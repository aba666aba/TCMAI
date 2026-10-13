# Copyright (C) 2026 aba666aba
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 本文件是中医药智能问答系统 (TCMAI) 的一部分。
#
# 本程序是自由软件：你可以依据自由软件基金会发布的 GNU Affero 通用公共
# 许可证条款（第 3 版或你选择的任何更新版本）重新分发和/或修改本程序。
#
# 本程序按“希望有用”的原则分发，但不提供任何担保，甚至不提供适销性或
# 特定用途适用性的默示担保。详见 GNU Affero 通用公共许可证。
#
# 你应已随本程序收到 GNU Affero 通用公共许可证副本。如未收到，
# 请见 <https://www.gnu.org/licenses/>。
#
# 注意：以网络服务方式提供本程序的服务端修改版本，同样必须按 AGPL-3.0
# 第 13 条向所有用户提供对应源代码。
"""NLU 引擎 — 基于 Qwen2.5 的自然语言→结构化症状提取"""
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# 本地模型路径 — 惰性计算以适配 PyInstaller 打包环境
_DEFAULT_MODEL_PATH: str | None = None


def _resolve_default_model_path() -> str:
    """运行时解析默认模型路径，兼容开发 / frozen / 已设置 TCMAI_ROOT 三种场景。"""
    tcm_root = os.environ.get("TCMAI_ROOT")
    if tcm_root:
        return os.path.join(tcm_root, "models", "nlu", "qwen")
    if getattr(sys, "frozen", False):
        return str(Path(sys.executable).parent / "models" / "nlu" / "qwen")
    # 开发模式：从 src/nlu/engine.py 向上 3 层到项目根
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models", "nlu", "qwen"
    )

# 系统提示词
_SYSTEM_PROMPT = """你是一位中医药症状提取专家。你的任务是从患者的病情描述中**完整地提取所有症状**，并映射到标准症状名称。

规则：
1. 仔细阅读患者的病情描述
2. 从以下标准症状列表中选取**所有匹配**的症状名称
3. **只输出列表中存在的症状名**，不要输出任何不在列表中的词
4. 如果描述中的口语词不在列表中，选择列表中意思最接近的标准症状名
5. **宁可多选不要漏选** — 只要描述中提到的症状，都要提取
6. 输出格式为 JSON 数组：["症状1", "症状2", ...]
7. 只输出 JSON，不要任何解释或其他文字

示例：
患者描述："胃部隐痛两天，吃不下饭，反酸嗳气"
输出：["胃痛", "食欲不振", "嗳气"]

患者描述："浑身没劲，走几步就喘，晚上睡不着，心慌"
输出：["乏力", "气短", "失眠", "心悸"]

患者描述："怕冷，手脚冰凉，腰酸腿软，夜尿多"
输出：["畏寒", "手足不温", "腰膝酸软", "夜尿多"]

患者描述："拉肚子，肚子胀"
输出：["腹泻", "腹胀"]

标准症状列表（共 229 种）：
{症状列表}"""


class NLUEngine:
    """自然语言理解引擎，用 LLM 提取结构化症状"""

    def __init__(self, model_path: str | None = None,
                 device: str | None = None, use_4bit: bool = True):
        # 仅使用本地模型路径，不提供网络回退
        self.model_path = model_path or _resolve_default_model_path()
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_4bit = use_4bit and (self.device == "cuda")
        self._model = None
        self._tokenizer = None
        self._symptom_list_text = ""
        self._vocab_cache = None

    def _build_symptom_list_text(self) -> str:
        if self._symptom_list_text:
            return self._symptom_list_text
        try:
            from tcm_recommender.extended_config import EXTENDED_SYMPTOM_LIST
            items = "、".join(EXTENDED_SYMPTOM_LIST)
            self._symptom_list_text = items
            return items
        except ImportError:
            return "胃痛、头痛、发热、咳嗽、失眠、乏力、腹痛、腹泻"

    def _load_vocabulary(self) -> tuple[set[str], dict[str, str]]:
        """加载标准症状词表与别名映射（与 recommend.py 使用同一份数据）。

        提示词里的"只输出列表内症状"是软约束，小模型经常不遵守
        （实测：输入"睡不着、没力气、头晕"会输出词表外的"疲劳""乏累"）。
        因此需要在解析后做一次硬校验。
        """
        if self._vocab_cache is not None:
            return self._vocab_cache
        try:
            from tcm_recommender.extended_config import (
                EXTENDED_SYMPTOM_LIST,
                SYMPTOM_ALIAS_MAP,
            )

            vocab = set(EXTENDED_SYMPTOM_LIST)
            aliases = {k: v for k, v in SYMPTOM_ALIAS_MAP.items() if v in vocab}
        except ImportError:
            vocab = {"胃痛", "头痛", "发热", "咳嗽", "失眠", "乏力", "腹痛", "腹泻"}
            aliases = {}
        self._vocab_cache = (vocab, aliases)
        return self._vocab_cache

    def _normalize_symptoms(self, symptoms: list[str]) -> list[str]:
        """把模型输出约束到 229 条标准词表内。

        处理顺序：
          1. 精确命中标准词 → 直接保留
          2. 命中别名表（如 疲劳→乏力、睡不着→失眠）→ 映射为标准词
          3. 前两步都没有结果时，才尝试"标准词包含该词"的保守匹配
             （如 乏累→乏力），且仅当唯一命中时才采纳
          4. 仍无法归一的 → 丢弃并记警告，避免词表外词汇流向下游
             （下游 TCMRecommender 用词表索引症状，词表外词无法编码）
        """
        vocab, aliases = self._load_vocabulary()
        resolved: list[str] = []
        unmapped: list[str] = []

        for raw in symptoms:
            item = str(raw).strip()
            if not item:
                continue
            if item in vocab:
                target = item
            elif item in aliases:
                target = aliases[item]
            else:
                target = None
            if target is None:
                unmapped.append(item)
            elif target not in resolved:
                resolved.append(target)

        # 保守兜底：仅在无任何命中时启用子串匹配，且要求唯一命中
        if not resolved and unmapped:
            for item in unmapped:
                candidates = [s for s in vocab if s in item or item in s]
                if len(candidates) == 1 and candidates[0] not in resolved:
                    resolved.append(candidates[0])

        if unmapped:
            logger.warning(
                "NLU 输出含词表外症状，已丢弃或归一: %s", "、".join(unmapped)
            )
        return resolved

    def load(self):
        if self._model is not None:
            return

        model_source = self.model_path
        if not os.path.exists(model_source):
            raise FileNotFoundError(
                f"本地 NLU 模型不存在: {model_source}\n"
                f"请确保 Qwen2.5 模型已放置在正确位置。"
            )

        logger.info("加载 NLU 模型: %s (device=%s, 4bit=%s)",
                     model_source, self.device, self.use_4bit)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            kwargs = {"trust_remote_code": True}
            if self.use_4bit and self.device == "cuda":
                try:
                    import bitsandbytes  # noqa: F401
                    kwargs["load_in_4bit"] = True
                    kwargs["device_map"] = "auto"
                    logger.info("使用 4-bit 量化加载")
                except ImportError:
                    logger.warning("bitsandbytes 不可用，回退到 float16")
                    kwargs["torch_dtype"] = "float16"
                    kwargs["device_map"] = "auto"
            elif self.device == "cuda":
                kwargs["torch_dtype"] = "float16"
                kwargs["device_map"] = "auto"
            else:
                kwargs["device_map"] = "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_source, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_source, **kwargs
            )
            self._model.eval()

            # 为 generate() 设置 pad_token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            logger.info("NLU 模型加载完成")

        except Exception as exc:
            logger.error("NLU 模型加载失败: %s", exc)
            self._pipe = None
            raise

    def extract_symptoms(self, text: str) -> list[str]:
        if self._model is None:
            logger.warning("NLU 模型未加载，尝试加载...")
            try:
                self.load()
            except Exception as exc:
                logger.error("NLU 模型加载失败，无法提取症状: %s", exc)
                raise RuntimeError(f"NLU 模型加载失败: {exc}") from exc

        symptom_list = self._build_symptom_list_text()
        prompt = _SYSTEM_PROMPT.format(症状列表=symptom_list)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"患者描述：{text}"},
        ]

        try:
            import torch

            # 应用 chat template → tokenize
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=2048
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            # 解码（跳过输入部分）
            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[0, input_len:]
            raw_output = self._tokenizer.decode(generated, skip_special_tokens=True)
            logger.info("NLU 原始输出: %s", raw_output[:200])

            symptoms = self._parse_json_output(raw_output)
            normalized = self._normalize_symptoms(symptoms)
            if len(normalized) != len(symptoms):
                logger.info(
                    "NLU 词表校验: %d 条输出 → %d 条标准症状 %s",
                    len(symptoms),
                    len(normalized),
                    normalized,
                )
            return normalized

        except Exception as exc:
            logger.error("NLU 推理失败: %s", exc)
            raise RuntimeError(f"NLU 推理失败: {exc}") from exc

    def _parse_json_output(self, raw: str):
        raw = raw.strip()
        # 找 JSON 数组
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end > start:
            json_str = raw[start : end + 1]
            try:
                result = json.loads(json_str)
                if isinstance(result, list):
                    return [str(s).strip() for s in result if str(s).strip()]
            except json.JSONDecodeError:
                pass
        # 逐行提取
        symptoms = []
        for line in raw.split("\n"):
            line = line.strip().strip('"').strip("'").strip("-,.。，、 ")
            if line and len(line) >= 2 and line not in symptoms:
                symptoms.append(line)
        return symptoms

    def unload(self):
        self._model = None
        self._tokenizer = None
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("NLU 模型已卸载")


# 全局单例
_engine: NLUEngine | None = None


def get_engine() -> NLUEngine:
    global _engine
    if _engine is None:
        _engine = NLUEngine()
    return _engine


def extract_symptoms(text: str) -> list[str]:
    engine = get_engine()
    return engine.extract_symptoms(text)
