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
            return symptoms

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
