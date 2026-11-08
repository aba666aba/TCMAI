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
import concurrent.futures
import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from core.data_desensitize import desensitize_query_text
from core.knowledge_base import KnowledgeBase
from core.result_verify import ResultVerifier
from core.tcm_config import generate_health_tips


# 统一根目录解析
def _get_project_root() -> Path:
    """获取项目根目录：TCMAI_ROOT > frozen > __file__ 降级链"""
    tcm_root = os.environ.get("TCMAI_ROOT")
    if tcm_root:
        return Path(tcm_root)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[2]

# 结果校验器（无外部依赖，模块加载时直接实例化）
_result_verifier: ResultVerifier | None = None


def _get_result_verifier() -> ResultVerifier:
    """获取结果校验器单例"""
    global _result_verifier
    if _result_verifier is None:
        _result_verifier = ResultVerifier()
    return _result_verifier


# NLU 引擎（延迟导入，避免启动时加载大模型）
_nlu_engine: Any | None = None
_nlu_loading: bool = False
_nlu_ready: bool = False
_nlu_error: str | None = None
_nlu_lock = threading.Lock()
_nlu_callbacks: list = []


def _get_nlu_engine():
    """获取 NLU 引擎单例（首次调用时加载模型，后续直接返回）

    双重检查加锁：预加载线程与首次查询并发时不会重复加载 ~1GB 模型。

    Returns:
        NLUEngine 实例，已调用 load() 完成模型加载
    """
    global _nlu_engine
    if _nlu_engine is None:
        with _nlu_lock:
            if _nlu_engine is None:
                import torch

                from nlu import NLUEngine
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # 有 GPU 时启用 4bit 量化节省显存，CPU 模式用全精度
                engine = NLUEngine(device=device, use_4bit=(device == "cuda"))
                engine.load()
                _nlu_engine = engine
                logger.info("NLU 引擎已就绪")
    return _nlu_engine


def preload_nlu_async(callback=None):
    """在后台线程预加载 NLU 模型，加载完成后调用 callback(success, message)"""
    global _nlu_loading, _nlu_ready, _nlu_error

    with _nlu_lock:
        if _nlu_ready or _nlu_loading:
            if callback:
                callback(_nlu_ready, _nlu_error or "模型已就绪")
            return
        _nlu_loading = True
        _nlu_error = None

    if callback:
        _nlu_callbacks.append(callback)

    def _load():
        global _nlu_loading, _nlu_ready, _nlu_error
        try:
            logger.info("开始后台预加载 NLU 模型...")
            _get_nlu_engine()
            with _nlu_lock:
                _nlu_ready = True
                _nlu_loading = False
            logger.info("NLU 模型后台预加载完成")
            for cb in _nlu_callbacks:
                try:
                    cb(True, "模型已就绪")
                except Exception:
                    pass
        except Exception as exc:
            with _nlu_lock:
                _nlu_loading = False
                _nlu_error = str(exc)
            logger.error("NLU 模型后台预加载失败: %s", exc)
            for cb in _nlu_callbacks:
                try:
                    cb(False, str(exc))
                except Exception:
                    pass

    thread = threading.Thread(target=_load, daemon=True, name="nlu-preload")
    thread.start()


def is_nlu_ready() -> bool:
    """查询 NLU 模型是否已就绪"""
    return _nlu_ready


def is_nlu_loading() -> bool:
    """查询 NLU 模型是否正在加载"""
    return _nlu_loading


def get_nlu_status() -> tuple[bool, bool, str | None]:
    """返回 (loading, ready, error)"""
    return _nlu_loading, _nlu_ready, _nlu_error


def _expand_syndrome_terms(
    text: str, max_syndromes: int = 3, max_extra_symptoms: int = 12
) -> str:
    """检测用户输入中的证型名（如"肾虚""脾虚"），附加对应症状提升召回。

    仅当证型名**完整出现**在输入中时才展开。原实现还包含
    `set(text) ⊆ set(证型名)` 的字符集合子集匹配与 startswith 前缀匹配，
    会把"气血"这类短词误命中多个证型、向 NLU 注入大量无关症状，已删除。
    展开症状数量有上限，避免淹没用户真实描述。
    """
    expanded = list(text)
    try:
        from tcm_recommender.extended_config import EXTENDED_SYNDROME_DEFINITIONS

        matched_syndromes = 0
        extra_symptoms = 0
        for syn_name, syn_info in EXTENDED_SYNDROME_DEFINITIONS.items():
            if matched_syndromes >= max_syndromes or extra_symptoms >= max_extra_symptoms:
                break
            if syn_name and syn_name in text:
                logger.info("检测到证型名: %s，展开症状", syn_name)
                for symptom in syn_info["symptoms"]:
                    if extra_symptoms >= max_extra_symptoms:
                        break
                    expanded.append(symptom)
                    extra_symptoms += 1
                matched_syndromes += 1
    except ImportError:
        pass
    return "".join(expanded)


def detect_symptoms_nlu(text: str) -> list[str]:
    """仅使用 NLU 引擎提取症状，失败则抛出异常（降级请用 extract_symptoms_with_fallback）"""
    enhanced_text = _expand_syndrome_terms(text)

    # ── NLU 提取（无降级）──
    engine = _get_nlu_engine()
    if not engine:
        raise RuntimeError("NLU 引擎不可用，无法提取症状")
    result = engine.extract_symptoms(enhanced_text)
    if not result:
        raise RuntimeError("NLU 引擎未提取到任何症状")
    logger.info("NLU 提取症状: %s", result)
    return result


def _extract_symptoms_by_keywords(text: str) -> list[str]:
    """关键词匹配降级：别名映射 + 标准症状词表直接匹配（不依赖任何模型）。"""
    try:
        from core.tcm_config import SYMPTOM_ALIAS_MAP, SYMPTOM_LIST
    except ImportError:
        logger.exception("关键词降级所需的词表导入失败")
        return []
    found: set[str] = set()
    for alias, standard in SYMPTOM_ALIAS_MAP.items():
        if alias and alias in text:
            found.add(standard)
    for symptom in SYMPTOM_LIST:
        if symptom in text:
            found.add(symptom)
    return sorted(found)[:20]


def extract_symptoms_with_fallback(text: str) -> tuple[list[str], str]:
    """症状提取三级降级链，保证查询不因 NLU 失败而失败。

    1. "nlu"     — Qwen 语义抽取（首选，失败/为空时降级）
    2. "keyword" — 别名映射 + 标准词表关键词匹配
    3. "none"    — 返回空列表，由 service.query 走纯智能搜索回答

    Returns:
        (symptoms, mode)，mode ∈ {"nlu", "keyword", "none"}
    """
    try:
        result = detect_symptoms_nlu(text)
        return result, "nlu"
    except Exception as exc:
        logger.warning("NLU 提取失败(%s: %s)，降级到关键词匹配", type(exc).__name__, exc)

    keywords = _extract_symptoms_by_keywords(text)
    if keywords:
        logger.info("关键词匹配到症状: %s", keywords)
        return keywords, "keyword"
    return [], "none"


logger = logging.getLogger(__name__)


def format_intelligent_knowledge_for_display(intel_result: dict[str, Any]) -> str:
    """格式化智能搜索结果为展示文本"""
    if not intel_result:
        return "暂无相关知识"

    parts: list[str] = []

    # 症状识别摘要
    symptoms = intel_result.get("symptoms", [])
    if symptoms:
        parts.append(f"【症状识别】{'、'.join(symptoms)}")

    # 搜索摘要
    summary = intel_result.get("search_summary", "")
    if summary:
        parts.append(f"【搜索摘要】{summary}")

    # 推理出的证型
    syndromes = intel_result.get("inferred_syndromes", [])
    if syndromes:
        syndrome_parts = []
        for syn in syndromes[:2]:
            syndrome_parts.append(f"{syn['syndrome']}({syn['score']:.0%})")
        parts.append(f"【推断证型】{'，'.join(syndrome_parts)}")

    # 推荐的方剂
    recommended = intel_result.get("recommended_formulas", [])
    if recommended:
        formula_parts = []
        for ref in recommended[:5]:
            formula_parts.append(
                f"{ref['name']}（{ref['syndrome']}：{ref['description']}）"
            )
        parts.append(f"【推荐方剂】{'；'.join(formula_parts)}")

    # 知识库搜索到的方剂
    db_formulas = intel_result.get("formulas", [])
    if db_formulas:
        parts.append(f"【知识库方剂】{', '.join(f['name'] for f in db_formulas[:5])}")

    # 药材
    db_herbs = intel_result.get("herbs", [])
    if db_herbs:
        parts.append(f"【相关药材】{', '.join(h['name'] for h in db_herbs[:5])}")

    # 中成药
    db_patents = intel_result.get("patent_medicines", [])
    if db_patents:
        parts.append(f"【相关中成药】{', '.join(p['name'] for p in db_patents[:5])}")

    return "\n".join(parts) if parts else "暂无相关知识"


class TCMDesktopService:
    MAX_HISTORY_ITEMS = 500

    def __init__(self) -> None:
        self.kb = KnowledgeBase()
        # print(f"[DEBUG] KnowledgeBase loaded, available={self.kb.is_available}")
        self.history_lock = threading.Lock()
        # TODO: 历史记录应该用sqlite，json文件并发有问题
        self.history_file = self._resolve_history_file()
        self.history_lock_file = self.history_file.with_suffix(".lock")
        self.history_data = self.load_history()
        self._recommender = None
        self._recommender_load_attempted = False
        self._recommender_available = False

    def _resolve_history_file(self):
        # 可写数据目录（打包环境为 %LOCALAPPDATA%/TCMAI/data，开发环境为项目 data/）
        from core.paths import get_history_file

        return get_history_file()

    @contextmanager
    def _acquire_file_lock(self):
        self.history_lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_lock_file, "a+b") as handle:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.lockf(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.lockf(handle, fcntl.LOCK_UN)

    def _sanitize_history_item(self, item):
        symptoms = item.get("symptoms", [])
        if not isinstance(symptoms, list):
            symptoms = []
        return {
            "id": int(item.get("id", 0)),
            "query": str(item.get("query", ""))[:500],
            "symptoms": [str(s)[:64] for s in symptoms[:20]],
            "answer": str(item.get("answer", ""))[:4000],
            "timestamp": str(item.get("timestamp", ""))[:32],
            "mode": item.get("mode", "ai"),
        }

    def _normalize_history(self, data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, list):
            return []
        normalized: list[dict[str, Any]] = []
        used_ids = set()
        next_id = 1
        for raw_item in data[: self.MAX_HISTORY_ITEMS]:
            if not isinstance(raw_item, dict):
                continue
            item = self._sanitize_history_item(raw_item)
            if item["id"] <= 0 or item["id"] in used_ids:
                item["id"] = next_id
            used_ids.add(item["id"])
            next_id = max(next_id, item["id"] + 1)
            normalized.append(item)
        return normalized

    def load_history(self) -> list[dict[str, Any]]:
        # TODO: json读写没有原子性，多进程会丢数据
        if not self.history_file.exists():
            return []
        try:
            with self._acquire_file_lock():
                with open(self.history_file, encoding="utf-8") as handle:
                    data = json.load(handle)
            return self._normalize_history(data)
        except json.JSONDecodeError:
            backup = self.history_file.with_suffix(".corrupt.json")
            try:
                if self.history_file.exists():
                    os.replace(self.history_file, backup)
            except Exception:
                pass
            logger.error("历史记录文件损坏，已重置为空并保留损坏副本")
            return []
        except Exception as exc:
            logger.error("加载历史记录失败: %s", exc)
            return []

    def save_history(self) -> None:
        try:
            sanitized = self._normalize_history(self.history_data)[
                -self.MAX_HISTORY_ITEMS :
            ]
            with self._acquire_file_lock():
                with NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    delete=False,
                    dir=str(self.history_file.parent),
                    prefix="history_",
                    suffix=".tmp",
                ) as handle:
                    json.dump(sanitized, handle, ensure_ascii=False, indent=2)
                    temp_name = handle.name
                os.replace(temp_name, self.history_file)
            self.history_data = sanitized
        except Exception as exc:
            logger.error("保存历史记录失败: %s", exc)

    def _get_recommender(self):
        """懒加载推荐模型单例"""
        if self._recommender_load_attempted:
            return self._recommender

        self._recommender_load_attempted = True
        try:
            import torch

            from tcm_recommender.recommend import TCMRecommender

            root = str(_get_project_root())
            from core.paths import ensure_database

            writable_db = ensure_database()
            db_path = str(writable_db) if writable_db else os.path.join(root, "data", "tcm_knowledge.db")
            model_path = os.path.join(root, "tcm_recommender", "best_model.pth")
            if not os.path.exists(db_path) or not os.path.exists(model_path):
                logger.warning("推荐模型或数据库文件不存在，AI 模式不可用")
                return None

            self._recommender = TCMRecommender(
                model_path=model_path,
                db_path=db_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self._recommender_available = True
            logger.info("推荐模型加载成功，AI 模式可用")
        except ImportError:
            logger.warning("PyTorch 未安装，AI 模式使用降级方案")
        except Exception as exc:
            logger.error("推荐模型加载失败: %s", exc)

        return self._recommender

    def _get_kg_enhanced(self):
        """懒加载 KG 增强推荐器（包裹已加载的 DL 推荐器 + 知识图谱）。

        依赖 networkx 反序列化 kg_cache.pkl；任一环节失败均返回 None，
        由调用方优雅降级（与 NLU 降级哲学一致）。
        """
        if hasattr(self, "_kg_enhanced"):
            return self._kg_enhanced
        self._kg_enhanced = None
        try:
            from kg.recommender import KGEnhancedRecommender
            from kg.query import KnowledgeGraphQuery

            rec = self._get_recommender()
            if rec is None:
                logger.warning("DL 推荐器不可用，跳过 KG 增强")
                return None

            import pickle

            from core.paths import get_data_dir

            root = str(_get_project_root())
            candidates = [
                os.path.join(str(get_data_dir()), "kg_cache.pkl"),
                os.path.join(root, "data", "kg_cache.pkl"),
            ]
            kg_cache = next((p for p in candidates if os.path.exists(p)), None)
            if not kg_cache:
                logger.warning("未找到 kg_cache.pkl，跳过 KG 增强")
                return None

            with open(kg_cache, "rb") as f:
                graph = pickle.load(f)
            kg = KnowledgeGraphQuery(graph)
            self._kg_enhanced = KGEnhancedRecommender(rec, kg)
            logger.info("KG 增强推荐器初始化成功")
        except Exception as exc:
            logger.warning("KG 增强初始化失败（已跳过）: %s", exc)
            self._kg_enhanced = None
        return self._kg_enhanced

    def _build_enrichment(self, query_text, symptoms, intel_result, base_safety):
        """构建「AI 解读」+「知识图谱辨证」+「组方草案」三段增强文本。

        把 KG 融合接进主链路：每次查询都运行 KGEnhancedRecommender，
        取其推理路径 / 毒性与配伍禁忌洞察，并结合检索证据合成连贯解读。
        任何异常都返回 ("", "", "")，保证主链路不受影响（优雅降级）。
        """
        interpret_text = ""
        kg_text = ""
        rx_text = ""
        try:
            kg_paths = None
            kg_safety_warn = []
            kg_insights = {}
            kg = self._get_kg_enhanced()
            if kg is not None and symptoms:
                kg_res = kg.recommend(
                    " ".join(symptoms), top_k_herb=10, top_k_formula=5
                )
                kg_paths = kg_res.get("reasoning_paths")
                kg_insights = kg_res.get("kg_insights") or {}
                kg_safety_warn = (kg_res.get("safety") or {}).get("kg_warnings") or []
                rx_text = self._generate_rx_text(symptoms, kg_res, kg)

            from rag.synthesizer import synthesize

            interpret_text = synthesize(
                query_text, symptoms, intel_result,
                kg_paths=kg_paths, safety=base_safety,
            )

            lines = []
            if kg_paths:
                for p in kg_paths[:4]:
                    if p.get("detail"):
                        lines.append(f"· {p['detail']}")
            if kg_insights.get("toxic_warning"):
                lines.append(f"⚠️ {kg_insights['toxic_warning']}")
            for w in kg_insights.get("incompatible_warning") or []:
                lines.append(f"⚠️ {w}")
            for w in kg_safety_warn:
                lines.append(f"⚠️ {w}")
            if lines:
                kg_text = "【知识图谱辨证】\n" + "\n".join(lines)
        except Exception as exc:
            logger.warning("RAG/KG 增强生成失败（已跳过）: %s", exc)
        # rx_text 独立返回：既并入 AI 解读尾部，也单独供「组方草案」弹窗使用
        if rx_text:
            kg_text = (kg_text + "\n\n" + rx_text).strip()
        return interpret_text, kg_text, rx_text

    def _generate_rx_text(self, symptoms: list[str], kg_res: dict, kg) -> str:
        """AI 组方草案（药性驱动 + 配伍硬约束 + 已入库方剂备选，供医师审核）。

        任何异常返回 ""，保证主链路不受影响。
        """
        try:
            gen = getattr(self, "_rx_gen", None)
            if gen is None:
                from rx.generator import FormulaGenerator

                gen = FormulaGenerator(
                    getattr(getattr(kg, "model", None), "db_path", None) or None
                )
                self._rx_gen = gen
            res = gen.generate(
                symptoms=symptoms,
                herb_candidates=[
                    {"name": h.get("name", ""), "score": h.get("score", 0)}
                    for h in (kg_res.get("herbs") or [])
                ],
                syndrome_name=(kg_res.get("syndrome") or {}).get("name"),
                matched_syndromes=(kg_res.get("syndrome_analysis") or {}).get(
                    "matched_syndromes"
                ),
                alternative_formulas=[
                    f.get("name", "") for f in (kg_res.get("formulas") or [])
                ],
            )
            return res.get("text") or ""
        except Exception as exc:
            logger.warning("AI 组方生成失败（已跳过）: %s", exc)
            return ""

    @staticmethod
    def _scores_saturated(scores: list[float], threshold: float = 0.999) -> bool:
        """检测模型输出分数是否大面积饱和（sigmoid 输出大量 1.0）。

        README「实现现状与已知局限」实测 Top-12 得分可全为 1.000——此时数值
        不具备任何置信度含义，继续展示"置信度 1.00"是虚假的精确暗示，
        应整体隐藏而不是照常显示。
        """
        scores = [s for s in scores if isinstance(s, (int, float))]
        if len(scores) < 2:
            return True  # 单条结果缺少对比基准，数值无意义
        saturated = sum(1 for s in scores if s >= threshold)
        return saturated / len(scores) >= 0.5

    @staticmethod
    def _short(text: str, n: int = 20) -> str:
        text = (text or "").strip().rstrip("。；;")
        return text[:n] + ("…" if len(text) > n else "")

    def _build_answer_parts(
        self,
        query_text: str,
        symptoms: list[str],
        result: dict[str, Any],
        safety: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """把推荐结果拆成「最简主文 + 模块化详情」。

        Returns:
            compact: 主文本框内容——只写最简的推荐方剂 + 推荐药材（一屏看全）
            sections: 模块化详情（由结果区下方按钮逐个弹窗展示）：
                      症状分析 / 安全与配伍 / 养生建议
                      （智能搜索发现、组方草案由 query 流程追加）
        """
        compact: list[str] = []
        if result.get("formulas"):
            compact.append("【推荐方剂】")
            for i, f in enumerate(result["formulas"][:3], 1):
                func = (f.get("functions") or "").strip()
                mark = " ☠️" if f.get("contains_toxic") else ""
                line = f"{i}. {f['name']}{mark}"
                if func:
                    line += f" —— {self._short(func, 18)}"
                compact.append(line)
        if result.get("herbs"):
            compact.append("")
            compact.append("【推荐药材】")
            for h in result["herbs"][:6]:
                eff = (h.get("efficacy") or "").strip()
                usage = (
                    h.get("usage") or self.get_herb_dosage(h["name"]) or ""
                ).strip()
                mark = " ☠️" if h.get("is_toxic") else ""
                segs = [h["name"] + mark]
                if eff:
                    segs.append(self._short(eff, 10))
                if usage:
                    segs.append(usage.replace("，", " "))
                compact.append(" · ".join(segs))
        if not compact:
            compact.append("（本次未生成结构化推荐，详见下方模块）")
        compact_text = "\n".join(compact)

        # ── 模块化详情 ──
        sections: dict[str, str] = {}

        analysis = ["【症状分析】"]
        analysis.append(
            f"您描述的症状：{'、'.join(symptoms) if symptoms else query_text}"
        )
        if result.get("symptoms"):
            analysis.append(f"系统识别到：{'、'.join(result['symptoms'])}")
        top_formula = (result.get("formulas") or [{}])[0].get("name", "")
        if top_formula:
            analysis.append(f"首选方剂：「{top_formula}」（辨证思路见「AI 解读」页）")
        sections["症状分析"] = "\n".join(analysis)

        safety_lines = []
        blocked = result.get("blocked_toxic_herbs") or []
        if blocked:
            safety_lines.append("【安全拦截】")
            safety_lines.append(
                f"⚠️ 以下毒性/峻烈药材已被安全策略拦截：{'、'.join(blocked)}"
            )
            safety_lines.append(
                "   此类药材必须由执业中医师根据病情、体质与炮制规范决定，本系统不予推荐。"
            )
        safety_lines.append("【安全提示】")
        if safety:
            for w in (safety.get("incompatibilities") or [])[:3]:
                safety_lines.append(f"⚠️ 配伍禁忌 — {self._short(w, 60)}")
            if safety.get("toxic_herbs"):
                safety_lines.append(
                    f"☠️ 含毒性药材：{'、'.join(safety['toxic_herbs'])}，须在医师指导下使用"
                )
            cautions = (safety.get("warnings") or [])[:3]
            if cautions:
                safety_lines.append("药材一般注意事项：")
                for w in cautions:
                    safety_lines.append(f"· {self._short(w, 46)}")
            if safety.get("is_safe"):
                safety_lines.append("✓ 未检测到十八反/十九畏类配伍禁忌，且不含毒性药材")
        else:
            safety_lines.append("⚠️ 未能完成安全校验，请务必咨询执业中医师后再使用")
        sections["安全与配伍"] = "\n".join(safety_lines)

        try:
            tips = generate_health_tips(set(result.get("symptoms", []) or symptoms))
        except Exception:
            tips = []
        if tips:
            sections["养生建议"] = "\n".join(tips)

        return compact_text, sections

    def _format_recommendation_as_answer(
        self,
        query_text: str,
        symptoms: list[str],
        result: dict[str, Any],
        safety: dict[str, Any] | None = None,
    ) -> str:
        """（兼容）完整文本回答：compact + 全部模块拼接，供历史记录使用。"""
        compact, sections = self._build_answer_parts(
            query_text, symptoms, result, safety
        )
        full = [compact]
        for body in sections.values():
            full.append("\n" + body)
        return "\n".join(full)

    def query(
        self, query_text: str, symptoms: list[str], nlu_mode: str = "nlu"
    ) -> dict[str, Any]:
        """纯 AI 模式查询（已移除快速模式降级）

        Args:
            query_text: 用户输入的查询文本
            symptoms: 已检测的症状列表（可能来自 NLU、关键词降级或为空）
            nlu_mode: 症状识别方式，"nlu"/"keyword"/"none"（用于回答中如实说明）

        Returns:
            包含 AI 推荐结果 + 智能搜索增强的完整回答

        Raises:
            ValueError: 输入参数校验失败
            RuntimeError: AI 模型不可用或推理失败
        """
        if not query_text or len(query_text.strip()) < 1 or len(query_text) > 1000:
            raise ValueError("查询内容长度需要在 1-1000 个字符之间")
        if not isinstance(symptoms, list):
            raise ValueError("症状参数必须为列表")
        if len(symptoms) > 20:
            raise ValueError("症状列表最多支持 20 项")

        filtered_symptoms = [s for s in symptoms if isinstance(s, str) and s.strip()]
        # print(f"[DEBUG] query={query_text!r}, symptoms={filtered_symptoms}")

        # ========== 输入脱敏（进入检索/模型/历史前去掉手机号、身份证、邮箱等） ==========
        safe_query = desensitize_query_text(query_text)
        safe_symptoms = [desensitize_query_text(s) for s in filtered_symptoms]

        # ========== 智能搜索（增强 AI 结果的补充知识） ==========
        intel_result = self.kb.intelligent_search(safe_query)
        intel_knowledge_text = format_intelligent_knowledge_for_display(intel_result)

        # ========== 加载 AI 模型 ==========
        recommender = self._get_recommender()
        if recommender is None:
            raise RuntimeError(
                "AI 模型不可用，无法生成推荐。\n"
                "\n"
                "最常见原因：本仓库的模型权重体积过大（~127MB），未纳入 git，\n"
                "直接 clone 后仓库里没有该文件。请先执行一次下载脚本：\n"
                "    pip install modelscope\n"
                "    python scripts/download_weights.py\n"
                "\n"
                "若已下载仍报错，请依次检查：\n"
                "1. 是否已安装 torch 与 accelerate（accelerate 缺失会导致 transformers 无法导入）\n"
                "2. 模型文件 tcm_recommender/best_model.pth 是否存在\n"
                "3. 知识库文件 data/tcm_knowledge.db 是否存在"
            )

        # ========== AI 模型推理（最多重试 2 次） ==========
        answer = None
        knowledge_result = None
        safety_result = None

        for attempt in range(2):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._run_recommender_inference,
                        recommender,
                        safe_query,
                        safe_symptoms,
                    )
                    answer, knowledge_result = future.result(timeout=60)
                break  # 成功则跳出重试循环
            except concurrent.futures.TimeoutError:
                if attempt == 0:
                    logger.warning("推荐模型推理超时，重试一次…")
                    continue
                raise RuntimeError(
                    "AI 模型推理超时，请稍后重试。若持续超时，请检查 GPU/CPU 负载。"
                ) from None
            except Exception as exc:
                logger.error("推荐模型推理失败(%s): %s", type(exc).__name__, exc)
                if attempt == 0:
                    logger.info("重试推荐模型推理…")
                    continue
                raise RuntimeError(f"AI 模型推理失败：{exc}") from exc

        # ========== 用智能搜索增强 AI 回答 ==========
        if answer is None:
            # 三级降级第 3 层：模型链路无有效症状 → 纯智能搜索回答
            logger.info("推荐模型无有效症状输入，降级为纯智能搜索回答")
            if not intel_knowledge_text or intel_knowledge_text == "暂无相关知识":
                raise RuntimeError(
                    "未能从输入中识别出可分析的症状，请尝试更具体的症状描述"
                    "（如：失眠、乏力、头晕、咳嗽）。"
                )
            final_answer = intel_knowledge_text
            knowledge = {"source": "intelligent_search"}
        else:
            final_answer = answer
            if intel_knowledge_text and intel_knowledge_text != "暂无相关知识":
                # 将智能搜索发现嵌入回答
                enrichment = f"\n【智能搜索发现】\n{intel_knowledge_text}"
                # 放在安全提示之前
                if "【安全提示】" in final_answer:
                    final_answer = final_answer.replace(
                        "【安全提示】", enrichment + "\n\n【安全提示】"
                    )
                else:
                    final_answer += enrichment

            knowledge = knowledge_result or {}
        if intel_result.get("recommended_formulas"):
            knowledge["intelligent_search"] = {
                "symptoms": intel_result.get("symptoms", []),
                "inferred_syndromes": [
                    s["syndrome"] for s in intel_result.get("inferred_syndromes", [])
                ],
                "recommended_formulas": [
                    f["name"] for f in intel_result.get("recommended_formulas", [])
                ],
            }

        # ========== RAG 解读 + 知识图谱辨证（增强，失败则跳过） ==========
        # 这两段为「基于检索证据的文本解读」，刻意不与结构化推荐混排，
        # 而是作为独立字段 ai_interpretation 返回，供桌面端在「AI 解读」面板展示。
        interpret_text, kg_text, rx_draft = self._build_enrichment(
            safe_query, safe_symptoms, intel_result, safety_result
        )
        ai_interpretation = "\n\n".join(
            p for p in (interpret_text, kg_text) if p
        ).strip()

        # ========== 组装模块化详情（结果区按钮弹窗逐个展示） ==========
        sections = dict(knowledge.get("answer_sections") or {})
        if (
            intel_knowledge_text
            and intel_knowledge_text != "暂无相关知识"
            and answer is not None
        ):
            sections["智能搜索发现"] = intel_knowledge_text
        if rx_draft:
            sections["组方草案"] = rx_draft

        # ========== 如实标注症状识别方式（降级时用户须知悉精度受限） ==========
        nlu_notice = {
            "keyword": "【症状识别方式】AI 语义模型本次不可用，已改用关键词匹配识别症状，识别能力有限。\n\n",
            "none": "【症状识别方式】未能识别出标准症状，以下为基于全文检索的相关知识，不构成症状分析。\n\n",
        }.get(nlu_mode, "")
        if nlu_notice:
            final_answer = nlu_notice + final_answer

        # ========== 结果校验（过滤"保证治愈/根治"等绝对化断言 + 典籍溯源 + 合规提示） ==========
        verification = _get_result_verifier().verify(final_answer, knowledge)
        final_answer = verification["answer"]
        disclaimer = verification["disclaimer"]
        has_warning = verification["has_warning"]

        history_item = self._append_history(
            safe_query, safe_symptoms, final_answer, "ai"
        )
        if history_item is not None:
            history_item["ai_interpretation"] = ai_interpretation
        return {
            "success": True,
            "answer": final_answer,
            "answer_compact": knowledge.get("answer_compact") or final_answer,
            "answer_sections": sections,
            "ai_interpretation": ai_interpretation,
            "knowledge": knowledge,
            "disclaimer": disclaimer,
            "mode": "ai",
            "nlu_mode": nlu_mode,
            "has_warning": has_warning,
            "history_item": history_item,
        }

    def _run_recommender_inference(
        self, recommender, query_text: str, symptoms: list[str]
    ):
        """在独立线程中运行推荐模型推理"""
        symptom_ids, valid, unknown = recommender.parse_symptoms(
            " ".join(symptoms) if symptoms else query_text
        )
        if not symptom_ids:
            return None, None

        result = recommender.recommend(symptom_ids, top_k_herb=10, top_k_formula=5)

        # 安全校验
        safety_result: dict[str, Any] | None = None
        try:
            herb_names = [h["name"] for h in result.get("herbs", [])[:10]]
            safety_result = recommender.check_safety(herb_names)
        except Exception as exc:
            logger.warning("安全校验异常（已跳过）: %s", exc)

        answer, answer_sections = self._build_answer_parts(
            query_text, symptoms, result, safety_result
        )
        knowledge = {
            "symptoms": result.get("symptoms", []),
            "syndrome": result.get("syndrome", {}),
            "herbs": [
                {"name": h["name"], "score": h["score"]}
                for h in result.get("herbs", [])[:10]
            ],
            "formulas": [
                {"name": f["name"], "score": f["score"]}
                for f in result.get("formulas", [])[:5]
            ],
            "source": "deep_learning_model",
            "answer_compact": answer,
            "answer_sections": answer_sections,
        }
        return answer, knowledge

    def _append_history(
        self, query_text: str, symptoms: list[str], answer: str, mode: str
    ) -> dict[str, Any]:
        with self.history_lock:
            next_id = (
                max((item.get("id", 0) for item in self.history_data), default=0) + 1
            )
            record = self._sanitize_history_item(
                {
                    "id": next_id,
                    "query": query_text,
                    "symptoms": symptoms,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "mode": mode,
                }
            )
            self.history_data.append(record)
            self.history_data = self.history_data[-self.MAX_HISTORY_ITEMS :]
            self.save_history()
            return record

    def list_history(self) -> list[dict[str, Any]]:
        return sorted(
            self.history_data[-self.MAX_HISTORY_ITEMS :],
            key=lambda item: item.get("timestamp", ""),
            reverse=True,
        )

    def export_history(self) -> list[dict[str, Any]]:
        return self.list_history()[:100]

    def delete_history_item(self, item_id: int) -> bool:
        with self.history_lock:
            before = len(self.history_data)
            self.history_data[:] = [
                item for item in self.history_data if item.get("id") != item_id
            ]
            changed = len(self.history_data) != before
            if changed:
                self.save_history()
            return changed

    def clear_history(self) -> None:
        with self.history_lock:
            self.history_data.clear()
            self.save_history()

    def get_herb_dosage(self, herb_name: str) -> str:
        """从数据库查询药材用法用量。

        查询失败时返回空串（用法用量为展示增强项，缺失不应中断推荐），
        但记录异常——静默返回空串会让"库损坏/字段缺失"与"该药材本就没有用法"
        无法区分。
        """
        if not self.kb or not self.kb.is_available:
            return ""
        try:
            with self.kb._lock:
                cur = self.kb.conn.execute(
                    "SELECT usage FROM herbs WHERE name = ? LIMIT 1",
                    (herb_name,),
                )
                row = cur.fetchone()
                if row and row["usage"]:
                    return row["usage"]
        except Exception:
            logger.exception("查询药材用法失败（herb=%s），已降级为空", herb_name)
        return ""

    def close(self) -> None:
        try:
            self.kb.close()
        except Exception as exc:
            logger.warning("知识库关闭异常: %s", exc)
        try:
            if self._recommender is not None:
                self._recommender.close()
        except Exception as exc:
            logger.warning("推荐模型关闭异常: %s", exc)
