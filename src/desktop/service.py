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

from core.knowledge_base import KnowledgeBase
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

# NLU 引擎（延迟导入，避免启动时加载大模型）
_nlu_engine: Any | None = None
_nlu_loading: bool = False
_nlu_ready: bool = False
_nlu_error: str | None = None
_nlu_lock = threading.Lock()
_nlu_callbacks: list = []


def _get_nlu_engine():
    global _nlu_engine
    if _nlu_engine is None:
        import torch

        from nlu import NLUEngine
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _nlu_engine = NLUEngine(device=device, use_4bit=(device == "cuda"))
        _nlu_engine.load()
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


def detect_symptoms_nlu(text: str) -> list[str]:
    """仅使用 NLU 引擎提取症状，失败则抛出异常"""
    # 证型→症状展开（用户可能直接输入"肾虚""脾虚"等）
    expanded = list(text)
    try:
        from tcm_recommender.extended_config import EXTENDED_SYNDROME_DEFINITIONS

        for syn_name, syn_info in EXTENDED_SYNDROME_DEFINITIONS.items():
            text_in_syn = text in syn_name
            syn_in_text = syn_name in text
            startswith = syn_name.startswith(text) or text.startswith(syn_name)
            char_match = set(text) <= set(syn_name) and len(text) >= 2
            if text_in_syn or syn_in_text or startswith or char_match:
                logger.info(
                    "检测到证型名: %s → %s，展开症状: %s",
                    text,
                    syn_name,
                    syn_info["symptoms"][:5],
                )
                expanded.extend(syn_info["symptoms"])
    except ImportError:
        pass
    enhanced_text = "".join(expanded)

    # ── NLU 提取（无降级）──
    engine = _get_nlu_engine()
    if not engine:
        raise RuntimeError("NLU 引擎不可用，无法提取症状")
    result = engine.extract_symptoms(enhanced_text)
    if not result:
        raise RuntimeError("NLU 引擎未提取到任何症状")
    logger.info("NLU 提取症状: %s", result)
    return result


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
        self.history_file = self._resolve_history_file()
        self.history_lock_file = self.history_file.with_suffix(".lock")
        self.history_data = self.load_history()
        self._recommender = None
        self._recommender_load_attempted = False
        self._recommender_available = False

    def _resolve_history_file(self) -> Path:
        root = _get_project_root()
        data_dir = (root / "data").resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "history.json"

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
            "mode": "ai" if item.get("mode") == "ai" else "ai",
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
        # TODO: 历史记录应该用sqlite，json文件并发有问题
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
            db_path = os.path.join(root, "data", "tcm_knowledge.db")
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

    def _format_recommendation_as_answer(
        self,
        query_text: str,
        symptoms: list[str],
        result: dict[str, Any],
        safety: dict[str, Any] | None = None,
    ) -> str:
        """将推荐模型的结构化结果格式化为自然语言回答"""
        parts = []

        parts.append("【症状分析】")
        parts.append(
            f"根据您描述的症状：{'、'.join(symptoms) if symptoms else query_text}"
        )

        if result.get("symptoms"):
            parts.append(f"系统识别到：{'、'.join(result['symptoms'])}")

        if result.get("formulas"):
            parts.append("\n【推荐方剂】")
            for i, formula in enumerate(result["formulas"][:5], 1):
                parts.append(
                    f"{i}. {formula['name']}（置信度：{formula['score']:.2f}）"
                )
                if formula.get("composition"):
                    parts.append(f"   组成：{formula['composition']}")
                if formula.get("functions"):
                    parts.append(f"   功效：{formula['functions']}")

        if result.get("herbs"):
            parts.append("\n【推荐药材】")
            for i, herb in enumerate(result["herbs"][:10], 1):
                toxic_tag = " ☠️" if herb.get("is_toxic") else ""
                parts.append(
                    f"{i}. {herb['name']}{toxic_tag}（置信度：{herb['score']:.2f}）"
                )
                if herb.get("efficacy"):
                    parts.append(f"   功效：{herb['efficacy'][:80]}")
                usage = herb.get("usage") or self.get_herb_dosage(herb["name"])
                if usage:
                    parts.append(f"   用量：{usage}")

        parts.append("\n【安全提示】")
        if safety and safety.get("warnings"):
            for w in safety["warnings"]:
                parts.append(w)
            if safety.get("toxic_herbs"):
                toxic_names = "、".join(safety["toxic_herbs"])
                parts.append(f"☠️ 含毒性药材：{toxic_names}，须在医师指导下使用")
            if not safety.get("is_safe"):
                parts.append("⚠️ 检测到配伍禁忌，请勿自行搭配使用！")
        else:
            parts.append("✓ 未检测到常见配伍禁忌")

        parts.append("\n【养生建议】")
        sym_set = set(result.get("symptoms", []) or symptoms)
        parts.extend(generate_health_tips(sym_set))

        return "\n".join(parts)

    def query(self, query_text: str, symptoms: list[str]) -> dict[str, Any]:
        """纯 AI 模式查询（已移除快速模式降级）

        Args:
            query_text: 用户输入的查询文本
            symptoms: 已检测的症状列表

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

        # ========== 智能搜索（增强 AI 结果的补充知识） ==========
        intel_result = self.kb.intelligent_search(query_text)
        intel_knowledge_text = format_intelligent_knowledge_for_display(intel_result)

        # ========== 加载 AI 模型 ==========
        recommender = self._get_recommender()
        if recommender is None:
            raise RuntimeError(
                "AI 模型不可用。请检查：\n"
                "1. PyTorch 是否已安装\n"
                "2. 模型文件 tcm_recommender/best_model.pth 是否存在\n"
                "3. 知识库文件 data/tcm_knowledge.db 是否存在"
            )

        # ========== AI 模型推理（最多重试 2 次） ==========
        answer = None
        knowledge_result = None

        for attempt in range(2):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._run_recommender_inference,
                        recommender,
                        query_text,
                        filtered_symptoms,
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

        if answer is None:
            raise RuntimeError("AI 模型未返回结果，请稍后重试。")

        # ========== 用智能搜索增强 AI 回答 ==========
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

        disclaimer = (
            "温馨提示：本结果基于深度学习模型推荐，仅供参考。实际用药请咨询专业中医师。"
        )

        history_item = self._append_history(
            query_text, filtered_symptoms, final_answer, "ai"
        )
        return {
            "success": True,
            "answer": final_answer,
            "knowledge": knowledge,
            "disclaimer": disclaimer,
            "mode": "ai",
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

        answer = self._format_recommendation_as_answer(
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
        """从数据库查询药材用法用量。"""
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
            pass
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
