import concurrent.futures
import json
import logging
import os
import platform
import re
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

from core.data_desensitize import DataDesensitizer
from core.fallback import generate_fallback_answer
from core.knowledge_base import KnowledgeBase
from core.result_verify import ResultVerifier
from core.symptom_pipeline import SymptomPipeline

logger = logging.getLogger(__name__)


def format_knowledge_for_display(kb_result: Dict[str, Any]) -> str:
    if not kb_result:
        return "暂无相关知识"

    parts: List[str] = []
    if kb_result.get("herbs"):
        parts.append(f"相关药材：{', '.join(item['name'] for item in kb_result['herbs'][:5])}")
    if kb_result.get("formulas"):
        parts.append(f"相关方剂：{', '.join(item['name'] for item in kb_result['formulas'][:5])}")
    if kb_result.get("patent_medicines"):
        parts.append(
            f"相关中成药：{', '.join(item['name'] for item in kb_result['patent_medicines'][:5])}"
        )
    if kb_result.get("folk_remedies"):
        parts.append(
            "相关民间验方："
            + ", ".join(
                item.get("remedy", item.get("disease", "验方"))
                for item in kb_result["folk_remedies"][:5]
            )
        )

    return "；".join(parts) if parts else "暂无相关知识"


class TCMDesktopService:
    MAX_HISTORY_ITEMS = 500

    def __init__(self) -> None:
        self.kb = KnowledgeBase()
        self.pipeline = SymptomPipeline()
        self.verifier = ResultVerifier()
        self.desensitizer = DataDesensitizer()
        self.history_lock = threading.Lock()
        self.history_file = self._resolve_history_file()
        self.history_lock_file = self.history_file.with_suffix(".lock")
        self.history_data = self.load_history()
        self._recommender = None
        self._recommender_load_attempted = False
        self._recommender_available = False

    def _resolve_history_file(self) -> Path:
        root = Path(os.environ.get("TCMAI_ROOT", Path(__file__).resolve().parents[2]))
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

    def _safe_desensitize(self, value: str) -> str:
        text = value or ""
        text = self.desensitizer.remove_pii(text)
        text = self.desensitizer.anonymize_query(text)
        text = re.sub(r"(我叫|我是|姓名是)([\u4e00-\u9fff]{2,4})", r"\1**", text)
        text = re.sub(r"患者[：: ]?([\u4e00-\u9fff]{2,4})", "患者**", text)
        return text[:4000]

    def _sanitize_history_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        symptoms = item.get("symptoms", [])
        if not isinstance(symptoms, list):
            symptoms = []
        return {
            "id": int(item.get("id", 0)),
            "query": self._safe_desensitize(str(item.get("query", ""))),
            "symptoms": [self._safe_desensitize(str(symptom))[:64] for symptom in symptoms[:20]],
            "answer": self._safe_desensitize(str(item.get("answer", ""))),
            "timestamp": str(item.get("timestamp", ""))[:32],
            "mode": "ai" if item.get("mode") == "ai" else "fast",
        }

    def _normalize_history(self, data: Any) -> List[Dict[str, Any]]:
        if not isinstance(data, list):
            return []
        normalized: List[Dict[str, Any]] = []
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

    def load_history(self) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            return []
        try:
            with self._acquire_file_lock():
                with open(self.history_file, "r", encoding="utf-8") as handle:
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
            sanitized = self._normalize_history(self.history_data)[-self.MAX_HISTORY_ITEMS :]
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
            from tcm_recommender.recommend import TCMRecommender

            db_path = os.path.join(
                os.environ.get("TCMAI_ROOT", str(Path(__file__).resolve().parents[2])),
                "data",
                "tcm_knowledge.db",
            )
            model_path = os.path.join(
                os.environ.get("TCMAI_ROOT", str(Path(__file__).resolve().parents[2])),
                "tcm_recommender",
                "best_model.pth",
            )
            if not os.path.exists(db_path) or not os.path.exists(model_path):
                logger.warning("推荐模型或数据库文件不存在，AI 模式不可用")
                return None

            self._recommender = TCMRecommender(
                model_path=model_path,
                db_path=db_path,
                device="cpu",
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
        symptoms: List[str],
        result: Dict[str, Any],
        safety: Optional[Dict[str, Any]] = None,
    ) -> str:
        """将推荐模型的结构化结果格式化为自然语言回答"""
        parts = []

        parts.append("【症状分析】")
        parts.append(f"根据您描述的症状：{'、'.join(symptoms) if symptoms else query_text}")

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
        parts.append("• 饮食调理：饮食宜清淡，避免生冷油腻")
        parts.append("• 起居调摄：保持规律作息，避免熬夜")
        parts.append("• 情志调畅：保持心情舒畅，避免过度思虑")

        return "\n".join(parts)

    def query(self, query_text: str, symptoms: List[str], answer_mode: str = "ai") -> Dict[str, Any]:
        if not query_text or len(query_text.strip()) < 1 or len(query_text) > 1000:
            raise ValueError("查询内容长度需要在 1-1000 个字符之间")
        if not isinstance(symptoms, list):
            raise ValueError("症状参数必须为列表")
        if len(symptoms) > 20:
            raise ValueError("症状列表最多支持 20 项")

        filtered_symptoms = [s for s in symptoms if isinstance(s, str) and s.strip()]
        _pipeline_result = self.pipeline.process(query_text, filtered_symptoms)
        kb_result = self.kb.search(query_text)
        answer = None
        knowledge: Any = {}
        disclaimer = ""
        mode = answer_mode

        if answer_mode == "ai":
            recommender = self._get_recommender()
            if recommender is not None:
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            self._run_recommender_inference,
                            recommender,
                            query_text,
                            filtered_symptoms,
                        )
                        answer, knowledge_result = future.result(timeout=15)
                except concurrent.futures.TimeoutError:
                    logger.warning("推荐模型推理超时，降级到快速模式")
                    mode = "fast"
                except Exception as exc:
                    logger.error("推荐模型推理失败: %s", exc)
                    mode = "fast"

                if answer is not None:
                    mode = "ai"
                    knowledge = knowledge_result
                    disclaimer = "温馨提示：本结果基于深度学习模型推荐，仅供参考。实际用药请咨询专业中医师。"

        if mode == "fast" or answer is None:
            fallback = generate_fallback_answer(query_text, kb_result)
            answer = fallback["answer"]
            knowledge = format_knowledge_for_display(kb_result)
            disclaimer = fallback.get("disclaimer", "")
            mode = "fast"

        history_item = self._append_history(query_text, filtered_symptoms, answer, mode)
        return {
            "success": True,
            "answer": answer,
            "knowledge": knowledge,
            "disclaimer": disclaimer,
            "mode": mode,
            "history_item": history_item,
        }

    def _run_recommender_inference(
        self, recommender, query_text: str, symptoms: List[str]
    ):
        """在独立线程中运行推荐模型推理"""
        symptom_ids, valid, unknown = recommender.parse_symptoms(
            " ".join(symptoms) if symptoms else query_text
        )
        if not symptom_ids:
            return None, None

        result = recommender.recommend(symptom_ids, top_k_herb=10, top_k_formula=5)

        # 安全校验
        safety_result: Optional[Dict[str, Any]] = None
        try:
            herb_names = [h["name"] for h in result.get("herbs", [])[:10]]
            safety_result = recommender.check_safety(herb_names)
        except Exception:
            pass

        answer = self._format_recommendation_as_answer(
            query_text, symptoms, result, safety_result
        )
        knowledge = {
            "symptoms": result.get("symptoms", []),
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

    def _append_history(self, query_text: str, symptoms: List[str], answer: str, mode: str) -> Dict[str, Any]:
        with self.history_lock:
            next_id = max((item.get("id", 0) for item in self.history_data), default=0) + 1
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

    def list_history(self) -> List[Dict[str, Any]]:
        return sorted(
            self.history_data[-self.MAX_HISTORY_ITEMS :],
            key=lambda item: item.get("timestamp", ""),
            reverse=True,
        )

    def export_history(self) -> List[Dict[str, Any]]:
        return self.list_history()

    def delete_history_item(self, item_id: int) -> bool:
        with self.history_lock:
            before = len(self.history_data)
            self.history_data[:] = [item for item in self.history_data if item.get("id") != item_id]
            changed = len(self.history_data) != before
            if changed:
                self.save_history()
            return changed

    def clear_history(self) -> None:
        with self.history_lock:
            self.history_data.clear()
            self.save_history()

    def close(self) -> None:
        try:
            self.kb.close()
        except Exception:
            pass
        try:
            if self._recommender is not None:
                self._recommender.close()
        except Exception:
            pass
