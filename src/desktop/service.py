import concurrent.futures
import json
import logging
import msvcrt
import os
import re
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

from ai.config import ModelConfig
from ai.infer import ModelInference
from core.data_desensitize import DataDesensitizer
from core.fallback import generate_fallback_answer
from core.knowledge_base import KnowledgeBase
from core.result_verify import ResultVerifier
from core.symptom_analysis import SymptomAnalyzer

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
        self.analyzer = SymptomAnalyzer()
        self.verifier = ResultVerifier()
        self.desensitizer = DataDesensitizer()
        model_config = ModelConfig()
        self.model = ModelInference(model_config.model_path, model_config.__dict__)
        self.history_lock = threading.Lock()
        self.history_file = self._resolve_history_file()
        self.history_lock_file = self.history_file.with_suffix(".lock")
        self.history_data = self.load_history()

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
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

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

    def query(self, query_text: str, symptoms: List[str], answer_mode: str = "fast") -> Dict[str, Any]:
        if not query_text or len(query_text.strip()) < 1 or len(query_text) > 1000:
            raise ValueError("查询内容长度需要在 1-1000 个字符之间")
        if not isinstance(symptoms, list):
            raise ValueError("症状参数必须为列表")
        if len(symptoms) > 20:
            raise ValueError("症状列表最多支持 20 项")

        self.analyzer.analyze(symptoms)
        kb_result = self.kb.search(query_text)
        answer = None
        knowledge: Any = {}
        disclaimer = ""
        mode = answer_mode

        if answer_mode == "ai" and self.model.model_available:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.model.infer, query_text, str(kb_result))
                    answer = future.result(timeout=10)
            except concurrent.futures.TimeoutError:
                mode = "fast"
            except Exception as exc:
                logger.error("AI 推理失败: %s", exc)
                mode = "fast"

            if answer is not None:
                verified = self.verifier.verify(answer, kb_result)
                answer = verified["answer"]
                knowledge = verified.get("knowledge", {})
                disclaimer = verified.get("disclaimer", "")
                mode = "ai"

        if mode == "fast" or answer is None:
            fallback = generate_fallback_answer(query_text, kb_result)
            answer = fallback["answer"]
            knowledge = format_knowledge_for_display(kb_result)
            disclaimer = fallback.get("disclaimer", "")
            mode = "fast"

        history_item = self._append_history(query_text, symptoms, answer, mode)
        return {
            "success": True,
            "answer": answer,
            "knowledge": knowledge,
            "disclaimer": disclaimer,
            "mode": mode,
            "history_item": history_item,
        }

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
            self.model.unload()
        except Exception:
            pass
