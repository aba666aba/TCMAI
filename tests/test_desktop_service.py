import json
import shutil
import sqlite3
import threading
from pathlib import Path

import pytest

from desktop.service import TCMDesktopService


TEST_ROOT = Path(__file__).resolve().parent / "_tmp_desktop_tests"


def create_case_dir(name: str) -> Path:
    case_dir = TEST_ROOT / name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def create_db_file(base_dir: Path) -> None:
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "tcm_knowledge.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE herbs (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            nature TEXT,
            flavor TEXT,
            meridian TEXT,
            efficacy TEXT,
            indications TEXT,
            usage TEXT,
            contraindications TEXT
        );
        CREATE TABLE formulas (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            source_book TEXT,
            composition TEXT,
            dosage TEXT,
            functions TEXT,
            indications TEXT,
            usage TEXT,
            modification TEXT,
            explanation TEXT
        );
        CREATE TABLE patent_medicines (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            composition TEXT,
            functions TEXT,
            indications TEXT,
            usage_dosage TEXT,
            contraindications TEXT
        );
        CREATE TABLE folk_remedies (
            id INTEGER PRIMARY KEY,
            disease TEXT,
            remedy TEXT,
            ingredients TEXT,
            method TEXT,
            source TEXT
        );
        INSERT INTO herbs (id, name, efficacy, indications) VALUES (1, '人参', '补气', '乏力');
        INSERT INTO formulas (id, name, composition, functions, indications) VALUES (1, '四君子汤', '人参', '健脾', '乏力');
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture
def desktop_service(monkeypatch):
    case_dir = create_case_dir("service_case")
    monkeypatch.setenv("TCMAI_ROOT", str(case_dir))
    create_db_file(case_dir)
    service = TCMDesktopService()
    yield service
    service.close()
    shutil.rmtree(case_dir, ignore_errors=True)


def test_history_is_desensitized(desktop_service):
    result = desktop_service.query(
        "我叫张三，手机号13812345678，最近乏力失眠怎么办",
        ["乏力", "失眠"],
        "fast",
    )
    item = result["history_item"]
    assert "13812345678" not in item["query"]
    assert "张三" not in item["query"]


def test_corrupt_history_is_recovered(monkeypatch):
    case_dir = create_case_dir("corrupt_case")
    monkeypatch.setenv("TCMAI_ROOT", str(case_dir))
    create_db_file(case_dir)
    history_path = case_dir / "data" / "history.json"
    history_path.write_text("{bad json", encoding="utf-8")

    service = TCMDesktopService()
    try:
        assert service.list_history() == []
        assert (case_dir / "data" / "history.corrupt.json").exists()
    finally:
        service.close()
        shutil.rmtree(case_dir, ignore_errors=True)


def test_atomic_threaded_history_appends(desktop_service):
    def worker(index: int):
        desktop_service.query(f"第{index}次查询，最近乏力", ["乏力"], "fast")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    history = desktop_service.list_history()
    assert len(history) == 40
    assert len({item["id"] for item in history}) == 40
    stored = json.loads(desktop_service.history_file.read_text(encoding="utf-8"))
    assert len(stored) == 40


def test_query_rejects_oversized_input(desktop_service):
    with pytest.raises(ValueError):
        desktop_service.query("a" * 1001, [], "fast")


def test_export_history_is_bounded(desktop_service):
    for index in range(desktop_service.MAX_HISTORY_ITEMS + 25):
        desktop_service.query(f"第{index}次查询，最近乏力", ["乏力"], "fast")
    exported = desktop_service.export_history()
    assert len(exported) == desktop_service.MAX_HISTORY_ITEMS
