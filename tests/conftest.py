import sqlite3
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_query():
    return "头痛发热怎么办"


@pytest.fixture
def sample_symptoms():
    return ["头痛", "发热", "咳嗽"]


@pytest.fixture
def memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS herbs (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            nature TEXT,
            flavor TEXT,
            meridian TEXT,
            efficacy TEXT,
            indications TEXT,
            usage TEXT,
            contraindications TEXT
        );
        CREATE TABLE IF NOT EXISTS formulas (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
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
        CREATE TABLE IF NOT EXISTS patent_medicines (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            composition TEXT,
            functions TEXT,
            indications TEXT,
            usage_dosage TEXT,
            contraindications TEXT
        );
        CREATE TABLE IF NOT EXISTS folk_remedies (
            id INTEGER PRIMARY KEY,
            disease TEXT,
            remedy TEXT,
            ingredients TEXT,
            method TEXT,
            source TEXT
        );
        INSERT OR IGNORE INTO herbs (id, name, category, nature, flavor, meridian, efficacy, indications)
        VALUES (1, '人参', '补气药', '微温', '甘、微苦', '脾、肺、心经', '大补元气，补脾益肺', '体虚欲脱，气短乏力');
        INSERT OR IGNORE INTO herbs (id, name, category, nature, flavor, meridian, efficacy, indications)
        VALUES (2, '黄芪', '补气药', '微温', '甘', '脾、肺经', '补气升阳，益卫固表', '气虚乏力，食少便溏');
        INSERT OR IGNORE INTO formulas (id, name, category, source_book, composition, functions, indications)
        VALUES (1, '四君子汤', '补益剂', '太平惠民和剂局方', '人参、白术、茯苓、甘草', '益气健脾', '脾胃气虚，面色萎白');
        INSERT OR IGNORE INTO patent_medicines (id, name, category, composition, functions, indications)
        VALUES (1, '六味地黄丸', '补益类', '熟地黄、山茱萸、山药、泽泻、牡丹皮、茯苓', '滋阴补肾', '肾阴虚证，腰膝酸软');
        INSERT OR IGNORE INTO folk_remedies (id, disease, remedy, ingredients, method, source)
        VALUES (1, '感冒', '姜汤', '生姜3片，红糖适量', '生姜加水煎煮，加红糖，趁热饮用', '民间验方');
        """
    )
    yield conn
    conn.close()

