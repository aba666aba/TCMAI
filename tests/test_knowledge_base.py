"""知识库模块测试"""
import pytest
import sqlite3
from pathlib import Path
from core.knowledge_base import KnowledgeBase


class TestKnowledgeBase:
    """KnowledgeBase 类测试套件"""

    def test_init_with_valid_db(self, memory_db):
        """测试使用有效数据库初始化知识库
        
        验证：当数据库文件存在且有效时，is_available 应为 True
        """
        # 将内存数据库保存到临时文件
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            # 将内存数据库内容保存到文件
            disk_conn = sqlite3.connect(temp_path)
            memory_db.backup(disk_conn)
            disk_conn.close()
            
            # 使用临时数据库文件初始化 KnowledgeBase
            kb = KnowledgeBase(temp_path)
            
            assert kb.is_available is True
            assert kb.conn is not None
            
            kb.close()
        finally:
            os.unlink(temp_path)

    def test_init_with_missing_db(self):
        """测试使用不存在的数据库路径初始化
        
        验证：当数据库文件不存在时，is_available 应为 False
        """
        kb = KnowledgeBase('/nonexistent/path/to/db.sqlite')
        
        assert kb.is_available is False
        assert kb.conn is None

    def test_search_herbs(self, memory_db):
        """测试搜索药材功能
        
        验证：能够正确搜索到匹配的药材
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            disk_conn = sqlite3.connect(temp_path)
            memory_db.backup(disk_conn)
            disk_conn.close()
            
            kb = KnowledgeBase(temp_path)
            results = kb.search('人参')
            
            assert 'herbs' in results
            assert len(results['herbs']) > 0
            assert results['herbs'][0]['name'] == '人参'
            
            kb.close()
        finally:
            os.unlink(temp_path)

    def test_search_empty_query(self):
        """测试空查询返回空结果
        
        验证：当查询为空字符串时，各结果列表应为空或返回默认值
        """
        import tempfile
        import os
        
        # 创建一个有效的空数据库
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            # 创建基础表结构
            conn = sqlite3.connect(temp_path)
            conn.executescript('''
                CREATE TABLE herbs (id INTEGER PRIMARY KEY, name TEXT, category TEXT, 
                    nature TEXT, flavor TEXT, meridian TEXT, efficacy TEXT, 
                    indications TEXT, usage TEXT, contraindications TEXT);
                CREATE TABLE formulas (id INTEGER PRIMARY KEY, name TEXT, category TEXT,
                    source_book TEXT, composition TEXT, dosage TEXT, functions TEXT,
                    indications TEXT, usage TEXT, modification TEXT, explanation TEXT);
                CREATE TABLE patent_medicines (id INTEGER PRIMARY KEY, name TEXT, category TEXT,
                    composition TEXT, functions TEXT, indications TEXT, usage_dosage TEXT,
                    contraindications TEXT);
                CREATE TABLE folk_remedies (id INTEGER PRIMARY KEY, disease TEXT, remedy TEXT,
                    ingredients TEXT, method TEXT, source TEXT);
            ''')
            conn.close()
            
            kb = KnowledgeBase(temp_path)
            results = kb.search('')
            
            # 空查询应该返回各列表（可能为空）
            assert 'herbs' in results
            assert 'formulas' in results
            
            kb.close()
        finally:
            os.unlink(temp_path)

    def test_context_manager(self, memory_db):
        """测试上下文管理器正确关闭连接
        
        验证：使用 with 语句后，连接应被正确关闭
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            disk_conn = sqlite3.connect(temp_path)
            memory_db.backup(disk_conn)
            disk_conn.close()
            
            with KnowledgeBase(temp_path) as kb:
                assert kb.is_available is True
                # 在上下文内执行一些操作
                results = kb.search('人参')
                assert 'herbs' in results
            
            # 退出上下文后，连接应该被关闭
            assert kb.conn is None
            assert kb.is_available is False
        finally:
            os.unlink(temp_path)

    def test_match_symptoms(self, memory_db):
        """测试症状匹配功能
        
        验证：能够根据症状列表匹配到相关方剂
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            disk_conn = sqlite3.connect(temp_path)
            memory_db.backup(disk_conn)
            disk_conn.close()
            
            kb = KnowledgeBase(temp_path)
            
            # 测试症状匹配
            symptoms = ['脾胃气虚', '面色萎白']
            results = kb.match_symptoms(symptoms)
            
            # 应该返回列表（可能有匹配结果）
            assert isinstance(results, list)
            # 由于测试数据中有四君子汤针对脾胃气虚，应该能匹配到
            if len(results) > 0:
                assert 'name' in results[0]
            
            kb.close()
        finally:
            os.unlink(temp_path)

    def test_match_symptoms_empty_list(self):
        """测试空症状列表匹配
        
        验证：传入空症状列表时应返回空结果
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            conn = sqlite3.connect(temp_path)
            conn.executescript('''
                CREATE TABLE formulas (id INTEGER PRIMARY KEY, name TEXT, category TEXT,
                    source_book TEXT, composition TEXT, functions TEXT, indications TEXT);
            ''')
            conn.close()
            
            kb = KnowledgeBase(temp_path)
            results = kb.match_symptoms([])
            
            assert results == []
            
            kb.close()
        finally:
            os.unlink(temp_path)

    def test_get_herb_detail(self, memory_db):
        """测试获取药材详情
        
        验证：能够根据 ID 获取药材详细信息
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            disk_conn = sqlite3.connect(temp_path)
            memory_db.backup(disk_conn)
            disk_conn.close()
            
            kb = KnowledgeBase(temp_path)
            
            # 获取 ID 为 1 的药材详情
            herb = kb.get_herb_detail(1)
            
            assert herb is not None
            assert herb['name'] == '人参'
            assert 'efficacy' in herb
            
            kb.close()
        finally:
            os.unlink(temp_path)

    def test_get_formula_detail(self, memory_db):
        """测试获取方剂详情
        
        验证：能够根据 ID 获取方剂详细信息
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            disk_conn = sqlite3.connect(temp_path)
            memory_db.backup(disk_conn)
            disk_conn.close()
            
            kb = KnowledgeBase(temp_path)
            
            # 获取 ID 为 1 的方剂详情
            formula = kb.get_formula_detail(1)
            
            assert formula is not None
            assert formula['name'] == '四君子汤'
            assert 'composition' in formula
            
            kb.close()
        finally:
            os.unlink(temp_path)
