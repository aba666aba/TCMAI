"""降级方案模块测试"""
import pytest
from core.fallback import (
    generate_fallback_answer,
    detect_symptoms,
    search_herbs,
    search_formulas,
    extract_keywords,
    is_model_output_valid,
    get_fallback_info,
    TCM_KNOWLEDGE,
    SYNDROMES
)


class TestFallback:
    """降级方案模块测试套件"""

    def test_generate_fallback_answer(self):
        """测试基本降级回答生成
        
        验证：能够生成包含症状分析、推荐方剂和养生建议的回答
        """
        query = "我最近头痛失眠怎么办"
        result = generate_fallback_answer(query)
        
        assert 'answer' in result
        assert 'knowledge' in result
        assert 'disclaimer' in result
        assert 'mode' in result
        assert result['mode'] == 'fallback'
        
        # 验证回答内容包含关键部分
        answer = result['answer']
        assert '【症状分析】' in answer or '【分析】' in answer
        assert '【养生建议】' in answer

    def test_fallback_with_no_kb(self):
        """测试无知识库结果时的降级
        
        验证：即使没有知识库结果，也能生成降级回答
        """
        query = "头痛发热"
        kb_result = None  # 模拟无知识库结果
        
        result = generate_fallback_answer(query, kb_result)
        
        assert result['answer'] is not None
        assert len(result['answer']) > 0
        assert result['mode'] == 'fallback'

    def test_detect_symptoms(self):
        """测试症状检测函数
        
        验证：能够从用户查询中正确识别症状
        """
        # 测试单一症状
        query1 = "我最近头痛得厉害"
        symptoms1 = detect_symptoms(query1)
        assert '头痛' in symptoms1
        
        # 测试多个症状
        query2 = "头痛失眠还有咳嗽"
        symptoms2 = detect_symptoms(query2)
        assert '头痛' in symptoms2
        assert '失眠' in symptoms2
        assert '咳嗽' in symptoms2
        
        # 测试无症状
        query3 = "今天天气不错"
        symptoms3 = detect_symptoms(query3)
        assert len(symptoms3) == 0

    def test_detect_symptoms_variations(self):
        """测试症状检测的变体识别
        
        验证：能够识别症状的不同表达方式
        """
        # 测试"头疼"（头痛的变体）
        query1 = "我头疼"
        symptoms1 = detect_symptoms(query1)
        assert '头痛' in symptoms1
        
        # 测试"睡不着"（失眠的变体）
        query2 = "最近睡不着"
        symptoms2 = detect_symptoms(query2)
        assert '失眠' in symptoms2
        
        # 测试"发烧"（发热的变体）
        query3 = "我发烧了"
        symptoms3 = detect_symptoms(query3)
        assert '发热' in symptoms3

    def test_fallback_has_disclaimer(self):
        """测试降级回答包含免责声明
        
        验证：生成的回答中包含医疗免责声明
        """
        query = "头痛怎么办"
        result = generate_fallback_answer(query)
        
        disclaimer = result.get('disclaimer', '')
        assert '温馨提示' in disclaimer or '医疗' in disclaimer or '专业' in disclaimer
        
        # 免责声明也应该在 answer 中
        answer = result['answer']
        assert len(answer) > 0

    def test_search_herbs(self):
        """测试搜索药材函数
        
        验证：能够根据关键词搜索内置药材知识
        """
        # 搜索人参
        results = search_herbs('人参')
        assert len(results) > 0
        assert any('人参' in herb['name'] for herb in results)
        
        # 搜索功效关键词
        results2 = search_herbs('补气')
        assert len(results2) > 0

    def test_search_formulas(self):
        """测试搜索方剂函数
        
        验证：能够根据关键词搜索内置方剂知识
        """
        # 搜索四君子汤
        results = search_formulas('四君子汤')
        assert len(results) > 0
        assert any('四君子汤' in formula['name'] for formula in results)
        
        # 搜索功效关键词
        results2 = search_formulas('健脾')
        assert len(results2) > 0

    def test_extract_keywords(self):
        """测试关键词提取函数
        
        验证：能够从查询中提取中医药关键词
        """
        query = "人参黄芪补气健脾"
        keywords = extract_keywords(query)
        
        assert '人参' in keywords
        assert '黄芪' in keywords
        assert '补气' in keywords
        assert '健脾' in keywords

    def test_is_model_output_valid(self):
        """测试模型输出有效性检查
        
        验证：能够正确判断模型输出是否有效
        """
        # 有效输出
        assert is_model_output_valid("这是一个有效的回答") is True
        assert is_model_output_valid("中医药知识") is True
        
        # 无效输出 - 空字符串
        assert is_model_output_valid("") is False
        assert is_model_output_valid("   ") is False
        
        # 无效输出 - 无汉字
        assert is_model_output_valid("abc 123") is False
        
        # 无效输出 - 重复字符
        assert is_model_output_valid("啊啊啊啊啊啊啊啊啊啊") is False

    def test_get_fallback_info_herb(self):
        """测试获取药材降级信息
        
        验证：能够根据药材名获取相关信息
        """
        result = get_fallback_info('人参')
        
        assert result['found'] is True
        assert result['type'] == 'herb'
        assert result['data'] is not None
        assert result['data']['name'] == '人参'

    def test_get_fallback_info_formula(self):
        """测试获取方剂降级信息
        
        验证：能够根据方剂名获取相关信息
        """
        result = get_fallback_info('四君子汤')
        
        assert result['found'] is True
        assert result['type'] == 'formula'
        assert result['data'] is not None
        assert result['data']['name'] == '四君子汤'

    def test_get_fallback_info_symptom(self):
        """测试获取症状降级信息
        
        验证：能够根据症状名获取相关信息
        """
        result = get_fallback_info('头痛')
        
        assert result['found'] is True
        assert result['type'] == 'symptom'
        assert result['data'] is not None

    def test_get_fallback_info_not_found(self):
        """测试获取不存在的主题信息
        
        验证：对于不存在的主题返回 found=False
        """
        result = get_fallback_info('不存在的药材')
        
        assert result['found'] is False
        assert result['type'] is None
        assert result['data'] is None

    def test_tcm_knowledge_structure(self):
        """测试内置知识库结构
        
        验证：TCM_KNOWLEDGE 包含预期的数据结构
        """
        assert 'herbs' in TCM_KNOWLEDGE
        assert 'formulas' in TCM_KNOWLEDGE
        assert 'symptoms' in TCM_KNOWLEDGE
        
        # 验证药材数据结构
        if TCM_KNOWLEDGE['herbs']:
            herb = TCM_KNOWLEDGE['herbs'][0]
            assert 'name' in herb
            assert 'efficacy' in herb
        
        # 验证方剂数据结构
        if TCM_KNOWLEDGE['formulas']:
            formula = TCM_KNOWLEDGE['formulas'][0]
            assert 'name' in formula
            assert 'composition' in formula

    def test_syndromes_structure(self):
        """测试证型数据结构
        
        验证：SYNDROMES 包含预期的数据结构
        """
        assert '脾胃虚弱' in SYNDROMES
        
        syndrome = SYNDROMES['脾胃虚弱']
        assert 'description' in syndrome
        assert 'symptoms' in syndrome
        assert 'treatment' in syndrome
        assert 'formula' in syndrome
