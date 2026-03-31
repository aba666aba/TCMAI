# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""
前端界面入口

Flask 应用启动文件
"""

from flask import Flask, render_template, request, jsonify
import logging
from core.knowledge_base import KnowledgeBase
from core.symptom_analysis import SymptomAnalyzer
from core.result_verify import ResultVerifier
from core.fallback import generate_fallback_answer
from ai.infer import ModelInference
from ai.config import ModelConfig

logger = logging.getLogger(__name__)


def format_knowledge_for_display(kb_result):
    """
    格式化知识库结果为中文文本
    
    Args:
        kb_result: 知识库查询结果字典
        
    Returns:
        格式化后的中文文本
    """
    if not kb_result:
        return '暂无相关知识'
    
    parts = []
    
    # 处理药材
    if kb_result.get('herbs') and len(kb_result['herbs']) > 0:
        herb_names = [h['name'] for h in kb_result['herbs'][:5]]
        parts.append(f"相关药材：{', '.join(herb_names)}")
    
    # 处理方剂
    if kb_result.get('formulas') and len(kb_result['formulas']) > 0:
        formula_names = [f['name'] for f in kb_result['formulas'][:5]]
        parts.append(f"相关方剂：{', '.join(formula_names)}")
    
    # 处理中成药
    if kb_result.get('patent_medicines') and len(kb_result['patent_medicines']) > 0:
        med_names = [m['name'] for m in kb_result['patent_medicines'][:5]]
        parts.append(f"相关中成药：{', '.join(med_names)}")
    
    # 处理民间验方
    if kb_result.get('folk_remedies') and len(kb_result['folk_remedies']) > 0:
        remedy_names = [r.get('remedy', r.get('disease', '验方')) for r in kb_result['folk_remedies'][:5]]
        parts.append(f"相关民间验方：{', '.join(remedy_names)}")
    
    if parts:
        return '; '.join(parts)
    return '暂无相关知识'


def create_app(config: dict = None) -> Flask:
    """
    创建 Flask 应用

    Args:
        config: 应用配置

    Returns:
        Flask 应用实例
    """
    # 获取模板和静态文件目录
    from pathlib import Path
    base_dir = Path(__file__).parent
    template_dir = base_dir / 'templates'
    static_dir = template_dir / 'static'
    
    app = Flask(__name__,
                template_folder=str(template_dir),
                static_folder=str(static_dir))

    # 优化配置：禁用模板自动重载，启用缓存
    app.config['TEMPLATES_AUTO_RELOAD'] = False
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # 静态文件缓存 1 小时

    # 加载配置
    if config:
        app.config.update(config)

    # 初始化服务
    kb = KnowledgeBase()
    analyzer = SymptomAnalyzer()
    verifier = ResultVerifier()

    # 初始化模型
    model_config = ModelConfig()
    model = ModelInference(model_config.model_path, model_config.__dict__)

    # 检查模型是否可用
    model_available = model.model_available
    if model_available:
        logger.info("AI 模型可用，使用 AI 推理模式")
    else:
        logger.warning("AI 模型不可用，自动启用降级方案（基于规则的中医药知识回答）")

    # 初始化历史记录存储
    import json
    import os
    from datetime import datetime

    history_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'history.json')

    def load_history():
        """加载历史记录"""
        if not os.path.exists(history_file):
            return []
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载历史记录失败：{e}")
            return []

    def save_history(history):
        """保存历史记录"""
        try:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存历史记录失败：{e}")

    # 加载历史记录
    history_data = load_history()

    @app.route('/')
    def index():
        """问答主界面"""
        return render_template('index.html')

    @app.route('/history')
    def history():
        """历史记录页面"""
        return render_template('history.html')

    @app.route('/about')
    def about():
        """关于页面"""
        return render_template('about.html')

    @app.route('/api/query', methods=['POST'])
    def query():
        """
        处理问答请求

        Request JSON:
            {
                "query": "用户问题",
                "symptoms": ["症状列表"],
                "mode": "fast|ai"  # fast=快速模式 (知识库), ai=AI 模式 (大模型推理)
            }

        Response JSON:
            {
                "answer": "AI 回答",
                "knowledge": "相关知识",
                "disclaimer": "合规提示",
                "mode": "fast|ai|fallback"  # 回答模式
            }
        """
        data = request.get_json()
        query_text = data.get('query', '')
        symptoms = data.get('symptoms', [])
        answer_mode = data.get('mode', 'fast')  # 默认快速模式

        logger.info(f"收到查询请求：{query_text[:50]}... [模式：{answer_mode}]")
        logger.info(f"模型对象：{model}")
        logger.info(f"模型可用：{model.model_available if hasattr(model, 'model_available') else 'N/A'}")

        try:
            # 症状分析
            analysis_result = analyzer.analyze(symptoms)

            # 知识库查询
            kb_result = kb.search(query_text)
            logger.info(f"知识库查询完成：{kb_result}")

            # 根据模式选择回答方式
            answer = None
            knowledge = {}
            disclaimer = ''
            mode = answer_mode

            if answer_mode == 'ai' and model.model_available:
                # AI 模式：使用大模型推理
                logger.info(f"开始 AI 模型推理...")
                answer = model.infer(query_text, str(kb_result))
                logger.info(f"模型推理结果：{answer[:50] if answer else 'None'}...")
                
                if answer is None:
                    logger.info("AI 模型输出无效，切换到快速模式")
                    mode = 'fast'
                else:
                    # 结果校验
                    verified_result = verifier.verify(answer, kb_result)
                    answer = verified_result['answer']
                    knowledge = verified_result.get('knowledge', {})
                    disclaimer = verified_result.get('disclaimer', '')
                    mode = 'ai'

            if mode == 'fast' or answer is None:
                # 快速模式：使用知识库 + 规则生成回答
                logger.info("使用快速模式生成回答")
                fallback_result = generate_fallback_answer(query_text, kb_result)
                answer = fallback_result['answer']
                # 格式化知识库结果为中文文本
                knowledge = format_knowledge_for_display(kb_result)
                disclaimer = fallback_result.get('disclaimer', '')
                mode = 'fast'

            # 保存历史记录
            history_item = {
                'id': len(history_data) + 1,
                'query': query_text,
                'symptoms': symptoms,
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
                'mode': mode  # 记录回答模式
            }
            history_data.append(history_item)
            save_history(history_data)

            return jsonify({
                'success': True,
                'answer': answer,
                'knowledge': knowledge,
                'disclaimer': disclaimer,
                'mode': mode  # 返回回答模式
            })

        except Exception as e:
            logger.error(f"查询处理失败：{str(e)}")
            # 即使出错也尝试使用降级方案
            try:
                fallback_result = generate_fallback_answer(query_text, None)
                return jsonify({
                    'success': True,
                    'answer': fallback_result['answer'],
                    'knowledge': fallback_result.get('knowledge', {}),
                    'disclaimer': fallback_result.get('disclaimer', ''),
                    'mode': 'fallback',
                    'error': str(e)  # 附带错误信息
                })
            except Exception as fallback_error:
                logger.error(f"降级方案也失败：{str(fallback_error)}")
                return jsonify({
                    'success': False,
                    'error': f"系统错误：{str(e)}"
                }), 500

    @app.route('/api/history', methods=['GET'])
    def get_history():
        """获取历史记录"""
        try:
            # 按时间倒序排序
            sorted_history = sorted(history_data, key=lambda x: x['timestamp'], reverse=True)
            return jsonify({
                'success': True,
                'history': sorted_history
            })
        except Exception as e:
            logger.error(f"获取历史记录失败：{e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/history/<int:item_id>', methods=['DELETE'])
    def delete_history_item(item_id):
        """删除单条历史记录"""
        try:
            nonlocal history_data
            original_count = len(history_data)
            history_data = [item for item in history_data if item['id'] != item_id]

            if len(history_data) < original_count:
                save_history(history_data)
                return jsonify({
                    'success': True,
                    'message': '删除成功'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '未找到该历史记录'
                }), 404
        except Exception as e:
            logger.error(f"删除历史记录失败：{e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/history', methods=['DELETE'])
    def clear_history():
        """清空所有历史记录"""
        try:
            nonlocal history_data
            history_data = []
            save_history(history_data)
            return jsonify({
                'success': True,
                'message': '清空成功'
            })
        except Exception as e:
            logger.error(f"清空历史记录失败：{e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/export', methods=['POST'])
    def export():
        """导出查询结果"""
        try:
            from .utils import export_result
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'error': '未提供导出数据'
                }), 400
            result = export_result(data)
            return jsonify({'success': True, 'file': result})
        except Exception as e:
            logger.error(f"导出失败：{e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
