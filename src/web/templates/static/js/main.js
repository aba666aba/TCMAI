/**
 * 中医药智能问答系统 - 主交互脚本
 * 原创功能：症状输入校验、问答交互、Markdown 渲染
 */

// 获取 CSRF token 的工具函数
function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : '';
}

// XSS 防护：HTML 转义函数
function escapeHtml(text) {
    if (typeof text !== 'string') return text;
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// XSS 防护：基础 HTML 净化函数
function basicSanitize(html) {
    if (typeof html !== 'string') return html;
    // 移除 script 标签
    html = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
    // 移除 on* 事件处理器
    html = html.replace(/\bon\w+\s*=\s*["'][^"']*["']/gi, '');
    html = html.replace(/\bon\w+\s*=\s*[^\s>]*/gi, '');
    // 移除 javascript: 协议
    html = html.replace(/javascript\s*:/gi, '');
    return html;
}

// DOM 元素
const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const answerSection = document.getElementById('answerSection');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const loadingTips = document.getElementById('loadingTips');
const resultContent = document.getElementById('resultContent');
const answerText = document.getElementById('answerText');
const knowledgeContent = document.getElementById('knowledgeContent');
const disclaimerText = document.getElementById('disclaimerText');
const symptomTags = document.querySelectorAll('.symptom-tags .tag');
const modeRadios = document.querySelectorAll('input[name="answerMode"]');

// 错误提示框
const errorBox = document.createElement('div');
errorBox.id = 'errorBox';
errorBox.className = 'error-box';
errorBox.style.display = 'none';

// 症状输入校验
class SymptomValidator {
    constructor() {
        // 敏感词列表
        this.sensitiveWords = [
            '癌症', '肿瘤', '晚期', '绝症', '死亡'
        ];

        // 症状关键词
        this.symptomKeywords = [
            '乏力', '疲倦', '食欲不振', '失眠', '头痛',
            '咳嗽', '发热', '腹痛', '腹泻', '便秘'
        ];
    }

    // 校验输入
    validate(text) {
        if (!text || text.trim().length === 0) {
            return {
                valid: false,
                message: '请输入您的问题或症状'
            };
        }

        if (text.length > 500) {
            return {
                valid: false,
                message: '输入内容过长，请控制在 500 字以内'
            };
        }

        // 检查敏感词
        for (const word of this.sensitiveWords) {
            if (text.includes(word)) {
                return {
                    valid: false,
                    message: `包含敏感词汇"${word}"，请修改后提交`
                };
            }
        }

        return {
            valid: true,
            message: '输入有效'
        };
    }

    // 提取症状
    // 注意：症状关键词列表需与后端 src/core/fallback.py 中的 detect_symptoms() 保持同步
    // 前端提取用于 UI 展示，后端检测用于实际分析
    extractSymptoms(text) {
        const symptoms = [];
        for (const keyword of this.symptomKeywords) {
            if (text.includes(keyword)) {
                symptoms.push(keyword);
            }
        }
        return symptoms;
    }
}

// Markdown 简单渲染器
class MarkdownRenderer {
    // 将 Markdown 转换为 HTML
    static render(text) {
        if (!text || typeof text !== 'string') {
            return '';
        }

        let html = text;

        // 转义 HTML 特殊字符（防止 XSS）
        html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // 标题
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

        // 粗体
        html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');

        // 斜体
        html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');

        // 列表
        html = html.replace(/^\s*[\-\*]\s+(.*$)/gim, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

        // 换行
        html = html.replace(/\n/g, '<br>');

        // 引用
        html = html.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

        // 代码块
        html = html.replace(/```([\s\S]*?)```/gim, '<pre><code>$1</code></pre>');
        html = html.replace(/`(.*?)`/gim, '<code>$1</code>');

        return html;
    }
}

// 初始化校验器
const validator = new SymptomValidator();

// 标签点击事件
symptomTags.forEach(tag => {
    tag.addEventListener('click', () => {
        const symptom = tag.dataset.symptom;
        const currentText = queryInput.value.trim();

        if (currentText) {
            queryInput.value = currentText + '，' + symptom;
        } else {
            queryInput.value = symptom;
        }

        queryInput.focus();
    });
});

// 获取当前选择的模式
function getSelectedMode() {
    for (const radio of modeRadios) {
        if (radio.checked) {
            return radio.value;
        }
    }
    return 'fast';
}

// 显示加载动画
function showLoading(isAIMode) {
    answerSection.style.display = 'block';
    loading.style.display = 'flex';
    resultContent.style.display = 'none';
    errorBox.style.display = 'none';

    if (isAIMode) {
        // AI 模式：显示思考步骤
        loadingText.textContent = 'AI 正在深入分析您的问题...';
        loadingTips.style.display = 'block';
        resetThinkingSteps();
    } else {
        // 快速模式：简单提示
        loadingText.textContent = '正在检索知识库，请稍候...';
        loadingTips.style.display = 'none';
    }

    // 滚动到回答区域
    answerSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 重置思考步骤动画
function resetThinkingSteps() {
    const steps = document.querySelectorAll('.thinking-steps .step');
    steps.forEach((step, index) => {
        step.className = 'step';
        if (index < 2) {
            step.classList.add('done');
        } else if (index === 2) {
            step.classList.add('active');
        }
    });
}

// 更新思考步骤
function updateThinkingStep(stepIndex) {
    const steps = document.querySelectorAll('.thinking-steps .step');
    steps.forEach((step, index) => {
        step.className = 'step';
        if (index < stepIndex) {
            step.classList.add('done');
        } else if (index === stepIndex) {
            step.classList.add('active');
        }
    });
}

// 隐藏加载动画
function hideLoading() {
    loading.style.display = 'none';
    resultContent.style.display = 'block';
}

// 显示错误
function showError(message) {
    hideLoading();
    errorBox.innerHTML = `
        <div class="error-icon">⚠️</div>
        <h3>出错了</h3>
        <p>${escapeHtml(message)}</p>
        <button onclick="document.getElementById('errorBox').style.display='none'" class="close-error">关闭</button>
    `;
    const resultContainer = document.getElementById('resultContent');
    if (resultContainer && !errorBox.parentNode) {
        resultContainer.parentNode.insertBefore(errorBox, resultContainer);
    }
    errorBox.style.display = 'block';
}

// 显示结果
function displayResult(data) {
    hideLoading();

    // 渲染回答
    if (data.answer && typeof data.answer === 'string') {
        const renderedAnswer = MarkdownRenderer.render(data.answer);
        answerText.innerHTML = basicSanitize(renderedAnswer);
    } else {
        answerText.innerHTML = '<p>抱歉，没有获取到有效回答。</p>';
    }

    // 显示相关知识
    const knowledgeBox = document.getElementById('knowledgeBox');
    if (data.knowledge) {
        let knowledgeText = '';
        
        // 处理不同类型的 knowledge 数据
        if (typeof data.knowledge === 'string') {
            // 字符串直接显示
            knowledgeText = data.knowledge;
        } else if (Array.isArray(data.knowledge)) {
            // 数组：格式化显示
            knowledgeText = formatKnowledgeArray(data.knowledge);
        } else if (typeof data.knowledge === 'object') {
            // 对象：提取有用信息
            knowledgeText = formatKnowledgeObject(data.knowledge);
        }
        
        if (knowledgeText && knowledgeText !== '暂无相关知识') {
            knowledgeContent.innerHTML = basicSanitize(MarkdownRenderer.render(knowledgeText));
            knowledgeBox.style.display = 'block';
        } else {
            knowledgeBox.style.display = 'none';
        }
    } else {
        knowledgeBox.style.display = 'none';
    }

    // 显示免责声明
    disclaimerText.textContent = data.disclaimer || '本回答仅供参考，不构成医疗诊断或治疗建议。如有健康问题请咨询专业医疗机构。';

    // 显示模式标签
    const modeBadge = document.createElement('div');
    modeBadge.className = `mode-badge mode-${data.mode}`;
    modeBadge.innerHTML = data.mode === 'ai' ? '🧠 AI 思考' : '⚡ 快速回答';
    answerText.insertBefore(modeBadge, answerText.firstChild);
}

// 格式化数组类型的知识
function formatKnowledgeArray(arr) {
    if (!arr || arr.length === 0) return '';
    
    let text = '';
    arr.forEach((item, index) => {
        if (typeof item === 'string') {
            text += `${index + 1}. ${item}\n`;
        } else if (item.name) {
            text += `${index + 1}. **${item.name}**`;
            if (item.efficacy) text += ` - ${item.efficacy}`;
            if (item.functions) text += ` - ${item.functions}`;
            text += '\n';
        }
    });
    return text;
}

// 格式化对象类型的知识
function formatKnowledgeObject(obj) {
    if (!obj) return '';
    
    // 兼容后端统一类型后的文本包装格式
    if (obj.display_text && typeof obj.display_text === 'string') {
        return obj.display_text;
    }
    
    let text = '';
    
    // 处理 herbs（药材）
    if (obj.herbs && obj.herbs.length > 0) {
        text += '**相关药材**：\n';
        obj.herbs.slice(0, 5).forEach((herb, i) => {
            text += `${i + 1}. **${herb.name}** - ${herb.efficacy || ''}\n`;
        });
        text += '\n';
    }
    
    // 处理 formulas（方剂）
    if (obj.formulas && obj.formulas.length > 0) {
        text += '**相关方剂**：\n';
        obj.formulas.slice(0, 5).forEach((formula, i) => {
            text += `${i + 1}. **${formula.name}** - ${formula.functions || formula.efficacy || ''}\n`;
        });
        text += '\n';
    }
    
    // 处理 patent_medicines（中成药）
    if (obj.patent_medicines && obj.patent_medicines.length > 0) {
        text += '**相关中成药**：\n';
        obj.patent_medicines.slice(0, 5).forEach((med, i) => {
            text += `${i + 1}. **${med.name}** - ${med.functions || ''}\n`;
        });
        text += '\n';
    }
    
    // 处理 folk_remedies（民间验方）
    if (obj.folk_remedies && obj.folk_remedies.length > 0) {
        text += '**相关民间验方**：\n';
        obj.folk_remedies.slice(0, 5).forEach((remedy, i) => {
            text += `${i + 1}. **${remedy.remedy || remedy.disease}** - ${remedy.ingredients || ''}\n`;
        });
        text += '\n';
    }
    
    // 处理 efficacy（功效搜索结果）
    if (obj.efficacy && obj.efficacy.length > 0) {
        text += '**功效相关**：\n';
        obj.efficacy.slice(0, 5).forEach((item, i) => {
            text += `${i + 1}. ${item.name}（${item.type === 'herb' ? '药材' : item.type === 'formula' ? '方剂' : '中成药'}） - ${item.efficacy || ''}\n`;
        });
    }
    
    // 处理 symptoms（症状）
    if (obj.symptoms && obj.symptoms.length > 0) {
        text += '**相关症状**：' + obj.symptoms.join('、') + '\n';
    }
    
    // 处理 source（来源）
    if (obj.source && obj.source !== 'fallback_mode') {
        text += `\n**来源**：${obj.source}\n`;
    }
    
    return text.trim() || '暂无相关知识';
}

// 提交按钮事件
submitBtn.addEventListener('click', async () => {
    const query = queryInput.value.trim();

    // 校验输入
    const validation = validator.validate(query);
    if (!validation.valid) {
        alert(validation.message);
        return;
    }

    // 获取选择的模式
    const mode = getSelectedMode();
    const isAIMode = mode === 'ai';

    // 禁用按钮防止重复提交
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-small"></span> 正在分析...';

    // 显示加载动画
    showLoading(isAIMode);

    // AI 模式：模拟思考步骤动画
    let thinkingInterval = null;
    if (isAIMode) {
        let currentStep = 0;
        thinkingInterval = setInterval(() => {
            currentStep++;
            if (currentStep <= 3) {
                updateThinkingStep(currentStep);
            }
        }, 1500);
    }

    try {
        // 发送请求
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                symptoms: validator.extractSymptoms(query),
                mode: mode  // 发送选择的模式
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP 错误：${response.status}`);
        }

        const data = await response.json();
        console.log('收到响应数据:', data);

        // 清除思考动画
        if (thinkingInterval) {
            clearInterval(thinkingInterval);
        }

        if (data.success) {
            displayResult(data);
        } else {
            showError(data.error || '查询失败，未知错误');
        }
    } catch (error) {
        console.error('请求错误:', error);
        showError('网络错误：' + error.message + '，请稍后重试');

        // 清除思考动画
        if (thinkingInterval) {
            clearInterval(thinkingInterval);
        }
    } finally {
        // 恢复按钮状态
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>🔍</span> 提交查询';
    }
});

// 回车键提交（Ctrl+Enter）
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        submitBtn.click();
    }
});

// 页面加载完成
document.addEventListener('DOMContentLoaded', () => {
    console.log('🌿 中医药智能问答系统已加载');

    // 添加错误盒到 DOM
    const resultContainer = document.getElementById('resultContent');
    if (resultContainer) {
        resultContainer.parentNode.insertBefore(errorBox, resultContainer);
    }
});
