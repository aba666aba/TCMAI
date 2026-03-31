/**
 * 历史记录页面交互脚本
 */

// DOM 元素
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const dateFilter = document.getElementById('dateFilter');
const exportBtn = document.getElementById('exportBtn');
const clearBtn = document.getElementById('clearBtn');
const historyList = document.getElementById('historyList');
const emptyState = document.getElementById('emptyState');

// 本地存储历史记录（用于搜索）
let history_data = [];

// 加载历史记录
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.success && data.history.length > 0) {
            history_data = data.history;
            renderHistory(data.history);
            emptyState.style.display = 'none';
            historyList.style.display = 'flex';
        } else {
            history_data = [];
            emptyState.style.display = 'block';
            historyList.style.display = 'none';
        }
    } catch (error) {
        console.error('加载历史失败:', error);
    }
}

// 渲染历史记录
function renderHistory(history) {
    historyList.innerHTML = '';
    
    history.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <div class="history-header">
                <span class="history-date">${item.date}</span>
                <span class="history-type">${item.type}</span>
            </div>
            <div class="history-content">
                <div class="history-query">
                    <strong>问：</strong>
                    <span>${item.query}</span>
                </div>
                <div class="history-answer">
                    <strong>答：</strong>
                    <span>${item.answer}</span>
                </div>
            </div>
            <div class="history-actions">
                <button class="view-btn">查看详情</button>
                <button class="delete-btn">删除</button>
            </div>
        `;
        
        // 绑定事件
        historyItem.querySelector('.view-btn').addEventListener('click', () => {
            viewDetail(item);
        });
        
        historyItem.querySelector('.delete-btn').addEventListener('click', () => {
            deleteHistory(item.id);
        });
        
        historyList.appendChild(historyItem);
    });
}

// 查看详情
function viewDetail(item) {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>历史记录详情</h3>
                <span class="close">&times;</span>
            </div>
            <div class="modal-body">
                <p><strong>时间：</strong>${item.timestamp}</p>
                <p><strong>症状：</strong>${item.symptoms ? item.symptoms.join('、') : '无'}</p>
                <p><strong>问题：</strong>${item.query}</p>
                <p><strong>回答：</strong>${item.answer}</p>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // 关闭按钮事件
    modal.querySelector('.close').addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    // 点击外部关闭
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            document.body.removeChild(modal);
        }
    });
}

// 搜索功能
searchBtn.addEventListener('click', () => {
    const keyword = searchInput.value.trim().toLowerCase();
    if (!keyword) {
        loadHistory();
        return;
    }
    
    const filtered = history_data.filter(item =>
        item.query.toLowerCase().includes(keyword) ||
        item.answer.toLowerCase().includes(keyword)
    );
    renderHistory(filtered);
});

// 删除历史记录
async function deleteHistory(id) {
    if (!confirm('确定要删除这条历史记录吗？')) {
        return;
    }

    try {
        const response = await fetch(`/api/history/${id}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            loadHistory();
        } else {
            alert('删除失败：' + data.error);
        }
    } catch (error) {
        console.error('删除失败:', error);
        alert('删除失败，请重试');
    }
}

// 导出功能
exportBtn.addEventListener('click', async () => {
    try {
        // 先获取历史记录
        const response = await fetch('/api/history');
        const data = await response.json();

        if (!data.success || data.history.length === 0) {
            alert('没有可导出的历史记录');
            return;
        }

        // 创建导出内容
        const exportData = {
            export_time: new Date().toISOString(),
            total_count: data.history.length,
            history: data.history
        };

        // 创建 Blob 并下载
        const blob = new Blob([JSON.stringify(exportData, null, 2)], 
            { type: 'application/json; charset=utf-8' });
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        const filename = `history_export_${timestamp}.json`;
        
        // 创建下载链接
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        alert('导出成功！文件将下载到您的默认下载文件夹');
    } catch (error) {
        console.error('导出失败:', error);
        alert('导出失败：' + error.message);
    }
});

// 清空历史
clearBtn.addEventListener('click', async () => {
    if (!confirm('确定要清空所有历史记录吗？')) {
        return;
    }
    
    try {
        const response = await fetch('/api/history', {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            loadHistory();
        }
    } catch (error) {
        console.error('清空失败:', error);
    }
});

// 页面加载完成
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
});
