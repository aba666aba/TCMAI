# 中药处方推荐系统使用说明

## 系统概述

中药处方推荐系统是一个基于图神经网络的智能推荐工具，可以根据输入的症状描述推荐相应的中药处方。

## 环境要求

- Python 3.7+
- PyTorch
- PyTorch Geometric
- Tkinter (通常包含在Python标准库中)

## 安装步骤

1. 确保已安装Python 3.7或更高版本
2. 安装必要的依赖包：
   ```
   pip install torch torchvision torchaudio
   pip install torch-geometric
   ```

## 使用方法

1. 确保以下文件存在于项目目录中：
   - `tcm_cleaned_data.json` (清洗后的数据文件)
   - `tcm_gnn_model.pth` (训练好的模型权重文件)

2. 运行界面程序：
   ```
   python tcm_model_interface.py
   ```

3. 在界面中输入症状描述，点击"推荐处方"按钮或按回车键获取推荐

## 功能说明

- **症状输入**：在输入框中输入患者的症状描述
- **处方推荐**：系统会根据输入的症状推荐相应的中药处方
- **结果显示**：推荐结果按照相关性得分排序显示

## 注意事项

1. 系统需要预先训练好的模型文件才能正常工作
2. 推荐结果仅供参考，实际用药请咨询专业中医师
3. 如果模型文件或数据文件缺失，系统将无法正常运行

## 文件说明

- `tcm_model_interface.py`：图形用户界面程序
- `tcm_cleaned_data.json`：清洗后的中药数据
- `tcm_gnn_model.pth`：训练好的模型权重文件

## 技术支持

如有任何问题，请联系技术支持团队。