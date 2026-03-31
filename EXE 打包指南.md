# 中医药智能问答系统 - EXE 打包指南

**日期**: 2026 年 3 月 23 日

---

## 📦 快速打包（推荐）

### 方法 1：双击打包

1. 双击运行 **`打包简单.bat`**
2. 等待 5-10 分钟
3. 完成后在 `dist` 文件夹找到 `中医药智能问答系统.exe`

### 方法 2：命令行打包

```bash
cd E:\Tom\TCMAI

# 一键打包
python -m PyInstaller --name="中医药智能问答系统" --windowed --onefile ^
  --add-data "src\web\templates;src\web\templates" ^
  --add-data "data;data" ^
  --hidden-import flask ^
  --hidden-import sqlite3 ^
  launcher.py
```

---

## 📁 打包后文件

### 单文件模式（--onefile）

```
dist/
└── 中医药智能问答系统.exe    (约 30-50 MB)
```

**优点**: 只有一个 exe 文件，方便分发  
**缺点**: 首次启动较慢（需要解压）

### 文件夹模式（默认）

```
dist/
└── 中医药智能问答系统/
    ├── 中医药智能问答系统.exe
    ├── data/
    ├── docs/
    └── (其他依赖)
```

**优点**: 启动快  
**缺点**: 文件较多

---

## 🚀 使用方法

### 单文件版

1. 复制 `中医药智能问答系统.exe` 到任意位置
2. 双击运行
3. 自动打开浏览器访问 http://localhost:5000

### 文件夹版

1. 复制整个 `中医药智能问答系统` 文件夹
2. 双击 `中医药智能问答系统.exe`
3. 自动打开浏览器

---

## ⚙️ 打包配置说明

### launcher.py

这是打包的入口文件，会：
1. 启动 Flask 服务器
2. 自动打开浏览器
3. 显示友好界面

### 关键参数

```bash
--name="中医药智能问答系统"   # 程序名称
--windowed                     # 不显示控制台（窗口应用）
--onefile                      # 打包成单个 exe
--add-data "源路径;目标路径"   # 添加数据文件
--hidden-import 模块名         # 添加隐藏依赖
```

---

## 📊 打包信息

| 项目 | 说明 |
|------|------|
| Python | 3.9.25 |
| PyInstaller | 6.19.0 |
| 打包后大小 | 30-50 MB |
| 打包时间 | 5-10 分钟 |
| 支持系统 | Windows 7/10/11 |

---

## ⚠️ 常见问题

### 1. 打包失败

**原因**: 依赖缺失或路径错误

**解决**:
```bash
# 清理旧文件
rmdir /s /q build dist

# 重新打包
python -m PyInstaller --clean ...
```

### 2. 运行时报错"找不到模块"

**原因**: 隐藏导入未包含

**解决**: 添加 `--hidden-import 模块名`

### 3. 数据库文件找不到

**原因**: 数据文件路径错误

**解决**: 检查 `--add-data` 参数

---

## 📝 完整打包命令

```bash
python -m PyInstaller ^
  --name="中医药智能问答系统" ^
  --windowed ^
  --onefile ^
  --add-data "src\web\templates;src\web\templates" ^
  --add-data "data;data" ^
  --add-data "docs;docs" ^
  --hidden-import flask ^
  --hidden-import sqlite3 ^
  --hidden-import core.knowledge_base ^
  --hidden-import core.symptom_analysis ^
  --hidden-import core.fallback ^
  launcher.py
```

---

## 🧪 测试打包

打包完成后：

1. **复制到其他目录** - 确保不依赖原路径
2. **在其他电脑运行** - 确保无需 Python
3. **测试功能** - 确保数据库查询正常

---

## 💡 优化建议

### 减小文件大小

```bash
# 使用 UPX 压缩（需单独安装）
--upx-dir=C:\upx

# 排除不需要的模块
--exclude-module matplotlib
--exclude-module numpy
```

### 添加图标

```bash
# 准备 icon.ico 文件
--icon=icon.ico
```

### 版本信息

创建 `version.txt`：

```
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    fileOS=0x40004,
    fileType=0x1,
  ),
  kids=[...],
)
```

使用：
```bash
--version-file=version.txt
```

---

## 📞 分发说明

### 分发给他人

1. 打包完成后，复制 exe 文件
2. 压缩为 ZIP
3. 发送给用户
4. 用户双击即可运行

### 无需安装

- ✅ 不需要 Python
- ✅ 不需要依赖
- ✅ 双击即运行
- ✅ 可放 U 盘携带

---

**最后更新**: 2026 年 3 月 23 日
