[Setup]
AppName=中医药智能问答系统
AppVersion=1.0
AppPublisher=TCMAI
DefaultDirName={commonpf}\TCMAI
DefaultGroupName=中医药智能问答系统
OutputBaseFilename=中医药智能问答系统_Setup
Compression=lzma2
SolidCompression=yes
OutputDir=installer_output
UsePreviousAppDir=no
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
; PyInstaller 打包后的文件 (代码和依赖)
; 注意：{app}（Program Files）对普通用户只读。运行时可写数据
; （历史记录/日志/知识库副本）由程序写入 %LOCALAPPDATA%\TCMAI\data，
; 首次启动时自动从下方 {app}\data 的种子库复制初始 tcm_knowledge.db。
; 卸载只清理 {app}，%LOCALAPPDATA%\TCMAI 下的用户数据会被保留。
Source: "dist\TCMAI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; 数据文件（种子数据库：只读模板，运行时不会直接写这个文件）
Source: "data\tcm_knowledge.db"; DestDir: "{app}\data"; Flags: ignoreversion

; 模型文件 (NLU 模型)
Source: "models\nlu\qwen\*"; DestDir: "{app}\models\nlu\qwen"; Flags: ignoreversion recursesubdirs createallsubdirs

; 推荐模型
Source: "tcm_recommender\best_model.pth"; DestDir: "{app}\tcm_recommender"; Flags: ignoreversion

[Icons]
Name: "{group}\中医药智能问答系统"; Filename: "{app}\TCMAI.exe"
Name: "{group}\卸载 中医药智能问答系统"; Filename: "{uninstallexe}"
Name: "{commondesktop}\中医药智能问答系统"; Filename: "{app}\TCMAI.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加图标:"

[Run]
Filename: "{app}\TCMAI.exe"; Description: "启动 中医药智能问答系统"; Flags: nowait postinstall skipifsilent
