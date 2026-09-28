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
Source: "dist\TCMAI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; 数据文件 (数据库)
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
