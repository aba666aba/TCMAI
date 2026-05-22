@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo        TCMAI Desktop Build
echo ========================================
echo.

python -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] PyInstaller module was not found for this Python.
    echo Install it first with:
    echo     python -m pip install pyinstaller
    echo.
    pause
    exit /b 1
)

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

python -m PyInstaller --noconfirm tcm_desktop.spec
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo [OK] Build completed.
echo Output folder:
echo     dist\TCMAI_Desktop\
echo.
pause
endlocal
