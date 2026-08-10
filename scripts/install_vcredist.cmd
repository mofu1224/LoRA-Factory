@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_vcredist.ps1"
if errorlevel 1 (
    echo.
    echo VC++ Redistributable installation did not complete. Review the message above.
    pause
    exit /b 1
)
pause
