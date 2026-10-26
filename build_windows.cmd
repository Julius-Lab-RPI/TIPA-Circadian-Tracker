@echo off
setlocal

if "%~1"=="" (
    echo Usage: build_windows.cmd "C:\path\to\python.exe"
    exit /b 2
)

set "PYTHON_EXE=%~1"
if not exist "%PYTHON_EXE%" (
    echo Python executable not found: %PYTHON_EXE%
    exit /b 2
)
for %%I in ("%PYTHON_EXE%") do set "PYTHON_DIR=%%~dpI"
set "PATH=%PYTHON_DIR%Library\bin;%PATH%"

pushd "%~dp0"
"%PYTHON_EXE%" -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo PyInstaller is not installed. Run:
    echo "%PYTHON_EXE%" -m pip install -r requirements-build.txt
    popd
    exit /b 1
)

if exist "dist\TpbmAnalysis" (
    powershell.exe -NoProfile -Command "Remove-Item -LiteralPath 'dist\TpbmAnalysis' -Recurse -Force"
    if errorlevel 1 (
        popd
        exit /b 1
    )
)
if exist "dist\TpbmAnalysis-Windows.zip" (
    del /q "dist\TpbmAnalysis-Windows.zip"
)

"%PYTHON_EXE%" -m PyInstaller --noconfirm --clean --windowed --onedir --name TpbmAnalysis --collect-all openpyxl run_tpbm_gui.py
if errorlevel 1 (
    popd
    exit /b 1
)

if not exist "dist\TpbmAnalysis\TpbmAnalysis.exe" (
    echo Expected executable was not created.
    popd
    exit /b 1
)

powershell.exe -NoProfile -Command "try { Compress-Archive -Path 'dist\TpbmAnalysis' -DestinationPath 'dist\TpbmAnalysis-Windows.zip' -Force -ErrorAction Stop } catch { Write-Error $_; exit 1 }"
if errorlevel 1 goto :archive_failed
if not exist "dist\TpbmAnalysis-Windows.zip" goto :archive_failed

echo Application: %CD%\dist\TpbmAnalysis\TpbmAnalysis.exe
echo Shareable package: %CD%\dist\TpbmAnalysis-Windows.zip
popd
exit /b 0

:archive_failed
echo Could not create the shareable ZIP package.
popd
exit /b 1
