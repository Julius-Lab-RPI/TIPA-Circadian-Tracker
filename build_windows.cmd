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
set "PIL_DLL_DIR=%PYTHON_DIR%Library\bin"

for %%D in (tiff.dll jpeg8.dll openjp2.dll zlib-ng2.dll deflate.dll Lerc.dll zstd.dll) do (
    if not exist "%PIL_DLL_DIR%\%%D" (
        echo Required Pillow runtime library not found: %PIL_DLL_DIR%\%%D
        echo Build with an environment that contains Pillow's native libraries.
        exit /b 1
    )
)

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

"%PYTHON_EXE%" -m PyInstaller --noconfirm --clean --windowed --onedir --name TpbmAnalysis --collect-all openpyxl --add-binary "%PIL_DLL_DIR%\tiff.dll;." --add-binary "%PIL_DLL_DIR%\jpeg8.dll;." --add-binary "%PIL_DLL_DIR%\openjp2.dll;." --add-binary "%PIL_DLL_DIR%\zlib-ng2.dll;." --add-binary "%PIL_DLL_DIR%\deflate.dll;." --add-binary "%PIL_DLL_DIR%\Lerc.dll;." --add-binary "%PIL_DLL_DIR%\zstd.dll;." run_tpbm_gui.py
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
