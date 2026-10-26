@echo off
REM Build AudioTranscribe.exe on Windows. Run this from the project folder.
REM PyInstaller cannot cross-compile, so this has to run on Windows itself.
setlocal

if not exist .venv (
    echo Creating the virtual environment...
    python -m venv .venv || goto :fail
)

echo Installing dependencies...
.venv\Scripts\python -m pip install --upgrade pip >nul
.venv\Scripts\pip install -r requirements-windows.txt || goto :fail
.venv\Scripts\pip install pyinstaller==6.11.1 || goto :fail

echo Checking the pipeline imports before building...
.venv\Scripts\python -c "import sys; sys.path.insert(0,'src'); import asr_backend, pipeline, webapp; print('backend:', asr_backend.backend())" || goto :fail

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building...
.venv\Scripts\pyinstaller --noconfirm --clean --log-level WARN AudioTranscribe-windows.spec || goto :fail

copy packaging\FIRST-RUN-WINDOWS.txt dist\AudioTranscribe\ >nul 2>&1

echo.
echo Built: dist\AudioTranscribe\AudioTranscribe.exe
echo Zip the whole dist\AudioTranscribe folder to share it.
goto :eof

:fail
echo.
echo BUILD FAILED - see the error above.
exit /b 1
