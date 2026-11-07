@echo off
rem Launch FlowDesk on Windows (creates the virtual environment on first run).
rem Usage: flowdesk.bat [workspace.fdw | file.fcs | folder ...]
rem Set FLOWDESK_CONSOLE=1 to run with a console window (shows error messages).
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run

where uv >nul 2>nul
if errorlevel 1 goto nouv
echo Creating the virtual environment with uv ...
uv venv --python 3.12 .venv || goto fail
uv pip install --python .venv\Scripts\python.exe -e . || goto fail
goto run

:nouv
echo Creating the virtual environment ...
py -3.12 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" goto nopython
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto fail
".venv\Scripts\python.exe" -m pip install -e . || goto fail

:run
if "%FLOWDESK_CONSOLE%"=="1" goto console
if not exist ".venv\Scripts\pythonw.exe" goto console
start "" ".venv\Scripts\pythonw.exe" -m flowdesk %*
goto end

:console
".venv\Scripts\python.exe" -m flowdesk %*
goto end

:nopython
echo Python 3.12 was not found. Install it from https://www.python.org/downloads/windows/
echo (tick "Add python.exe to PATH") or install uv, then run flowdesk.bat again.
pause
exit /b 1

:fail
echo Installing FlowDesk failed - see the messages above. Delete the .venv folder and retry.
pause
exit /b 1

:end
endlocal
