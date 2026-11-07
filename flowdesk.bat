@echo off
rem Launch FlowDesk on Windows (creates the virtual environment on first run).
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    where uv >nul 2>nul
    if %errorlevel%==0 (
        uv venv --python 3.12 .venv && uv pip install --python .venv\Scripts\python.exe -e .
    ) else (
        py -3.12 -m venv .venv 2>nul || python -m venv .venv
        .venv\Scripts\python.exe -m pip install --upgrade pip
        .venv\Scripts\python.exe -m pip install -e .
    )
)
start "" ".venv\Scripts\pythonw.exe" -m flowdesk %*
endlocal
