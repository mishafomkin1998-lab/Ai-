@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo ═══════════════════════════════════════════════════════════
echo                 💬 CHAT SYSTEM
echo ═══════════════════════════════════════════════════════════
echo.

REM Проверяем Ollama
ollama --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama не установлен!
    echo    Скачайте с https://ollama.com/download
    pause
    exit
)

REM Проверяем модель
ollama list | findstr "mymodel" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Модель 'mymodel' не найдена в Ollama
    echo    Создайте её командой: ollama create mymodel -f Modelfile
    echo.
)

echo Запускаю сервер...
echo Откройте в браузере: http://127.0.0.1:8000
echo.
echo Для остановки закройте это окно или нажмите Ctrl+C
echo.

python server.py

pause
