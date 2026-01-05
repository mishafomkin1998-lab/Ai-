@echo off
chcp 65001 >nul
echo.
echo ═══════════════════════════════════════════════════════════
echo                 УСТАНОВКА CHAT SYSTEM
echo ═══════════════════════════════════════════════════════════
echo.

echo [1/2] Устанавливаю зависимости Python...
pip install -r requirements.txt

echo.
echo [2/2] Проверяю Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠️  Ollama не найден!
    echo    Скачайте с https://ollama.com/download
    echo.
) else (
    echo ✅ Ollama установлен
)

echo.
echo ═══════════════════════════════════════════════════════════
echo                 ✅ УСТАНОВКА ЗАВЕРШЕНА
echo ═══════════════════════════════════════════════════════════
echo.
echo Для запуска используйте: start.bat
echo.
pause
