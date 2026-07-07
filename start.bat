@echo off
chcp 65001 >nul
title RozittaTranscriber (faster-whisper)

echo ============================================
echo   RozittaTranscriber — запуск без сборки exe
echo ============================================
echo.

cd /d "%~dp0"

echo Запуск сервера RozittaTranscriber на http://localhost:8010
echo (закройте это окно, чтобы остановить)
echo.

python server.py

if errorlevel 1 (
    echo.
    echo ОШИБКА: не удалось запустить.
    echo Проверьте, что установлены зависимости:
    echo   pip install -r requirements.txt
    echo.
    pause
)
