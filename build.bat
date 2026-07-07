@echo off
chcp 65001 >nul
title Сборка RozittaTranscriber.exe

echo ============================================
echo   Сборка RozittaTranscriber.exe через PyInstaller
echo ============================================
echo.

echo [1/4] Проверка Python...
python --version || (echo ОШИБКА: Python не найден в PATH & pause & exit /b 1)
echo.

echo [2/4] Установка зависимостей...
pip install -r requirements.txt || (echo ОШИБКА установки зависимостей & pause & exit /b 1)
pip install pyinstaller || (echo ОШИБКА установки PyInstaller & pause & exit /b 1)
echo.

echo [3/4] Сборка exe (это займёт 2-5 минут)...
pyinstaller transcriber.spec --noconfirm || (echo ОШИБКА сборки & pause & exit /b 1)
echo.

echo [4/4] Готово!
echo.
echo ============================================
echo   Экзешник создан:
echo   dist\transcriber\transcriber.exe (бренд: RozittaTranscriber)
echo ============================================
echo.
echo Чтобы запустить: двойной клик по transcriber.exe
echo Затем открыть в браузере: http://localhost:8010
echo.
pause
