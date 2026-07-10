@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title RozittaTranscriber — установщик зависимостей

REM ============================================================
REM  RozittaTranscriber — install.bat
REM ============================================================
REM
REM  Установщик зависимостей с автодетектом видеокарты.
REM
REM  Логика:
REM    - NVIDIA найдена (nvidia-smi работает)  → GPU-сборка (~4 ГБ)
REM    - NVIDIA не найдена / AMD / Intel       → CPU-сборка (~300 МБ)
REM
REM  GPU-сборка: pip install -r requirements.txt
REM    faster-whisper тянет ctranslate2 + nvidia-* (~1.93 ГБ)
REM    whisperx тянет torch CUDA-версии (~2 ГБ)
REM    всё нужно и используется.
REM
REM  CPU-сборка (3 шага, чтобы не качать CUDA-torch впустую):
REM    1) ставим CPU-torch ПЕРВЫМ (~200 МБ) с pytorch.org/whl/cpu
REM    2) pip install -r requirements.txt — whisperx видит, что torch
REM       уже стоит совместимой версии, и НЕ заменяет его на CUDA-версию
REM       (экономия ~2 ГБ скачивания). Но ctranslate2 всё равно тянет
REM       nvidia-* (~1.93 ГБ) — этого избежать нельзя.
REM    3) pip uninstall nvidia-* — удаляем CUDA-библиотеки (~1.93 ГБ),
REM       ctranslate2 на CPU работает без них.
REM    Итог: скачано ~2.13 ГБ, в .venv осталось ~300 МБ.
REM
REM  Ручное переопределение (--gpu / --cpu) нужно в случаях:
REM    1. Гибридный ноутбук (Intel + NVIDIA): автодетект найдёт NVIDIA,
REM       но вы хотите экономить батарею и принудительно ставить CPU-сборку.
REM    2. NVIDIA-карта БЕЗ драйвера: nvidia-smi не запустится, скрипт
REM       выберет CPU. Если планируете поставить драйвер позже —
REM       --gpu поставит GPU-сборку заранее.
REM    3. Сервер/CI без видеодрайвера: иногда nvidia-smi не в PATH или
REM       нет прав — --cpu явно ставит лёгкую сборку.
REM    4. Разработка/тесты: --cpu, чтобы сравнить скорость CPU vs GPU
REM       на одной машине без переустановки.
REM
REM  Использование:
REM    install.bat            автодетект (рекомендуется)
REM    install.bat --gpu      принудительно GPU-сборка
REM    install.bat --cpu      принудительно CPU-сборка
REM ============================================================

cd /d "%~dp0"

echo ============================================
echo   RozittaTranscriber — установщик зависимостей
echo ============================================
echo.

REM --- Проверка Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден в PATH.
    echo Установите Python 3.10-3.12 с https://www.python.org/downloads/
    echo (whisperX не поддерживает Python 3.13+)
    echo.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [ok] Python %PYVER%

REM --- Проверка версии Python (нужно 3.10-3.12) ---
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
if !PYMAJOR! neq 3 goto :pyver_ok
if !PYMINOR! lss 10 (
    echo [ERROR] Нужен Python 3.10-3.12, у вас %PYVER%.
    echo whisperX не поддерживает Python ниже 3.10.
    pause
    exit /b 1
)
if !PYMINOR! geq 13 (
    echo [WARN] У вас Python %PYVER%, но whisperX официально поддерживает 3.10-3.12.
    echo Возможны проблемы. Рекомендуется установить Python 3.12.
    echo.
    choice /c YN /m "Продолжить anyway"
    if errorlevel 2 exit /b 1
)
:pyver_ok

REM --- Проверка requirements.txt ---
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt не найден в текущей папке.
    echo Запустите install.bat из корня проекта RozittaTranscriber.
    pause
    exit /b 1
)

REM --- Определение режима ---
set MODE=auto
if /i "%~1"=="--gpu" set MODE=gpu
if /i "%~1"=="--cpu" set MODE=cpu

if "!MODE!"=="auto" (
    echo [detect] Проверяю NVIDIA через nvidia-smi...
    nvidia-smi >nul 2>&1
    if !errorlevel! equ 0 (
        echo [detect] NVIDIA обнаружена — GPU-сборка
        set MODE=gpu
    ) else (
        echo [detect] NVIDIA не обнаружена — CPU-сборка
        set MODE=cpu
    )
) else (
    echo [mode] Принудительный режим: !MODE!
)

echo.
echo ============================================
echo   Установка: !MODE!-сборка
echo ============================================
echo.

REM --- Обновление pip ---
echo [step 0] Обновление pip...
python -m pip install --upgrade pip wheel setuptools

if /i "!MODE!"=="gpu" goto :install_gpu
goto :install_cpu

:install_gpu
echo.
echo [step 1] Установка зависимостей (GPU)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Не удалось установить зависимости.
    pause
    exit /b 1
)
goto :install_done

:install_cpu
echo.
echo [step 1] Установка CPU-torch ПЕРВЫМ (~200 МБ, чтобы не качать CUDA-torch впустую)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo.
    echo [ERROR] Не удалось установить CPU-torch.
    echo Проверьте интернет-соединение.
    pause
    exit /b 1
)

echo.
echo [step 2] Установка зависимостей из requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Не удалось установить зависимости.
    pause
    exit /b 1
)

echo.
echo [step 3] Удаление CUDA-библиотек nvidia-* (~1.93 ГБ, не нужны на CPU)...
REM ctranslate2 тянет их как deps, но на CPU работает без них
pip uninstall -y nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 nvidia-cufile-cu12 2>nul
echo [ok] CUDA-библиотеки удалены (если были установлены).

:install_done
echo.
echo ============================================
echo   Установка завершена!
echo ============================================
echo.
echo Что дальше:
echo   1. Запустите сервер:  start.bat   (или: python server.py)
echo   2. Откройте браузер:  http://localhost:8010
echo.
echo При первом запуске скачается модель faster-whisper
echo (large-v3-turbo ~1.5 ГБ — по умолчанию).
echo.
echo Для диаризации (опционально) понадобится HuggingFace-токен —
echo см. README, раздел «Разделение по ролям».
echo.
pause
endlocal
