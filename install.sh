#!/usr/bin/env bash
# ============================================================
#  RozittaTranscriber — install.sh
# ============================================================
#
#  Установщик зависимостей с автодетектом видеокарты.
#  Работает на Linux и macOS.
#
#  Логика:
#    - NVIDIA найдена (nvidia-smi работает)  → GPU-сборка (~4 ГБ)
#    - NVIDIA не найдена / AMD / Intel       → CPU-сборка (~300 МБ)
#
#  GPU-сборка: pip install -r requirements.txt
#    faster-whisper тянет ctranslate2 + nvidia-* (~1.93 ГБ)
#    whisperx тянет torch CUDA-версии (~2 ГБ)
#    всё нужно и используется.
#
#  CPU-сборка (3 шага, чтобы не качать CUDA-torch впустую):
#    1) ставим CPU-torch ПЕРВЫМ (~200 МБ) с pytorch.org/whl/cpu
#    2) pip install -r requirements.txt — whisperx видит, что torch
#       уже стоит совместимой версии, и НЕ заменяет его на CUDA-версию
#       (экономия ~2 ГБ скачивания). Но ctranslate2 всё равно тянет
#       nvidia-* (~1.93 ГБ) — этого избежать нельзя.
#    3) pip uninstall nvidia-* — удаляем CUDA-библиотеки (~1.93 ГБ),
#       ctranslate2 на CPU работает без них.
#    Итог: скачано ~2.13 ГБ, в .venv осталось ~300 МБ.
#
#  Ручное переопределение (--gpu / --cpu) нужно в случаях:
#    1. Гибридный ноутбук (Intel + NVIDIA): автодетект найдёт NVIDIA,
#       но вы хотите экономить батарею и принудительно ставить CPU-сборку.
#    2. NVIDIA-карта БЕЗ драйвера: nvidia-smi не запустится, скрипт
#       выберет CPU. Если планируете поставить драйвер позже —
#       --gpu поставит GPU-сборку заранее.
#    3. Сервер/CI без видеодрайвера: иногда nvidia-smi не в PATH или
#       нет прав — --cpu явно ставит лёгкую сборку.
#    4. Разработка/тесты: --cpu, чтобы сравнить скорость CPU vs GPU
#       на одной машине без переустановки.
#
#  Использование:
#    ./install.sh            автодетект (рекомендуется)
#    ./install.sh --gpu      принудительно GPU-сборка
#    ./install.sh --cpu      принудительно CPU-сборка
# ============================================================

set -u  # выход при обращении к неопределённой переменной
# (set -e НЕ включаем — nvidia-smi может вернуть ненулевой код при детекте)

cd "$(dirname "$0")"

# --- Цвета (если терминал поддерживает) ---
if [ -t 1 ]; then
    C_GREEN='\033[0;32m'
    C_YELLOW='\033[1;33m'
    C_RED='\033[0;31m'
    C_BLUE='\033[0;34m'
    C_RESET='\033[0m'
else
    C_GREEN=''; C_YELLOW=''; C_RED=''; C_BLUE=''; C_RESET=''
fi

info()  { printf "${C_BLUE}[info]${C_RESET} %s\n" "$*"; }
ok()    { printf "${C_GREEN}[ok]${C_RESET} %s\n" "$*"; }
warn()  { printf "${C_YELLOW}[warn]${C_RESET} %s\n" "$*"; }
err()   { printf "${C_RED}[ERROR]${C_RESET} %s\n" "$*" >&2; }
step()  { printf "\n${C_BLUE}[step]${C_RESET} %s\n" "$*"; }

echo "============================================"
echo "  RozittaTranscriber — установщик зависимостей"
echo "============================================"
echo

# --- Проверка Python ---
if ! command -v python3 >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PY=python
    else
        err "Python не найден. Установите Python 3.10-3.12."
        err "  macOS:  brew install python@3.12"
        err "  Ubuntu: sudo apt install python3.12 python3.12-venv"
        exit 1
    fi
else
    PY=python3
fi

PYVER="$($PY --version 2>&1 | awk '{print $2}')"
ok "Python $PYVER"

# --- Проверка версии Python (нужно 3.10-3.12) ---
PYMAJOR="${PYVER%%.*}"
REST="${PYVER#*.}"
PYMINOR="${REST%%.*}"
if [ "$PYMAJOR" != "3" ] || [ "$PYMINOR" -lt 10 ] 2>/dev/null; then
    err "Нужен Python 3.10-3.12, у вас $PYVER. whisperX не поддерживает ниже 3.10."
    exit 1
fi
if [ "$PYMINOR" -ge 13 ] 2>/dev/null; then
    warn "У вас Python $PYVER, но whisperX официально поддерживает 3.10-3.12."
    warn "Возможны проблемы. Рекомендуется Python 3.12."
    read -r -p "Продолжить anyway? [y/N] " ans
    case "$ans" in
        y|Y|yes|д|Д) ;;
        *) exit 1 ;;
    esac
fi

# --- Проверка requirements.txt ---
if [ ! -f "requirements.txt" ]; then
    err "requirements.txt не найден в текущей папке."
    err "Запустите install.sh из корня проекта RozittaTranscriber."
    exit 1
fi

# --- Определение режима ---
MODE="auto"
case "${1:-}" in
    --gpu) MODE="gpu" ;;
    --cpu) MODE="cpu" ;;
    "")    MODE="auto" ;;
    *)     err "Неизвестный аргумент: $1"
           err "Использование: ./install.sh [--gpu|--cpu]"
           exit 1 ;;
esac

if [ "$MODE" = "auto" ]; then
    info "Проверяю NVIDIA через nvidia-smi..."
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        ok "NVIDIA обнаружена — GPU-сборка"
        MODE="gpu"
    else
        info "NVIDIA не обнаружена — CPU-сборка"
        MODE="cpu"
    fi
else
    info "Принудительный режим: $MODE"
fi

echo
echo "============================================"
echo "  Установка: $MODE-сборка"
echo "============================================"

# --- Обновление pip ---
step "Обновление pip..."
$PY -m pip install --upgrade pip wheel setuptools

# --- GPU-сборка ---
if [ "$MODE" = "gpu" ]; then
    step "Установка зависимостей (GPU)..."
    pip install -r requirements.txt || {
        err "Не удалось установить зависимости."
        exit 1
    }
# --- CPU-сборка ---
else
    step "Установка CPU-torch ПЕРВЫМ (~200 МБ, чтобы не качать CUDA-torch впустую)..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu || {
        err "Не удалось установить CPU-torch. Проверьте интернет-соединение."
        exit 1
    }

    step "Установка зависимостей из requirements.txt..."
    pip install -r requirements.txt || {
        err "Не удалось установить зависимости."
        exit 1
    }

    step "Удаление CUDA-библиотек nvidia-* (~1.93 ГБ, не нужны на CPU)..."
    # ctranslate2 тянет их как deps, но на CPU работает без них
    pip uninstall -y \
        nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12 nvidia-cufft-cu12 \
        nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
        nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 nvidia-cufile-cu12 \
        2>/dev/null || true
    ok "CUDA-библиотеки удалены (если были установлены)."
fi

echo
echo "============================================"
ok "Установка завершена!"
echo "============================================"
echo
echo "Что дальше:"
echo "  1. Запустите сервер:  ./start.sh   (или: python3 server.py)"
echo "  2. Откройте браузер:  http://localhost:8010"
echo
echo "При первом запуске скачается модель faster-whisper"
echo "(large-v3-turbo ~1.5 ГБ — по умолчанию)."
echo
echo "Для диаризации (опционально) понадобится HuggingFace-токен —"
echo "см. README, раздел «Разделение по ролям»."
