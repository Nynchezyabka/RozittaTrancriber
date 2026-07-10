#!/usr/bin/env bash
# ============================================================
#  RozittaTranscriber — start.sh
# ============================================================
#
#  Запуск сервера (после install.sh).
#  Linux / macOS версия start.bat.
#
#  Использование:
#    ./start.sh
#    Остановить: Ctrl+C
# ============================================================

cd "$(dirname "$0")"

echo "============================================"
echo "  RozittaTranscriber — запуск сервера"
echo "============================================"
echo
echo "Сервер: http://localhost:8010"
echo "Остановить: Ctrl+C"
echo

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python

"$PY" server.py
RC=$?

if [ $RC -ne 0 ]; then
    echo
    echo "[ERROR] Не удалось запустить сервер (код $RC)."
    echo "Проверьте, что установлены зависимости:"
    echo "  ./install.sh"
    echo
    exit $RC
fi
