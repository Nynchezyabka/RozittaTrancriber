# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec для сборки transcriber.exe.

Сборка (Windows):
    pip install pyinstaller
    pyinstaller transcriber.spec --noconfirm

Результат: dist/transcriber/transcriber.exe (+ папка с зависимостями).
Чтобы получить один .exe — заменить COLLECT на EXE с bundle=True (см. комментарий внизу),
но one-folder стартует быстрее и стабильнее с CTranslate2.
"""

import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# --- собираем всё содержимое пакетов, которые тащат данные/бинарники ---
datas = []
binaries = []
hiddenimports = []

for pkg in ['faster_whisper', 'ctranslate2', 'av', 'tokenizers', 'huggingface_hub',
            'uvicorn', 'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto',
            'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto',
            'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto',
            'uvicorn.lifespan', 'uvicorn.lifespan.on',
            'anyio', 'anyio._backends', 'anyio._backends._asyncio',
            'httptools', 'websockets', 'watchfiles',
            # опциональная диаризация (whisperX) — тащим на случай, если установлен
            'whisperx', 'pyannote', 'pyannote.audio', 'torchaudio', 'torchaudio.backend',
            'soundfile', 'librosa', 'scipy', 'sklearn', 'sklearn.cluster']:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# --- статика UI ---
datas += [('static', 'static')]

a = Analysis(
    ['server.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'matplotlib', 'PIL', 'IPython', 'pytest'],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --- one-folder: папка с exe + зависимостями (рекомендуется) ---
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='transcriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,           # консоль оставляем — видно логи сервера
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='docs/transcriber.ico',   # лягушка-логотип Rozitta
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='transcriber',
)

# --- если хотите один .exe (one-file), закомментируйте COLLECT выше и раскомментируйте блок ниже ---
# exe = EXE(
#     pyz, a.scripts, a.binaries, a.datas, [],
#     name='transcriber',
#     console=True,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     runtime_tmpdir=None,
#     console=True,
# )
