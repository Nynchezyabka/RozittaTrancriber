"""
server.py — FastAPI сервис транскрибации.

Запуск (на машине пользователя с установленным faster-whisper):
    pip install -r requirements.txt
    python server.py
    # или: uvicorn server:app --host 0.0.0.0 --port 8010 --reload

Порт зафиксирован: 8010.

Эндпоинты:
    GET  /                  — UI (static/index.html)
    GET  /api/status        — текущее состояние очереди
    POST /api/start         — запуск транскрибации (JSON body = TranscribeOptions)
    POST /api/stop          — мягкая остановка с сохранением готового
    GET  /api/browse?path=  — список подпапок (для диалога выбора папки)
    GET  /api/preview?stem= — содержимое .part/.md файла для превью
    WS   /ws                — прогресс + логи в реальном времени
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from transcribe import manager, TranscribeOptions, MEDIA_EXTS, fmt_ts, pick_device

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    _set_loop()
    yield


app = FastAPI(title="RozittaTranscriber", version="1.0.0", lifespan=lifespan)

# ---------- модели запросов ----------

class StartRequest(BaseModel):
    input_dir: str
    output_dir: Optional[str] = None
    model: str = "large-v3-turbo"
    language: str = "ru"
    prompt: Optional[str] = None
    timecodes: bool = True
    vad_filter: bool = True
    skip_existing: bool = True
    beam_size: int = 5
    # опциональная диаризация через whisperX
    diarize: bool = False
    hf_token: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


# ---------- WebSocket-подписчики (broadcast) ----------

class Hub:
    """Простейший broadcast-хаб для WS-клиентов."""
    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        # сразу шлём текущий статус
        await ws.send_json({"type": "status", "data": manager.status()})

    async def remove(self, ws: WebSocket):
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, msg: dict):
        dead = []
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)


hub = Hub()


# ---------- мост: колбэки менеджера → asyncio broadcast ----------

_loop: Optional[asyncio.AbstractEventLoop] = None


def _set_loop():
    global _loop
    try:
        _loop = asyncio.get_running_loop()
    except RuntimeError:
        _loop = None


def _bridge_progress(task):
    if _loop:
        asyncio.run_coroutine_threadsafe(
            hub.broadcast({"type": "progress", "data": task.to_dict()}), _loop
        )


def _bridge_log(msg: str, level: str):
    if _loop:
        asyncio.run_coroutine_threadsafe(
            hub.broadcast({"type": "log", "data": {"msg": msg, "level": level, "ts": time.time()}}), _loop
        )


def _bridge_state(state: str):
    if _loop:
        asyncio.run_coroutine_threadsafe(
            hub.broadcast({"type": "state", "data": state}), _loop
        )


manager.on_progress = _bridge_progress
manager.on_log = _bridge_log
manager.on_state = _bridge_state


# ---------- маршруты ----------

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def api_health():
    """Состояние сервиса: наличие faster-whisper, whisperx, устройство, версия."""
    fw_version = None
    try:
        import faster_whisper
        fw_version = getattr(faster_whisper, "__version__", "unknown")
    except Exception:
        fw_version = None
    wx_available = False
    try:
        import whisperx  # noqa: F401
        wx_available = True
    except Exception:
        wx_available = False
    dev, ct = pick_device()
    return JSONResponse({
        "ok": True,
        "faster_whisper": fw_version,
        "faster_whisper_available": fw_version is not None,
        "whisperx_available": wx_available,
        "device": dev,
        "compute_type": ct,
        "state": manager.state,
    })


@app.get("/api/status")
async def api_status():
    return JSONResponse(manager.status())


@app.post("/api/start")
async def api_start(req: StartRequest):
    if _loop is None:
        _set_loop()
    opts = TranscribeOptions(
        input_dir=req.input_dir,
        output_dir=req.output_dir,
        model=req.model,
        language=req.language,
        prompt=req.prompt,
        timecodes=req.timecodes,
        vad_filter=req.vad_filter,
        skip_existing=req.skip_existing,
        beam_size=req.beam_size,
        diarize=req.diarize,
        hf_token=req.hf_token,
        min_speakers=req.min_speakers,
        max_speakers=req.max_speakers,
    )
    ok, msg = manager.start(opts)
    return JSONResponse({"ok": ok, "message": msg, "status": manager.status()})


@app.post("/api/stop")
async def api_stop():
    stopped = manager.stop()
    return JSONResponse({"ok": stopped, "status": manager.status()})


@app.get("/api/browse")
async def api_browse(path: str = Query("")):
    """Список подпапок для диалога выбора. Возвращает {path, dirs}."""
    try:
        p = Path(path) if path else Path.home()
        if not p.is_absolute():
            p = Path.home() / path
        if not p.exists():
            p = Path.home()
        dirs = []
        try:
            for entry in sorted(p.iterdir()):
                if entry.is_dir() and not entry.name.startswith("."):
                    dirs.append(str(entry))
        except PermissionError:
            pass
        return JSONResponse({"path": str(p), "dirs": dirs})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/preview")
async def api_preview(stem: str = Query(...)):
    """Содержимое .md (или .part, если в работе) для превью."""
    out_dir = manager._out_dir()
    md = out_dir / f"{stem}.md"
    part = out_dir / f"{stem}.md.part"
    target = md if md.exists() else (part if part.exists() else None)
    if not target:
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        text = target.read_text(encoding="utf-8")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({
        "stem": stem,
        "name": target.name,
        "is_part": target.suffix == ".part",
        "text": text,
        "size": target.stat().st_size,
    })


@app.get("/api/download")
async def api_download(stem: str = Query(...)):
    """Скачать готовый .md."""
    out_dir = manager._out_dir()
    md = out_dir / f"{stem}.md"
    if not md.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(md, filename=md.name, media_type="text/markdown")


class RenameRequest(BaseModel):
    stem: str
    renames: dict  # {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"}


@app.post("/api/rename-speakers")
async def api_rename_speakers(req: RenameRequest):
    """Применяет переименование спикеров к готовому .md (in-place).
    Заменяет 'SPEAKER_00:' на 'Иван:' и т.д. в строках вида '**[MM:SS] SPEAKER_00:**'.
    Работает поверх ОРИГИНАЛЬНОГО текста (до переименования), хранящегося в памяти
    менеджера — поэтому повторное переименование корректно меняет любое имя.
    Атомарная запись через .tmp → rename."""
    import re
    out_dir = manager._out_dir()
    md = out_dir / f"{req.stem}.md"
    part = out_dir / f"{req.stem}.md.part"
    target = md if md.exists() else (part if part.exists() else None)
    if not target:
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        # берём оригинал (до переименования) — из кеша или из самого .md
        original = manager.get_original(req.stem)
        if original is None:
            # оригинал недоступен (уже переименовано без кеша) — берём текущий
            original = target.read_text(encoding="utf-8")
        text = original
        applied = {}
        # паттерн учитывает реальный формат '**[MM:SS] SPEAKER_00:**'
        for old, new in req.renames.items():
            if not old or not new or old == new:
                continue
            pattern = re.compile(r"(\*\*\[[^\]]+\]\s)" + re.escape(old) + r"(:\*\*)")
            new_text, n = pattern.subn(r"\g<1>" + new + r"\g<2>", text)
            if n > 0:
                text = new_text
                applied[old] = {"new": new, "count": n}
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)
        return JSONResponse({
            "ok": True,
            "stem": req.stem,
            "applied": applied,
            "text": text,
            "size": target.stat().st_size,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await hub.add(ws)
    try:
        while True:
            # держим соединение; клиент может слать ping
            await ws.receive_text()
    except WebSocketDisconnect:
        await hub.remove(ws)
    except Exception:
        await hub.remove(ws)


# статика (если нужны доп. ассеты)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")
