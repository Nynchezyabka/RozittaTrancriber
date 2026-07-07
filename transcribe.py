"""
transcribe.py — ядро транскрибации (перенос transcribe_videos.py).

Сохраняет все функции оригинала:
  - faster-whisper (один говорящий, без диаризации)
  - автоопределение CUDA/CPU
  - на файл → один .md с заголовком "# <имя>"
  - таймкоды **[MM:SS]** каждые 5 минут (TIMECODE_STEP)
  - idempotent: пропуск файлов, где .md уже есть
  - crash-safe: запись в .part → атомарный rename
  - прогресс по сегментам (% + затраченное время)
  - устойчивость: сбой одного файла не роняет батч
  - graceful stop: флаг self.stop_requested

Дополнения для веба:
  - TranscribeManager: одноэлементный менеджер очереди (потокобезопасный)
  - колбэки on_progress / on_log для WebSocket
  - статус-машина: idle / running / stopping / finished

Опциональная диаризация (whisperX):
  - если opts.diarize=True и установлен whisperx + hf_token →
    распознанный текст разбивается по спикерам:
    **[MM:SS] Спикер N:** текст
  - если whisperx не установлен или нет токена — мягкое падение:
    транскрипция идёт без диаризации, в логе предупреждение
"""

from __future__ import annotations

import os
import sys
import time
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

MEDIA_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3",
              ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac"}

TIMECODE_STEP = 300  # секунд между метками (5 минут)


def fmt_ts(seconds: float) -> str:
    """1234.5 -> '20:34' (или '1:05:12' если больше часа)."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def pick_device() -> tuple[str, str]:
    """Выбор устройства: CUDA, если доступна, иначе CPU."""
    try:
        from ctranslate2 import get_cuda_device_count
        if get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


# ---------- состояние одного файла ----------

@dataclass
class FileTask:
    path: Path
    size: int = 0
    duration: float = 0.0
    status: str = "queued"        # queued | running | done | skipped | error | stopped
    progress: float = 0.0         # 0..100
    elapsed: float = 0.0          # секунд затрачено
    eta: float = 0.0              # секунд осталось
    error: str = ""
    current_seg: float = 0.0      # позиция текущего сегмента (сек)
    md_path: Optional[Path] = None
    word_count: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def name(self) -> str:
        return self.path.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stem": self.path.stem,
            "size": self.size,
            "duration": self.duration,
            "status": self.status,
            "progress": round(self.progress, 1),
            "elapsed": round(self.elapsed, 1),
            "eta": round(self.eta, 1),
            "error": self.error,
            "current_seg": round(self.current_seg, 1),
            "md_name": self.md_path.name if self.md_path else None,
            "word_count": self.word_count,
        }


@dataclass
class TranscribeOptions:
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


# ---------- менеджер очереди (потокобезопасный, singleton) ----------

class TranscribeManager:
    """Один активный запуск транскрибации. Поток в фоне, прогресс через колбэки."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self.options: Optional[TranscribeOptions] = None
        self.tasks: list[FileTask] = []
        self.state: str = "idle"        # idle | running | stopping | finished
        self.device: str = ""
        self.compute_type: str = ""
        self.model_name: str = ""
        self.started_at: float = 0.0
        self.finished_at: float = 0.0
        # оригинальные тексты .md (до переименования спикеров): stem -> text
        # хранятся в памяти, чтобы повторное переименование работало корректно
        self._originals: dict[str, str] = {}
        # колбэки: on_progress(task), on_log(msg, level), on_state(state)
        self.on_progress: Optional[Callable[[FileTask], None]] = None
        self.on_log: Optional[Callable[[str, str], None]] = None
        self.on_state: Optional[Callable[[str], None]] = None

    # --- публичное API ---

    @property
    def is_running(self) -> bool:
        return self.state == "running"

    def status(self) -> dict:
        with self._lock:
            total = len(self.tasks)
            done = sum(1 for t in self.tasks if t.status == "done")
            running = sum(1 for t in self.tasks if t.status == "running")
            queued = sum(1 for t in self.tasks if t.status == "queued")
            errors = sum(1 for t in self.tasks if t.status == "error")
            skipped = sum(1 for t in self.tasks if t.status == "skipped")
            current = next((t for t in self.tasks if t.status == "running"), None)
            overall = 0.0
            if total:
                overall = sum(t.progress for t in self.tasks) / total
            return {
                "state": self.state,
                "device": self.device,
                "compute_type": self.compute_type,
                "model": self.model_name,
                "input_dir": self.options.input_dir if self.options else "",
                "output_dir": str(self._out_dir()) if self.options else "",
                "total": total,
                "done": done,
                "running": running,
                "queued": queued,
                "errors": errors,
                "skipped": skipped,
                "overall_progress": round(overall, 1),
                "current": current.to_dict() if current else None,
                "tasks": [t.to_dict() for t in self.tasks],
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    def start(self, opts: TranscribeOptions) -> tuple[bool, str]:
        with self._lock:
            if self.state == "running":
                return False, "Транскрибация уже идёт"
            src = Path(opts.input_dir)
            if not src.is_dir():
                return False, f"Папка не найдена: {src}"
            out = self._out_dir_for(opts)
            out.mkdir(parents=True, exist_ok=True)

            files = sorted(p for p in src.iterdir()
                           if p.is_file() and p.suffix.lower() in MEDIA_EXTS)
            if not files:
                return False, f"В папке нет медиафайлов: {src}"

            self.options = opts
            self.tasks = [
                FileTask(
                    path=p,
                    size=p.stat().st_size if p.exists() else 0,
                    md_path=out / (p.stem + ".md"),
                ) for p in files
            ]
            # помечаем уже готовые как skipped
            if opts.skip_existing:
                for t in self.tasks:
                    if t.md_path and t.md_path.exists():
                        t.status = "skipped"

            self._stop_flag.clear()
            self.state = "running"
            self.started_at = time.time()
            self.finished_at = 0.0
            self.device = ""
            self.compute_type = ""
            self.model_name = opts.model
            self._originals.clear()  # новый запуск — сбрасываем кеш оригиналов
            self._emit_state("running")
            self._log(f"Старт: {len(files)} файлов, модель {opts.model}, "
                      f"язык {opts.language}", "info")

            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True, "Запущено"

    def stop(self) -> bool:
        if self.state != "running":
            return False
        self._stop_flag.set()
        self.state = "stopping"
        self._emit_state("stopping")
        self._log("Получен сигнал остановки (с сохранением готового)", "warn")
        return True

    # --- внутреннее ---

    def _out_dir(self) -> Path:
        if not self.options:
            return Path(".")
        return self._out_dir_for(self.options)

    @staticmethod
    def _out_dir_for(opts: TranscribeOptions) -> Path:
        if opts.output_dir:
            return Path(opts.output_dir)
        return Path(opts.input_dir) / "transcripts"

    def get_original(self, stem: str) -> Optional[str]:
        """Возвращает оригинальный текст .md (до переименования спикеров).
        Если оригинал не в кеше — читает текущий .md. Если в нём есть SPEAKER_NN,
        считает его оригиналом и кеширует. Иначе возвращает None (оригинал утерян)."""
        if stem in self._originals:
            return self._originals[stem]
        out_dir = self._out_dir()
        md = out_dir / f"{stem}.md"
        if not md.exists():
            return None
        try:
            text = md.read_text(encoding="utf-8")
        except Exception:
            return None
        # если в тексте есть SPEAKER_NN — считаем это оригиналом
        import re as _re
        if _re.search(r"\bSPEAKER_\d+\b", text):
            self._originals[stem] = text
            return text
        return None

    def _log(self, msg: str, level: str = "info"):
        if self.on_log:
            try:
                self.on_log(msg, level)
            except Exception:
                pass

    def _emit_state(self, state: str):
        if self.on_state:
            try:
                self.on_state(state)
            except Exception:
                pass

    def _emit_progress(self, task: FileTask):
        if self.on_progress:
            try:
                self.on_progress(task)
            except Exception:
                pass

    def _run(self):
        """Главный цикл воркера (в отдельном потоке)."""
        opts = self.options
        try:
            device, compute_type = pick_device()
            self.device = device
            self.compute_type = compute_type
            self._log(f"Устройство: {device} ({compute_type}), модель: {opts.model}", "info")
            self._log("Загрузка модели (при первом запуске — скачивание)...", "info")
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                self._log("faster-whisper не установлен. Установите: pip install faster-whisper", "error")
                self.state = "finished"
                self.finished_at = time.time()
                self._emit_state("finished")
                return

            model = WhisperModel(opts.model, device=device, compute_type=compute_type)
            self._log("Модель загружена", "info")

            # диаризация (опционально)
            diar_pipeline = None
            if opts.diarize:
                diar_pipeline = self._load_diarization(opts)
                if diar_pipeline is None:
                    self._log("Диаризация отключена: продолжаем без разделения по ролям", "warn")
                else:
                    self._log("Диаризация готова (whisperX + pyannote)", "info")

            todo = [t for t in self.tasks if t.status == "queued"]
            self._log(f"В очереди: {len(todo)}, уже готово/пропущено: "
                      f"{len(self.tasks) - len(todo)}", "info")

            for i, task in enumerate(todo, 1):
                if self._stop_flag.is_set():
                    task.status = "stopped"
                    self._emit_progress(task)
                    break
                self._log(f"[{i}/{len(todo)}] {task.name}", "info")
                try:
                    self._transcribe_file(model, task, opts, diar_pipeline)
                except KeyboardInterrupt:
                    task.status = "stopped"
                    self._emit_progress(task)
                    self._log("Прервано пользователем. Готовые файлы сохранены.", "warn")
                    break
                except Exception as e:
                    task.status = "error"
                    task.error = str(e)
                    self._emit_progress(task)
                    self._log(f"ОШИБКА [{task.name}]: {e}", "error")
                    self._log(f"Подсказка: можно извлечь аудио — "
                              f'ffmpeg -i "{task.name}" -ac 1 -ar 16000 out.wav', "info")

            self.state = "finished" if not self._stop_flag.is_set() else "finished"
            self.finished_at = time.time()
            self._emit_state("finished")
            done = sum(1 for t in self.tasks if t.status == "done")
            errs = sum(1 for t in self.tasks if t.status == "error")
            self._log(f"Итог: успешно {done}, ошибок {errs}", "info")

        except Exception as e:
            self._log(f"Критическая ошибка воркера: {e}", "error")
            self._log(traceback.format_exc(), "error")
            self.state = "finished"
            self.finished_at = time.time()
            self._emit_state("finished")

    def _load_diarization(self, opts: TranscribeOptions):
        """Пытается загрузить пайплайн диаризации whisperX. Возвращает None при неудаче."""
        try:
            import whisperx
        except ImportError:
            self._log("whisperx не установлен — диаризация недоступна. "
                      "Установите: pip install whisperx", "warn")
            return None
        if not opts.hf_token:
            self._log("Не задан HuggingFace-токен — диаризация недоступна. "
                      "Получите токен: https://huggingface.co/settings/tokens "
                      "(нужны права Read и Accept User Conditions для pyannote)", "warn")
            return None
        try:
            device = self.device or "cpu"
            self._log("Загрузка модели диаризации pyannote/speaker-diarization-3.1...", "info")
            pipeline = whisperx.DiarizationPipeline(
                use_auth_token=opts.hf_token,
                device=device,
            )
            self._log("Модель диаризации загружена", "info")
            return pipeline
        except Exception as e:
            self._log(f"Не удалось загрузить модель диаризации: {e}", "warn")
            self._log("Проверьте: 1) верный HF-токен; 2) принято соглашение на "
                      "https://huggingface.co/pyannote/speaker-diarization-3.1; "
                      "3) на https://huggingface.co/pyannote/segmentation-3.0", "warn")
            return None

    def _transcribe_file(self, model, task: FileTask, opts: TranscribeOptions, diar_pipeline=None):
        """Транскрибирует один файл. Атомарная запись через .part.
        Если передан diar_pipeline — текст разбивается по спикерам."""
        dst = task.md_path
        tmp = dst.with_suffix(dst.suffix + ".part")
        t0 = time.monotonic()
        task.status = "running"
        task.started_at = time.time()
        self._emit_progress(task)

        segments, info = model.transcribe(
            str(task.path),
            language=opts.language if opts.language and opts.language != "auto" else None,
            vad_filter=opts.vad_filter,
            beam_size=opts.beam_size,
            condition_on_previous_text=False,
            initial_prompt=opts.prompt if opts.prompt else None,
        )
        duration = info.duration or 0.0
        task.duration = duration
        self._log(f"  длительность: {fmt_ts(duration)}, язык: {info.language} "
                  f"(p={info.language_probability:.2f})", "info")

        # собираем сегменты в список (нужно для диаризации и для расчёта прогресса)
        seg_list = []
        for seg in segments:
            if self._stop_flag.is_set():
                break
            seg_list.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            })

        # если включена диаризация — накладываем спикеров
        if diar_pipeline is not None and not self._stop_flag.is_set():
            try:
                seg_list = self._apply_diarization(
                    diar_pipeline, str(task.path), seg_list, opts
                )
            except Exception as e:
                self._log(f"  диаризация не удалась ({e}), пишу без разделения", "warn")

        # пишем .md
        next_mark = 0.0
        word_count = 0
        last_speaker = None
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(f"# {task.path.stem}\n\n")
            for s in seg_list:
                if self._stop_flag.is_set():
                    break
                speaker = s.get("speaker")  # None, если диаризации нет
                # таймкод каждые 5 минут (без диаризации) или при смене спикера
                if speaker is None and opts.timecodes and s["start"] >= next_mark:
                    f.write(f"\n**[{fmt_ts(s['start'])}]**\n\n")
                    next_mark = s["start"] - (s["start"] % TIMECODE_STEP) + TIMECODE_STEP
                if speaker is not None:
                    if speaker != last_speaker:
                        f.write(f"\n**[{fmt_ts(s['start'])}] {speaker}:** {s['text']}\n")
                        last_speaker = speaker
                    else:
                        # продолжение реплики того же спикера
                        f.write(f" {s['text']}")
                else:
                    f.write(s["text"] + "\n")
                word_count += len(s["text"].split())
                task.current_seg = s["end"]
                if duration > 0:
                    pct = s["end"] / duration * 100
                    task.progress = min(pct, 100.0)
                    task.elapsed = time.monotonic() - t0
                    if pct > 0:
                        task.eta = task.elapsed * (100 - pct) / pct
                    self._emit_progress(task)

        # если остановлено до завершения — оставляем .part, не переименовываем
        if self._stop_flag.is_set():
            task.status = "stopped"
            tmp.unlink(missing_ok=True)  # удаляем недописанный .part
            self._emit_progress(task)
            return

        tmp.replace(dst)
        task.status = "done"
        task.progress = 100.0
        task.elapsed = time.monotonic() - t0
        task.finished_at = time.time()
        task.word_count = word_count
        self._emit_progress(task)
        self._log(f"  готово за {fmt_ts(task.elapsed)} → {dst.name} "
                  f"({word_count} слов)" +
                  (" [с диаризацией]" if diar_pipeline else ""), "info")

    def _apply_diarization(self, pipeline, audio_path: str, segments: list, opts: TranscribeOptions) -> list:
        """Накладывает метки спикеров на распознанные сегменты.
        Возвращает новый список сегментов с добавленным полем 'speaker'."""
        import whisperx
        # диаризуем аудио
        diar_kwargs = {}
        if opts.min_speakers:
            diar_kwargs["min_speakers"] = opts.min_speakers
        if opts.max_speakers:
            diar_kwargs["max_speakers"] = opts.max_speakers
        diar_segments = pipeline(audio_path, **diar_kwargs)
        # сопоставляем слова из транскрипции со спикерами
        result = {"segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments]}
        result = whisperx.assign_word_speakers(diar_segments, result)
        # собираем обратно, прокидывая speaker в каждый сегмент
        out = []
        for s in result.get("segments", []):
            out.append({
                "start": s["start"],
                "end": s["end"],
                "text": s["text"].strip(),
                "speaker": s.get("speaker"),
            })
        return out


# глобальный singleton
manager = TranscribeManager()
