#!/usr/bin/env python3
"""Persistent STT/TTS worker — fariza-til venvida ishlaydi, modellarni bir marta yuklab xotirada saqlaydi.

Protokol: stdin/stdout orqali har qatorda bitta JSON.
  {"id": 1, "cmd": "status"}
  {"id": 2, "cmd": "stt", "path": "/abs/file.wav", "lang": "uz"}
  {"id": 3, "cmd": "tts", "text": "...", "out": "/abs/out.wav", "speaker": "dilnavoz", "sample_rate": 16000}
  {"id": 4, "cmd": "quit"}

Javoblar: {"id": 1, "ok": true, ...} yoki {"id": 1, "ok": false, "error": "..."}
Ishga tushganda: {"ready": true, "stage": "silero"|"asr", ...}
"""
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

TIL_PATH = Path("/home/tony/.fariza/config/opencode/skills/fariza-til/til.py")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_til():
    spec = importlib.util.spec_from_file_location("fariza_til", TIL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg):
    sys.stderr.write(f"[worker] {msg}\n")
    sys.stderr.flush()


def main():
    til = load_til()
    state = {"silero": None, "asr": None, "device": None, "asr_error": None}

    try:
        model, device = til.load_silero(False)
        state["silero"] = model
        state["device"] = device.type
        emit({"ready": True, "stage": "silero", "device": device.type})
        log(f"silero yuklandi ({device.type})")
    except Exception as exc:
        emit({"ready": False, "error": f"silero: {exc}"})
        return

    try:
        state["asr"] = til.load_asr(til.DEFAULT_STT_MODEL, False)
        emit({"ready": True, "stage": "asr", "device": state["device"]})
        log("whisper yuklandi")
    except Exception as exc:
        state["asr_error"] = str(exc)
        emit({"ready": True, "stage": "asr_failed", "error": str(exc)})
        log(f"whisper xato: {exc}")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        rid = req.get("id")
        cmd = req.get("cmd")
        try:
            if cmd == "status":
                emit({
                    "id": rid, "ok": True,
                    "device": state["device"],
                    "silero": state["silero"] is not None,
                    "asr": state["asr"] is not None,
                    "asr_error": state["asr_error"],
                })

            elif cmd == "stt":
                if state["asr"] is None:
                    state["asr"] = til.load_asr(til.DEFAULT_STT_MODEL, False)
                    state["asr_error"] = None
                audio = til.load_audio(Path(req["path"]))
                text = til.transcribe(state["asr"], audio, req.get("lang", "uz"))
                emit({"id": rid, "ok": True, "text": text, "device": state["device"]})

            elif cmd == "tts":
                rate = int(req.get("sample_rate", 48000))
                speaker = req.get("speaker", til.SILERO_SPEAKER)
                audio = til.synth(state["silero"], req["text"], speaker, rate, False)
                out = Path(req["out"])
                out.parent.mkdir(parents=True, exist_ok=True)
                til.save_wav(audio, out, rate)
                emit({
                    "id": rid, "ok": True, "file": str(out),
                    "seconds": round(len(audio) / rate, 2), "speaker": speaker,
                })

            elif cmd == "quit":
                emit({"id": rid, "ok": True})
                break

            else:
                emit({"id": rid, "ok": False, "error": f"noma'lum buyruq: {cmd}"})

        except Exception as exc:
            log(traceback.format_exc())
            emit({"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
