#!/usr/bin/env python3
"""Pocket AI Server — Raspberry Pi Pico W cho'ntak yordamchisi uchun AI backend.

Oqim:  audio (wav/pcm16)  ->  STT (Whisper uz)  ->  LLM (deepseek-flash)  ->  TTS (Silero uz)  ->  javob wav

Endpointlar:
  GET  /healthz                     holat
  POST /chat                        {"text": "...", "session_id": "..."} — matnli test
  POST /voice                       multipart: file=<wav>, session_id=<opt>
  POST /voice/raw                   xom body: wav yoki ?format=pcm16&rate=16000
  GET  /audio/{name}                javob wav faylini yuklab olish

Ishga tushirish: ./run.sh   (yoki .venv/bin/python server.py)
"""
import json
import queue
import subprocess
import threading
import time
import urllib.request
import uuid
import wave
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

BASE = Path(__file__).resolve().parent
SKILL = Path("/home/tony/.fariza/config/opencode/skills/fariza-til")
VENV_PY = SKILL / ".venv/bin/python"
WORKER_PY = BASE / "stt_tts_worker.py"
DATA_IN = BASE / "data/in"
DATA_OUT = BASE / "data/out"
AUTH_FILE = Path("/home/tony/.fariza/data/opencode/auth.json")

CONFIG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
LLM_CFG = CONFIG["llm"]
ASST_CFG = CONFIG["assistant"]
STT_CFG = CONFIG["stt"]
TTS_CFG = CONFIG["tts"]

DATA_IN.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)


def read_llm_key():
    provider = LLM_CFG.get("auth_provider", "deepseek")
    try:
        auth = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        return auth[provider]["key"]
    except Exception:
        return None


class Worker:
    """fariza-til venvidagi persistent STT/TTS jarayoni bilan JSON-lines aloqa."""

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.pending = {}
        self.next_id = 1
        self.ready = False
        self.stage = None
        self.device = None
        self.error = None
        self._stage_event = threading.Event()

    def start(self, wait_seconds=420):
        self.proc = subprocess.Popen(
            [str(VENV_PY), str(WORKER_PY)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._reader, daemon=True).start()
        got = self._stage_event.wait(timeout=wait_seconds)
        if not got:
            self.error = "worker tayyor bo'lmadi (timeout)"
        return self.ready

    def _reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ready" in msg:
                if msg.get("ready") is False:
                    self.error = msg.get("error")
                    self._stage_event.set()
                else:
                    self.ready = True
                    self.device = msg.get("device") or self.device
                    self.stage = msg.get("stage")
                    if msg.get("stage") in ("asr", "asr_failed"):
                        self._stage_event.set()
                continue
            rid = msg.get("id")
            q = self.pending.get(rid)
            if q is not None:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass

    def request(self, cmd, timeout=180, **kw):
        with self.lock:
            if not self.proc or self.proc.poll() is not None:
                return {"ok": False, "error": "worker jarayoni to'xtagan"}
            rid = self.next_id
            self.next_id += 1
            q = queue.Queue(maxsize=1)
            self.pending[rid] = q
            payload = json.dumps({"id": rid, "cmd": cmd, **kw}, ensure_ascii=False)
            try:
                self.proc.stdin.write(payload + "\n")
                self.proc.stdin.flush()
            except Exception as exc:
                self.pending.pop(rid, None)
                return {"ok": False, "error": f"worker yozish xato: {exc}"}
            try:
                return q.get(timeout=timeout)
            except queue.Empty:
                return {"ok": False, "error": f"worker javob bermadi ({timeout}s)"}
            finally:
                self.pending.pop(rid, None)


WORKER = Worker()

SESSIONS = {}
SESSION_LOCK = threading.Lock()
LLM_KEY = read_llm_key()

app = FastAPI(title="Pocket AI Server", version="0.1")


def session_history(session_id):
    with SESSION_LOCK:
        return list(SESSIONS.get(session_id, []))


def session_append(session_id, role, content):
    limit = int(CONFIG.get("max_history", 8)) * 2
    with SESSION_LOCK:
        hist = SESSIONS.setdefault(session_id, [])
        hist.append({"role": role, "content": content})
        if len(hist) > limit:
            del hist[:-limit]


def llm_reply(session_id, user_text):
    if not LLM_KEY:
        return "", "LLM kaliti topilmadi (auth.json)"
    messages = [{"role": "system", "content": ASST_CFG["system_prompt"]}]
    messages += session_history(session_id)
    messages.append({"role": "user", "content": user_text})
    body = json.dumps({
        "model": LLM_CFG["model"],
        "messages": messages,
        "max_tokens": int(LLM_CFG.get("max_tokens", 300)),
        "temperature": float(LLM_CFG.get("temperature", 0.6)),
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        LLM_CFG.get("base_url", "https://api.deepseek.com") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=int(LLM_CFG.get("timeout", 40))) as resp:
            data = json.load(resp)
        reply = (data["choices"][0]["message"]["content"] or "").strip()
        return reply, None
    except Exception as exc:
        return "", f"LLM xato: {exc}"


def analyze_and_reply(in_path, session_id):
    t0 = time.time()
    stt = WORKER.request("stt", path=str(in_path), lang=STT_CFG.get("lang", "uz"))
    t1 = time.time()
    if not stt.get("ok"):
        return {"ok": False, "stage": "stt", "error": stt.get("error")}
    transcript = (stt.get("text") or "").strip()
    if not transcript:
        return {"ok": True, "transcript": "", "reply": "", "reply_url": None,
                "timings": {"stt": round(t1 - t0, 2)}}

    session_append(session_id, "user", transcript)
    reply, err = llm_reply(session_id, transcript)
    t2 = time.time()
    if err:
        return {"ok": False, "stage": "llm", "transcript": transcript, "error": err}
    session_append(session_id, "assistant", reply)

    out_name = f"{uuid.uuid4().hex[:12]}.wav"
    out_path = DATA_OUT / out_name
    tts = WORKER.request("tts", text=reply, out=str(out_path),
                         speaker=TTS_CFG.get("speaker", "dilnavoz"),
                         sample_rate=int(TTS_CFG.get("sample_rate", 16000)))
    t3 = time.time()
    if not tts.get("ok"):
        return {"ok": False, "stage": "tts", "transcript": transcript,
                "reply": reply, "error": tts.get("error")}

    return {
        "ok": True,
        "session_id": session_id,
        "transcript": transcript,
        "reply": reply,
        "reply_url": f"/audio/{out_name}",
        "timings": {
            "stt": round(t1 - t0, 2),
            "llm": round(t2 - t1, 2),
            "tts": round(t3 - t2, 2),
            "total": round(t3 - t0, 2),
        },
    }


def pcm16_to_wav(raw, rate):
    path = DATA_IN / f"{uuid.uuid4().hex[:12]}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(raw)
    return path


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "worker_ready": WORKER.ready,
        "worker_stage": WORKER.stage,
        "device": WORKER.device,
        "worker_error": WORKER.error,
        "llm_model": LLM_CFG["model"],
        "llm_key": bool(LLM_KEY),
        "sessions": len(SESSIONS),
    }


@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "JSON body kerak")
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text bo'sh")
    session_id = body.get("session_id") or "default"
    t0 = time.time()
    session_append(session_id, "user", text)
    reply, err = llm_reply(session_id, text)
    if err:
        raise HTTPException(502, err)
    session_append(session_id, "assistant", reply)
    return {"ok": True, "reply": reply, "session_id": session_id,
            "seconds": round(time.time() - t0, 2)}


@app.post("/voice")
async def voice(file: UploadFile = File(...), session_id: str = Form("default")):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "audio bo'sh")
    in_path = DATA_IN / f"{uuid.uuid4().hex[:12]}.wav"
    in_path.write_bytes(raw)
    result = analyze_and_reply(in_path, session_id)
    return JSONResponse(result, status_code=200 if result.get("ok") else 502)


@app.post("/voice/raw")
async def voice_raw(request: Request, format: str = "wav", rate: int = 16000,
                    session_id: str = "default"):
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "audio bo'sh")
    if format == "pcm16":
        in_path = pcm16_to_wav(raw, rate)
    else:
        in_path = DATA_IN / f"{uuid.uuid4().hex[:12]}.wav"
        in_path.write_bytes(raw)
    result = analyze_and_reply(in_path, session_id)
    return JSONResponse(result, status_code=200 if result.get("ok") else 502)


@app.get("/audio/{name}")
def audio(name: str):
    path = (DATA_OUT / name).resolve()
    if not str(path).startswith(str(DATA_OUT)) or not path.exists():
        raise HTTPException(404, "fayl topilmadi")
    return FileResponse(path, media_type="audio/wav")


if __name__ == "__main__":
    print("worker ishga tushirilmoqda (modellar yuklanmoqda)...", flush=True)
    ok = WORKER.start()
    print(f"worker ready={WORKER.ready} stage={WORKER.stage} device={WORKER.device} err={WORKER.error}", flush=True)
    if not ok and WORKER.error:
        print("DIQQAT: worker muammosi:", WORKER.error, flush=True)
    uvicorn.run(app, host=CONFIG.get("host", "0.0.0.0"), port=int(CONFIG.get("port", 8400)))
