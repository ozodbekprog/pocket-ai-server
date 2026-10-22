#!/usr/bin/env python3
"""End-to-end test: matn -> TTS (test audiosi) -> serverga /voice -> transkript + javob + javob wav.

Ishlatish (server ishlab turishi kerak):
    .venv/bin/python client_test.py "Bugun Toshkentda havo qanday?"
"""
import json
import subprocess
import sys
from pathlib import Path

import httpx

BASE = Path(__file__).resolve().parent
TIL_PY = Path("/home/tony/.fariza/config/opencode/skills/fariza-til/til.py")
TIL_VENV_PY = Path("/home/tony/.fariza/config/opencode/skills/fariza-til/.venv/bin/python")

CONFIG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
PORT = int(CONFIG.get("port", 8400))


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Salom, men cho'ntak yordamchisiman. Bugun ob-havo qanday?"
    in_wav = Path("/tmp/opencode/pocket_test_in.wav")
    in_wav.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] TTS bilan test audiosi yaratilmoqda: {in_wav}")
    subprocess.run(
        [str(TIL_VENV_PY), str(TIL_PY), "tts", text, "--out", str(in_wav)],
        check=True, capture_output=True, text=True,
    )

    print(f"[2/3] POST /voice -> http://127.0.0.1:{PORT}")
    with httpx.Client(timeout=180) as client:
        with open(in_wav, "rb") as f:
            resp = client.post(
                f"http://127.0.0.1:{PORT}/voice",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"session_id": "test"},
            )
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if not data.get("ok"):
            return 1

        reply_url = data.get("reply_url")
        if reply_url:
            out_wav = Path("/tmp/opencode/pocket_test_reply.wav")
            r = client.get(f"http://127.0.0.1:{PORT}{reply_url}")
            out_wav.write_bytes(r.content)
            print(f"[3/3] Javob audiosi saqlandi: {out_wav}")
            subprocess.run(["paplay", str(out_wav)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
