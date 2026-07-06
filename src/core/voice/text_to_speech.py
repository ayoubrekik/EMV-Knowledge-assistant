from pathlib import Path
from typing import Dict
import urllib.request
import subprocess
import uuid


AUDIO_DIR = Path("/app/src/storage/tts_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

PIPER_DIR = Path("/root/.cache/huggingface/piper/en_US-amy-medium")
PIPER_MODEL = PIPER_DIR / "en_US-amy-medium.onnx"
PIPER_CONFIG = PIPER_DIR / "en_US-amy-medium.onnx.json"

MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"

def download_piper_voice() -> None:
    PIPER_DIR.mkdir(parents=True, exist_ok=True)

    if not PIPER_MODEL.exists():
        urllib.request.urlretrieve(MODEL_URL, PIPER_MODEL)

    if not PIPER_CONFIG.exists():
        urllib.request.urlretrieve(CONFIG_URL, PIPER_CONFIG)


def text_to_speech(text: str, language: str = "en", slow: bool = False) -> Dict:
    try:
        text = (text or "").strip()

        if not text:
            return {
                "success": False,
                "error": "Empty text"
            }

        download_piper_voice()

        audio_filename = f"response_{uuid.uuid4().hex[:8]}.wav"
        audio_path = AUDIO_DIR / audio_filename

        subprocess.run(
            [
                "piper",
                "--model",
                str(PIPER_MODEL),
                "--output_file",
                str(audio_path),
            ],
            input=text,
            text=True,
            check=True,
        )

        return {
            "success": True,
            "audio_filename": audio_filename,
            "audio_path": str(audio_path)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }