from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid

from src.core.voice.speech_to_text import speech_to_text
from fastapi.responses import FileResponse
from src.core.voice.text_to_speech import text_to_speech, AUDIO_DIR

router = APIRouter(prefix="/voice", tags=["Voice"])

TEMP_AUDIO_DIR = Path("/app/src/storage/temp_audio")
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/transcribe")
async def transcribe_voice(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Audio file is required")

    suffix = Path(audio.filename).suffix or ".webm"
    temp_path = TEMP_AUDIO_DIR / f"voice_{uuid.uuid4().hex}{suffix}"

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        result = speech_to_text(str(temp_path))

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error"))

        return {
            "success": True,
            "transcript": result.get("text", ""),
            "language": result.get("language")
        }

    finally:
        if temp_path.exists():
            temp_path.unlink()



@router.post("/tts")
async def tts_voice(payload: dict):
    text = (payload.get("text") or "").strip()
    language = payload.get("language", "en")
    slow = bool(payload.get("slow", False))

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    result = text_to_speech(
        text=text,
        language=language,
        slow=slow
    )

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error"))

    return {
        "success": True,
        "audio_url": f"/voice/audio/{result['audio_filename']}"
    }


@router.get("/audio/{filename}")
async def serve_tts_audio(filename: str):
    audio_path = AUDIO_DIR / filename

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav"
    )