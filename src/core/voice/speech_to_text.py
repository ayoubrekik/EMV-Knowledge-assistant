from pathlib import Path
from typing import Dict, Optional
from faster_whisper import WhisperModel
import os

from src.core.db.chroma_client import resolve_hf_snapshot

_model: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    global _model

    if _model is None:
        # model_name = os.getenv(
        #     "WHISPER_MODEL",
        #     "Systran/faster-whisper-small"
        # )
        model_name="Systran/faster-whisper-small"
        model_path = resolve_hf_snapshot(model_name)

        _model = WhisperModel(
            model_path,
            device="cpu",#os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type="int8"#os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        )

    return _model


def speech_to_text(audio_file_path: str) -> Dict:
    try:
        path = Path(audio_file_path)

        if not path.exists():
            return {"success": False, "error": "Audio file not found"}

        print("AUDIO PATH:", path)
        print("AUDIO SIZE:", path.stat().st_size, "bytes")

        if path.stat().st_size < 2000:
            return {
                "success": False,
                "error": "Audio file is too small or empty"
            }

        model = get_whisper_model()

        segments, info = model.transcribe(
            str(path),
            language="en",
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=False,
            no_speech_threshold=0.9,
            log_prob_threshold=-2.0,
            compression_ratio_threshold=2.8
        )

        segments = list(segments)

        print("DETECTED LANGUAGE:", info.language)
        print("SEGMENTS:", segments)

        text = " ".join(s.text.strip() for s in segments).strip()

        return {
            "success": True,
            "text": text,
            "language": info.language
        }

    except Exception as e:
        return {"success": False, "error": str(e)}