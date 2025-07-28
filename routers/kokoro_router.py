from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from schemas.kokoro_schema import TTSRequest
from services.kokoro_service import generate_tts_audio
import os

router = APIRouter()

@router.get("/download/{filename}")
def download_file(filename: str):
    filepath = os.path.join("output", filename)
    if os.path.exists(filepath):
        return FileResponse(path=filepath, filename=filename, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="File not found")

@router.post("/tts")
async def synthesize_tts(request: TTSRequest):
    try:
        result = generate_tts_audio(
            text=request.text,
            voice=request.voice,
            lang_code=request.lang_code,
            speed=request.speed
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from constants.kokoro_meta import LANGUAGE_VOICES

@router.get("/languages")
def get_languages():
    return [
        {"lang_code": code, "language": meta["name"]}
        for code, meta in LANGUAGE_VOICES.items()
    ]

@router.get("/voices/{lang_code}")
def get_voices_by_language(lang_code: str):
    if lang_code not in LANGUAGE_VOICES:
        raise HTTPException(status_code=404, detail="Language code not found")
    return {
        "lang_code": lang_code,
        "language": LANGUAGE_VOICES[lang_code]["name"],
        "voices": LANGUAGE_VOICES[lang_code]["voices"]
    }
