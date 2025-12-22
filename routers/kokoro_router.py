from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from schemas.kokoro_schema import TTSRequest
from services.kokoro_service import generate_tts_audio
import os

router = APIRouter()

@router.get("/")
def read_root():
    return {"message": "Welcome to Kokoro TTS API"}


from fastapi.responses import Response
from fastapi import HTTPException
import os

@router.get("/download/{filename}")
@router.head("/download/{filename}")
def download_file(filename: str):
    filepath = os.path.join("output", filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    with open(filepath, "rb") as f:
        audio_bytes = f.read()

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',

            # 🔥 Your requested headers:
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
            "Access-Control-Allow-Headers": "Range, Authorization",

            # 🔥 These help audio scrubbing in browsers:
            "Accept-Ranges": "bytes",
        }
    )

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
