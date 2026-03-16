import os
import uuid
import logging
import requests
from typing import Optional, Any

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    FastAPI
)
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from google import genai
from google.genai import types

from config.settings import SETTINGS
from config.logger import logger

app = FastAPI(title="Indic Text To Speech")

# =========================================================
# CONFIG
# =========================================================

ELEVENLABS_API_KEY = SETTINGS.ELEVENLABS_API_KEY
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_TTS_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"
AUDIO_FILE_PATH = SETTINGS.AUDIO_FILE_PATH

ELEVENLABS_VOICE_MAP = {
    "en": "21m00Tcm4obsnIqyzB0n",
    "hi": "pFmYtHwV4pP13J24b6x7",
    "ta": "pN9z3LhFv23v99g2g5n9",
    "will": "21m00Tcm4TlvDq8ikWAM",
    "rachel": "bIHbv24MWmeRgasZH58o",
}

os.makedirs(AUDIO_FILE_PATH, exist_ok=True)

# =========================================================
# ROUTER
# =========================================================

router = APIRouter(prefix="/transcribe", tags=["speech-text"])

# =========================================================
# SCHEMAS
# =========================================================

class STTResponse(BaseModel):
    transcription: str


class TTSRequest(BaseModel):
    text: str
    lang: str
    model_name: Optional[str] = None


class TTSResponse(BaseModel):
    audio_url: str
    mouth_cues: Any = []


class TTTRequest(BaseModel):
    text: str
    src_lang: str
    target_lang: str


class TTTResponse(BaseModel):
    transcription: str


# =========================================================
# INTERNAL HELPERS (SERVICE LOGIC)
# =========================================================

def _check_elevenlabs_key():
    if not ELEVENLABS_API_KEY:
        raise HTTPException(500, "ELEVENLABS_API_KEY not configured")


def _get_voice_id(lang: str, model_name: Optional[str]):
    if model_name and model_name in ELEVENLABS_VOICE_MAP:
        return ELEVENLABS_VOICE_MAP[model_name]
    if lang in ELEVENLABS_VOICE_MAP:
        return ELEVENLABS_VOICE_MAP[lang]
    return ELEVENLABS_VOICE_MAP["en"]


async def _translate_text(text: str, src: str, tgt: str) -> str:
    client = genai.Client(api_key=SETTINGS.LLM_PROVIDER_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text,
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=f"""
Translate text only.
Source: {src}
Target: {tgt}
No explanations.
""",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


# =========================================================
# ROUTES
# =========================================================

@router.post("/stt", response_model=STTResponse)
async def speech_to_text(
    file: UploadFile = File(...),
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
):
    _check_elevenlabs_key()

    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    files = {"file": (file.filename, await file.read(), file.content_type)}
    data = {"model_id": "scribe_v1"}

    response = requests.post(
        ELEVENLABS_STT_URL, headers=headers, files=files, data=data
    )

    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)

    text = response.json().get("text", "")

    if source_lang and target_lang and source_lang != target_lang:
        text = await _translate_text(text, source_lang, target_lang)

    return {"transcription": text}


@router.post("/tts", response_model=TTSResponse)
async def text_to_speech(req: TTSRequest):
    _check_elevenlabs_key()

    voice_id = _get_voice_id(req.lang, req.model_name)
    file_id = f"{req.lang}_{uuid.uuid4()}.mp3"
    file_path = os.path.join(AUDIO_FILE_PATH, file_id)

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
    }

    payload = {
        "text": req.text,
        "model_id": "eleven_multilingual_v2",
    }

    response = requests.post(
        f"{ELEVENLABS_TTS_BASE_URL}/{voice_id}",
        headers=headers,
        json=payload,
    )

    if response.status_code != 200:
        raise HTTPException(response.status_code, response.text)

    with open(file_path, "wb") as f:
        f.write(response.content)

    return {
        "audio_url": f"{SETTINGS.BASE_URL_AUDIO}/download/{file_id}",
        "mouth_cues": [],
    }


@router.get("/download/{filename}")
async def download_audio(filename: str):
    file_path = os.path.join(AUDIO_FILE_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Audio file not found")

    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)


@router.post("/ttt", response_model=TTTResponse)
async def text_to_text(req: TTTRequest):
    text = await _translate_text(req.text, req.src_lang, req.target_lang)
    return {"transcription": text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=INDIC_PORT)
