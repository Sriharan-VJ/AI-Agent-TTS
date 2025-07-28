import os
import uuid
import torch
import logging
import traceback
import numpy as np
import soundfile as sf

from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config.cors_options import configure_cors
from config.settings import (
    DEVICE,
    ARABIC_PORT,
    ARABIC_BASE_URL,
    THAI_MODEL_PATH,
    ARABIC_MODEL_PATH,
    AUDIO_FILE_PATH,
)

from transformers import VitsModel, AutoTokenizer

app = FastAPI(title="Arabic and Thai")
configure_cors(app)
app.mount("/static", StaticFiles(directory="static"), name="static")


arabic_model = VitsModel.from_pretrained(ARABIC_MODEL_PATH).to(DEVICE)
arabic_tokenizer = AutoTokenizer.from_pretrained(ARABIC_MODEL_PATH)

thai_model = VitsModel.from_pretrained(THAI_MODEL_PATH).to(DEVICE)
thai_tokenizer = AutoTokenizer.from_pretrained(THAI_MODEL_PATH)


class TTSRequest(BaseModel):
    lang: str
    text: str
    model_name: Optional[str] = None


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(AUDIO_FILE_PATH, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    print("TEXT:", request.text)

    if request.lang not in ["ar", "th"]:
        logging.error(f"Unsupported language code: {request.lang}")
        raise HTTPException(status_code=400, detail="Unsupported language code")

    try:
        unique_id = str(uuid.uuid4())

        if request.lang == "ar":
            inputs = arabic_tokenizer(request.text, return_tensors="pt")
            inputs["input_ids"] = inputs["input_ids"].long()
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            if inputs["input_ids"].shape[1] == 0:
                raise ValueError(
                    "Tokenized input is empty. Please provide valid Arabic text."
                )

            with torch.no_grad():
                output = arabic_model(**inputs).waveform

            if not isinstance(output, np.ndarray):
                output = output.cpu().numpy()

            output_filename = f"{AUDIO_FILE_PATH}/ar_{unique_id}.mp3"
            sf.write(
                output_filename, output[0], samplerate=arabic_model.config.sampling_rate
            )

            download_url = f"{ARABIC_BASE_URL}/download/ar_{unique_id}.mp3"
            return {"audio_url": download_url}

        elif request.lang == "th":
            inputs = thai_tokenizer(request.text, return_tensors="pt")
            inputs = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            if inputs["input_ids"].shape[1] == 0:
                raise ValueError(
                    "Tokenized input is empty. Please provide valid Thai text."
                )

            with torch.no_grad():
                output = thai_model(**inputs).waveform

            if not isinstance(output, np.ndarray):
                output = output.cpu().numpy()

            output_filename = f"{AUDIO_FILE_PATH}/th_{unique_id}.mp3"
            sf.write(
                output_filename, output[0], samplerate=arabic_model.config.sampling_rate
            )

            download_url = f"{ARABIC_BASE_URL}/download/th_{unique_id}.mp3"
            return {"audio_url": download_url}

    except Exception as e:
        logging.error(f"Exception: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print(
        """

███████████████████████████████████████████████████████████████████████████████████████████████
█░░░░░░░░░░░░░░█░░░░░░░░░░░░░░░░███░░░░░░░░░░░░░░█░░░░░░░░░░░░░░█░░░░░░██░░░░░░█░░░░░░░░░░░░░░█
█░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀▄▀▄▀▄▀▄▀▄▀░░███░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀░░██░░▄▀░░█░░▄▀▄▀▄▀▄▀▄▀░░█
█░░▄▀░░░░░░▄▀░░█░░▄▀░░░░░░░░▄▀░░███░░▄▀░░░░░░▄▀░░█░░░░░░▄▀░░░░░░█░░▄▀░░██░░▄▀░░█░░▄▀░░░░░░▄▀░░█
█░░▄▀░░██░░▄▀░░█░░▄▀░░████░░▄▀░░███░░▄▀░░██░░▄▀░░█████░░▄▀░░█████░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░█
█░░▄▀░░░░░░▄▀░░█░░▄▀░░░░░░░░▄▀░░███░░▄▀░░░░░░▄▀░░█████░░▄▀░░█████░░▄▀░░░░░░▄▀░░█░░▄▀░░░░░░▄▀░░█
█░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀▄▀▄▀▄▀▄▀▄▀░░███░░▄▀▄▀▄▀▄▀▄▀░░█████░░▄▀░░█████░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀▄▀▄▀▄▀▄▀░░█
█░░▄▀░░░░░░▄▀░░█░░▄▀░░░░░░▄▀░░░░███░░▄▀░░░░░░▄▀░░█████░░▄▀░░█████░░▄▀░░░░░░▄▀░░█░░▄▀░░░░░░▄▀░░█
█░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░█████░░▄▀░░██░░▄▀░░█████░░▄▀░░█████░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░█
█░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░░░░░█░░▄▀░░██░░▄▀░░█████░░▄▀░░█████░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░█
█░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀▄▀▄▀░░█░░▄▀░░██░░▄▀░░█████░░▄▀░░█████░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░█
█░░░░░░██░░░░░░█░░░░░░██░░░░░░░░░░█░░░░░░██░░░░░░█████░░░░░░█████░░░░░░██░░░░░░█░░░░░░██░░░░░░█
███████████████████████████████████████████████████████████████████████████████████████████████"""
    )
    uvicorn.run(app, host="0.0.0.0", port=ARABIC_PORT)
