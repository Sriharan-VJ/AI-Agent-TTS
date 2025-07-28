import os
import uuid
import torch
import soundfile as sf

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from transformers import AutoModel, AutoTokenizer
from config.settings import AUDIO_FILE_PATH, INDIC_PORT, INDIC_MODEL_PATH, DEVICE, BASE_URL


app = FastAPI(title="Indic Text To Speech")

model = AutoModel.from_pretrained(INDIC_MODEL_PATH, trust_remote_code=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(INDIC_MODEL_PATH, trust_remote_code=True)


class TTSRequest(BaseModel):
    text: str
    speaker_id: int = 16
    style_id: int = 16

@app.get("/download/{filename}")
def download_audio(filename: str):
    file_path = os.path.join(AUDIO_FILE_PATH, filename)

    if not os.path.isfile(file_path):
        return {"error": "File not found"}

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )


@app.post("/tts")
def generate_tts(req: TTSRequest):
    inputs = tokenizer(text=req.text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(
            inputs["input_ids"], speaker_id=req.speaker_id, emotion_id=req.style_id
        )

    waveform = outputs.waveform.squeeze().detach().cpu().numpy()
    filename = f"{uuid.uuid4()}.wav"
    audio_path = os.path.join(AUDIO_FILE_PATH, filename)
    sf.write(audio_path, waveform, model.config.sampling_rate)

    download_url = f"{BASE_URL}:4545/download/{filename}"
    return {"message": "TTS audio generated", "download_url": download_url}


if __name__ == "__main__":
    print(
        """
██╗███╗░░██╗██████╗░██╗░█████╗░  ████████╗████████╗░██████╗
██║████╗░██║██╔══██╗██║██╔══██╗  ╚══██╔══╝╚══██╔══╝██╔════╝
██║██╔██╗██║██║░░██║██║██║░░╚═╝  ░░░██║░░░░░░██║░░░╚█████╗░
██║██║╚████║██║░░██║██║██║░░██╗  ░░░██║░░░░░░██║░░░░╚═══██╗
██║██║░╚███║██████╔╝██║╚█████╔╝  ░░░██║░░░░░░██║░░░██████╔╝
╚═╝╚═╝░░╚══╝╚═════╝░╚═╝░╚════╝░  ░░░╚═╝░░░░░░╚═╝░░░╚═════╝░"""
    )
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=INDIC_PORT)
