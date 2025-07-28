from kokoro import KPipeline
import soundfile as sf
import os
import uuid

pipeline_cache = {}
unique_id = str(uuid.uuid4())

BASE_URL = "https://b8252b5121bc.ngrok-free.app"


def get_pipeline(lang_code: str):
    if lang_code not in pipeline_cache:
        pipeline_cache[lang_code] = KPipeline(lang_code=lang_code)
    return pipeline_cache[lang_code]


def generate_tts_audio(text: str, voice: str, lang_code: str, speed: float):
    pipeline = get_pipeline(lang_code)
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")

    audio_combined = []
    for _, _, audio in generator:
        audio_combined.extend(audio)

    filename = f"{lang_code}_{unique_id}.wav"
    output_path = os.path.join("output", filename)
    sf.write(output_path, audio_combined, 24000)

    download_url = f"{BASE_URL}/api/download/{filename}"
    return {"audio_url": download_url}
