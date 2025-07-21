from kokoro import KPipeline
import soundfile as sf
import os
import uuid

pipeline_cache = {}
unique_id = str(uuid.uuid4())

BASE_URL = "http://172.18.14.131:8998"


def get_pipeline(lang_code: str):
    if lang_code not in pipeline_cache:
        pipeline_cache[lang_code] = KPipeline(lang_code=lang_code)
    return pipeline_cache[lang_code]


def generate_tts_audio(text: str, voice: str, lang_code: str, speed: float):
    pipeline = get_pipeline(lang_code)
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")

    audio_combined = []

    for i, item in enumerate(generator):
        audio = item[2]
        audio_combined.extend(audio)
    output_path = os.path.join("output", f"{lang_code} {unique_id}.wav")
    sf.write(output_path, audio_combined, 24000)

    download_url = f"{BASE_URL}/download/{lang_code}_{unique_id}.wav"

    return {"audio_url": download_url}
