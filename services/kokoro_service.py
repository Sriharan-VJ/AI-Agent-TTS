import os
import re
import uuid
import soundfile as sf
from kokoro import KPipeline
from services.lipsynk import generate_mouth_cues

pipeline_cache = {}
unique_id = str(uuid.uuid4())



def get_pipeline(lang_code: str):
    if lang_code not in pipeline_cache:
        pipeline_cache[lang_code] = KPipeline(lang_code=lang_code)
    return pipeline_cache[lang_code]


def split_text_only(input_text):
    cleaned_text = re.sub(r'<[^>]+>.*?</[^>]+>', '', input_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    return [f" {line}" for i, line in enumerate(lines)]



def generate_tts_audio(text: str, voice: str, lang_code: str, speed: float):
    pipeline = get_pipeline(lang_code)  # Assuming you have this
    mouth_cues =[]
    print("Text:", text)
    text = split_text_only(text)
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")

    print("Split Text:",text)
    audio_combined = []
    for _, _, audio in generator:
        audio_combined.extend(audio)

    os.makedirs("output", exist_ok=True)

    unique_id = str(uuid.uuid4())
    filename = f"{lang_code}_{unique_id}.wav"
    output_path = os.path.join("output", filename)

    # Save original TTS (24kHz float WAV)
    sf.write(output_path, audio_combined, 24000)

    print("Generate MouthCues with Rhubarb")
    # mouth_cues = generate_mouth_cues(output_path)
    print("mouth cues:",mouth_cues)

    download_url = f"/download/{filename}"
    return {
        "audio_url": download_url,
        "mouth_cues": mouth_cues
    }
