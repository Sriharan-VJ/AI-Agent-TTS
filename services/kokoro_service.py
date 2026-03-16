# import os
# import re
# import uuid
# import soundfile as sf
# from kokoro import KPipeline
# from services.lipsynk import generate_mouth_cues

# pipeline_cache = {}
# unique_id = str(uuid.uuid4())



# def get_pipeline(lang_code: str):
#     if lang_code not in pipeline_cache:
#         pipeline_cache[lang_code] = KPipeline(lang_code=lang_code, device="cuda", trf=False)
#     return pipeline_cache[lang_code]


# import re

# def split_text_only(input_text):
#     # Remove HTML tags
#     cleaned_text = re.sub(r'<[^>]+>.*?</[^>]+>', '', input_text, flags=re.DOTALL)
#     # Remove URLs
#     cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)

#     # Split only numbers >= 10 digits into spaced digits
#     def split_long_numbers(match):
#         number = match.group()
#         if len(number) >= 10:  # mobile or long IDs
#             return " ".join(number)
#         return number  # leave shorter numbers as-is

#     cleaned_text = re.sub(r'\d+', split_long_numbers, cleaned_text)

#     # Collapse multiple spaces to a single space
#     cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

#     # Split by lines and strip whitespace
#     lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
#     return [f" {line}" for line in lines]





# def generate_tts_audio(text: str, voice: str, lang_code: str, speed: float):
#     print("Function Start")
#     pipeline = get_pipeline(lang_code)  # Assuming you have this
#     mouth_cues =[]
#     print("22222222222222222222")
#     print("Original Text:", text)
#     text = split_text_only(text)
#     print("Split Text:",text)
#     generator = pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
#     print("3333333333333333")
#     audio_combined = []
#     for _, _, audio in generator:
#         audio_combined.extend(audio)
#     print("44444444444444444444")
#     os.makedirs("output", exist_ok=True)
#     print("555555555555555555")
#     unique_id = str(uuid.uuid4())
#     filename = f"{lang_code}_{unique_id}.wav"
#     output_path = os.path.join("output", filename)
#     print("666666666666666")
#     # Save original TTS (24kHz float WAV)
#     sf.write(output_path, audio_combined, 24000)
#     print("777777777777777")
#     print("Generate MouthCues with Rhubarb")
#     # mouth_cues = generate_mouth_cues(output_path)
#     print("Function End")
#     download_url = f"/download/{filename}"
#     return {
#         "audio_url": download_url,
#         "mouth_cues": mouth_cues
# #     }
# import os
# import re
# import uuid
# import numpy as np
# import soundfile as sf
# import torch
# from kokoro import KPipeline
# from services.lipsynk import generate_mouth_cues

# # ------------------------------------------------------------
# # 🔥 GLOBAL OPTIMIZATIONS FOR 4090
# # ------------------------------------------------------------

# torch.set_default_device("cuda")          # force GPU everywhere
# torch.set_grad_enabled(False)             # disable autograd for speed

# pipeline_cache = {}
# voice_cache = {}                          # cache voices on GPU


# # ------------------------------------------------------------
# # ⚡ FAST & GPU-FORCED PIPELINE LOADING
# # ------------------------------------------------------------

# def get_pipeline(lang_code: str):
#     if lang_code not in pipeline_cache:
#         print(f"Loading Kokoro pipeline for: {lang_code}")
#         pipeline_cache[lang_code] = KPipeline(
#             lang_code=lang_code,
#             device="cuda",               # force GPU
#             trf=False                    # fastest G2P mode
#         )
#     return pipeline_cache[lang_code]


# # ------------------------------------------------------------
# # 🧼 TEXT CLEANING
# # ------------------------------------------------------------

# def split_text_only(input_text):
#     cleaned_text = re.sub(r'<[^>]+>.*?</[^>]+>', '', input_text, flags=re.DOTALL)
#     cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)
#     lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
#     return [f" {line}" for i, line in enumerate(lines)]


# # ------------------------------------------------------------
# # 🎤 CORE TTS GENERATOR (GPU-OPTIMIZED)
# # ------------------------------------------------------------

# def generate_tts_audio(text: str, voice: str, lang_code: str, speed: float):
#     pipeline = get_pipeline(lang_code)

#     print("Text:", text)
#     text = split_text_only(text)

#     # ----------------------------------------------------------
#     # ⚡ PRELOAD & CACHE VOICE ON GPU (HUGE SPEED BOOST)
#     # ----------------------------------------------------------
#     if voice not in voice_cache:
#         print(f"Loading voice (GPU-cache): {voice}")
#         v = pipeline.load_voice(voice)            # load CPU
#         v = v.to("cuda").half()                   # move to GPU + FP16
#         voice_cache[voice] = v
#     else:
#         v = voice_cache[voice]

#     # ----------------------------------------------------------
#     # ⚡ RUN PIPELINE (NO CPU MOVES, NO COPYING)
#     # ----------------------------------------------------------
#     generator = pipeline(
#         text,
#         voice=voice,              # pass preloaded GPU voice
#         speed=speed,
#         split_pattern=r"\n+"
#     )

#     print("Split Text:", text)

#     # ----------------------------------------------------------
#     # ⚡ FAST CONCAT (NumPy, not Python list.extend)
#     # ----------------------------------------------------------
#     audio_list = []
#     for _, _, audio in generator:
#         audio_list.append(audio.squeeze().cpu().numpy())  # CPU copy at final stage only

#     if audio_list:
#         audio_combined = np.concatenate(audio_list)
#     else:
#         audio_combined = np.zeros(1, dtype=np.float32)

#     # ----------------------------------------------------------
#     # 💾 SAVE OUTPUT
#     # ----------------------------------------------------------
#     os.makedirs("output", exist_ok=True)
#     unique_id = str(uuid.uuid4())
#     filename = f"{lang_code}_{unique_id}.wav"
#     output_path = os.path.join("output", filename)

#     sf.write(output_path, audio_combined, 24000)

#     print("Generate MouthCues with Rhubarb")
#     mouth_cues = []  # generate_mouth_cues(output_path)

#     print("mouth cues:", mouth_cues)

#     download_url = f"/download/{filename}"
#     return {
#         "audio_url": download_url,
#         "mouth_cues": mouth_cues
#     }


import os
import re
import uuid
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from kokoro import KPipeline
from services.lipsynk import generate_mouth_cues

torch.set_default_device("cuda")
torch.set_grad_enabled(False)

pipeline_cache = {}
voice_cache = {}

def get_pipeline(lang_code: str):
    if lang_code not in pipeline_cache:
        print(f"Loading Kokoro pipeline for: {lang_code}")
        pipeline_cache[lang_code] = KPipeline(
            lang_code=lang_code,
            device="cuda",
            trf=False
        )
    return pipeline_cache[lang_code]

def split_text_only(input_text):
    cleaned_text = re.sub(r'<[^>]+>.*?</[^>]+>', '', input_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    return [f" {line}" for i, line in enumerate(lines)]

def generate_tts_audio(text: str, voice: str, lang_code: str, speed: float):
    pipeline = get_pipeline(lang_code)

    print("Text:", text)
    text = split_text_only(text)

    if voice not in voice_cache:
        print(f"Loading voice (GPU-cache): {voice}")
        v = pipeline.load_voice(voice)
        v = v.to("cuda").half()
        voice_cache[voice] = v
    else:
        v = voice_cache[voice]

    generator = pipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=r"\n+"
    )

    print("Split Text:", text)

    audio_list = []
    for _, _, audio in generator:
        audio_list.append(audio.squeeze().cpu().numpy())

    if audio_list:
        audio_combined = np.concatenate(audio_list)
    else:
        audio_combined = np.zeros(1, dtype=np.float32)

    os.makedirs("output", exist_ok=True)
    unique_id = str(uuid.uuid4())

    # Temporary WAV
    temp_wav_path = f"output/{lang_code}_{unique_id}.temp.wav"
    sf.write(temp_wav_path, audio_combined, 24000)

    # Final MP3
    final_mp3 = f"output/{lang_code}_{unique_id}.mp3"
    audio = AudioSegment.from_wav(temp_wav_path)
    audio.export(final_mp3, format="mp3", bitrate="128k")

    os.remove(temp_wav_path)

    print("Generate MouthCues with Rhubarb")

    download_url = f"/download/{os.path.basename(final_mp3)}"
    return {
        "audio_url": download_url,
    }
