import os
import json
import subprocess
import numpy as np
import soundfile as sf

def convert_to_rhubarb_format(input_path):
    """Convert WAV to Rhubarb's preferred format: PCM16 mono 44.1kHz."""
    data, samplerate = sf.read(input_path)
    if len(data.shape) > 1:  # Stereo → Mono
        data = np.mean(data, axis=1)
    temp_path = input_path.replace(".wav", "_rhubarb.wav")
    sf.write(temp_path, data, 44100, subtype="PCM_16")
    return temp_path


def generate_mouth_cues(audio_path: str, transcript: str = None):
    rhubarb_path = "/home/genai/Downloads/Projects/Rhubarb-Lip-Sync-1.14.0-Linux/rhubarb"

    # Create temp JSON path
    json_path = audio_path.replace(".wav", "_mouthcues.json")

    # Build rhubarb command
    cmd = [
        rhubarb_path, audio_path,
        "-f", "json",
        "-o", json_path
    ]
    if transcript:
        cmd += ["-r", transcript]

    # Run rhubarb
    subprocess.run(cmd, check=True)

    # Read generated JSON
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return data.get("mouthCues", [])
    else:
        return []
