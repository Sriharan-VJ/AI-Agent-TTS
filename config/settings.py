import os
import torch
from starlette.config import Config

current_dir = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.abspath(os.path.join(current_dir, "../.env"))

config = Config(ENV_PATH)

####################################
# SERVER CONFIG
####################################

PORT = config("PORT", cast=int, default=8000)
HOST = config("HOST", cast=str, default="localhost")
ENVIRONMENT = config("ENVIRONMENT", cast=str, default="development")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################
# CORS
####################################

ALLOWED_ORIGIN = config("ALLOWED_ORIGIN", cast=str, default="*")

####################################
# MODEL
####################################

ASTEROID_MODEL_PATH = config("ASTEROID_MODEL_PATH", cast=str, default="")


ARABIC_MODEL_PATH = config("ARABIC_MODEL_PATH", cast=str, default="")
THAI_MODEL_PATH = config("THAI_MODEL_PATH", cast=str, default="")

ARABIC_PORT = int(os.getenv("ARABIC_PORT", 7777))
ARABIC_BASE_URL = config("ARABIC_BASE_URL", cast=str, default="")
AUDIO_FILE_PATH = config("AUDIO_FILE_PATH", cast=str, default="")

INDIC_PORT = int(os.getenv("INDIC_PORT",4545))
INDIC_MODEL_PATH = config("INDIC_MODEL_PATH", cast=str, default="")

BASE_URL = config("BASE_URL", cast=str, default="")

INDIC_BASE_URL = config("INDIC_BASE_URL", cast=str, default="")
KOKORO_BASE_URL = config("KOKORO_BASE_URL", cast=str, default="")
TRANSLATOR_PATH = config("TRANSLATOR_PATH", cast=str, default="")
GEMINI_API_KEY = config("GEMINI_API_KEY", cast=str, default="")