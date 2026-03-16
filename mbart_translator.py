import logging

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config.settings import TRANSLATOR_PATH
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
)

if not TRANSLATOR_PATH:
    raise RuntimeError("TRANSLATOR_PATH not set in .env")

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# FastAPI
# --------------------------------------------------
app = FastAPI(title="mBART Translator")

# --------------------------------------------------
# Request schema
# --------------------------------------------------
class T2TRequest(BaseModel):
    text: str
    src_lang: str
    target_lang: str

# --------------------------------------------------
# Language map (human → mBART)
# --------------------------------------------------
LANGUAGE_MAP = {
    # -------------------------
    # Core / Common
    # -------------------------
    "en": "en_XX",
    "english": "en_XX",

    "ar": "ar_AR",
    "arabic": "ar_AR",

    "cs": "cs_CZ",
    "czech": "cs_CZ",

    "de": "de_DE",
    "german": "de_DE",

    "es": "es_XX",
    "spanish": "es_XX",

    "et": "et_EE",
    "estonian": "et_EE",

    "fi": "fi_FI",
    "finnish": "fi_FI",

    "fr": "fr_XX",
    "french": "fr_XX",

    "it": "it_IT",
    "italian": "it_IT",

    "ja": "ja_XX",
    "japanese": "ja_XX",

    "ko": "ko_KR",
    "korean": "ko_KR",

    "nl": "nl_XX",
    "dutch": "nl_XX",

    "pt": "pt_XX",
    "portuguese": "pt_XX",

    "ru": "ru_RU",
    "russian": "ru_RU",

    "tr": "tr_TR",
    "turkish": "tr_TR",

    "vi": "vi_VN",
    "vietnamese": "vi_VN",

    "zh": "zh_CN",
    "chinese": "zh_CN",

    # -------------------------
    # Indian Languages
    # -------------------------
    "gu": "gu_IN",
    "gujarati": "gu_IN",

    "hi": "hi_IN",
    "hindi": "hi_IN",

    "bn": "bn_IN",
    "bengali": "bn_IN",

    "ml": "ml_IN",
    "malayalam": "ml_IN",

    "mr": "mr_IN",
    "marathi": "mr_IN",

    "ta": "ta_IN",
    "tamil": "ta_IN",

    "te": "te_IN",
    "telugu": "te_IN",

    "ur": "ur_PK",
    "urdu": "ur_PK",

    "ne": "ne_NP",
    "nepali": "ne_NP",

    "si": "si_LK",
    "sinhala": "si_LK",

    # -------------------------
    # European Languages
    # -------------------------
    "lt": "lt_LT",
    "lithuanian": "lt_LT",

    "lv": "lv_LV",
    "latvian": "lv_LV",

    "ro": "ro_RO",
    "romanian": "ro_RO",

    "hr": "hr_HR",
    "croatian": "hr_HR",

    "mk": "mk_MK",
    "macedonian": "mk_MK",

    "pl": "pl_PL",
    "polish": "pl_PL",

    "sv": "sv_SE",
    "swedish": "sv_SE",

    "uk": "uk_UA",
    "ukrainian": "uk_UA",

    "sl": "sl_SI",
    "slovene": "sl_SI",

    "gl": "gl_ES",
    "galician": "gl_ES",

    # -------------------------
    # Middle East / Central Asia
    # -------------------------
    "fa": "fa_IR",
    "persian": "fa_IR",

    "he": "he_IL",
    "hebrew": "he_IL",

    "ps": "ps_AF",
    "pashto": "ps_AF",

    "kk": "kk_KZ",
    "kazakh": "kk_KZ",

    "az": "az_AZ",
    "azerbaijani": "az_AZ",

    "ka": "ka_GE",
    "georgian": "ka_GE",

    # -------------------------
    # Southeast / East Asia
    # -------------------------
    "my": "my_MM",
    "burmese": "my_MM",

    "km": "km_KH",
    "khmer": "km_KH",

    "mn": "mn_MN",
    "mongolian": "mn_MN",

    "th": "th_TH",
    "thai": "th_TH",

    "tl": "tl_XX",
    "tagalog": "tl_XX",

    "id": "id_ID",
    "indonesian": "id_ID",

    # -------------------------
    # African Languages
    # -------------------------
    "af": "af_ZA",
    "afrikaans": "af_ZA",

    "sw": "sw_KE",
    "swahili": "sw_KE",

    "xh": "xh_ZA",
    "xhosa": "xh_ZA",
}


def normalize_lang(lang: str) -> str:
    lang = lang.strip().lower()
    if lang not in LANGUAGE_MAP:
        raise ValueError(f"Unsupported language: {lang}")
    return LANGUAGE_MAP[lang]

# --------------------------------------------------
# Load model ONCE
# --------------------------------------------------
logger.info("Loading mBART model from local path...")

tokenizer = MBart50TokenizerFast.from_pretrained(TRANSLATOR_PATH)
model = MBartForConditionalGeneration.from_pretrained(
    TRANSLATOR_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.eval()

logger.info("mBART model loaded successfully")

# --------------------------------------------------
# Warmup (IMPORTANT)
# --------------------------------------------------
logger.info("Warming up model...")
with torch.inference_mode():
    tokenizer.src_lang = "en_XX"
    inputs = tokenizer("hello", return_tensors="pt").to(model.device)
    _ = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["ta_IN"],
        max_new_tokens=8,
    )
logger.info("Warmup complete")

# --------------------------------------------------
# Translation function
# --------------------------------------------------
def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    src_lang_code = normalize_lang(src_lang)
    tgt_lang_code = normalize_lang(tgt_lang)

    tokenizer.src_lang = src_lang_code
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.inference_mode():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code],
            max_new_tokens=128,
        )

    return tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0]

# --------------------------------------------------
# API endpoint
# --------------------------------------------------
@app.post("/")
def translate(request: T2TRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    try:
        print("Before Translation:", request.text)
        translated_text = translate_text(
            request.text, request.src_lang, request.target_lang
        )
        print("After Translation:", translated_text)
        return {"transcription": translated_text}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Translation failed")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
