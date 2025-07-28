# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
# import torch

# # Load the model and tokenizer
# model_name = "facebook/m2m100_418M"
# tokenizer = M2M100Tokenizer.from_pretrained(model_name)
# model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Set source and target language
# src_text = "How are you?"
# src_lang = "en"
# tgt_lang = "ta"  # Tamil

# # Set tokenizer source language
# tokenizer.src_lang = src_lang

# # Encode input
# encoded = tokenizer(src_text, return_tensors="pt").to(device)

# # Force BOS token for Tamil
# tgt_lang_id = tokenizer.get_lang_id(tgt_lang)

# # Generate translation
# generated = model.generate(
#     **encoded,
#     forced_bos_token_id=tgt_lang_id,
#     max_length=100
# )

# # Decode and print
# translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
# print("Translated text:", translation)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

app = FastAPI()

# Load model and tokenizer once
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input schema
class TranslationRequest(BaseModel):
    text: str
    src_lang: str   # default English
    tgt_lang: str   # default Tamil

@app.post("/translate")
def translate_text(payload: TranslationRequest):
    try:
        tokenizer.src_lang = payload.src_lang
        encoded = tokenizer(payload.text, return_tensors="pt").to(device)
        tgt_lang_id = tokenizer.get_lang_id(payload.tgt_lang)

        generated = model.generate(
            **encoded,
            forced_bos_token_id=tgt_lang_id,
            max_length=100
        )

        translated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return {"translated_text": translated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn

    print(
        """
██╗███╗░░██╗██████╗░██╗░█████╗░  ████████╗██████╗░░█████╗░███╗░░██╗░██████╗
██║████╗░██║██╔══██╗██║██╔══██╗  ╚══██╔══╝██╔══██╗██╔══██╗████╗░██║██╔════╝
██║██╔██╗██║██║░░██║██║██║░░╚═╝  ░░░██║░░░██████╔╝███████║██╔██╗██║╚█████╗░
██║██║╚████║██║░░██║██║██║░░██╗  ░░░██║░░░██╔══██╗██╔══██║██║╚████║░╚═══██╗
██║██║░╚███║██████╔╝██║╚█████╔╝  ░░░██║░░░██║░░██║██║░░██║██║░╚███║██████╔╝
╚═╝╚═╝░░╚══╝╚═════╝░╚═╝░╚════╝░  ░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚══╝╚═════╝░"""
    )

    uvicorn.run(app, host="0.0.0.0", port=1221)
