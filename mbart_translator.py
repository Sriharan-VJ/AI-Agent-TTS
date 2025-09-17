# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# from constants.kokoro_meta import LANG_NAME_TO_CODE
# from fastapi import FastAPI, HTTPException
# from config.settings import TRANSLATOR_PATH
# from pydantic import BaseModel
# from config.cors_options import configure_cors



# import google.generativeai as genai, types

# genai.configure(api_key="AIzaSyAzrm7jo0TgS0koxqeo2A_3QfyGJuwD6qM")

# model = genai.GenerativeModel("gemini-2.5-flash-lite")



# # FastAPI setup
# app = FastAPI(title="Mbart Translator")
# configure_cors(app)


# # Request body
# class T2TRequest(BaseModel):
#     text: str
#     src_lang: str
#     target_lang: str


# @app.get("/")
# def root():
#     return {"status": "Mbart Translator is running"}



# async def transcribe_text_to_text(
#         text: str, source_lang: str, target_lang: str
# ):
#     """
#     Transcribes text from one language to another using the Google Gemini API.
#     This function is specifically designed to translate Arabic text to Tamil.
#     """
#     print(f"Transcribing text from {source_lang} to {target_lang}: {text}")
 
#     try:
#         client = genai.Client(api_key="AIzaSyAzrm7jo0TgS0koxqeo2A_3QfyGJuwD6qM")
#         model = "gemini-2.5-flash"  # Using the model specified in your previous example
 
#         generate_content_config = types.GenerateContentConfig(
#         response_mime_type="text/plain",
#         temperature=0,
#         system_instruction=types.Part.from_text(text=f"""
#     You are a translation engine.
#     Input will be in {source_lang}.
#     Output must be in {target_lang}.
#     Do not explain, do not answer questions, do not add any extra text — only provide the direct translation.
#     If the input is already in the target language, return it unchanged.
#     """),
#         thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
#     )
 
#         contents = [
#             types.Content(
#                 role="user",
#                 parts=[
#                     types.Part.from_text(text=text),  # Use the text directly
#                 ],
#             ),
#         ]
 
#         response = client.models.generate_content(
#             model=model,
#             contents=contents,
#             config=generate_content_config,
#         )
 
#         print(f"transcription: {response.text}")
#         print(f"Response token Type: {response.usage_metadata.total_token_count}")
 
#         return JSONResponse(content={"transcription": response.text})
#     except Exception as e:
#         # Catch any exceptions that occur during the API call or processing
#         raise HTTPException(
#             status_code=500,
#             detail=f"Gemini API Error: {str(e)}"
#         )








# @app.post("/")
# async def translate(request: T2TRequest):
#     if not request.text.strip():
#         raise HTTPException(status_code=400, detail="Input text is empty.")
#     try:
#            translated_text = transcribe_text_to_text(request.text, request.src_lang, request.target_lang)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Translation failed: {e}")
#     print("Before Translation:", request.text)
#     print("After Translation:", translated_text)
#     return {"transcription": translated_text}


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)





from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key="AIzaSyAzrm7jo0TgS0koxqeo2A_3QfyGJuwD6qM")

# Load model
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# FastAPI setup
app = FastAPI(title="Gemini Translator")

# Request body
class T2TRequest(BaseModel):
    text: str
    src_lang: str
    target_lang: str

@app.get("/")
def root():
    return {"status": "Gemini Translator is running"}

async def transcribe_text_to_text(text: str, source_lang: str, target_lang: str):
    """
    Translates text from one language to another using Google Gemini API.
    """
    print(f"Translating from {source_lang} to {target_lang}: {text}")

    try:
        prompt = f"""
        You are a translation engine.
        Input language: {source_lang}
        Output language: {target_lang}
        Only return the direct translation, no explanations.
        
        Text: {text}
        """

        response = model.generate_content(prompt)

        if not response.text:
            raise HTTPException(status_code=500, detail="Empty response from Gemini API")

        print(f"Translation: {response.text}")
        return response.text.strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

@app.post("/")
async def translate(request: T2TRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")
    translated_text = await transcribe_text_to_text(
        request.text, request.src_lang, request.target_lang
    )
    return JSONResponse(content={"transcription": translated_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
