from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    lang_code: str = "a"
    speed: float = 1.0
