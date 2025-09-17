from pydantic import BaseModel
from typing import Optional

class TTSRequest(BaseModel):
    text: str
    voice: str 
    lang_code: str 
    speed: Optional[float] = 1.0
