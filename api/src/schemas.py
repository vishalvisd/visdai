from pydantic import BaseModel, Field
from typing import Optional, List

class FindAnswer(BaseModel):
    question: str
