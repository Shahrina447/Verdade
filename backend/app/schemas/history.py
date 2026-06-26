from pydantic import BaseModel


class HistoryItem(BaseModel):
    id:           str
    text_preview: str
    prediction:   str
    confidence:   float
    timestamp:    str
