from pydantic import BaseModel, Field
from app.config import BATCH_MAX, TEXT_MAX_LEN, TEXT_MIN_LEN


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=TEXT_MIN_LEN, max_length=TEXT_MAX_LEN)


class PredictResponse(BaseModel):
    prediction:      str
    confidence:      float
    confidence_real: float
    confidence_fake: float
    verdict_text:    str
    prediction_id:   str
    timestamp:       str


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=BATCH_MAX)


class BatchSummary(BaseModel):
    total:           int
    fake:            int
    real:            int
    fake_percentage: float


class BatchResponse(BaseModel):
    results: list[PredictResponse]
    summary: BatchSummary
