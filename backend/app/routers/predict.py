import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.schemas.predict import (
    BatchPredictRequest,
    BatchResponse,
    BatchSummary,
    PredictRequest,
    PredictResponse,
)
from app.services import history as history_svc
from app.services import inference as inference_svc

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    result        = inference_svc.run_inference(req.text)
    prediction_id = str(uuid.uuid4())
    timestamp     = datetime.now(timezone.utc).isoformat()

    hist = history_svc.load()
    hist = history_svc.append(
        hist,
        prediction_id=prediction_id,
        text=req.text,
        prediction=result["prediction"],
        confidence=result["confidence"],
        timestamp=timestamp,
    )
    history_svc.save(hist)

    return PredictResponse(**result, prediction_id=prediction_id, timestamp=timestamp)


@router.post("/batch", response_model=BatchResponse)
def predict_batch(req: BatchPredictRequest) -> BatchResponse:
    results: list[PredictResponse] = []
    hist = history_svc.load()

    for text in req.texts:
        text = text.strip()
        if len(text) < 10:
            raise HTTPException(
                status_code=422,
                detail="Each text must be at least 10 characters.",
            )

        result        = inference_svc.run_inference(text)
        prediction_id = str(uuid.uuid4())
        timestamp     = datetime.now(timezone.utc).isoformat()

        hist = history_svc.append(
            hist,
            prediction_id=prediction_id,
            text=text,
            prediction=result["prediction"],
            confidence=result["confidence"],
            timestamp=timestamp,
        )
        results.append(
            PredictResponse(**result, prediction_id=prediction_id, timestamp=timestamp)
        )

    history_svc.save(hist)

    fake_count = sum(1 for r in results if r.prediction == "FAKE")
    real_count = len(results) - fake_count

    return BatchResponse(
        results=results,
        summary=BatchSummary(
            total=len(results),
            fake=fake_count,
            real=real_count,
            fake_percentage=round(fake_count / len(results) * 100, 1) if results else 0.0,
        ),
    )
