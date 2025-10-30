import time
from fastapi import APIRouter
from ..core.model import model_bundle
from ..core.config import settings
from ..schemas.predict import (
    PredictIn, PredictOut, BatchPredictIn, BatchPredictOut
)

router = APIRouter()


@router.get("/")
def root():
    return {
        "name": settings.title,
        "version": settings.version,
        "model_id": settings.model_id,
        "device": model_bundle.device_str,
        "labels": model_bundle.id2label,
    }


@router.get("/health")
def healthz():
    return {"status": "ok", "model_loaded": model_bundle.pipe is not None, "device": model_bundle.device_str}


@router.post("/v1/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    t0 = time.perf_counter()
    out = model_bundle.predict_one(inp.text, inp.threshold)
    t1 = time.perf_counter()
    return PredictOut(**out, latency_ms=(t1 - t0) * 1000.0)


@router.post("/v1/batch_predict", response_model=BatchPredictOut)
def batch_predict(inp: BatchPredictIn):
    t0 = time.perf_counter()
    outs = model_bundle.predict_batch(inp.texts, inp.threshold)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    per = total_ms / max(1, len(outs))
    results = [PredictOut(**o, latency_ms=per) for o in outs]
    return BatchPredictOut(results=results, latency_ms=total_ms)
