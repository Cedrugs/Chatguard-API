from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PredictOut(BaseModel):
    label: str
    probs: Dict[str, float]
    latency_ms: float


class BatchPredictIn(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BatchPredictOut(BaseModel):
    results: List[PredictOut]
    latency_ms: float
