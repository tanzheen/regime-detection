from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .regime_pipeline import (
    DEFAULT_MODEL_PATH,
    SlicedWassersteinRegimePipeline,
)


app = FastAPI(
    title="Regime Detection API",
    description=(
        "FastAPI service for the log_ret + vol_zscore sliced Wasserstein "
        "regime model."
    ),
    version="0.1.0",
)


class PriceBar(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: str = Field(..., examples=["2026-05-22"])
    adjClose: float = Field(..., examples=[745.64])
    adjVolume: float = Field(..., examples=[41762006])


class PredictRequest(BaseModel):
    records: list[PriceBar]
    include_features: bool = False


@lru_cache(maxsize=1)
def get_pipeline() -> SlicedWassersteinRegimePipeline:
    return SlicedWassersteinRegimePipeline(DEFAULT_MODEL_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "model": pipeline.metadata.__dict__,
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    pipeline = get_pipeline()
    return pipeline.metadata.__dict__


@app.post("/predict/latest")
def predict_latest(request: PredictRequest) -> dict[str, Any]:
    try:
        pipeline = get_pipeline()
        frame = pd.DataFrame.from_records([_dump_model(row) for row in request.records])
        result = pipeline.predict_latest(
            frame,
            include_features=request.include_features,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"result": result}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        pipeline = get_pipeline()
        records = [_dump_model(row) for row in request.records]
        result = pipeline.predict_records(
            records,
            include_features=request.include_features,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"result": result}


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
