"""
FraudShield AI — FastAPI Backend
=================================
All REST API endpoints for the fraud detection system.
"""

from __future__ import annotations

import io
import json
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import ml.pipeline as pipeline
from ml.synthetic import generate_synthetic_dataset
from schemas import (
    BatchPredictionResponse,
    DashboardResponse,
    DatasetStats,
    EvaluationResponse,
    HealthResponse,
    InsightsResponse,
    PredictionResult,
    TrainRequest,
    TrainResponse,
    TransactionFeatures,
)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FraudShield API",
    description="Credit Card Fraud Detection using Data Mining & Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Dataset persistence paths ──────────────────────────────────────────────────
SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"
DATASET_PATH = SAVED_MODELS_DIR / "active_dataset.pkl.gz"
DATASET_META_PATH = SAVED_MODELS_DIR / "active_dataset_meta.json"

# ── State ─────────────────────────────────────────────────────────────────────

CURRENT_DF: Optional[pd.DataFrame] = None
IS_SAMPLE_DATA: bool = True

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load persisted dataset or fallback to synthetic data on startup."""
    global CURRENT_DF, IS_SAMPLE_DATA
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists():
        try:
            CURRENT_DF = pd.read_pickle(DATASET_PATH, compression="gzip")
            IS_SAMPLE_DATA = False
            print(f"[FraudShield] Loaded persisted dataset: {len(CURRENT_DF)} rows")
            return
        except Exception as e:
            print(f"[FraudShield] Failed to load persisted dataset ({e}), falling back to synthetic.")

    CURRENT_DF = generate_synthetic_dataset(n_samples=2000, fraud_ratio=0.025)
    IS_SAMPLE_DATA = True


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    return {
        "status": "ok",
        "message": "FraudShield AI backend is running.",
        "model_loaded": len(pipeline.MODEL_REGISTRY) > 0,
        "dataset_loaded": CURRENT_DF is not None,
    }


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/api/dashboard", tags=["Dashboard"])
async def dashboard():
    global CURRENT_DF, IS_SAMPLE_DATA
    if CURRENT_DF is None:
        if DATASET_PATH.exists():
            try:
                CURRENT_DF = pd.read_pickle(DATASET_PATH, compression="gzip")
                IS_SAMPLE_DATA = False
            except Exception:
                CURRENT_DF = generate_synthetic_dataset()
                IS_SAMPLE_DATA = True
        else:
            CURRENT_DF = generate_synthetic_dataset()
            IS_SAMPLE_DATA = True
    try:
        data = pipeline.get_dashboard_data(CURRENT_DF, IS_SAMPLE_DATA)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dataset endpoints ─────────────────────────────────────────────────────────

@app.get("/api/dataset/status", tags=["Dataset"])
async def dataset_status():
    global CURRENT_DF, IS_SAMPLE_DATA
    if CURRENT_DF is None:
        return {"loaded": False, "is_sample_data": True}

    fraud_count = int(CURRENT_DF["Class"].sum()) if "Class" in CURRENT_DF.columns else 0
    total_rows = len(CURRENT_DF)

    meta = {}
    if DATASET_META_PATH.exists():
        try:
            meta = json.loads(DATASET_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "loaded": True,
        "is_sample_data": IS_SAMPLE_DATA,
        "filename": meta.get("filename", "Synthetic Demo Dataset" if IS_SAMPLE_DATA else "Custom Dataset"),
        "rows": total_rows,
        "columns": len(CURRENT_DF.columns),
        "fraud_count": fraud_count,
        "genuine_count": total_rows - fraud_count,
        "fraud_rate": round(fraud_count / max(total_rows, 1) * 100, 3),
        "upload_time": meta.get("upload_time"),
    }


@app.post("/api/dataset/reset", tags=["Dataset"])
async def reset_dataset():
    global CURRENT_DF, IS_SAMPLE_DATA
    if DATASET_PATH.exists():
        try:
            DATASET_PATH.unlink()
        except Exception:
            pass
    if DATASET_META_PATH.exists():
        try:
            DATASET_META_PATH.unlink()
        except Exception:
            pass
    CURRENT_DF = generate_synthetic_dataset(n_samples=2000, fraud_ratio=0.025)
    IS_SAMPLE_DATA = True
    return {"status": "ok", "message": "Reset to synthetic demo dataset."}


@app.post("/api/upload", tags=["Dataset"])
async def upload_dataset(file: UploadFile = File(...)):
    global CURRENT_DF, IS_SAMPLE_DATA

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    stats = pipeline.validate_dataframe(df)

    if stats["is_valid"]:
        CURRENT_DF = df
        IS_SAMPLE_DATA = False

        # Persist to disk so dataset survives restarts
        try:
            SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            df.to_pickle(DATASET_PATH, compression="gzip")
            meta = {
                "filename": file.filename,
                "rows": len(df),
                "columns": len(df.columns),
                "is_sample_data": False,
                "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            DATASET_META_PATH.write_text(json.dumps(meta), encoding="utf-8")
        except Exception as e:
            print(f"[FraudShield] Warning: Failed to persist dataset to disk: {e}")

        # Clear old models so user re-trains on new data
        pipeline.MODEL_REGISTRY.clear()
        pipeline.TRAINED_METRICS.clear()

    return JSONResponse(content=stats)


# ── Single transaction prediction ─────────────────────────────────────────────

@app.post("/api/predict", tags=["Prediction"])
async def predict_single(payload: TransactionFeatures):
    try:
        result = pipeline.predict_single(payload.features, payload.model_name)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ── Batch prediction ──────────────────────────────────────────────────────────

@app.post("/api/predict/batch", tags=["Prediction"])
async def predict_batch(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(default=None),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV is empty.")

    try:
        result = pipeline.predict_batch(df, model_name)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


# ── Train ─────────────────────────────────────────────────────────────────────

@app.post("/api/train", tags=["Training"])
async def train(payload: TrainRequest):
    global CURRENT_DF
    if CURRENT_DF is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Upload a dataset first.")
    if "Class" not in CURRENT_DF.columns:
        raise HTTPException(status_code=400, detail="Dataset must have a 'Class' column for training.")

    try:
        result = pipeline.train_models(
            CURRENT_DF,
            model_names=payload.model_names,
            test_size=payload.test_size,
            use_smote=payload.use_smote,
        )
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {e}\n{tb}")


# ── Evaluation ────────────────────────────────────────────────────────────────

@app.get("/api/evaluation", tags=["Evaluation"])
async def evaluation():
    if not pipeline.TRAINED_METRICS:
        raise HTTPException(
            status_code=404,
            detail="No evaluation results available. Train the models first via /api/train.",
        )
    best = max(pipeline.TRAINED_METRICS, key=lambda m: m["f1_score"])
    return JSONResponse(content={
        "models": pipeline.TRAINED_METRICS,
        "best_model": best["model_name"],
        "note": (
            "Metrics are computed on a held-out test set (stratified split). "
            "Precision and recall are emphasized because the dataset is highly imbalanced."
        ),
    })


# ── Insights ──────────────────────────────────────────────────────────────────

@app.get("/api/insights", tags=["Insights"])
async def insights():
    global CURRENT_DF, IS_SAMPLE_DATA
    if CURRENT_DF is None:
        CURRENT_DF = generate_synthetic_dataset()
        IS_SAMPLE_DATA = True
    try:
        data = pipeline.get_insights(CURRENT_DF)
        data["is_sample_data"] = IS_SAMPLE_DATA
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "name": "FraudShield AI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
