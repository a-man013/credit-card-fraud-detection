"""
Pydantic schemas for FraudShield AI API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    dataset_loaded: bool


# ── Dashboard ─────────────────────────────────────────────────────────────────

class KPICards(BaseModel):
    total_transactions: int
    genuine_count: int
    fraud_count: int
    fraud_rate: float  # percentage


class TimeSeriesPoint(BaseModel):
    date: str
    genuine: int
    fraud: int


class RecentTransaction(BaseModel):
    id: int
    amount: float
    prediction: str   # "Genuine" | "Fraud"
    probability: Optional[float]
    timestamp: str


class FraudAlert(BaseModel):
    id: int
    amount: float
    probability: float
    timestamp: str
    severity: str  # "High" | "Medium" | "Low"


class SystemStatus(BaseModel):
    model_trained: bool
    dataset_loaded: bool
    last_trained: Optional[str]
    records_processed: int
    active_model: Optional[str]


class DashboardResponse(BaseModel):
    kpi: KPICards
    time_series: List[TimeSeriesPoint]
    recent_transactions: List[RecentTransaction]
    fraud_alerts: List[FraudAlert]
    system_status: SystemStatus
    is_sample_data: bool


# ── Dataset Upload ────────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_pct: float


class DatasetStats(BaseModel):
    rows: int
    columns: int
    missing_cells: int
    missing_pct: float
    duplicate_rows: int
    column_info: List[ColumnInfo]
    class_distribution: Optional[Dict[str, int]]
    preview: List[Dict[str, Any]]
    validation_messages: List[str]
    warnings: List[str]
    is_valid: bool


# ── Predict ───────────────────────────────────────────────────────────────────

class TransactionFeatures(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        example={
            "Amount": 149.62,
            "V1": -1.359807134,
            "V2": -0.072781173,
        },
    )
    model_name: Optional[str] = "Random Forest"


class PredictionResult(BaseModel):
    prediction: str           # "Genuine" | "Potential Fraud"
    predicted_class: int      # 0 or 1
    probability_fraud: Optional[float]
    confidence: Optional[str]
    model_used: str
    features_used: List[str]


class BatchPredictionRow(BaseModel):
    row_index: int
    features: Dict[str, Any]
    prediction: str
    predicted_class: int
    probability_fraud: Optional[float]
    actual_class: Optional[int]
    is_correct: Optional[bool]


class BatchPredictionResponse(BaseModel):
    total: int
    fraud_count: int
    genuine_count: int
    fraud_rate: float
    model_used: str
    predictions: List[BatchPredictionRow]
    accuracy: Optional[float]   # only when actual labels available
    has_actual_labels: bool


# ── Train ─────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model_names: Optional[List[str]] = ["Logistic Regression", "Decision Tree", "Random Forest"]
    test_size: float = 0.2
    use_smote: bool = True


class TrainResponse(BaseModel):
    status: str
    message: str
    models_trained: List[str]
    training_samples: int
    test_samples: int
    elapsed_seconds: float


# ── Evaluation ────────────────────────────────────────────────────────────────

class ConfusionMatrixData(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int
    labels: List[str]


class ROCPoint(BaseModel):
    fpr: float
    tpr: float


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    confusion_matrix: ConfusionMatrixData
    roc_curve: Optional[List[ROCPoint]]


class EvaluationResponse(BaseModel):
    models: List[ModelMetrics]
    best_model: str
    note: str


# ── Insights ──────────────────────────────────────────────────────────────────

class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    genuine_count: int
    fraud_count: int


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class AnomalyRecord(BaseModel):
    row_index: int
    amount: float
    anomaly_score: float
    is_fraud: Optional[bool]


class InsightsResponse(BaseModel):
    amount_distribution: List[HistogramBin]
    feature_importances: List[FeatureImportance]
    anomalies: List[AnomalyRecord]
    stats: Dict[str, Any]
    is_sample_data: bool
