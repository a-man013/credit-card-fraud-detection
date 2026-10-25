"""
Basic tests for the FraudShield AI ML pipeline.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.synthetic import generate_synthetic_dataset
from ml import pipeline


@pytest.fixture(scope="module")
def sample_df():
    return generate_synthetic_dataset(n_samples=500, fraud_ratio=0.05, seed=1)


@pytest.fixture(scope="module")
def trained_state(sample_df):
    result = pipeline.train_models(
        sample_df,
        model_names=["Logistic Regression", "Random Forest"],
        test_size=0.2,
        use_smote=False,
    )
    return result


# ── Synthetic data tests ──────────────────────────────────────────────────────

def test_synthetic_shape(sample_df):
    assert sample_df.shape[0] == 500
    assert "Class" in sample_df.columns
    assert "Amount" in sample_df.columns
    assert "Time" in sample_df.columns


def test_synthetic_class_distribution(sample_df):
    fraud = (sample_df["Class"] == 1).sum()
    assert fraud > 0
    assert fraud < len(sample_df)


# ── Validation tests ──────────────────────────────────────────────────────────

def test_validate_good_df(sample_df):
    stats = pipeline.validate_dataframe(sample_df)
    assert stats["is_valid"] is True
    assert stats["rows"] == 500
    assert stats["columns"] > 0


def test_validate_empty_df():
    empty = pd.DataFrame()
    stats = pipeline.validate_dataframe(empty)
    assert stats["is_valid"] is False
    assert any("empty" in m.lower() for m in stats["validation_messages"])


def test_validate_missing_values(sample_df):
    df_missing = sample_df.copy()
    df_missing.loc[0, "Amount"] = np.nan
    stats = pipeline.validate_dataframe(df_missing)
    assert stats["missing_cells"] >= 1


# ── Training tests ────────────────────────────────────────────────────────────

def test_train_returns_metrics(trained_state):
    assert trained_state["status"] == "success"
    assert len(trained_state["models_trained"]) >= 1


def test_train_models_saved():
    assert "Logistic Regression" in pipeline.MODEL_REGISTRY or \
           "Random Forest" in pipeline.MODEL_REGISTRY


def test_evaluation_metrics():
    metrics = pipeline.TRAINED_METRICS
    assert len(metrics) >= 1
    for m in metrics:
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1_score"] <= 1.0


# ── Prediction tests ──────────────────────────────────────────────────────────

def test_predict_single(sample_df):
    feat_cols = pipeline.FEATURE_COLUMNS
    row = sample_df.drop(columns=["Class"]).iloc[0]
    features = {k: float(v) for k, v in row.items() if k in feat_cols}

    result = pipeline.predict_single(features)
    assert result["prediction"] in ["Genuine", "Potential Fraud"]
    assert result["predicted_class"] in [0, 1]


def test_predict_batch(sample_df):
    result = pipeline.predict_batch(sample_df.head(50))
    assert result["total"] == 50
    assert result["fraud_count"] + result["genuine_count"] == 50
    assert result["has_actual_labels"] is True


def test_predict_without_class_column(sample_df):
    df_no_class = sample_df.drop(columns=["Class"]).head(10)
    result = pipeline.predict_batch(df_no_class)
    assert result["has_actual_labels"] is False
    assert result["total"] == 10


# ── Insights tests ────────────────────────────────────────────────────────────

def test_insights_shape(sample_df):
    insights = pipeline.get_insights(sample_df)
    assert "amount_distribution" in insights
    assert "anomalies" in insights
    assert isinstance(insights["amount_distribution"], list)
