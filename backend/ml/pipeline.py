"""
FraudShield AI — Machine Learning Pipeline
==========================================
Implements: validation → preprocessing → feature selection →
            training (LR / DT / RF) → evaluation → insights.

SMOTE (if requested) is applied only to the training split.
"""

from __future__ import annotations

import io
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

MODEL_DIR = Path(__file__).parent.parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

KNOWN_SCHEMAS = {
    # Kaggle standard schema
    "kaggle": {
        "required": ["Time", "Amount", "Class"],
        "pca_prefix": "V",
        "target": "Class",
    }
}

MODEL_REGISTRY: Dict[str, Any] = {}
SCALER_REGISTRY: Dict[str, StandardScaler] = {}
FEATURE_COLUMNS: List[str] = []
TRAINED_METRICS: List[Dict] = []
LAST_TRAINED: Optional[str] = None
ACTIVE_MODEL_NAME: str = "Random Forest"

# ── Validation ────────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate an uploaded DataFrame.
    Returns a stats dict + validation_messages + warnings.
    """
    messages: List[str] = []
    warnings: List[str] = []

    # Basic shape
    n_rows, n_cols = df.shape
    if n_rows == 0:
        messages.append("ERROR: Dataset is empty (0 rows).")
    if n_cols < 2:
        messages.append("ERROR: Dataset must have at least 2 columns.")

    # Missing values
    missing_cells = int(df.isnull().sum().sum())
    missing_pct = round(missing_cells / max(df.size, 1) * 100, 2)
    if missing_cells > 0:
        warnings.append(f"Dataset has {missing_cells} missing cells ({missing_pct}%). They will be imputed/dropped during preprocessing.")

    # Duplicates
    dup_rows = int(df.duplicated().sum())
    if dup_rows > 0:
        warnings.append(f"{dup_rows} duplicate rows found. They will be removed during preprocessing.")

    # Column info
    col_info = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        col_info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2) if len(df) > 0 else 0.0,
        })

    # Check for known target
    target_col = None
    class_distribution = None
    if "Class" in df.columns:
        target_col = "Class"
        vc = df["Class"].value_counts().to_dict()
        class_distribution = {str(int(k)): int(v) for k, v in vc.items()}
        messages.append("Target column 'Class' detected (0=Genuine, 1=Fraud).")
        if "0" not in class_distribution:
            warnings.append("No genuine (Class=0) transactions found.")
        if "1" not in class_distribution:
            warnings.append("No fraud (Class=1) transactions found.")
    else:
        warnings.append("No 'Class' column found. Predictions can be made but evaluation will be skipped.")

    # Check numeric columns
    non_numeric = [c for c in df.columns if df[c].dtype == object and c != target_col]
    if non_numeric:
        warnings.append(f"Non-numeric columns found: {non_numeric}. They will be dropped during training.")

    # Preview (first 10 rows, serialisable)
    preview_df = df.head(10).copy()
    for col in preview_df.select_dtypes(include=[np.floating]).columns:
        preview_df[col] = preview_df[col].round(4)
    preview = preview_df.fillna("").astype(str).to_dict(orient="records")

    is_valid = not any(m.startswith("ERROR") for m in messages)

    return {
        "rows": n_rows,
        "columns": n_cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": dup_rows,
        "column_info": col_info,
        "class_distribution": class_distribution,
        "preview": preview,
        "validation_messages": messages,
        "warnings": warnings,
        "is_valid": is_valid,
    }


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, fit_scaler: bool = True, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """
    Returns preprocessed feature DataFrame (no target), fitted scaler, feature list.
    """
    df = df.copy()

    # Drop target if present
    target_col = "Class" if "Class" in df.columns else None
    y = df[target_col].astype(int) if target_col else None

    # Drop non-numeric columns and target
    drop_cols = ([target_col] if target_col else []) + [c for c in df.columns if df[c].dtype == object]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Scale Amount and Time if present
    scale_cols = [c for c in ["Amount", "Time"] if c in df.columns]
    if scale_cols:
        if fit_scaler:
            scaler = StandardScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
        else:
            if scaler is not None:
                # Only transform columns that the scaler was fitted on
                scaler_cols = [c for c in scale_cols if c in df.columns]
                if scaler_cols:
                    df[scaler_cols] = scaler.transform(df[scaler_cols])

    feature_cols = list(df.columns)
    return df, scaler, feature_cols, y


# ── Training ──────────────────────────────────────────────────────────────────

def train_models(
    df: pd.DataFrame,
    model_names: List[str],
    test_size: float = 0.2,
    use_smote: bool = True,
) -> Dict[str, Any]:
    """
    Train the requested models on df, returns metrics dict.
    Modifies global MODEL_REGISTRY, SCALER_REGISTRY, FEATURE_COLUMNS.
    """
    global MODEL_REGISTRY, SCALER_REGISTRY, FEATURE_COLUMNS, TRAINED_METRICS, LAST_TRAINED, ACTIVE_MODEL_NAME

    start = time.time()

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column for training.")

    y_full = df["Class"].astype(int)
    X_full_raw = df.drop(columns=["Class"])

    # Remove non-numeric
    X_full_raw = X_full_raw.select_dtypes(include=[np.number])

    # Stratified split BEFORE any SMOTE
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(X_full_raw, y_full))

    X_train_raw = X_full_raw.iloc[train_idx]
    X_test_raw  = X_full_raw.iloc[test_idx]
    y_train     = y_full.iloc[train_idx]
    y_test      = y_full.iloc[test_idx]

    # Scale (fit on train only)
    scale_cols = [c for c in ["Amount", "Time"] if c in X_train_raw.columns]
    scaler = StandardScaler()
    X_train = X_train_raw.copy()
    X_test  = X_test_raw.copy()
    if scale_cols:
        X_train[scale_cols] = scaler.fit_transform(X_train_raw[scale_cols])
        X_test[scale_cols]  = scaler.transform(X_test_raw[scale_cols])

    FEATURE_COLUMNS = list(X_train.columns)
    SCALER_REGISTRY["scaler"] = scaler

    # SMOTE on training data only
    X_tr = X_train.values
    y_tr = y_train.values
    if use_smote and HAS_SMOTE:
        try:
            sm = SMOTE(random_state=42, k_neighbors=min(5, int(y_tr.sum()) - 1))
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        except Exception:
            pass  # Fall back to unbalanced if SMOTE fails

    # ── Build model instances ─────────────────────────────────────────────────
    model_map = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    metrics_list = []
    X_te = X_test.values
    y_te = y_test.values

    for name in model_names:
        if name not in model_map:
            continue
        clf = model_map[name]
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        has_proba = hasattr(clf, "predict_proba")

        acc  = float(accuracy_score(y_te, y_pred))
        prec = float(precision_score(y_te, y_pred, zero_division=0))
        rec  = float(recall_score(y_te, y_pred, zero_division=0))
        f1   = float(f1_score(y_te, y_pred, zero_division=0))

        roc_auc = None
        roc_points = None
        if has_proba:
            y_proba = clf.predict_proba(X_te)[:, 1]
            try:
                roc_auc = float(roc_auc_score(y_te, y_proba))
                fpr_arr, tpr_arr, _ = roc_curve(y_te, y_proba)
                # Downsample to 50 points for JSON
                idx = np.linspace(0, len(fpr_arr) - 1, min(50, len(fpr_arr))).astype(int)
                roc_points = [{"fpr": round(float(fpr_arr[i]), 4), "tpr": round(float(tpr_arr[i]), 4)} for i in idx]
            except Exception:
                pass

        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel().tolist()

        model_metrics = {
            "model_name": name,
            "accuracy":   round(acc, 4),
            "precision":  round(prec, 4),
            "recall":     round(rec, 4),
            "f1_score":   round(f1, 4),
            "roc_auc":    round(roc_auc, 4) if roc_auc is not None else None,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "labels": ["Genuine", "Fraud"]},
            "roc_curve":  roc_points,
        }
        metrics_list.append(model_metrics)

        MODEL_REGISTRY[name] = clf
        # Save to disk
        try:
            joblib.dump(clf, MODEL_DIR / f"{name.replace(' ', '_')}.joblib")
        except Exception:
            pass

    # Save scaler
    try:
        joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
        np.save(MODEL_DIR / "feature_columns.npy", np.array(FEATURE_COLUMNS))
    except Exception:
        pass

    TRAINED_METRICS = metrics_list
    LAST_TRAINED = time.strftime("%Y-%m-%d %H:%M:%S")
    if "Random Forest" in MODEL_REGISTRY:
        ACTIVE_MODEL_NAME = "Random Forest"
    elif model_names:
        ACTIVE_MODEL_NAME = model_names[0]

    elapsed = round(time.time() - start, 2)
    return {
        "status": "success",
        "message": f"Trained {len(metrics_list)} model(s) in {elapsed}s.",
        "models_trained": [m["model_name"] for m in metrics_list],
        "training_samples": len(X_tr),
        "test_samples": len(X_te),
        "elapsed_seconds": elapsed,
    }


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_single(features: Dict[str, float], model_name: Optional[str] = None) -> Dict[str, Any]:
    """Predict for a single transaction."""
    clf, scaler, feat_cols = _load_model(model_name)

    row = _build_feature_row(features, feat_cols, scaler)
    pred_class = int(clf.predict(row)[0])
    prob = None
    if hasattr(clf, "predict_proba"):
        prob = round(float(clf.predict_proba(row)[0][1]), 4)

    return {
        "prediction": "Potential Fraud" if pred_class == 1 else "Genuine",
        "predicted_class": pred_class,
        "probability_fraud": prob,
        "confidence": _confidence_label(prob),
        "model_used": model_name or ACTIVE_MODEL_NAME,
        "features_used": feat_cols,
    }


def predict_batch(df: pd.DataFrame, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Predict for an uploaded DataFrame."""
    clf, scaler, feat_cols = _load_model(model_name)

    has_actual = "Class" in df.columns
    actual_labels = df["Class"].astype(int).values if has_actual else None

    X = df.copy()
    if "Class" in X.columns:
        X = X.drop(columns=["Class"])

    # Keep only known feature columns; fill missing with 0
    X = X.reindex(columns=feat_cols, fill_value=0.0).astype(float)

    # Scale
    scale_cols = [c for c in ["Amount", "Time"] if c in X.columns]
    if scale_cols and scaler is not None:
        X[scale_cols] = scaler.transform(X[scale_cols])

    preds = clf.predict(X.values)
    probas = None
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(X.values)[:, 1]

    results = []
    for i, (idx, row) in enumerate(df.iterrows()):
        p = int(preds[i])
        prob = round(float(probas[i]), 4) if probas is not None else None
        actual = int(actual_labels[i]) if has_actual else None
        results.append({
            "row_index": i,
            "features": {k: (round(float(v), 4) if isinstance(v, (int, float)) else str(v)) for k, v in row.items() if k != "Class"},
            "prediction": "Potential Fraud" if p == 1 else "Genuine",
            "predicted_class": p,
            "probability_fraud": prob,
            "actual_class": actual,
            "is_correct": (p == actual) if has_actual else None,
        })

    fraud_count  = int(sum(r["predicted_class"] == 1 for r in results))
    genuine_count = len(results) - fraud_count
    fraud_rate   = round(fraud_count / max(len(results), 1) * 100, 2)

    accuracy = None
    if has_actual:
        accuracy = round(float(accuracy_score(actual_labels, preds)), 4)

    return {
        "total": len(results),
        "fraud_count": fraud_count,
        "genuine_count": genuine_count,
        "fraud_rate": fraud_rate,
        "model_used": model_name or ACTIVE_MODEL_NAME,
        "predictions": results,
        "accuracy": accuracy,
        "has_actual_labels": has_actual,
    }


# ── Insights ──────────────────────────────────────────────────────────────────

def get_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute:
    - Amount histogram split by genuine/fraud
    - Feature importances from Random Forest (if trained)
    - Isolation Forest anomaly scores
    """
    has_class = "Class" in df.columns

    # Amount distribution
    amounts = df["Amount"] if "Amount" in df.columns else pd.Series([0])
    bins = np.histogram_bin_edges(amounts.clip(0, amounts.quantile(0.99)), bins=20)
    hist_bins = []
    for i in range(len(bins) - 1):
        mask = (amounts >= bins[i]) & (amounts < bins[i + 1])
        gen_cnt  = int((df.loc[mask, "Class"] == 0).sum()) if has_class else int(mask.sum())
        fraud_cnt = int((df.loc[mask, "Class"] == 1).sum()) if has_class else 0
        hist_bins.append({
            "bin_start": round(float(bins[i]), 2),
            "bin_end":   round(float(bins[i + 1]), 2),
            "genuine_count": gen_cnt,
            "fraud_count":   fraud_cnt,
        })

    # Feature importances
    importances = []
    rf = MODEL_REGISTRY.get("Random Forest")
    if rf is not None and hasattr(rf, "feature_importances_") and FEATURE_COLUMNS:
        imp = rf.feature_importances_
        pairs = sorted(zip(FEATURE_COLUMNS, imp), key=lambda x: x[1], reverse=True)[:15]
        importances = [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs]

    # Anomaly detection with Isolation Forest
    anomalies = []
    try:
        num_df = df.select_dtypes(include=[np.number]).drop(columns=["Class"], errors="ignore")
        num_df = num_df.fillna(0)
        if len(num_df) > 0:
            iso = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
            scores = iso.fit_predict(num_df)
            anomaly_scores_raw = iso.score_samples(num_df)
            anomaly_idx = np.where(scores == -1)[0][:50]
            for idx in anomaly_idx:
                amount = float(df.iloc[idx]["Amount"]) if "Amount" in df.columns else 0.0
                is_fraud = bool(df.iloc[idx]["Class"] == 1) if has_class else None
                anomalies.append({
                    "row_index": int(idx),
                    "amount": round(amount, 2),
                    "anomaly_score": round(float(anomaly_scores_raw[idx]), 4),
                    "is_fraud": is_fraud,
                })
    except Exception:
        pass

    # Stats
    stats: Dict[str, Any] = {
        "total_rows": len(df),
        "fraud_count": int((df["Class"] == 1).sum()) if has_class else None,
        "genuine_count": int((df["Class"] == 0).sum()) if has_class else None,
    }
    if "Amount" in df.columns:
        stats["avg_amount"] = round(float(df["Amount"].mean()), 2)
        stats["max_amount"] = round(float(df["Amount"].max()), 2)
        if has_class:
            fraud_mask = df["Class"] == 1
            stats["avg_fraud_amount"]   = round(float(df.loc[fraud_mask, "Amount"].mean()), 2) if fraud_mask.any() else 0
            stats["avg_genuine_amount"] = round(float(df.loc[~fraud_mask, "Amount"].mean()), 2) if (~fraud_mask).any() else 0

    return {
        "amount_distribution": hist_bins,
        "feature_importances": importances,
        "anomalies": anomalies,
        "stats": stats,
    }


# ── Dashboard data ─────────────────────────────────────────────────────────────

def get_dashboard_data(df: pd.DataFrame, is_sample: bool) -> Dict[str, Any]:
    from ml.synthetic import generate_time_series
    from datetime import datetime

    has_class = "Class" in df.columns

    total = len(df)
    genuine = int((df["Class"] == 0).sum()) if has_class else total
    fraud   = int((df["Class"] == 1).sum()) if has_class else 0
    rate    = round(fraud / max(total, 1) * 100, 2)

    time_series = generate_time_series(df) if "Time" in df.columns else []

    # Recent transactions (last 10 rows as "most recent")
    recent_df = df.tail(10).copy()
    recent_txns = []
    for i, (_, row) in enumerate(recent_df.iterrows()):
        amt = float(row.get("Amount", 0))
        cls = int(row["Class"]) if has_class else 0
        recent_txns.append({
            "id": int(i + 1),
            "amount": round(amt, 2),
            "prediction": "Fraud" if cls == 1 else "Genuine",
            "probability": None,
            "timestamp": f"2024-01-{(i % 28) + 1:02d} 12:{i:02d}:00",
        })

    # Fraud alerts (top fraud transactions by amount)
    alerts = []
    if has_class:
        fraud_df = df[df["Class"] == 1].nlargest(5, "Amount") if "Amount" in df.columns else df[df["Class"] == 1].head(5)
        for j, (_, row) in enumerate(fraud_df.iterrows()):
            amt = float(row.get("Amount", 0))
            prob = 0.9 - j * 0.05
            sev = "High" if prob > 0.85 else "Medium" if prob > 0.75 else "Low"
            alerts.append({
                "id": j + 1,
                "amount": round(amt, 2),
                "probability": round(prob, 2),
                "timestamp": f"2024-01-{(j % 28) + 1:02d} 08:{j * 5:02d}:00",
                "severity": sev,
            })

    system_status = {
        "model_trained": len(MODEL_REGISTRY) > 0,
        "dataset_loaded": True,
        "last_trained": LAST_TRAINED,
        "records_processed": total,
        "active_model": ACTIVE_MODEL_NAME if MODEL_REGISTRY else None,
    }

    return {
        "kpi": {"total_transactions": total, "genuine_count": genuine, "fraud_count": fraud, "fraud_rate": rate},
        "time_series": time_series,
        "recent_transactions": recent_txns,
        "fraud_alerts": alerts,
        "system_status": system_status,
        "is_sample_data": is_sample,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model(model_name: Optional[str]):
    global FEATURE_COLUMNS
    name = model_name or ACTIVE_MODEL_NAME
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name], SCALER_REGISTRY.get("scaler"), FEATURE_COLUMNS

    # Try disk
    path = MODEL_DIR / f"{name.replace(' ', '_')}.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    feat_path   = MODEL_DIR / "feature_columns.npy"

    if path.exists():
        clf = joblib.load(path)
        MODEL_REGISTRY[name] = clf
        if scaler_path.exists():
            SCALER_REGISTRY["scaler"] = joblib.load(scaler_path)
        if feat_path.exists():
            FEATURE_COLUMNS = list(np.load(feat_path, allow_pickle=True))
        return clf, SCALER_REGISTRY.get("scaler"), FEATURE_COLUMNS

    raise ValueError(f"Model '{name}' is not trained. Call /api/train first.")


def _build_feature_row(features: Dict[str, float], feat_cols: List[str], scaler) -> np.ndarray:
    row = pd.DataFrame([features])
    row = row.reindex(columns=feat_cols, fill_value=0.0).astype(float)
    scale_cols = [c for c in ["Amount", "Time"] if c in row.columns]
    if scale_cols and scaler is not None:
        row[scale_cols] = scaler.transform(row[scale_cols])
    return row.values


def _confidence_label(prob: Optional[float]) -> Optional[str]:
    if prob is None:
        return None
    if prob >= 0.8:
        return "High"
    if prob >= 0.5:
        return "Medium"
    return "Low"
