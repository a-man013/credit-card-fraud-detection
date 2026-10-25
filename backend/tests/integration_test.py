"""Quick integration test for all API endpoints."""
import urllib.request
import json
import sys

BASE = "http://127.0.0.1:8000"

def get(path):
    r = urllib.request.urlopen(BASE + path, timeout=30)
    return json.loads(r.read())

def post_json(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    r = urllib.request.urlopen(req, timeout=120)
    return json.loads(r.read())

errors = []

try:
    h = get("/api/health")
    print(f"[OK] GET /api/health  status={h['status']}  model_loaded={h['model_loaded']}")
except Exception as e:
    print(f"[FAIL] GET /api/health: {e}")
    errors.append(e)

try:
    d = get("/api/dashboard")
    print(f"[OK] GET /api/dashboard  total_txns={d['kpi']['total_transactions']}  is_sample={d['is_sample_data']}")
except Exception as e:
    print(f"[FAIL] GET /api/dashboard: {e}")
    errors.append(e)

try:
    t = post_json("/api/train", {
        "model_names": ["Random Forest", "Logistic Regression", "Decision Tree"],
        "test_size": 0.2,
        "use_smote": True
    })
    print(f"[OK] POST /api/train  {t['message']}")
except Exception as e:
    print(f"[FAIL] POST /api/train: {e}")
    errors.append(e)

try:
    ev = get("/api/evaluation")
    print(f"[OK] GET /api/evaluation  models={len(ev['models'])}  best={ev['best_model']}")
    for m in ev["models"]:
        print(f"       {m['model_name']}: acc={m['accuracy']}  prec={m['precision']}  recall={m['recall']}  f1={m['f1_score']}")
except Exception as e:
    print(f"[FAIL] GET /api/evaluation: {e}")
    errors.append(e)

try:
    p = post_json("/api/predict", {
        "features": {"Amount": 149.62, "V1": -1.359807, "V2": -0.072781, "Time": 0},
        "model_name": "Random Forest"
    })
    print(f"[OK] POST /api/predict  prediction={p['prediction']}  prob={p['probability_fraud']}")
except Exception as e:
    print(f"[FAIL] POST /api/predict: {e}")
    errors.append(e)

try:
    ins = get("/api/insights")
    print(f"[OK] GET /api/insights  hist_bins={len(ins['amount_distribution'])}  anomalies={len(ins['anomalies'])}  importances={len(ins['feature_importances'])}")
except Exception as e:
    print(f"[FAIL] GET /api/insights: {e}")
    errors.append(e)

print()
if errors:
    print(f"[SUMMARY] {len(errors)} endpoint(s) FAILED")
    sys.exit(1)
else:
    print("[SUMMARY] All endpoints passed!")
