# FraudShield 🛡️
### Credit Card Fraud Detection Using Data Mining

A full-stack, production-ready web application for real-time and batch credit card fraud detection using machine learning algorithms.

---

## 🖥️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19, Vite, Tailwind CSS v4, Recharts, Lucide Icons |
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **ML Pipeline** | Scikit-learn (Random Forest, Decision Tree, Logistic Regression, Isolation Forest), Imbalanced-learn (SMOTE), Pandas, NumPy |
| **State & API** | Axios, TanStack React Query |

---

## 📁 Project Structure

```
CREDIT CARD FRAUD DETECTION/
├── backend/
│   ├── main.py              ← FastAPI app & REST endpoints
│   ├── schemas.py           ← Pydantic request/response models
│   ├── requirements.txt     ← Python backend dependencies
│   ├── ml/
│   │   ├── pipeline.py      ← Preprocessing, SMOTE, training, and evaluation
│   │   └── synthetic.py     ← 2,000-row synthetic dataset generator (fallback)
│   ├── saved_models/        ← Persisted model weights and feature scalers
│   └── tests/
│       └── test_pipeline.py ← Pytest unit test suite
├── frontend/
│   ├── index.html
│   ├── vite.config.js       ← Vite config + API proxy (/api → :8000)
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx          ← Router configuration
│       ├── index.css        ← Minimalist dark theme tokens
│       ├── api/client.js    ← Axios API client
│       ├── components/
│       │   ├── Layout/      ← Sidebar, Topbar, AppLayout
│       │   └── UI/          ← Card, Badge, Spinner, AlertBanner
│       └── pages/
│           ├── Dashboard.jsx
│           ├── TransactionAnalyzer.jsx
│           ├── ModelPerformance.jsx
│           ├── FraudInsights.jsx
│           └── DatasetManagement.jsx
├── .gitignore
├── .env.example
├── package.json             ← Root script runner
├── start.bat                ← One-click Windows launcher
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** (Tested on Python 3.11)
- **Node.js 18+**

---

### Option 1: One-Click Launch (Windows)

Simply double-click or run:
```powershell
.\start.bat
```
This automatically launches both the FastAPI backend and Vite frontend in separate terminals.

---

### Option 2: Manual Terminal Setup

#### 1. Start the Backend (Terminal 1)
```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

#### 2. Start the Frontend (Terminal 2)
```powershell
cd frontend
npm install
npm run dev
```
- **Frontend App:** [http://localhost:5173](http://localhost:5173) (or [http://localhost:5174](http://localhost:5174))

---

## 🧪 Running Tests

To verify the data processing pipeline, SMOTE, and model inference:

```powershell
cd backend
python -m pytest tests/test_pipeline.py -v
```

All 12 unit tests should pass.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Backend and model health status |
| `GET` | `/api/dashboard` | Dashboard KPIs, activity timeline, and fraud alerts |
| `POST` | `/api/upload` | Upload & validate CSV dataset |
| `POST` | `/api/predict` | Real-time single transaction scoring |
| `POST` | `/api/predict/batch` | Bulk CSV transaction prediction |
| `POST` | `/api/train` | Train and cross-evaluate models on uploaded data |
| `GET` | `/api/evaluation` | Held-out test split evaluation metrics |
| `GET` | `/api/insights` | Feature importances, anomaly scores, and amount distribution |

---

## 📊 Getting the Kaggle Benchmark Dataset

The system includes a pre-packaged 2,000-row synthetic dataset to run immediately out-of-the-box. To benchmark with real European cardholder data:

1. Download **`creditcard.csv`** from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Open the app and navigate to **Dataset Management**.
3. Upload `creditcard.csv` to validate and load.
4. Navigate to **Model Performance** and click **Train Models**.

---

## 📄 License
This project is open-source under the MIT License.
