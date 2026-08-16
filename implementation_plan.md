# Implementation Plan: CureBlend Healthcare Recommendation System Backend

CureBlend is an AI-driven healthcare decision-support backend built with FastAPI, Scikit-Learn/XGBoost, SHAP (XAI), Rule Engines (Safety, Severity, Dual Recommendations), and MongoDB Atlas persistence.

## User Review Required

> [!IMPORTANT]
> - The dataset for 24 core medical conditions will be synthesized with realistic medical symptom variations in `train.py` so the model can be trained and export model artifacts (`model.joblib`, `tfidf.joblib`, `label_encoder.joblib`) locally without needing external API credentials during setup.
> - MongoDB Atlas connection will support standard environment variable `MONGODB_URL` with a standard fallback/mock mode if MongoDB is not reachable so backend testing works out of the box.

## Proposed Components & Files

### 1. Root & Configuration
- **[NEW] `cureblend-backend/requirements.txt`**: Complete Python package requirements (`fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `xgboost`, `shap`, `joblib`, `motor`, `pymongo`, `python-dotenv`, `pandas`, `numpy`).
- **[NEW] `cureblend-backend/app/config.py`**: Configuration management using `pydantic-settings` or `os.getenv` for MongoDB URI, DB Name, environment, and model paths.

### 2. Knowledge Base & Data Generation
- **[NEW] `cureblend-backend/data/knowledge_base.json`**: Structured dictionary mapping 24 core conditions to severity weights, pharmaceutical remedies (OTC usage, contraindications), and herbal/lifestyle guidance.
- **[NEW] `cureblend-backend/train.py`**: Offline model training script generating realistic symptom texts across 24 conditions, fitting TF-IDF (`ngram_range=(1,2)`) and `XGBClassifier`, and dumping `.joblib` model artifacts into `data/`.

### 3. Pydantic Validation Schemas
- **[NEW] `cureblend-backend/app/models/schemas.py`**:
  - `AssessmentRequest`: Input symptoms list, natural language text, age, lifestyle factors, existing conditions.
  - `AssessmentResponse`: Probability predictions, dynamic severity level/score, SHAP feature weights, emergency status, and dual recommendations.
  - `PredictionHistoryItem`: MongoDB schema for stored assessment logs.

### 4. ML & XAI Pipeline (`app/ml/`)
- **[NEW] `cureblend-backend/app/ml/pipeline.py`**: Model artifact loader & inference predictor returning multi-disease calibrated probability distribution.
- **[NEW] `cureblend-backend/app/ml/explainer.py`**: SHAP `TreeExplainer` feature weight attribution extractor explaining top predicted conditions.

### 5. Rule Engines & Services (`app/services/`)
- **[NEW] `cureblend-backend/app/services/safety.py`**: Emergency red-flag interceptor for life-threatening symptom keywords.
- **[NEW] `cureblend-backend/app/services/severity.py`**: Dynamic score calculator (Low, Moderate, High, Critical) using age, symptom count, and disease severity weights.
- **[NEW] `cureblend-backend/app/services/recommender.py`**: Evidence-based dual recommendation engine with patient contraindication checks.

### 6. Database & API Endpoints (`app/database/`, `app/main.py`)
- **[NEW] `cureblend-backend/app/database/connection.py`**: Async Motor driver setup & history CRUD operations with graceful fallback.
- **[NEW] `cureblend-backend/app/main.py`**: FastAPI application with CORS middleware, health check `/api/v1/health`, assessment endpoint `/api/v1/assess`, and history endpoint `/api/v1/history`.

---

## Verification Plan

### Automated Verification
1. Environment & Package installation in Python venv.
2. Execute `python train.py` to train the model and generate `data/model.joblib`, `data/tfidf.joblib`, and `data/label_encoder.joblib`.
3. Test inference, SHAP XAI, Red-Flag Interceptor, Severity Scoring, and Dual Recommendations via Python test scripts.
4. Run FastAPI app via Uvicorn and verify OpenAPI docs (`/docs`) and API responses for `/api/v1/assess`, `/api/v1/history`, and `/api/v1/health`.

### Manual Verification
- Test emergency payload (e.g. "chest pain and shortness of breath") to confirm `is_emergency: True`.
- Test standard symptom payload (e.g. "fever, cough, body aches") to verify disease probabilities, SHAP weights, severity level, and filtered dual recommendations.
