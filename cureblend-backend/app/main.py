"""
CureBlend — FastAPI Application & Route Handlers
==================================================
Main entry point for the CureBlend AI Healthcare Recommendation API.

Endpoints:
  POST /api/v1/assess   → Full pipeline (NLP → ML Predict → SHAP → Safety → Severity → Recommendations)
  GET  /api/v1/history   → Retrieve past assessment records
  GET  /api/v1/health    → API health check
"""

import sys
from pathlib import Path

# Ensure project root (cureblend-backend) is in sys.path for uvicorn subprocess reloader
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    APP_NAME, APP_VERSION, APP_DESCRIPTION,
    CORS_ORIGINS, TOP_K_PREDICTIONS
)
from app.models.schemas import (
    AssessmentRequest, AssessmentResponse,
    ConditionProbability, FeatureWeight,
    EmergencyAlert, SeverityAssessment,
    DualRecommendation, PharmaceuticalRecommendation, HerbalRecommendation,
    HistoryResponse, PredictionHistoryItem
)
from app.ml.pipeline import predict_conditions, get_artifacts
from app.ml.explainer import compute_feature_importance
from app.services.safety import scan_for_emergencies
from app.services.severity import calculate_severity
from app.services.recommender import get_recommendations
from app.database.connection import db_manager, save_assessment, get_history


# ══════════════════════════════════════════════════════════════
#  LIFESPAN EVENTS (startup / shutdown)
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # ── Startup ──
    print("\n" + "=" * 60)
    print(f"  {APP_NAME} v{APP_VERSION} — Starting up...")
    print("=" * 60)
    
    # Load ML model artifacts
    try:
        get_artifacts()
        print("  [OK] ML Pipeline loaded.")
    except FileNotFoundError as e:
        print(f"  [!] ML Pipeline not loaded: {e}")
        print("  -> Run `python train.py` first to generate model artifacts.")
    
    # Connect to MongoDB
    await db_manager.connect()
    
    print("=" * 60)
    print(f"  {APP_NAME} is ready! -> http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    yield  # Application is running
    
    # ── Shutdown ──
    print("\n  Shutting down...")
    await db_manager.disconnect()
    print("  [OK] Shutdown complete.\n")


# ══════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
#  UTILITY: Compute Lifestyle Risk Score
# ══════════════════════════════════════════════════════════════

def _compute_lifestyle_risk(lifestyle_factors: list[str]) -> float:
    """Compute a numeric risk score from lifestyle factors."""
    risk_weights = {
        "smoking": 1.5, "smoker": 1.5,
        "alcohol": 1.0, "heavy drinking": 1.5,
        "sedentary": 0.8, "no exercise": 0.8,
        "obesity": 1.2, "obese": 1.2,
        "poor diet": 0.7, "junk food": 0.7,
        "stress": 0.5, "sleep deprivation": 0.6,
        "drug use": 1.5, "substance abuse": 1.5,
    }
    
    score = 1.0  # Base score
    for factor in (lifestyle_factors or []):
        factor_lower = factor.lower()
        for keyword, weight in risk_weights.items():
            if keyword in factor_lower:
                score += weight
                break
        else:
            score += 0.3  # Unknown factor gets small weight
    
    return min(score, 5.0)


# ══════════════════════════════════════════════════════════════
#  ENDPOINT: POST /api/v1/assess
# ══════════════════════════════════════════════════════════════

@app.post(
    "/api/v1/assess",
    response_model=AssessmentResponse,
    summary="Run Full Health Assessment Pipeline",
    description="Accepts symptoms and patient info, runs ML prediction, "
                "SHAP explanation, safety check, severity scoring, and "
                "returns dual pharmaceutical + herbal recommendations."
)
async def assess_health(request: AssessmentRequest):
    """
    Full CureBlend assessment pipeline:
      1. Safety Red-Flag Scan (emergency check FIRST)
      2. ML Multi-Disease Prediction (TF-IDF + XGBoost)
      3. SHAP Feature Importance Explanation
      4. Dynamic Severity Score Calculation
      5. Dual Recommendation Lookup (with contraindication filtering)
      6. MongoDB History Persistence
    """
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)
    
    # ── 1. SAFETY RED-FLAG SCAN (runs FIRST, before ML) ──────
    emergency_result = scan_for_emergencies(
        symptom_text=request.symptom_text,
        symptoms=request.symptoms
    )
    
    emergency = EmergencyAlert(
        is_emergency=emergency_result["is_emergency"],
        matched_flags=emergency_result["matched_flags"],
        advisory=emergency_result["advisory"]
    )
    
    # ── 2. ML MULTI-DISEASE PREDICTION ──────────────────────
    try:
        lifestyle_risk = _compute_lifestyle_risk(request.lifestyle_factors)
        
        predictions_raw = predict_conditions(
            symptom_text=request.symptom_text,
            symptoms=request.symptoms,
            age=request.age,
            existing_conditions_count=len(request.existing_conditions),
            lifestyle_risk_score=lifestyle_risk,
            top_k=TOP_K_PREDICTIONS
        )
        
        predictions = [
            ConditionProbability(
                condition=p["condition"],
                probability=p["probability"],
                rank=p["rank"]
            )
            for p in predictions_raw
        ]
        
        top_condition = predictions_raw[0]["condition"] if predictions_raw else "Unknown"
        top_confidence = predictions_raw[0]["probability"] if predictions_raw else 0.0
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run `python train.py` first to train the model."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ML prediction error: {str(e)}"
        )
    
    # ── 3. SHAP / FEATURE IMPORTANCE ────────────────────────
    try:
        shap_raw = compute_feature_importance(
            symptom_text=request.symptom_text,
            symptoms=request.symptoms,
            predicted_condition=top_condition
        )
        
        shap_importance = [
            FeatureWeight(feature=f["feature"], weight=f["weight"])
            for f in shap_raw
        ]
    except Exception as e:
        print(f"  [!] SHAP computation failed: {e}")
        shap_importance = []
    
    # ── 4. SEVERITY SCORING ─────────────────────────────────
    severity_raw = calculate_severity(
        predicted_condition=top_condition,
        symptoms=request.symptoms,
        age=request.age,
        existing_conditions=request.existing_conditions,
        lifestyle_factors=request.lifestyle_factors,
        prediction_confidence=top_confidence
    )
    
    severity = SeverityAssessment(
        score=severity_raw["score"],
        level=severity_raw["level"],
        breakdown=severity_raw["breakdown"]
    )
    
    # Override severity to Critical if emergency detected
    if emergency.is_emergency:
        severity = SeverityAssessment(
            score=max(severity_raw["score"], 95),
            level="Critical",
            breakdown={**severity_raw["breakdown"], "emergency_override": True}
        )
    
    # ── 5. DUAL RECOMMENDATIONS ─────────────────────────────
    reco_raw = get_recommendations(
        predicted_condition=top_condition,
        age=request.age,
        existing_conditions=request.existing_conditions,
        lifestyle_factors=request.lifestyle_factors
    )
    
    recommendations = DualRecommendation(
        condition=reco_raw["condition"],
        pharmaceutical=[
            PharmaceuticalRecommendation(
                medication=p["medication"],
                usage=p["usage"],
                precautions=p["precautions"]
            )
            for p in reco_raw.get("pharmaceutical", [])
        ],
        herbal=[
            HerbalRecommendation(
                remedy=h["remedy"],
                usage=h["usage"],
                precautions=h["precautions"]
            )
            for h in reco_raw.get("herbal", [])
        ],
        contraindication_warnings=reco_raw.get("contraindication_warnings", [])
    )
    
    # ── 6. PERSIST TO MONGODB ───────────────────────────────
    reco_summary = f"Top: {top_condition} ({top_confidence:.0%}). " \
                   f"Pharma: {len(reco_raw.get('pharmaceutical', []))} meds. " \
                   f"Herbal: {len(reco_raw.get('herbal', []))} remedies."
    
    history_record = {
        "request_id": request_id,
        "timestamp": timestamp,
        "input_symptoms": request.symptoms,
        "symptom_text": request.symptom_text,
        "age": request.age,
        "existing_conditions": request.existing_conditions,
        "lifestyle_factors": request.lifestyle_factors,
        "top_prediction": top_condition,
        "confidence": top_confidence,
        "severity_level": severity.level,
        "severity_score": severity.score,
        "is_emergency": emergency.is_emergency,
        "recommendations_summary": reco_summary
    }
    
    await save_assessment(history_record)
    
    # ── 7. BUILD & RETURN RESPONSE ──────────────────────────
    return AssessmentResponse(
        request_id=request_id,
        timestamp=timestamp,
        emergency=emergency,
        predictions=predictions,
        shap_importance=shap_importance,
        severity=severity,
        recommendations=recommendations
    )


# ══════════════════════════════════════════════════════════════
#  ENDPOINT: GET /api/v1/history
# ══════════════════════════════════════════════════════════════

@app.get(
    "/api/v1/history",
    response_model=HistoryResponse,
    summary="Retrieve Assessment History",
    description="Fetch past assessment records from MongoDB with pagination support."
)
async def get_assessment_history(
    limit: int = Query(default=50, ge=1, le=200, description="Max records to return"),
    skip: int = Query(default=0, ge=0, description="Records to skip for pagination")
):
    """Retrieve paginated assessment history from MongoDB."""
    records, total = await get_history(limit=limit, skip=skip)
    
    history_items = []
    for record in records:
        try:
            history_items.append(
                PredictionHistoryItem(
                    request_id=record.get("request_id", ""),
                    timestamp=record.get("timestamp", datetime.now(timezone.utc)),
                    input_symptoms=record.get("input_symptoms", []),
                    symptom_text=record.get("symptom_text", ""),
                    age=record.get("age", 0),
                    top_prediction=record.get("top_prediction", "Unknown"),
                    confidence=record.get("confidence", 0.0),
                    severity_level=record.get("severity_level", "Unknown"),
                    is_emergency=record.get("is_emergency", False),
                    recommendations_summary=record.get("recommendations_summary", "")
                )
            )
        except Exception:
            continue  # Skip malformed records
    
    return HistoryResponse(
        total_records=total,
        records=history_items
    )


# ══════════════════════════════════════════════════════════════
#  ENDPOINT: GET /api/v1/health
# ══════════════════════════════════════════════════════════════

@app.get(
    "/api/v1/health",
    summary="API Health Check",
    description="Returns API status, model availability, and database connectivity."
)
async def health_check():
    """Health check endpoint for monitoring and deployment verification."""
    
    # Check model status
    try:
        arts = get_artifacts()
        model_status = "loaded"
        model_classes = len(arts.label_encoder.classes_)
    except Exception:
        model_status = "not_loaded"
        model_classes = 0
    
    return {
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "components": {
            "ml_model": {
                "status": model_status,
                "conditions_supported": model_classes
            },
            "database": {
                "status": "connected" if db_manager.is_connected else "disconnected",
                "engine": "MongoDB"
            }
        },
        "endpoints": {
            "assess": "POST /api/v1/assess",
            "history": "GET /api/v1/history",
            "health": "GET /api/v1/health",
            "docs": "GET /docs"
        }
    }


# ══════════════════════════════════════════════════════════════
#  UVICORN ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
