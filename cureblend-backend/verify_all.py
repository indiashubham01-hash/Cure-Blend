"""
Comprehensive Verification Script for CureBlend Backend
Tests all modules, ML pipeline, XAI, Rule Engines, Database, and FastAPI handlers.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import (
    APP_NAME, APP_VERSION, MODEL_PATH, TFIDF_PATH, 
    LABEL_ENCODER_PATH, SCALER_PATH, MLB_PATH, KNOWLEDGE_BASE_PATH
)
from app.ml.pipeline import get_artifacts, predict_conditions, preprocess_text
from app.ml.explainer import compute_feature_importance
from app.services.safety import scan_for_emergencies, check_contraindications
from app.services.severity import calculate_severity
from app.services.recommender import get_recommendations
from app.database.connection import db_manager, save_assessment, get_history, get_assessment_by_id
from app.models.schemas import AssessmentRequest, AssessmentResponse
from app.main import assess_health, get_assessment_history, health_check


async def run_all_checks():
    print("=" * 70)
    print(f"[+] STARTING SYSTEM-WIDE HEALTH & VERIFICATION CHECK FOR {APP_NAME} v{APP_VERSION}")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. FILE & ARTIFACT EXISTENCE CHECK
    # -------------------------------------------------------------
    print("\n[1/7] Checking Model & Knowledge Base Artifacts on Disk...")
    files_to_check = [
        ("Model", MODEL_PATH),
        ("TF-IDF Vectorizer", TFIDF_PATH),
        ("Label Encoder", LABEL_ENCODER_PATH),
        ("Scaler", SCALER_PATH),
        ("Multi-Label Binarizer", MLB_PATH),
        ("Knowledge Base", KNOWLEDGE_BASE_PATH)
    ]
    all_files_ok = True
    for name, path in files_to_check:
        exists = path.exists()
        size_kb = (path.stat().st_size / 1024) if exists else 0
        status = f"[OK] ({size_kb:.1f} KB)" if exists else "[FAIL] MISSING"
        print(f"  * {name:<22}: {path.name:<24} -> {status}")
        if not exists:
            all_files_ok = False
    
    assert all_files_ok, "Missing required artifacts!"
    print("  -> All required artifacts exist!")

    # -------------------------------------------------------------
    # 2. ML ARTIFACT LOADING & INFERENCE
    # -------------------------------------------------------------
    print("\n[2/7] Testing ML Pipeline Artifact Loading & Inference...")
    arts = get_artifacts()
    classes = list(arts.label_encoder.classes_)
    print(f"  * Classes supported ({len(classes)}): {', '.join(classes[:8])}...")
    
    test_cases = [
        {
            "desc": "Flu-like symptoms",
            "text": "I have high fever, dry cough, severe fatigue, and body ache",
            "symptoms": ["fever", "cough", "fatigue", "body_ache"],
            "age": 32,
            "existing": [],
            "lifestyle": []
        },
        {
            "desc": "Migraine symptoms",
            "text": "Intense throbbing headache on one side of head, sensitivity to light, nausea",
            "symptoms": ["headache", "nausea", "light_sensitivity"],
            "age": 28,
            "existing": [],
            "lifestyle": ["stress"]
        },
        {
            "desc": "GERD / Acid reflux symptoms",
            "text": "Severe heartburn, acid reflux after eating, burning chest discomfort",
            "symptoms": ["heartburn", "acid_reflux", "chest_burning"],
            "age": 45,
            "existing": [],
            "lifestyle": ["poor diet"]
        }
    ]

    for tc in test_cases:
        preds = predict_conditions(
            symptom_text=tc["text"],
            symptoms=tc["symptoms"],
            age=tc["age"],
            existing_conditions_count=len(tc["existing"]),
            lifestyle_risk_score=1.0,
            top_k=3
        )
        print(f"  * Test Case '{tc['desc']}':")
        for p in preds:
            print(f"     Rank {p['rank']}: {p['condition']:<22} (Probability: {p['probability']*100:.2f}%)")
        assert len(preds) > 0, "Prediction returned empty list!"

    # -------------------------------------------------------------
    # 3. SHAP / XAI FEATURE ATTRIBUTION
    # -------------------------------------------------------------
    print("\n[3/7] Testing XAI Feature Attribution (SHAP / TF-IDF)...")
    shap_results = compute_feature_importance(
        symptom_text="I have high fever, persistent dry cough, and headache",
        symptoms=["fever", "cough", "headache"],
        predicted_condition=preds[0]["condition"],
        top_k=5
    )
    print(f"  * Feature weights for '{preds[0]['condition']}':")
    for item in shap_results:
        print(f"     Feature: {item['feature']:<20} | Weight: {item['weight']:.4f}")
    assert len(shap_results) > 0, "Feature importance failed to compute!"

    # -------------------------------------------------------------
    # 4. RED-FLAG SAFETY INTERCEPTOR
    # -------------------------------------------------------------
    print("\n[4/7] Testing Red-Flag Safety Interceptor...")
    # Test 4a: Emergency trigger
    emerg_res = scan_for_emergencies(
        symptom_text="I have severe crushing chest pain, difficulty breathing, and dizziness",
        symptoms=["chest_pain", "difficulty_breathing"]
    )
    print(f"  * Emergency Case: is_emergency = {emerg_res['is_emergency']}")
    print(f"    Matched flags: {emerg_res['matched_flags']}")
    assert emerg_res["is_emergency"] is True, "Safety scan failed to catch chest pain emergency!"
    
    # Test 4b: Non-emergency normal case
    normal_res = scan_for_emergencies(
        symptom_text="Mild runny nose and sneezing for two days",
        symptoms=["runny_nose", "sneezing"]
    )
    print(f"  * Non-Emergency Case: is_emergency = {normal_res['is_emergency']}")
    assert normal_res["is_emergency"] is False, "Safety scan raised false emergency!"

    # -------------------------------------------------------------
    # 5. DYNAMIC SEVERITY SCORING & RECOMMENDATIONS
    # -------------------------------------------------------------
    print("\n[5/7] Testing Severity Calculator & Dual Recommendations...")
    # Young healthy vs Elderly with comorbidities
    sev_young = calculate_severity(
        predicted_condition="Common Cold",
        symptoms=["runny_nose", "sneezing"],
        age=25,
        existing_conditions=[],
        lifestyle_factors=[],
        prediction_confidence=0.9
    )
    sev_elderly = calculate_severity(
        predicted_condition="Pneumonia",
        symptoms=["fever", "cough", "fatigue", "chills", "shortness_of_breath"],
        age=78,
        existing_conditions=["hypertension", "diabetes"],
        lifestyle_factors=["smoker"],
        prediction_confidence=0.85
    )
    print(f"  * Young patient score: {sev_young['score']}/100 (Level: {sev_young['level']})")
    print(f"  * Elderly high-risk score: {sev_elderly['score']}/100 (Level: {sev_elderly['level']})")
    assert sev_elderly["score"] > sev_young["score"], "Elderly risk not scored higher!"

    # Dual recommendations & Contraindications
    recos = get_recommendations(
        predicted_condition="Hypertension",
        age=65,
        existing_conditions=["kidney disease", "asthma"],
        lifestyle_factors=["smoking", "sedentary"]
    )
    print(f"  * Recos for {recos['condition']}:")
    print(f"    Pharma remedies: {len(recos['pharmaceutical'])}")
    print(f"    Herbal remedies: {len(recos['herbal'])}")
    print(f"    Contraindication warnings: {len(recos['contraindication_warnings'])}")
    for w in recos['contraindication_warnings'][:3]:
        safe_w = w.encode("ascii", "ignore").decode("ascii")
        print(f"      - {safe_w.strip()}")

    # -------------------------------------------------------------
    # 6. DATABASE (MongoDB + Local SQLite Fallback)
    # -------------------------------------------------------------
    print("\n[6/7] Testing Database Storage & Fallback...")
    await db_manager.connect()
    test_record = {
        "request_id": "test-verification-12345",
        "input_symptoms": ["fever", "cough"],
        "symptom_text": "testing db persistence",
        "age": 30,
        "existing_conditions": [],
        "lifestyle_factors": [],
        "top_prediction": "Influenza",
        "confidence": 0.92,
        "severity_level": "Moderate",
        "severity_score": 45,
        "is_emergency": False,
        "recommendations_summary": "Test summary"
    }
    saved = await save_assessment(test_record)
    print(f"  * Record Save Result: {saved}")
    assert saved is True, "Failed to save record to database/fallback!"

    records, total = await get_history(limit=5, skip=0)
    print(f"  * History fetch result: total = {total}, records returned = {len(records)}")
    assert total > 0, "History query returned 0 records!"

    fetched_single = await get_assessment_by_id("test-verification-12345")
    assert fetched_single is not None, "Failed to retrieve single record by ID!"
    print(f"  * Single record fetched: ID = {fetched_single['request_id']}, Condition = {fetched_single.get('top_prediction')}")

    # -------------------------------------------------------------
    # 7. FASTAPI FULL END-TO-END ENDPOINT HANDLERS
    # -------------------------------------------------------------
    print("\n[7/7] Testing FastAPI Endpoints (Health, Assess, History)...")
    
    # 7a. Health check
    health_res = await health_check()
    print(f"  * GET /api/v1/health: Status = {health_res['status']}, ML = {health_res['components']['ml_model']['status']}")
    assert health_res["status"] == "healthy"
    
    # 7b. Full assess endpoint
    assess_req = AssessmentRequest(
        symptom_text="I have severe fever with chills, body ache, and persistent cough",
        symptoms=["fever", "cough", "chills", "body_ache"],
        age=35,
        lifestyle_factors=["stress"],
        existing_conditions=[]
    )
    assess_res: AssessmentResponse = await assess_health(assess_req)
    print(f"  * POST /api/v1/assess Result:")
    print(f"     Request ID: {assess_res.request_id}")
    print(f"     Top Prediction: {assess_res.predictions[0].condition} ({assess_res.predictions[0].probability*100:.1f}%)")
    print(f"     Severity: Level {assess_res.severity.level} (Score {assess_res.severity.score})")
    print(f"     Emergency Alert: {assess_res.emergency.is_emergency}")
    print(f"     Top SHAP Feature: {assess_res.shap_importance[0].feature if assess_res.shap_importance else 'N/A'}")
    print(f"     Pharma Recos: {len(assess_res.recommendations.pharmaceutical)}")
    print(f"     Herbal Recos: {len(assess_res.recommendations.herbal)}")
    
    # 7c. History endpoint
    history_res = await get_assessment_history(limit=10, skip=0)
    print(f"  * GET /api/v1/history: Total records = {history_res.total_records}, returned items = {len(history_res.records)}")
    assert len(history_res.records) > 0

    await db_manager.disconnect()

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL SYSTEM CHECKS PASSED PERFECTLY! 100% OPERATIONAL.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_all_checks())
