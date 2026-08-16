"""
CureBlend — Dynamic Severity Score Calculator
===============================================
Computes a severity score (0–100) based on:
  - Symptom load (count of symptoms)
  - Patient age (elderly >60 get higher risk)
  - Existing medical conditions
  - Lifestyle risk factors
  - Disease-specific severity weight from knowledge base

Categories:
  Low:      0–39
  Moderate: 40–69
  High:     70–89
  Critical: 90–100
"""

import json
from pathlib import Path
from app.config import KNOWLEDGE_BASE_PATH


# ══════════════════════════════════════════════════════════════
#  LOAD DISEASE SEVERITY WEIGHTS
# ══════════════════════════════════════════════════════════════

def _load_severity_weights() -> dict[str, int]:
    """Load disease → severity_weight mapping from knowledge_base.json."""
    try:
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
        return {
            disease: data.get("severity_weight", 30)
            for disease, data in kb.get("conditions", {}).items()
        }
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback: default weights
        return {}


# Cache on module load
_SEVERITY_WEIGHTS = _load_severity_weights()

# High-risk lifestyle factor multiplier keywords
HIGH_RISK_LIFESTYLE = {
    "smoking": 1.15,
    "smoker": 1.15,
    "heavy drinking": 1.10,
    "alcohol": 1.08,
    "sedentary": 1.05,
    "obesity": 1.12,
    "obese": 1.12,
    "drug use": 1.15,
    "substance abuse": 1.15,
    "poor diet": 1.05,
    "sleep deprivation": 1.05,
    "stress": 1.03,
    "no exercise": 1.05,
}


# ══════════════════════════════════════════════════════════════
#  SEVERITY CALCULATOR
# ══════════════════════════════════════════════════════════════

def calculate_severity(
    predicted_condition: str,
    symptoms: list[str],
    age: int,
    existing_conditions: list[str],
    lifestyle_factors: list[str],
    prediction_confidence: float = 0.5
) -> dict:
    """
    Compute a dynamic severity score (0–100) and categorical level.
    
    Scoring formula:
      Base Score = disease_severity_weight (from knowledge base)
      + Symptom Load Bonus (more symptoms → higher severity)
      + Age Risk Factor (elderly >60 → bonus points)
      + Condition Risk (existing conditions → compounding risk)
      + Lifestyle Multiplier (risky habits → multiplied score)
      + Confidence Adjustment (higher ML confidence → stronger weight)
    
    Args:
        predicted_condition: Top predicted disease name
        symptoms: List of symptom strings
        age: Patient age
        existing_conditions: Pre-existing conditions
        lifestyle_factors: Lifestyle habits
        prediction_confidence: ML model confidence for top prediction
    
    Returns:
        dict: {
            "score": int (0–100),
            "level": str,
            "breakdown": {component: value, ...}
        }
    """
    
    # ── 1. Base Disease Severity Weight ──────────────────────
    disease_weight = _SEVERITY_WEIGHTS.get(predicted_condition, 30)
    
    # Scale to 0–40 range (max disease weight is ~85)
    base_score = (disease_weight / 100) * 40
    
    # ── 2. Symptom Load Score (0–20) ─────────────────────────
    symptom_count = len(symptoms) if symptoms else 1
    # Logarithmic scaling: more symptoms add diminishing returns
    symptom_score = min(20, symptom_count * 2.5)
    
    # ── 3. Age Risk Factor (0–15) ────────────────────────────
    if age >= 75:
        age_score = 15
    elif age >= 60:
        age_score = 10
    elif age >= 45:
        age_score = 5
    elif age <= 5:
        age_score = 12  # Pediatric vulnerability
    elif age <= 12:
        age_score = 8
    else:
        age_score = 2  # Healthy adult range
    
    # ── 4. Existing Condition Risk (0–15) ────────────────────
    condition_count = len(existing_conditions) if existing_conditions else 0
    condition_score = min(15, condition_count * 5)
    
    # ── 5. Confidence Adjustment (0–10) ──────────────────────
    # Higher model confidence means the assessment is more certain
    confidence_score = prediction_confidence * 10
    
    # ── 6. Aggregate Raw Score ───────────────────────────────
    raw_score = base_score + symptom_score + age_score + condition_score + confidence_score
    
    # ── 7. Lifestyle Risk Multiplier ─────────────────────────
    lifestyle_multiplier = 1.0
    for factor in (lifestyle_factors or []):
        factor_lower = factor.lower()
        for keyword, mult in HIGH_RISK_LIFESTYLE.items():
            if keyword in factor_lower:
                lifestyle_multiplier *= mult
                break
    
    final_score = raw_score * lifestyle_multiplier
    
    # ── 8. Clamp to 0–100 ───────────────────────────────────
    final_score = int(min(100, max(0, round(final_score))))
    
    # ── 9. Categorize ───────────────────────────────────────
    if final_score >= 90:
        level = "Critical"
    elif final_score >= 70:
        level = "High"
    elif final_score >= 40:
        level = "Moderate"
    else:
        level = "Low"
    
    return {
        "score": final_score,
        "level": level,
        "breakdown": {
            "disease_base_weight": round(base_score, 1),
            "symptom_load": round(symptom_score, 1),
            "age_factor": round(age_score, 1),
            "condition_risk": round(condition_score, 1),
            "confidence_factor": round(confidence_score, 1),
            "lifestyle_multiplier": round(lifestyle_multiplier, 2),
            "raw_score_before_lifestyle": round(raw_score, 1)
        }
    }
