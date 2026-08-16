"""
CureBlend — Dual Recommendation Engine
========================================
Provides evidence-based Pharmaceutical guidance AND Herbal/Lifestyle
recommendations for predicted conditions, with patient-specific
contraindication filtering based on age and existing conditions.

Data source: data/knowledge_base.json
"""

import json
from pathlib import Path
from app.config import KNOWLEDGE_BASE_PATH
from app.services.safety import check_contraindications


# ══════════════════════════════════════════════════════════════
#  LOAD KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════

def _load_knowledge_base() -> dict:
    """Load the full knowledge base from JSON."""
    try:
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  ⚠ Knowledge base load error: {e}")
        return {"conditions": {}}


# Cache on module load
_KNOWLEDGE_BASE = _load_knowledge_base()


# ══════════════════════════════════════════════════════════════
#  AGE-BASED SAFETY CHECKS
# ══════════════════════════════════════════════════════════════

def _age_warnings(age: int) -> list[str]:
    """Generate age-specific safety warnings."""
    warnings = []
    if age < 2:
        warnings.append("⚠️ INFANT SAFETY: Most medications are NOT safe for children under 2. Consult a pediatrician BEFORE giving any medication.")
    elif age < 12:
        warnings.append("⚠️ PEDIATRIC PATIENT: Medication doses must be adjusted for children. Consult a pediatrician for appropriate dosing.")
    elif age < 18:
        warnings.append("⚠️ ADOLESCENT PATIENT: Some medications have different dosing for teens. Verify with a healthcare provider.")
    elif age >= 75:
        warnings.append("⚠️ ELDERLY PATIENT (75+): Increased risk of drug interactions and side effects. Lower doses may be required. Medical supervision recommended.")
    elif age >= 60:
        warnings.append("⚠️ SENIOR PATIENT: Monitor for drug interactions with existing medications. Regular check-ups advised.")
    
    return warnings


# ══════════════════════════════════════════════════════════════
#  DUAL RECOMMENDATION LOOKUP
# ══════════════════════════════════════════════════════════════

def get_recommendations(
    predicted_condition: str,
    age: int,
    existing_conditions: list[str],
    lifestyle_factors: list[str]
) -> dict:
    """
    Retrieve filtered pharmaceutical + herbal/lifestyle recommendations
    for a predicted condition, with contraindication safety checks.
    
    Args:
        predicted_condition: The top predicted disease name
        age: Patient age
        existing_conditions: Patient's pre-existing conditions
        lifestyle_factors: Patient's lifestyle habits
    
    Returns:
        dict: {
            "condition": str,
            "pharmaceutical": [
                {"medication": str, "usage": str, "precautions": str}, ...
            ],
            "herbal": [
                {"remedy": str, "usage": str, "precautions": str}, ...
            ],
            "contraindication_warnings": [str, ...]
        }
    """
    conditions_db = _KNOWLEDGE_BASE.get("conditions", {})
    
    # Find matching condition (case-insensitive)
    condition_data = None
    matched_name = predicted_condition
    
    for cond_name, cond_info in conditions_db.items():
        if cond_name.lower() == predicted_condition.lower():
            condition_data = cond_info
            matched_name = cond_name
            break
    
    # If no exact match, try partial match
    if condition_data is None:
        for cond_name, cond_info in conditions_db.items():
            if (predicted_condition.lower() in cond_name.lower() or
                cond_name.lower() in predicted_condition.lower()):
                condition_data = cond_info
                matched_name = cond_name
                break
    
    # If still no match, return generic guidance
    if condition_data is None:
        return _generic_recommendation(predicted_condition, age)
    
    # ── Process Pharmaceutical Recommendations ──
    all_contraindication_warnings = []
    all_contraindication_warnings.extend(_age_warnings(age))
    
    pharma_results = []
    for med in condition_data.get("pharmaceutical", []):
        # Check contraindications
        med_contras = med.get("contraindicated_conditions", [])
        contra_warnings = check_contraindications(med_contras, existing_conditions)
        all_contraindication_warnings.extend(contra_warnings)
        
        # Build precautions string (include contra warnings if any)
        precautions = med.get("precautions", "")
        if contra_warnings:
            precautions += " | " + " | ".join(contra_warnings)
        
        pharma_results.append({
            "medication": med.get("medication", ""),
            "usage": med.get("usage", ""),
            "precautions": precautions
        })
    
    # ── Process Herbal/Lifestyle Recommendations ──
    herbal_results = []
    for herb in condition_data.get("herbal", []):
        # Check contraindications
        herb_contras = herb.get("contraindicated_conditions", [])
        contra_warnings = check_contraindications(herb_contras, existing_conditions)
        all_contraindication_warnings.extend(contra_warnings)
        
        precautions = herb.get("precautions", "")
        if contra_warnings:
            precautions += " | " + " | ".join(contra_warnings)
        
        herbal_results.append({
            "remedy": herb.get("remedy", ""),
            "usage": herb.get("usage", ""),
            "precautions": precautions
        })
    
    # ── Lifestyle-Specific Additions ──
    lifestyle_tips = _lifestyle_additions(lifestyle_factors)
    if lifestyle_tips:
        for tip in lifestyle_tips:
            herbal_results.append(tip)
    
    # Deduplicate contraindication warnings
    unique_warnings = list(dict.fromkeys(all_contraindication_warnings))
    
    return {
        "condition": matched_name,
        "pharmaceutical": pharma_results,
        "herbal": herbal_results,
        "contraindication_warnings": unique_warnings
    }


def _generic_recommendation(condition: str, age: int) -> dict:
    """Fallback generic recommendation when condition isn't in knowledge base."""
    return {
        "condition": condition,
        "pharmaceutical": [
            {
                "medication": "Consult a Healthcare Provider",
                "usage": f"The condition '{condition}' requires professional medical evaluation "
                         "for appropriate medication recommendations.",
                "precautions": "Do not self-medicate. Visit a qualified doctor for proper diagnosis and treatment."
            }
        ],
        "herbal": [
            {
                "remedy": "General Wellness Support",
                "usage": "Stay hydrated (2–3 liters water daily), maintain balanced nutrition, "
                         "and ensure adequate rest (7–8 hours sleep).",
                "precautions": "These are general wellness tips and do not replace medical treatment."
            },
            {
                "remedy": "Monitor Symptoms",
                "usage": "Keep a symptom diary noting time, severity, and triggers. "
                         "Share with your healthcare provider.",
                "precautions": "Seek immediate medical attention if symptoms worsen or new symptoms appear."
            }
        ],
        "contraindication_warnings": _age_warnings(age)
    }


def _lifestyle_additions(lifestyle_factors: list[str]) -> list[dict]:
    """Generate additional herbal/lifestyle tips based on reported lifestyle factors."""
    tips = []
    
    lifestyle_lower = [f.lower() for f in (lifestyle_factors or [])]
    
    if any("smok" in f for f in lifestyle_lower):
        tips.append({
            "remedy": "Smoking Cessation Support",
            "usage": "Consider nicotine replacement therapy (patches/gum). "
                     "Smoking worsens most health conditions and delays healing.",
            "precautions": "Consult doctor before starting NRT if on heart medications."
        })
    
    if any("sedentary" in f or "no exercise" in f for f in lifestyle_lower):
        tips.append({
            "remedy": "Gradual Physical Activity",
            "usage": "Start with 15-minute daily walks and gradually increase to "
                     "30 minutes of moderate activity 5 days/week.",
            "precautions": "Consult doctor before starting exercise if you have heart or joint conditions."
        })
    
    if any("alcohol" in f or "drinking" in f for f in lifestyle_lower):
        tips.append({
            "remedy": "Alcohol Moderation",
            "usage": "Limit alcohol intake. Many medications interact with alcohol. "
                     "Stay within recommended limits (1 drink/day women, 2/day men).",
            "precautions": "Complete abstinence recommended with liver conditions or certain medications."
        })
    
    if any("stress" in f or "anxiety" in f for f in lifestyle_lower):
        tips.append({
            "remedy": "Stress Management Techniques",
            "usage": "Practice deep breathing (4-7-8 technique), progressive muscle relaxation, "
                     "or mindfulness meditation for 10 minutes daily.",
            "precautions": "Seek professional help if stress/anxiety significantly impacts daily functioning."
        })
    
    if any("poor diet" in f or "junk food" in f for f in lifestyle_lower):
        tips.append({
            "remedy": "Nutritional Improvement",
            "usage": "Increase intake of fruits, vegetables, whole grains, and lean proteins. "
                     "Reduce processed foods, refined sugar, and excess salt.",
            "precautions": "Dietary changes should be gradual. Consult a nutritionist for personalized plans."
        })
    
    return tips
