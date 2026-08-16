"""
CureBlend — Red-Flag Safety Interceptor
=========================================
Scans patient symptoms against a strict red-flag dictionary of
life-threatening conditions. If matched, returns is_emergency: True
with immediate medical action instructions BEFORE standard ML predictions.

This module acts as a critical safety gate in the assessment pipeline.
"""


# ══════════════════════════════════════════════════════════════
#  RED-FLAG KEYWORD DICTIONARY
# ══════════════════════════════════════════════════════════════
#  Organized by emergency category for structured advisory output.

RED_FLAG_RULES = {
    # ── Cardiovascular Emergencies ──
    "chest pain": {
        "category": "Cardiovascular Emergency",
        "advisory": "Possible cardiac event. Call emergency services (112/911) IMMEDIATELY. "
                     "Chew 1 aspirin (325mg) if not allergic. Do NOT lie flat — sit upright. "
                     "Do NOT drive yourself to the hospital."
    },
    "chest tightness": {
        "category": "Cardiovascular Emergency",
        "advisory": "Chest tightness may indicate angina or heart attack. "
                     "Seek emergency medical attention immediately. Avoid exertion."
    },
    "crushing chest pressure": {
        "category": "Cardiovascular Emergency",
        "advisory": "Crushing chest pressure is a hallmark of myocardial infarction. "
                     "Call emergency services NOW. Time is critical."
    },
    "irregular heartbeat": {
        "category": "Cardiovascular Emergency",
        "advisory": "Irregular heartbeat with dizziness may indicate cardiac arrhythmia. "
                     "Seek immediate medical evaluation. Avoid caffeine and stimulants."
    },

    # ── Respiratory Emergencies ──
    "shortness of breath": {
        "category": "Respiratory Emergency",
        "advisory": "Acute shortness of breath requires urgent evaluation. "
                     "Sit upright, use prescribed inhaler if available. "
                     "Call emergency services if breathing worsens rapidly."
    },
    "severe breathlessness": {
        "category": "Respiratory Emergency",
        "advisory": "Severe breathlessness is a medical emergency. "
                     "Call emergency services immediately. Keep airway open and sit upright."
    },
    "hemoptysis": {
        "category": "Respiratory Emergency",
        "advisory": "Coughing up blood (hemoptysis) requires IMMEDIATE medical attention. "
                     "Do NOT ignore. Go to the nearest emergency room NOW."
    },
    "coughing blood": {
        "category": "Respiratory Emergency",
        "advisory": "Blood in cough may indicate serious pulmonary condition. "
                     "Seek emergency care immediately. Avoid cough suppressants."
    },
    "difficulty breathing": {
        "category": "Respiratory Emergency",
        "advisory": "Difficulty breathing warrants urgent medical assessment. "
                     "If oxygen levels drop below 92%, call emergency services."
    },
    "choking": {
        "category": "Respiratory Emergency",
        "advisory": "Choking is a life-threatening emergency. Perform Heimlich maneuver. "
                     "Call emergency services immediately if obstruction persists."
    },

    # ── Neurological Emergencies ──
    "loss of consciousness": {
        "category": "Neurological Emergency",
        "advisory": "Loss of consciousness requires IMMEDIATE emergency response. "
                     "Place person in recovery position. Call emergency services. "
                     "Do NOT give food or water."
    },
    "sudden weakness": {
        "category": "Neurological Emergency (Possible Stroke)",
        "advisory": "Sudden weakness may indicate stroke. Use FAST test: "
                     "Face drooping, Arm weakness, Speech difficulty, Time to call emergency. "
                     "Every minute counts. Call emergency services NOW."
    },
    "sudden paralysis": {
        "category": "Neurological Emergency (Possible Stroke)",
        "advisory": "Sudden paralysis is a stroke indicator. Call emergency services IMMEDIATELY. "
                     "Note the time of onset. Do NOT give aspirin until confirmed by doctor."
    },
    "facial drooping": {
        "category": "Neurological Emergency (Possible Stroke)",
        "advisory": "Facial drooping is a critical stroke sign. Call emergency services NOW. "
                     "Note time symptoms started. Do NOT wait for symptoms to improve."
    },
    "slurred speech": {
        "category": "Neurological Emergency (Possible Stroke)",
        "advisory": "Slurred speech with other symptoms may indicate stroke. "
                     "FAST response is critical. Call emergency services immediately."
    },
    "severe headache sudden onset": {
        "category": "Neurological Emergency",
        "advisory": "Sudden severe headache ('thunderclap') may indicate subarachnoid hemorrhage. "
                     "This is a neurosurgical emergency. Go to ER immediately."
    },
    "seizure": {
        "category": "Neurological Emergency",
        "advisory": "During seizure: Clear area of sharp objects. Do NOT restrain or put anything in mouth. "
                     "Time the seizure. Call emergency services if seizure lasts >5 minutes."
    },
    "convulsions": {
        "category": "Neurological Emergency",
        "advisory": "Convulsions require emergency medical attention. "
                     "Protect from injury. Place in recovery position after episode. Call emergency services."
    },

    # ── Allergic / Anaphylactic Emergencies ──
    "anaphylaxis": {
        "category": "Allergic Emergency",
        "advisory": "Anaphylaxis is life-threatening. Use epinephrine auto-injector (EpiPen) IMMEDIATELY. "
                     "Call emergency services. Lie flat with legs elevated unless breathing difficulty."
    },
    "throat swelling": {
        "category": "Allergic Emergency",
        "advisory": "Throat swelling may indicate anaphylaxis. This can obstruct airway. "
                     "Use EpiPen if available. Call emergency services immediately."
    },
    "severe allergic reaction": {
        "category": "Allergic Emergency",
        "advisory": "Severe allergic reaction requires immediate epinephrine and emergency care. "
                     "Do NOT wait to see if symptoms improve."
    },

    # ── Trauma & Bleeding ──
    "uncontrolled bleeding": {
        "category": "Trauma Emergency",
        "advisory": "Apply direct firm pressure with clean cloth. Elevate injured area. "
                     "Call emergency services. Do NOT remove embedded objects."
    },
    "severe bleeding": {
        "category": "Trauma Emergency",
        "advisory": "Apply direct pressure to wound. Use tourniquet only as last resort. "
                     "Call emergency services immediately."
    },
    "head injury": {
        "category": "Trauma Emergency",
        "advisory": "Head injury with confusion, vomiting, or loss of consciousness requires "
                     "immediate emergency evaluation. Do NOT give blood thinners."
    },

    # ── Abdominal Emergencies ──
    "vomiting blood": {
        "category": "Gastrointestinal Emergency",
        "advisory": "Vomiting blood (hematemesis) is a medical emergency indicating possible "
                     "GI bleeding. Go to emergency room immediately. Do NOT eat or drink."
    },
    "blood in stool": {
        "category": "Gastrointestinal Emergency",
        "advisory": "Significant blood in stool may indicate serious GI bleeding. "
                     "Seek urgent medical evaluation. Monitor for signs of shock."
    },
    "severe abdominal pain": {
        "category": "Abdominal Emergency",
        "advisory": "Severe abdominal pain may indicate surgical emergency (appendicitis, perforation). "
                     "Do NOT take painkillers before evaluation. Go to ER immediately."
    },

    # ── Metabolic Emergencies ──
    "confusion and high fever": {
        "category": "Sepsis / Meningitis Alert",
        "advisory": "Confusion with high fever may indicate sepsis or meningitis. "
                     "This is a medical emergency. Seek immediate hospital care."
    },
    "diabetic emergency": {
        "category": "Metabolic Emergency",
        "advisory": "Diabetic emergency (hypo/hyperglycemia). If conscious: give sugar for low blood sugar. "
                     "Call emergency services. Check blood glucose if meter available."
    },
    "suicidal thoughts": {
        "category": "Mental Health Emergency",
        "advisory": "If you or someone is having suicidal thoughts, contact crisis helpline immediately. "
                     "India: iCall 9152987821 | US: 988 | UK: 116 123. You are not alone."
    }
}

# Flattened set of all red-flag keywords for fast scanning
ALL_RED_FLAG_KEYWORDS = set(RED_FLAG_RULES.keys())


# ══════════════════════════════════════════════════════════════
#  SAFETY SCAN FUNCTION
# ══════════════════════════════════════════════════════════════

def scan_for_emergencies(
    symptom_text: str,
    symptoms: list[str]
) -> dict:
    """
    Scan patient input for red-flag emergency keywords.
    
    Args:
        symptom_text: Free-form natural language symptom description
        symptoms: List of structured symptom chip strings
    
    Returns:
        dict: {
            "is_emergency": bool,
            "matched_flags": [str, ...],
            "advisory": str  (combined emergency instructions)
        }
    """
    # Normalize input for matching
    combined_input = symptom_text.lower()
    if symptoms:
        combined_input += " " + " ".join([s.lower().replace("_", " ") for s in symptoms])
    
    matched_flags = []
    advisories = []
    categories = set()
    
    # Scan against all red-flag keywords
    for keyword, rule in RED_FLAG_RULES.items():
        if keyword in combined_input:
            matched_flags.append(keyword)
            categories.add(rule["category"])
            advisories.append(rule["advisory"])
    
    if not matched_flags:
        return {
            "is_emergency": False,
            "matched_flags": [],
            "advisory": ""
        }
    
    # ── Build combined emergency advisory ──
    header = (
        "⚠️ EMERGENCY ALERT ⚠️\n"
        f"Red-flag symptoms detected: {', '.join(matched_flags)}\n"
        f"Categories: {', '.join(sorted(categories))}\n\n"
        "IMMEDIATE ACTIONS:\n"
    )
    
    # Deduplicate advisories while preserving order
    unique_advisories = list(dict.fromkeys(advisories))
    numbered_steps = "\n".join(
        [f"{i+1}. {adv}" for i, adv in enumerate(unique_advisories)]
    )
    
    footer = (
        "\n\n⚠️ DISCLAIMER: This is an AI-generated emergency alert. "
        "It is NOT a substitute for professional medical evaluation. "
        "Please call your local emergency number IMMEDIATELY."
    )
    
    return {
        "is_emergency": True,
        "matched_flags": matched_flags,
        "advisory": header + numbered_steps + footer
    }


def check_contraindications(
    medication_contraindications: list[str],
    patient_conditions: list[str]
) -> list[str]:
    """
    Cross-check medication contraindications against patient's existing conditions.
    
    Args:
        medication_contraindications: List of condition keys this medication is contraindicated for
        patient_conditions: Patient's reported existing conditions
    
    Returns:
        List of matched contraindication warnings
    """
    warnings = []
    
    # Normalize patient conditions for matching
    normalized_patient = [c.lower().replace(" ", "_") for c in patient_conditions]
    
    for contraindication in medication_contraindications:
        normalized_contra = contraindication.lower().replace(" ", "_")
        
        # Check for exact match or partial match
        for patient_cond in normalized_patient:
            if (normalized_contra in patient_cond or
                patient_cond in normalized_contra or
                normalized_contra == patient_cond):
                warnings.append(
                    f"⚠️ CONTRAINDICATED: This remedy may be unsafe with your "
                    f"condition '{patient_cond.replace('_', ' ')}'. Consult your doctor."
                )
                break
    
    return warnings
