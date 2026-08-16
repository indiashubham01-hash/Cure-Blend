"""
CureBlend — SHAP / Feature Weight XAI Explainer
=================================================
Computes top-k contributing symptom features for the predicted condition
using SHAP TreeExplainer (for tree-based models) or falls back to
normalized TF-IDF feature weight attribution.

Output format:
    [{"feature": "fever", "weight": 0.45}, {"feature": "cough", "weight": 0.32}, ...]
"""

import numpy as np
from app.ml.pipeline import get_artifacts, preprocess_text
from app.config import TOP_K_FEATURES

# Attempt to import SHAP — use TF-IDF fallback if unavailable
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("  [!] SHAP not available. Using TF-IDF weight attribution fallback.")


def compute_feature_importance(
    symptom_text: str,
    symptoms: list[str],
    predicted_condition: str,
    top_k: int = TOP_K_FEATURES
) -> list[dict]:
    """
    Compute feature importance for the top predicted condition.
    
    Strategy:
      1. Try SHAP TreeExplainer on the base XGBoost estimators (if calibrated model wraps them)
      2. If SHAP fails or is not available, fallback to TF-IDF coefficient-based attribution
    
    Args:
        symptom_text: Raw symptom text from user
        symptoms: Structured symptom chips
        predicted_condition: The top predicted disease name
        top_k: Number of top features to return
    
    Returns:
        List of dicts: [{"feature": str, "weight": float}, ...]
    """
    arts = get_artifacts()
    
    # ── Combine text for feature extraction ──
    combined_text = symptom_text
    if symptoms:
        chip_text = " ".join([s.replace("_", " ") for s in symptoms])
        combined_text = f"{symptom_text} {chip_text}".strip()
    
    if not combined_text:
        combined_text = "general discomfort"
    
    cleaned_text = preprocess_text(combined_text)
    tfidf_vector = arts.tfidf.transform([cleaned_text])
    
    # Get feature names from TF-IDF vocabulary
    tfidf_feature_names = list(arts.tfidf.get_feature_names_out())
    mlb_feature_names = list(arts.mlb.classes_)
    demographic_names = ["age", "existing_conditions_count", "lifestyle_risk_score"]
    all_feature_names = tfidf_feature_names + mlb_feature_names + demographic_names
    
    # ── Attempt SHAP Explanation ──
    if SHAP_AVAILABLE:
        try:
            return _shap_explanation(
                arts, tfidf_vector, symptoms, cleaned_text,
                predicted_condition, all_feature_names, tfidf_feature_names,
                top_k
            )
        except Exception as e:
            print(f"  [!] SHAP explanation failed ({e}). Using TF-IDF fallback.")
    
    # ── Fallback: TF-IDF Weight Attribution ──
    return _tfidf_weight_attribution(
        tfidf_vector, tfidf_feature_names, symptoms, top_k
    )


def _shap_explanation(
    arts, tfidf_vector, symptoms, cleaned_text,
    predicted_condition, all_feature_names, tfidf_feature_names,
    top_k
) -> list[dict]:
    """
    Use SHAP to explain the prediction. Handles CalibratedClassifierCV
    by extracting the underlying base estimators.
    """
    import pandas as pd
    from scipy.sparse import hstack, csr_matrix
    
    # Build the full feature vector (same as pipeline.py)
    normalized_symptoms = [s.lower().replace(" ", "_") for s in symptoms]
    multi_hot = arts.mlb.transform([normalized_symptoms])
    demographics_df = pd.DataFrame([[30, 0, 1.0]], columns=["Age", "Existing_Conditions_Count", "Lifestyle_Risk_Score"])
    demographics = arts.scaler.transform(demographics_df)
    
    feature_matrix = hstack([
        tfidf_vector,
        csr_matrix(multi_hot),
        csr_matrix(demographics)
    ])
    
    # Extract the base XGBoost estimator from calibrated wrapper
    base_model = arts.model
    if hasattr(base_model, "calibrated_classifiers_"):
        # CalibratedClassifierCV wraps multiple estimators — use the first
        base_estimator = base_model.calibrated_classifiers_[0].estimator
    elif hasattr(base_model, "estimator"):
        base_estimator = base_model.estimator
    else:
        base_estimator = base_model
    
    # Get class index for predicted condition
    condition_idx = list(arts.label_encoder.classes_).index(predicted_condition)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(base_estimator)
    
    # Convert sparse to dense for SHAP (required for TreeExplainer)
    dense_features = feature_matrix.toarray()
    shap_values = explainer.shap_values(dense_features)
    
    # shap_values shape: (n_samples, n_features, n_classes) or list of arrays
    if isinstance(shap_values, list):
        # Multi-class: list of arrays, one per class
        class_shap = shap_values[condition_idx][0]  # First (only) sample
    elif len(shap_values.shape) == 3:
        class_shap = shap_values[0, :, condition_idx]
    else:
        class_shap = shap_values[0]
    
    # Get absolute importance and map to feature names
    abs_importance = np.abs(class_shap)
    
    # Truncate feature names to match SHAP output
    n_features = len(class_shap)
    feature_names = all_feature_names[:n_features]
    
    # Sort by absolute importance
    top_indices = np.argsort(abs_importance)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if abs_importance[idx] > 0.0001:  # Filter negligible features
            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            results.append({
                "feature": feature_name.replace("_", " "),
                "weight": round(float(class_shap[idx]), 4)
            })
    
    return results if results else _tfidf_weight_attribution(
        tfidf_vector, tfidf_feature_names, symptoms, top_k
    )


def _tfidf_weight_attribution(
    tfidf_vector,
    feature_names: list[str],
    symptoms: list[str],
    top_k: int
) -> list[dict]:
    """
    Fallback: Use TF-IDF feature weights as a proxy for feature importance.
    Normalizes the non-zero TF-IDF values for the input text.
    """
    # Get TF-IDF values for the input
    tfidf_array = tfidf_vector.toarray().flatten()
    
    # Get non-zero feature indices
    nonzero_indices = np.nonzero(tfidf_array)[0]
    
    if len(nonzero_indices) == 0:
        # If no TF-IDF features matched, return symptom chips as features
        return [
            {"feature": s.replace("_", " "), "weight": round(1.0 / max(len(symptoms), 1), 4)}
            for s in symptoms[:top_k]
        ]
    
    # Normalize to 0–1 range
    max_val = tfidf_array[nonzero_indices].max()
    if max_val > 0:
        normalized = tfidf_array / max_val
    else:
        normalized = tfidf_array
    
    # Sort by value descending
    sorted_indices = nonzero_indices[np.argsort(normalized[nonzero_indices])[::-1]]
    top_indices = sorted_indices[:top_k]
    
    results = []
    for idx in top_indices:
        feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        results.append({
            "feature": feature_name.replace("_", " "),
            "weight": round(float(normalized[idx]), 4)
        })
    
    # Also add symptom chips that aren't already in TF-IDF results
    existing_features = {r["feature"] for r in results}
    for sym in symptoms:
        clean_sym = sym.replace("_", " ")
        if clean_sym not in existing_features and len(results) < top_k:
            results.append({"feature": clean_sym, "weight": 0.3})
    
    return results[:top_k]
