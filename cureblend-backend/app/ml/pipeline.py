"""
CureBlend — ML Inference Pipeline
===================================
Loads trained model artifacts (TF-IDF, XGBoost, LabelEncoder, Scaler, MLB)
and provides multi-disease probability prediction from symptom input.

Integrates the user's hybrid feature engineering approach:
  TF-IDF text features + Multi-Hot symptom chips + Scaled demographics
"""

import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
import joblib
from pathlib import Path

from app.config import (
    MODEL_PATH, TFIDF_PATH, LABEL_ENCODER_PATH,
    SCALER_PATH, MLB_PATH, TOP_K_PREDICTIONS
)

# ── NLTK Resources ────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ══════════════════════════════════════════════════════════════
#  TEXT PREPROCESSOR (mirrors train.py preprocessing)
# ══════════════════════════════════════════════════════════════

def preprocess_text(text: str) -> str:
    """Lowercasing, punctuation removal, stopword removal, and lemmatization."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    
    tokens = text.split()
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]
    
    return " ".join(cleaned_tokens)


# ══════════════════════════════════════════════════════════════
#  MODEL LOADER (Singleton-style lazy loading)
# ══════════════════════════════════════════════════════════════

class ModelArtifacts:
    """Lazy-loaded singleton for all ML model artifacts."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance
    
    def load(self):
        """Load all artifacts from disk if not already loaded."""
        if self._loaded:
            return
        
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {MODEL_PATH}. "
                "Run `python train.py` first to generate model files."
            )
        
        self.model = joblib.load(MODEL_PATH)
        self.tfidf = joblib.load(TFIDF_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.mlb = joblib.load(MLB_PATH)
        self._loaded = True
        print("  [OK] All model artifacts loaded successfully.")
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Global singleton
artifacts = ModelArtifacts()


def get_artifacts() -> ModelArtifacts:
    """Get the loaded model artifacts, loading them if necessary."""
    if not artifacts.is_loaded:
        artifacts.load()
    return artifacts


# ══════════════════════════════════════════════════════════════
#  PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════

def predict_conditions(
    symptom_text: str,
    symptoms: list[str],
    age: int = 30,
    existing_conditions_count: int = 0,
    lifestyle_risk_score: float = 1.0,
    top_k: int = TOP_K_PREDICTIONS
) -> list[dict]:
    """
    Run the full inference pipeline:
      1. Preprocess natural language symptom text
      2. TF-IDF vectorize the cleaned text
      3. Multi-hot encode structured symptom chips
      4. Scale demographic features
      5. Merge into final feature matrix
      6. Predict multi-class probabilities
    
    Args:
        symptom_text: Free-form symptom description
        symptoms: List of structured symptom chip strings
        age: Patient age
        existing_conditions_count: Number of pre-existing conditions
        lifestyle_risk_score: Computed lifestyle risk (0–5)
        top_k: Number of top predictions to return
    
    Returns:
        List of dicts: [{"condition": str, "probability": float, "rank": int}, ...]
    """
    arts = get_artifacts()
    
    # ── 1. Combine symptom text + chip names for richer text input ──
    combined_text = symptom_text
    if symptoms:
        # Convert chip names (e.g. "burning_urination") to natural text
        chip_text = " ".join([s.replace("_", " ") for s in symptoms])
        combined_text = f"{symptom_text} {chip_text}".strip()
    
    if not combined_text:
        combined_text = "general discomfort"
    
    # ── 2. Preprocess & TF-IDF ──
    cleaned_text = preprocess_text(combined_text)
    tfidf_features = arts.tfidf.transform([cleaned_text])
    
    # ── 3. Multi-hot encode symptom chips ──
    # Normalize chip names to match training vocabulary
    normalized_symptoms = [s.lower().replace(" ", "_") for s in symptoms]
    multi_hot = arts.mlb.transform([normalized_symptoms])
    
    # ── 4. Scale demographics ──
    demographics_df = pd.DataFrame(
        [[age, existing_conditions_count, lifestyle_risk_score]],
        columns=["Age", "Existing_Conditions_Count", "Lifestyle_Risk_Score"]
    )
    demographics = arts.scaler.transform(demographics_df)
    
    # ── 5. Merge feature matrix ──
    feature_matrix = hstack([
        tfidf_features,
        csr_matrix(multi_hot),
        csr_matrix(demographics)
    ])
    
    # ── 6. Predict probabilities ──
    probabilities = arts.model.predict_proba(feature_matrix)[0]
    
    # ── 7. Sort by probability and return top-K ──
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        condition_name = arts.label_encoder.inverse_transform([idx])[0]
        prob = float(probabilities[idx])
        results.append({
            "condition": condition_name,
            "probability": round(prob, 4),
            "rank": rank
        })
    
    return results


def get_feature_names() -> list[str]:
    """Return all feature names from the fitted TF-IDF vectorizer."""
    arts = get_artifacts()
    return list(arts.tfidf.get_feature_names_out())
