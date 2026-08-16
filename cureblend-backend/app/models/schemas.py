"""
CureBlend Backend — Pydantic V2 Schemas
========================================
Strict request/response validation models for the FastAPI endpoints.
Covers assessment input, multi-disease predictions, SHAP explanations,
severity scoring, dual recommendations, and history records.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
#  REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════

class AssessmentRequest(BaseModel):
    """Input payload for the /api/v1/assess endpoint."""
    symptoms: list[str] = Field(
        default_factory=list,
        description="Structured symptom chips selected by the user (e.g., ['fever', 'cough', 'headache'])"
    )
    symptom_text: str = Field(
        default="",
        description="Free-form natural language symptom description from the user"
    )
    age: int = Field(
        default=30,
        ge=0,
        le=120,
        description="Patient age in years"
    )
    lifestyle_factors: list[str] = Field(
        default_factory=list,
        description="Lifestyle habits (e.g., ['smoking', 'sedentary', 'alcohol'])"
    )
    existing_conditions: list[str] = Field(
        default_factory=list,
        description="Pre-existing medical conditions (e.g., ['diabetes', 'hypertension'])"
    )


# ══════════════════════════════════════════════════════════════
#  RESPONSE SUB-SCHEMAS
# ══════════════════════════════════════════════════════════════

class ConditionProbability(BaseModel):
    """Single disease prediction with confidence score."""
    condition: str = Field(description="Predicted disease/condition name")
    probability: float = Field(ge=0.0, le=1.0, description="Calibrated probability (0–1)")
    rank: int = Field(ge=1, description="Rank position (1 = most likely)")


class FeatureWeight(BaseModel):
    """Single SHAP / feature importance entry."""
    feature: str = Field(description="Symptom or feature token")
    weight: float = Field(description="Importance weight (higher = more influential)")


class PharmaceuticalRecommendation(BaseModel):
    """Evidence-based pharmaceutical guidance."""
    medication: str = Field(description="Generic medication name")
    usage: str = Field(description="Common OTC usage instructions")
    precautions: str = Field(description="Safety precautions and warnings")


class HerbalRecommendation(BaseModel):
    """Natural / lifestyle remedy guidance."""
    remedy: str = Field(description="Herbal or lifestyle remedy name")
    usage: str = Field(description="How to use this remedy")
    precautions: str = Field(description="Contraindications or warnings")


class DualRecommendation(BaseModel):
    """Combined pharmaceutical + herbal/lifestyle guidance for a condition."""
    condition: str = Field(description="Target condition name")
    pharmaceutical: list[PharmaceuticalRecommendation] = Field(default_factory=list)
    herbal: list[HerbalRecommendation] = Field(default_factory=list)
    contraindication_warnings: list[str] = Field(
        default_factory=list,
        description="Active contraindication alerts based on patient profile"
    )


class EmergencyAlert(BaseModel):
    """Emergency / red-flag alert information."""
    is_emergency: bool = Field(default=False)
    matched_flags: list[str] = Field(default_factory=list, description="Red-flag keywords detected")
    advisory: str = Field(default="", description="Immediate action instructions")


class SeverityAssessment(BaseModel):
    """Dynamic severity scoring output."""
    score: int = Field(ge=0, le=100, description="Numeric severity score (0–100)")
    level: str = Field(description="Categorical level: Low, Moderate, High, or Critical")
    breakdown: dict = Field(
        default_factory=dict,
        description="Score component breakdown (symptom_load, age_factor, condition_risk, disease_weight)"
    )


# ══════════════════════════════════════════════════════════════
#  MAIN RESPONSE SCHEMA
# ══════════════════════════════════════════════════════════════

class AssessmentResponse(BaseModel):
    """Complete response payload for the /api/v1/assess endpoint."""
    request_id: str = Field(description="Unique identifier for this assessment")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Emergency check (evaluated first)
    emergency: EmergencyAlert
    
    # ML predictions (top-K diseases with probabilities)
    predictions: list[ConditionProbability] = Field(default_factory=list)
    
    # XAI explanation (feature importance for top prediction)
    shap_importance: list[FeatureWeight] = Field(default_factory=list)
    
    # Severity scoring
    severity: SeverityAssessment
    
    # Dual recommendations (pharma + herbal) with contraindication filtering
    recommendations: DualRecommendation
    
    # Disclaimer
    disclaimer: str = Field(
        default="This is an AI-generated health assessment for informational purposes only. "
                "It is NOT a substitute for professional medical diagnosis or treatment. "
                "Please consult a qualified healthcare provider for any medical concerns."
    )


# ══════════════════════════════════════════════════════════════
#  HISTORY SCHEMA
# ══════════════════════════════════════════════════════════════

class PredictionHistoryItem(BaseModel):
    """Schema for stored assessment records retrieved from MongoDB."""
    request_id: str
    timestamp: datetime
    input_symptoms: list[str]
    symptom_text: str
    age: int
    top_prediction: str
    confidence: float
    severity_level: str
    is_emergency: bool
    recommendations_summary: str = Field(
        default="",
        description="Brief summary of recommendations provided"
    )


class HistoryResponse(BaseModel):
    """Response wrapper for the /api/v1/history endpoint."""
    total_records: int
    records: list[PredictionHistoryItem]
