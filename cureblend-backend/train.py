"""
CureBlend — Offline Model Training Script (v2 — Real + Augmented Data)
========================================================================
Loads REAL clinical symptom–disease datasets from CSV archives and
augments with synthetic samples for conditions not in the archives.

Trains a TF-IDF + XGBoost multi-class classification pipeline with
calibrated probability output, and exports model artifacts to data/.

Data Sources:
  - data/datasets/DiseaseAndSymptoms.csv         (41 diseases, 4920 rows)
  - data/datasets/Diseases_and_Symptoms_dataset.csv (100 diseases, 96K rows)
  - Synthetic generation for COVID-19, Influenza, etc. (not in archives)

Usage:
    python train.py

Outputs:
    data/model.joblib          - Calibrated XGBClassifier
    data/tfidf.joblib          - Fitted TfidfVectorizer
    data/label_encoder.joblib  - Fitted LabelEncoder
    data/scaler.joblib         - Fitted StandardScaler (demographic features)
    data/mlb.joblib            - Fitted MultiLabelBinarizer (symptom chips)
"""

import sys
import re
import csv
import random
import warnings
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# ── NLTK Resource Download ─────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
DATA_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  1. TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """Lowercasing, punctuation/special char removal, stopword removal, and lemmatization."""
    if not isinstance(text, str):
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
#  2. TARGET DISEASE LIST & NORMALIZATION MAP
# ══════════════════════════════════════════════════════════════

# Map from archive disease names -> our canonical names
DISEASE_NAME_MAP = {
    # Archive 1 (DiseaseAndSymptoms.csv) mappings
    "Common Cold": "Common Cold",
    "Pneumonia": "Pneumonia",
    "Bronchial Asthma": "Bronchial Asthma",
    "Tuberculosis": "Tuberculosis",
    "Dengue": "Dengue Fever",
    "Malaria": "Malaria",
    "Typhoid": "Typhoid Fever",
    "Gastroenteritis": "Gastroenteritis",
    "GERD": "GERD",
    "Hypertension": "Hypertension",
    "Diabetes": "Diabetes Type 2",
    "Migraine": "Migraine",
    "Urinary tract infection": "Urinary Tract Infection",
    "Arthritis": "Arthritis",
    "hepatitis A": "Hepatitis",
    "Hepatitis B": "Hepatitis",
    "Hepatitis C": "Hepatitis",
    "Hepatitis D": "Hepatitis",
    "Hepatitis E": "Hepatitis",
    "Alcoholic hepatitis": "Hepatitis",
    "Psoriasis": "Psoriasis",
    "Chicken pox": "Chicken Pox",
    "Fungal infection": "Fungal Infection",
    "Allergy": "Allergy",
    "Acne": "Acne",
    "Jaundice": "Jaundice",
    "Hyperthyroidism": "Hyperthyroidism",
    "Hypothyroidism": "Hypothyroidism",
    "Heart attack": "Heart Attack",
    "Varicose veins": "Varicose Veins",
    "AIDS": "AIDS/HIV",
    "Cervical spondylosis": "Cervical Spondylosis",
    "Impetigo": "Impetigo",
    "Dimorphic hemmorhoids(piles)": "Hemorrhoids",
    "Drug Reaction": "Drug Reaction",
    "Hypoglycemia": "Hypoglycemia",
    "Chronic cholestasis": "Chronic Cholestasis",
    "Peptic ulcer diseae": "Peptic Ulcer Disease",
    "(vertigo) Paroymsal  Positional Vertigo": "Vertigo",
    "Osteoarthristis": "Osteoarthritis",
    "Paralysis (brain hemorrhage)": "Paralysis",

    # Archive 2 (Diseases_and_Symptoms_dataset.csv) mappings
    "common cold": "Common Cold",
    "pneumonia": "Pneumonia",
    "asthma": "Bronchial Asthma",
    "appendicitis": "Appendicitis",
    "eczema": "Eczema",
    "urinary tract infection": "Urinary Tract Infection",
    "arthritis of the hip": "Arthritis",
    "infectious gastroenteritis": "Gastroenteritis",
    "noninfectious gastroenteritis": "Gastroenteritis",
    "conjunctivitis": "Conjunctivitis",
    "conjunctivitis due to allergy": "Conjunctivitis",
    "acute bronchitis": "Acute Bronchitis",
    "acute sinusitis": "Acute Sinusitis",
    "depression": "Depression",
    "anxiety": "Anxiety",
    "panic disorder": "Anxiety",
    "heart attack": "Heart Attack",
    "heart failure": "Heart Failure",
    "gout": "Gout",
    "strep throat": "Strep Throat",
    "sepsis": "Sepsis",
    "gallstone": "Gallstone",
    "hemorrhoids": "Hemorrhoids",
    "psoriasis": "Psoriasis",
    "hypoglycemia": "Hypoglycemia",
    "drug reaction": "Drug Reaction",
    "allergy": "Allergy",
    "seasonal allergies (hay fever)": "Allergic Rhinitis",
    "diverticulitis": "Diverticulitis",
    "chronic obstructive pulmonary disease (copd)": "COPD",
}

# Final target conditions we want in the model
TARGET_CONDITIONS = [
    "Common Cold", "Influenza", "Pneumonia", "Bronchial Asthma", "COVID-19",
    "Tuberculosis", "Dengue Fever", "Malaria", "Typhoid Fever", "Gastroenteritis",
    "GERD", "Peptic Ulcer Disease", "Appendicitis", "Hypertension", "Diabetes Type 2",
    "Anemia", "Migraine", "Allergic Rhinitis", "Urinary Tract Infection",
    "Kidney Stones", "Hepatitis", "Arthritis", "Eczema",
    # New conditions from archives
    "Conjunctivitis", "COPD", "Depression", "Anxiety", "Heart Attack",
    "Heart Failure", "Gout", "Strep Throat", "Chicken Pox", "Psoriasis",
    "Hemorrhoids", "Acute Sinusitis", "Acute Bronchitis", "Sepsis",
    "Gallstone", "Diverticulitis", "Fungal Infection", "Jaundice",
    "Hyperthyroidism", "Hypothyroidism", "Acne", "Drug Reaction",
    "Allergy", "Varicose Veins", "AIDS/HIV", "Cervical Spondylosis",
    "Hypoglycemia",
]


# ══════════════════════════════════════════════════════════════
#  3. LOAD REAL DATA FROM ARCHIVE CSVs
# ══════════════════════════════════════════════════════════════

def load_archive1_data() -> pd.DataFrame:
    """Load DiseaseAndSymptoms.csv (archive 1): 41 diseases, symptom column format."""
    csv_path = DATASETS_DIR / "DiseaseAndSymptoms.csv"
    if not csv_path.exists():
        print("  [!] DiseaseAndSymptoms.csv not found, skipping archive 1.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        disease_raw = str(row["Disease"]).strip()
        canonical = DISEASE_NAME_MAP.get(disease_raw)
        if canonical is None or canonical not in TARGET_CONDITIONS:
            continue

        # Collect non-empty symptom columns
        symptoms = []
        for col in df.columns[1:]:
            val = str(row[col]).strip()
            if val and val.lower() not in ("nan", ""):
                symptoms.append(val.strip().replace("_", " ").strip())

        if not symptoms:
            continue

        # Build natural language description from symptoms
        desc = "I am experiencing " + ", ".join(symptoms[:5])
        if len(symptoms) > 5:
            desc += " and other symptoms"

        records.append({
            "Disease": canonical,
            "Raw_Description": desc,
            "Symptoms_List": [s.replace(" ", "_").lower() for s in symptoms],
        })

    return pd.DataFrame(records)


def load_archive2_data(max_rows_per_disease: int = 150) -> pd.DataFrame:
    """Load Diseases_and_Symptoms_dataset.csv (archive 2): binary symptom matrix using pandas."""
    csv_path = DATASETS_DIR / "Diseases_and_Symptoms_dataset.csv"
    if not csv_path.exists():
        print("  [!] Diseases_and_Symptoms_dataset.csv not found, skipping archive 2.")
        return pd.DataFrame()

    # Read CSV fast using pandas
    df_raw = pd.read_csv(csv_path)
    df_raw["diseases"] = df_raw["diseases"].astype(str).str.strip()
    df_raw["canonical"] = df_raw["diseases"].map(DISEASE_NAME_MAP)
    df_raw = df_raw[df_raw["canonical"].isin(TARGET_CONDITIONS)]

    symptom_cols = [c for c in df_raw.columns if c not in ("diseases", "canonical")]

    records = []
    # Process sampled rows per disease for speed and balance
    for canonical_name, group in df_raw.groupby("canonical"):
        sample_group = group.sample(n=min(len(group), max_rows_per_disease), random_state=42)
        for _, row in sample_group.iterrows():
            # Get active symptoms
            active_mask = row[symptom_cols] == 1
            active_symptoms = [col.replace("_", " ").strip() for col in symptom_cols if active_mask[col]]
            if not active_symptoms:
                continue

            selected = random.sample(active_symptoms, min(6, len(active_symptoms)))
            desc = "I have been having " + ", ".join(selected)

            records.append({
                "Disease": canonical_name,
                "Raw_Description": desc,
                "Symptoms_List": [s.replace(" ", "_").lower() for s in active_symptoms],
            })

    return pd.DataFrame(records)



# ══════════════════════════════════════════════════════════════
#  4. SYNTHETIC DATA FOR CONDITIONS NOT IN ARCHIVES
# ══════════════════════════════════════════════════════════════

SYNTHETIC_CONDITIONS = {
    "Influenza": {
        "symptoms": ["high_fever", "body_ache", "fatigue", "headache", "chills", "dry_cough", "muscle_pain", "sore_throat"],
        "descriptions": [
            "Severe body aches with high fever of 103F and extreme fatigue and chills",
            "I have been having high fever with terrible body pain and shivering",
            "Sudden onset of fever with severe muscle aches headache and dry cough",
            "Extreme fatigue with high temperature chills and whole body is aching",
            "High fever since morning with severe headache muscle pain and dry cough",
            "I feel completely exhausted with fever chills and pain in my joints and muscles",
            "Terrible body aches with temperature of 102 and persistent dry cough",
            "Shivering with high fever severe fatigue and throat is also sore",
            "Sudden high fever with intense headache body pain and feeling very weak",
            "My whole body hurts with high grade fever chills and dry cough",
        ]
    },
    "COVID-19": {
        "symptoms": ["fever", "dry_cough", "fatigue", "loss_of_taste", "loss_of_smell", "body_ache", "sore_throat", "headache"],
        "descriptions": [
            "Fever with dry cough and I have completely lost my sense of taste and smell",
            "Lost sense of smell and taste with persistent dry cough and body aches",
            "Dry cough with fever fatigue and I cannot smell or taste anything",
            "Body aches with headache fever and my food tastes like nothing at all",
            "Persistent dry cough with complete loss of smell and feeling extremely tired",
            "High fever with dry cough severe fatigue and loss of taste sensation",
            "I cannot smell anything and have dry cough with fever and body pain",
            "Sore throat with dry cough fever and everything I eat tastes bland",
            "Extreme tiredness with fever dry cough and loss of smell for three days",
            "Body ache with dry cough and both taste and smell have disappeared",
        ]
    },
    "Anemia": {
        "symptoms": ["fatigue", "weakness", "pale_skin", "dizziness", "shortness_of_breath", "cold_hands", "brittle_nails", "headache"],
        "descriptions": [
            "Extreme fatigue and weakness with pale skin and feeling dizzy when standing",
            "I feel weak and tired all the time with pale complexion and brittle nails",
            "Constant tiredness with dizziness shortness of breath and very pale skin",
            "Weakness and fatigue with cold hands and feet and my nails break easily",
            "Feeling exhausted with pallor dizziness on standing and shortness of breath",
            "My skin looks very pale and I feel weak dizzy and short of breath",
            "Chronic fatigue with headaches pale skin and feeling cold all the time",
            "Extreme weakness with pale complexion brittle nails and dizziness",
            "I get breathless easily and feel tired with pale skin and cold extremities",
            "Persistent tiredness with dizziness headache and my skin has lost its color",
        ]
    },
    "Kidney Stones": {
        "symptoms": ["severe_flank_pain", "back_pain", "blood_in_urine", "nausea", "vomiting", "painful_urination", "groin_pain"],
        "descriptions": [
            "Excruciating pain in my side and back that comes in waves with blood in urine",
            "Severe pain in lower back radiating to groin with nausea and vomiting",
            "Intense colicky pain in flank that comes and goes with painful urination",
            "Sharp pain in side spreading to groin with blood in urine and nausea",
            "Unbearable pain in back and side with waves of pain and I vomited twice",
            "Severe flank pain radiating to lower abdomen with hematuria and nausea",
            "Pain comes in waves from back to groin with burning urination and blood",
            "Excruciating side pain with nausea vomiting and pink colored urine",
            "Sudden severe pain in lower back moving to groin with painful urination",
            "Colicky pain in flank with blood in urine nausea and cannot find comfortable position",
        ]
    },
}


def generate_synthetic_data(samples_per_condition: int = 80) -> pd.DataFrame:
    """Generate synthetic data for conditions not in the archive CSVs."""
    records = []
    variation_prefixes = [
        "", "For the past few days ", "Since last week ",
        "I have been experiencing ", "Lately ", "Recently ",
        "For about three days now ", "Since yesterday ",
        "It started suddenly ", "Gradually over a week "
    ]
    variation_suffixes = [
        "", " and it is getting worse", " and I feel very unwell",
        " and I am worried", " especially at night",
        " mostly in the morning", " that comes and goes",
        " and I cannot do my daily activities",
        " and it is affecting my sleep", " and I need help"
    ]

    for condition, cond_info in SYNTHETIC_CONDITIONS.items():
        descriptions = cond_info["descriptions"]
        symptom_chips = cond_info["symptoms"]

        for i in range(samples_per_condition):
            base_desc = descriptions[i % len(descriptions)]
            desc = random.choice(variation_prefixes) + base_desc + random.choice(variation_suffixes)

            num_symptoms = max(2, int(len(symptom_chips) * random.uniform(0.7, 1.0)))
            selected_symptoms = random.sample(symptom_chips, min(num_symptoms, len(symptom_chips)))

            records.append({
                "Disease": condition,
                "Raw_Description": desc,
                "Symptoms_List": selected_symptoms,
            })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════
#  5. DATASET BALANCING & PREPARATION
# ══════════════════════════════════════════════════════════════

def balance_dataset(df: pd.DataFrame, max_per_class: int = 200, min_per_class: int = 40) -> pd.DataFrame:
    """Balance the dataset by capping overrepresented and augmenting underrepresented classes."""
    balanced_frames = []
    for disease in df["Disease"].unique():
        subset = df[df["Disease"] == disease]
        count = len(subset)

        if count > max_per_class:
            # Downsample
            subset = subset.sample(n=max_per_class, random_state=42)
        elif count < min_per_class:
            # Upsample by repeating rows with slight variation
            repeats_needed = min_per_class - count
            augmented = subset.sample(n=repeats_needed, replace=True, random_state=42).copy()
            subset = pd.concat([subset, augmented], ignore_index=True)

        balanced_frames.append(subset)

    result = pd.concat(balanced_frames, ignore_index=True)
    return result.sample(frac=1, random_state=42).reset_index(drop=True)


def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add random demographic features for training."""
    ages = []
    cond_counts = []
    lifestyle_scores = []

    for _ in range(len(df)):
        age = random.choice([
            random.randint(18, 35),
            random.randint(36, 55),
            random.randint(56, 75),
            random.randint(2, 17),
        ])
        ages.append(age)
        cond_counts.append(random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0])
        lifestyle_scores.append(round(random.uniform(0.5, 5.0), 2))

    df["Age"] = ages
    df["Existing_Conditions_Count"] = cond_counts
    df["Lifestyle_Risk_Score"] = lifestyle_scores
    return df


# ══════════════════════════════════════════════════════════════
#  6. TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════

def train_model():
    """Main training function: loads real + synthetic data, fits transformers & model, saves artifacts."""

    print("=" * 60)
    print("  CureBlend -- Model Training Pipeline (v2)")
    print("=" * 60)

    # ── Step 1: Load Real Data ────────────────────────────────
    print("\n[1/7] Loading real datasets from archives...")

    df_archive1 = load_archive1_data()
    print(f"  -> Archive 1 (DiseaseAndSymptoms): {len(df_archive1)} rows, "
          f"{df_archive1['Disease'].nunique() if len(df_archive1) > 0 else 0} conditions")

    df_archive2 = load_archive2_data()
    print(f"  -> Archive 2 (Diseases_and_Symptoms): {len(df_archive2)} rows, "
          f"{df_archive2['Disease'].nunique() if len(df_archive2) > 0 else 0} conditions")

    # ── Step 2: Generate Synthetic Data ───────────────────────
    print("\n[2/7] Generating synthetic data for missing conditions...")
    df_synthetic = generate_synthetic_data(samples_per_condition=80)
    print(f"  -> Synthetic: {len(df_synthetic)} rows, {df_synthetic['Disease'].nunique()} conditions")

    # ── Step 3: Merge All Data ────────────────────────────────
    print("\n[3/7] Merging and balancing dataset...")
    frames = [f for f in [df_archive1, df_archive2, df_synthetic] if len(f) > 0]
    df = pd.concat(frames, ignore_index=True)
    print(f"  -> Combined: {len(df)} rows, {df['Disease'].nunique()} conditions")

    # Filter to target conditions only
    df = df[df["Disease"].isin(TARGET_CONDITIONS)]
    print(f"  -> After filtering to targets: {len(df)} rows, {df['Disease'].nunique()} conditions")

    # Balance
    df = balance_dataset(df, max_per_class=200, min_per_class=40)
    print(f"  -> After balancing: {len(df)} rows")

    # Add demographics
    df = add_demographic_features(df)

    # Print per-condition counts
    print("\n  Condition distribution:")
    for disease, count in df["Disease"].value_counts().items():
        print(f"    {disease}: {count}")

    # ── Step 4: Text Preprocessing ────────────────────────────
    print("\n[4/7] Preprocessing text descriptions...")
    df["Clean_Description"] = df["Raw_Description"].apply(preprocess_text)

    # ── Step 5: Feature Extraction ────────────────────────────
    print("\n[5/7] Extracting features (TF-IDF + Multi-Hot + Demographics)...")

    # Collect all unique symptom chips
    all_symptom_chips = sorted(list(set(
        chip for symptoms in df["Symptoms_List"] for chip in symptoms
    )))

    # 5a. TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    tfidf_features = tfidf_vectorizer.fit_transform(df["Clean_Description"])
    print(f"  -> TF-IDF features: {tfidf_features.shape[1]}")

    # 5b. Multi-Hot symptom chips
    mlb = MultiLabelBinarizer(classes=all_symptom_chips)
    multi_hot_symptoms = mlb.fit_transform(df["Symptoms_List"])
    print(f"  -> Multi-hot symptom features: {multi_hot_symptoms.shape[1]}")

    # 5c. Demographics
    scaler = StandardScaler()
    scaled_demographics = scaler.fit_transform(
        df[["Age", "Existing_Conditions_Count", "Lifestyle_Risk_Score"]]
    )
    print(f"  -> Demographic features: {scaled_demographics.shape[1]}")

    # 5d. Merge
    final_feature_matrix = hstack([
        tfidf_features,
        csr_matrix(multi_hot_symptoms),
        csr_matrix(scaled_demographics)
    ])
    print(f"  -> Final merged matrix: {final_feature_matrix.shape}")

    # ── Step 6: Label Encoding & Training ─────────────────────
    print("\n[6/7] Encoding labels and training XGBoost...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Disease"])
    n_classes = len(label_encoder.classes_)
    print(f"  -> Classes ({n_classes}): {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        final_feature_matrix, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    xgb_base = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=xgb_base,
        method="sigmoid",
        cv=3
    )
    calibrated_model.fit(X_train, y_train)

    # ── Step 7: Evaluation ────────────────────────────────────
    print("\n[7/7] Evaluating model performance...")
    y_pred = calibrated_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  -> Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    # ── Save Artifacts ────────────────────────────────────────
    print("\nSaving model artifacts to data/...")
    joblib.dump(calibrated_model, DATA_DIR / "model.joblib")
    joblib.dump(tfidf_vectorizer, DATA_DIR / "tfidf.joblib")
    joblib.dump(label_encoder, DATA_DIR / "label_encoder.joblib")
    joblib.dump(scaler, DATA_DIR / "scaler.joblib")
    joblib.dump(mlb, DATA_DIR / "mlb.joblib")

    for name in ["model", "tfidf", "label_encoder", "scaler", "mlb"]:
        path = DATA_DIR / f"{name}.joblib"
        print(f"  [OK] {name}.joblib ({path.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 60)
    print(f"  Training Complete! {n_classes} conditions trained.")
    print(f"  Test Accuracy: {accuracy:.2%}")
    print("  All artifacts saved to data/")
    print("=" * 60)

    return calibrated_model, tfidf_vectorizer, label_encoder, scaler, mlb


if __name__ == "__main__":
    train_model()
