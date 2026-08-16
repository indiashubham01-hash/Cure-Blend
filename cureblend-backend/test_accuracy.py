import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ml.pipeline import predict_conditions, get_artifacts

arts = get_artifacts()
print(f"Total Model Classes ({len(arts.label_encoder.classes_)}):")
print(list(arts.label_encoder.classes_))
print("-" * 60)

test_cases = [
    ("Influenza / Flu", ["high_fever", "body_ache", "fatigue", "dry_cough", "chills"], "Severe body aches with high fever of 103F and extreme fatigue and chills"),
    ("Migraine", ["headache", "nausea", "visual_disturbances", "acidity"], "One sided throbbing headache with severe nausea and sensitivity to light"),
    ("GERD", ["heartburn", "acidity", "stomach_pain"], "Severe heartburn and acid reflux coming up the throat after meals"),
    ("Pneumonia", ["cough", "fever", "breathlessness", "chest_pain"], "High fever with productive cough, chills and severe shortness of breath"),
    ("Urinary Tract Infection", ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine"], "Burning sensation when urinating with frequent urination and pelvic discomfort"),
    ("Dengue Fever", ["high_fever", "joint_pain", "headache", "vomiting"], "Sudden high fever with intense bone pain, headache behind the eyes and weakness"),
    ("Malaria", ["chills", "vomiting", "high_fever", "sweating"], "High fever with shivering and cold chills, sweating and nausea"),
    ("COVID-19", ["fever", "dry_cough", "loss_of_taste", "loss_of_smell"], "Lost sense of smell and taste with persistent dry cough and body aches"),
    ("Bronchial Asthma", ["breathlessness", "cough", "wheezing"], "Shortness of breath with wheezing sound and persistent coughing at night"),
    ("Diabetes Type 2", ["polyuria", "increased_appetite", "excessive_hunger", "fatigue"], "Frequent urination especially at night, extreme thirst and excessive hunger"),
    ("Peptic Ulcer Disease", ["abdominal_pain", "indigestion", "loss_of_appetite"], "Burning abdominal pain and nausea aggravated on empty stomach"),
]

correct = 0
for expected, chips, text in test_cases:
    preds = predict_conditions(symptom_text=text, symptoms=chips, top_k=3)
    top1 = preds[0]["condition"] if preds else "None"
    top1_prob = preds[0]["probability"] * 100 if preds else 0
    is_match = (top1.lower() in expected.lower() or expected.lower() in top1.lower())
    if is_match:
        correct += 1
        mark = "[PASS]"
    else:
        mark = "[FAIL]"
    
    print(f"{mark} | Expected: {expected} | Predicted: {top1} ({top1_prob:.1f}%)")
    for p in preds:
        print(f"     Rank {p['rank']}: {p['condition']} ({p['probability']*100:.1f}%)")
    print()

print("=" * 60)
print(f"Accuracy on Clinical Presets: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
