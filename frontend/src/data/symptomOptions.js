export const symptomCategories = {
  "Common & General": [
    "fever", "high fever", "fatigue", "chills", "body ache", "sweating", "weakness", "loss of appetite"
  ],
  "Respiratory & ENT": [
    "cough", "dry cough", "productive cough", "sore throat", "runny nose", "sneezing", "nasal congestion", 
    "shortness of breath", "breathlessness", "wheezing", "chest tightness", "loss of taste", "loss of smell"
  ],
  "Neurological & Head": [
    "headache", "throbbing headache", "dizziness", "light sensitivity", "visual aura", "confusion", "insomnia"
  ],
  "Digestive & Gastro": [
    "nausea", "vomiting", "abdominal pain", "heartburn", "acid reflux", "diarrhea", "indigestion", "bloating", "burning stomach pain"
  ],
  "Musculoskeletal & Skin": [
    "joint pain", "joint stiffness", "neck pain", "back pain", "skin rash", "itching", "hives", "dry skin", "pimples"
  ],
  "Urinary & Renal": [
    "burning urination", "frequent urination", "pelvic pain", "cloudy urine", "flank pain"
  ]
};

export const defaultSymptomOptions = Array.from(new Set(Object.values(symptomCategories).flat()));

export const commonLifestyleFactors = [
  "smoking", "alcohol", "sedentary", "no exercise", "poor diet", "stress", "sleep deprivation"
];

export const commonConditions = [
  "diabetes", "hypertension", "asthma", "thyroid disorder", "arthritis", "heart disease", "kidney disease", "anxiety", "depression"
];

export const samplePresets = [
  {
    name: "Flu & Viral Fever",
    badge: "Respiratory",
    symptoms: ["high fever", "dry cough", "fatigue", "body ache", "chills"],
    text: "High fever of 103F for 2 days with dry cough, shivering chills, and severe body aches.",
    age: 34,
    lifestyle: ["stress"],
    conditions: []
  },
  {
    name: "Migraine Episode",
    badge: "Neurological",
    symptoms: ["headache", "throbbing headache", "light sensitivity", "nausea"],
    text: "Severe throbbing headache on one side with intense light sensitivity, nausea, and sound sensitivity.",
    age: 28,
    lifestyle: ["sleep deprivation", "stress"],
    conditions: []
  },
  {
    name: "Acid Reflux / GERD",
    badge: "Digestive",
    symptoms: ["heartburn", "acid reflux", "bloating", "indigestion"],
    text: "Severe burning sensation in upper chest after eating, frequent sour taste, and acid regurgitation.",
    age: 42,
    lifestyle: ["poor diet", "sedentary"],
    conditions: ["hypertension"]
  },
  {
    name: "Chest Emergency (Red-Flag Demo)",
    badge: "Emergency Demo",
    isEmergencyDemo: true,
    symptoms: ["chest pain", "shortness of breath", "dizziness"],
    text: "Sudden crushing chest pain radiating to left arm and jaw with difficulty breathing and cold sweating.",
    age: 58,
    lifestyle: ["smoking"],
    conditions: ["hypertension", "diabetes"]
  }
];
