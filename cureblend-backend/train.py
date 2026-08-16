"""
CureBlend — High-Precision Clinical Model Training Pipeline (50 Conditions)
============================================================================
Trains an XGBoost multi-class classifier with TF-IDF text features,
multi-hot encoded symptom chips, and scaled demographic features.
Covers all 50 medical conditions with clinical symptom profiles and calibrated probabilities.

Usage:
    python train.py
"""

import sys
import re
import json
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

# ── NLTK Setup ────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def preprocess_text(text: str) -> str:
    """Lowercasing, alphanumeric cleaning, stopword removal, and lemmatization."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    cleaned = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 1
    ]
    return " ".join(cleaned)


# ══════════════════════════════════════════════════════════════
#  CLINICAL PROFILES FOR ALL 50 CONDITIONS
# ══════════════════════════════════════════════════════════════

CLINICAL_PROFILES = {
    "Common Cold": {
        "chips": ["sneezing", "runny_nose", "nasal_congestion", "sore_throat", "mild_cough", "low_grade_fever", "mild_headache", "cough", "fever", "headache"],
        "keywords": ["sneezing", "runny nose", "stuffy nose", "nasal congestion", "scratchy sore throat", "mild cough", "watery eyes", "low grade fever"],
        "templates": [
            "I have a runny nose, continuous sneezing, mild sore throat, and nasal congestion for 2 days.",
            "Stuffy nose with clear mucus discharge, sneezing, mild headache, and scratchy throat.",
            "Woke up with head congestion, sneezing fits, runny nose, and feeling slightly feverish.",
            "Catching a cold with constant sneezing, watery eyes, blocked nose, and mild throat irritation.",
        ]
    },
    "Influenza": {
        "chips": ["high_fever", "fever", "body_ache", "muscle_pain", "fatigue", "chills", "dry_cough", "cough", "headache", "sore_throat", "weakness"],
        "keywords": ["high fever", "body aches", "muscle pain", "shivering chills", "severe fatigue", "dry cough", "intense headache", "exhaustion", "sudden high temperature"],
        "templates": [
            "Sudden onset of high fever of 103F with severe body aches, shivering chills, and extreme fatigue.",
            "Whole body hurts, high grade fever since yesterday with persistent dry cough and debilitating tiredness.",
            "Severe muscle pain in back and legs, shivering, high fever, headache, and complete exhaustion.",
            "High temperature, shaking chills, dry hacking cough, painful body ache, and inability to get out of bed.",
            "Severe body aches with high fever of 103F and extreme fatigue and chills.",
        ]
    },
    "COVID-19": {
        "chips": ["fever", "high_fever", "dry_cough", "cough", "loss_of_taste", "loss_of_smell", "fatigue", "shortness_of_breath", "breathlessness", "body_ache", "sore_throat", "headache"],
        "keywords": ["loss of smell", "loss of taste", "anosmia", "ageusia", "dry cough", "fever", "fatigue", "chest tightness"],
        "templates": [
            "I have completely lost my sense of smell and taste, along with fever, dry cough, and fatigue.",
            "Persistent dry cough, mild fever, fatigue, and I cannot smell or taste any food for the past 3 days.",
            "Lost taste and smell completely, experiencing body ache, dry cough, headache, and mild breathlessness.",
            "Fever with dry cough, fatigue, sore throat, and total loss of smell and taste.",
            "Lost sense of smell and taste with persistent dry cough and body aches.",
        ]
    },
    "Pneumonia": {
        "chips": ["high_fever", "fever", "productive_cough", "cough", "yellow_sputum", "green_sputum", "shortness_of_breath", "breathlessness", "chest_pain", "chest_pain_on_breathing", "chills", "fatigue"],
        "keywords": ["cough with yellow phlegm", "green sputum", "pleuritic chest pain", "difficulty breathing", "high fever", "shivering", "crackling breath", "productive cough", "lung congestion"],
        "templates": [
            "High fever with heavy productive cough producing thick green and yellow phlegm, plus sharp chest pain when breathing.",
            "Severe shortness of breath, high fever with chills, coughing up yellowish mucus, and stabbing pain in lungs.",
            "Persistent fever, deep chest rattling cough with phlegm, breathlessness, and pleuritic chest pain.",
            "Fever, shaking chills, productive cough with dark yellow sputum, and difficulty taking deep breaths.",
            "High fever with productive cough, chills, and severe shortness of breath.",
        ]
    },
    "Bronchial Asthma": {
        "chips": ["wheezing", "shortness_of_breath", "breathlessness", "chest_tightness", "dry_cough", "cough", "nocturnal_cough", "difficulty_exhaling"],
        "keywords": ["wheezing sound", "whistling breath", "chest tightness", "shortness of breath", "asthma attack", "nocturnal cough", "breathlessness"],
        "templates": [
            "Experiencing tight chest with high pitched wheezing sound and severe shortness of breath, worse at night.",
            "Sudden attack of breathlessness with audible whistling wheeze, chest constriction, and dry coughing.",
            "Difficulty exhaling, tight bands around chest, wheezing, and coughing fits triggered by cold air.",
            "Shortness of breath with chest tightness and wheezing sound in lungs especially during exertion and nighttime.",
            "Shortness of breath with wheezing sound and persistent coughing at night.",
        ]
    },
    "COPD": {
        "chips": ["chronic_cough", "productive_cough", "cough", "chronic_breathlessness", "shortness_of_breath", "breathlessness", "wheezing", "fatigue", "chest_fullness"],
        "keywords": ["chronic productive cough", "longterm breathlessness", "smoker cough", "wheezing", "chronic sputum", "exertion dyspnea"],
        "templates": [
            "Longstanding chronic cough with daily mucus production, progressive breathlessness even on short walks, and wheezing.",
            "Chronic smoker with persistent productive morning cough, constant shortness of breath, and chest heaviness.",
            "Getting breathless on mild exertion, chronic daily phlegm cough, and constant lung tightness.",
            "Progressive difficulty breathing over months with daily sputum expectoration and fatigue in an older smoker.",
        ]
    },
    "Tuberculosis": {
        "chips": ["chronic_cough", "cough", "blood_in_sputum", "hemoptysis", "night_sweats", "sweating", "weight_loss", "low_grade_fever", "fever", "fatigue"],
        "keywords": ["cough lasting over 3 weeks", "coughing up blood", "night sweats", "unexplained weight loss", "evening fever", "loss of appetite"],
        "templates": [
            "Persistent cough for more than 4 weeks, coughing up streaks of blood, drenching night sweats, and significant weight loss.",
            "Chronic productive cough with hemoptysis, low grade fever every evening, night sweats, and extreme weight loss.",
            "Unexplained weight loss, coughing up blood stained sputum, fatigue, and waking up drenched in sweat at night.",
            "Longterm cough over a month with blood in phlegm, loss of appetite, fatigue, and night chills with sweats.",
        ]
    },
    "Dengue Fever": {
        "chips": ["high_fever", "fever", "retro_orbital_pain", "severe_joint_pain", "joint_pain", "body_ache", "muscle_pain", "skin_rash", "nausea", "vomiting", "headache", "fatigue", "weakness"],
        "keywords": ["intense eye pain", "pain behind the eyes", "breakbone joint pain", "sudden high fever", "petechial rash", "severe muscle ache"],
        "templates": [
            "Sudden high fever with excruciating pain behind the eyes (retro-orbital), severe joint and bone pain, and nausea.",
            "Breakbone fever with intense body aches, severe eye socket pain, high temperature of 104F, and small red skin rash.",
            "High fever, terrible joint and muscle pain, severe headache behind eyes, nausea, and extreme physical weakness.",
            "High grade fever, intense retro-orbital headache, aching bones, red spots on arms, and vomiting.",
            "Sudden high fever with intense bone pain, headache behind the eyes and weakness.",
        ]
    },
    "Malaria": {
        "chips": ["cyclical_fever", "high_fever", "fever", "shivering_chills", "chills", "profuse_sweating", "sweating", "headache", "nausea", "vomiting", "fatigue"],
        "keywords": ["cold stage shivering chills", "hot stage fever", "sweating stage", "cyclical fever spikes", "rigors", "body ache", "shivering with fever and sweating"],
        "templates": [
            "Experiencing intense shivering cold chills followed by high fever spikes and then profuse drenching sweats.",
            "Cyclical episodes of high fever with uncontrollable teeth chattering chills, sweating, headache, and nausea.",
            "Fever that comes every alternate day with violent shivering, followed by high temperature and intense sweating.",
            "High fever with periodic chills and rigors, profuse sweating, body pain, and vomiting after mosquito exposure.",
            "High fever with shivering and cold chills, sweating, and nausea.",
        ]
    },
    "Typhoid Fever": {
        "chips": ["step_ladder_fever", "high_fever", "fever", "abdominal_pain", "stomach_pain", "headache", "fatigue", "loss_of_appetite", "constipation", "rose_spots"],
        "keywords": ["prolonged step ladder fever", "continuous high fever", "abdominal tenderness", "pea soup diarrhea or constipation", "weakness"],
        "templates": [
            "Gradually rising step-ladder high fever for 8 days with stomach pain, severe headache, extreme weakness, and poor appetite.",
            "Continuous high fever for over a week with abdominal cramping, constipation, tongue coating, and extreme lethargy.",
            "High sustained fever with loss of appetite, headache, abdominal discomfort, and faint rose colored spots on abdomen.",
            "Prolonged fever not going away, stomach ache, fatigue, dry cough, and malaise after consuming outside street food.",
        ]
    },
    "Gastroenteritis": {
        "chips": ["watery_diarrhea", "diarrhea", "vomiting", "nausea", "abdominal_cramps", "abdominal_pain", "stomach_pain", "low_grade_fever", "fever", "dehydration"],
        "keywords": ["frequent watery loose stools", "stomach flu", "vomiting", "nausea", "abdominal cramps", "food poisoning", "dehydration"],
        "templates": [
            "Severe watery diarrhea occurring 6 times today with persistent vomiting, stomach cramps, and dehydration.",
            "Food poisoning symptoms with sudden watery loose motions, nausea, vomiting, and abdominal cramping.",
            "Frequent loose watery stools, stomach cramps, nausea, low grade fever, and extreme thirst from dehydration.",
            "Constant vomiting and diarrhea after eating spoiled food, accompanied by severe abdominal pain and weakness.",
        ]
    },
    "GERD": {
        "chips": ["heartburn", "acid_reflux", "acidity", "sour_taste", "chest_burning", "regurgitation", "indigestion", "bloating", "stomach_pain"],
        "keywords": ["burning sensation behind breastbone", "acid coming up throat", "sour regurgitation", "heartburn after meals", "pyrosis", "worse lying down"],
        "templates": [
            "Severe burning sensation behind breastbone and in throat after eating spicy food, with sour acid regurgitation.",
            "Heartburn and acid reflux worsening when lying flat at night, accompanied by sour taste in mouth and bloating.",
            "Constant acid reflux rising into esophagus, burning chest discomfort after meals, and frequent belching.",
            "Retrosternal chest burning, acid regurgitation in back of throat, and indigestion relieved by antacids.",
            "Severe heartburn and acid reflux coming up the throat after meals.",
        ]
    },
    "Peptic Ulcer Disease": {
        "chips": ["epigastric_pain", "burning_stomach_pain", "abdominal_pain", "stomach_pain", "indigestion", "bloating", "nausea", "loss_of_appetite"],
        "keywords": ["burning upper abdominal pain", "pain on empty stomach", "pain relieved by food", "epigastric burning", "indigestion", "belching"],
        "templates": [
            "Gnawing, burning pain in the upper abdomen that gets worse on an empty stomach and improves temporarily after eating.",
            "Severe epigastric burning pain between meals and during the night, with nausea, bloating, and loss of appetite.",
            "Burning stomach ache in the upper center abdomen, indigestion, feeling full quickly, and mild nausea.",
            "Sharp burning upper gastric pain relieved by antacids or drinking milk, accompanied by bloating and belching.",
            "Burning abdominal pain and nausea aggravated on empty stomach.",
        ]
    },
    "Appendicitis": {
        "chips": ["right_lower_quadrant_pain", "periumbilical_pain", "abdominal_pain", "stomach_pain", "fever", "nausea", "vomiting", "loss_of_appetite"],
        "keywords": ["sharp pain right lower abdomen", "McBurney point pain", "pain shifted from belly button to right groin", "rebound tenderness", "nausea"],
        "templates": [
            "Sharp severe pain that started around my belly button and has moved to the right lower abdomen with fever and nausea.",
            "Excruciating right lower quadrant abdominal pain, worsening with walking or coughing, accompanied by vomiting and fever.",
            "Sudden severe pain in right lower side of stomach, loss of appetite, low grade fever, and intense tenderness on touching.",
            "Acute right lower abdominal cramping, unable to stand straight due to sharp pain, nausea, and elevated temperature.",
        ]
    },
    "Gallstone": {
        "chips": ["right_upper_quadrant_pain", "biliary_colic", "abdominal_pain", "stomach_pain", "pain_radiating_to_shoulder", "nausea", "vomiting", "indigestion"],
        "keywords": ["sharp right upper abdominal pain", "pain radiating to right scapula shoulder", "pain after fatty meal", "biliary colic", "nausea"],
        "templates": [
            "Severe sharp pain under right ribs radiating to right shoulder blade, occurring 1 hour after eating a fatty meal with nausea.",
            "Intense colicky pain in right upper quadrant of abdomen spreading to back, with vomiting and severe indigestion.",
            "Sudden severe right sided upper abdominal ache lasting several hours after greasy food, accompanied by nausea.",
            "Episodes of severe pain beneath right ribcage radiating towards right shoulder with bloating and vomiting.",
        ]
    },
    "Diverticulitis": {
        "chips": ["left_lower_quadrant_pain", "abdominal_tenderness", "abdominal_pain", "stomach_pain", "fever", "nausea", "constipation", "bloating"],
        "keywords": ["sharp pain left lower abdomen", "left sided stomach pain", "fever", "constipation", "abdominal tenderness", "bloating"],
        "templates": [
            "Constant sharp pain in the left lower abdomen with mild fever, nausea, abdominal bloating, and constipation.",
            "Severe tenderness and cramping in lower left side of stomach, fever, and change in bowel habits.",
            "Persistent cramping pain in lower left quadrant, feeling bloated with low grade fever and nausea.",
            "Left lower abdominal ache that is getting progressively worse, with localized tenderness, fever, and constipation.",
        ]
    },
    "Kidney Stones": {
        "chips": ["severe_flank_pain", "back_pain", "groin_pain", "blood_in_urine", "hematuria", "painful_urination", "burning_urination", "nausea", "vomiting"],
        "keywords": ["excruciating side back pain", "colicky flank pain radiating to groin", "blood in urine pink urine", "painful burning urination", "renal colic"],
        "templates": [
            "Excruciating sharp pain in the lower back and flank that radiates down to the groin, with visible blood in pinkish urine.",
            "Sudden unbearable colicky side pain coming in severe waves, accompanied by nausea, vomiting, and burning urination.",
            "Severe flank pain shooting into lower abdomen and groin, painful urination with reddish urine, and severe restlessness.",
            "Intense renal colic pain in back, nausea, vomiting, and frequent painful urination with hematuria.",
        ]
    },
    "Urinary Tract Infection": {
        "chips": ["burning_urination", "burning_micturition", "dysuria", "painful_urination", "frequent_urination", "urgency", "pelvic_pain", "bladder_discomfort", "cloudy_urine", "foul_smell_of_urine", "foul_smelling_urine"],
        "keywords": ["burning sensation when peeing", "dysuria", "frequent urge to urinate", "cloudy foul smelling urine", "lower pelvic pain", "bladder pain", "burning urination"],
        "templates": [
            "Intense burning sensation while urinating, frequent urge to pee every few minutes, and pelvic pressure.",
            "Painful burning urination (dysuria), cloudy foul-smelling urine, and constant lower abdominal discomfort.",
            "Frequent and urgent urination with sharp burning pain at the end of micturition, and suprapubic aching.",
            "Burning when passing urine, having to go constantly with only drops coming out, and lower pelvic pain.",
            "Burning sensation when urinating with frequent urination and pelvic discomfort.",
        ]
    },
    "Migraine": {
        "chips": ["throbbing_headache", "headache", "unilateral_headache", "photophobia", "phonophobia", "light_sensitivity", "nausea", "vomiting", "visual_aura", "visual_disturbances", "acidity"],
        "keywords": ["one sided throbbing pulsating headache", "light sensitivity", "sound sensitivity", "visual zigzag aura", "nausea", "migraine attack"],
        "templates": [
            "Severe throbbing pulsating headache on one side of my head with intense sensitivity to light and sound, plus nausea.",
            "Unilateral pounding headache around right eye and temple, preceded by flashing lights (visual aura), with nausea.",
            "Debilitating one-sided throbbing head pain, made worse by light and noise, accompanied by vomiting.",
            "Severe pulsating headache on the left side, feeling nauseous and needing to lie in a dark, quiet room.",
            "One sided throbbing headache with severe nausea and sensitivity to light.",
        ]
    },
    "Hypertension": {
        "chips": ["high_blood_pressure", "morning_headache", "headache", "dizziness", "blurry_vision", "palpitations", "chest_tightness"],
        "keywords": ["elevated blood pressure", "occipital morning headache", "dizziness", "pounding in ears", "blurred vision", "palpitations"],
        "templates": [
            "Waking up with throbbing headaches at the back of the head, dizziness, and blood pressure reading above 160/100.",
            "Frequent morning headaches, feeling dizzy and lightheaded, with heart palpitations and elevated blood pressure.",
            "Pounding sensation in the head and neck, blurred vision episodes, dizziness, and high BP readings.",
            "Occipital head pressure in mornings, fatigue, dizziness, and feeling flushed with high blood pressure.",
        ]
    },
    "Heart Attack": {
        "chips": ["crushing_chest_pain", "chest_pain", "chest_pressure", "pain_radiating_to_left_arm", "pain_radiating_to_jaw", "shortness_of_breath", "breathlessness", "cold_sweat", "sweating", "diaphoresis", "dizziness"],
        "keywords": ["crushing central chest pressure", "elephant on chest", "pain radiating down left arm", "pain to jaw", "cold sweating", "severe breathlessness"],
        "templates": [
            "Severe crushing pressure in the center of my chest radiating down my left arm and jaw, with shortness of breath and cold sweat.",
            "Heavy squeezing chest pain like an elephant sitting on my chest, sudden breathlessness, dizziness, and profuse cold sweating.",
            "Acute substernal chest tightness radiating to neck and shoulder, extreme difficulty breathing, nausea, and diaphoresis.",
            "Sudden intense crushing chest pain, radiating to left arm, lightheadedness, and cold clammy skin.",
        ]
    },
    "Heart Failure": {
        "chips": ["shortness_of_breath_lying_flat", "orthopnea", "shortness_of_breath", "breathlessness", "leg_swelling", "pedal_edema", "fatigue", "rapid_weight_gain", "chronic_cough", "cough"],
        "keywords": ["shortness of breath when lying flat", "need 3 pillows to sleep", "swollen feet and ankles", "pedal edema", "extreme fatigue"],
        "templates": [
            "Severe shortness of breath when lying flat on bed requiring 3 pillows to sleep (orthopnea), with swollen ankles and legs.",
            "Progressive breathlessness, swollen feet and lower legs leaving pit marks, fatigue, and waking up breathless at night.",
            "Bilateral leg and ankle swelling, extreme tiredness, unable to climb stairs without gasping for breath.",
            "Shortness of breath on mild exertion, fluid retention in legs, persistent cough with white frothy sputum, and fatigue.",
        ]
    },
    "Diabetes Type 2": {
        "chips": ["frequent_urination", "polyuria", "excessive_thirst", "polydipsia", "excessive_hunger", "increased_appetite", "unexplained_weight_loss", "fatigue", "blurry_vision"],
        "keywords": ["peeing very frequently at night", "extreme unquenchable thirst", "always hungry", "unexplained weight loss", "blurry vision", "fatigue"],
        "templates": [
            "Constantly thirsty with unquenchable thirst, urinating multiple times throughout the night, fatigue, and blurry vision.",
            "Excessive urination (polyuria), drinking liters of water daily (polydipsia), increased hunger, and unexplained weight loss.",
            "Feeling exhausted all the time with frequent urination, extreme dry mouth and thirst, and slow healing cuts on feet.",
            "High blood sugar symptoms including constant thirst, frequent nighttime urination, blurry eyesight, and fatigue.",
            "Frequent urination especially at night, extreme thirst and excessive hunger.",
        ]
    },
    "Hypoglycemia": {
        "chips": ["shakiness", "tremors", "cold_sweat", "sweating", "dizziness", "rapid_heartbeat", "palpitations", "confusion", "extreme_hunger", "irritability", "weakness"],
        "keywords": ["sudden shaking hands", "cold sweat", "fast heartbeat palpitations", "dizzy and confused", "extreme sudden hunger", "low blood sugar"],
        "templates": [
            "Sudden onset of severe hand tremors, cold clammy sweat, dizziness, rapid heart pounding, and extreme hunger.",
            "Feeling shaky, confused, sweating profusely with racing heartbeat, and feeling faint before eating.",
            "Sudden low blood sugar episode with trembling, dizziness, pale sweaty skin, weakness, and blurred vision.",
            "Shakiness in hands, fast heart rate, sudden anxiety, cold sweats, and feeling lightheaded.",
        ]
    },
    "Hyperthyroidism": {
        "chips": ["rapid_weight_loss", "palpitations", "heat_intolerance", "hand_tremors", "tremors", "anxiety", "insomnia", "excessive_sweating", "sweating"],
        "keywords": ["unintentional weight loss despite high appetite", "racing heartbeat", "cannot tolerate heat", "shaky trembling hands", "nervousness"],
        "templates": [
            "Losing weight rapidly despite eating more than usual, experiencing rapid heart palpitations, heat intolerance, and shaky hands.",
            "Racing heartbeat, excessive sweating, feeling nervous and jittery all the time, hand tremors, and poor sleep.",
            "Weight loss, heat intolerance with profuse sweating, trembling fingers, anxiety, and bulging or irritated eyes.",
            "Constant palpitations, feeling hot when others are cold, unexplained weight loss, diarrhea, and restlessness.",
        ]
    },
    "Hypothyroidism": {
        "chips": ["weight_gain", "extreme_fatigue", "fatigue", "cold_intolerance", "dry_skin", "constipation", "hair_loss", "puffy_face", "weakness"],
        "keywords": ["unexplained weight gain", "constant fatigue and sluggishness", "cannot tolerate cold weather", "dry skin", "hair thinning", "constipation"],
        "templates": [
            "Gaining weight without change in diet, feeling exhausted and sluggish constantly, intolerant to cold, with dry skin.",
            "Extreme fatigue, constipation, hair thinning, feeling cold all the time, and puffy swelling in face and eyes.",
            "Low energy, weight gain, brittle dry hair, dry flaky skin, muscle aches, and mental brain fog.",
            "Persistent tiredness, constipation, sensitivity to cold temperatures, weight gain, and hoarse voice.",
        ]
    },
    "Anemia": {
        "chips": ["extreme_fatigue", "fatigue", "weakness", "pale_skin", "pallor", "dizziness", "shortness_of_breath", "breathlessness", "cold_hands", "brittle_nails"],
        "keywords": ["pale skin and conjunctiva", "feeling exhausted with minor activity", "dizziness on standing", "cold extremities", "brittle spoon nails"],
        "templates": [
            "Extreme fatigue and weakness, skin and inner eyelids look pale, dizziness when standing up, and shortness of breath.",
            "Constant exhaustion, pale complexion, cold hands and feet, dizziness, and feeling lightheaded with brittle nails.",
            "Severe lack of energy, pale skin, shortness of breath upon climbing stairs, headache, and heart racing.",
            "Feeling faint, very pale face, chronic tiredness, brittle spoon-shaped nails, and cold extremities.",
        ]
    },
    "Hepatitis": {
        "chips": ["jaundice", "yellow_eyes", "dark_urine", "pale_stool", "fatigue", "right_upper_quadrant_pain", "abdominal_pain", "stomach_pain", "nausea", "loss_of_appetite"],
        "keywords": ["yellowing of skin and eyes", "dark brown tea colored urine", "clay colored stool", "right upper quadrant liver pain", "jaundice", "nausea"],
        "templates": [
            "Yellowing of eyes and skin (jaundice), dark tea-colored urine, pale clay-colored stools, and dull ache in right upper abdomen.",
            "Severe fatigue, yellow sclera in eyes, loss of appetite, nausea, and right-sided upper stomach discomfort.",
            "Jaundice with itchy skin, dark urine, light stools, extreme weakness, and elevated liver enzymes.",
            "Yellow skin, loss of appetite, nausea, vomiting, dark urine, and aching discomfort over the liver area.",
        ]
    },
    "Jaundice": {
        "chips": ["yellow_skin", "yellow_eyes", "scleral_icterus", "dark_urine", "pale_stool", "skin_itching", "itching", "fatigue"],
        "keywords": ["yellow skin discoloration", "yellow eyes scleral icterus", "dark tea colored urine", "pale stools", "pruritus itching"],
        "templates": [
            "Noticeable yellow discoloration of skin and whites of eyes, passing dark brown urine and pale colored stools.",
            "Yellow eyes (scleral icterus), persistent generalized skin itching, dark urine, fatigue, and loss of appetite.",
            "Skin and eyes turned yellow over the past week, with deep yellow-brown urine and pale stools.",
            "Yellowing of skin, dark urine, intense itching all over the body, fatigue, and mild abdominal discomfort.",
        ]
    },
    "Arthritis": {
        "chips": ["joint_pain", "joint_stiffness", "joint_swelling", "morning_stiffness", "reduced_mobility"],
        "keywords": ["joint pain", "morning stiffness lasting over an hour", "swollen painful joints", "difficulty moving fingers/knees", "arthralgia"],
        "templates": [
            "Severe joint pain and stiffness in knees and hands, especially for the first hour after waking up in the morning.",
            "Swollen, warm, and painful joints in fingers and wrists, with reduced range of motion and morning stiffness.",
            "Aching pain in both knees when walking or standing, with joint creaking sounds and stiffness after resting.",
            "Chronic joint swelling, tenderness, and stiffness in multiple joints making daily tasks painful.",
        ]
    },
    "Gout": {
        "chips": ["big_toe_pain", "severe_joint_pain", "joint_pain", "joint_redness", "joint_swelling", "sudden_nocturnal_pain"],
        "keywords": ["excruciating big toe pain", "swollen red hot joint", "woke up at night with intense toe pain", "podagra", "cannot touch bedsheet to toe"],
        "templates": [
            "Woke up in the middle of the night with excruciating, throbbing pain, redness, and swelling in my big toe joint.",
            "Intense severe pain, redness, and swelling in the base of the big toe; even the bedsheet touching it is unbearable.",
            "Sudden attack of hot, red, extremely swollen joint pain in big toe and ankle with elevated uric acid.",
            "Acute severe podagra with swollen bright red big toe joint, intense heat, and throbbing pain.",
        ]
    },
    "Cervical Spondylosis": {
        "chips": ["neck_pain", "neck_stiffness", "radiating_arm_pain", "numbness_in_hands", "tingling_in_fingers", "shoulder_pain"],
        "keywords": ["chronic neck pain", "neck stiffness", "pain shooting down arm", "numbness and tingling in fingers", "cervical spine pain"],
        "templates": [
            "Chronic neck pain and stiffness that radiates down my shoulder and right arm, with tingling and numbness in fingers.",
            "Stiff neck with difficulty turning head, radiating pain into upper back and arm, and pins-and-needles sensation in hands.",
            "Neck ache aggravated by working on computer, with shoulder numbness, hand weakness, and tingling in fingertips.",
            "Severe cervical pain radiating down both arms with morning neck stiffness and loss of grip strength.",
        ]
    },
    "Allergic Rhinitis": {
        "chips": ["sneezing", "itchy_nose", "itchy_eyes", "watery_eyes", "nasal_congestion", "clear_runny_nose", "runny_nose", "post_nasal_drip"],
        "keywords": ["bouts of sneezing", "itchy watery eyes", "clear watery runny nose", "hay fever", "seasonal allergies", "nasal itching"],
        "templates": [
            "Non-stop bouts of sneezing, extremely itchy and watery eyes, clear runny nose, and itchy palate/throat.",
            "Seasonal hay fever symptoms with continuous sneezing, nasal congestion, itchy watery eyes, and clear nasal drip.",
            "Itchy nose and eyes, sudden explosive sneezing fits when exposed to dust and pollen, with clear runny discharge.",
            "Watery itchy eyes, itchy roof of mouth, sneezing, and persistent clear nasal congestion.",
        ]
    },
    "Allergy": {
        "chips": ["skin_rash", "hives", "urticaria", "itching", "red_welts", "facial_swelling"],
        "keywords": ["itchy red raised hives", "skin welts", "generalized itching", "facial swelling", "allergic reaction rash"],
        "templates": [
            "Sudden appearance of red, raised, extremely itchy hives and welts all over the body after eating seafood.",
            "Generalized skin itching with red raised rash (urticaria) and mild swelling around eyes and lips.",
            "Widespread itchy allergic rash with red patches and intense skin irritation.",
            "Sudden itchy hives and wheals across chest and arms after contact with an allergen.",
        ]
    },
    "Eczema": {
        "chips": ["dry_skin", "itching", "red_scaly_patches", "skin_rash", "cracked_skin", "skin_flaking", "lichenification"],
        "keywords": ["intense itchy dry patches", "red cracked skin", "skin flaking in elbows and knees", "atopic dermatitis", "bleeding from itching"],
        "templates": [
            "Very dry, red, scaly, and intensely itchy skin patches inside elbow creases and behind knees.",
            "Chronic itchy dermatitis with cracked, bleeding, and dry thickened skin on hands and arms.",
            "Severe itching with red inflamed patches of skin that flake and ooze when scratched.",
            "Dry, scaly, itchy rash on face and neck with intense urge to scratch, especially at night.",
        ]
    },
    "Psoriasis": {
        "chips": ["red_plaques", "silvery_scales", "dry_skin_patches", "skin_itching", "itching", "skin_rash", "pitted_nails", "joint_pain"],
        "keywords": ["well demarcated red plaques", "silvery white scales", "scaling on elbows and knees", "scalp scaling", "pitted nails"],
        "templates": [
            "Thick red plaques covered with silvery white scales on elbows, knees, and scalp with itching and flaking.",
            "Red scaly raised skin patches with silvery crusts on lower back and joints, with tiny pits on fingernails.",
            "Dry itchy red skin plaques shedding silver scales on knees and scalp with occasional skin cracking.",
            "Well-defined red patches with thick silvery scales that itch and bleed when scales are removed.",
        ]
    },
    "Acne": {
        "chips": ["pimples", "blackheads", "whiteheads", "cystic_lesions", "oily_skin", "skin_redness", "inflamed_bumps"],
        "keywords": ["facial pimples", "blackheads and whiteheads", "painful red cysts", "oily face", "breakouts on forehead and cheeks"],
        "templates": [
            "Painful red inflamed pimples, cysts, and blackheads on forehead, cheeks, and chin with very oily skin.",
            "Persistent acne breakouts with pus-filled pimples, whiteheads, and red scarring on face and upper back.",
            "Deep painful cystic acne lesions on jawline with oily skin and surface comedones.",
            "Severe breakouts of pimples and red blemishes across face and chest with skin inflammation.",
        ]
    },
    "Fungal Infection": {
        "chips": ["ring_shaped_rash", "skin_rash", "intense_itching", "itching", "red_peeling_skin", "macerated_skin_folds", "discolored_nails"],
        "keywords": ["ring shaped red circular rash", "intense itching in groin or feet", "tinea ringworm", "athlete's foot", "peeling skin between toes"],
        "templates": [
            "Red, circular, ring-shaped rash with raised scaly borders and intense itching on inner thighs and groin.",
            "Intense itching, redness, and peeling macerated skin between toes (athlete's foot) with foul odor.",
            "Ringworm circular red rash that is spreading outward with clearing in the center and severe itching.",
            "Red itchy fungal rash under skin folds with scaling, peeling, and irritation.",
        ]
    },
    "Chicken Pox": {
        "chips": ["itchy_blisters", "vesicular_rash", "skin_rash", "fever", "high_fever", "loss_of_appetite", "headache", "fatigue", "red_spots"],
        "keywords": ["fluid filled itchy blisters", "crops of vesicles", "red itchy spots turning into blisters", "fever and rash", "varicella"],
        "templates": [
            "High fever with intense itchy red spots all over body that quickly turned into fluid-filled blisters (vesicles).",
            "Crops of itchy blisters starting on chest and spreading to face and limbs, with fever and tiredness.",
            "Fluid-filled itchy vesicular rash, scabbing blisters, fever, loss of appetite, and body ache in young patient.",
            "Intensely itchy rash with red spots, clear fluid blisters, and scabs with moderate fever and malaise.",
        ]
    },
    "Strep Throat": {
        "chips": ["severe_sore_throat", "sore_throat", "painful_swallowing", "swollen_tonsils", "tonsillar_exudate", "swollen_lymph_nodes", "fever", "high_fever", "headache"],
        "keywords": ["sudden severe sore throat", "pain swallowing odynophagia", "white patches on tonsils", "swollen neck glands", "fever with NO cough"],
        "templates": [
            "Sudden severe throat pain making swallowing extremely painful, fever of 102F, swollen tonsils with white pus patches, and no cough.",
            "Intense sore throat, swollen tender lymph nodes in neck, bright red swollen tonsils with exudates, and fever.",
            "Severe odynophagia (painful swallowing), high fever, white spots on throat and tonsils, and headache without runny nose.",
            "Very painful sore throat with fever, swollen neck glands, and enlarged tonsils with white streaks.",
        ]
    },
    "Acute Sinusitis": {
        "chips": ["facial_pain", "sinus_pressure", "nasal_congestion", "thick_yellow_nasal_discharge", "headache", "toothache", "post_nasal_drip"],
        "keywords": ["facial pressure under eyes and forehead", "throbbing sinus headache", "thick green nasal discharge", "pain bending forward", "sinus congestion"],
        "templates": [
            "Intense throbbing facial pain and pressure under eyes and forehead, worsening when bending forward, with thick yellow nasal discharge.",
            "Severe sinus headache, blocked nose with thick discolored mucus, facial tenderness over cheeks, and upper toothache.",
            "Facial pressure over maxillary and frontal sinuses, post-nasal drip, headache, and persistent nasal obstruction.",
            "Pain and fullness in forehead and cheekbones, thick green nasal discharge, reduced smell, and fever.",
        ]
    },
    "Acute Bronchitis": {
        "chips": ["persistent_cough", "productive_cough", "cough", "chest_discomfort", "chest_tightness", "wheezing", "low_grade_fever", "fever", "fatigue", "mucus_production"],
        "keywords": ["chest cold", "hacking cough with clear or yellow phlegm", "burning chest discomfort on coughing", "mild fever", "wheeze"],
        "templates": [
            "Persistent deep hacking cough following a cold, producing yellowish mucus with chest soreness and fatigue.",
            "Constant productive cough lasting 10 days, chest burning sensation when coughing, mild wheezing, and low fever.",
            "Chest cold with heavy coughing fits, clear-to-yellow sputum, mild shortness of breath, and tiredness.",
            "Irritating bronchial cough with mucus expectoration, rib soreness from coughing, and low grade fever.",
        ]
    },
    "Conjunctivitis": {
        "chips": ["red_eyes", "eye_discharge", "eyelid_crusting", "gritty_eye_sensation", "eye_itching", "watery_eyes"],
        "keywords": ["pink eye", "bloodshot red eyes", "yellow eye crusting in morning", "gritty feeling in eyes", "sticky discharge", "itchy eyes"],
        "templates": [
            "Woke up with eyelids stuck together with crusty yellow discharge, red bloodshot eyes, and gritty burning sensation.",
            "Pink red eye with watery sticky discharge, intense itching, and eyelids swollen and crusted shut in morning.",
            "Redness in both eyes with yellowish discharge, foreign body sensation, and tearing.",
            "Bloodshot pink eyes with gritty irritation, tearing, and sticky morning discharge.",
        ]
    },
    "Hemorrhoids": {
        "chips": ["rectal_bleeding", "bright_red_blood_on_tissue", "anal_pain", "anal_itching", "swollen_anal_lump", "painful_bowel_movements"],
        "keywords": ["bright red blood on toilet paper", "painful bowel movements", "swollen painful lump around anus", "anal itching", "piles"],
        "templates": [
            "Painless bright red blood dripping into toilet bowl during bowel movement, with swollen painful lump at the anal opening.",
            "Severe anal pain and itching, with swelling around anus and streaks of bright red blood on toilet paper after passing hard stool.",
            "Painful swollen hemorrhoid lump near anus, bleeding during defecation, and discomfort while sitting.",
            "Bright red rectal bleeding with bowel movement, accompanied by anal irritation, itching, and a tender bulge.",
        ]
    },
    "Varicose Veins": {
        "chips": ["swollen_twisted_veins", "aching_legs", "leg_heaviness", "lower_leg_swelling", "night_leg_cramps", "itching_around_veins"],
        "keywords": ["enlarged blue purple twisted veins on legs", "aching heavy legs after standing", "swelling in calves and ankles", "spider veins"],
        "templates": [
            "Visible dark blue, twisted, bulging veins on calves and thighs with aching heaviness and throbbing in legs after standing.",
            "Heavy tired feeling in legs, swollen ankles at end of the day, and painful bulging veins on lower legs with night cramps.",
            "Aching and throbbing pain in lower limbs with prominent bulging surface veins and itchy dry skin over veins.",
            "Swollen twisted veins on back of legs, leg fatigue upon prolonged standing, and calf tightness.",
        ]
    },
    "Sepsis": {
        "chips": ["high_fever", "fever", "hypothermia", "severe_shivering", "chills", "rapid_heartbeat", "palpitations", "low_blood_pressure", "rapid_breathing", "shortness_of_breath", "confusion", "extreme_pain"],
        "keywords": ["severe shivering and shaking", "confusion and disorientation", "very fast heart rate", "extremely low blood pressure", "rapid shallow breathing", "clammy mottled skin"],
        "templates": [
            "Extremely high fever with violent uncontrollable shivering, racing heart rate of 130 bpm, confusion, and drop in blood pressure.",
            "Severe infection worsening rapidly with rapid shallow breathing, mottled clammy skin, disorientation, and extreme body pain.",
            "Low body temperature, severe lethargy and confusion, fast weak pulse, and inability to stand.",
            "Rapid deterioration from an infection with tachycardia, hypotension, fever with rigors, and mental confusion.",
        ]
    },
    "AIDS/HIV": {
        "chips": ["prolonged_fever", "fever", "chronic_fatigue", "fatigue", "swollen_lymph_nodes", "rapid_weight_loss", "night_sweats", "sweating", "chronic_diarrhea", "diarrhea", "oral_thrush"],
        "keywords": ["swollen lymph glands in neck and armpits", "recurrent infections", "unexplained severe weight loss", "chronic diarrhea", "night sweats", "white oral patches"],
        "templates": [
            "Persistent swollen lymph nodes in neck and groin for months, recurring fevers, drenching night sweats, and rapid weight loss.",
            "Chronic diarrhea lasting weeks, severe unexplained weight loss, white fungal patches in mouth (oral thrush), and fatigue.",
            "Frequent recurrent opportunistic infections, profound exhaustion, enlarged lymph glands, and night sweats.",
            "Unexplained chronic fever, severe weight loss, night sweats, and persistent swollen lymph nodes.",
        ]
    },
    "Drug Reaction": {
        "chips": ["sudden_skin_rash", "skin_rash", "hives", "itching", "facial_swelling", "fever_after_medication", "fever"],
        "keywords": ["sudden rash after starting new medication", "allergic drug eruption", "hives and itching", "lip swelling after antibiotic"],
        "templates": [
            "Developed widespread red itchy rash and hives all over body 2 days after starting a new antibiotic medication.",
            "Sudden allergic skin eruption with itching, red macules, and mild fever following intake of prescription pills.",
            "Skin broke out in itchy red patches and welts shortly after taking a new prescription drug.",
            "Drug-induced skin rash with generalized erythema, itching, and mild lip swelling after medication.",
        ]
    },
    "Depression": {
        "chips": ["persistent_sadness", "loss_of_interest", "anhedonia", "low_energy", "fatigue", "insomnia", "feelings_of_worthlessness", "poor_concentration"],
        "keywords": ["constant feeling of sadness and emptiness", "loss of interest in all activities", "cannot sleep or sleeping too much", "extreme fatigue", "hopelessness"],
        "templates": [
            "Persistent feelings of sadness and emptiness for over a month, complete loss of interest in hobbies (anhedonia), and extreme fatigue.",
            "Feeling hopeless and worthless, inability to concentrate at work, severe insomnia, and lack of energy to do basic tasks.",
            "Chronic low mood, loss of appetite, sleep disturbances, fatigue, and feeling disconnected from life.",
            "Deep sadness, lack of motivation, feelings of guilt and worthlessness, and difficulty sleeping or getting out of bed.",
        ]
    },
    "Anxiety": {
        "chips": ["rapid_heartbeat", "palpitations", "nervousness", "restlessness", "panic_attacks", "hyperventilation", "chest_tightness", "insomnia", "sweating", "dizziness"],
        "keywords": ["sudden panic attack", "racing heart and sweating", "constant feeling on edge", "difficulty catching breath from panic", "restlessness", "dread"],
        "templates": [
            "Sudden panic attack with racing heart palpitations, trembling, hyperventilating, and overwhelming feeling of impending dread.",
            "Constant nervousness and worrying, feeling restless and tense, heart pounding, and inability to calm down or sleep.",
            "Episodes of intense anxiety with chest tightness, shortness of breath, dizziness, and shaking hands.",
            "Severe restlessness, rapid breathing, racing pulse, difficulty concentrating, and persistent sleep problems due to worry.",
        ]
    },
}


# ══════════════════════════════════════════════════════════════
#  SYNTHESIZE ACCURATE CLINICAL TRAINING DATASET
# ══════════════════════════════════════════════════════════════

def generate_full_clinical_dataset(samples_per_condition: int = 120) -> pd.DataFrame:
    """Generate high-fidelity clinical dataset across all 50 conditions."""
    records = []

    variation_prefixes = [
        "", "I have been having ", "Experiencing ", "For the past few days, ",
        "Since yesterday, ", "Lately I noticed ", "Suffering from ",
        "Doctor, I have ", "It started with ", "Current symptoms include "
    ]
    variation_suffixes = [
        "", " and it is quite uncomfortable.", " making it hard to work.",
        " which is getting worse.", " along with general malaise.",
        " and feeling very weak.", " causing severe distress.",
        " especially in the evening.", " and I need medical advice."
    ]

    for condition, profile in CLINICAL_PROFILES.items():
        chips = profile["chips"]
        keywords = profile["keywords"]
        templates = profile["templates"]

        for i in range(samples_per_condition):
            if i < len(templates) * 5:
                base = templates[i % len(templates)]
            else:
                k_sample = random.sample(keywords, min(len(keywords), random.randint(3, 5)))
                base = ", ".join(k_sample)

            prefix = random.choice(variation_prefixes)
            suffix = random.choice(variation_suffixes)
            raw_desc = f"{prefix}{base}{suffix}".strip()

            num_chips = random.randint(min(2, len(chips)), len(chips))
            selected_chips = random.sample(chips, num_chips)

            if condition in ["Common Cold", "Influenza", "COVID-19", "Allergy", "Acne", "Migraine", "Gastroenteritis"]:
                age = random.randint(16, 45)
            elif condition in ["Heart Attack", "Heart Failure", "COPD", "Diabetes Type 2", "Hypertension", "Cervical Spondylosis", "Gout", "Varicose Veins"]:
                age = random.randint(48, 78)
            elif condition in ["Chicken Pox"]:
                age = random.randint(4, 25)
            else:
                age = random.randint(20, 65)

            records.append({
                "Disease": condition,
                "Raw_Description": raw_desc,
                "Symptoms_List": selected_chips,
                "Age": age,
                "Existing_Conditions_Count": random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0],
                "Lifestyle_Risk_Score": round(random.uniform(0.5, 4.5), 2),
            })

    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════

def train_and_export():
    print("=" * 65)
    print("  CureBlend — High-Precision Clinical Model Training (50 Conditions)")
    print("=" * 65)

    print("\n[1/6] Generating comprehensive clinical datasets...")
    df = generate_full_clinical_dataset(samples_per_condition=130)
    n_conditions = df["Disease"].nunique()
    print(f"  -> Total records: {len(df)} across {n_conditions} conditions.")

    print("\n[2/6] Preprocessing clinical narratives...")
    df["Clean_Description"] = df["Raw_Description"].apply(preprocess_text)

    all_symptom_chips = sorted(list(set(
        chip for chips in df["Symptoms_List"] for chip in chips
    )))

    print("\n[3/6] Extracting TF-IDF, Multi-Hot Chips, and Demographics...")
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    tfidf_matrix = tfidf.fit_transform(df["Clean_Description"])
    print(f"  -> TF-IDF Vocabulary Size: {tfidf_matrix.shape[1]}")

    mlb = MultiLabelBinarizer(classes=all_symptom_chips)
    mlb_matrix = mlb.fit_transform(df["Symptoms_List"])
    print(f"  -> Multi-Hot Symptom Chip Features: {mlb_matrix.shape[1]}")

    scaler = StandardScaler()
    demographics_matrix = scaler.fit_transform(
        df[["Age", "Existing_Conditions_Count", "Lifestyle_Risk_Score"]]
    )
    print(f"  -> Demographic Features: {demographics_matrix.shape[1]}")

    X = hstack([
        tfidf_matrix,
        csr_matrix(mlb_matrix),
        csr_matrix(demographics_matrix)
    ])
    print(f"  -> Final Merged Matrix Shape: {X.shape}")

    print("\n[4/6] Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Disease"])
    num_classes = len(label_encoder.classes_)
    print(f"  -> Classes ({num_classes}): {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    print("\n[5/6] Training XGBoost Multi-Class Classifier with Probability Calibration...")
    xgb_base = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        objective="multi:softprob",
        num_class=num_classes,
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

    print("\n[6/6] Evaluating model accuracy...")
    y_pred = calibrated_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n  =======================================================")
    print(f"  -> HELD-OUT TEST ACCURACY: {test_acc * 100:.2f}%")
    print(f"  =======================================================\n")

    print("Saving calibrated model artifacts to data/...")
    joblib.dump(calibrated_model, DATA_DIR / "model.joblib")
    joblib.dump(tfidf, DATA_DIR / "tfidf.joblib")
    joblib.dump(label_encoder, DATA_DIR / "label_encoder.joblib")
    joblib.dump(scaler, DATA_DIR / "scaler.joblib")
    joblib.dump(mlb, DATA_DIR / "mlb.joblib")

    for fname in ["model", "tfidf", "label_encoder", "scaler", "mlb"]:
        p = DATA_DIR / f"{fname}.joblib"
        print(f"  [OK] {fname}.joblib saved ({p.stat().st_size / 1024:.1f} KB)")

    print(f"\nTraining Complete. {num_classes} conditions fully calibrated and saved.")
    return calibrated_model


if __name__ == "__main__":
    train_and_export()
