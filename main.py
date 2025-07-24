# main.py
# This file combines functionalities from app.py (Flask) and backend_app.py (FastAPI)
# into a single FastAPI application.
# It prioritizes backend_app.py's logic for medicine extraction and suggestion,
# while integrating app.py's NER and summarization.
#
# IMPORTANT CHANGES:
# 1. Switched spellchecker library to 'TextBlob' for better compatibility on Render.
# 2. Removed the 'Grammar Correction' model entirely to reduce memory usage for Render's free tier.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
import uvicorn
import re
import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    pipeline,
)
import numpy as np
import torch
from word2number import w2n
# Changed from spellchecker/pyspellchecker to TextBlob
from textblob import TextBlob
# You also need to install the TextBlob data: python -m textblob.download_corpora
from rapidfuzz import fuzz

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Medicare Unified Backend",
    description="Combined API for medical text processing, NER, summarization, and medicine prescription management.",
    version="1.0.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Adjust this in production for security.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Global Variables and Model/Data Loading ---

# Device configuration (from both app.py and backend_app.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model placeholders (will be loaded on startup)
ner_tokenizer = None
ner_model = None
ner_pipeline = None
# Removed grammar_tokenizer and grammar_model to save memory
summary_tokenizer = None
summary_model = None
summary_pipeline = None

# Medicine data and feedback storage (from backend_app.py)
MEDICINE_DATA_FILE = "medicines_combined.json" # Assuming this file is in the same directory as main.py
LOADED_MEDICINE_NAMES: List[str] = []
LOADED_MEDICINE_NAMES_LOWER_SET: set = set()
LEARNED_FEEDBACK: List[Dict] = [] # In-memory storage for feedback

FUZZY_MATCH_THRESHOLD = 65 # From backend_app.py

# Mapping for common spelled-out numbers to digits (from backend_app.py, more robust)
NUMBER_WORDS = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
    'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
    'half': '0.5' # For dosages like "half tablet"
}

# --- Startup Event: Load all models and data when the app starts ---
@app.on_event("startup")
async def load_all_models_and_data():
    global ner_tokenizer, ner_model, ner_pipeline
    # Removed global grammar_tokenizer, grammar_model
    global summary_tokenizer, summary_model, summary_pipeline
    global LOADED_MEDICINE_NAMES, LOADED_MEDICINE_NAMES_LOWER_SET

    # --- Load Biomedical NER model (from backend_app.py's robust logic) ---
    FINE_TUNED_MODEL_PATH = "./fine_tuned_biobert_model"
    GENERIC_BIOBERT_MODEL = "d4data/biomedical-ner-all"
    try:
        if os.path.exists(FINE_TUNED_MODEL_PATH):
            print(f"Attempting to load fine-tuned BioBERT tokenizer and model from local path: {FINE_TUNED_MODEL_PATH}...")
            ner_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
            ner_model = AutoModelForTokenClassification.from_pretrained(FINE_TUNED_MODEL_PATH).to(device)
            ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
            print("SUCCESS: Fine-tuned BioBERT model and tokenizer loaded.")
        else:
            print(f"INFO: Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}.")
            print(f"Attempting to download/load pre-trained NER model from Hugging Face Hub: {GENERIC_BIOBERT_MODEL}...")
            ner_tokenizer = AutoTokenizer.from_pretrained(GENERIC_BIOBERT_MODEL)
            ner_model = AutoModelForTokenClassification.from_pretrained(GENERIC_BIOBERT_MODEL).to(device)
            ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
            print(f"SUCCESS: NER model '{GENERIC_BIOBERT_MODEL}' loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load any BioBERT model. Medicine extraction will fall back to basic keyword matching and regex.")
        print(f"Detailed error: {e}")
        ner_tokenizer = None
        ner_model = None
        ner_pipeline = None

    # --- Removed Grammar Correction model loading to save memory ---
    # grammar_model_name = "vennify/t5-base-grammar-correction"
    # print(f"Loading Grammar Correction model: {grammar_model_name}...")
    # try:
    #     grammar_tokenizer = AutoTokenizer.from_pretrained(grammar_model_name)
    #     grammar_model = AutoModelForSeq2SeqLM.from_pretrained(grammar_model_name).to(device)
    #     print("SUCCESS: Grammar Correction model loaded.")
    # except Exception as e:
    #     print(f"ERROR: Failed to load Grammar Correction model: {e}")
    #     raise RuntimeError(f"Failed to load Grammar Correction model: {e}")

    # --- Load Abstractive Summarization Model (from app.py) ---
    summary_model_name = "t5-base"
    print(f"Loading Summarization model: {summary_model_name}...")
    try:
        summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
        summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name).to(device)
        summary_pipeline = pipeline(
            "summarization",
            model=summary_model,
            tokenizer=summary_tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        print("SUCCESS: Abstractive Summarization model loaded.")
    except Exception as e:
        print(f"ERROR: Failed to load Summarization model: {e}")
        raise RuntimeError(f"Failed to load Summarization model: {e}")

    # --- Load medicine data from JSON file (from backend_app.py's load_medicine_names) ---
    print(f"Loading medicine data from {MEDICINE_DATA_FILE}...")
    if not os.path.exists(MEDICINE_DATA_FILE):
        print(f"ERROR: {MEDICINE_DATA_FILE} not found in the backend directory.")
        print("Please ensure you have copied 'medicines_combined.json' from your Flutter assets to the 'medicare_backend' folder.")
        # Fallback to a small dummy list if file not found, for continued operation
        LOADED_MEDICINE_NAMES = ["Paracetamol", "Vitamin C", "Amoxicillin", "Ibuprofen", "Diphtheria Antitoxin", "Vita Ch Tablet"]
        LOADED_MEDICINE_NAMES_LOWER_SET = {name.lower() for name in LOADED_MEDICINE_NAMES}
        print("WARNING: Using a dummy medicine list due to missing JSON file.")
        return

    try:
        with open(MEDICINE_DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            names_to_load = []
            for item in data:
                if 'name' in item and isinstance(item['name'], str):
                    names_to_load.append(item['name'].strip())
                
                if 'strength' in item and isinstance(item['strength'], str):
                    strength_match = re.match(r'([A-Za-z\s]+?)(?:\s*\(?\d+.*|\s+\d+.*|$)', item['strength'].strip())
                    if strength_match:
                        core_name = strength_match.group(1).strip()
                        if core_name and core_name.lower() not in LOADED_MEDICINE_NAMES_LOWER_SET:
                            names_to_load.append(core_name)
            
            unique_names = sorted(list(set(names_to_load)))
            
            LOADED_MEDICINE_NAMES = unique_names
            LOADED_MEDICINE_NAMES_LOWER_SET = {n.lower() for n in unique_names}
            
            print(f"Successfully loaded {len(LOADED_MEDICINE_NAMES)} unique medicine names from {MEDICINE_DATA_FILE}.")
            if "paracetamol" in LOADED_MEDICINE_NAMES_LOWER_SET:
                print("DEBUG: 'Paracetamol' (lowercase) IS found in loaded medicine names set.")
            else:
                print("DEBUG: 'Paracetamol' (lowercase) NOT found in loaded medicine names set. Check JSON 'strength' field extraction.")
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse {MEDICINE_DATA_FILE}. Ensure it's valid JSON. Error: {e}")
        print("WARNING: Using a dummy medicine list due to JSON parsing error.")
        LOADED_MEDICINE_NAMES = ["Paracetamol", "Vitamin C", "Amoxicillin", "Ibuprofen", "Diphtheria Antitoxin", "Vita Ch Tablet"]
        LOADED_MEDICINE_NAMES_LOWER_SET = {name.lower() for name in LOADED_MEDICINE_NAMES}
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading {MEDICINE_DATA_FILE}: {e}")
        print("WARNING: Using a dummy medicine list due to unexpected error.")
        LOADED_MEDICINE_NAMES = ["Paracetamol", "Vitamin C", "Amoxicillin", "Ibuprofen", "Diphtheria Antitoxin", "Vita Ch Tablet"]
        LOADED_MEDICINE_NAMES_LOWER_SET = {name.lower() for name in LOADED_MEDICINE_NAMES}


# --- Pydantic Models (from backend_app.py) ---
class MedicineDetail(BaseModel): # Matches MedicinePrescription in Flutter
    name: str
    dosage: str
    duration: str
    frequency: str
    timing: str

class MedicineRequest(BaseModel):
    text: str

class MedicineResponse(BaseModel):
    name: str
    dosage: str
    duration: str
    frequency: str
    timing: str

class SuggestionRequest(BaseModel):
    input_text: str
    patient_summary: str

class SuggestionResponse(BaseModel):
    suggestion: str

class FeedbackRequest(BaseModel):
    original_text: str
    corrected_medicines: List[MedicineDetail]

# Pydantic Model for /ner endpoint's response (from app.py's expected output)
class NERResponse(BaseModel):
    entities: List[Dict]
    summary: str
    medication_prescriptions: List[Dict]


# --- Helper Functions (Consolidated from both app.py and backend_app.py) ---

# Utility function to make float/int JSON serializable (from app.py)
def convert_to_serializable(obj):
    """Recursively converts numpy types and torch tensors to standard Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj

# Spell Correction (Using TextBlob - REPLACED OLD SPELLCHECKER)
def spell_correct_text(text):
    """Corrects common spelling mistakes in the input text using TextBlob."""
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

# Removed grammar_correct function entirely to save memory.
# def grammar_correct(text):
#     """
#     Corrects grammar of the input text using the loaded T5-based model.
#     """
#     if not grammar_tokenizer or not grammar_model:
#         raise RuntimeError("Grammar Correction model not initialized.")
#     input_text = f"grammar: {text}"
#     input_ids = grammar_tokenizer.encode(input_text, return_tensors="pt").to(device)
#     outputs = grammar_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
#     corrected_text = grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return corrected_text

# Normalize Number Words (from app.py)
def normalize_number_words(text):
    """
    Normalizes common spoken number patterns (e.g., 'six fifty' to 'six hundred fifty')
    and converts number words to digits to improve extraction accuracy.
    """
    # First, handle patterns like "six fifty" to "six hundred fifty"
    text = re.sub(
        r'\b(one|two|three|four|five|six|seven|eight|nine)\s+(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b',
        r'\1 hundred \2',
        text,
        flags=re.IGNORECASE
    )
    # Then, convert all recognized number words to digits
    words = text.split()
    normalized_words = []
    for word in words:
        lower_word = word.lower()
        if lower_word in NUMBER_WORDS:
            normalized_words.append(NUMBER_WORDS[lower_word])
        else:
            normalized_words.append(word)
    return " ".join(normalized_words)

# Merge subword tokens from NER results (from app.py, renamed to fit backend_app.py's naming convention)
def _merge_ner_tokens(ner_results):
    """
    Merges tokens that are part of the same entity and handles subword tokens (##).
    Also tries to merge consecutive entities of the same type if they are close.
    Crucially, preserves start/end indices for merged tokens.
    """
    merged_entities = []
    current_entity = None

    # Sort entities by start index to ensure correct processing order
    ner_results.sort(key=lambda x: x.get('start', 0))

    for ent in ner_results:
        word = ent["word"]
        # Use 'entity_group' from raw NER results, then map to 'entity' for consistency with app.py's merge_tokens output
        entity_type = ent.get("entity_group", "UNKNOWN")
        score = ent["score"]
        start = ent.get("start")
        end = ent.get("end")

        # Check if the current token can be merged with the previous one
        if current_entity and (word.startswith("##") or \
           (entity_type == current_entity["entity"] and \
            start is not None and current_entity["end"] is not None and (start - current_entity["end"] <= 2))):
            
            # Merge word
            if word.startswith("##"):
                current_entity["word"] += word[2:]
            else:
                current_entity["word"] += " " + word
            
            # Update end and average score
            current_entity["end"] = end
            current_entity["score"] = (current_entity["score"] + score) / 2
        else:
            # If it's a new entity, add the previous one (if any) and start a new one
            if current_entity:
                merged_entities.append(current_entity)
            current_entity = {"word": word, "entity": entity_type, "score": score, "start": start, "end": end}
            
    if current_entity:
        merged_entities.append(current_entity)
        
    return merged_entities

# Extract General Advice/Recommendations (from app.py)
def extract_general_advice_via_summarization(text: str) -> list:
    """
    Generates general health advice from the text by summarizing relevant sections using the T5 summarization model.
    """
    if not summary_pipeline:
        raise RuntimeError("Summarization pipeline not initialized.")

    advice_list = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    advice_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        if any(keyword in sentence_lower for keyword in [
            "should", "must", "try to", "avoid", "get enough", "do light", "eat", "drink", "take",
            "consult", "report", "follow up", "rest", "hydrate", "exercise", "limit", "reduce", "maintain"
        ]) and not any(symptom_kw in sentence_lower for symptom_kw in [
            "palpitations", "pain", "fever", "cough", "abdominal", "symptom", "disease", "procedure", "test"
        ]):
            advice_sentences.append(sentence)

    if advice_sentences:
        advice_text_to_summarize = " ".join(advice_sentences)
        print(f"[DEBUG] Text for Advice Summarization: {advice_text_to_summarize}")
        
        generated_advice_summary = summary_pipeline(
            advice_text_to_summarize,
            max_new_tokens=50,
            min_length=10,
            do_sample=False
        )[0]['summary_text']
        
        if generated_advice_summary:
            advice_list.append(generated_advice_summary.strip())
            
    print(f"[DEBUG] Extracted Advice: {advice_list}")
    return advice_list

# Helper for converting word numbers to digits (from backend_app.py)
def word_to_num(text_segment):
    """Converts a string segment containing spelled-out numbers to digits.
    Handles simple single words and common two-word numbers like "six fifty".
    """
    text_segment_lower = text_segment.lower().strip()
    
    if text_segment_lower in NUMBER_WORDS:
        return NUMBER_WORDS[text_segment_lower]

    parts = text_segment_lower.split()
    if len(parts) == 2:
        if parts[0] in NUMBER_WORDS and parts[1] in NUMBER_WORDS:
            try:
                val1 = float(NUMBER_WORDS[parts[0]])
                val2 = float(NUMBER_WORDS[parts[1]])
                if val1 < 100 and val2 >= 10:
                    return str(int(val1 * 100 + val2))
            except ValueError:
                pass
    
    try:
        if re.match(r'^\d+(\.\d+)?$', text_segment_lower):
            return text_segment_lower
    except ValueError:
        pass

    return text_segment

# Regex extraction for dosage, duration, frequency, timing (from backend_app.py)
def _extract_dosage(text: str) -> str:
    num_pattern = r'(?:(?:\d+(?:\.\d+)?)|' + r'|'.join(NUMBER_WORDS.keys()) + r')'
    units_pattern = r'(?:mg|g|ml|mcg|unit|tablet|pill|capsule|spoon(?:ful)?|units?|tabs?|caps?|bottles?|vials?|sachets?|pouches?|drops?|puffs?|sprays?|inhalations?|patches?|ml|drops|units|tabs|caps|bottles|vials|sachets|pouches|puffs|sprays|inhalations|patches)\b'
    
    regex = re.search(rf'({num_pattern}(?:\s*{num_pattern})*\s*{units_pattern})', text, re.IGNORECASE)
    if regex:
        matched_str = regex.group(0).strip()
        num_part_match = re.search(rf'({num_pattern}(?:\s*{num_pattern})*)', matched_str, re.IGNORECASE)
        unit_part_match = re.search(units_pattern, matched_str, re.IGNORECASE)
        
        if num_part_match and unit_part_match:
            converted_value = word_to_num(num_part_match.group(0))
            return f"{converted_value} {unit_part_match.group(0).strip()}"
        return matched_str
    
    regex_just_num = re.search(rf'(\b{num_pattern}(?:\s*{num_pattern})*\b)', text, re.IGNORECASE)
    if regex_just_num:
        converted_value = word_to_num(regex_just_num.group(0))
        if not re.search(units_pattern, text, re.IGNORECASE):
            return f"{converted_value} mg" 
        return converted_value
    
    return 'N/A'

def _extract_duration(text: str) -> str:
    num_pattern = r'(?:(?:\d+(?:\.\d+)?)|' + r'|'.join(NUMBER_WORDS.keys()) + r')'
    duration_units = r'(?:day|week|month|year|hr|hour)s?'
    regex = re.search(rf'(?:for\s+)?({num_pattern}(?:\s*{num_pattern})*\s*{duration_units})\b', text, re.IGNORECASE)
    if regex:
        matched_str = regex.group(0).strip()
        num_part_match = re.search(rf'({num_pattern}(?:\s*{num_pattern})*)', matched_str, re.IGNORECASE)
        unit_part_match = re.search(duration_units, matched_str, re.IGNORECASE)
        if num_part_match and unit_part_match:
            converted_value = word_to_num(num_part_match.group(0))
            return f"{converted_value} {unit_part_match.group(0).strip()}"
        return matched_str
    
    return 'N/A'

def _extract_frequency(text: str) -> str:
    num_pattern = r'(?:(?:\d+(?:\.\d+)?)|' + r'|'.join(NUMBER_WORDS.keys()) + r')'
    frequency_terms = r'(twice daily|once a day|thrice daily|three times a day|four times a day|daily|every\s+' + num_pattern + r'\s*hours|b\.?d\.?|t\.?i\.?d\.?|o\.?d\.?|q\.?i\.?d\.?|bd|tid|od|qid|bid|tds|qds|qd|prn|stat|as needed|every other day|alternate day|weekly|monthly|once)\b'
    regex = re.search(frequency_terms, text, re.IGNORECASE)
    if regex:
        matched_str = regex.group(0).strip()
        num_match = re.search(num_pattern, matched_str, re.IGNORECASE)
        if num_match:
            converted_num = word_to_num(num_match.group(0))
            return re.sub(re.escape(num_match.group(0)), converted_num, matched_str, flags=re.IGNORECASE)
        return matched_str
    return 'N/A'

def _extract_timing(text: str) -> str:
    timing_terms = r'(before food|after food|at night|morning|evening|bedtime|before meal|after meal|empty stomach|with food|after breakfast|after lunch|after dinner|before breakfast|before lunch|before dinner)\b'
    regex = re.search(timing_terms, text, re.IGNORECASE)
    if regex:
        return regex.group(0).strip()
    return 'N/A'

# Core Extraction Logic (Prioritizes Learned Feedback - from backend_app.py)
def _extract_medicines(text: str) -> List[Dict]:
    print(f"DEBUG: ner_pipeline status at _extract_medicines start: {ner_pipeline is not None}")
    text_lower = text.lower()

    # 1. Check LEARNED_FEEDBACK first for highly similar inputs
    for feedback_entry in LEARNED_FEEDBACK:
        original_feedback_text_lower = feedback_entry['original_text'].lower()
        similarity_score = fuzz.ratio(text_lower, original_feedback_text_lower)
        if similarity_score > 90:
            print(f"DEBUG: Found highly similar input in learned feedback (score: {similarity_score}). Returning corrected data.")
            return [med.model_dump() if hasattr(med, 'model_dump') else med for med in feedback_entry['corrected_medicines']]

    # 2. If no direct feedback match, proceed with NER model (or fallback)
    if ner_pipeline:
        print(f"DEBUG: Processing input text with NER model: '{text}'")
        return _extract_medicines_with_biobert(text)
    else:
        print(f"DEBUG: Processing input text with basic keyword matching: '{text}'")
        return _extract_medicines_basic(text)

# NER Model-based Extraction (from backend_app.py, adjusted for _merge_ner_tokens)
def _extract_medicines_with_biobert(text: str) -> List[Dict]:
    extracted_data = []
    raw_ner_results = ner_pipeline(text)
    print(f"DEBUG: Raw NER results from NER model (before merging): {raw_ner_results}")
    
    ner_results = _merge_ner_tokens(convert_to_serializable(raw_ner_results))
    print(f"DEBUG: Merged NER results: {ner_results}")

    identified_medicine_names = set()

    for entity in ner_results:
        if entity['entity'] in ['Chemical', 'CHEMICAL', 'DRUG', 'MEDICINE', 'COMPOUND', 'Medication']:
            potential_med_name = entity['word'].strip()
            print(f"DEBUG: Potential medicine recognized by NER model: '{potential_med_name}' (Entity Group: {entity['entity']})")
            
            best_match_from_list = "N/A"
            max_similarity = 0.0
            
            if potential_med_name.lower() in LOADED_MEDICINE_NAMES_LOWER_SET:
                best_match_from_list = next((n for n in LOADED_MEDICINE_NAMES if n.lower() == potential_med_name.lower()), potential_med_name)
                max_similarity = 100.0
                print(f"DEBUG: Exact match found for '{potential_med_name}': '{best_match_from_list}'")
            else:
                for loaded_name in LOADED_MEDICINE_NAMES:
                    current_similarity = fuzz.ratio(potential_med_name.lower(), loaded_name.lower())
                    if current_similarity > max_similarity:
                        max_similarity = current_similarity
                        best_match_from_list = loaded_name
                print(f"DEBUG: Best fuzzy match for '{potential_med_name}' from loaded list: '{best_match_from_list}' (Similarity: {max_similarity})")

            if max_similarity > FUZZY_MATCH_THRESHOLD and best_match_from_list != "N/A":
                if best_match_from_list.lower() not in identified_medicine_names:
                    identified_medicine_names.add(best_match_from_list.lower())
                    extracted_data.append({
                        "name": best_match_from_list,
                        "dosage": _extract_dosage(text),
                        "duration": _extract_duration(text), 
                        "frequency": _extract_frequency(text),
                        "timing": _extract_timing(text),
                    })
                    print(f"DEBUG: Added extracted medicine (high similarity): {best_match_from_list}")
            else:
                print(f"DEBUG: Skipped '{potential_med_name}' (Similarity: {max_similarity}) - below threshold or not a valid match.")
        else: 
            print(f"DEBUG: Entity '{entity['word']}' with group '{entity['entity']}' is not a recognized medicine type.")

    print(f"DEBUG: Final extracted data from NER model: {extracted_data}")
    return extracted_data

# Fallback basic extraction (from backend_app.py)
def _extract_medicines_basic(text: str) -> List[Dict]:
    extracted_data = []
    text_lower = text.lower()
    sorted_available_medicines = sorted(LOADED_MEDICINE_NAMES, key=len, reverse=True)
    matched_names = set()
    print(f"DEBUG: Basic extraction for text: '{text}'")

    for med_name in sorted_available_medicines:
        med_name_lower = med_name.lower()
        if med_name_lower in text_lower and med_name_lower not in matched_names:
            print(f"DEBUG: Basic direct containment match found: '{med_name}'")
            extracted_data.append({
                "name": med_name,
                "dosage": _extract_dosage(text),
                "duration": _extract_duration(text),
                "frequency": _extract_frequency(text),
                "timing": _extract_timing(text),
            })
            matched_names.add(med_name_lower)
        else:
            similarity_score = fuzz.ratio(med_name_lower, text_lower)
            if similarity_score > 60 and med_name_lower not in matched_names:
                print(f"DEBUG: Basic fuzzy match found: '{med_name}' (Similarity: {similarity_score})")
                extracted_data.append({
                    "name": med_name,
                    "dosage": _extract_dosage(text),
                    "duration": _extract_duration(text),
                    "frequency": _extract_frequency(text),
                    "timing": _extract_timing(text),
                })
                matched_names.add(med_name_lower)
    print(f"DEBUG: Final extracted data from basic: {extracted_data}")
    return extracted_data


# Medicine Suggestion Logic (from backend_app.py, adjusted for _merge_ner_tokens)
def _get_medicine_suggestion(input_text: str, patient_summary: str) -> str:
    input_lower = input_text.lower()
    
    # Prioritize learned feedback for suggestions too
    for feedback_entry in LEARNED_FEEDBACK:
        original_feedback_text_lower = feedback_entry['original_text'].lower()
        similarity_score = fuzz.ratio(input_lower, original_feedback_text_lower)
        if similarity_score > 90:
            for corrected_med in feedback_entry['corrected_medicines']:
                med_name = corrected_med['name'] if isinstance(corrected_med, dict) else corrected_med.name
                if fuzz.ratio(input_lower, med_name.lower()) > 75:
                    print(f"DEBUG: Suggestion from learned feedback: {med_name}")
                    return med_name
            return "N/A"
    
    if ner_pipeline:
        print("DEBUG: No direct feedback match for suggestion. Using NER model.")
        return _get_medicine_suggestion_with_biobert(input_text, patient_summary)
    else:
        print("DEBUG: No direct feedback match for suggestion and NER model not loaded. Falling back to basic matching.")
        best_match = "N/A"
        highest_similarity = 0.0
        for med_name in LOADED_MEDICINE_NAMES:
            current_similarity = fuzz.ratio(input_lower, med_name.lower())
            if current_similarity > 60:
                highest_similarity = current_similarity
                best_match = med_name
        return best_match if highest_similarity > 60 else "N/A"

def _get_medicine_suggestion_with_biobert(input_text: str, patient_summary: str) -> str:
    input_lower = input_text.lower()
    
    raw_ner_results = ner_pipeline(input_text)
    ner_results = _merge_ner_tokens(convert_to_serializable(raw_ner_results))
    
    potential_drug_entity = None
    for entity in ner_results:
        if entity['entity'] in ['Chemical', 'CHEMICAL', 'DRUG', 'MEDICINE', 'COMPOUND', 'Medication']:
            potential_drug_entity = entity['word'].strip()
            break

    if potential_drug_entity:
        best_match_name = "N/A"
        max_similarity = 0.0
        
        if potential_drug_entity.lower() in LOADED_MEDICINE_NAMES_LOWER_SET:
            best_match_name = next((n for n in LOADED_MEDICINE_NAMES if n.lower() == potential_drug_entity.lower()), potential_drug_entity)
            max_similarity = 100.0
            print(f"DEBUG: Exact match found for suggestion '{potential_drug_entity}': '{best_match_name}'")
        else:
            for loaded_name in LOADED_MEDICINE_NAMES:
                current_similarity = fuzz.ratio(potential_drug_entity.lower(), loaded_name.lower())
                if current_similarity > max_similarity:
                    max_similarity = current_similarity
                    best_match_name = loaded_name
            print(f"DEBUG: Best fuzzy match for suggestion '{potential_drug_entity}': '{best_match_name}' (Similarity: {max_similarity})")
        
        if max_similarity > FUZZY_MATCH_THRESHOLD:
            return best_match_name
    
    return "N/A"

# Helper for post-processing summary text (from app.py)
def _clean_summary_text_post_processing(text: str):
    """
    Performs final cleanup on the summary text.
    """
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Ensure proper punctuation at the end
    if text and not text.endswith(('.', '!', '?')):
        text += "."
    # Fix double periods if any
    text = re.sub(r'\.\s*\.', '.', text)
    return text


# --- API Endpoints ---

@app.get("/")
async def home():
    return {"message": "AI-Powered Medical Backend is running!"}

@app.post("/ner", response_model=NERResponse)
async def extract_entities_api(request_data: MedicineRequest):
    """
    Processes input text to apply spell correction, extract entities,
    and generate a structured medical summary.
    This endpoint integrates logic from the original app.py's /ner route.
    """
    # Removed grammar_tokenizer and grammar_model from this check
    if not ner_pipeline or not summary_pipeline:
        raise HTTPException(status_code=500, detail="One or more core models (NER/Summarization) not initialized on backend. Check server logs.")

    try:
        text = request_data.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Missing or empty 'text' field")

        # --- Step 1: Pre-processing (Spell Check -> Number Word Normalization) ---
        # Grammar correction step removed to save memory.
        spell_corrected_text = spell_correct_text(text)
        print(f"[INFO] Spell-Corrected Text: {spell_corrected_text}")
        
        # Normalize number words before NER and summarization
        # Directly use spell_corrected_text as grammar correction is removed
        processed_text_for_models = normalize_number_words(spell_corrected_text)
        print(f"[INFO] Normalized Number Words Text: {processed_text_for_models}")

        # --- Step 2: Named Entity Recognition (NER) using BioBERT ---
        raw_entities = ner_pipeline(processed_text_for_models)
        # Convert to serializable format and merge subword tokens, preserving spans
        cleaned_entities = _merge_ner_tokens(convert_to_serializable(raw_entities))
        print(f"[DEBUG] Cleaned Entities (with spans): {cleaned_entities}") 

        # --- Step 3: Extract & Normalize Specific Details from NER Output ---
        
        # Symptoms (Filter out semantically incorrect ones like "signs of recovery")
        all_symptoms = [ent["word"] for ent in cleaned_entities if ent["entity"] == "Sign_symptom"]
        extracted_symptoms = list(set([s for s in all_symptoms if "recovery" not in s.lower()]))

        # Diseases/Conditions
        extracted_diseases = list(set([ent["word"] for ent in cleaned_entities if ent["entity"] == "Disease"]))
        
        # Tests/Procedures
        extracted_tests_procedures = list(set([ent["word"] for ent in cleaned_entities if ent["entity"] == "Procedure"])) 
        if "check your body temperature" in processed_text_for_models.lower() and "body temperature check" not in extracted_tests_procedures:
            extracted_tests_procedures.append("Body temperature check")

        # --- IMPORTANT: Medication Extraction using backend_app.py's robust logic ---
        medication_prescriptions = _extract_medicines(processed_text_for_models)
        print(f"[DEBUG] Medication Prescriptions (from _extract_medicines): {medication_prescriptions}")
        
        # Extract General Advice using the new summarization-based function
        general_advice = extract_general_advice_via_summarization(processed_text_for_models)

        # --- Step 4: Construct the Structured Summary ---
        structured_summary_parts = []

        # 4.1: Patient Overview / Chief Complaints
        if extracted_symptoms or extracted_diseases:
            patient_summary_line = "Patient reports "
            if extracted_symptoms:
                patient_summary_line += f"symptoms of {', '.join(extracted_symptoms)}"
            if extracted_symptoms and extracted_diseases:
                patient_summary_line += " and "
            if extracted_diseases:
                patient_summary_line += f"diagnosed with {', '.join(extracted_diseases)}"
            patient_summary_line += "."
            structured_summary_parts.append(patient_summary_line)
        else:
            # Fallback to a general summary from T5 if no specific symptoms/diseases extracted
            generated_general_summary = summary_pipeline(
                processed_text_for_models,
                max_new_tokens=100,
                min_length=20,
                do_sample=False
            )[0]['summary_text']
            if generated_general_summary:
                structured_summary_parts.append(generated_general_summary.strip())


        # 4.2: Tests/Procedures Recommended
        if extracted_tests_procedures:
            structured_summary_parts.append(f"Tests/Procedures recommended: {', '.join(extracted_tests_procedures)}.")

        # 4.3: Prescribed Medications
        if medication_prescriptions:
            med_lines = []
            for med in medication_prescriptions:
                line = f"{med['name']}"
                if med['dosage'] != "N/A":
                    line += f" {med['dosage']}"
                if med['frequency'] != "N/A":
                    line += f" {med['frequency']}"
                if med['duration'] != "N/A":
                    line += f" for {med['duration']}"
                med_lines.append(line)
            structured_summary_parts.append(f"Prescribed medications: {'; '.join(med_lines)}.")

        # 4.4: Additional Advice
        if general_advice:
            structured_summary_parts.append(f"Additional advice: {'; '.join(general_advice)}.")
        
        final_structured_summary = " ".join(structured_summary_parts).strip()
        final_structured_summary = _clean_summary_text_post_processing(final_structured_summary)


        # --- Return results ---
        return NERResponse(
            entities=cleaned_entities,
            summary=final_structured_summary,
            medication_prescriptions=medication_prescriptions
        )

    except Exception as e:
        print(f"[ERROR] during /ner processing: {e}")
        raise HTTPException(status_code=500, detail={"error": "Failed to process text", "details": str(e)})

# --- API Endpoints from backend_app.py (copied directly) ---

@app.post("/extract_medicines", response_model=List[MedicineResponse])
async def extract_medicines_api(request: MedicineRequest):
    """
    Extracts medicine prescriptions from a given text (summary or voice input)
    by prioritizing learned feedback, then using BioBERT, then basic matching.
    """
    if not LOADED_MEDICINE_NAMES:
        raise HTTPException(status_code=500, detail="Medicine data not loaded on backend. Check server logs.")

    extracted = _extract_medicines(request.text)
    
    if not extracted:
        return []
    
    # Ensure proper Pydantic conversion for response
    return [MedicineResponse(**med) for med in extracted]

@app.post("/suggest_medicine", response_model=SuggestionResponse)
async def suggest_medicine_api(request: SuggestionRequest):
    """
    Suggests a medicine name based on input, patient summary, and available medicines,
    prioritizing learned feedback, then leveraging BioBERT.
    """
    if not LOADED_MEDICINE_NAMES:
        raise HTTPException(status_code=500, detail="Medicine data not loaded on backend. Check server logs.")

    suggestion = _get_medicine_suggestion(
        request.input_text,
        request.patient_summary
    )
    return {"suggestion": suggestion}

@app.post("/feedback_extraction")
async def feedback_extraction(feedback: FeedbackRequest):
    """
    Receives feedback on extracted medicines to 'learn' from user corrections.
    This data is stored in-memory for demonstration.
    """
    LEARNED_FEEDBACK.append(feedback.model_dump()) # Use model_dump() for Pydantic v2
    print(f"DEBUG: Received feedback. Current learned feedback count: {len(LEARNED_FEEDBACK)}")
    print(f"DEBUG: Stored feedback for original text (first 50 chars): {feedback.original_text[:50]}...")
    return {"message": "Feedback received and stored conceptually."}

# --- Main execution block for local development ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
