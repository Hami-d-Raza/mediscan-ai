"""
services/nlp_model.py
---------------------
HuggingFace Transformers-based NER service for MediScan AI.

Model:
  d4data/biomedical-ner-all  — Bio_ClinicalBERT fine-tuned on multiple
  biomedical corpora. Recognises 18 entity types including diseases,
  symptoms, medications, and anatomical structures.

  Model card: https://huggingface.co/d4data/biomedical-ner-all

Loading strategy:
  - On first call, pipeline is downloaded & cached by HuggingFace (~400 MB).
  - Subsequent calls re-use the cached singleton.
  - If loading fails (no internet, low RAM, etc.) → SIMULATION mode activates:
    keyword-based fallback runs on the raw text so the full API response
    shape is always returned.

Public API:
  analyze_report(text: str) -> dict
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity label → output bucket mapping
# Labels from d4data/biomedical-ner-all
# ---------------------------------------------------------------------------
_LABEL_MAP: dict[str, str] = {
    # Diseases / disorders
    "Disease_disorder":        "diseases",
    "Pathological_formation":  "diseases",
    "Cancer":                  "diseases",
    # Symptoms / signs
    "Sign_symptom":            "symptoms",
    "Severity":                "symptoms",
    # Medications / drugs
    "Medication":              "medications",
    "Therapeutic_procedure":   "medications",
    "Clinical_event":          "medications",
    # Anatomical / imaging (routed to other bucket)
    "Anatomical_location":     "other",
    "Diagnostic_procedure":    "other",
    "Lab_value":               "other",
    # (everything else is ignored in the top-level buckets)
}

# Buckets always present in the response even when empty
_EMPTY_BUCKETS: dict[str, list] = {
    "diseases":    [],
    "symptoms":    [],
    "medications": [],
    "other":       [],
}

# ---------------------------------------------------------------------------
# Singleton pipeline
# ---------------------------------------------------------------------------
_ner_pipeline = None
_simulation_mode: bool = False


def _get_pipeline():
    """
    Lazy-load the HuggingFace NER pipeline.
    Cached after first call — thread-safe for single-worker FastAPI.
    """
    global _ner_pipeline, _simulation_mode

    if _ner_pipeline is not None:
        return _ner_pipeline

    try:
        from transformers import pipeline, logging as hf_logging
        hf_logging.set_verbosity_error()   # suppress HF info spam

        logger.info("Loading biomedical NER model (d4data/biomedical-ner-all)...")
        _ner_pipeline = pipeline(
            task="ner",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="first",    # merge sub-word tokens → whole words
            device=-1,                       # force CPU; change to 0 for GPU
        )
        _simulation_mode = False
        logger.info("NER model loaded successfully.")

    except Exception as exc:
        logger.warning(
            "Could not load NER model: %s — falling back to SIMULATION mode.", exc
        )
        _simulation_mode = True
        _ner_pipeline = None   # keep None so simulation path is taken

    return _ner_pipeline


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_ner_pipeline(text: str) -> list[dict]:
    """
    Run HuggingFace NER pipeline on text.
    Handles texts longer than the model's 512-token limit by chunking.
    """
    pipe = _get_pipeline()
    if pipe is None:
        return []

    # Chunk text into ~400-word segments to stay under the 512 token limit
    words  = text.split()
    chunk_size = 400
    chunks = [
        " ".join(words[i: i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    all_entities: list[dict] = []
    for chunk in chunks:
        if chunk.strip():
            try:
                all_entities.extend(pipe(chunk))
            except Exception as exc:
                logger.warning("NER chunk failed: %s", exc)

    return all_entities


def _clean_word(word: str) -> str:
    """Remove BERT subword artefacts and normalise whitespace/punctuation."""
    # Remove all ## subword prefixes (e.g. '##lioma' → 'lioma')
    word = re.sub(r"##", "", word)
    # Remove stray spaces that sometimes appear around subword joins
    word = re.sub(r"\s+", " ", word).strip()
    # Strip leading/trailing punctuation
    word = word.strip(".,;:()[]\"'")
    return word


# ---------------------------------------------------------------------------
# Blacklist: generic descriptors that are NOT meaningful diseases/symptoms
# ---------------------------------------------------------------------------
_GENERIC_BLACKLIST: set[str] = {
    # Adjectives / descriptors
    "intact", "normal", "mild", "moderate", "severe", "marked", "significant",
    "bilateral", "unilateral", "focal", "diffuse", "acute", "chronic",
    "high", "low", "intermediate", "heterogeneous", "homogeneous",
    "medial", "lateral", "anterior", "posterior", "superior", "inferior",
    "central", "peripheral", "proximal", "distal",
    # Generic anatomical noise
    "bone", "cartilage", "fibre", "fibres", "tissue", "fat", "fluid",
    "stump", "floor", "roof", "surface", "layer", "margin", "wall",
    # Generic medical noise
    "loss", "change", "changes", "signal", "finding", "findings",
    "pattern", "appearance", "morphology", "heterogeneity",
    "impingement", "involvement", "extension",
    # Imaging sequence names
    "t1w", "t2w", "pd", "stir", "flair", "dwi", "adc", "swi",
    "mri", "mra", "ct", "radiology",
    # Generic procedures
    "surgery", "procedure", "injection", "biopsy",
}

# Date pattern: matches "2024-01-01", "01/12/2024", "- 05 - 12" etc.
_DATE_PATTERN = re.compile(
    r'^[\-\s]*\d{1,4}[\-\/\s]+\d{1,2}[\-\/\s]+\d{1,4}[\-\s]*$'
)


def _is_blacklisted(word: str) -> bool:
    """Return True if the word should be filtered out."""
    # Check normalised lowercase against the blacklist
    if word.lower() in _GENERIC_BLACKLIST:
        return True
    # Filter out date-like tokens
    if _DATE_PATTERN.match(word):
        return True
    # Filter out pure digit strings
    if re.match(r'^[\d\s\.\-\/]+$', word):
        return True
    return False


def _group_entities(raw_entities: list[dict]) -> dict[str, list[str]]:
    """
    Map raw HuggingFace NER output to the four response buckets.
    Deduplicates and normalises entity text (title-case).
    """
    buckets: dict[str, set[str]] = {k: set() for k in _EMPTY_BUCKETS}

    for ent in raw_entities:
        label = ent.get("entity_group", "")
        word  = ent.get("word", "").strip()
        score = ent.get("score", 0.0)

        if not word or score < 0.70:        # discard low-confidence tokens
            continue

        # Skip raw subword fragments that the aggregation didn't merge
        if word.startswith("##"):
            continue

        word = _clean_word(word)

        # Skip if too short after cleaning (likely a noise fragment)
        if len(word) < 3:
            continue

        # Skip if still contains ## (malformed token)
        if "##" in word:
            continue

        # Skip generic / blacklisted / date tokens
        if _is_blacklisted(word):
            continue

        bucket = _LABEL_MAP.get(label, "other")
        buckets[bucket].add(word.title())   # normalise to Title Case

    return {k: sorted(v) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# Keyword-based simulation (fallback when model unavailable)
# ---------------------------------------------------------------------------

# MRI / neurological disease terms
_DISEASE_KEYWORDS = [
    # Brain tumors
    "glioma", "glioblastoma", "meningioma", "pituitary tumor", "astrocytoma",
    "brain tumor", "brain metastasis", "acoustic neuroma", "medulloblastoma",
    # Neurodegenerative
    "alzheimer", "alzheimer's disease", "dementia", "parkinson", "parkinson's disease",
    # Cerebrovascular
    "stroke", "ischemic stroke", "hemorrhagic stroke", "tia",
    "transient ischemic attack", "cerebral infarction",
    "subarachnoid hemorrhage", "cerebral hemorrhage",
    # Demyelinating
    "multiple sclerosis", "ms lesion", "white matter lesion", "demyelination",
    # Spinal
    "disc herniation", "spinal stenosis", "spondylosis", "radiculopathy",
    "myelopathy", "disc prolapse", "vertebral fracture",
    # General neurological
    "hydrocephalus", "cerebral edema", "cerebral atrophy", "epilepsy",
    "encephalitis", "meningitis", "abscess", "aneurysm",
    "arteriovenous malformation",
    # General medical
    "diabetes", "hypertension", "tuberculosis", "pneumonia",
    "tumor", "cancer", "fracture", "infection", "anemia",
]

# Neurological and general symptom terms
_SYMPTOM_KEYWORDS = [
    # Neurological
    "headache", "migraine", "seizure", "convulsion", "memory loss",
    "cognitive decline", "confusion", "disorientation", "aphasia",
    "dysarthria", "dysphagia", "diplopia", "vision loss", "blurred vision",
    "weakness", "hemiplegia", "hemiparesis", "paralysis", "numbness",
    "tingling", "ataxia", "tremor", "vertigo", "dizziness",
    "loss of consciousness", "syncope", "papilledema",
    # Spinal / musculoskeletal
    "back pain", "neck pain", "radicular pain", "sciatica",
    "muscle weakness", "bowel dysfunction", "bladder dysfunction",
    # General
    "fever", "fatigue", "nausea", "vomiting", "pain",
    "shortness of breath", "chest pain", "cough",
]

# Neurological and general medication / treatment terms
_MEDICATION_KEYWORDS = [
    # Anti-tumour / oncology
    "temozolomide", "bevacizumab", "radiotherapy", "chemotherapy",
    "dexamethasone", "corticosteroid",
    # Anti-epileptic
    "levetiracetam", "valproate", "phenytoin", "carbamazepine",
    # Dementia
    "donepezil", "memantine", "rivastigmine",
    # Stroke
    "alteplase", "warfarin", "aspirin", "clopidogrel", "heparin",
    "antihypertensive", "lisinopril", "amlodipine",
    # MS
    "interferon beta", "glatiramer acetate", "natalizumab", "ocrelizumab",
    # General
    "metformin", "paracetamol", "amoxicillin", "ibuprofen",
    "insulin", "atorvastatin", "omeprazole", "prednisone",
]

# MRI-specific imaging findings (surfaced in the 'other' bucket)
_MRI_FINDINGS_KEYWORDS = [
    "hyperintense", "hypointense", "isointense",
    "enhancement", "contrast enhancement", "ring enhancement",
    "mass effect", "midline shift", "herniation",
    "lesion", "lesions", "plaque", "signal abnormality",
    "white matter changes", "periventricular",
    "diffusion restriction", "adc", "dwi", "flair",
    "t1", "t2",
]


def _simulate_extraction(text: str) -> dict[str, list[str]]:
    """
    Keyword scan on lowercased text — runs when the model is unavailable.
    Returns the same bucket structure as the real pipeline.
    Also detects MRI imaging findings and places them in the 'other' bucket.
    """
    text_lower = text.lower()

    def _find(keywords: list[str]) -> list[str]:
        return sorted({
            kw.title() for kw in keywords
            if kw in text_lower and not _is_blacklisted(kw)
        })

    return {
        "diseases":    _find(_DISEASE_KEYWORDS),
        "symptoms":    _find(_SYMPTOM_KEYWORDS),
        "medications": _find(_MEDICATION_KEYWORDS),
        "other":       [v for v in _find(_MRI_FINDINGS_KEYWORDS) if not _DATE_PATTERN.match(v)],
    }


# ---------------------------------------------------------------------------
# Condition → suggested medications map
# Each entry: condition keyword (lowercase) → list of suggestion dicts
# ---------------------------------------------------------------------------
_CONDITION_MEDICATIONS: list[dict] = [
    {
        "conditions": ["glioma", "glioblastoma", "brain tumor", "brain tumour", "astrocytoma"],
        "suggestions": [
            {"drug": "Dexamethasone",   "note": "Corticosteroid to reduce cerebral edema"},
            {"drug": "Levetiracetam",   "note": "Prophylactic anti-epileptic"},
            {"drug": "Temozolomide",    "note": "Chemotherapy agent (oncologist-prescribed)"},
            {"drug": "Omeprazole",      "note": "Gastric protection while on steroids"},
        ],
    },
    {
        "conditions": ["meningioma"],
        "suggestions": [
            {"drug": "Dexamethasone",   "note": "For perilesional edema management"},
            {"drug": "Levetiracetam",   "note": "Seizure prophylaxis if symptomatic"},
        ],
    },
    {
        "conditions": ["pituitary", "pituitary adenoma", "pituitary tumor"],
        "suggestions": [
            {"drug": "Cabergoline",     "note": "For prolactinoma (endocrinologist-directed)"},
            {"drug": "Bromocriptine",   "note": "Dopamine agonist alternative for prolactinoma"},
        ],
    },
    {
        "conditions": ["multiple sclerosis", "ms lesion", "demyelination"],
        "suggestions": [
            {"drug": "Methylprednisolone", "note": "IV pulse for acute relapse (neurologist-directed)"},
            {"drug": "Interferon Beta-1a", "note": "Disease-modifying therapy"},
            {"drug": "Vitamin D3",         "note": "Supplementation often recommended in MS"},
        ],
    },
    {
        "conditions": ["stroke", "ischemic stroke", "cerebral infarction", "tia"],
        "suggestions": [
            {"drug": "Aspirin 75–300 mg",  "note": "Anti-platelet after ischaemic stroke"},
            {"drug": "Clopidogrel",        "note": "Anti-platelet alternative / dual therapy"},
            {"drug": "Atorvastatin",       "note": "Statin for secondary prevention"},
            {"drug": "Lisinopril",         "note": "ACE inhibitor for blood pressure control"},
        ],
    },
    {
        "conditions": ["disc herniation", "radiculopathy", "disc prolapse", "spinal stenosis"],
        "suggestions": [
            {"drug": "Ibuprofen 400 mg",   "note": "NSAID for pain/inflammation (after meals)"},
            {"drug": "Diclofenac",         "note": "NSAID alternative for radicular pain"},
            {"drug": "Pregabalin",         "note": "Neuropathic/radicular pain relief"},
            {"drug": "Omeprazole",         "note": "Gastric protection with NSAIDs"},
            {"drug": "Physiotherapy",      "note": "Core strengthening exercises are essential"},
        ],
    },
    {
        "conditions": ["meningitis", "encephalitis", "abscess", "infection"],
        "suggestions": [
            {"drug": "Ceftriaxone",        "note": "IV antibiotic (hospital-administered)"},
            {"drug": "Dexamethasone",      "note": "Adjunct to reduce inflammation"},
        ],
    },
    {
        "conditions": ["epilepsy", "seizure", "convulsion"],
        "suggestions": [
            {"drug": "Levetiracetam",      "note": "First-line anti-epileptic"},
            {"drug": "Valproate",          "note": "Broad-spectrum anti-epileptic"},
            {"drug": "Carbamazepine",      "note": "Focal seizure management"},
        ],
    },
    {
        "conditions": ["headache", "migraine"],
        "suggestions": [
            {"drug": "Paracetamol 500–1000 mg", "note": "First-line for tension headache"},
            {"drug": "Ibuprofen 400 mg",        "note": "For moderate-severity headache"},
            {"drug": "Sumatriptan",             "note": "For confirmed migraine attacks"},
        ],
    },
    {
        "conditions": ["hypertension"],
        "suggestions": [
            {"drug": "Amlodipine",         "note": "Calcium channel blocker"},
            {"drug": "Lisinopril",         "note": "ACE inhibitor"},
            {"drug": "Lifestyle changes",  "note": "Low-salt diet, exercise, weight management"},
        ],
    },
    {
        "conditions": ["vertigo", "dizziness"],
        "suggestions": [
            {"drug": "Betahistine",        "note": "For vestibular vertigo (Ménière's)"},
            {"drug": "Cinnarizine",        "note": "Anti-vertigo / anti-nausea"},
        ],
    },
]


def _suggest_medications(diseases: list[str], symptoms: list[str]) -> list[dict]:
    """
    Given detected diseases and symptoms, return a list of suggested
    medications with their notes.
    Each suggestion: {"drug": str, "note": str, "for_condition": str}
    """
    combined_lower = " ".join(diseases + symptoms).lower()
    seen_drugs: set[str] = set()
    suggestions: list[dict] = []

    for entry in _CONDITION_MEDICATIONS:
        matched_condition = next(
            (c for c in entry["conditions"] if c in combined_lower), None
        )
        if matched_condition:
            for s in entry["suggestions"]:
                if s["drug"].lower() not in seen_drugs:
                    seen_drugs.add(s["drug"].lower())
                    suggestions.append({
                        "drug":          s["drug"],
                        "note":          s["note"],
                        "for_condition": matched_condition.title(),
                    })

    return suggestions


def _generate_summary(entities: dict[str, list[str]], text: str) -> str:
    """Build a concise MRI-oriented summary from extracted entities."""
    parts = []
    if entities["diseases"]:
        parts.append(f"Detected condition(s): {', '.join(entities['diseases'])}.")
    if entities["symptoms"]:
        parts.append(f"Reported symptom(s): {', '.join(entities['symptoms'])}.")
    if entities["medications"]:
        parts.append(f"Medication / treatment noted: {', '.join(entities['medications'])}.")
    # Surface MRI imaging findings in summary (limit to 5)
    mri_other = entities.get("other", [])
    if mri_other:
        parts.append(f"MRI finding(s): {', '.join(mri_other[:5])}.")
    if not parts:
        parts.append("No specific medical entities were identified in the report.")

    word_count = len(text.split())
    parts.append(f"Report length: ~{word_count} words.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_report(text: str) -> dict:
    """
    Extract structured medical entities from plain text using NER.

    Args:
        text: Raw text content of the medical report.

    Returns:
        {
            "diseases":    ["Pneumonia", ...],
            "symptoms":    ["Fever", "Cough", ...],
            "medications": ["Amoxicillin", ...],
            "other":       [...],
            "summary":     "Identified condition(s): ...",
            "char_count":  1024,
            "simulated":   false
        }
    """
    if not text or not text.strip():
        return {
            **_EMPTY_BUCKETS,
            "summary":   "No text provided for analysis.",
            "char_count": 0,
            "simulated":  False,
        }

    # Trigger lazy model load (sets _simulation_mode as side-effect)
    _get_pipeline()

    # Extract entities
    if _simulation_mode:
        logger.info("Running NER in simulation (keyword) mode.")
        entities = _simulate_extraction(text)
    else:
        logger.info("Running NER pipeline on %d characters.", len(text))
        raw = _run_ner_pipeline(text)
        entities = _group_entities(raw)

        # If model ran but found nothing meaningful, surface keyword hits
        if not any(entities[k] for k in ("diseases", "symptoms", "medications")):
            logger.info("NER returned no confident hits — supplementing with keyword scan.")
            fallback = _simulate_extraction(text)
            for bucket in ("diseases", "symptoms", "medications"):
                entities[bucket] = sorted(
                    set(entities[bucket]) | set(fallback[bucket])
                )

    summary = _generate_summary(entities, text)
    suggested = _suggest_medications(entities["diseases"], entities["symptoms"])

    return {
        "diseases":             entities["diseases"],
        "symptoms":             entities["symptoms"],
        "medications":          entities["medications"],
        "other":                entities.get("other", []),
        "suggested_medications": suggested,
        "summary":              summary,
        "char_count":           len(text),
        "simulated":            _simulation_mode,
    }
