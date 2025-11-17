import re
from collections import defaultdict

MERGE_LABELS = [
    "SKILL_Tools_Frameworks",
    "SKILL_Programming_Languages",
    "EDUCATION",
    "COMPANY",
    "ROLE",
    "EXPERIENCE",
    "DEGREE"
]

NOISE_WORDS = {"experience", "work experience", "professional summary", "summary", "projects", "skills"}
LOCATIONS = ["ha noi", "ho chi minh", "hcm", "viet nam"]

def is_noise(text: str) -> bool:
    text_lower = text.strip().lower()
    return text_lower in NOISE_WORDS or re.fullmatch(r"(19|20)\d{2}", text) is not None

def is_location(text: str) -> bool:
    return text.strip().lower() in LOCATIONS

def merge_entities(entities: list[dict]) -> list[dict]:
    merged = []
    for ent in entities:
        label = ent["label"]
        text = ent["text"].strip()
        if not text or is_noise(text):
            continue
        if merged and label in MERGE_LABELS and merged[-1]["label"] == label:
            if text[0].islower() or text[0].isdigit():
                merged[-1]["text"] += " " + text
                continue
        merged.append({"label": label, "text": text})
    return merged

def postprocess_ner_final(entities: list[dict]) -> dict:
    merged = merge_entities(entities)
    result = defaultdict(list)
    for ent in merged:
        label = ent["label"]
        text = ent["text"].strip()
        if is_noise(text):
            continue
        if label == "PERSON" and is_location(text):
            continue
        if label == "contact":
            label = "EMAIL"
        result[label].append(text)
    # Loại trùng lặp
    for k in result:
        result[k] = list(dict.fromkeys(result[k]))
    return dict(result)
