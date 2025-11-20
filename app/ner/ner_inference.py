import os
from .model_loader import load_ner_model
from .ner_helper import postprocess_entities, postprocess_ner_final

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model-best-1")

nlp = load_ner_model(MODEL_PATH)

def run_spacy_ner(text: str):
    doc = nlp(text)
    raw_entities = postprocess_entities(doc)
    return postprocess_ner_final(raw_entities)
