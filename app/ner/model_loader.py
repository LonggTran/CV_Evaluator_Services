import spacy

def load_ner_model(model_path: str):
    return spacy.load(model_path)
