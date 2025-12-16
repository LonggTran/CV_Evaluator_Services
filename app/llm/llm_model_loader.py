#llm/llm_model_loader
import os

import torch
from unsloth import FastLanguageModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "final_llm")

model = None
tokenizer = None

def get_llm_model():
    global model, tokenizer

    if model is None or tokenizer is None:
        print("🔄 Loading LLM model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,   # thư mục model
            max_seq_length=2048,
            dtype=None               # để mặc định (FP32/FP16 tùy môi trường)
        )

        model = FastLanguageModel.for_inference(model)
        if torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")

        print(f"✅ Model loaded on {next(model.parameters()).device}")
        print("✅ Model loaded successfully.")

    return model, tokenizer
