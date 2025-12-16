import torch
from app.llm.llm_model_loader import get_llm_model
from app.common.utils import safe_json_parse

# Load model khi server start
model, tokenizer = get_llm_model()

# ---- FIXED FIELDS ----
CV_FIELDS = [
    "PERSON", "ROLE", "COMPANY", "CONTACT", "PHONE",
    "DEGREE", "GPA", "EXPERIENCE",
    "SKILL_Programming Languages", "SKILL_Cloud & DevOps",
    "SKILL_Databases", "SKILL_Tools & Frameworks", "SKILL_Soft Skills"
]

JD_FIELDS = [
    "JOB_TITLE", "COMPANY", "LOCATION", "EXPERIENCE_REQUIRED",
    "EDUCATION_REQUIRED", "SKILL_Programming_Languages",
    "SKILL_Cloud_DevOps", "SKILL_Databases", "SKILL_Tools_Frameworks",
    "SKILL_Soft_Skills", "RESPONSIBILITIES", "BENEFITS", "GPA"
]

# ---- SYSTEM PROMPTS ----
PROMPTS = {
    "cv": f"""
You are an expert CV/Resume NER extractor.
Extract exactly the following fields: {', '.join(CV_FIELDS)}.
Return ONLY JSON. Missing field = "Không có thông tin".
""",
    "jd": f"""
You are an expert Job Description (JD) NER extractor.
Extract exactly the following fields: {', '.join(JD_FIELDS)}.
Return ONLY JSON. Missing field = "Không có thông tin".
"""
}


def llm_extract(text: str, mode: str):
    """ mode = ['cv', 'jd'] """

    if mode not in ["cv", "jd"]:
        raise ValueError("mode must be 'cv' or 'jd'")

    FIXED_KEYS = CV_FIELDS if mode == "cv" else JD_FIELDS
    SYSTEM_PROMPT = PROMPTS[mode]

    try:
        user_prompt = f"Input:\n{text}\n\nReturn ONLY JSON with the fixed keys above."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
            attention_mask = attention_mask.to(model.device)
        else:
            input_ids = encoded.to(model.device)
            attention_mask = torch.ones_like(input_ids).to(model.device)

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False
        )

        raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        json_text = raw_output[start:end] if start != -1 else None

        data = safe_json_parse(json_text) if json_text else {}

        return { key: data.get(key, "Không có thông tin") for key in FIXED_KEYS }

    except Exception as e:
        return { key: "Không có thông tin" for key in FIXED_KEYS } | {"error": str(e)}