import re
import torch
from sympy.parsing.sympy_parser import null
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


class ResumeParserCore:
    def __init__(self, model_path="deberta-ner-final"):
        print("Đang tải AI Model DeBERTa vào bộ nhớ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        device = 0 if torch.cuda.is_available() else -1
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple",
                                     device=device)

        self.expected_labels = ["NAME", "EMAIL", "PHONE", "LOCATION", "JOB_TITLE", "COMPANY", "SKILL", "DEGREE",
                                "UNIVERSITY", "CERTIFICATE", "PROJECT", "URL"]
        self.multi_value = ["SKILL", "JOB_TITLE", "COMPANY", "DEGREE", "UNIVERSITY", "CERTIFICATE", "PROJECT", "URL"]

        # Tiếng Anh chuẩn (Ngăn chặn nhiễu)
        self.section_keywords = {
            "EDUCATION": ["EDUCATION", "ACADEMIC", "STUDIES"],
            "SKILL": ["SKILL", "TECHNICAL SKILLS", "TECHNOLOGIES", "TOOLS"],
            "EXPERIENCE": ["EXPERIENCE", "WORK HISTORY", "EMPLOYMENT"],
            "PROJECT": ["PROJECT", "PERSONAL PROJECTS"],
            "CERTIFICATE": ["CERTIFICATE", "CERTIFICATIONS", "AWARDS"]
        }

    def _get_text_zones(self, text: str) -> list:
        zones = []
        lines = text.split('\n')
        current_idx = 0
        for line in lines:
            line_clean = line.strip().upper()
            if 3 < len(line_clean) < 40 and not "@" in line_clean and not re.search(r'\d{8,}', line_clean):
                for zone_type, keywords in self.section_keywords.items():
                    if any(kw in line_clean for kw in keywords):
                        zones.append({"zone": zone_type, "start": current_idx, "end": len(text)})
                        if len(zones) > 1: zones[-2]["end"] = current_idx
                        break
            current_idx += len(line) + 1
        return zones

    def _get_active_zone(self, start_idx: int, zones: list) -> str:
        for z in zones:
            if z["start"] <= start_idx <= z["end"]: return z["zone"]
        return "GENERAL"

    def _fallback_extract_name(self, text: str) -> str:
        lines = text.strip().split('\n')
        for line in lines[:5]:
            line = line.strip()
            if not line or "@" in line or re.search(r'\d', line) or line.lower() in ["cv", "resume",
                                                                                     "profile"]: continue
            words = line.split()
            if 1 < len(words) <= 5:
                return re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', line).strip()
        return None

    def parse(self, text: str) -> dict:
        structured_output = {label: ([] if label in self.multi_value else None) for label in self.expected_labels}

        # Lớp 1: Regex
        if found_emails := re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text):
            structured_output["EMAIL"] = found_emails[0]
        if found_phones := re.findall(r'(?:\+84|0)(?:\d[-.\s]?){9,10}', text):
            structured_output["PHONE"] = found_phones[0]
        if found_urls := re.findall(
                r'(?:https?://)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
                text):
            for url in found_urls:
                if "@" not in url and url not in structured_output["URL"]: structured_output["URL"].append(url)

        # Lớp 2: AI Zone-Aware Extraction
        text_zones = self._get_text_zones(text)
        raw_predictions = self.ner_pipeline(text)

        for entity in raw_predictions:
            if entity['score'] < 0.10: continue
            label = entity['entity_group']
            word = entity['word'].replace('Ġ', ' ').strip()
            if len(word) <= 1 and word not in ["C", "R"]: continue

            active_zone = self._get_active_zone(entity.get('start', 0), text_zones)
            if label == "COMPANY" and active_zone == "EDUCATION":
                label = "UNIVERSITY"
            elif active_zone == "PROJECT" and label not in ["SKILL", "URL"]:
                label = "PROJECT"
            elif active_zone == "CERTIFICATE" and label not in ["URL", "SKILL"]:
                label = "CERTIFICATE"

            if label in structured_output and label not in ["EMAIL", "PHONE", "URL"]:
                if label in self.multi_value:
                    if word not in structured_output[label]: structured_output[label].append(word)
                else:
                    if structured_output[label] is None or entity['score'] > 0.80: structured_output[label] = word

        # Lớp 3: Fallback & Clean
        if not structured_output["NAME"]: structured_output["NAME"] = self._fallback_extract_name(text)

        final_display = {}
        for label, value in structured_output.items():
            out_key = "ADDITIONAL_URLS" if label == "URL" else label
            if isinstance(value, list):
                final_display[out_key] = ", ".join(value) if len(value) > 0 else None
            else:
                if value:
                    clean_val = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9+]+$', '', value).strip()
                    final_display[out_key] = clean_val if len(clean_val) > 1 else None
                else:
                    final_display[out_key] = None

        return final_display