"""
Structured feature extraction pipeline (JSON output)
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from openai import OpenAI

# ===================== CONFIG =====================
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data/reports"
OUTPUT_DIR = BASE_DIR / "outputs"
CSV_PATH = OUTPUT_DIR / "structured_results.csv"
JSON_OUT_DIR = OUTPUT_DIR / "json_results"
JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gpt-5.2"
MAX_API_RETRIES = 8
MAX_PARSE_RETRIES = 3
TEMPERATURE = 0
DEIDENTIFY = True   # IRB-safe default

# ===================== LOGGING =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== CLIENT =====================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===================== TEMPLATE =====================
def load_templates():
    return (
        (BASE_DIR / "prompts/json_template.txt").read_text(encoding="utf-8"),
        (BASE_DIR / "prompts/prompt_template.txt").read_text(encoding="utf-8")
    )

JSON_TEMPLATE, PROMPT_TEMPLATE = load_templates()

# ===================== UTILS =====================
def deidentify(text: str) -> str:
    text = re.sub(r"姓名[:：]\s*\S+", "姓名: [REDACTED]", text)
    text = re.sub(r"\b\d{8,}\b", "[ID]", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[DATE]", text)
    text = re.sub(r"\b(\d{3,4}-\d{3,4}-\d{3,4})\b", "[PHONE]", text)
    return text


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(
        JSON_TEMPLATE=JSON_TEMPLATE,
        report_text=text
    )


def extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group())


# ===================== MODEL =====================
def call_model(prompt: str) -> str:
    for i in range(MAX_API_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            if "rate" in str(e).lower():
                time.sleep(min(2 ** i, 30))
            else:
                raise

    raise RuntimeError("API failed")


# ===================== CORE =====================
def extract_features(report_text: str, study_id: str) -> Dict[str, Any]:

    if DEIDENTIFY:
        report_text = deidentify(report_text)

    prompt = build_prompt(report_text)

    for attempt in range(MAX_PARSE_RETRIES):
        content = call_model(prompt)

        try:
            return extract_json(content)
        except Exception as e:
            logger.warning(f"Parse failed {study_id} (attempt {attempt+1})")
            time.sleep(1)

    raise RuntimeError("JSON parsing failed")


# ===================== MAIN =====================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []


    for idx, file in enumerate(sorted(INPUT_DIR.glob("*.txt"))):
        study_id = f"case_{idx:04d}"

        try:
            logger.info(f"Processing {study_id}")

            report_text = file.read_text(encoding="utf-8")
            data = extract_features(report_text, study_id)


            json_path = JSON_OUT_DIR / f"{study_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)


            rows.append({
                "study_id": study_id,
                **data
            })

        except Exception as e:
            logger.error(f"Failed {file.name}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(CSV_PATH, index=False)
        logger.info(f"Saved aggregated CSV -> {CSV_PATH}")
        logger.info(f"Saved individual JSONs -> {JSON_OUT_DIR}")
    else:
        logger.warning("No valid results")


if __name__ == "__main__":
    main()