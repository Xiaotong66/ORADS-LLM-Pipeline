"""
O-RADS classification pipeline using LLMs
"""

import os
import json
import re
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from openai import OpenAI
import anthropic

# ===================== CONFIG =====================
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data/reports"
OUTPUT_PATH = BASE_DIR / "outputs/orads_results.xlsx"

MODEL_NAME = "gemini"   # gpt / gemini / qwen / grok / deepseek / claude
MAX_RETRIES = 8
TEMPERATURE = 0

# ===================== LOGGING =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== CLIENT =====================
gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

grok_client = OpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1"
)

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

qwen_client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

claude_client = anthropic.Anthropic(
    api_key=os.getenv("CLAUDE_API_KEY")
)

# ===================== UTILS =====================
def retry(func):
    for i in range(MAX_RETRIES):
        try:
            return func()
        except Exception as e:
            if "rate" in str(e).lower() or "quota" in str(e).lower():
                time.sleep(min(2 ** i + random.random(), 30))
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def clean_json(text: str) -> str:
    return re.sub(r'```(?:json)?\s*|\s*```', '', text).strip()


def parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except:
        return json.loads(clean_json(text))


def parse_patient(folder: Path):
    parts = folder.name.split("-")
    pid = "-".join(parts[:2])
    return pid


# ===================== MODEL =====================
def call_model(prompt: str) -> str:
    def _call():
        if MODEL_NAME == "gpt":
            return gpt_client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE
            )

        if MODEL_NAME == "qwen":
            return qwen_client.chat.completions.create(
                model="qwen3-max",
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE
            )

        if MODEL_NAME == "grok":
            return grok_client.chat.completions.create(
                model="grok-4-1-fast",
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE
            )

        if MODEL_NAME == "deepseek":
            return deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE
            )

        if MODEL_NAME == "claude":
            return claude_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )

        raise ValueError("Unknown model")

    response = retry(_call)

    if MODEL_NAME == "claude":
        return response.content[0].text
    return response.choices[0].message.content.strip()


# ===================== CORE =====================
def extract_orads(report_text: str, prompt_template: str) -> List[Dict]:
    prompt = prompt_template.replace("{report_text}", report_text)

    content = call_model(prompt)
    data = parse_json(content)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


def process_all(prompt_template: str) -> pd.DataFrame:
    rows = []

    for folder in INPUT_DIR.iterdir():
        if not folder.is_dir():
            continue

        report_file = folder / "enhanced_report.txt"
        if not report_file.exists():
            continue

        pid = parse_patient(folder)

        logger.info(f"Processing {folder.name}")

        report_text = load_text(report_file)
        lesions = extract_orads(report_text, prompt_template)

        for lesion in lesions:
            rows.append({
                "ID": pid,
                "Region": lesion.get("region"),
                "Diameter": lesion.get("maximum_diameter"),
                "O-RADS": lesion.get("o_rads"),
            })

    return pd.DataFrame(rows)


# ===================== MAIN =====================
def main():
    prompt_template = load_text(BASE_DIR / "prompts/orads_prompt.txt")
    rules = load_text(BASE_DIR / "prompts/O-RADS Rules.txt")

    prompt_template = prompt_template.replace("{O_RADS_RULES}", rules)

    df = process_all(prompt_template)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    df.to_excel(OUTPUT_PATH, index=False)
    logger.info(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()