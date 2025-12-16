import pandas as pd
import random
import json
import re
from google import genai
from tqdm import tqdm
import time
import os

# config
INPUT_CASES_JSON = "episodes.json"
OUTPUT_DATASET = "final_rag_dataset.jsonl"
NUM_SAMPLES = 249
API_KEY = None

GEMINI_MODEL = "gemini-2.5-flash"
client = genai.Client(api_key=API_KEY)

# util functions
def clean_case_name(raw):
    if not raw:
        return None

    cleaned = re.sub(r"[`'’“”\"_\-]", "", raw)
    cleaned = cleaned.replace("  ", " ").strip()
    cleaned = re.sub(r"\brn\b", "m", cleaned)
    cleaned = re.sub(r"\bnr\b", "m", cleaned)
    cleaned = re.sub(r"\bvv\b", "w", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not re.search(r"\bv\.?\b", cleaned):
        return None

    return cleaned.lower()


def normalize_current_case(case_name):
    cleaned = case_name.lower()
    cleaned = re.sub(r"[`'’“”\"_\-]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def filter_statutes(stat_list):
    allowed = []
    for s in stat_list:
        if re.search(r"\b(6|8|19|22)\s*U\.?S\.?C\.?", s, re.IGNORECASE):
            allowed.append(s)
        if re.search(r"\b(6|8|19|22)\s*C\.?F\.?R\.?", s, re.IGNORECASE):
            allowed.append(s)
    return allowed


# field extractors
def extract_list_field(text, field):
    pattern = rf'"?{field}"?\s*:\s*\[(.*?)\]'
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    block = match.group(1)
    items = re.findall(r'"(.*?)"', block)
    return items

def extract_field(text, field):
    pattern = rf'"?{field}"?\s*:\s*(.*?)(?=\n\s*[a-zA-Z_]+\s*:|$)'
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip()
    value = re.sub(r"^[\[\{]", "", value)
    value = re.sub(r"[\]\}]", "", value)
    value = value.strip().strip(",")
    return value if value else None


# gemini calls
def ask_gemini(case_name, case_text, retries=5):
    prompt = f"""
You are reading a U.S. federal court decision.

CASE NAME:
{case_name}

FULL CASE TEXT:
{case_text}

TASKS:
1. Generate a situational legal question based on the case issues.
2. Generate a golden answer based ONLY on the case ruling.
3. List all statutes/sections quoted in the text as "golden_context".
4. List all previously cited cases as "golden_case_context".

OUTPUT STRICTLY AS:
question: ...
golden_answer: ...
golden_context: ["...", "..."]
golden_case_context: ["...", "..."]
"""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            raw = response.text
            break

        except Exception as e:
            print(f"⚠️ Gemini error on attempt {attempt+1}/{retries}: {e}")
            time.sleep(5)
            raw = None

    if raw is None:
        return None

    question = extract_field(raw, "question")
    golden_answer = extract_field(raw, "golden_answer")
    raw_context = extract_list_field(raw, "golden_context")
    raw_cases = extract_list_field(raw, "golden_case_context")

    golden_context = filter_statutes(raw_context)

    current_case = normalize_current_case(case_name)
    golden_case_context = [current_case]

    for c in raw_cases:
        cc = clean_case_name(c)
        if cc and cc not in golden_case_context:
            golden_case_context.append(cc)

    return {
        "question": question,
        "golden_answer": golden_answer,
        "golden_context": golden_context,
        "golden_case_context": golden_case_context
    }

# main pipeline
def build_dataset():
    print("Loading cases...")
    df = pd.read_json(INPUT_CASES_JSON)

    print("Checking previously processed cases...")
    already_done = set()

    if os.path.exists(OUTPUT_DATASET):
        print("Resuming from existing dataset...")
        with open(OUTPUT_DATASET, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    already_done.add(data["case_name"])
                except:
                    continue
    else:
        print("No previous file. Starting fresh.")

    print(f"Already processed: {len(already_done)} cases")

    remaining_df = df[~df["n.name"].isin(already_done)]

    print(f"Sampling {NUM_SAMPLES} cases from remaining {len(remaining_df)}")

    sampled = remaining_df.sample(min(NUM_SAMPLES, len(remaining_df)))

    print("Querying Gemini for each case...")

    with open(OUTPUT_DATASET, "a", encoding="utf-8") as out_file:
        for idx, row in tqdm(sampled.iterrows(), total=len(sampled), unit="case"):
            case_name = row["n.name"]
            case_text = row["n.content"]

            g = ask_gemini(case_name, case_text)

            # if completely failed, skip
            if g is None:
                continue

            # skip if no valid statutes
            if not g["golden_context"]:
                continue

            out_file.write(json.dumps({
                "case_name": case_name,
                "question": g["question"],
                "golden_answer": g["golden_answer"],
                "golden_context": g["golden_context"],
                "golden_case_context": g["golden_case_context"]
            }) + "\n")

            out_file.flush()

            time.sleep(2)

    print("\nDataset updated:", OUTPUT_DATASET)

if __name__ == "__main__":
    build_dataset()