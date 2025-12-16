import pandas as pd
import re
import csv
import json
from transformers import AutoTokenizer, AutoModel
import torch
import nltk
from tqdm import tqdm

nltk.download('punkt')

# Load LegalBERT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Embedding function
def embed_text(text: str):
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)

        outputs = model(**inputs)
        cls_embed = outputs.last_hidden_state[:, 0, :]
        return cls_embed[0].cpu().numpy().tolist()

# chunker
def smart_legal_chunk(text: str, max_tokens=350, overlap=0.15):
    text = re.sub(r"\n\s*\n", "\n\n", text.strip())

    section_splits = re.split(
        r"\n(?=[A-Z][A-Z ]{4,}\n)",
        text
    )

    chunks = []

    for section in section_splits:
        sentences = nltk.sent_tokenize(section)
        token_count = 0
        current_chunk = []
        section_chunks = []

        for sent in sentences:
            sent_tokens = len(tokenizer.tokenize(sent))

            if token_count + sent_tokens > max_tokens:
                section_chunks.append(" ".join(current_chunk))

                overlap_size = int(len(current_chunk) * overlap)
                current_chunk = current_chunk[-overlap_size:]
                token_count = sum(len(tokenizer.tokenize(s)) for s in current_chunk)

            current_chunk.append(sent)
            token_count += sent_tokens

        if current_chunk:
            section_chunks.append(" ".join(current_chunk))

        chunks.extend(section_chunks)

    return [c.strip() for c in chunks if len(c.strip()) > 20]

def process_json(input_json, output_csv):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} cases from JSON.\n")
    print("\n📌 Building chunks for all cases (pass 1)...")

    all_chunks = [] 

    for row in data:
        case_name = row["case"]
        text = row["opinion"]

        chunks = smart_legal_chunk(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append((case_name, i, chunk))

    total_chunks = len(all_chunks)
    print(f"Total chunks to embed: {total_chunks}")
    print("\n📌 Embedding all chunks (pass 2)...")

    rows = []

    for (case_name, chunk_id, chunk_text) in tqdm(all_chunks, total=total_chunks, desc="Embedding All Cases", unit="chunk"):
        emb = embed_text(chunk_text)
        rows.append({
            "case_name": case_name,
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
            "embedding": emb
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_name", "chunk_id", "chunk_text", "embedding"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} chunks → {output_csv}\n")

process_json(
    input_json="all_cases.json",
    output_csv="cases_chunked_with_embeddings.csv"
)