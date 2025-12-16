import requests
import re
import csv
from bs4 import BeautifulSoup
from typing import List, Dict
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# load LegalBERT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
section = 6

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def embed_text(text: str):
    """Return LegalBERT embedding as a Python list of floats."""
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)

        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_embedding[0].cpu().numpy().tolist()

# cleaning
def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# legal structure regex
SUBSECTION_RE = re.compile(r"^\([a-z]\)\s")      # (a)
PARAGRAPH_RE = re.compile(r"^\(\d+\)\s")         # (1)
SUBPARAGRAPH_RE = re.compile(r"^\([A-Z]\)\s")    # (A)
CLAUSE_RE = re.compile(r"^\([ivxlcdm]+\)\s")     # (i)
SUBCLAUSE_RE = re.compile(r"^\([IVXLCDM]+\)\s")  # (I)


# HTML download and section extraction
def download_html(url: str) -> str:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text

def extract_sections(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")

    sections = []

    # Find all section headers: <h3 class="section-head">
    section_heads = soup.find_all("h3", class_="section-head")

    for head in section_heads:
        sec_title = clean(head.get_text())
        sec_id = "sec_" + re.sub(r"[^0-9]+", "", sec_title)  # like sec_1531

        # Collect subsection tags until next section-head
        chunks = []
        for sib in head.next_siblings:
            if isinstance(sib, str):
                continue

            # Stop when next § section begins
            if sib.name == "h3" and "section-head" in sib.get("class", []):
                break

            # append any <h4 class="subsection-head"> or <h4 class="paragraph-head">
            if sib.name == "h4" and (
                "subsection-head" in sib.get("class", []) or
                "paragraph-head" in sib.get("class", [])
            ):
                chunks.append(clean(sib.get_text()))

            # append statutory body paragraphs
            if sib.name == "p" and (
                "statutory-body" in sib.get("class", []) or
                "statutory-body-1em" in sib.get("class", []) or
                "statutory-body-2em" in sib.get("class", []) or
                "statutory-body-3em" in sib.get("class", [])
            ):
                chunks.append(clean(sib.get_text()))

        full_text = " ".join(chunks).strip()

        if full_text:
            sections.append({
                "section_id": sec_id,
                "section_title": sec_title,
                "text": full_text
            })

    return sections


# structural splitting
def split_by_structure(text: str) -> List[str]:
    """
    Split according to USC hierarchical boundaries.
    """
    lines = text.split(". ")
    chunks = []
    current = []

    for line in lines:
        l = line.strip()
        if not l:
            continue

        is_boundary = (
            SUBSECTION_RE.match(l)
            or PARAGRAPH_RE.match(l)
            or SUBPARAGRAPH_RE.match(l)
            or CLAUSE_RE.match(l)
            or SUBCLAUSE_RE.match(l)
        )

        if is_boundary and current:
            chunks.append(clean(" ".join(current)))
            current = [l]
        else:
            current.append(l)

    if current:
        chunks.append(clean(" ".join(current)))

    return chunks

# keep chunks with size enforcement
def enforce_size(chunks: List[str], max_len=1500, overlap=200):
    final = []
    for c in chunks:
        words = c.split()
        if len(words) <= max_len:
            final.append(c)
        else:
            start = 0
            while start < len(words):
                end = start + max_len
                window = " ".join(words[start:end])
                final.append(window)
                start = max(0, end - overlap)
    return final

def smart_chunk_sections(sections, max_len=1500, overlap=200):
    out = []
    for sec in sections:
        structured = split_by_structure(sec["text"])
        sized = enforce_size(structured, max_len=max_len, overlap=overlap)

        for i, ch in enumerate(sized):
            out.append({
                "section_id": sec["section_id"],
                "section_title": sec["section_title"],
                "chunk_index": i,
                "chunk": ch
            })
    return out

def save_with_embeddings(chunks, out_csv):
    """
    Produces a CSV with:
    - section_id
    - section_title
    - chunk_index
    - chunk
    - embedding_0 ... embedding_767
    """

    from tqdm import tqdm  # progress bar

    example_emb = embed_text("test")
    emb_dim = len(example_emb)

    fieldnames = ["section_id", "section_title", "chunk_index", "chunk"] + \
                 [f"emb_{i}" for i in range(emb_dim)]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for chunk in tqdm(chunks, desc="Embedding chunks with LegalBERT", unit="chunk"):
            emb = embed_text(chunk["chunk"])

            row = {
                "section_id": chunk["section_id"],
                "section_title": chunk["section_title"],
                "chunk_index": chunk["chunk_index"],
                "chunk": chunk["chunk"]
            }

            for i, v in enumerate(emb):
                row[f"emb_{i}"] = v

            w.writerow(row)

    print(f"✓ Saved {len(chunks)} chunks with LegalBERT embeddings → {out_csv}")

def process_usc_to_csv(url, out_csv):
    print("Downloading HTML…")
    html = download_html(url)

    print("Extracting sections…")
    sections = extract_sections(html)
    print(f"Found {len(sections)} sections.")

    print("Chunking with legal structure…")
    chunks = smart_chunk_sections(sections)
    print(f"Generated {len(chunks)} chunks.")

    print("Embedding with LegalBERT…")
    save_with_embeddings(chunks, out_csv)

if __name__ == "__main__":
    sec = str(section)
    url = f"https://www.govinfo.gov/content/pkg/USCODE-2024-title{sec}/html/USCODE-2024-title{sec}.htm"
    out_csv = f"usc_title{sec}.csv"
    process_usc_to_csv(url, out_csv)