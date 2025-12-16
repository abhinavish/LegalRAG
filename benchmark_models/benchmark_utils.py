import json
import ast
import numpy as np
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoTokenizer, AutoModel

class Helpers():
    @staticmethod
    def embed_text(tokenizer, device, model, text: str) -> np.ndarray:
        """Return LegalBERT CLS embedding as 1D numpy float32 array."""
        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            vec = cls_embedding[0].cpu().numpy()
        return vec.astype(np.float32)
    
    @staticmethod
    def normalize_matrix(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / norms

    @staticmethod
    def normalize_vector(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm

    @staticmethod
    def parse_golden_cite(cite: str):
        """
        Robustly parse a citation like:
        '8 U.S.C. § 1153(b)(5)'
        '8 USC §1153'
        '8 C.F.R. § 204.6(e)'
        '8 CFR 204.6'
        Returns a dict:
        {"title_type": "usc"|"cfr", "title_number": "8", "section_number": "1153" or "204.6"}
        Returns None ONLY if completely unparsable.
        """
        if not cite:
            return None

        c = cite.lower().strip()

        title_type = None
        title_number = None

        # USC variants
        m = re.search(r"(\d+)\s*(u\.?s\.?c\.?)", c)
        if m:
            title_type = "usc"
            title_number = m.group(1)

        # CFR variants
        if title_type is None:
            m = re.search(r"(\d+)\s*(c\.?f\.?r\.?)", c)
            if m:
                title_type = "cfr"
                title_number = m.group(1)

        # If still not found, attempt fallback
        if title_type is None:
            # look for something like "title 8"
            m = re.search(r"title\s*(\d+)", c)
            if m:
                title_type = "usc"  # assume USC by default
                title_number = m.group(1)

        if title_type is None or title_number is None:
            # cannot parse title info
            return None

        section_number = None

        # If "§" exists, parse directly
        if "§" in c:
            after = c.split("§", 1)[1].strip()
            # numerical part until first non-digit/dot
            sec = re.match(r"([\d\.]+)", after)
            if sec:
                section_number = sec.group(1)

        # If no "§", fallback patterns
        if section_number is None:
            # e.g. "8 USC 1153(b)(5)"
            m = re.search(r"\b(\d+\.\d+|\d+)", c)
            if m:
                section_number = m.group(1)

        if section_number is None:
            return None

        return {
            "title_type": title_type,
            "title_number": title_number,
            "section_number": section_number,
        }

    @staticmethod
    def parse_corpus_title_from_filename(filename: str):
        """
        Extract ('usc','6') from 'usc_title6.csv', etc.
        """
        name = filename.lower()
        title_type = None
        if "usc" in name:
            title_type = "usc"
        elif "cfr" in name:
            title_type = "cfr"

        m = re.search(r"title(\d+)", name)
        title_number = m.group(1) if m else None

        return title_type, title_number

    @staticmethod
    def parse_corpus_section_number(title: str):
        """
        Extract numeric section from corpus section_title, e.g.:
        '§1153. Immigrant investors' -> '1153'
        """
        if not isinstance(title, str):
            return None
        t = title.lower().strip()
        if "§" not in t:
            return None
        part = t.split("§", 1)[1]
        sec = re.split(r"[^\d\.]", part)[0]  # keep digits + dots
        sec = sec.strip()
        return sec or None

    @staticmethod
    def normalize_case_name(name: str) -> str:
        """
        Make case names comparable between:
        - 'Case_massachusetts_coalition_for_immigration_reform_v_us_citizenship_and_immigration_services'
        - 'massachusetts coalition for immigration reform v. us citizenship and immigration services'
        """
        if not name:
            return ""
        n = name.strip().lower()

        # strip 'case_' prefix
        if n.startswith("case_"):
            n = n[len("case_"):]

        # underscores → spaces
        n = n.replace("_", " ")

        # normalize 'v.' / 'vs.' / 'vs'
        n = n.replace(" vs. ", " v ")
        n = n.replace(" vs ", " v ")
        n = n.replace(" v. ", " v ")

        # collapse whitespace
        n = re.sub(r"\s+", " ", n)

        return n

    @staticmethod
    def precision_recall_at_k(golden_set, retrieved_list, k):
        """
        golden_set: set of labels (e.g., law triples or case names)
        retrieved_list: list of labels (with possible duplicates)
        Recall denominator is MIN(k, len(golden_set)).
        So questions with < k golden answers are evaluated fairly.

        Also enforces that each retrieved label is counted once (deduped)
        for precision/recall.
        """
        if k > len(retrieved_list):
            k = len(retrieved_list)

        # Top-k retrieved
        top_k = retrieved_list[:k]

        # Compute hits
        hits = [x for x in top_k if x['title'] in golden_set]
        num_hits = len(hits)

        # Precision denominator = number retrieved (unique)
        denom_ret = len(top_k) if top_k else 1
        precision = num_hits / denom_ret

        # Recall denominator = min(k, |golden_set|)
        gold_count = len(golden_set)
        denom_gold = min(k, gold_count) if gold_count > 0 else 1
        recall = num_hits / denom_gold

        return precision, recall

    @staticmethod
    def summarize_metrics(name, metrics_dict):
        print(f"\n=== {name} retrieval metrics ===")
        for k in sorted(metrics_dict.keys()):
            prec_list = metrics_dict[k]["precision"]
            rec_list = metrics_dict[k]["recall"]
            avg_p = float(np.mean(prec_list)) if prec_list else 0.0
            avg_r = float(np.mean(rec_list)) if rec_list else 0.0
            print(f"@{k}: precision={avg_p:.4f}, recall={avg_r:.4f}")


class Loaders():
    @staticmethod
    def load_test_dataset(path: str):
        path = Path(path)
        items = []
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        else:
            # assume JSON (list or single object)
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                items = data
            else:
                items = [data]
        return items

    @staticmethod
    def _load_emb_matrix_from_cols(df: pd.DataFrame, prefix: str = "emb_") -> np.ndarray:
        emb_cols = [c for c in df.columns if c.startswith(prefix)]
        if not emb_cols:
            raise ValueError("No embedding columns starting with 'emb_' found.")
        return df[emb_cols].values.astype(np.float32)

    @staticmethod
    def load_law_chunks(files):
        """
        Load all law CSVs and attach a '_source_file' column
        to each row to know which title they came from.
        Returns:
            df_all, emb_all
        """
        dfs = []
        for fname in files:
            df = pd.read_csv(fname)
            df["_source_file"] = fname
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)

        if any(c.startswith("emb_") for c in df_all.columns):
            emb_mat = Loaders._load_emb_matrix_from_cols(df_all, prefix="emb_")
        elif "embedding" in df_all.columns:
            emb_mat = Loaders._load_emb_matrix_from_single_col(df_all, col="embedding")
        else:
            raise ValueError("Law corpus CSVs must have 'emb_0..' or 'embedding' column(s).")

        emb_mat = Helpers.normalize_matrix(emb_mat)
        return df_all, emb_mat

    @staticmethod
    def load_case_chunks(file):
        df = pd.read_csv(file)

        if any(c.startswith("emb_") for c in df.columns):
            emb_mat = Loaders._load_emb_matrix_from_cols(df, prefix="emb_")
        elif "embedding" in df.columns:
            emb_mat = Loaders._load_emb_matrix_from_single_col(df, col="embedding")
        else:
            raise ValueError("Case chunks CSV must have 'emb_0..' or 'embedding' column(s).")

        emb_mat = Helpers.normalize_matrix(emb_mat)
        return df, emb_mat

    @staticmethod
    def _load_emb_matrix_from_single_col(df: pd.DataFrame, col: str = "embedding") -> np.ndarray:
        if col not in df.columns:
            raise ValueError(f"Embedding column '{col}' not found in dataframe.")
        vectors = []
        for s in df[col].tolist():
            if isinstance(s, str):
                v = np.array(ast.literal_eval(s), dtype=np.float32)
            else:
                v = np.array(s, dtype=np.float32)
            vectors.append(v)
        return np.stack(vectors, axis=0)