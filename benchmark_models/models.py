import json
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from benchmark_utils import Loaders, Helpers

class BaseRAG():
    def __init__(self, test_path, case_path, k_values, device, model_name):
        self.test_path = test_path
        self.case_path = case_path
        self.k_values = k_values
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
    
    def load(self):
        pass

    def eval(self):
        pass

    def build_csv_outputs(self):
        pass

    def execute(self):
        pass

class NaiveRAG(BaseRAG):
    def __init__(self, test_path, case_path, law_path, k_values, device, model_name):
        super().__init__(test_path, case_path, k_values, device, model_name)
        self.law_path = law_path

    def load(self):
        print("Loading LegalBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        print("Loading test dataset...")
        test_items = Loaders.load_test_dataset(self.test_path)
        print(f"Loaded {len(test_items)} test questions.")

        print("Loading law corpus chunks...")
        law_df, law_embs = Loaders.load_law_chunks(self.law_path)
        print(f"Law corpus chunks: {len(law_df)}")

        print("Loading case chunks...")
        case_df, case_embs = Loaders.load_case_chunks(self.case_path)
        print(f"Case chunks: {len(case_df)}")

        return test_items, law_df, law_embs, case_df, case_embs

    def eval(self, test_items, law_df, law_embs, case_df, case_embs):
        law_metrics = {k: {"precision": [], "recall": []} for k in self.k_values}
        case_metrics = {k: {"precision": [], "recall": []} for k in self.k_values}

        MAX_K = max(self.k_values)
        all_results = []

        print("Evaluating retrieval for Naive RAG...")
        for item in tqdm(test_items, desc="Questions"):
            question = item["question"]
            golden_triples = set()
            for cite in item.get("golden_context", []):
                parsed = Helpers.parse_golden_cite(cite)
                if parsed and parsed["section_number"]:
                    triple = (
                        parsed["title_type"],
                        parsed["title_number"],
                        parsed["section_number"],
                    )
                    golden_triples.add(triple)

            golden_case_raw = item.get("golden_case_context", [])
            golden_case_set = {Helpers.normalize_case_name(c) for c in golden_case_raw if c}

            # Embed question
            q_vec = Helpers.embed_text(self.tokenizer, self.device, self.model, question)
            q_vec_norm = Helpers.normalize_vector(q_vec)

            # corpus retrieval
            law_sims = law_embs @ q_vec_norm
            top_idx = np.argpartition(-law_sims, MAX_K - 1)[:MAX_K]
            top_idx = top_idx[np.argsort(-law_sims[top_idx])]

            retrieved_statute_chunks = []
            for idx in top_idx:
                row = law_df.iloc[idx]
                filename = str(row["_source_file"])
                title_type, title_number = Helpers.parse_corpus_title_from_filename(filename)
                section_number = Helpers.parse_corpus_section_number(row["section_title"])
                
                retrieved_statute_chunks.append({
                    "title": (title_type, title_number, section_number),
                    "chunk": row["chunk"] if "chunk" in row else row.get("text", "")
                })

            per_q_law_prec = {}
            per_q_law_rec = {}

            for k in self.k_values:
                p, r = Helpers.precision_recall_at_k(golden_triples, retrieved_statute_chunks, k)
                law_metrics[k]["precision"].append(p)
                law_metrics[k]["recall"].append(r)
                per_q_law_prec[k] = p
                per_q_law_rec[k] = r

            # case retrieval
            case_sims = case_embs @ q_vec_norm
            top_idx_case = np.argpartition(-case_sims, MAX_K - 1)[:MAX_K]
            top_idx_case = top_idx_case[np.argsort(-case_sims[top_idx_case])]

            retrieved_case_chunks = []
            for idx in top_idx_case:
                row = case_df.iloc[idx]
                cname_raw = str(row["case_name"])
                retrieved_case_chunks.append({
                    "title": Helpers.normalize_case_name(cname_raw),
                    "chunk": row["chunk_text"] if "chunk_text" in row else row.get("text", "")
                })

            per_q_case_prec = {}
            per_q_case_rec = {}

            for k in self.k_values:
                p, r = Helpers.precision_recall_at_k(golden_case_set, retrieved_case_chunks, k)
                case_metrics[k]["precision"].append(p)
                case_metrics[k]["recall"].append(r)
                per_q_case_prec[k] = p
                per_q_case_rec[k] = r

            # store per-question results for CSVs
            all_results.append({
                "question": question,
                "golden_answer": item.get("golden_answer", ""),
                "golden_law_context": list(golden_triples),
                "golden_case_context": list(golden_case_set),
                "retrieved_statutes": retrieved_statute_chunks,
                "retrieved_case_names": retrieved_case_chunks,
                "law_precision": per_q_law_prec,
                "law_recall": per_q_law_rec,
                "case_precision": per_q_case_prec,
                "case_recall": per_q_case_rec,
            })

        return law_metrics, case_metrics, all_results

    def build_csv_outputs(self, test_items, all_results, k, out_dir="rag_outputs"):
        """
        Each row corresponds to a single question.
        """
        Path(out_dir).mkdir(exist_ok=True)
        
        retrieved_data = []

        for res in all_results:
            q = res["question"]

            retrieved_law = res["retrieved_statutes"][:k]
            
            any_law_hit = int(res["law_recall"][k] > 0)

            # case
            retrieved_case = res["retrieved_case_names"][:k]
            any_case_hit = int(res["case_recall"][k] > 0)

            retrieved_data.append({
                "question": q,
                "golden_answers": res["golden_answer"],
                "golden_law_context": res["golden_law_context"],
                "golden_case_context": res["golden_case_context"],
                "top_k_law_statutes": retrieved_law,
                "top_k_case_statutes": retrieved_case,
                "any_correct_law_retrieved": any_law_hit,
                "any_correct_case_retrieved": any_case_hit,
                "any_correct_retrieved": int(any_law_hit or any_case_hit),
            })

            pd.DataFrame(retrieved_data).to_csv(
                f"{out_dir}/naive_retrieval_k{k}.csv", index=False
            )

        print(f"Saved CSV files for k={k}")

    def execute(self, k):
        test_items, law_df, law_embs, case_df, case_embs = self.load()
        law_metrics, case_metrics, all_results = self.eval(
            test_items, law_df, law_embs, case_df, case_embs
        )

        # summary printouts
        Helpers.summarize_metrics("Law (USC/CFR triple match)", law_metrics)
        Helpers.summarize_metrics("Cases", case_metrics)

        # ---- NEW: write CSV outputs ----
        self.build_csv_outputs(test_items, all_results, k)

import json
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from benchmark_utils import Loaders, Helpers

class CBRRAG(BaseRAG):
    def __init__(self, test_path, case_path, k_values, device, model_name,
                 w_intra=0.40, w_inter=0.60):
        super().__init__(test_path, case_path, k_values, device, model_name)
        self.w_intra = w_intra
        self.w_inter = w_inter

    def load(self):
        print("Loading LegalBERT model for CBR-RAG...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        print("Loading test dataset...")
        test_items = Loaders.load_test_dataset(self.test_path)
        print(f"Loaded {len(test_items)} test questions.")

        print("Loading case chunks...")
        case_df, case_embs = Loaders.load_case_chunks(self.case_path)
        # case_embs are already normalized in Loaders.load_case_chunks
        retrieval_embs = case_embs

        if "chunk_text" not in case_df.columns:
            raise ValueError("CBRRAG requires a 'chunk_text' column in the case chunks CSV.")

        print("Computing matching embeddings for case chunks (INTRA)...")
        match_embs_list = []
        for txt in tqdm(case_df["chunk_text"].tolist(), desc="CBR matching embeddings"):
            v = Helpers.embed_text(self.tokenizer, self.device, self.model, txt + "\nType: Matching")
            match_embs_list.append(Helpers.normalize_vector(v))
        match_embs = np.stack(match_embs_list, axis=0)

        return test_items, case_df, retrieval_embs, match_embs

    # Evaluation
    def eval(self, test_items, case_df, retrieval_embs, match_embs):
        case_metrics = {k: {"precision": [], "recall": []} for k in self.k_values}
        MAX_K = max(self.k_values)
        all_results = []

        print("Evaluating CBR-RAG")
        for item in tqdm(test_items, desc="Questions"):
            question = item["question"]

            # Build golden case set
            golden_case_raw = item.get("golden_case_context", [])
            golden_case_set = {
                Helpers.normalize_case_name(c) for c in golden_case_raw if c
            }

            # Question embeddings (CBR)
            q_match = Helpers.embed_text(self.tokenizer, self.device, self.model, question + "\nType: Matching")
            q_match = Helpers.normalize_vector(q_match)

            q_ret = Helpers.embed_text(self.tokenizer, self.device, self.model, question + "\nType: Retrieval")
            q_ret = Helpers.normalize_vector(q_ret)

            # Hybrid scores: W_INTRA * INTRA + W_INTER * INTER
            intra_scores = match_embs @ q_match          
            inter_scores = retrieval_embs @ q_ret     
            hybrid_scores = self.w_intra * intra_scores + self.w_inter * inter_scores

            # Top-K retrieval indices
            top_idx = np.argpartition(-hybrid_scores, MAX_K - 1)[:MAX_K]
            top_idx = top_idx[np.argsort(-hybrid_scores[top_idx])]

            # Build retrieved case chunks
            retrieved_case_chunks = []
            for idx in top_idx:
                row = case_df.iloc[idx]
                cname_raw = str(row["case_name"])
                retrieved_case_chunks.append({
                    "title": Helpers.normalize_case_name(cname_raw),
                    "chunk": row["chunk_text"] if "chunk_text" in row else row.get("text", "")
                })

            # Metrics for each k
            per_q_case_prec = {}
            per_q_case_rec = {}
            for k in self.k_values:
                p, r = Helpers.precision_recall_at_k(golden_case_set, retrieved_case_chunks, k)
                case_metrics[k]["precision"].append(p)
                case_metrics[k]["recall"].append(r)
                per_q_case_prec[k] = p
                per_q_case_rec[k] = r

            # Store per-question results
            all_results.append({
                "question": question,
                "golden_answer": item.get("golden_answer", ""),
                "golden_case_context": list(golden_case_set),
                "retrieved_case_chunks": retrieved_case_chunks,
                "case_precision": per_q_case_prec,
                "case_recall": per_q_case_rec,
            })

        return case_metrics, all_results

    # csv output
    def build_csv_outputs(self, test_items, all_results, k, out_dir="rag_outputs_cbr"):
        """
        Each row corresponds to a single question.
        Only case retrieval is recorded. No law outputs.
        """
        Path(out_dir).mkdir(exist_ok=True)

        rows = []
        for res in all_results:
            q = res["question"]

            retrieved_case = res["retrieved_case_chunks"][:k]
            any_case_hit = int(res["case_recall"][k] > 0)

            rows.append({
                "question": q,
                "golden_answers": res["golden_answer"],
                "golden_case_context": res["golden_case_context"],
                "top_k_case_chunks": retrieved_case,
                "any_correct_case_retrieved": any_case_hit,
            })

        df = pd.DataFrame(rows)
        df.to_csv(f"{out_dir}/cbr_retrieval_k{k}.csv", index=False)
        print(f"Saved CSV for k={k}")

    # execute
    def execute(self, k):
        test_items, case_df, retrieval_embs, match_embs = self.load()
        case_metrics, all_results = self.eval(test_items, case_df, retrieval_embs, match_embs)

        Helpers.summarize_metrics("Cases (CBR-RAG hybrid)", case_metrics)
        self.build_csv_outputs(test_items, all_results, k)