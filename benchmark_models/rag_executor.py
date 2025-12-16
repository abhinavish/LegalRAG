import torch
from models import NaiveRAG, CBRRAG

# Test dataset. Each item has:
# - case_name
# - question
# - golden_answer
# - golden_context (list of US code/cfr cites)
# - golden_case_context (list of case names)
TEST_DATASET_PATH = "immigration_test/test_cases.jsonl"

# section_id, section_title, chunk_index, chunk, embedding vector
LAW_CHUNKS_FILES = [
    "us_code_embeddings/usc_title6.csv",
    "us_code_embeddings/usc_title8.csv",
    "us_code_embeddings/usc_title19.csv",
    "us_code_embeddings/usc_title22.csv",
]

# case_name,chunk_id,chunk_text,emb_0..emb_767  (or embedding col)
CASE_CHUNKS_FILE = "case_embeddings/cases_chunked_with_embeddings.csv"
K_VALUES = [1, 3, 5, 10]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

naive_rag = NaiveRAG(TEST_DATASET_PATH, CASE_CHUNKS_FILE, LAW_CHUNKS_FILES, K_VALUES, DEVICE, MODEL_NAME)
naive_rag.execute(3)

cbr_rag = CBRRAG(TEST_DATASET_PATH, CASE_CHUNKS_FILE, K_VALUES, DEVICE, MODEL_NAME)
cbr_rag.execute(3)