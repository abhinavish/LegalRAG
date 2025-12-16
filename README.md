# LegalRAG: Immigration Legal Chatbot
---

## Project Overview

**LegalRAG** is a Retrieval-Augmented Generation (RAG) system that provides accessible immigration law assistance by combining real court opinions with US Code statutes in a knowledge graph, powered by LLMs.

- **Project Type:** Full-stack RAG application with knowledge graph construction

---

## What It Does

LegalRAG helps self-represented immigrants understand US immigration procedures and rights by:
- **Retrieving** relevant case law from immigration court decisions
- **Connecting** those cases to applicable statutes in the US Code
- **Generating** conversational explanations in simple, accessible language
- **Citing** legal sources so users know where information comes from

**Example Queries:**
- "How does having a US citizen spouse help my case?"
- "What forms do I need to file for asylum?"
- "How long does the process typically take?"

The system answers with specific legal precedents and statutes, not just general information.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Neo4j instance (local or cloud)
- API keys: Google Gemini (required), OpenAI/Hugging Face (optional)
- CourtListener API key (for case ingestion)
- uv package manager

### Quick Start

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/CS6220-LegalRAG.git
cd CS6220-LegalRAG
```

2. **Install Dependencies with uv**
```bash
uv sync
# or
uv pip install -r requirements.txt
```

3. **Configure Environment**
```bash
cp .envexample .env
# Edit .env with your API keys:
# NEO4J_URI=bolt://localhost:7687
# PROVIDER_API_KEY=your_gemini_key
# COURTLISTENER_API_KEY=your_key
```

4. **Start Neo4j Desktop**


5. **Run Executable Scripts**
```bash
# Ingest cases
python ingest_cases.py

# Ingest US Code
python ingest_us_code.py

# Test chatbot
python main.py

# Evaluate RAG quality
python evaluation.py
```

---

## Complete Project Structure

```
└── CS6220-LegalRAG
    
    └── benchmark_models
        ├── benchmark_utils.py
        ├── models.py
        ├── rag_executor.py
    └── case_embeddings
        ├── case_chunker.py
    └── clients
        └── __pycache__
        ├── __init__.py
        ├── courtListener_client.py
        ├── graphiti_client.py
    └── immigration_test
        ├── generate_test_cases.py
    └── models
        └── __pycache__
        ├── __init__.py
        ├── config.py
        ├── entities.py
        ├── relations.py
    └── prompts
        └── __pycache__
        ├── __init__.py
        ├── immigration_prompts.py
    └── providers
        └── __pycache__
        ├── __init__.py
        ├── base.py
        ├── gemini.py
        ├── hugging_face.py
        ├── openai.py
        ├── registry.py
    └── results
        └── generation
            └── eval_results
                ├── cbr_retrieval_k3_unified_eval_results.csv
                ├── lawRAG_retrieval_k_eval_results.csv
                ├── naive_retrieval_k3_eval_results.csv
                ├── no_rag_llm_eval_results.csv
            ├── cbr_retrieval_k3_unified.csv
            ├── lawRAG_retrieval_k.csv
            ├── naive_retrieval_k3.csv
            ├── no_rag_llm.csv
    └── scripts
        └── __pycache__
        └── data_structure
            └── __pycache__
            ├── graph.py
            ├── node.py
        ├── cluster_visualizer.py
        ├── gemini_batch_ai_studio.py
        ├── gemini_prompt.py
        ├── ingest_us_code.py
        ├── neo4j_integration_uscode.py
        ├── us_code_extracter.py
        ├── uslm_converter.py
        ├── xml_extractor.py
        ├── xml_processor.py
    └── services
        └── __pycache__
        ├── __init__.py
        ├── case_service.py
        ├── chatbot_service.py
    └── testing
        └── data
            ├── immigration_test_dataset.json
        └── results
            ├── results.json
    └── us_code_embeddings
        ├── us_code_chunker.py
    ├── .envexample
    ├── .gitignore
    ├── .python-version
    ├── clusters.png
    ├── data_parser.py
    ├── db_analysis.json
    ├── db_analysis.py
    ├── eval.py
    ├── evaluation.py
    ├── explore_eval.ipynb
    ├── ingest_cases.py
    ├── main.py
    ├── pyproject.toml
    ├── README.md
    ├── requirements.txt
    └── uv.lock
```

---

## Executable

- main.py
- ingest_cases.py
- evaluation.py
- db_analysis.py

---

## Core Components

### 1. **clients/** 

#### graphiti_client.py
**Purpose:** Async episode ingestion and knowledge graph search


#### courtListener_client.py
**Purpose:** CourtListener API integration for fetching court opinions


---

### 2. **services/** 

#### chatbot_service.py
**Purpose:** Multi-turn conversation engine with knowledge graph retrieval

#### case_service.py
**Purpose:** Legal opinion parsing into structured format



---

### 3. **models/** 

#### entities.py
**Purpose:** Pydantic models for legal entities


#### relations.py
**Purpose:** Pydantic models for entity relationships


#### config.py
**Purpose:** Entity and edge type mappings

---

### 4. **prompts/** 

#### immigration_prompts.py
**Purpose:** System prompts, formatting templates, LangChain integration





---

### 5. **providers/** 

**Purpose:** Support multiple LLM providers with unified interface

**Providers:**

- **Gemini Provider**
- **OpenAI Provider**
- **Hugging Face Provider**
- **Registry**

---

### 6. **scripts/**

#### Data Structures

- **graph.py**
- **node.py**


#### US Code Processing

- **us_code_extracter.py**
- **uslm_converter.py**
- **xml_processor.py**
- **xml_extractor.py**

#### Neo4j Integration

- **neo4j_integration_uscode.py**


#### Summarization & Visualization

- **gemini_batch_ai_studio.py**
- **gemini_prompt.py**
- **cluster_visualizer.py**


---

### 7. **utils/**

#### ingest_cases.py
**Purpose:** Ingest immigration court cases from CourtListener



#### data_parser.py
**Purpose:** Parse raw US Code JSON data


#### evaluation.py
**Purpose:** Measure RAG retrieval quality


#### main.py
**Purpose:** Interactive test suite with 4 modes

---

### 8. **benchmark_models/** - RAG Benchmarking

#### benchmark_utils.py
**Purpose:** Utility classes for LegalBERT-based embedding, citation parsing, and retrieval evaluation

#### models.py
**Purpose:** RAG system implementations for benchmarking (BaseRAG, NaiveRAG, CBRRAG)

#### rag_executor.py
**Purpose:** Entry point for running RAG benchmarks and comparative testing

---

### 9. **case_embeddings/** - Case Processing

#### case_chunker.py
**Purpose:** Break court opinions into semantic chunks with LegalBERT embeddings

---

### 10. **us_code_embeddings/** - Statute Processing

#### us_code_chunker.py
**Purpose:** Download, parse, and embed US Code statutes with legal structure awareness

---

### 11. **immigration_test/** - Test Dataset Generation

#### generate_test_cases.py
**Purpose:** Auto-generate Q&A pairs from case law using Gemini

---

## Data Sources & Datasets

### 1. CourtListener Immigration Cases
**Source:** https://www.courtlistener.com/api/
**Description:** US federal court immigration opinions
**Size:** 250 cases
**Format:** JSON with full opinion text
**Update Frequency:** Real-time (daily new cases)
**Quality:** Official federal court records



### 2. US Code (USLM Format)
**Source:** https://github.com/usgpo/uslm
**Description:** US Code statutes in XML format
**Titles:** 8, 22, 42, 18, 19, 18A, 50, 50A (immigration-relevant)
**Size:** 50 MB raw XML
**Update Frequency:** Annually
**Quality:** Official government source



### 3. Test Dataset
**File:** `testing/data/immigration_test_dataset.json`
**Description:** 100 curated immigration law Q&A pairs
**Size:** 50 KB
**Format:** JSON with question, golden context, query type
**Coverage:** Procedures, eligibility, forms, timeline
**Quality:** Manually curated, verified legal information



---

## Open Source Dependencies

| Package | GitHub | Purpose |
|---------|--------|---------|
| **Graphiti** | https://github.com/getzep/graphiti | Knowledge graph framework, entity extraction | 
| **Neo4j Driver** | github.com/neo4j/neo4j-python-driver | Graph DB client, Cypher queries | 
| **LangChain** | github.com/langchain-ai/langchain | LLM orchestration, prompting | 
| **Google Generative AI** | https://ai.google.dev| Gemini API, embeddings, reranker | 
| **SentenceTransformers** | github.com/UKPLab/sentence-transformers | Embedding generation |


---
