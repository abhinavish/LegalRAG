import os
import ast
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# LLM and RAGAS imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset

# Metrics imports
from bert_score import score
import textstat

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

def load_csv_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df

def format_contexts(row: pd.Series) -> List[str]:
    contexts = []
    try:
        law_statutes = ast.literal_eval(row.get('top_k_law_statutes', '[]'))
        if isinstance(law_statutes, list):
            for stat in law_statutes:
                if isinstance(stat, dict):
                    title = stat.get('title', '')
                    chunk = stat.get('chunk', '')
                    if chunk:
                        if isinstance(title, tuple):
                            title_str = f"{title[0]} {title[1]} {title[2]}"
                        else:
                            title_str = str(title)
                        contexts.append(f"{title_str}: {chunk}")
    except (ValueError, SyntaxError):
        pass
    
    # Parse case statutes/chunks
    try:
        case_chunks = ast.literal_eval(row.get('top_k_case_statutes', '[]'))
        if isinstance(case_chunks, list):
            for case in case_chunks:
                if isinstance(case, dict):
                    title = case.get('title', '')
                    chunk = case.get('chunk', '')
                    if chunk:
                        contexts.append(f"{title}: {chunk}")
    except (ValueError, SyntaxError):
        pass
    
    return contexts

def generate_answer(question: str, contexts: List[str], llm: ChatGoogleGenerativeAI, optimize_readability: bool = False) -> str:
    context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
    
    if optimize_readability:
        # Readability-optimized prompt for LawRAG
        prompt = f"""Use the following context to answer the question. Write your answer in a clear, accessible way that is easy to understand.

Guidelines for your answer:
- Use shorter sentences (aim for 15-20 words per sentence)
- Prefer simple, everyday words over complex legal jargon when possible
- Write clearly and directly
- Break down complex ideas into smaller parts
- Use active voice when possible
- If the context does not contain enough information to answer the question, say so clearly

Context:
{context_text}

Question: {question}

Answer:"""
    else:
        # Standard prompt for other models
        prompt = f"""Use the following context to answer the question. If the context does not contain enough information to answer the question, say so.

Context:
{context_text}

Question: {question}

Answer:"""
    
    try:
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"ERROR: {str(e)}"

def generate_answer_no_rag(question: str, llm: ChatGoogleGenerativeAI) -> str:
    """Generate answer without any retrieved context (closed-book)."""
    prompt = f"""Answer the following question based on your knowledge.

Question: {question}

Answer:"""
    
    try:
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"ERROR: {str(e)}"

def setup_gemini_judge(model_name: str = None) -> LangchainLLMWrapper:
    if model_name is None:
        model_name = GEMINI_MODEL
    
    chat_model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0
    )
    gemini_llm = LangchainLLMWrapper(chat_model)
    faithfulness.llm = gemini_llm 
    return gemini_llm

def calculate_bertscore_batch(generated_list: List[str], reference_list: List[str]) -> List[float]:
    if not generated_list or not reference_list:
        return []
    
    try:
        P, R, F1 = score(
            generated_list, 
            reference_list, 
            model_type='microsoft/deberta-v3-base', 
            lang='en', 
            verbose=False,
            device='cpu' 
        )
        return [float(f.item()) for f in F1]
    except Exception as e:
        print(f"Deberta failed, falling back to roberta: {e}")
        try:
            P, R, F1 = score(
                generated_list, 
                reference_list, 
                model_type='roberta-large', 
                lang='en', 
                verbose=False,
                device='cpu'
            )
            return [float(f.item()) for f in F1]
        except Exception:
            return [0.0] * len(generated_list)

def calculate_readability(text: str) -> Dict[str, float]:
    try:
        return {
            'grade_level': textstat.flesch_kincaid_grade(text),
            'reading_ease': textstat.flesch_reading_ease(text)
        }
    except:
        return {'grade_level': 0.0, 'reading_ease': 0.0}

def evaluate_dataset(csv_path: str, output_path: str, max_rows: Optional[int] = None):
    print(f"\nEvaluating: {csv_path}")
    df = load_csv_data(csv_path)
    
    if max_rows:
        df = df.head(max_rows)

    is_lawrag = "lawRAG" in csv_path.lower() or "lawrag" in csv_path.lower()
    if is_lawrag:
        print("Detected LawRAG model - using readability-optimized prompts")
    

    # Setup LLMs
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2)
    gemini_judge = setup_gemini_judge()
    
    # 1. GENERATION PHASE
    questions = []
    golden_answers = []
    generated_answers = []
    contexts_list = []
    is_no_rag = False  # Track if this is a no-RAG evaluation
    
    print("Generating answers...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        q = row.get('question', '')
        g = row.get('golden_answers', '')
        if not q: continue
            
        ctx = format_contexts(row)
        
        # Check if this is a no-RAG scenario (no contexts available)
        if not ctx or (len(ctx) == 0):
            is_no_rag = True
            gen_ans = generate_answer_no_rag(q, llm)
        else:
            gen_ans = generate_answer(q, ctx, llm, optimize_readability=is_lawrag)
        
        questions.append(q)
        golden_answers.append(g)
        generated_answers.append(gen_ans)
        contexts_list.append(ctx)

    # 2. EVALUATION PHASE (BATCHED)
    
    # BertScore
    print("Calculating BertScore...")
    bert_scores = calculate_bertscore_batch(generated_answers, golden_answers)
    
    # Readability
    print("Calculating Readability...")
    readability_scores = [calculate_readability(ans) for ans in generated_answers]
    

    # RAGAS Faithfulness (BATCHED)
    print("Calculating Faithfulness (RAGAS)...")
    faithfulness_scores = []

    # Filter to only rows with contexts for RAGAS evaluation
    questions_with_context = []
    generated_answers_with_context = []
    contexts_list_with_context = []
    golden_answers_with_context = []
    row_indices_with_context = []

    for i, ctx in enumerate(contexts_list):
        if ctx and len(ctx) > 0:
            questions_with_context.append(questions[i])
            generated_answers_with_context.append(generated_answers[i])
            contexts_list_with_context.append(ctx)
            golden_answers_with_context.append(golden_answers[i])
            row_indices_with_context.append(i)

    if len(questions_with_context) == 0:
        print("No rows with contexts found - skipping faithfulness calculation")
        faithfulness_scores = [None] * len(questions)
    else:
        # Construct RAGAS Dataset only for rows with contexts
        ragas_data = {
            'question': questions_with_context,
            'answer': generated_answers_with_context,
            'contexts': contexts_list_with_context,
            'ground_truth': golden_answers_with_context
        }
        ragas_dataset = Dataset.from_dict(ragas_data)
        
        try:
            # Evaluate the whole dataset at once
            ragas_results = evaluate(
                ragas_dataset, 
                metrics=[faithfulness],
                llm=gemini_judge
            )
            # Map results back to original indices
            faithfulness_scores = [None] * len(questions)
            for idx, result in zip(row_indices_with_context, ragas_results['faithfulness']):
                faithfulness_scores[idx] = result
        except Exception as e:
            print(f"RAGAS Evaluation failed: {e}")
            faithfulness_scores = [None] * len(questions)

    # 3. SAVE RESULTS
    results = []
    for i in range(len(questions)):
        result_row = {
            'question': questions[i],
            'golden_answer': golden_answers[i],
            'generated_answer': generated_answers[i],
            'retrieved_contexts': str(contexts_list[i]),
            'faithfulness_score': faithfulness_scores[i] if i < len(faithfulness_scores) else None,
            'bertscore_f1': bert_scores[i] if i < len(bert_scores) else 0.0,
            'readability_grade_level': readability_scores[i]['grade_level'],
            'readability_reading_ease': readability_scores[i]['reading_ease']
        }
        # Merge original data
        original_row = df.iloc[i].to_dict()
        # Update original with results
        original_row.update(result_row)
        results.append(original_row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    BASE_DIR = Path("/home/hice1/hpark339/scratch/CS6220-LegalRAG/data/generation")
    OUTPUT_DIR = Path("/home/hice1/hpark339/scratch/CS6220-LegalRAG/data/generation/eval_results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_files = ["naive_retrieval_k3.csv", "cbr_retrieval_k3_unified.csv", "lawRAG_retrieval_k.csv", "no_rag_llm.csv"]
    
    for csv_file in csv_files:
        csv_path = BASE_DIR / csv_file
        if csv_path.exists():
            evaluate_dataset(str(csv_path), str(OUTPUT_DIR / csv_file.replace('.csv', '_eval_results.csv')), max_rows=200)

