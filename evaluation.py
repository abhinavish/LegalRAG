import os
import asyncio
from dotenv import load_dotenv
from clients.graphiti_client import graphitiClient
from services.chatbot_service import ChatbotService
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import re
import json

load_dotenv()


neo4j_uri = os.getenv('NEO4J_URI')
neo4j_user = os.getenv('NEO4J_USER')
neo4j_password = os.getenv('NEO4J_PASSWORD')
llm_provider = os.getenv('LLM_PROVIDER')
gemini_key = os.getenv('PROVIDER_API_KEY')
llm_model = os.getenv('INGESTION_LLM')
embedding_model = os.getenv('INGESTION_EMBEDDING')
reranker_model = os.getenv('INGESTION_RERANKER')

SYS_PROMPT = """You are tasked with evaluating the quality of document retrieval in a RAG (Retrieval-Augmented Generation) pipeline.
                                
                                You will be given:
                                1. **Query**: The user's question
                                2. **Retrieved Context**: The documents/facts retrieved by the RAG system
                                3. **Golden Context**: The ideal documents/facts that should have been retrieved (ground truth)
                                
                                Your task is to evaluate the retrieval system across the following dimensions:
                                
                                ## EVALUATION CRITERIA
                                
                                ### 1. CONTEXT PRECISION (Score: 1-5)
                                What proportion of the retrieved context is actually relevant to answering the query?
                                - Score 5: All retrieved items are highly relevant (90-100% precision)
                                - Score 4: Most retrieved items are relevant (70-89% precision)
                                - Score 3: About half of retrieved items are relevant (50-69% precision)
                                - Score 2: Few retrieved items are relevant (30-49% precision)
                                - Score 1: Very few or no retrieved items are relevant (<30% precision)

                                ### 2. CONTEXT RECALL (Score: 1-5)
                                How many of the golden/expected context items were successfully retrieved?
                                - Score 5: Retrieved all or nearly all golden context items (90-100 recall)
                                - Score 4: Retrieved most golden context items (70-89 recall)
                                - Score 3: Retrieved about half of golden context items (50-69 recall)
                                - Score 2: Retrieved few golden context items (30-49 recall)
                                - Score 1: Retrieved very few or no golden context items (<30 recall)

                                ### 3. CONTEXT RELEVANCE (Score: 1-5)
                                How relevant is the retrieved context to the specific query, regardless of golden context?
                                - Score 5: All retrieved items directly address the query topic
                                - Score 4: Most retrieved items address the query with minor off-topic content
                                - Score 3: Some retrieved items address the query, significant noise present
                                - Score 2: Few retrieved items address the query
                                - Score 1: Retrieved items do not address the query

                                ### 4. COVERAGE (Score: 1-5)
                                Does the retrieved context cover all aspects needed to answer the query comprehensively?
                                - Score 5: Retrieved context covers all necessary aspects
                                - Score 4: Retrieved context covers most aspects, minor gaps
                                - Score 3: Retrieved context covers some aspects, notable gaps
                                - Score 2: Retrieved context has significant gaps
                                - Score 1: Retrieved context fails to cover critical aspects


                                **Calculate**: (Number of golden items retrieved / Total golden items) × 100

                                ## OUTPUT FORMAT

                                Provide your evaluation in the following JSON format:

                                {
                                  "precision": {
                                    "score": <1-5>,
                                    "percentage": <calculated precision %>,
                                    "relevant_count": <number of relevant items>,
                                    "total_retrieved": <total retrieved items>,
                                  },
                                  "context_recall": {
                                    "score": <1-5>,
                                    "retrieved_golden_items": <number of golden items found>,
                                    "total_golden_items": <total golden items>,
                                    "missing_items": ["<list of missing golden items>"],
                                  },
                                  "relevance": {
                                    "score": <1-5>,
                                  },
                                  "coverage": {
                                    "score": <1-5>,
                                    "percentage": <calculated coverage %>,
                                    "found_count": <golden items found>,
                                    "total_golden": <total golden items>,
                                    "missing_items": ["<list of missing golden context>"],
                                  },
                                  "overall": {
                                    "average_score": <average of 3 scores>,
                                    "pass": <true if average >= 2.5, false otherwise>,
                                  }
                                }

                                ## GUIDELINES
                                - Be objective and specific
                                - List which retrieved items are irrelevant (for precision)
                                - List which golden items are missing (for coverage)
                                - Focus on factual matching, not paraphrasing

                                Now evaluate the following retrieval system output:

                                """
                                
                                
DATASET_PATH="test_data/immigration_test_dataset.json"

def load_data():

    data = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data

async def eval(case, model, llm_judge, num_context=5):
    # Retrieve context
    results = await model.search_knowledge_graph(
        case['question'], 
        limit=20
        
    )
    facts = results.get('facts', [])[:num_context]
    nodes = results.get('nodes', [])[:num_context]

    retrieved = "**Retrieved:**\n"
    for i, fact in enumerate(facts, 1):
        retrieved += f"{i}. {fact}\n"
    
    for i, node in enumerate(nodes, 1):
        idx=len(results.get('facts', [])) + i
        name=node.get('name')
        summary=node.get('summary', '')
        retrieved += f"{idx}. {name} - {summary}\n"
   
    golden = case.get('golden_case_context', [])
    golden_str = "\n".join([f"- {item}" for item in golden])
    
    user_msg = f"""
                **Query:** {case['question']}

                {retrieved}

                **Golden Context:**
                {golden_str}
                """
    
    messages = [
        SystemMessage(content=SYS_PROMPT),
        HumanMessage(content=user_msg)
    ]
    
    response = await llm_judge.ainvoke(messages)

    text = response.content

    # Try to extract JSON inside ```json ... ```
    code_block = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)

    if code_block:
        text = code_block.group(1)   # Only the JSON content
    else:
        text = text.strip()          # Fallback (maybe it's raw JSON)

    # Parse JSON
    result = json.loads(text)
    
    # Add metadata
    result['case_name'] = case.get('case_name', 'Unknown')
    result['num_retrieved'] = len(facts) + len(nodes)
    result['num_golden'] = len(golden)
    
    return result

    

async def main():

    data=load_data()

    graphiti = await graphitiClient(
        neo4j_uri, neo4j_user, neo4j_password, 
        gemini_key, llm_model, embedding_model, 
        reranker_model, llm_provider, 
        use_huggingface_embedding=True
    )

    chatbot = ChatbotService(
        graphiti_client=graphiti,
        llm_api_key=gemini_key,
        model_name=llm_model,
        full_search=True

    )


    llm_judge = ChatGoogleGenerativeAI(
        model=llm_model,
        google_api_key=gemini_key,
        temperature=0.1
    )

    results = []

    for test in data:
        result = await eval(test, chatbot, llm_judge)
        results.append(result)
    
    # Save to JSON file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

        

if __name__=="__main__":
    asyncio.run(main())