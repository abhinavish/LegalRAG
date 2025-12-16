import os
import asyncio

from dotenv import load_dotenv
from clients import graphitiClient
from services import caseParser
from clients import courtListenerClient
from graphiti_core.nodes import EpisodeType

from models.config import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES
from scripts.ingest_us_code import ingest_us_code
import datetime

load_dotenv()


neo4j_uri = os.getenv('NEO4J_URI')
neo4j_user = os.getenv('NEO4J_USER')
neo4j_password = os.getenv('NEO4J_PASSWORD')
db_name = os.getenv('DB_NAME', "neo4j")
llm_provider=os.getenv('LLM_PROVIDER')
gemini_key=os.getenv('PROVIDER_API_KEY')
llm_model=os.getenv('INGESTION_LLM')
embedding_model=os.getenv('INGESTION_EMBEDDING')
reranker_model=os.getenv("INGESTION_RERANKER")
courtlistener_api=os.getenv("COURTLISTENER_API_KEY")
small_model=os.getenv("SMALL_MODEL")

def getEpisodes(query="immigration",num_cases=5, skip_results=0):
    courtListener=courtListenerClient(courtlistener_api)
    caseTransformer=caseParser()
    cases=courtListener.fetchOpinions(search_query=query, max_res=num_cases, skip_results=skip_results)
    
    print(f"Fetched {len(cases)} opinions with full text\n")

    episodes=[]
    for case in cases:
        ep=caseTransformer.opinion_to_episode(case)
        episodes.append(ep)

    return episodes

async def main():
    gp=None
    try:
        gp = await graphitiClient(neo4j_uri, neo4j_user, neo4j_password, gemini_key, llm_model, embedding_model, reranker_model, small_model=small_model, llm_provider=llm_provider, use_huggingface_embedding=True)

        episodes = getEpisodes(num_cases=5, skip_results=150)

        await gp.addEpisodes(episodes=episodes, sleep=10)
        

    finally:
        if gp is not None:
            await gp.close()


if __name__ == "__main__":
    #asyncio.run(main())
    ingest_us_code(neo4j_uri, neo4j_user, neo4j_password, db_name, gemini_key, llm_model)
    pass