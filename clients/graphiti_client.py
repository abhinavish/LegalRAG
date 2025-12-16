
import json
import asyncio
from datetime import datetime, timezone
from logging import INFO

from graphiti_core import Graphiti
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

from graphiti_core.nodes import EpisodeType


from providers import graphitiProviderRegistry, baseGraphitiProvider, huggingFaceConfig, huggingFaceEmbedding
from models import ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP

from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
from datetime import datetime, timezone
import time

REGISTRY=graphitiProviderRegistry()

class graphitiClient():
    
    def __init__(self,
                n4j_uri: str,
                n4j_user: str,
                n4j_password:str,
                api_key:str,
                ingestion_llm:str,
                embedding_model:str,
                reranker_llm:str,
                small_model:str,
                llm_provider: str='google',
                use_huggingface_embedding: bool= False):
        
        
        self.llm_provider: baseGraphitiProvider = REGISTRY.get(llm_provider)()
        self.llm_client=self.llm_provider.getLlmClient(api_key=api_key, model=ingestion_llm, small_model=small_model)

        if not use_huggingface_embedding:
            self.embedding_client=self.llm_provider.getEmbedder(api_key=api_key, embedding_model=embedding_model)
        else:
        
            emb_config=huggingFaceConfig(embedding_model=embedding_model)
            self.embedding_client=huggingFaceEmbedding(config=emb_config)


        self.reranker_client=self.llm_provider.getReranker(api_key=api_key, reranker_model=reranker_llm, small_model=small_model)

        self.graphiti=Graphiti(uri=n4j_uri, \
                               user=n4j_user,\
                               password=n4j_password,\
                               llm_client=self.llm_client,\
                               embedder=self.embedding_client,\
                               cross_encoder=self.reranker_client)
        
    
    
    def __await__(self):
        return self._async_init().__await__()
        
    async def _async_init(self):
        await self.graphiti.build_indices_and_constraints()
        return self
    
    async def close(self):
        await self.graphiti.close()

    async def addEpisodes(self, episodes, sleep=60, max_retries=3):
        for i, episode in enumerate(episodes):
            
            retry_num = 0
            retry_time = 120  
            done = False

            print(f"\nProcessing episode: {episode.get("id",[])}")

            while retry_num < max_retries and not done:
                try:
                    await self.graphiti.add_episode(
                        name=f'Case_{episode.get("id", i)}',
                        episode_body=episode['content'] if isinstance(episode['content'], str) else json.dumps(episode['content']),
                        source_description=episode['description'],
                        reference_time=episode.get("valid_at", datetime.now(timezone.utc)),
                        group_id="legal_cases",
                        entity_types=ENTITY_TYPES,
                        excluded_entity_types=["Entity"],     
                        edge_types=EDGE_TYPES,         
                        edge_type_map=EDGE_TYPE_MAP     
                    )

                    done = True
                    print(f'Added episode: Case_{episode.get("id", i)} - {episode["description"]}')

                except Exception as e:
                    error_str = str(e).lower()

                    if any(keyword in error_str for keyword in ['429', 'rate limit', 'resource_exhausted', 'quota']):
                        retry_num += 1

                        if retry_num < max_retries:
                            print(f"\nRate limit error on episode {i} (attempt {retry_num}/{max_retries})")
                            print(f"Error: {str(e)}")
                            print(f"Retrying in {retry_time} seconds")

                            for rem in range(retry_time, 0, -1):
                                mins, secs = divmod(rem, 60)
                                timer = f"{mins:02d}:{secs:02d}"
                                print(f"\rRetry cooldown: {timer}  ", end='', flush=True)
                                await asyncio.sleep(1)
                            print(f"\rRetry cooldown complete")

                           
                            retry_time = min(retry_time, 600)  
                        else:
                            print(f"\nFailed episode {i} after {max_retries} attempts due to rate limit")
                            break
                    else:
                        
                        print(f"\nFailed to add episode {i}: {str(e)}")
                        raise
                    
            if done and i < len(episodes) - 1:
                print(f"\n⏸Rate limit cooldown: {sleep} seconds")

                for rem in range(sleep, 0, -1):
                    mins, secs = divmod(rem, 60)
                    timer = f"{mins:02d}:{secs:02d}"
                    print(f"\rTime remaining: {timer}  ", end='', flush=True)
                    await asyncio.sleep(1)
                print(f"\rCooldown complete\n")


    async def addEpisodesParallel(self, episodes, batch_size: int = 20, sleep: int = 100, max_retries: int = 3):
    
    
        total = (len(episodes) - 1) // batch_size + 1
    
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i + batch_size]
            curr_batch = i // batch_size + 1
    
            print(f"\n{'='*70}")
            print(f"BATCH {curr_batch}/{total} - Processing {len(batch)} episodes")
            print(f"{'='*70}")
    
            for attempt in range(max_retries):
                try:
                    tasks = []
                    for j, episode in enumerate(batch):
                        task = self.graphiti.add_episode(
                            name=f'Case_{episode.get("id", i+j)}',
                            episode_body=episode['content'],
                            source_description=episode['description'],
                            reference_time=episode.get("valid_at", datetime.now(timezone.utc)),
                            group_id="legal_cases",
                            entity_types=ENTITY_TYPES,
                            excluded_entity_types=["Entity"],     
                            edge_types=EDGE_TYPES,         
                            edge_type_map=EDGE_TYPE_MAP
                        )
                        tasks.append(task)
    
                    await asyncio.gather(*tasks)
                    
                    print(f"\nBatch {curr_batch} Complete - Added {len(batch)} cases:")
                    print(f"{'─'*70}")
    
                    for j, episode in enumerate(batch, 1):
                        name = episode.get('metadata', {}).get('case_name', 'Unknown Case')
                        court = episode.get('metadata', {}).get('court', 'Unknown Court')
                        date_filed = episode.get('metadata', {}).get('date_filed', 'Unknown Date')
    
                        if len(name) > 50:
                            name = name[:47] + "..."
    
                        print(f"  {j:2d}. {name}")
                        print(f"      {court} | {date_filed}")
    
                    print(f"{'─'*70}")
                    break
                
                except Exception as e:
                    if attempt < max_retries - 1:
                       
                        print(f"\nBatch {curr_batch} failed (attempt {attempt + 1}/{max_retries})")
                        print(f"   Error: {str(e)}")
                        print(f"   Retrying in {sleep} seconds")
                        
                        for rem in range(sleep, 0, -1):
                            m, s = divmod(rem, 60)
                            timer = f"{m:02d}:{s:02d}"
                            print(f"\rRetry cooldown: {timer}  ", end='', flush=True)
                            await asyncio.sleep(1)
                        print(f"\rRetry cooldown complete\n")
                    else:
                        
                        print(f"\nBatch {curr_batch} failed after {max_retries} attempts")
                        print(f"Final error: {str(e)}")
                        raise
                    
            if curr_batch < total:
                print(f"\nRate limit cooldown: {sleep} seconds")
    
                for rem in range(sleep, 0, -1):
                    m, s = divmod(rem, 60)
                    timer = f"{m:02d}:{s:02d}"
                    print(f"\rTime remaining: {timer}  ", end='', flush=True)
                    await asyncio.sleep(1)
    
                print(f"\rCooldown complete!\n")
    
        print(f"\n{'='*70}")
        print(f"INGESTION COMPLETE - All {len(episodes)} episodes added successfully")
        print(f"{'='*70}\n")

    async def addEpisodesBulk(self, episodes):
        print(f"Adding {len(episodes)} episodes in bulk")

        
        raw_ep = []

        for i, ep in enumerate(episodes):
            
            raw_episode = RawEpisode(  
                name=f'Case_{ep.get("id", i)}',
                content=ep['content'],
                source_description=ep['description'],
                source=EpisodeType.text,
                reference_time=ep.get("valid_at", datetime.now(timezone.utc))
            )
            raw_ep.append(raw_episode)

        
        await self.graphiti.add_episode_bulk(
            raw_ep,  
            group_id="legal_cases",
            entity_types=ENTITY_TYPES,
            excluded_entity_types=["Entity"],
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP
        )

        print(f'Bulk added {len(episodes)} episodes')

    async def searchGraph(self, query: str,group_id:str = "legal_cases",limit: int = 10):
        
        results = await self.graphiti.search(
            query=query,
            group_ids=[group_id] if group_id else None,
            num_results=limit
        )
    
        return results
    
    async def searchGraphFull(self, query: str,
                          recipe=COMBINED_HYBRID_SEARCH_RRF,
                          group_id: str = "legal_cases",
                          limit: int = 10):
        results = await self.graphiti._search(
            query=query,
            config=recipe,
            group_ids=[group_id] if group_id else None
        )

        return results