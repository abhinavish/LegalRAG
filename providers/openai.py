from .base import baseGraphitiProvider
from .registry import graphitiProviderRegistry

from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

@graphitiProviderRegistry.register('openai')
class openAIProvider(baseGraphitiProvider):
    def getLlmClient(self, api_key: str, model: str = None, small_model=None):
        config = LLMConfig(
            api_key=api_key,
            model=model,
            small_model=small_model
        )
        return OpenAIClient(config=config)
    
    def getEmbedder(self, api_key: str, embedding_model: str = None):
        config = OpenAIEmbedderConfig(
            api_key=api_key,
            embedding_model=embedding_model
        )
        return OpenAIEmbedder(config=config)
    
    def getReranker(self, api_key: str, reranker_model: str = None, small_model=None):
        config = LLMConfig(
            api_key=api_key,
            model=reranker_model,
            small_model=small_model
        )
        return OpenAIRerankerClient(config=config)