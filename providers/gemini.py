
from .registry import graphitiProviderRegistry
from .base import baseGraphitiProvider
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient



@graphitiProviderRegistry.register('google')
class geminiProvider(baseGraphitiProvider):

    def getLlmClient(self, api_key: str, model: str = None, small_model=None):
        config = LLMConfig(api_key=api_key, model=model, small_model=small_model)
        client=GeminiClient(config=config)
        return client
    
    def getEmbedder(self, api_key: str, embedding_model: str = None):
        config = GeminiEmbedderConfig(
            api_key=api_key,
            embedding_model=embedding_model
        )
        return GeminiEmbedder(config=config)
    
    def getReranker(self, api_key: str, reranker_model: str = None, small_model=None):
        config = LLMConfig(api_key=api_key, model=reranker_model, small_model=small_model)
        return GeminiRerankerClient(config=config)