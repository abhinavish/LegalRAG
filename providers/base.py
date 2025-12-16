from typing import Dict, Type, Optional
from abc import ABC, abstractmethod

class baseGraphitiProvider(ABC):
    
    
    @abstractmethod
    def getLlmClient(self, api_key: str, model: str = None):
        
        pass
    
    @abstractmethod
    def getEmbedder(self, api_key: str, embedding_model: str = None):
        
        pass
    
    @abstractmethod
    def getReranker(self, api_key: str, reranker_model: str = None):
        pass