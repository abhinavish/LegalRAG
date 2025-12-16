from .base import baseGraphitiProvider
from typing import Type

class graphitiProviderRegistry():

    providers={}

    @classmethod
    def register(cls, name: str):
        def decorator(provider: Type[baseGraphitiProvider]):
            cls.providers[name.lower()]=provider
            return provider
        return decorator
    
    @classmethod
    def get(cls,name: str):
        provider_name=name.lower()
        return cls.providers.get(provider_name, f"Provider '{provider_name}' does not exist")


