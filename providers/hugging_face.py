from sentence_transformers import SentenceTransformer
from pydantic import Field
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig


class huggingFaceConfig(EmbedderConfig):
    embedding_model: str = Field(default=None)
    device: str = 'cpu'
    embedding_dim:int =None

class huggingFaceEmbedding(EmbedderClient):
    def __init__(self, config: huggingFaceConfig, batch_size:int=None):
        assert config is not None, "Please provide config for huggingface embedder"

        self.config = config
        self.model = SentenceTransformer(
                config.embedding_model,
                device=config.device
            )
        
        self.batch_size = batch_size if batch_size is not None else 100


    async def create(self, input_data):
        embd = self.model.encode(
            input_data,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        if embd.ndim == 2:
            if embd.shape[0] == 1:
                embd = embd[0]
            else:
                
                embd = embd[0]
        elif embd.ndim != 1:
            raise ValueError(f"Unexpected embedding shape: {embd.shape}")

        embs = embd.tolist()

        if self.config.embedding_dim and len(embs) != self.config.embedding_dim:
            print(
                f'Embedding dimension mismatch: expected {self.config.embedding_dim} got {len(embs)}')

        return embs

    async def create_batch(self, input_data_list):
        if not input_data_list:
            return []

        batch_size = self.batch_size
        embeds = []

        
        for i in range(0, len(input_data_list), batch_size):
            batch = input_data_list[i : i + batch_size]

            try:
                embedder = self.model.encode(
                    batch,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=len(batch)
                )

        
                batch_emb = embedder.tolist()
                embeds.extend(batch_emb)

            except Exception as e:

                for item in batch:
                    try:
                        embedder = self.model.encode(
                            item,
                            normalize_embeddings=True,
                            convert_to_numpy=True
                        )
                        embeds.append(embedder.tolist())

                    except Exception as individual_error:
                        print(f'Failed to embed individual item: {individual_error}')
                        raise individual_error

        return embeds


