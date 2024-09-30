from transformers import AutoModel


class JinaEmbeddingModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v3", trust_remote_code=True)

    def encode(self, texts: list[str], task: str = "text-matching"):
        return self.model.encode(texts, task=task)


jina_embedding_model = JinaEmbeddingModel()


if __name__ == "__main__":
    embedding_model = JinaEmbeddingModel()
    print(embedding_model.encode(["Hello, world!"]))
