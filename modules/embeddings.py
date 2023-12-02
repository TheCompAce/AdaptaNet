from transformers import AutoModel
import torch

class EmbeddingsModel:
    def __init__(self, model_name='jinaai/jina-embeddings-v2-base-en'):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def get_embeddings(self, text, max_length=8192):
        # Encode the text and apply mean pooling
        inputs = self.model.encode(text, max_length=max_length, return_tensors='pt')
        return inputs.mean(dim=1)

if __name__ == "__main__":
    text = ["Example text to get embeddings"]
    embeddings_model = EmbeddingsModel()
    embeddings = embeddings_model.get_embeddings(text)
    print("Embeddings:", embeddings)
