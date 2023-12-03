# Assume the previous imports and VectorHouse definition are here
import torch
from modules.llms.llm import LLM
from modules.utils import ascii_to_string, int_list_to_uuid, unix_timestamp_to_datetime
from modules.vector import ExtendedVectorHouse, VectorHouse

class EmbeddingVectorHouse(VectorHouse):
    def __init__(self, input_data: str, embeddings_model, max_vector_length: int = None):
        super().__init__(input_data, max_vector_length)
        self.embeddings_model = embeddings_model
        self.embedded_data = self.add_embeddings_to_vectors()

    def __str__(self):
        return str(self.embedded_data)

    @staticmethod
    def from_tensor(tensor, token_length):
        # Convert tensor back to list of vectors and embeddings
        vectors = tensor.tolist()

        # Basic reconstruction of vector elements and embeddings
        reconstructed_vectors = []
        for vec in vectors:
            vector_data = {
                "id": int_list_to_uuid(vec[0:16]),
                "link_id": int_list_to_uuid(vec[16:32]),
                "data_type": vec[32],
                "token": ascii_to_string(vec[33:33+token_length]),
                "sentence_id": int_list_to_uuid(vec[33+token_length:33+token_length+16]),
                "sentence_position": vec[49+token_length],
                "date": unix_timestamp_to_datetime(vec[50+token_length]),
                "nsfw_score": vec[51+token_length]
            }

            # Extract embeddings
            embeddings = vec[52+token_length:52+token_length+embeddings_length]  # Adjust embeddings_length as per the new model

            # Combine data and embeddings
            reconstructed_vector = {**vector_data, "embeddings": embeddings}
            reconstructed_vectors.append(reconstructed_vector)

        return reconstructed_vectors
    
    @staticmethod
    def mean_pooling(embeddings):
        # Assuming embeddings is a tensor of shape (num_sentences, num_tokens, embedding_dim)
        # Averaging across the token dimension
        pooled = torch.mean(embeddings, dim=1)
        return pooled
    
    def add_embeddings_to_vectors(self):
        # Existing logic to get embeddings
        embeddings = self.embeddings_model.get_embeddings(self.input_data)

        # Apply mean pooling to the embeddings
        pooled_embeddings = self.mean_pooling(embeddings)

        # Integrate pooled embeddings with existing vectors
        extended_vectors = []
        for vec, emb in zip(self.vectors, pooled_embeddings):
            extended_vector = vec + emb.tolist()
            extended_vectors.append(extended_vector)

        return extended_vectors

# Example usage
if __name__ == "__main__":
    embeddings_model = LLM(model_name='jinaai/jina-embeddings-v2-base-en', use_causal_pretrained=False, trust_remote_code=True) # Initialize your embeddings model here
    input_string = "hello world"
    extended_vector_instance = ExtendedVectorHouse(input_string, embeddings_model)
    
    print("Extended Vector representation with embeddings:", extended_vector_instance)

    tensor = extended_vector_instance.to_tensor()
    embeddings_length = 8192  # 1024 Update this value based on the output size of jina-embeddings-v2-base-en
    reconstructed_data = ExtendedVectorHouse.from_tensor(tensor, embeddings_length)

    print("Reconstructed Vector representation from tensor with embeddings:", reconstructed_data)
