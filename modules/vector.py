from typing import List
import uuid
from datetime import datetime

import torch
import torch.nn as nn

from modules.utils import ascii_to_string, datetime_to_unix_timestamp, int_list_to_uuid, unix_timestamp_to_datetime, uuid_to_int_list

class VectorHouse:
    def __init__(self, input_data: str, max_vector_length: int = None) -> None:
        self.input_data = input_data
        self.max_vector_length = max_vector_length
        self.vectors = self.process_data()

    def process_data(self) -> List[List]:
        nsfw_score = self.get_nsfw_score(self.input_data)
        tokens = self.get_tokens_from_data(self.input_data)
        data_type = 0
        return [self.create_vector(token, idx + 1, data_type, nsfw_score) for idx, data_type, token in enumerate(tokens)]

    def create_vector(self, token: str, position: int, data_type: int = 0, nsfw_score: float = 0.0) -> List:   
        vector = [
            uuid_to_int_list(uuid.uuid4()),                  # Vector Id as list of integers
            uuid_to_int_list(uuid.uuid4()),                  # Link Id as list of integers (for extending the data to end of file)
            data_type,
            uuid_to_int_list(uuid.uuid4()),                  # Sentence Id as list of integers            
            position,                                       # Sentence Position

            datetime_to_unix_timestamp(datetime.now()),     # Date Value as Unix timestamp
            nsfw_score,                                      # NSFW Score
            self.string_to_ascii(token)                     # Token Value as ASCII values            
        ]
        vector.extend(self.append_new_vectors())

        # Pad or truncate the vector based on self.max_vector_length
        if self.max_vector_length is not None:
            vector_length = len(vector)
            if vector_length < self.max_vector_length:
                vector.extend([0] * (self.max_vector_length - vector_length))  # Padding
            elif vector_length > self.max_vector_length:
                vector = vector[:self.max_vector_length]  # Truncation

        return vector
    
    
    def to_tensor(self):
        # Assuming self.vectors is a list of lists and all data is numerical
        tensor = torch.tensor(self.vectors, dtype=torch.float32)
        return tensor
    
    @staticmethod
    def from_tensor(tensor, token_length):
        # Convert tensor back to list of vectors
        # This will depend on the structure of your original vectors
        vectors = tensor.tolist()
        # Basic reconstruction of vector elements
        reconstructed_vectors = []
        for vec in vectors:
            vector_data = {
                "id": int_list_to_uuid(vec[0:16]),  # Adjust index as per your UUID size
                "link_id": int_list_to_uuid(vec[16:32]),
                "data_type": vec[32],
                "token": ascii_to_string(vec[33:33+token_length]),  # Adjust token_length accordingly
                "sentence_id": int_list_to_uuid(vec[33+token_length:33+token_length+16]),
                "sentence_position": vec[49+token_length],
                "date": unix_timestamp_to_datetime(vec[50+token_length]),
                "nsfw_score": vec[51+token_length]
            }

            # Additional data can be added by overriding this method in subclasses
            reconstructed_vectors.append(vector_data)
        return reconstructed_vectors

    def loss_function(self, output, target, alpha = 0.5, beta = 0.5):
        # Calculate Cross-Entropy Loss
        cross_entropy_loss = nn.CrossEntropyLoss()(output, target)

        # Calculate Mean Squared Error
        mse_loss = nn.MSELoss()(output, target)

        # A* path algorithm or other decision-making logic
        # For simplicity, let's assume a weighted sum here
        # Adjust weights according to your preference
        # alpha = 0.5  # weight for cross_entropy_loss
        # beta = 0.5   # weight for mse_loss
        combined_loss = alpha * cross_entropy_loss + beta * mse_loss

        return combined_loss

    def get_tokens_from_data(self, input_data):
        data_type = 0
        if isinstance(input_data, str):
            # Tokenize by words for string data
            tokens = input_data.split(" ")
        elif isinstance(input_data, bytes):
            # Tokenize by byte values for byte data
            tokens = [str(byte) for byte in input_data]
            data_type = 1
        else:
            # Handle other data types or raise an error
            raise ValueError("Unsupported data type for tokenization")
        return data_type, tokens
    
    def get_nsfw_score(self, token: str) -> int:
        return 0

    def append_new_vectors(self) -> List:
        # Override in child class to add new vectors to each vector
        return []

    def __str__(self) -> str:
        return str(self.vectors)
    

# Example subclass extending Vector class
class ExtendedVectorHouse(VectorHouse):
    @staticmethod
    def from_tensor(tensor):
        base_vectors = super().from_tensor(tensor)
        # Process additional data specific to ExtendedVectorHouse
        for vec in base_vectors:
            # Add or modify vector data as needed
            # Example: vec["additional_data"] = some_processing_function(...)
            pass
        return base_vectors

class ByteVector(VectorHouse):
    def get_tokens_from_data(self, input_data):
        if isinstance(input_data, bytes):
            return [str(byte) for byte in input_data]
        raise ValueError("Input data must be byte data")


if __name__ == "__main__":
    input_string = "hello world"
    extended_vector_instance = ExtendedVectorHouse(input_string)
    
    # Convert to tensor
    tensor = extended_vector_instance.to_tensor()
    print("Tensor representation:", tensor)

    # Convert back from tensor using the subclass method
    vectors_from_extended_tensor = ExtendedVectorHouse.from_tensor(tensor)
    print("Extended Vector representation from tensor:", vectors_from_extended_tensor)
