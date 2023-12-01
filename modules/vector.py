from typing import List
import uuid
from datetime import datetime

import torch

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
    def from_tensor(tensor):
        # Convert tensor back to list of vectors
        # This will depend on the structure of your original vectors
        vectors = tensor.tolist()
        # Basic reconstruction of vector elements
        reconstructed_vectors = []
        for vec in vectors:
            data = {
                "id": int_list_to_uuid(vec[0]),
                "link_id": int_list_to_uuid(vec[1]),
                "data_type": int_list_to_uuid(vec[2]),
                "token": ascii_to_string(vec[3]),
                "sentence_id": int_list_to_uuid(vec[4]),
                "sentence_position": vec[5],
                "date": unix_timestamp_to_datetime(vec[6]),
                "nsfw_score": vec[7]
            }
            # Additional data can be added by overriding this method in subclasses
            reconstructed_vectors.append(data)
        return reconstructed_vectors


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
    
def uuid_to_int_list(uuid_value):
    return [int(byte) for byte in uuid_value.bytes]

def int_list_to_uuid(int_list):
    try:
        uuid_bytes = bytes(int_list)
        return uuid.UUID(bytes=uuid_bytes)
    except ValueError:
        raise ValueError("Invalid integer list for UUID conversion")


def datetime_to_unix_timestamp(date_value):
    return int(date_value.timestamp())

def unix_timestamp_to_datetime(unix_timestamp):
    return datetime.fromtimestamp(unix_timestamp)

def ascii_to_string(ascii_list):
    return ''.join(chr(ascii_value) for ascii_value in ascii_list)

def string_to_ascii(string_token: str) -> List[int]:
    return [ord(char) for char in string_token]

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
