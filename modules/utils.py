from typing import List
from datetime import datetime
import os
import random
import uuid

import json

def read_json(file_path):
    """Reads a JSON file and returns its content."""
    with open(file_path, 'r') as file:
        return json.load(file)

def write_json(file_path, data):
    """Writes data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_prompt_from_settings(file_path, key):
    """Retrieves a prompt from settings.json based on the given key."""
    settings = read_json(file_path)
    prompt_setting = settings.get('embedding_prompts', {}).get(key)

    if os.path.isfile(prompt_setting):
        with open(prompt_setting, 'r') as file:
            return file.read()
    else:
        return prompt_setting


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

def a_star_pathfinding(cross_entropy_loss, mse_loss, step=0.25, steps=10):
    best_cost = float('inf')
    best_g, best_h = 0, 0

    for i in range(steps):
        g = random.uniform(0, 1)
        h = random.uniform(0, 1)

        for j in range(steps):
            cost = g * cross_entropy_loss + h * mse_loss
            if cost < best_cost:
                best_cost = cost
                best_g, best_h = g, h
            
            g += step
            h += step
            if g > 1: g = 1
            if h > 1: h = 1

    return best_g, best_h
