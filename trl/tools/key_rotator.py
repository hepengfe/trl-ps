import itertools
import os
import openai

class APIKeyRotator:
    def __init__(self, keys):
        self.keys = itertools.cycle(keys)

    def next(self):
        return next(self.keys)

# OpenAI and Google key pools
OPENAI_KEYS = [
    os.getenv("OPENAI_KEY_1", "YOUR_API_KEY_1"),
    os.getenv("OPENAI_KEY_2", "YOUR_API_KEY_2"),
    # Add more keys as needed
]

GOOGLE_KEYS = [
    os.getenv("GOOGLE_KEY_1", "YOUR_GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_KEY_2", "YOUR_GOOGLE_API_KEY_2"),
    # Add more keys as needed
]

openai_key_rotator = APIKeyRotator(OPENAI_KEYS)
google_key_rotator = APIKeyRotator(GOOGLE_KEYS)

def get_openai_client():
    key = openai_key_rotator.next()
    return openai.OpenAI(api_key=key)
