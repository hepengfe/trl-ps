import itertools
import os
import openai


class APIKeyRotator:
    def __init__(self, keys, name="API"):
        self.name = name
        self._keys = keys
        self.keys = itertools.cycle(keys)

    def next(self):
        if not self._keys:
            raise RuntimeError(
                f"No {self.name} keys configured. Set the relevant environment variables."
            )
        return next(self.keys)


def _keys_from_env(*names):
    seen = []
    for name in names:
        value = os.getenv(name)
        if value and value not in seen:
            seen.append(value)
    return seen


OPENAI_KEYS = _keys_from_env("OPENAI_KEY_1", "OPENAI_KEY_2", "OPENAI_API_KEY")
GOOGLE_KEYS = _keys_from_env("GOOGLE_KEY_1", "GOOGLE_KEY_2", "GOOGLE_API_KEY")

openai_key_rotator = APIKeyRotator(OPENAI_KEYS, name="OpenAI")
google_key_rotator = APIKeyRotator(GOOGLE_KEYS, name="Google")


def get_openai_client():
    return openai.OpenAI(api_key=openai_key_rotator.next())
