import json
import os
from pathlib import Path


def load_api_keys(api_key_path=None):
    """Load provider keys from a JSON file and environment variables."""
    api_keys = {}

    if api_key_path and Path(api_key_path).exists():
        with open(api_key_path, "r") as f:
            api_keys.update(json.load(f))

    if os.environ.get("OPENAI_API_KEY"):
        api_keys.setdefault("openai", os.environ["OPENAI_API_KEY"])
    if os.environ.get("GEMINI_API_KEY"):
        api_keys.setdefault("gemini", os.environ["GEMINI_API_KEY"])
    if os.environ.get("GOOGLE_API_KEY"):
        api_keys.setdefault("gemini", os.environ["GOOGLE_API_KEY"])

    return api_keys
