# app/config.py
import os
from dotenv import load_dotenv

load_dotenv(override=True)

CHROMA_SEARCH_URL = "http://0.0.0.0:8324/search"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHROMA_ACCESS_TOKEN = os.getenv("CHROMA_ACCESS_TOKEN")
AGENT_ACCESS_TOKEN = os.getenv("AGENT_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not CHROMA_ACCESS_TOKEN or not AGENT_ACCESS_TOKEN:
    raise RuntimeError("Missing required access tokens")
