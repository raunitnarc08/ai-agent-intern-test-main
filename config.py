import os
from datetime import datetime, timezone
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge-base"
DATA_DIR = BASE_DIR / "data"
ORDERS_PATH = DATA_DIR / "orders.json"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TURN_LOGS_PATH = LOGS_DIR / "agent_turns.jsonl"
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Load .env manually if python-dotenv is not installed or just simple parse
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key not in os.environ:
                    os.environ[key] = val

# LLM & Embedding Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Retrieval parameters
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.38"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))

# Operational snapshot timestamp for Aster & Row mock data
SNAPSHOT_TIME_STR = "2026-08-15T12:00:00Z"
SNAPSHOT_DATETIME = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
