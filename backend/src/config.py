# src/config.py

import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
print("Loading .env from:", env_path)  # 🔍 Debug print
load_dotenv(dotenv_path=env_path)

print("DEBUG POSTGRES_USER =", os.getenv("POSTGRES_USER"))  # Should not be None

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# ✅ Construct the async PostgreSQL URL
DATABASE_URL = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

print("🔍 Loaded config:")
print("POSTGRES_USER =", POSTGRES_USER)
print("POSTGRES_PASSWORD =", POSTGRES_PASSWORD)
print("POSTGRES_DB =", POSTGRES_DB)
