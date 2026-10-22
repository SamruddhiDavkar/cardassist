"""Central configuration. Values can be overridden with environment variables."""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A fixed "today" keeps the fake data, the policy checks and the evaluation reproducible.
TODAY = date.fromisoformat(os.getenv("CARDASSIST_TODAY", "2026-10-21"))

DB_PATH = Path(os.getenv("CARDASSIST_DB", ROOT / "data" / "cardassist.db"))

OLLAMA_MODEL = os.getenv("CARDASSIST_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Dispute rules
DISPUTE_WINDOW_DAYS = 60          # charge must be at most 60 days old
REVIEW_THRESHOLD = 500.00         # amounts strictly over this need a human
FRAUD_KEYWORDS = (
    "fraud", "stolen", "hacked", "unauthorized", "unauthorised",
    "identity theft", "someone used my card", "compromised",
)

MAX_AGENT_STEPS = 6
