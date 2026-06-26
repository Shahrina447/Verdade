"""
Central configuration — all constants and paths in one place.
Values can be overridden via backend/.env
"""
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent   # backend/
MODEL_PATH   = BASE_DIR / "xlmroberta_urdu_best.pt"
HISTORY_FILE = BASE_DIR / "data" / "history.json"

# ── Server ─────────────────────────────────────────────────────────────────────
HOST            = os.getenv("HOST", "localhost")
PORT            = int(os.getenv("PORT", "8000"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# ── Model ──────────────────────────────────────────────────────────────────────
HF_MODEL_NAME = "xlm-roberta-base"
MAX_LEN       = 256
NUM_LABELS    = 2
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── API ────────────────────────────────────────────────────────────────────────
TEXT_MIN_LEN = 10
TEXT_MAX_LEN = 5_000
BATCH_MAX    = 50
PREVIEW_LEN  = 120
