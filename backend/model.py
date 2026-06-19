import os
import torch
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─── CPU ONLY — never cuda ────────────────────────────────────────────────────
DEVICE = torch.device("cpu")

MODEL_PATH = os.getenv("MODEL_PATH", "./saved_model")

# ─── Decision threshold ───────────────────────────────────────────────────────
# REAL is predicted only when real_conf >= REAL_THRESHOLD.
# Default 0.5 = standard. Lower it (e.g. 0.4) to catch more fake news
# at the cost of some false positives on real news.
# Set via env: REAL_THRESHOLD=0.4
REAL_THRESHOLD = float(os.getenv("REAL_THRESHOLD", "0.55"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "1.5"))

# Global model / tokenizer — loaded once at startup
_model = None
_tokenizer = None


def load_model():
    """Load the fine-tuned xlm-RoBERTa model onto CPU."""
    global _model, _tokenizer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # float32 ONLY — never float16 on CPU
    )
    _model.to(DEVICE)
    _model.eval()
    print(f"[SachAI] Model loaded from '{MODEL_PATH}' on {DEVICE}")
    print(f"[SachAI] REAL threshold = {REAL_THRESHOLD} | Temperature = {TEMPERATURE}")


def predict(text: str) -> dict:
    """Run xlm-RoBERTa inference on CPU and return a prediction dict."""
    from preprocess import clean_urdu_text

    cleaned = clean_urdu_text(text)
    inputs = _tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits / TEMPERATURE, dim=-1)[0]
    # LABEL_0 = FAKE, LABEL_1 = REAL (as encoded during training)
    fake_conf = round(probs[0].item(), 3)
    real_conf = round(probs[1].item(), 3)

    # Apply configurable threshold — model must be >= REAL_THRESHOLD confident
    # in REAL before we label it as such, otherwise it's FAKE.
    prediction = "REAL" if real_conf >= REAL_THRESHOLD else "FAKE"

    if prediction == "REAL" and real_conf >= 0.80:
        verdict = "✓ Authentic linguistic patterns detected. Content appears credible."
    elif prediction == "REAL" and real_conf >= 0.65:
        verdict = "⚠ Likely credible but verify with trusted sources before sharing."
    elif prediction == "REAL":
        verdict = "⚠ Mixed signals detected. Treat with caution and verify independently."
    elif fake_conf >= 0.80:
        verdict = "✗ Strong misinformation indicators: sensational tone, unverifiable claims, suspicious phrasing."
    else:
        verdict = "✗ Misinformation indicators detected: sensational tone and unverifiable claims."

    return {
        "prediction": prediction,
        "confidence": max(real_conf, fake_conf),
        "confidence_real": real_conf,
        "confidence_fake": fake_conf,
        "verdict_text": verdict,
        "prediction_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
    }
