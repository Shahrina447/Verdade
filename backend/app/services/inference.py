"""
Inference service — tokenize → forward pass → structured result.

Label mapping (matches training):
  label 0 → REAL
  label 1 → FAKE

Post-processing pipeline:
  1. Confidence threshold  — below UNCERTAIN_THRESHOLD → prediction = "UNCERTAIN"
  2. Heuristic overrides   — known conspiracy/misinformation patterns → FAKE
"""
from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from app.config import DEVICE, MAX_LEN

# Populated by the lifespan handler in main.py
tokenizer: AutoTokenizer | None = None
model: Any = None

# ── Thresholds ─────────────────────────────────────────────────────────────────
# If max(conf_real, conf_fake) is below this the model is not confident enough
# to commit to REAL or FAKE — we return UNCERTAIN instead.
UNCERTAIN_THRESHOLD = 0.65

# ── Heuristic patterns ─────────────────────────────────────────────────────────
# Urdu phrases strongly associated with conspiracy / miracle-cure misinformation.
# Each entry is a plain substring; matched case-insensitively against the text.
_HEURISTIC_PATTERNS: list[str] = [
    # suppression / cover-up
    "حکومت اسے چھپا",
    "حکومت چھپا رہی",
    "حکومت نے چھپایا",
    "میڈیا چھپا رہا",
    "راز چھپایا جا رہا",
    "آپ کو نہیں بتایا جاتا",
    # miracle cures / pseudoscience
    "تمام بیماریاں ٹھیک",
    "تمام بیماریوں کا علاج",
    "صرف ایک دن میں ٹھیک",
    "صرف ایک ہفتے میں",
    "ایک گھنٹے میں علاج",
    "100 فیصد علاج",
    "کینسر ختم",
    "شوگر ختم",
    "ڈاکٹر نہیں بتاتے",
    "ڈاکٹر چھپاتے ہیں",
    # unverifiable sensational claims
    "حیرت انگیز دریافت",
    "معجزاتی علاج",
    "جادوئی دوا",
    "فوری شفا",
    # conspiracy framing
    "سازش کا انکشاف",
    "بڑا انکشاف",
    "چونکا دینے والا راز",
]

# Pre-compile as a single regex for efficiency
_HEURISTIC_RE = re.compile(
    "|".join(re.escape(p) for p in _HEURISTIC_PATTERNS),
    flags=re.IGNORECASE,
)


# ── Main inference ──────────────────────────────────────────────────────────────

def run_inference(text: str) -> dict:
    """
    Returns a dict with:
      prediction, confidence, confidence_real, confidence_fake, verdict_text
    """
    if tokenizer is None or model is None:
        raise RuntimeError("Model has not been loaded yet.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs     = F.softmax(logits, dim=-1).squeeze()  # shape [2]
    conf_real = float(probs[0])   # label 0 = REAL
    conf_fake = float(probs[1])   # label 1 = FAKE
    confidence = max(conf_real, conf_fake)

    # Step 1 — raw model prediction
    if confidence < UNCERTAIN_THRESHOLD:
        prediction = "UNCERTAIN"
    else:
        prediction = "REAL" if conf_real >= conf_fake else "FAKE"

    result = {
        "prediction":      prediction,
        "confidence":      round(confidence, 4),
        "confidence_real": round(conf_real, 4),
        "confidence_fake": round(conf_fake, 4),
        "verdict_text":    "",          # filled below
        "heuristic_match": False,       # internal flag, stripped before response
    }

    # Step 2 — heuristic override
    result = _apply_heuristics(text, result)

    result["verdict_text"] = _build_verdict(
        result["prediction"],
        result["confidence"],
        result["heuristic_match"],
    )

    # Remove internal flag before returning to router
    result.pop("heuristic_match", None)
    return result


def _apply_heuristics(text: str, result: dict) -> dict:
    """Override prediction to FAKE when known misinformation patterns are found."""
    match = _HEURISTIC_RE.search(text)
    if match:
        # Flip the probabilities to be consistent with the FAKE verdict
        # so the UI doesn't show "Real: 95%, Fake: 80%"
        raw_fake = result["confidence_fake"]
        boosted_fake = max(raw_fake, 0.80)
        boosted_real = 1.0 - boosted_fake

        result["prediction"]       = "FAKE"
        result["heuristic_match"]  = True
        result["confidence"]       = round(boosted_fake, 4)
        result["confidence_fake"]  = round(boosted_fake, 4)
        result["confidence_real"]  = round(boosted_real, 4)
    return result


def _build_verdict(prediction: str, confidence: float, heuristic: bool = False) -> str:
    pct = round(confidence * 100)

    if prediction == "UNCERTAIN":
        return (
            f"The model is not confident enough to classify this article ({pct}% max confidence). "
            "Language patterns are ambiguous — manually verify with a trusted source before sharing."
        )

    if prediction == "FAKE":
        if heuristic:
            return (
                "This article contains known misinformation patterns — "
                "conspiracy framing, miracle-cure claims, or suppression language. "
                "Treat with high skepticism and verify independently."
            )
        if confidence >= 0.85:
            return f"High likelihood of misinformation detected ({pct}% confidence). Do not share without verification."
        if confidence >= 0.65:
            return f"This article shows several fake-news indicators ({pct}% confidence). Verify before sharing."
        return f"Content leans toward fake news but signals are mixed ({pct}%). Treat with caution."

    # REAL
    if confidence >= 0.85:
        return f"This article shows strong indicators of authentic, credible reporting ({pct}% confidence)."
    if confidence >= 0.65:
        return f"This article appears to be real news with moderate confidence ({pct}%)."
    return f"The content leans toward real news but signals are mixed ({pct}%). Cross-check before sharing."
