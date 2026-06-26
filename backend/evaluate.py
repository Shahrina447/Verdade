"""
Evaluate the trained XLM-RoBERTa model on test.csv

Usage:
    uv run python evaluate.py
    uv run python evaluate.py --csv path/to/test.csv --batch-size 16

Expected CSV columns:
    "News Items"  — Urdu text
    "label"       — integer label (0 = REAL, 1 = FAKE)
    OR "Label"    — string label (REAL / FAKE / TRUE / FALSE)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from transformers import AutoTokenizer

# ── Add project root so `app` package is importable ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from app.config import DEVICE, MAX_LEN
from app.models.loader import load_model

# ── Label normalisation ────────────────────────────────────────────────────────
# Training: 0 = REAL, 1 = FAKE
_LABEL_MAP: dict[str, int] = {
    "0": 0, "real": 0, "true": 0,
    "1": 1, "fake": 1, "false": 1,
}


def normalise_labels(series: pd.Series) -> list[int]:
    """Convert raw label column (int or string) to list of 0/1 ints."""
    out = []
    for v in series:
        key = str(v).strip().lower()
        if key not in _LABEL_MAP:
            raise ValueError(f"Unknown label value: {v!r}. Expected 0/1/REAL/FAKE/TRUE/FALSE.")
        out.append(_LABEL_MAP[key])
    return out


# ── Batch inference ────────────────────────────────────────────────────────────

def predict_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    batch_size: int = 32,
) -> list[int]:
    """Run inference on a list of texts and return predicted label indices."""
    all_preds: list[int] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits          # (B, 2)

        probs = F.softmax(logits, dim=-1)            # (B, 2)
        preds = torch.argmax(probs, dim=-1).tolist() # label 0=REAL, 1=FAKE
        all_preds.extend(preds)

        done = min(i + batch_size, len(texts))
        print(f"  [{done}/{len(texts)}]", end="\r", flush=True)

    print()
    return all_preds


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SachAI model on test.csv")
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent / "test.csv"),
        help="Path to test CSV file (default: backend/test.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32)",
    )
    parser.add_argument(
        "--text-col",
        default=None,
        help="Name of the text column (auto-detected if omitted)",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Name of the label column (auto-detected if omitted)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    # ── Load CSV ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SachAI — Model Evaluation")
    print(f"{'='*60}")
    print(f"\n→ Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}   Columns: {list(df.columns)}")

    # Auto-detect text column
    text_col = args.text_col
    if text_col is None:
        candidates = ["News Items", "news_items", "text", "Text", "content", "Content"]
        for c in candidates:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            print(f"ERROR: Could not auto-detect text column. Available: {list(df.columns)}")
            sys.exit(1)
    print(f"  Text column  : '{text_col}'")

    # Auto-detect label column  (prefer integer 'label' over string 'Label')
    label_col = args.label_col
    if label_col is None:
        for c in ["label", "Label", "labels"]:
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            print(f"ERROR: Could not auto-detect label column. Available: {list(df.columns)}")
            sys.exit(1)
    print(f"  Label column : '{label_col}'")

    # Drop rows with missing text or label
    before = len(df)
    df = df.dropna(subset=[text_col, label_col])
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with missing values.")

    texts  = df[text_col].astype(str).tolist()
    y_true = normalise_labels(df[label_col])

    label_dist = pd.Series(y_true).value_counts().to_dict()
    print(f"  Label dist   : REAL={label_dist.get(0,0):,}  FAKE={label_dist.get(1,0):,}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\n→ Loading model …")
    t0 = time.time()
    tokenizer, model = load_model()
    print(f"  Done in {time.time()-t0:.1f}s  |  device={DEVICE}")

    # ── Run inference ───────────────────────────────────────────────────────
    print(f"\n→ Running inference (batch_size={args.batch_size}) …")
    t1 = time.time()
    y_pred = predict_batch(texts, tokenizer, model, batch_size=args.batch_size)
    elapsed = time.time() - t1
    print(f"  Done in {elapsed:.1f}s  ({len(texts)/elapsed:.0f} samples/sec)")

    # ── Metrics ─────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["REAL (0)", "FAKE (1)"],
        digits=4,
    )

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    print(f"\n  Accuracy : {acc*100:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred REAL   Pred FAKE")
    print(f"  True REAL   {cm[0][0]:>9,}   {cm[0][1]:>9,}")
    print(f"  True FAKE   {cm[1][0]:>9,}   {cm[1][1]:>9,}")
    print(f"\n  Classification Report:\n")
    print(report)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
