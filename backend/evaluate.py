"""
Evaluate xlmroberta_urdu_best.pt on test_400.csv
  — 400 stratified samples (200 REAL + 200 FAKE) drawn from test.csv

Usage:
    uv run python evaluate.py
"""
from __future__ import annotations

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
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).parent))

from app.config import DEVICE, MAX_LEN
from app.models.loader import load_model

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH   = Path(__file__).parent / "test_samples.csv"
BATCH_SIZE = 16

# test_samples.csv label convention: 0 = TRUE news, 1 = FAKE news
# We remap to match model space where probe auto-detects FAKE index
CSV_FAKE_LABEL = 1   # what the CSV uses for FAKE
CSV_TRUE_LABEL = 0   # what the CSV uses for TRUE


def probe_label_order(df: pd.DataFrame, tokenizer, model) -> int:
    """
    Runs one known FAKE and one known TRUE sample through the model
    and returns which output index corresponds to FAKE.
    """
    fake_text = df[df["label"] == CSV_FAKE_LABEL]["text"].iloc[0]
    real_text = df[df["label"] == CSV_TRUE_LABEL]["text"].iloc[0]

    results = {}
    for name, text in [("fake", fake_text), ("real", real_text)]:
        inp = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=MAX_LEN, padding=True
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits
        probs = F.softmax(logits, dim=-1).squeeze()
        results[name] = probs.tolist()

    fake_idx = int(results["fake"][1] > results["fake"][0])
    print(f"  FAKE sample probs : {[round(p, 4) for p in results['fake']]}")
    print(f"  REAL sample probs : {[round(p, 4) for p in results['real']]}")
    print(f"  → FAKE index = {fake_idx}  |  REAL index = {1 - fake_idx}")
    return fake_idx


def predict_batch(texts: list[str], tokenizer, model, fake_idx: int) -> tuple[list[int], list[float]]:
    """
    Returns:
        y_pred — list[int]   : 1 = FAKE, 0 = REAL  (matches CSV label)
        y_prob — list[float] : FAKE probability per sample (for AUC)
    """
    all_preds: list[int]   = []
    all_probs: list[float] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch  = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits      # (B, 2)

        probs    = F.softmax(logits, dim=-1)     # (B, 2)
        raw_pred = torch.argmax(probs, dim=-1)   # model-space index

        if fake_idx == 1:
            # model index 1 = FAKE → CSV_FAKE_LABEL, model index 0 = TRUE → CSV_TRUE_LABEL
            preds     = [CSV_FAKE_LABEL if p == 1 else CSV_TRUE_LABEL for p in raw_pred.tolist()]
            fake_prob = probs[:, 1].tolist()
        else:
            # model index 0 = FAKE → CSV_FAKE_LABEL, model index 1 = TRUE → CSV_TRUE_LABEL
            preds     = [CSV_FAKE_LABEL if p == 0 else CSV_TRUE_LABEL for p in raw_pred.tolist()]
            fake_prob = probs[:, 0].tolist()

        all_preds.extend(preds)
        all_probs.extend(fake_prob)

        done = min(i + BATCH_SIZE, len(texts))
        print(f"  Progress : [{done:>3}/{len(texts)}]", end="\r", flush=True)

    print()
    return all_preds, all_probs


def main() -> None:
    print(f"\n{'='*64}")
    print("   SachAI  —  Model Evaluation on test_samples.csv")
    print(f"{'='*64}")

    # ── Load the pre-generated 400-sample file ──────────────────────────────
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run the sampling step first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    # Stratified sample — 200 per class
    fake_df = df[df["label"] == CSV_FAKE_LABEL].sample(n=200, random_state=42)
    true_df = df[df["label"] == CSV_TRUE_LABEL].sample(n=200, random_state=42)
    sample  = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    texts  = sample["text"].astype(str).tolist()
    # Keep labels in CSV space (0=TRUE, 1=FAKE) — predictions will be mapped to match
    y_true = sample["label"].tolist()

    real_count = y_true.count(CSV_TRUE_LABEL)
    fake_count = y_true.count(CSV_FAKE_LABEL)

    print(f"\n  File     : {CSV_PATH.name}  ({len(df):,} total rows)")
    print(f"  Sampled  : 400 rows — 200 TRUE + 200 FAKE (stratified)")
    print(f"  Device   : {DEVICE}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\n→ Loading model …")
    t0 = time.time()
    tokenizer, model = load_model()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # ── Auto-detect label order ─────────────────────────────────────────────
    print(f"\n→ Probing label order …")
    fake_idx = probe_label_order(df, tokenizer, model)

    # ── Run inference ───────────────────────────────────────────────────────
    print(f"\n→ Running inference (batch_size={BATCH_SIZE}) …")
    t1 = time.time()
    y_pred, y_prob = predict_batch(texts, tokenizer, model, fake_idx)
    infer_time = time.time() - t1
    print(f"  Done in {infer_time:.1f}s  ({len(texts) / infer_time:.1f} samples/sec)")

    # ── Compute metrics ─────────────────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    prec_fake = precision_score(y_true, y_pred, pos_label=CSV_FAKE_LABEL)
    rec_fake  = recall_score(y_true, y_pred, pos_label=CSV_FAKE_LABEL)
    f1_fake   = f1_score(y_true, y_pred, pos_label=CSV_FAKE_LABEL)
    prec_real = precision_score(y_true, y_pred, pos_label=CSV_TRUE_LABEL)
    rec_real  = recall_score(y_true, y_pred, pos_label=CSV_TRUE_LABEL)
    f1_real   = f1_score(y_true, y_pred, pos_label=CSV_TRUE_LABEL)
    f1_macro  = f1_score(y_true, y_pred, average="macro")
    prec_mac  = precision_score(y_true, y_pred, average="macro")
    rec_mac   = recall_score(y_true, y_pred, average="macro")
    mcc       = matthews_corrcoef(y_true, y_pred)
    auc       = roc_auc_score(y_true, y_prob)
    cm        = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ── Print results ───────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("   RESULTS")
    print(f"{'='*64}")

    print(f"""
  ┌─────────────────────────────────────┐
  │         Overall Metrics             │
  ├──────────────────────┬──────────────┤
  │  Accuracy            │  {acc*100:>7.2f} %  │
  │  ROC-AUC             │  {auc:>8.4f}  │
  │  MCC                 │  {mcc:>8.4f}  │
  │  Precision (macro)   │  {prec_mac:>8.4f}  │
  │  Recall    (macro)   │  {rec_mac:>8.4f}  │
  │  F1        (macro)   │  {f1_macro:>8.4f}  │
  └──────────────────────┴──────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │              Per-Class Metrics                             │
  ├───────────┬───────────┬──────────┬──────────┬─────────────┤
  │  Class    │ Precision │  Recall  │    F1    │   Support   │
  ├───────────┼───────────┼──────────┼──────────┼─────────────┤
  │  REAL (0) │  {prec_real:.4f}   │  {rec_real:.4f}  │  {f1_real:.4f}  │     {real_count:>3}       │
  │  FAKE (1) │  {prec_fake:.4f}   │  {rec_fake:.4f}  │  {f1_fake:.4f}  │     {fake_count:>3}       │
  └───────────┴───────────┴──────────┴──────────┴─────────────┘

  ┌──────────────────────────────────────────┐
  │           Confusion Matrix               │
  │                                          │
  │              Pred REAL    Pred FAKE      │
  │  True REAL   {tn:>8,}    {fp:>8,}       │
  │  True FAKE   {fn:>8,}    {tp:>8,}       │
  └──────────────────────────────────────────┘
""")

    print(f"  Model load : {load_time:.1f}s   |   Inference : {infer_time:.1f}s")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
