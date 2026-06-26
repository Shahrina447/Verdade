"""
Model loading logic.

Handles three common .pt save formats:
  1. torch.save(model)                     → full nn.Module
  2. torch.save(model.state_dict())        → bare OrderedDict
  3. torch.save({"model_state_dict": ...}) → wrapped dict
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from app.config import DEVICE, HF_MODEL_NAME, MODEL_PATH, NUM_LABELS

log = logging.getLogger("verdade.loader")


def load_model() -> tuple[AutoTokenizer, Any]:
    """Return (tokenizer, model) ready for inference."""

    log.info("Loading tokenizer '%s' …", HF_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    log.info("Loading checkpoint from '%s' …", MODEL_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # ── Case 1: full nn.Module ──────────────────────────────────────────────
    if isinstance(checkpoint, torch.nn.Module):
        log.info("Format: full nn.Module — using directly.")
        model = checkpoint.to(DEVICE).eval()
        return tokenizer, model

    # ── Case 2 / 3: state dict (bare or wrapped) ───────────────────────────
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        log.info("Format: state_dict — building architecture from config only.")

        # Use from_config to avoid re-downloading the 1 GB base weights —
        # they get overwritten by our .pt file immediately anyway.
        config = AutoConfig.from_pretrained(HF_MODEL_NAME, num_labels=NUM_LABELS)
        model = AutoModelForSequenceClassification.from_config(config)
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE).eval()
        return tokenizer, model

    raise RuntimeError(
        f"Unrecognised checkpoint format: {type(checkpoint)}. "
        "Expected nn.Module or a state dict."
    )
