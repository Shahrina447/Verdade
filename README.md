<div align="center">

<img src="https://img.shields.io/badge/سچ_کی_پہچان-Recognition_of_Truth-7c3aed?style=for-the-badge&labelColor=0b1020" alt="SachAI tagline"/>

# SachAI — Urdu Fake News Detection System

**AI-powered misinformation detection for the Urdu language using xlm-RoBERTa**

[![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14.2-black?style=flat-square&logo=next.js&logoColor=white)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2_CPU-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-xlm--RoBERTa-FFD21E?style=flat-square)](https://huggingface.co/xlm-roberta-base)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=flat-square)](LICENSE)

<br/>

[Overview](#overview) · [Features](#features) · [Tech Stack](#tech-stack) · [Project Structure](#project-structure) · [Setup & Run](#setup--run) · [Model](#model) · [Model Performance](#model-performance) · [API Reference](#api-reference) · [Dataset](#dataset) · [Developer](#developer)

</div>

---

## Overview

SachAI is a full-stack NLP web application that detects misinformation in Urdu news articles using a fine-tuned **xlm-RoBERTa** transformer model. A user pastes any Urdu text and receives an instant verdict — **REAL** or **FAKE** — with a confidence percentage and a human-readable explanation.

The model is fine-tuned on a combined corpus of **Ax-to-Grind Urdu** and **Hook & Bait Urdu** — two expert-annotated Pakistani news datasets. The backend runs entirely on CPU using PyTorch float32, no GPU required.

---

## Features

| Feature | Description |
|---|---|
| Instant verdict | Paste any Urdu text and get REAL / FAKE with confidence % |
| Credibility meter | Animated circular meter color-coded by verdict |
| Sample texts | 3 pre-loaded Urdu articles for quick demo |
| Batch analysis | Test up to 10 articles at once |
| Prediction history | Full log with per-item delete and bulk clear |
| Urdu rendering | Native RTL text with Noto Nastaliq Urdu font |
| Text normalization | Arabic-to-Urdu character fix (`ي → ی`, `ك → ک`, `ه → ہ`) |
| CPU inference | Runs on any machine, no GPU needed |
| Persistent storage | SQLite database for prediction history |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Next.js 14 + TypeScript | UI, detector, history page |
| Styling | Tailwind CSS + custom CSS | Dark theme, RTL Urdu support |
| HTTP Client | Axios | API calls from frontend |
| Backend | FastAPI 0.111 + Python 3.12 | REST API, inference pipeline |
| Server | Uvicorn | ASGI server |
| Validation | Pydantic v2 | Request/response models |
| Database | SQLite via aiosqlite | Prediction history storage |
| ML Framework | PyTorch 2.2 (CPU-only) | Model inference |
| NLP Library | HuggingFace Transformers 4.40 | Tokenizer + model loading |
| Base Model | xlm-roberta-base | 278M param multilingual transformer |

---

## Project Structure

```
SachAI/
├── .gitignore
├── README.md
│
├── backend/
│   ├── main.py                  # FastAPI app, routes, CORS, lifespan
│   ├── model.py                 # xlm-RoBERTa inference (CPU-only)
│   ├── preprocess.py            # Urdu text normalization & cleaning
│   ├── schemas.py               # Pydantic request/response models
│   ├── database.py              # SQLite CRUD for prediction history
│   ├── evaluate.py              # Model evaluation script
│   ├── synthetic_test.csv       # Hand-crafted synthetic test data
│   ├── requirements.txt
│   ├── .env.example
│   └── saved_model/             # Trained weights — not in git (1.1 GB)
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       └── model.safetensors
│
└── frontend/
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx
    │   ├── globals.css
    │   └── history/
    │       └── page.tsx
    ├── components/
    │   ├── Navbar.tsx
    │   ├── HeroSection.tsx
    │   ├── DetectorSection.tsx
    │   ├── BatchAnalyzer.tsx
    │   ├── HowItWorks.tsx
    │   ├── StatsSection.tsx
    │   ├── FeaturesSection.tsx
    │   ├── MarqueeSection.tsx
    │   ├── CTASection.tsx
    │   └── Footer.tsx
    ├── lib/
    │   └── api.ts
    ├── package.json
    └── tailwind.config.ts
```

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.12 | [python.org](https://python.org) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| npm | 9+ | Comes with Node.js |

---

## Setup & Run

### 1 · Clone

```bash
git clone https://github.com/Shahrina447/SachAI.git
cd SachAI
```

### 2 · Backend

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install CPU-only PyTorch first (required before requirements.txt)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env

# Place trained model files in backend/saved_model/
# Required files: config.json, tokenizer.json, tokenizer_config.json,
#                 special_tokens_map.json, model.safetensors

# Start the server
uvicorn main:app --reload --port 8000
```

### 3 · Frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
# → http://localhost:3000
```

### 4 · Verify

```bash
# Health check
curl http://localhost:8000/api/health

# Test prediction
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "سائنسدانوں نے دعویٰ کیا کہ گرم پانی سے کینسر مکمل طور پر ختم ہو جاتا ہے!"}' \
  | python3 -m json.tool
```

---

## Environment Variables

**`backend/.env`**
```env
HOST=127.0.0.1
PORT=8000
MODEL_PATH=./saved_model
REAL_THRESHOLD=0.55        # min confidence to label as REAL
TEMPERATURE=1.5            # softmax temperature scaling
DB_PATH=./predictions.db
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

**`frontend/.env.local`**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Model

### Architecture

**Base:** `xlm-roberta-base` — a multilingual transformer pre-trained on 100 languages including Urdu (278M parameters).

**Task:** Binary sequence classification — `FAKE (0)` vs `TRUE (1)`

### Training

Fine-tuned on Google Colab (T4 GPU) using HuggingFace `Trainer` API on a combined dataset of Ax-to-Grind Urdu and Hook & Bait Urdu.

| Parameter | Value |
|---|---|
| Base model | xlm-roberta-base |
| Max token length | 128 |
| Batch size | 16 |
| Epochs | 2 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Precision | float32 (CPU inference) |

### Inference Settings

| Setting | Value | Purpose |
|---|---|---|
| Temperature | 1.5 | Softens overconfident softmax scores |
| REAL threshold | 0.55 | Model must be ≥55% confident to label REAL |
| Device | CPU | No GPU required |

> Model weights (`model.safetensors`, 1.1 GB) are not stored in this repository. Place trained files in `backend/saved_model/` before running.

---

## Model Performance

### Test 1 — Combined Dataset Held-Out Test
*1000 balanced samples (500 FAKE + 500 TRUE) from training distribution*

| Metric | Value |
|---|---|
| Accuracy | **94.40%** |
| Precision | 0.9784 |
| Recall | 0.9080 |
| F1 Score | 0.9419 |

| | Predicted FAKE | Predicted TRUE |
|---|---|---|
| **Actual FAKE** | 454 TP | 46 FN |
| **Actual TRUE** | 10 FP | 490 TN |

---

### Test 2 — Synthetic Out-of-Distribution Test
*50 hand-crafted Urdu news articles (25 FAKE + 25 TRUE) — completely outside training data*

| Metric | Value |
|---|---|
| Accuracy | **84.00%** |
| Precision | 1.0000 |
| Recall | 0.6800 |
| F1 Score | 0.8095 |

| | Predicted FAKE | Predicted TRUE |
|---|---|---|
| **Actual FAKE** | 17 TP | 8 FN |
| **Actual TRUE** | 0 FP | 25 TN |

**Key observations:**
- Precision = 1.0 — zero false alarms, every FAKE prediction was correct
- 8 missed fake news were all written in neutral journalistic tone — a known challenge in NLP
- 0 real news incorrectly flagged as fake

---

### Performance Summary

| Test Set | Type | Samples | Accuracy | F1 |
|---|---|---|---|---|
| Combined dataset | In-distribution | 1000 | **94.4%** | 0.942 |
| Synthetic data | Out-of-distribution | 50 | **84.0%** | 0.810 |

Retraining on a combined dataset improved cross-domain accuracy from 74.5% (single dataset) to 94.4%.

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Server health check |
| `POST` | `/api/predict` | Analyze a single Urdu text |
| `POST` | `/api/predict/batch` | Analyze up to 10 texts |
| `GET` | `/api/history` | Fetch last 50 predictions |
| `DELETE` | `/api/history/{id}` | Delete a specific prediction |
| `DELETE` | `/api/history` | Clear all predictions |

### POST /api/predict

**Request**
```json
{ "text": "اردو خبر کا متن یہاں — 10 to 1000 characters" }
```

**Response**
```json
{
  "prediction": "FAKE",
  "confidence": 0.997,
  "confidence_real": 0.003,
  "confidence_fake": 0.997,
  "verdict_text": "✗ Strong misinformation indicators detected.",
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-06-19T10:00:00"
}
```

**Error codes**

| Status | Reason |
|---|---|
| `400` | Text shorter than 10 characters |
| `400` | Text longer than 1000 characters |
| `422` | Malformed request body |
| `500` | Internal inference error |

---

## Dataset

| Dataset | Samples | Labels | Role | Source |
|---|---|---|---|---|
| Ax-to-Grind Urdu | 10,083 | FAKE / TRUE | Primary training | [arXiv:2403.14037](https://arxiv.org/abs/2403.14037) |
| Hook & Bait Urdu | ~59,000 | FAKE / TRUE | Combined training | Internal corpus |
| Synthetic test | 50 | FAKE / TRUE | Out-of-distribution eval | Hand-crafted |

> Dataset files are not included in this repository. Download from the sources above.

---

## Running the Evaluation Script

```bash
cd backend
source venv/bin/activate

# Evaluate on any labeled CSV
python evaluate.py --file test.csv --text-col "News Items" --label-col Label

# Balanced random sample (faster)
python evaluate.py --file test.csv --text-col "News Items" --label-col Label --sample 200

# Evaluate on synthetic test data
python evaluate.py --file synthetic_test.csv --text-col "News Items" --label-col Label
```

Output includes accuracy, precision, recall, F1, confusion matrix, and a detailed `*_results.csv` file.

---

## Developer

**Shahrina Khan**  
Computer Science Student — UMT Sialkot  
GitHub: [@Shahrina447](https://github.com/Shahrina447)
