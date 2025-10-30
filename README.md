# Chatguard API
Chat smarter, speak kinder.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-teal.svg)](#)
[![Transformers](https://img.shields.io/badge/Transformers-4.44%2B-purple.svg)](#)
[![Torch](https://img.shields.io/badge/PyTorch-2.3%2B-red.svg)](#)
[![Docker](https://img.shields.io/badge/Docker-ready-0db7ed.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

Production-ready FastAPI service for **toxic language detection** using a fine‑tuned XLM‑RoBERTa model hosted on Hugging Face Hub. Modular, type‑safe, and deployable on CPU or GPU.

---

## Features
- Loads a Hugging Face model at startup and serves low-latency inference
- Clean modular layout (config, model, schemas, routes)
- Single and batch prediction with optional probability thresholding
- JSON responses with stable probability keys (`clean`, `toxic`)
- OpenAPI docs via `/docs` and `/redoc`
- Health endpoint at `/health`

---

## Project Structure
```
toxicity-api/
├─ app/
│  ├─ main.py               # FastAPI entry point
│  ├─ core/
│  │  ├─ config.py          # Env + settings
│  │  └─ model.py           # Model loader + inference logic
│  ├─ api/
│  │  └─ routes.py          # HTTP endpoints
│  └─ schemas/
│     └─ predict.py         # Pydantic I/O models
├─ requirements.txt
└─ README.md
```

---

## Quickstart

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Configure
```bash
export MODEL_ID=your-username/toxic-xlmr
# optional
export DEVICE=cuda      # or cpu
export MAX_LENGTH=256
```

Or you can use the default on https://huggingface.co/cedrugs/toxic-xlmr (cedrugs/toxic-xlmr)

### 3) Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open docs at `http://localhost:8000/docs`.

---

## Environment Variables

| Variable     | Description                                   | Default                          |
|--------------|-----------------------------------------------|----------------------------------|
| `MODEL_ID`   | Hugging Face model repo (e.g. user/model)     | `your-username/toxic-xlmr`       |
| `DEVICE`     | `cpu` or `cuda` (auto-detect if unset)        | *auto*                           |
| `MAX_LENGTH` | Max token length per input                    | `256`                            |

---

## API

### Health
```
GET /health
```
Response:
```json
{ "status": "ok", "model_loaded": true, "device": "cuda" }
```

### Metadata
```
GET /
```

### Predict (single)
```
POST /v1/predict
```
Request:
```json
{ "text": "you are so dumb", "threshold": 0.5 }
```
Response:
```json
{
  "label": "toxic",
  "probs": { "clean": 0.12, "toxic": 0.88 },
  "latency_ms": 4.2
}
```

### Predict (batch)
```
POST /v1/batch_predict
```
Request:
```json
{ "texts": ["you suck", "have a nice day"], "threshold": 0.5 }
```
Response:
```json
{
  "results": [
    { "label": "toxic", "probs": { "clean": 0.09, "toxic": 0.91 }, "latency_ms": 2.1 },
    { "label": "clean", "probs": { "clean": 0.97, "toxic": 0.03 }, "latency_ms": 2.1 }
  ],
  "latency_ms": 4.2
}
```

---

## Docker

Build & run:
```bash
docker build -t chatguard-api .
docker run --rm -p 8000:8000 \
  -e MODEL_ID=your-username/toxic-xlmr \
  -e DEVICE=cpu \
  chatguard-api
```

---

## Production Notes
- For GPU: set `DEVICE=cuda` and ensure CUDA drivers are available.
- Prefer one worker per GPU. For CPU-bound scaling:
  ```bash
  gunicorn -k uvicorn.workers.UvicornWorker -w 4 app.main:app --bind 0.0.0.0:8000
  ```
- Pin model revisions in `MODEL_ID` for reproducible deployments (e.g., `user/model@sha`).
- Consider enabling request timeouts and reverse proxying behind Traefik/Caddy.

---

## Model Requirements
This API expects a Hugging Face repo containing a binary classifier with standard files:

```
pytorch_model.bin
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

Pushing to Hub example:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model.push_to_hub("toxic-xlmr")
tokenizer.push_to_hub("toxic-xlmr")
```

---

## License
MIT

---

## Acknowledgments
Built with FastAPI, Transformers, and PyTorch. Deployed anywhere from laptops to GPUs in the cloud.
