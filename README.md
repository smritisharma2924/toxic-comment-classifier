# 🛡️ Toxic Comment Classifier

> A production-grade multi-label toxic comment classifier powered by fine-tuned DistilBERT, served via FastAPI, containerized with Docker, and deployed on GCP Cloud Run with async SQLite request logging.

**🔗 Live Demo:** [toxic-classifier-62230506999.us-central1.run.app](https://toxic-classifier-62230506999.us-central1.run.app)

---

## 📌 What It Does

This system classifies user-submitted comments across **6 toxicity labels simultaneously**:

| Label | Description |
|---|---|
| `toxic` | General toxic language |
| `severe_toxic` | Highly offensive content |
| `obscene` | Obscene or vulgar language |
| `threat` | Direct or implicit threats |
| `insult` | Targeted insults |
| `identity_hate` | Hate speech targeting identity groups |

A comment can belong to **multiple labels at once** — making this a true multi-label classification problem, not a simple binary classifier.

---

## 🏗️ Architecture

```
User Input (Browser)
        │
        ▼
  FastAPI Backend  ◄──── Preprocessing (clean_text)
        │
        ▼
  DistilBERT Model  ──► 6 sigmoid outputs
        │
        ▼
  Prediction Response
  {label, confidence, uncertain}
        │
        ├──► Async SQLite Logger (BackgroundTasks)
        │
        ▼
   GCP Cloud Run (Docker Container)
```

**Stack:**
- **ML:** PyTorch, HuggingFace Transformers, Scikit-learn, Keras
- **Backend:** FastAPI, Uvicorn, Pydantic, aiosqlite
- **Frontend:** Vanilla HTML/CSS/JS (served via FastAPI static files)
- **Infrastructure:** Docker, GCP Cloud Run, GCP Container Registry
- **Training:** Kaggle GPU (T4), local M4 Mac (MPS)

---

## 📊 Model Comparison

Three models were trained progressively on the full **Jigsaw Toxic Comment dataset (159,571 rows)**:

| Model | Micro F1 | Precision | Recall | Parameters | Training Time | Threat F1 | Identity Hate F1 |
|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.61 | 0.47 | 0.88 | ~30K | 2.59s | 0.00 | 0.00 |
| BiLSTM | 0.74 | 0.79 | 0.69 | 2.75M | 138.49s | 0.00 | 0.00 |
| **DistilBERT (selected)** | **0.78** | **0.77** | **0.78** | **66.4M** | **65.51 min** | **0.51** | **0.51** |

**Why DistilBERT was selected for deployment:**
- Best overall Micro F1 (0.78)
- Most balanced precision/recall tradeoff
- Only model capable of detecting rare labels (`threat`, `identity_hate`)
- BiLSTM's higher precision (0.79) does not compensate for complete failure on rare, safety-critical labels

---

## 🔁 Project Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Monorepo scaffolding, environment setup | ✅ |
| 1 | Preprocessing pipeline (`clean_text`) | ✅ |
| 2 | Baseline model (TF-IDF + LogReg) | ✅ |
| 3 | Mid-tier model (BiLSTM) | ✅ |
| 4 | Advanced model (DistilBERT fine-tuning) | ✅ |
| 5 | Formal evaluation & model selection | ✅ |
| 6 | FastAPI skeleton (schemas, routers) | ✅ |
| 7 | Model serving integration | ✅ |
| 8 | Async SQLite request logging | ✅ |
| 9 | Failure mode audit & mitigation | ✅ |
| 10 | Frontend UI | ✅ |
| 11 | Docker containerization | ✅ |
| 12 | GCP Cloud Run deployment | ✅ |
| 13 | README | ✅ |

---

## 🔌 API Reference

### `POST /predict`

Classifies a comment for toxicity.

**Request:**
```json
{
  "text": "you are an idiot"
}
```

**Response:**
```json
{
  "label": "toxic",
  "confidence": 0.9952,
  "uncertain": false
}
```

**Response Fields:**

| Field | Type | Description |
|---|---|---|
| `label` | string | Predicted toxicity category |
| `confidence` | float | Model confidence (0.0 – 1.0) |
| `uncertain` | bool | True if confidence < 0.5 — treat as safe |

**Display Logic:**
- `uncertain: true` → show "No toxic content detected" regardless of label
- `uncertain: false` → show label and confidence score

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.12+
- Conda (for ML environment)
- Git

### 1. Clone the repo
```bash
git clone https://github.com/smritisharma2924/toxic-comment-classifier.git
cd toxic-comment-classifier
```

### 2. Set up ML environment
```bash
conda create -n toxic-ml python=3.12
conda activate toxic-ml
pip install torch transformers scikit-learn keras
```

### 3. Set up backend environment
```bash
cd backend_api
python -m venv toxicvenv
source toxicvenv/bin/activate
pip install -r requirements.txt
```

### 4. Add model artifacts
Download `distilbert_model.pt` and `distilbert_tokenizer/` and place them in:
```
backend_api/app/models/
```

### 5. Run the server
```bash
cd backend_api
source toxicvenv/bin/activate
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

---

## 🐳 Docker

```bash
# Build
docker build -t toxic-classifier .

# Run
docker run -p 8000:8000 toxic-classifier
```

---

## ⚠️ Known Failure Modes

| Failure | Example | Behaviour | Status |
|---|---|---|---|
| **Implicit threats** | "someone should make you disappear" | Model scores 0.6% — misses it | ✅ Mitigated via rule-based pattern matching |
| **Truncation** | Long comment with toxicity after 128 tokens | Truncated — toxicity never seen | Documented |
| **Multilingual inconsistency** | Hindi toxic → uncertain; French toxic → 99% | Inconsistent cross-language behaviour | Documented |
| **Sarcasm** | "oh great, another brilliant idea" | Correctly flagged as uncertain | ✅ Handled by uncertainty threshold |
| **Clean text forced label** | "have a wonderful day" | Returns label with uncertain: true | ✅ Handled by uncertainty threshold |

---

## 📁 Project Structure

```
toxic-comment-classifier/
├── backend_api/
│   ├── app/
│   │   ├── db/              # SQLite logger
│   │   ├── models/          # Model artifacts
│   │   ├── routers/         # API endpoints
│   │   ├── schemas/         # Pydantic models
│   │   ├── services/        # Inference, preprocessing, model loader
│   │   ├── static/          # Frontend (index.html)
│   │   ├── main.py          # FastAPI app
│   │   └── state.py         # Shared model store
│   └── requirements.txt
├── ml_pipeline/
│   ├── notebooks/           # Training notebooks
│   └── saved_models/        # Trained model artifacts
├── frontend_ui/             # Original frontend source
├── Dockerfile
└── README.md
```

---

## 👩‍💻 About

Built by **Smriti Sharma** — sophomore B.Tech CSE (AI/ML specialisation) as a flagship portfolio project.

**Training environment:** Kaggle GPU (T4) for BiLSTM and DistilBERT, local Apple M4 for baseline and inference testing.

**Dataset:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) — 159,571 labelled comments.