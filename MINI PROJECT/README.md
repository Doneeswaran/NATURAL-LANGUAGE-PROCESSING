<div align="center">
  <h1>🧠 NLP Answer Evaluator</h1>
  <p><b>An AI-powered, privacy-first local application for evaluating student answers against reference keys using advanced Natural Language Processing.</b></p>

  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" />
</div>

<br/>

## ✨ Features

- **🔒 100% Local & Private**: No API keys required. No data is sent to OpenAI or Anthropic. Everything runs on your own hardware.
- **🚀 State-of-the-Art ML**: Utilizes Hugging Face's `all-mpnet-base-v2` Sentence Transformer for extremely accurate zero-shot semantic matching.
- **📚 Advanced Grammar Parsing**: Uses `spaCy` to extract Noun Chunks and Named Entities, going beyond simple keyword matching to actually understand the concepts.
- **📊 Comprehensive Scoring**: Grades answers on a 0-100 scale across multiple dimensions: *Semantic Similarity, Completeness, Clarity,* and *Relevance*.
- **📄 Multi-Format Support**: Upload `.txt` or `.pdf` files directly in the browser.

---

## 🛠️ Architecture

1. **Frontend (`nlp_evaluator_v5.html`)**: A beautiful, vanilla HTML/CSS/JS interface that parses PDFs locally and sends raw text to the backend.
2. **Backend (`backend/app.py`)**: A lightning-fast Python `FastAPI` server that loads the ML models into memory and exposes a `/evaluate` endpoint.

---

## 🚀 Getting Started

### Prerequisites
- macOS, Linux, or Windows
- Python 3.10+ installed

### 1. Start the Machine Learning Server
Open your terminal and navigate to the backend folder to install the required libraries and boot up the server.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```
> **Note:** The first time you run this, it will pause for a minute or two to download the 420MB `all-mpnet-base-v2` model and the `spaCy` English grammar model.

### 2. Open the Interface
Simply double-click the `nlp_evaluator_v5.html` file to open it in your web browser. 

Upload your Question, Reference Answer, and Student Answer documents, then click **Evaluate Answer**!

---

## ⚙️ Evaluation Modes
- **Strict**: Heavily weights exact factual completeness. Punishes missing terminology.
- **Balanced**: A fair 50/50 split between semantic understanding and factual completeness.
- **Lenient**: Heavily weights semantic understanding. Good for rewarding conceptual knowledge even if exact terminology is missed.
