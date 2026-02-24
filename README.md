# Finance Planning – NLP-Based Financial Text Classification

A web-based personal finance management system enhanced with NLP classification using Transformer models.

The system classifies Indonesian financial text into five categories:

- Needs  
- Wants  
- Debt  
- Invest  
- Saving  

This project consists of:
- **Production system** → React + FastAPI (deploying IndoBERT)
- **Research phase** → Comparative experiments across multiple models

---

# Tech Stack

- Frontend: React (Vite)
- Backend: FastAPI
- NLP Models: IndoBERT, mBERT, XLM-R, SVM, Decision Tree
- Language: Python, JavaScript
- Framework: PyTorch, Transformers (HuggingFace)

---

# Model Downloads

Model weights are excluded from this repository due to GitHub file size limits.

---

## Main Production Model (Required)

This is the deployed model used by the FastAPI backend.

👉 **Download IndoBERT Model:**  
`https://drive.google.com/file/d/1Plvh-Z6boBXFqmeZDP33Q9Vu8fCdKN90/view?usp=sharing`

After downloading, place the folder inside:

```
finance-backend/
```

Expected structure:

```
finance-backend/
└── indobert-dataset-final/
```

This model is required to run the backend server.

---

## Full Experimental Models (Optional)

This archive contains all trained models used during the research phase:

- IndoBERT (baseline & tuned)
- mBERT
- XLM-RoBERTa
- SVM
- Decision Tree

👉 **Download Full Experimental Models (Optional):**  
`https://drive.google.com/file/d/1SrOmJI1shU0lvsdTS1MA86kRHBFlgI2k/view?usp=sharing`

These models are **not required** to run the web application.

All comparative experiments were conducted in:

```
experiments/train.ipynb
```

---

# Installation & Setup

Two terminals are required.

---

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/finance-nlp-indobert.git
cd finance-nlp-indobert
```

---

# Backend Setup (FastAPI)

## Step 1 – Navigate to Backend

```bash
cd finance-backend
```

## Step 2 – Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

**Windows**
```bash
.venv\Scripts\activate
```

**Mac/Linux**
```bash
source .venv/bin/activate
```

## Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4 – Ensure Model Exists

Make sure the IndoBERT model folder exists:

```
finance-backend/indobert-dataset-final/
```

## Step 5 – Run Backend Server

```bash
uvicorn main:app --reload --port 8000
```

Backend runs at:

```
http://localhost:8000
```

---

# Frontend Setup (React + Vite)

Open a new terminal.

## Step 1 – Navigate to Frontend

```bash
cd finance-frontend
```

## Step 2 – Install Dependencies

```bash
npm install
```

## Step 3 – Run Development Server

```bash
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

# Experimental Phase

All model training and comparative evaluation were conducted inside:

```
experiments/train.ipynb
```

Models evaluated:

- IndoBERT
- mBERT
- XLM-RoBERTa
- Support Vector Machine (SVM)
- Decision Tree

Only **IndoBERT** is deployed in the production backend.

---

# Project Structure

```
finance-nlp-indobert/
│
├── finance-backend/
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── indobert-dataset-final/ (not included)
│
├── finance-frontend/
│
├── experiments/
│   └── train.ipynb
│
├── README.md
└── .gitignore
```

---

# Model Overview

- Architecture: Transformer-based (IndoBERT)
- Task: Multi-class text classification
- Language: Indonesian
- Deployment: REST API via FastAPI

---

# Contributors

| Name | Contribution | GitHub Username |
|------|-------------|--------|
| Delicia Nathania | EDA & mBert Model | [@delicia1220](https://github.com/delicia1220) |
| Jonea Kristiawan | Decision Tree Model | [@joneakristiawan](https://github.com/joneakristiawan) |
| Kevin Tanwioutra | SVM Model | [@TakeshiKei](https://github.com/TakeshiKei) |
| Nicholas William | EDA & XLM-RoBERTa Model | [@NicholasWilliam](https://github.com/NicholasWilliam) |
| Rio Dwi Oktavianto | IndoBERT Model & Fine-Tuning | [@ReygaDirga](https://github.com/ReygaDirga) |

---

# Notes

- Large model files are excluded from the repository.
- Download the main IndoBERT model before running the backend.
- Backend must be running before starting the frontend.

