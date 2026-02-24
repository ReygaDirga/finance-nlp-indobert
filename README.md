# 📊 Finance Planning – NLP-Based Financial Text Classification

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

# 🧠 Tech Stack

- Frontend: React (Vite)
- Backend: FastAPI
- NLP Models: IndoBERT, mBERT, XLM-R, SVM, Decision Tree
- Language: Python, JavaScript
- Framework: PyTorch, Transformers (HuggingFace)

---

# 📥 Model Downloads

Model weights are excluded from this repository due to GitHub file size limits.

---

## 1️⃣ Main Production Model (Required)

This is the deployed model used by the FastAPI backend.

👉 **Download IndoBERT Model:**  
`PASTE_MAIN_INDOBERT_LINK_HERE`

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

## 2️⃣ Full Experimental Models (Optional)

This archive contains all trained models used during the research phase:

- IndoBERT (baseline & tuned)
- mBERT
- XLM-RoBERTa
- SVM
- Decision Tree

👉 **Download Full Experimental Models (Optional):**  
`PASTE_FULL_MODELS_LINK_HERE`

These models are **not required** to run the web application.

All comparative experiments were conducted in:

```
experiments/train.ipynb
```

---

# ⚙️ Installation & Setup

Two terminals are required.

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/finance-nlp-indobert.git
cd finance-nlp-indobert
```

---

# 🖥 Backend Setup (FastAPI)

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

# 🌐 Frontend Setup (React + Vite)

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

# 🔬 Experimental Phase

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

# 🏗 Project Structure

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

# 📊 Model Overview

- Architecture: Transformer-based (IndoBERT)
- Task: Multi-class text classification
- Language: Indonesian
- Deployment: REST API via FastAPI

---

# 👥 Contributors

| Name | Contribution | GitHub |
|------|-------------|--------|
| A | EDA & XLM-RoBERTa Model | https://github.com/USERNAME_A |
| B | EDA & mBERT Model | https://github.com/USERNAME_B |
| C | IndoBERT Model & Fine-Tuning (Main Deployment Model) | https://github.com/USERNAME_C |
| D | SVM Model | https://github.com/USERNAME_D |
| E | Decision Tree Model | https://github.com/USERNAME_E |

---

# 📌 Notes

- Large model files are excluded from the repository.
- Download the main IndoBERT model before running the backend.
- Backend must be running before starting the frontend.
