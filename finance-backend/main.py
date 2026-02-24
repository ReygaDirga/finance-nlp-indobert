# main.py
from __future__ import annotations

from typing import Optional, Literal, Dict, List
from datetime import datetime, date, time, timedelta
import os

from fastapi import FastAPI, Depends, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func

from pydantic import BaseModel, Field, EmailStr

from db import Base, engine, get_db
from models import User, Income, Transaction
from auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

# ==== ML (IndoBERT) ====
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

# =========================
# Pydantic Schemas
# =========================
class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=6, max_length=200)

class LoginIn(BaseModel):
    identifier: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MeOut(BaseModel):
    id: int
    username: str
    email: str

class IncomeIn(BaseModel):
    amount: int = Field(gt=0)

class IncomeOut(BaseModel):
    amount: int

Cat = Literal["needs", "wants", "debts", "savings", "invests"]
VALID_CATS = {"needs", "wants", "debts", "savings", "invests"}

class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    category: str
    confidence: float

class TransactionCreate(BaseModel):
    text: str
    amount: int = Field(gt=0)
    category: Cat
    confidence: Optional[float] = None 

class TransactionUpdate(BaseModel):
    text: Optional[str] = None
    amount: Optional[int] = Field(default=None, gt=0)
    category: Optional[Cat] = None

class TransactionOut(BaseModel):
    id: int
    text: str
    amount: int
    category: Cat
    confidence: Optional[float] = None
    created_at: datetime

class TransactionsSummaryOut(BaseModel):
    needs: int = 0
    wants: int = 0
    debts: int = 0
    savings: int = 0
    invests: int = 0

class DashboardOut(BaseModel):
    username: str
    period_year: int
    period_month: int
    income: int
    targets: Dict[str, int]
    spent: Dict[str, int]

class DayBucket(BaseModel):
    day: str
    totals: TransactionsSummaryOut
    total_amount: int = 0
    count: int = 0

class CalendarDayTx(BaseModel):
    id: int
    text: str
    amount: int
    category: Cat
    confidence: Optional[float] = None
    created_at: datetime

class CalendarDayOut(BaseModel):
    day: str
    total_amount: int = 0
    count: int = 0
    by_category: TransactionsSummaryOut
    transactions: List[CalendarDayTx] = []

class ReportSummaryOut(BaseModel):
    period: Literal["daily", "monthly", "yearly"]
    start: str
    end: str
    totals: TransactionsSummaryOut
    total_amount: int = 0
    count: int = 0
    buckets: List[DayBucket] = []

# =========================
# ML PREDICT CONFIG
# =========================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "indobert-dataset-final")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

_tokenizer, _model, _label_encoder = None, None, None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global _tokenizer, _model, _label_encoder
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
        _model.eval()
    if _label_encoder is None:
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)

@app.on_event("startup")
def _startup():
    try: load_model()
    except Exception as e: print(f"[WARN] Model error: {e}")

# =========================
# HELPERS
# =========================
def _dt_range_for_month_simple(year: int, month: int):
    start = date(year, month, 1)
    if month == 12: end = date(year + 1, 1, 1)
    else: end = date(year, month + 1, 1)
    return datetime.combine(start, time.min), datetime.combine(end, time.min)

def _sum_by_category(db: Session, user_id: int, s: datetime, e: datetime) -> dict:
    rows = db.query(Transaction.category, func.coalesce(func.sum(Transaction.amount), 0), func.count(Transaction.id)).filter(
        Transaction.user_id == user_id, Transaction.created_at >= s, Transaction.created_at < e
    ).group_by(Transaction.category).all()
    out = {c: 0 for c in VALID_CATS}
    total, count = 0, 0
    for cat, t, c in rows:
        if cat in out: out[cat] = int(t or 0)
        total += int(t or 0); count += int(c or 0)
    return {"by_cat": out, "total_amount": total, "count": count}

# =========================
# ROUTES
# =========================
@app.get("/")
def root(): return {"ok": True, "app": "AIFI"}

# FIXED: Register diperkuat dengan log dan pengecekan ganda
@app.post("/auth/register")
def register(inp: RegisterIn, db: Session = Depends(get_db)):
    print(f"[AUTH] Attempting register: {inp.username}")
    u_strip = inp.username.strip()
    e_strip = inp.email.strip().lower()
    
    # Cek username atau email sudah ada belum
    existing = db.query(User).filter((User.username == u_strip) | (User.email == e_strip)).first()
    if existing:
        print(f"[AUTH] Register failed: User already exists")
        raise HTTPException(status_code=400, detail="Username or Email already taken")
    
    try:
        u = User(username=u_strip, email=e_strip, password_hash=hash_password(inp.password))
        db.add(u); db.commit()
        print(f"[AUTH] User {u_strip} registered successfully")
        return {"message": "registered"}
    except Exception as e:
        db.rollback()
        print(f"[ERROR] DB Register Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/auth/login", response_model=TokenOut)
def login(inp: LoginIn, db: Session = Depends(get_db)):
    ident = inp.identifier.strip()
    u = db.query(User).filter((User.username == ident) | (User.email == ident)).first()
    if not u or not verify_password(inp.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_access_token(subject=u.username), "token_type": "bearer"}

@app.get("/me", response_model=MeOut)
def me(u: User = Depends(get_current_user)): return u

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    load_model()
    with torch.no_grad():
        enc = _tokenizer(inp.text.strip(), truncation=True, padding=True, max_length=128, return_tensors="pt").to(_device)
        probs = torch.softmax(_model(**enc).logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
    try:
        raw = str(_label_encoder.inverse_transform([pred_idx])[0]).lower().strip()
        if raw == "invest": raw = "invests"
        if raw == "saving": raw = "savings"
        if raw == "debt": raw = "debts"
        final = raw if raw in VALID_CATS else "needs"
    except: final = "needs"
    return PredictOut(category=final, confidence=float(probs[pred_idx].item()))

@app.get("/dashboard", response_model=DashboardOut)
def dashboard(year: Optional[int]=None, month: Optional[int]=None, user: User=Depends(get_current_user), db: Session=Depends(get_db)):
    inc = db.query(Income).filter(Income.user_id == user.id).first()
    if not inc: raise HTTPException(status_code=404, detail="Income not set")
    now = datetime.now()
    y, m = (year or now.year), (month or now.month)
    s, e = _dt_range_for_month_simple(y, m)
    sums = _sum_by_category(db, user.id, s, e)
    return DashboardOut(username=user.username, period_year=y, period_month=m, income=inc.amount, targets={"needs": 50, "wants": 15, "debts": 0, "savings": 15, "invests": 20}, spent=sums["by_cat"])

@app.post("/transactions", response_model=TransactionOut)
def create_tx(inp: TransactionCreate, user: User=Depends(get_current_user), db: Session=Depends(get_db)):
    tx = Transaction(user_id=user.id, text=inp.text, amount=inp.amount, category=inp.category, confidence=inp.confidence)
    db.add(tx); db.commit(); db.refresh(tx); return tx

@app.get("/transactions", response_model=List[TransactionOut])
def list_tx(user: User=Depends(get_current_user), db: Session=Depends(get_db)):
    return db.query(Transaction).filter(Transaction.user_id==user.id).order_by(Transaction.created_at.desc()).limit(200).all()

@app.patch("/transactions/{tx_id}", response_model=TransactionOut)
@app.put("/transactions/{tx_id}", response_model=TransactionOut)
def update_tx(tx_id: int, inp: TransactionUpdate, user: User=Depends(get_current_user), db: Session=Depends(get_db)):
    tx = db.query(Transaction).filter(Transaction.id==tx_id, Transaction.user_id==user.id).first()
    if not tx: raise HTTPException(status_code=404)
    if inp.text: tx.text = inp.text
    if inp.amount: tx.amount = inp.amount
    if inp.category and inp.category in VALID_CATS: tx.category = inp.category
    db.commit(); db.refresh(tx); return tx

@app.delete("/transactions/{tx_id}")
def delete_tx(tx_id: int, user: User=Depends(get_current_user), db: Session=Depends(get_db)):
    tx = db.query(Transaction).filter(Transaction.id==tx_id, Transaction.user_id==user.id).first()
    if not tx: raise HTTPException(status_code=404)
    db.delete(tx); db.commit(); return {"ok": True}

@app.get("/reports/summary", response_model=ReportSummaryOut)
def reports_summary(period: str="monthly", year: int=2026, month: int=2, start: str=None, end: str=None, user: User=Depends(get_current_user), db: Session=Depends(get_db)):
    if period == "yearly": s, e = datetime(year, 1, 1), datetime(year+1, 1, 1)
    elif period == "monthly": s, e = _dt_range_for_month_simple(year, month)
    else:
        d1, d2 = date.fromisoformat(start), date.fromisoformat(end)
        s, e = datetime.combine(d1, time.min), datetime.combine(d2 + timedelta(days=1), time.min)
    sums = _sum_by_category(db, user.id, s, e)
    buckets = []
    if period == "daily":
        cur = s.date()
        while cur < e.date():
            cs, ce = datetime.combine(cur, time.min), datetime.combine(cur + timedelta(days=1), time.min)
            ds = _sum_by_category(db, user.id, cs, ce)
            buckets.append(DayBucket(day=cur.isoformat(), totals=TransactionsSummaryOut(**ds["by_cat"]), total_amount=ds["total_amount"], count=ds["count"]))
            cur += timedelta(days=1)
    return ReportSummaryOut(period=period, start=s.date().isoformat(), end=(e-timedelta(seconds=1)).date().isoformat(), totals=TransactionsSummaryOut(**sums["by_cat"]), total_amount=sums["total_amount"], count=sums["count"], buckets=buckets)

@app.get("/reports/calendar", response_model=List[CalendarDayOut])
def reports_calendar(year: int=2026, month: int=2, user: User=Depends(get_current_user), db: Session=Depends(get_db)):
    s, e = _dt_range_for_month_simple(year, month)
    rows = db.query(Transaction).filter(Transaction.user_id==user.id, Transaction.created_at>=s, Transaction.created_at<e).all()
    day_map = { (s.date()+timedelta(days=i)).isoformat(): CalendarDayOut(day=(s.date()+timedelta(days=i)).isoformat(), by_category=TransactionsSummaryOut(), transactions=[]) for i in range((e-s).days) }
    for r in rows:
        ds = r.created_at.date().isoformat()
        if ds in day_map:
            item = day_map[ds]; item.total_amount += r.amount; item.count += 1
            cat_d = item.by_category.model_dump(); cat_d[r.category] += r.amount
            item.by_category = TransactionsSummaryOut(**cat_d)
            item.transactions.append(CalendarDayTx(id=r.id, text=r.text, amount=r.amount, category=r.category, confidence=r.confidence, created_at=r.created_at))
    return list(day_map.values())

@app.post("/income")
def set_inc(
    inp: Optional[IncomeIn] = None, 
    amount: Optional[int] = Body(None),
    user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    final_amount = 0
    if inp and inp.amount: final_amount = inp.amount
    elif amount: final_amount = amount
    else: raise HTTPException(status_code=400, detail="Amount is required")

    inc = db.query(Income).filter(Income.user_id == user.id).first()
    if inc: inc.amount = final_amount
    else: db.add(Income(user_id=user.id, amount=final_amount))
    db.commit(); return {"ok": True, "amount": final_amount}

@app.get("/income", response_model=IncomeOut)
def get_inc(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    inc = db.query(Income).filter(Income.user_id == user.id).first()
    if not inc: raise HTTPException(status_code=404, detail="Income not set")
    return IncomeOut(amount=inc.amount)