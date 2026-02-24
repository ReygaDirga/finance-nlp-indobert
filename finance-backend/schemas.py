from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


# -------- AUTH --------
class RegisterIn(BaseModel):
    username: str
    email: str
    password: str


class LoginIn(BaseModel):
    identifier: str  # username or email
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeOut(BaseModel):
    id: int
    username: str
    email: str


# -------- INCOME --------
class IncomeIn(BaseModel):
    amount: int = Field(ge=0)


class IncomeOut(BaseModel):
    amount: int


# -------- PREDICT --------
class PredictIn(BaseModel):
    text: str


class PredictOut(BaseModel):
    category: str
    confidence: float  # 0..1


# -------- TRANSACTIONS --------
class TransactionCreate(BaseModel):
    text: str
    amount: int
    category: str
    confidence: Optional[float] = None

class TransactionOut(BaseModel):
    id: int
    text: str
    amount: int
    category: str
    confidence: Optional[float] = None
    created_at: datetime

class TransactionsSummaryOut(BaseModel):
    needs: int
    wants: int
    savings: int
    debts: int
    invests: int

# -------- DASHBOARD --------
class DashboardOut(BaseModel):
    income: int
    targets: dict
    spent: dict
