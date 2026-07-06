from pydantic import BaseModel
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from passlib.context import CryptContext

from src.core.db.deps import get_db
from src.core.db.models import User


router = APIRouter(prefix="/auth", tags=["Auth"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: Literal["user", "admin"]


@router.post("/register")
def register_user(payload: RegisterRequest, db: Session = Depends(get_db)):
    username = payload.username.strip()
    email = payload.email.strip().lower()
    password = payload.password
    role = payload.role

    existing_user = (
        db.query(User)
        .filter(or_(User.username == username, User.email == email))
        .first()
    )

    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already exists")

    user = User(
        username=username,
        email=email,
        password_hash=pwd_context.hash(password),
        role=role,
        is_active=True
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {"success": True}