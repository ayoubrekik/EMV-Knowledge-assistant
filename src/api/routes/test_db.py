# src/api/routes/test_db.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.core.db.deps import get_db

router = APIRouter()

@router.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    return {"status": "DB connected"}