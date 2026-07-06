from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID

from src.core.db.deps import get_db
from src.core.db.models import User
from src.core.auth import require_admin


router = APIRouter(prefix="/settings", tags=["Settings"])


@router.get("/users")
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    users = (
        db.query(User)
        .order_by(User.created_at.desc())
        .all()
    )

    return {
        "success": True,
        "users_count": len(users),
        "users": [
            {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat() if user.created_at else None,
            }
            for user in users
        ],
    }


@router.delete("/users/{user_id}")
def delete_user(
    user_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")

    db.delete(user)
    db.commit()

    return {
        "success": True,
        "message": "User deleted successfully",
        "deleted_user_id": str(user_id),
    }