"""Tracking routes: log page views, get activity history."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.models import ActivityLog, User, get_db
from api.routes.auth_routes import get_current_user, require_admin

router = APIRouter(prefix="/tracking", tags=["tracking"])


# --- Schemas ---

class LogEventRequest(BaseModel):
    page: str
    action: str = "view"
    details: str | None = None


class ActivityResponse(BaseModel):
    id: int
    user_id: int
    username: str
    page: str
    action: str
    details: str | None
    timestamp: datetime

    class Config:
        from_attributes = True


# --- Endpoints ---

@router.post("/log", status_code=201)
def log_event(body: LogEventRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Log a page view or action for the current user."""
    entry = ActivityLog(
        user_id=user.id,
        page=body.page,
        action=body.action,
        details=body.details,
    )
    db.add(entry)
    db.commit()
    return {"status": "ok"}


@router.get("/activity", response_model=list[ActivityResponse])
def get_activity(
    limit: int = Query(100, le=1000),
    user_id: int | None = None,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Get activity logs (admin-only). Optionally filter by user_id."""
    query = db.query(ActivityLog).join(User)
    if user_id:
        query = query.filter(ActivityLog.user_id == user_id)
    logs = query.order_by(ActivityLog.timestamp.desc()).limit(limit).all()
    return [
        ActivityResponse(
            id=log.id,
            user_id=log.user_id,
            username=log.user.username,
            page=log.page,
            action=log.action,
            details=log.details,
            timestamp=log.timestamp,
        )
        for log in logs
    ]
