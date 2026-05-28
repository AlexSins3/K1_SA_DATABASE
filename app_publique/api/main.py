"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api.config import ADMIN_EMAIL, ADMIN_USERNAME
from api.models import User, get_db, init_db
from api.auth import hash_password
from api.routes.auth_routes import router as auth_router, get_current_user
from api.routes.tracking_routes import router as tracking_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB and bootstrap admin on startup."""
    init_db()
    _bootstrap_admin()
    yield


def _bootstrap_admin():
    """Create admin user if none exists (first run)."""
    from api.models import SessionLocal
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.role == "admin").first():
            admin = User(
                username=ADMIN_USERNAME,
                email=ADMIN_EMAIL,
                hashed_password=hash_password("admin"),  # Change immediately via .env or CLI
                role="admin",
            )
            db.add(admin)
            db.commit()
            print(f"[API] Admin user '{ADMIN_USERNAME}' created with default password. CHANGE IT!")
    finally:
        db.close()


app = FastAPI(
    title="K1 SA Database - Auth API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(tracking_router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}
