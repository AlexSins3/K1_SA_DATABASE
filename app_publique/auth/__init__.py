"""Authentication module — Streamlit client for the FastAPI auth backend."""

from auth.client import (
    get_current_user,
    is_admin,
    is_authenticated,
    log_page_view,
    login,
    logout,
    require_auth,
)

__all__ = [
    "get_current_user",
    "is_admin",
    "is_authenticated",
    "log_page_view",
    "login",
    "logout",
    "require_auth",
]
