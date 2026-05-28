"""Streamlit authentication client — communicates with the FastAPI backend."""

import requests
import streamlit as st

from config import API_BASE_URL


def _api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def _headers() -> dict:
    token = st.session_state.get("token", "")
    return {"token": token}


def login(username: str, password: str) -> bool:
    """Authenticate against the API. Stores token & user in session_state."""
    try:
        resp = requests.post(
            _api_url("/auth/login"),
            json={"username": username, "password": password},
            timeout=5,
        )
    except requests.ConnectionError:
        st.error("Impossible de contacter le serveur d'authentification.")
        return False

    if resp.status_code == 200:
        data = resp.json()
        st.session_state["token"] = data["access_token"]
        st.session_state["user"] = data["user"]
        st.session_state["authenticated"] = True
        return True
    else:
        st.error("Identifiants incorrects.")
        return False


def logout():
    """Clear session."""
    for key in ("token", "user", "authenticated"):
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    return st.session_state.get("authenticated", False)


def get_current_user() -> dict | None:
    return st.session_state.get("user")


def is_admin() -> bool:
    user = get_current_user()
    return user is not None and user.get("role") == "admin"


def log_page_view(page: str, action: str = "view", details: str | None = None):
    """Send a tracking event to the API (fire-and-forget)."""
    if not is_authenticated():
        return
    try:
        requests.post(
            _api_url("/tracking/log"),
            json={"page": page, "action": action, "details": details},
            headers=_headers(),
            timeout=2,
        )
    except requests.RequestException:
        pass  # Don't break the UI for tracking failures


def require_auth():
    """
    Gate function: shows login form if not authenticated.
    Call at the top of app.py — returns True if user is logged in.
    """
    if is_authenticated():
        return True

    st.markdown(
        "<h2 style='text-align:center;'>🔐 Connexion requise</h2>",
        unsafe_allow_html=True,
    )
    with st.form("login_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submitted = st.form_submit_button("Se connecter", use_container_width=True)

    if submitted and username and password:
        login(username, password)
        if is_authenticated():
            st.rerun()

    return False
