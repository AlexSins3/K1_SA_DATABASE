"""Admin panel tab — activity logs & user management (admin-only)."""

import json

import requests
import streamlit as st

from auth import get_current_user, is_admin
from config import API_BASE_URL

API_BASE = API_BASE_URL


def _headers() -> dict:
    return {"token": st.session_state.get("token", "")}


def show_admin_tab():
    """Render the admin panel. Only visible to admin users."""
    if not is_admin():
        st.warning("Accès réservé à l'administrateur.")
        return

    st.header("🛠️ Panneau d'administration")

    section = st.radio(
        "Section",
        ["📊 Activité des utilisateurs", "👥 Gestion des comptes"],
        horizontal=True,
    )

    if section == "📊 Activité des utilisateurs":
        _show_activity_logs()
    else:
        _show_user_management()


def _show_activity_logs():
    st.subheader("Historique des consultations")

    col1, col2 = st.columns([1, 3])
    with col1:
        limit = st.number_input("Nombre d'entrées", min_value=10, max_value=1000, value=50, step=10)

    try:
        resp = requests.get(
            f"{API_BASE}/tracking/activity",
            headers=_headers(),
            params={"limit": limit},
            timeout=5,
        )
    except requests.ConnectionError:
        st.error("API indisponible.")
        return

    if resp.status_code != 200:
        st.error(f"Erreur API : {resp.status_code}")
        return

    logs = resp.json()

    if not logs:
        st.info("Aucune activité enregistrée pour l'instant.")
        return

    # Summary metrics
    import pandas as pd
    df = pd.DataFrame(logs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total consultations", len(df))
    col2.metric("Utilisateurs actifs", df["username"].nunique())
    col3.metric("Pages vues", df["page"].nunique())

    # Breakdown by page
    st.markdown("**Répartition par page :**")
    page_counts = df["page"].value_counts()
    st.bar_chart(page_counts)

    # Breakdown by user
    st.markdown("**Activité par utilisateur :**")
    user_counts = df.groupby("username")["page"].count().sort_values(ascending=False)
    st.dataframe(user_counts.reset_index().rename(columns={"page": "consultations"}), use_container_width=True)

    # Full log table
    with st.expander("📋 Journal complet", expanded=False):
        st.dataframe(
            df[["timestamp", "username", "page", "action", "details"]].sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def _show_user_management():
    st.subheader("Comptes utilisateurs")

    # List existing users
    try:
        resp = requests.get(f"{API_BASE}/auth/users", headers=_headers(), timeout=5)
    except requests.ConnectionError:
        st.error("API indisponible.")
        return

    if resp.status_code != 200:
        st.error(f"Erreur : {resp.status_code}")
        return

    users = resp.json()

    import pandas as pd
    df = pd.DataFrame(users)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%d/%m/%Y %H:%M")
    df["last_login"] = pd.to_datetime(df["last_login"]).dt.strftime("%d/%m/%Y %H:%M")

    st.dataframe(
        df[["username", "email", "role", "is_active", "created_at", "last_login"]],
        use_container_width=True,
        hide_index=True,
    )

    # Create new user
    st.markdown("---")
    st.markdown("**Créer un nouveau compte :**")

    with st.form("create_user_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_username = st.text_input("Nom d'utilisateur")
            new_email = st.text_input("Email")
        with col2:
            new_password = st.text_input("Mot de passe", type="password")
            new_role = st.selectbox("Rôle", ["user", "admin"])

        submitted = st.form_submit_button("Créer le compte", use_container_width=True)

    if submitted:
        if not new_username or not new_email or not new_password:
            st.error("Tous les champs sont requis.")
            return
        try:
            resp = requests.post(
                f"{API_BASE}/auth/register",
                headers=_headers(),
                json={
                    "username": new_username,
                    "email": new_email,
                    "password": new_password,
                    "role": new_role,
                },
                timeout=5,
            )
        except requests.ConnectionError:
            st.error("API indisponible.")
            return

        if resp.status_code == 200:
            st.success(f"Compte '{new_username}' créé avec succès !")
            st.rerun()
        else:
            st.error(f"Erreur : {resp.json().get('detail', 'Inconnu')}")

    # Change password for any user (admin)
    st.markdown("---")
    st.markdown("**Changer le mot de passe d'un utilisateur :**")

    with st.form("admin_change_password_form"):
        col1, col2 = st.columns(2)
        with col1:
            target_user = st.selectbox("Utilisateur", [u["username"] for u in users])
        with col2:
            new_pwd = st.text_input("Nouveau mot de passe", type="password", key="admin_new_pwd")
            confirm_pwd = st.text_input("Confirmer", type="password", key="admin_confirm_pwd")

        pwd_submitted = st.form_submit_button("Modifier le mot de passe", use_container_width=True)

    if pwd_submitted:
        if not new_pwd:
            st.error("Le mot de passe ne peut pas être vide.")
        elif new_pwd != confirm_pwd:
            st.error("Les mots de passe ne correspondent pas.")
        else:
            try:
                resp = requests.post(
                    f"{API_BASE}/auth/admin/change-password",
                    headers=_headers(),
                    json={"username": target_user, "new_password": new_pwd},
                    timeout=5,
                )
            except requests.ConnectionError:
                st.error("API indisponible.")
                return
            if resp.status_code == 200:
                st.success(f"Mot de passe de '{target_user}' mis à jour !")
            else:
                st.error(f"Erreur : {resp.json().get('detail', 'Inconnu')}")
