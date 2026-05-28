"""CLI utility to manage users (change password, create admin, etc.)."""

import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.models import init_db, SessionLocal, User
from api.auth import hash_password


def change_password(username: str, new_password: str):
    init_db()
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    if not user:
        print(f"User '{username}' not found.")
        return
    user.hashed_password = hash_password(new_password)
    db.commit()
    print(f"Password updated for '{username}'.")
    db.close()


def create_user(username: str, email: str, password: str, role: str = "user"):
    init_db()
    db = SessionLocal()
    if db.query(User).filter(User.username == username).first():
        print(f"User '{username}' already exists.")
        db.close()
        return
    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        role=role,
    )
    db.add(user)
    db.commit()
    print(f"User '{username}' ({role}) created.")
    db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_users.py change-password <username> <new_password>")
        print("  python manage_users.py create-user <username> <email> <password> [role]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "change-password" and len(sys.argv) == 4:
        change_password(sys.argv[2], sys.argv[3])
    elif cmd == "create-user" and len(sys.argv) >= 5:
        role = sys.argv[5] if len(sys.argv) > 5 else "user"
        create_user(sys.argv[2], sys.argv[3], sys.argv[4], role)
    else:
        print("Invalid arguments. Run without args for usage.")
