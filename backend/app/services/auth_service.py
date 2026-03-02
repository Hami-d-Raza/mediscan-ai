"""
services/auth_service.py
------------------------
Business logic for user authentication.
Handles password hashing, user lookup, and JWT token generation.
In a production system, replace the in-memory store with a real database (e.g., PostgreSQL).
"""

from datetime import datetime, timedelta
from typing import Optional
import hashlib
import hmac
import uuid

from jose import jwt

from app.config import settings
from app.models.user import UserRegister, UserLogin, TokenResponse

# ---------------------------------------------------------------------------
# Temporary in-memory user store  (replace with DB in production)
# ---------------------------------------------------------------------------
_users: dict[str, dict] = {}


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def register_user(user: UserRegister) -> bool:
    """Register a new user. Returns False if the email is already taken."""
    if user.email in _users:
        return False
    _users[user.email] = {
        "id": str(uuid.uuid4()),
        "name": user.name,
        "email": user.email,
        "hashed_password": _hash_password(user.password),
    }
    return True


def authenticate_user(credentials: UserLogin) -> Optional[TokenResponse]:
    """Validate credentials and return a JWT token, or None on failure."""
    user = _users.get(credentials.email)
    if not user:
        return None
    if user["hashed_password"] != _hash_password(credentials.password):
        return None

    token = _create_access_token({"sub": user["email"], "name": user["name"]})
    return TokenResponse(access_token=token, token_type="bearer")
