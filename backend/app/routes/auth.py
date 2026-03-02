"""
routes/auth.py
--------------
Authentication endpoints: user registration and login.
Delegates business logic to services/auth_service.py.
Returns a JWT access token on successful login.
"""

from fastapi import APIRouter, HTTPException, status
from app.models.user import UserRegister, UserLogin, TokenResponse
from app.services.auth_service import register_user, authenticate_user

router = APIRouter()


@router.post("/register", status_code=status.HTTP_201_CREATED, summary="Register a new user")
async def register(user: UserRegister):
    """Create a new user account."""
    result = register_user(user)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists.",
        )
    return {"message": "User registered successfully."}


@router.post("/login", response_model=TokenResponse, summary="Login and get access token")
async def login(credentials: UserLogin):
    """Authenticate user and return a JWT access token."""
    token = authenticate_user(credentials)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
