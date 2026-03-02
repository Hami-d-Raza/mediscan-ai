"""
models/user.py
--------------
Pydantic schemas (request / response models) for user authentication.
These are used for request validation and automatic API docs in FastAPI.
"""

from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    """Payload required to create a new account."""
    name: str     = Field(..., min_length=2, max_length=100, example="Hamid Raza")
    email: EmailStr = Field(..., example="hamid@example.com")
    password: str = Field(..., min_length=6, example="securepassword")


class UserLogin(BaseModel):
    """Payload required to log in."""
    email: EmailStr = Field(..., example="hamid@example.com")
    password: str   = Field(..., example="securepassword")


class TokenResponse(BaseModel):
    """JWT token returned after successful login."""
    access_token: str
    token_type: str = "bearer"
