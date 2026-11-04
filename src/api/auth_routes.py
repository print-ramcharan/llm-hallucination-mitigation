"""Authentication endpoints for the multi-user workspace."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.auth import default_auth_store

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class AuthResponse(BaseModel):
    user_id: str
    username: str
    access_token: str
    token_type: str = "bearer"


@router.post(
    "/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED
)
def register(payload: Credentials) -> AuthResponse:
    try:
        return AuthResponse(
            **default_auth_store.register(payload.username, payload.password)
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.post("/login", response_model=AuthResponse)
def login(payload: Credentials) -> AuthResponse:
    result = default_auth_store.login(payload.username, payload.password)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username or password is incorrect.",
        )
    return AuthResponse(**result)
