"""Password-backed local accounts and signed bearer-token authentication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_user_context: ContextVar[str | None] = ContextVar("current_user_id", default=None)
_bearer = HTTPBearer(auto_error=False)


class AuthStore:
    """SQLite account storage with PBKDF2 password hashing and HMAC tokens."""

    def __init__(
        self, db_path: str | Path = "data/auth/auth.db", token_ttl: int = 86400 * 7
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_ttl = token_ttl
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, salt BLOB NOT NULL, password_hash BLOB NOT NULL, created_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS auth_config (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO auth_config(key, value) VALUES('signing_key', ?)",
                (secrets.token_bytes(32),),
            )
            conn.commit()
            self._signing_key = bytes(
                conn.execute(
                    "SELECT value FROM auth_config WHERE key='signing_key'"
                ).fetchone()[0]
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _hash_password(password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)

    def register(self, username: str, password: str) -> dict[str, str]:
        normalized = username.strip().casefold()
        if not 3 <= len(normalized) <= 64 or not all(
            c.isalnum() or c in "._-@" for c in normalized
        ):
            raise ValueError(
                "Username must be 3–64 characters using letters, numbers, '.', '_', '-' or '@'."
            )
        if len(password) < 8:
            raise ValueError("Password must contain at least 8 characters.")
        salt = secrets.token_bytes(16)
        user_id = f"usr_{uuid.uuid4().hex}"
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO users(user_id, username, salt, password_hash, created_at) VALUES(?,?,?,?,?)",
                    (
                        user_id,
                        normalized,
                        salt,
                        self._hash_password(password, salt),
                        int(time.time()),
                    ),
                )
                conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("An account with that username already exists.") from exc
        return {
            "user_id": user_id,
            "username": normalized,
            "access_token": self._issue_token(user_id),
        }

    def login(self, username: str, password: str) -> dict[str, str] | None:
        normalized = username.strip().casefold()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, salt, password_hash FROM users WHERE username=?",
                (normalized,),
            ).fetchone()
        if not row or not hmac.compare_digest(
            self._hash_password(password, row["salt"]), row["password_hash"]
        ):
            return None
        return {
            "user_id": row["user_id"],
            "username": normalized,
            "access_token": self._issue_token(row["user_id"]),
        }

    def _issue_token(self, user_id: str) -> str:
        payload = json.dumps(
            {"sub": user_id, "exp": int(time.time()) + self.token_ttl},
            separators=(",", ":"),
        ).encode()
        body = base64.urlsafe_b64encode(payload).rstrip(b"=")
        signature = hmac.new(self._signing_key, body, hashlib.sha256).digest()
        return f"{body.decode()}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"

    def verify(self, token: str) -> str | None:
        try:
            body_part, signature_part = token.split(".", 1)
            body = body_part.encode()
            supplied = base64.urlsafe_b64decode(
                signature_part + "=" * (-len(signature_part) % 4)
            )
            expected = hmac.new(self._signing_key, body, hashlib.sha256).digest()
            if not hmac.compare_digest(supplied, expected):
                return None
            payload = json.loads(
                base64.urlsafe_b64decode(body + b"=" * (-len(body) % 4))
            )
            if int(payload["exp"]) <= int(time.time()):
                return None
            with self._connect() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM users WHERE user_id=?", (payload["sub"],)
                ).fetchone()
            return payload["sub"] if exists else None
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None


default_auth_store = AuthStore()


async def authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    token = (
        credentials.credentials
        if credentials and credentials.scheme.lower() == "bearer"
        else ""
    )
    user_id = default_auth_store.verify(token) if token else None
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to access this resource.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


class UserContextMiddleware:
    """Bind the verified bearer identity to the request context for sync route code."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        parts = authorization.split(None, 1)
        user_id = (
            default_auth_store.verify(parts[1])
            if len(parts) == 2 and parts[0].lower() == "bearer"
            else None
        )
        context_token = _user_context.set(user_id)
        try:
            await self.app(scope, receive, send)
        finally:
            _user_context.reset(context_token)


def current_user_id() -> str:
    """Return the authenticated owner ID inside a protected route."""
    user_id = _user_context.get()
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required."
        )
    return user_id


def owner_filters(filters: dict | None = None) -> dict:
    """Add an unoverrideable ownership constraint to retrieval filters."""
    scoped = dict(filters or {})
    scoped["owner_id"] = current_user_id()
    return scoped


def require_owned_session(session_id: str) -> None:
    """Raise not-found for missing or foreign sessions to avoid ID enumeration."""
    from src.memory.session_registry import default_session_registry

    session = default_session_registry.get_session(session_id)
    if not session or session.user_id != current_user_id():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
        )
