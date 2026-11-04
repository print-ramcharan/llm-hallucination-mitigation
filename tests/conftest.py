"""Shared authenticated test identity for protected application API routes."""

import pytest

from src.api.app import app
from src.auth import _user_context, authenticated_user


@pytest.fixture(autouse=True)
def authenticated_api_test_user():
    async def test_identity():
        _user_context.set("test-user")
        return "test-user"

    previous = app.dependency_overrides.get(authenticated_user)
    app.dependency_overrides[authenticated_user] = test_identity
    yield
    if previous is None:
        app.dependency_overrides.pop(authenticated_user, None)
    else:
        app.dependency_overrides[authenticated_user] = previous
