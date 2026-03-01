# Exercise: FastAPI Advanced — Dependency Injection and Auth
# Practice with Depends(), OAuth2, and background tasks.
#
# Run: pip install fastapi uvicorn httpx pytest

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Annotated

app = FastAPI()


# Exercise 1: Create a dependency chain
# Implement get_db → get_current_user → get_admin_user
# Each depends on the previous one.

class FakeDB:
    users = {
        "alice": {"username": "alice", "role": "admin", "token": "alice-token"},
        "bob": {"username": "bob", "role": "user", "token": "bob-token"},
    }


# TODO: Implement get_db dependency (yield-based, prints open/close)

# TODO: Implement get_current_user (extracts token from OAuth2, looks up user)

# TODO: Implement get_admin_user (depends on get_current_user, checks role)


# Exercise 2: Rate limiter dependency
# Create a dependency that limits each IP to 5 requests per minute.
# Use an in-memory dict to track request counts.

# TODO: Implement rate_limiter dependency


# Exercise 3: Protected endpoints
# GET /me — returns current user (any authenticated user)
# GET /admin/users — returns all users (admin only)
# POST /admin/reset — admin-only action with background task

# TODO: Implement the endpoints above


# Exercise 4: Caching dependency
# Create a dependency that caches results for 60 seconds.
# GET /expensive?key=... — returns cached or computed result

# TODO: Implement cache dependency and endpoint


if __name__ == "__main__":
    print("Implement the exercises and test with:")
    print("  uvicorn 03_fastapi_auth:app --reload")
    print("  pytest 03_fastapi_auth.py -v")
