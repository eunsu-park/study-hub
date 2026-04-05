"""
FastAPI Advanced — Dependency Injection and Authentication
Demonstrates: Depends(), OAuth2 password flow, JWT tokens, background tasks.

Run: pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt]
"""

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# In production, use python-jose and passlib — simplified here for clarity
app = FastAPI(title="Auth Demo")

# --- Fake DB ---
fake_users = {
    "alice": {
        "username": "alice",
        "email": "alice@example.com",
        "hashed_password": "fakehash_secret123",
        "disabled": False,
    }
}


# --- Models ---

class User(BaseModel):
    username: str
    email: str
    disabled: bool = False


class Token(BaseModel):
    access_token: str
    token_type: str


# --- Dependencies ---

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def fake_hash(password: str) -> str:
    return f"fakehash_{password}"


def fake_decode_token(token: str) -> User | None:
    """In production, decode a real JWT here."""
    user_data = fake_users.get(token)
    if user_data:
        return User(**user_data)
    return None


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> User:
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# --- Dependency Chaining Example ---

class DBSession:
    """Simulates a database session with cleanup."""
    def __init__(self):
        self.connected = True
        print("DB session opened")

    def close(self):
        self.connected = False
        print("DB session closed")


async def get_db():
    """Yield-based dependency — cleanup runs after response."""
    db = DBSession()
    try:
        yield db
    finally:
        db.close()


# --- Routes ---

@app.post("/token", response_model=Token)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = fake_users.get(form_data.username)
    if not user or user["hashed_password"] != fake_hash(form_data.password):
        raise HTTPException(status_code=400, detail="Incorrect credentials")
    # In production, create a real JWT token here
    return Token(access_token=form_data.username, token_type="bearer")


@app.get("/users/me", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_active_user)],
):
    return current_user


# --- Background Tasks ---

def write_log(message: str):
    """Runs after the response is sent."""
    with open("api.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} - {message}\n")


@app.post("/notify")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks,
    user: Annotated[User, Depends(get_active_user)],
):
    background_tasks.add_task(write_log, f"Notification sent to {email} by {user.username}")
    return {"message": f"Notification will be sent to {email}"}


@app.get("/db-demo")
async def db_demo(db: Annotated[DBSession, Depends(get_db)]):
    return {"db_connected": db.connected}
