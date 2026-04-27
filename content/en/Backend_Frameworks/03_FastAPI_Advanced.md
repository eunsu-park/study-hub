# 03. FastAPI Advanced

**Previous**: [FastAPI Basics](./02_FastAPI_Basics.md) | **Next**: [FastAPI Database](./04_FastAPI_Database.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Implement the Dependency Injection pattern using `Depends()` to share logic across endpoints
- Build a JWT-based authentication flow with OAuth2 password bearer scheme
- Design modular applications using `APIRouter` with proper prefix and tag organization
- Implement WebSocket endpoints for real-time bidirectional communication
- Manage application lifecycle events for startup and shutdown resource management

---

## Table of Contents

1. [Dependency Injection with Depends()](#1-dependency-injection-with-depends)
2. [Authentication: OAuth2 with JWT](#2-authentication-oauth2-with-jwt)
3. [File Uploads](#3-file-uploads)
4. [Background Tasks](#4-background-tasks)
5. [WebSocket Support](#5-websocket-support)
6. [Custom Middleware](#6-custom-middleware)
7. [APIRouter for Modular Apps](#7-apirouter-for-modular-apps)
8. [Lifespan Events](#8-lifespan-events)
9. [Practice Problems](#9-practice-problems)
10. [References](#10-references)

---

## 1. Dependency Injection with Depends()

Dependency Injection (DI) is FastAPI's mechanism for sharing reusable logic across endpoints. Instead of calling a function inside each handler, you declare it as a dependency and FastAPI resolves it automatically.

### Theory: Dependency Injection: A Graph, Not a Function Call

`Depends(callable)` is FastAPI's marker that says "compute the value of this parameter by calling that callable, and inject the result". The naïve mental model is "call the dependency function and pass the return value". The actual mechanism is more interesting — and explains the cache, override, and yield semantics.

#### A.1 The dependency DAG

Each handler has zero or more `Depends(...)` parameters. Each dependency *itself* is a callable that may have its own `Depends(...)` parameters. The transitive closure of these references forms a directed acyclic graph (DAG):

```
get_current_user
    ↓ depends on
get_token_from_header
    ↓ depends on
verify_signing_key
    ↓ depends on
settings
```

When a request arrives, FastAPI:

1. Topologically sorts the DAG so leaves resolve before their dependents.
2. Walks the sorted list, calling each dependency, awaiting if it is `async`, and stashing the result.
3. Passes the resolved values into the handler.

The graph is built once at startup (during the type-hint introspection from Lesson 02) and cached. Per-request work is just walking it. There is no reflection per request.

#### A.2 Per-request caching

By default, if the same dependency callable appears in multiple places in a single request's DAG, it is called *once*. The result is cached for the rest of that request and reused. This is what makes patterns like "every endpoint in this router needs the current user" cheap — even if `get_current_user` appears as a transitive dep five places, the JWT is decoded once.

The cache scope is *one request*. Two requests have two independent caches. Override the default with `Depends(callable, use_cache=False)` if a dependency must run every time it appears (rare).

#### A.3 `yield`-based dependencies: setup and teardown

A dependency function can use `yield` instead of `return`:

```python
async def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
```

Everything before `yield` runs at dependency resolution. The yielded value is what gets injected. Everything *after* `yield` runs after the response has been sent — a finalizer phase. This is structurally identical to a `contextlib.contextmanager`, and FastAPI implements it that way. It is the right pattern for any resource that must be released: DB sessions, file handles, HTTP clients, cache connections.

The teardown runs even if the handler raises — analogous to `try`/`finally`. If the handler raises an `HTTPException`, the dependency's `finally` block still runs, then the exception is converted to an HTTP response.

#### A.4 Sub-application overrides

For testing, `app.dependency_overrides[real_dep] = fake_dep` rewires a dependency without touching the handler. Because resolution happens at request time through the DAG, the override is picked up automatically — the handler never knew the difference. This is dependency injection's payoff for testing: you can substitute a fake database, fake auth, or fake clock without monkeypatching imports.

### Basic Dependencies

```python
from fastapi import FastAPI, Depends, Query

app = FastAPI()

# A dependency is just a callable (function or class)
async def common_pagination(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=10, ge=1, le=100),
) -> dict[str, int]:
    """Reusable pagination logic. Instead of repeating these
    query parameters in every list endpoint, declare them once."""
    return {"skip": skip, "limit": limit}

@app.get("/users")
async def list_users(pagination: dict = Depends(common_pagination)):
    # pagination = {"skip": 0, "limit": 10}
    return {"users": [], **pagination}

@app.get("/posts")
async def list_posts(pagination: dict = Depends(common_pagination)):
    # Same pagination logic, zero duplication
    return {"posts": [], **pagination}
```

### Class-Based Dependencies

```python
from dataclasses import dataclass

@dataclass
class PaginationParams:
    """Class-based dependency. FastAPI calls the constructor with
    query parameters matched by name, just like function dependencies."""
    skip: int = Query(default=0, ge=0)
    limit: int = Query(default=10, ge=1, le=100)

@app.get("/items")
async def list_items(params: PaginationParams = Depends()):
    # When using Depends() with no argument on a type-annotated parameter,
    # FastAPI uses the annotation (PaginationParams) as the dependency
    return {"skip": params.skip, "limit": params.limit}
```

### Dependency Chains

Dependencies can depend on other dependencies, forming a chain that FastAPI resolves in order:

```
get_current_user
    └── depends on: get_token_from_header
                        └── depends on: oauth2_scheme (extracts Bearer token)
```

```python
from fastapi import Header, HTTPException

async def get_api_key(x_api_key: str = Header(...)):
    """First level: extract API key from header."""
    return x_api_key

async def verify_api_key(api_key: str = Depends(get_api_key)):
    """Second level: validate the extracted key.
    Dependencies form a chain -- FastAPI resolves them in order."""
    valid_keys = {"key-abc-123", "key-def-456"}
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.get("/protected")
async def protected_route(api_key: str = Depends(verify_api_key)):
    # By the time this runs, the API key is already validated
    return {"message": "Access granted", "key": api_key}
```

### Yield Dependencies (Resource Cleanup)

```python
from typing import AsyncGenerator

async def get_db_session() -> AsyncGenerator:
    """Yield dependencies run setup code before yield and cleanup after.
    This pattern ensures database sessions are always closed,
    even if the endpoint raises an exception."""
    session = AsyncSession()
    try:
        yield session      # Endpoint runs here
    finally:
        await session.close()  # Always executes (like a context manager)

@app.get("/data")
async def get_data(db: AsyncSession = Depends(get_db_session)):
    result = await db.execute(select(User))
    return result.scalars().all()
```

---

## 2. Authentication: OAuth2 with JWT

FastAPI has built-in support for OAuth2 flows. The most common pattern for APIs is the **password bearer** flow with **JWT** (JSON Web Tokens).

### Overview

```
┌──────────┐        ┌──────────────┐        ┌──────────┐
│  Client   │        │  FastAPI      │        │  Database │
│           │        │  Server       │        │          │
│  POST /token       │               │        │          │
│  username+password  │               │        │          │
│ ─────────────────▶ │               │        │          │
│           │        │  Verify creds ──────▶  │          │
│           │        │               │ ◀────  │          │
│           │        │  Generate JWT │        │          │
│ ◀───────────────── │               │        │          │
│  {access_token}    │               │        │          │
│           │        │               │        │          │
│  GET /users/me     │               │        │          │
│  Authorization:    │               │        │          │
│  Bearer <JWT>      │               │        │          │
│ ─────────────────▶ │               │        │          │
│           │        │  Decode JWT   │        │          │
│           │        │  Extract user │        │          │
│ ◀───────────────── │               │        │          │
│  {user data}       │               │        │          │
└──────────┘        └──────────────┘        └──────────┘
```

### Implementation

```python
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

app = FastAPI()

# --- Configuration ---
# In production, load these from environment variables, never hardcode
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Password Hashing ---
# bcrypt is the recommended algorithm for password hashing
# because it's intentionally slow, making brute-force attacks impractical
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- OAuth2 Scheme ---
# tokenUrl tells the Swagger UI where to send login requests
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Models ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str
    email: str
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

# --- Fake Database ---
fake_users_db = {
    "alice": {
        "username": "alice",
        "email": "alice@example.com",
        "hashed_password": pwd_context.hash("secret123"),
        "disabled": False,
    }
}

# --- Helper Functions ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Compare a plain password against its hash.
    passlib handles the salt extraction and comparison internally."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT with an expiration time.
    The 'sub' (subject) claim identifies the user."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(username: str, password: str) -> UserInDB | None:
    """Verify credentials. Returns the user if valid, None otherwise.
    Always hash-compare even for non-existent users to prevent
    timing attacks that reveal valid usernames."""
    user_data = fake_users_db.get(username)
    if not user_data:
        return None
    user = UserInDB(**user_data)
    if not verify_password(password, user.hashed_password):
        return None
    return user

# --- Dependencies ---
async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)]
) -> User:
    """Decode the JWT and return the current user.
    This dependency is used by any endpoint that requires authentication."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user_data = fake_users_db.get(username)
    if user_data is None:
        raise credentials_exception
    return User(**user_data)

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Chain dependency: checks that the user is not disabled."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- Endpoints ---
@app.post("/token", response_model=Token)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """OAuth2 password flow endpoint.
    The Swagger UI provides a login form that posts to this URL."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=access_token, token_type="bearer")

@app.get("/users/me", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """Protected endpoint. Only accessible with a valid JWT."""
    return current_user
```

---

## 3. File Uploads

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Maximum file size: 5 MB
MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload with validation.
    UploadFile is preferred over bytes because it uses a spooled
    temporary file -- large uploads don't consume all memory."""

    # Validate content type before reading the file
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not allowed. "
                   f"Allowed: {ALLOWED_TYPES}"
        )

    # Read and check size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")

    # Save to disk with sanitized filename
    safe_name = Path(file.filename).name  # Strip directory components
    save_path = UPLOAD_DIR / safe_name
    save_path.write_bytes(contents)

    return {
        "filename": safe_name,
        "size_bytes": len(contents),
        "content_type": file.content_type,
    }

@app.post("/upload-multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    """Accept multiple files in a single request."""
    results = []
    for f in files:
        contents = await f.read()
        results.append({"filename": f.filename, "size": len(contents)})
    return {"uploaded": results}
```

---

## 4. Background Tasks

Background tasks run **after** the response is sent to the client. They are ideal for non-critical operations like sending emails or writing audit logs.

```python
from fastapi import FastAPI, BackgroundTasks
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

def send_welcome_email(email: str, name: str):
    """Simulates sending an email. This runs after the response
    is already sent, so the client doesn't wait for email delivery."""
    logger.info(f"Sending welcome email to {email} for {name}")
    # In production: use an email service like SendGrid, SES, etc.

def write_audit_log(action: str, user_id: int, details: str):
    """Write to an audit log file. Background tasks are perfect for
    logging because the client shouldn't wait for I/O operations."""
    logger.info(f"AUDIT: {action} by user {user_id}: {details}")

@app.post("/users", status_code=201)
async def create_user(
    name: str,
    email: str,
    background_tasks: BackgroundTasks,
):
    """Create a user and schedule post-creation tasks.
    The response returns immediately; tasks run in the background."""
    user_id = 42  # Simulated DB insert

    # Queue multiple background tasks -- they run in order
    background_tasks.add_task(send_welcome_email, email, name)
    background_tasks.add_task(write_audit_log, "CREATE_USER", user_id, f"name={name}")

    # Client receives this immediately, doesn't wait for email/logging
    return {"id": user_id, "name": name, "email": email}
```

**Important**: Background tasks share the same process. For heavy or long-running jobs (video processing, ML inference), use a proper task queue like Celery or Arq.

---

### Theory: Async/Await: Coroutines on the Event Loop

FastAPI handlers can be `async def` or plain `def`. The choice has direct consequences for how the handler runs and what it must not do.

#### B.1 What `async def` actually buys you

An `async def` function is *not* faster than a `def` function. It does not magically parallelize work. What it does is provide *suspension points* — every `await` is a place where the function can pause and let the event loop run something else.

The win is exclusively for **I/O-bound** workloads. While your handler is awaiting `await db.execute(...)`, the event loop can serve other requests on the same worker. With sync code, the worker is blocked end-to-end. So:

```
async def: 1 worker can serve thousands of concurrent in-flight requests
def      : 1 worker serves 1 request at a time, sequentially
```

#### B.2 The cardinal rule: never block the loop

An `async def` handler that does CPU work or calls a *synchronous* blocking I/O function freezes the entire event loop. Other requests on the same worker stop being served. Symptoms: tail latency spikes, mysterious timeouts, "the server hangs under load".

The blocking culprits to watch for:

- `time.sleep(...)` instead of `await asyncio.sleep(...)`
- `requests.get(...)` instead of `await httpx.AsyncClient().get(...)`
- `psycopg2.connect(...)` instead of `await asyncpg.connect(...)` or `SQLAlchemy` async engine
- Heavy NumPy / Pandas / image processing on the main thread

For sync I/O libraries that you cannot replace, declare the handler as `def` (not `async def`). FastAPI will run it in a thread pool — the loop stays free, and one thread blocks instead of the whole worker. The cost is that thread pools are bounded; under heavy load they are still a bottleneck.

#### B.3 The mixed-route pattern

A real FastAPI app routinely has both `async def` and `def` handlers. The framework dispatches them differently:

| Handler | Runs in | Risk |
|---------|---------|------|
| `async def` | Event loop directly | Blocking the loop kills concurrency |
| `def` | `anyio.to_thread.run_sync` thread pool | Thread pool exhaustion under load |

The right rule: `async def` if every I/O dep is async; otherwise `def`. Mixing async and sync I/O in one async handler is the worst case — you pay for async without getting it.

## 5. WebSocket Support

WebSockets provide full-duplex communication over a single TCP connection. Unlike HTTP's request/response pattern, both client and server can send messages at any time.

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dataclasses import dataclass, field

app = FastAPI()

@dataclass
class ConnectionManager:
    """Manages active WebSocket connections.
    In production, you'd use Redis pub/sub for multi-server support."""
    active_connections: list[WebSocket] = field(default_factory=list)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """Send a message to all connected clients.
        Uses a copy of the list to avoid modification during iteration."""
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws/chat/{room_id}")
async def websocket_chat(websocket: WebSocket, room_id: str):
    """WebSocket endpoint for a chat room.
    The infinite loop keeps the connection alive until the client disconnects."""
    await manager.connect(websocket)
    try:
        await manager.broadcast(f"User joined room {room_id}")
        while True:
            # Wait for a message from the client
            data = await websocket.receive_text()
            # Broadcast to all connected clients
            await manager.broadcast(f"[{room_id}] {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"User left room {room_id}")
```

### Client-Side JavaScript

```javascript
// Minimal WebSocket client for testing
const ws = new WebSocket("ws://localhost:8000/ws/chat/general");

ws.onopen = () => {
    console.log("Connected");
    ws.send("Hello from client!");
};

ws.onmessage = (event) => {
    console.log("Received:", event.data);
};

ws.onclose = () => {
    console.log("Disconnected");
};
```

---

## 6. Custom Middleware

Middleware wraps every request/response cycle. It runs before the route handler and after the response is generated.

```python
import time
import uuid
from fastapi import FastAPI, Request, Response

app = FastAPI()

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign a unique ID to every request for distributed tracing.
    The ID is added to the response headers and can be used to
    correlate logs across services."""
    request_id = str(uuid.uuid4())
    # Store on request state so handlers can access it
    request.state.request_id = request_id

    response: Response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Measure and log request processing time.
    Middleware executes in reverse order of declaration,
    so this wraps the request_id_middleware above."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response
```

### Theory: Middleware: The Onion Model

Middleware sits between the ASGI server and your handlers. Each piece of middleware can transform the incoming request, transform the outgoing response, or both. They compose as concentric layers — the request traverses them inward, the response traverses them outward.

#### C.1 The request/response trace

For three middlewares A, B, C registered in that order, one request runs:

```
        IN                    OUT
client → A → B → C → handler → C → B → A → client
```

A sees the request first and the response last. C sees them in the reverse order. The order of `add_middleware()` calls *matters* because it controls the layering. The conventional order, outside-in, is: trusted host → CORS → GZip → session → auth → request id / logging → handler.

#### C.2 The contract: call_next

A Starlette/FastAPI middleware is a callable that receives the `request` and a `call_next` async function. It must `await call_next(request)` to get the response from the inner layer, then return a response — possibly modified, possibly the original:

```python
@app.middleware("http")
async def add_request_id(request, call_next):
    request.state.req_id = uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.req_id
    return response
```

Anything before `call_next` runs on the way in. Anything after runs on the way out. Returning early without calling `call_next` short-circuits — useful for auth middleware that rejects unauthenticated requests with a `401` before the handler ever runs.

#### C.3 Why middleware is *not* dependency injection

Both can attach data to the request and run before/after handlers. They differ in two crucial ways:

1. **Granularity.** Middleware applies to *every* request to the app. Depends applies to one or many specific endpoints. Choose middleware for cross-cutting concerns (CORS, logging); choose Depends for endpoint-specific logic (auth on protected routes only).
2. **Type integration.** Depends results are injected as typed parameters with full IDE support. Middleware results live on `request.state` as untyped attributes. This is why "current user" is usually a Depends, not a middleware.

A useful heuristic: if it changes HTTP headers or short-circuits responses, it is middleware; if it produces a typed value the handler needs, it is Depends.

### Execution Order

```
Request  ──▶  timing_middleware (enter)
                 ──▶  request_id_middleware (enter)
                         ──▶  route handler
                 ◀──  request_id_middleware (exit)
         ◀──  timing_middleware (exit)  ──▶  Response
```

Middleware declared **first** wraps the **outermost** layer. The handler sees the final, fully-processed request.

---

## 7. APIRouter for Modular Apps

As your application grows, putting everything in one file becomes unmaintainable. `APIRouter` lets you split endpoints into modules.

### Project Structure

```
app/
├── main.py           # Application factory, includes routers
├── routers/
│   ├── __init__.py
│   ├── users.py      # User-related endpoints
│   ├── posts.py      # Post-related endpoints
│   └── admin.py      # Admin-only endpoints
├── models/
│   ├── __init__.py
│   └── schemas.py    # Pydantic models
└── dependencies.py   # Shared dependencies
```

### Router Definition

```python
# app/routers/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from ..dependencies import get_current_active_user
from ..models.schemas import UserCreate, UserResponse

# prefix and tags apply to all routes in this router
router = APIRouter(
    prefix="/api/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=list[UserResponse])
async def list_users():
    """This becomes GET /api/users/ thanks to the router prefix."""
    return []

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """This becomes GET /api/users/{user_id}."""
    ...

@router.post("/", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    ...
```

```python
# app/routers/admin.py
from fastapi import APIRouter, Depends
from ..dependencies import get_current_admin_user

# All routes in this router require admin authentication
router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(get_current_admin_user)],  # Applied to ALL routes
)

@router.get("/stats")
async def get_stats():
    """Only accessible by admin users.
    The dependency is declared on the router, not each endpoint."""
    return {"total_users": 100, "active_today": 42}
```

### Main Application

```python
# app/main.py
from fastapi import FastAPI
from .routers import users, posts, admin

app = FastAPI(title="My Modular API", version="1.0.0")

# Include all routers. Each router's prefix and tags are preserved.
app.include_router(users.router)
app.include_router(posts.router)
app.include_router(admin.router)

@app.get("/")
async def root():
    return {"message": "API is running"}
```

---

## 8. Lifespan Events

Lifespan events handle setup and teardown logic that runs once when the application starts and stops. Common uses: database connection pools, ML model loading, cache warming.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Simulated resources
ml_models: dict = {}
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """The lifespan context manager replaces the deprecated
    @app.on_event("startup") and @app.on_event("shutdown") decorators.
    Everything before yield runs at startup; after yield runs at shutdown."""

    # --- STARTUP ---
    print("Loading ML model...")
    ml_models["sentiment"] = load_model("sentiment-v2")

    print("Creating database pool...")
    db_pool = await create_pool(
        "postgresql://localhost/mydb",
        min_size=5,
        max_size=20,
    )
    # Store on app.state so dependencies can access it
    app.state.db_pool = db_pool

    print("Application ready!")

    yield  # Application runs here, handling requests

    # --- SHUTDOWN ---
    print("Closing database pool...")
    await db_pool.close()

    print("Unloading ML models...")
    ml_models.clear()

    print("Shutdown complete.")

# Pass the lifespan to the FastAPI constructor
app = FastAPI(lifespan=lifespan)

@app.get("/predict")
async def predict(text: str):
    """Access resources initialized during startup."""
    model = ml_models.get("sentiment")
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"sentiment": model.predict(text)}
```

### Why Lifespan Instead of on_event?

The `@app.on_event()` decorators are deprecated as of FastAPI 0.93+. The `lifespan` context manager is preferred because:

1. **Guaranteed cleanup**: If startup fails, shutdown code still runs
2. **Shared state**: You can pass variables from startup to shutdown via closure
3. **Testability**: You can override the lifespan in tests
4. **Clarity**: One function shows both setup and teardown together

---

## 9. Practice Problems

### Problem 1: Dependency Injection Chain

Build a three-level dependency chain:
1. `get_settings()` -- loads configuration (database URL, API keys)
2. `get_db(settings)` -- creates a database connection using settings
3. `get_user_repo(db)` -- creates a user repository with the database connection

Create three endpoints that use `get_user_repo` to list, create, and delete users. Verify that changing `get_settings` propagates through the chain.

### Problem 2: Role-Based Access Control

Extend the JWT authentication from this lesson to support **roles** (admin, editor, viewer). Create:
- A `require_role(role: str)` dependency factory that returns a dependency
- `GET /admin/dashboard` -- requires admin role
- `POST /posts` -- requires editor or admin role
- `GET /posts` -- requires any authenticated user

The role should be stored in the JWT payload.

### Problem 3: WebSocket Chat with Rooms

Extend the WebSocket chat example to support:
- Multiple rooms (each room has its own connection list)
- A `/rooms` REST endpoint that lists active rooms and user counts
- Username identification (passed as a query parameter when connecting)
- Message history (store the last 50 messages per room in memory)

### Problem 4: Modular Application

Restructure a monolithic FastAPI app into modules:
- `routers/auth.py` -- login, register, refresh token
- `routers/items.py` -- CRUD for items
- `routers/admin.py` -- admin-only stats, user management
- `dependencies.py` -- shared dependencies (auth, DB, pagination)

Each router should have appropriate prefixes, tags, and router-level dependencies.

### Problem 5: Lifespan Resource Management

Create a FastAPI app with a lifespan that:
1. On startup: creates an in-memory SQLite database, runs migrations, seeds test data
2. On shutdown: exports statistics (total requests served, uptime) to a JSON file
3. Uses `app.state` to share the database connection with endpoints
4. Includes a middleware that increments a request counter on every request

---

## 10. References

- [FastAPI Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [FastAPI Security - OAuth2 with JWT](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [FastAPI APIRouter](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/)
- [python-jose (JWT Library)](https://github.com/mpdavis/python-jose)
- [Passlib Documentation](https://passlib.readthedocs.io/)

---

**Previous**: [FastAPI Basics](./02_FastAPI_Basics.md) | **Next**: [FastAPI Database](./04_FastAPI_Database.md)
