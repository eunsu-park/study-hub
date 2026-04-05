#!/bin/bash
# Exercises for Lesson 11: Validation and Serialization
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: Pydantic Model Validation ===
# Problem: Build a comprehensive Pydantic model with custom validators,
# field constraints, and nested models for a user registration endpoint.
exercise_1() {
    echo "=== Exercise 1: Pydantic Model Validation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
from datetime import date
import re


class Address(BaseModel):
    """Nested model for address validation."""
    street: str = Field(..., min_length=1, max_length=200)
    city: str = Field(..., min_length=1, max_length=100)
    state: str = Field(..., min_length=2, max_length=2, description="US state code")
    zip_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$")
    country: str = Field("US", max_length=2)

    @field_validator("state")
    @classmethod
    def validate_state_code(cls, v: str) -> str:
        """Normalize to uppercase and validate."""
        v = v.upper()
        valid_states = {"CA", "NY", "TX", "FL", "WA", "OR"}  # Subset for demo
        if v not in valid_states:
            raise ValueError(f"Invalid state code: {v}")
        return v


class UserRegistration(BaseModel):
    """Registration model with comprehensive validation."""
    username: str = Field(
        ..., min_length=3, max_length=30,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$",
        description="Must start with a letter, alphanumeric + _ -",
    )
    email: str = Field(
        ..., max_length=254,
        description="Valid email address",
    )
    password: str = Field(..., min_length=8, max_length=128)
    password_confirm: str = Field(..., min_length=8)
    date_of_birth: date = Field(..., description="YYYY-MM-DD format")
    phone: Optional[str] = Field(None, pattern=r"^\+?1?\d{10,15}$")
    address: Optional[Address] = None
    terms_accepted: bool = Field(..., description="Must be true")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation and normalization."""
        v = v.lower().strip()
        if not re.match(r"^[\w.-]+@[\w.-]+\.\w{2,}$", v):
            raise ValueError("Invalid email format")
        return v

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Enforce password complexity rules."""
        if not any(c.isupper() for c in v):
            raise ValueError("Must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Must contain at least one digit")
        if not any(c in "!@#$%^&*()-_=+[]{}|;:',.<>?/" for c in v):
            raise ValueError("Must contain at least one special character")
        return v

    @field_validator("date_of_birth")
    @classmethod
    def validate_age(cls, v: date) -> date:
        """User must be at least 18 years old."""
        today = date.today()
        age = today.year - v.year - ((today.month, today.day) < (v.month, v.day))
        if age < 18:
            raise ValueError("Must be at least 18 years old")
        return v

    @model_validator(mode="after")
    def passwords_match(self):
        """Cross-field validation: passwords must match."""
        if self.password != self.password_confirm:
            raise ValueError("Passwords do not match")
        return self

    @model_validator(mode="after")
    def terms_must_be_accepted(self):
        if not self.terms_accepted:
            raise ValueError("Terms of service must be accepted")
        return self
SOLUTION
}

# === Exercise 2: Response Serialization Patterns ===
# Problem: Implement different response serialization strategies:
# separate input/output models, computed fields, and field exclusion.
exercise_2() {
    echo "=== Exercise 2: Response Serialization Patterns ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from pydantic import BaseModel, Field, computed_field
from typing import Optional
from datetime import datetime, timezone


# Pattern 1: Separate Input and Output Models
# Input models accept what the client sends.
# Output models return what the client should see.

class UserCreate(BaseModel):
    """Input: what the client sends."""
    username: str
    email: str
    password: str          # Accepted in input
    role: Optional[str] = None  # Ignored (server decides)


class UserResponse(BaseModel):
    """Output: what the client receives."""
    id: str
    username: str
    email: str
    # password is NOT here — never return passwords
    role: str
    created_at: datetime

    @computed_field
    @property
    def display_name(self) -> str:
        """Computed field — calculated on serialization, not stored."""
        return f"@{self.username}"


class UserDetailResponse(UserResponse):
    """Extended output with additional fields for authorized users."""
    phone: Optional[str] = None
    last_login: Optional[datetime] = None
    is_verified: bool = False


# Pattern 2: Field Exclusion with model_dump
class InternalUser(BaseModel):
    id: str
    username: str
    email: str
    password_hash: str        # Internal only
    api_key: str              # Internal only
    role: str
    created_at: datetime


def to_public_response(user: InternalUser) -> dict:
    """Convert internal model to public response, excluding sensitive fields."""
    return user.model_dump(
        exclude={"password_hash", "api_key"},
        mode="json",  # Ensures datetime is serialized to string
    )


# Pattern 3: Conditional serialization
class ProductResponse(BaseModel):
    id: str
    name: str
    price: float
    cost: Optional[float] = None      # Only for admin
    margin: Optional[float] = None    # Only for admin

    def for_role(self, role: str) -> dict:
        """Return different fields based on user role."""
        if role == "admin":
            return self.model_dump()
        else:
            return self.model_dump(exclude={"cost", "margin"})


# Usage in FastAPI:
from fastapi import FastAPI, Depends

app = FastAPI()

@app.post("/api/v1/users", response_model=UserResponse, status_code=201)
def create_user(body: UserCreate):
    """Input uses UserCreate (accepts password).
    Output uses UserResponse (excludes password)."""
    user = UserResponse(
        id="user_1",
        username=body.username,
        email=body.email,
        role="user",
        created_at=datetime.now(timezone.utc),
    )
    return user
SOLUTION
}

# === Exercise 3: Custom Serializers ===
# Problem: Implement custom JSON serialization for special types
# (datetime, Decimal, Enum, UUID).
exercise_3() {
    echo "=== Exercise 3: Custom Serializers ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from pydantic import BaseModel, field_serializer, ConfigDict
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


class MoneyResponse(BaseModel):
    """Model with custom serialization for special types."""
    model_config = ConfigDict(
        json_encoders={
            Decimal: lambda v: str(v),  # Global: Decimal → string
        }
    )

    id: UUID
    amount: Decimal
    currency: Currency
    created_at: datetime

    @field_serializer("amount")
    def serialize_amount(self, v: Decimal) -> str:
        """Serialize Decimal as string to preserve precision.

        JSON has no Decimal type. Using float loses precision:
        float(Decimal("19.99")) might become 19.990000000000002

        Always serialize money as strings in API responses.
        """
        return f"{v:.2f}"

    @field_serializer("created_at")
    def serialize_datetime(self, v: datetime) -> str:
        """Serialize datetime as ISO 8601 with Z suffix (not +00:00)."""
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.strftime("%Y-%m-%dT%H:%M:%SZ")

    @field_serializer("id")
    def serialize_uuid(self, v: UUID) -> str:
        """Serialize UUID as string (without dashes is also common)."""
        return str(v)


# Usage:
money = MoneyResponse(
    id=uuid4(),
    amount=Decimal("19.99"),
    currency=Currency.USD,
    created_at=datetime.now(timezone.utc),
)

print(money.model_dump_json(indent=2))
# {
#   "id": "a1b2c3d4-e5f6-...",
#   "amount": "19.99",       ← String, not 19.990000000000002
#   "currency": "USD",        ← String value of enum
#   "created_at": "2025-06-15T10:30:00Z"  ← Clean ISO format
# }
SOLUTION
}

# === Exercise 4: Request/Response Middleware ===
# Problem: Build middleware that transforms request/response bodies
# (e.g., snake_case to camelCase conversion for JavaScript clients).
exercise_4() {
    echo "=== Exercise 4: Request/Response Middleware ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import json
import re
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def convert_keys(obj, converter):
    """Recursively convert all dictionary keys."""
    if isinstance(obj, dict):
        return {converter(k): convert_keys(v, converter) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys(item, converter) for item in obj]
    return obj


app = FastAPI()


@app.middleware("http")
async def camel_case_middleware(request: Request, call_next):
    """Convert camelCase request → snake_case for Python.
    Convert snake_case response → camelCase for JavaScript.

    This lets the API use Python conventions internally while
    JavaScript clients use their preferred camelCase convention.
    """
    # Convert incoming camelCase body to snake_case
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        if body:
            try:
                data = json.loads(body)
                converted = convert_keys(data, camel_to_snake)
                # Reconstruct request with converted body
                request._body = json.dumps(converted).encode()
            except json.JSONDecodeError:
                pass

    response = await call_next(request)

    # Convert outgoing snake_case response to camelCase
    if response.headers.get("content-type", "").startswith("application/json"):
        body = b""
        async for chunk in response.body_iterator:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        try:
            data = json.loads(body)
            converted = convert_keys(data, snake_to_camel)
            return JSONResponse(
                content=converted,
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        except json.JSONDecodeError:
            pass

    return response


from pydantic import BaseModel

class UserCreate(BaseModel):
    first_name: str      # Python: snake_case
    last_name: str
    email_address: str

@app.post("/api/v1/users")
def create_user(body: UserCreate):
    return {"first_name": body.first_name, "last_name": body.last_name}

# Client sends:  {"firstName": "Alice", "lastName": "Smith", "emailAddress": "..."}
# Server sees:   {"first_name": "Alice", "last_name": "Smith", "email_address": "..."}
# Client gets:   {"firstName": "Alice", "lastName": "Smith"}
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 11: Validation and Serialization"
echo "================================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
