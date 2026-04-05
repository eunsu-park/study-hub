#!/usr/bin/env python3
"""Example: API Documentation

Demonstrates automatic OpenAPI spec generation and Swagger UI with FastAPI:
- Rich schema descriptions with Field and examples
- Grouped endpoints via tags and tag metadata
- Custom OpenAPI extensions
- Programmatic spec export for CI/doc pipelines

Related lesson: 10_API_Documentation.md

Run:
    pip install "fastapi[standard]"
    uvicorn 10_api_documentation:app --reload --port 8000

    # Interactive docs:  http://localhost:8000/docs  (Swagger UI)
    # Alternative docs:  http://localhost:8000/redoc (ReDoc)
    # Raw OpenAPI JSON:  http://localhost:8000/openapi.json
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from fastapi import FastAPI, Path, Query, status
from pydantic import BaseModel, Field

# =============================================================================
# TAG METADATA — Organizes endpoints in Swagger UI
# =============================================================================
# Tags group related endpoints and can include external documentation links.

tags_metadata = [
    {
        "name": "Articles",
        "description": "CRUD operations on articles. Supports filtering and pagination.",
        "externalDocs": {"description": "Content guidelines", "url": "https://example.com/docs"},
    },
    {
        "name": "Health",
        "description": "Service health and readiness probes.",
    },
]


# =============================================================================
# APPLICATION — Comprehensive metadata for the generated spec
# =============================================================================

app = FastAPI(
    title="Blog API",
    version="1.2.0",
    summary="A demonstration API with rich OpenAPI documentation.",
    description=(
        "## Overview\n\n"
        "This API manages blog articles. It showcases how to produce\n"
        "well-documented OpenAPI specs directly from code.\n\n"
        "### Authentication\n"
        "Pass `Authorization: Bearer <token>` for write operations.\n"
    ),
    contact={"name": "API Support", "email": "api@example.com"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    openapi_tags=tags_metadata,
)


# =============================================================================
# ENUMS — Rendered as dropdown selects in Swagger UI
# =============================================================================

class ArticleStatus(str, Enum):
    draft = "draft"
    published = "published"
    archived = "archived"


# =============================================================================
# SCHEMAS — Detailed descriptions and examples drive doc quality
# =============================================================================

class ArticleCreate(BaseModel):
    """Schema for creating an article. All fields are documented with
    descriptions and realistic examples so consumers understand the
    expected format without reading external docs."""

    title: str = Field(
        ..., min_length=1, max_length=200,
        description="Article headline displayed to readers.",
        examples=["Getting Started with FastAPI"],
    )
    body: str = Field(
        ..., min_length=10,
        description="Markdown-formatted article body.",
        examples=["# Introduction\n\nFastAPI makes it easy to ..."],
    )
    status: ArticleStatus = Field(
        ArticleStatus.draft,
        description="Publication status. New articles default to `draft`.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Freeform tags for categorization.",
        examples=[["python", "api"]],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Getting Started with FastAPI",
                    "body": "# Introduction\n\nFastAPI is a modern web framework...",
                    "status": "draft",
                    "tags": ["python", "api"],
                }
            ]
        }
    }


class ArticleResponse(BaseModel):
    id: str = Field(..., description="Server-generated UUID.")
    title: str
    body: str
    status: ArticleStatus
    tags: list[str]
    created_at: datetime
    word_count: int = Field(..., description="Computed word count of the body.")


class ArticleList(BaseModel):
    data: list[ArticleResponse]
    total: int = Field(..., description="Total matching articles (before pagination).")


# =============================================================================
# IN-MEMORY STORE
# =============================================================================

from uuid import uuid4

_db: dict[str, dict] = {}


# =============================================================================
# ROUTES — Each operation has summary, description, and response docs
# =============================================================================

@app.post(
    "/api/v1/articles",
    response_model=ArticleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an article",
    description="Creates a new article. Returns the full resource with generated fields.",
    responses={
        201: {"description": "Article created successfully."},
        422: {"description": "Validation error — check field constraints."},
    },
    tags=["Articles"],
)
def create_article(body: ArticleCreate):
    aid = str(uuid4())
    article = {
        "id": aid, "title": body.title, "body": body.body,
        "status": body.status.value, "tags": body.tags,
        "created_at": datetime.now(timezone.utc),
        "word_count": len(body.body.split()),
    }
    _db[aid] = article
    return article


@app.get(
    "/api/v1/articles",
    response_model=ArticleList,
    summary="List articles",
    description="Retrieve a paginated, optionally filtered list of articles.",
    tags=["Articles"],
)
def list_articles(
    status_filter: Optional[ArticleStatus] = Query(
        None, alias="status", description="Filter by publication status."
    ),
    limit: int = Query(20, ge=1, le=100, description="Page size."),
    offset: int = Query(0, ge=0, description="Number of items to skip."),
):
    items = list(_db.values())
    if status_filter:
        items = [a for a in items if a["status"] == status_filter.value]
    return {"data": items[offset:offset + limit], "total": len(items)}


@app.get(
    "/api/v1/articles/{article_id}",
    response_model=ArticleResponse,
    summary="Get an article by ID",
    tags=["Articles"],
    responses={404: {"description": "Article not found."}},
)
def get_article(
    article_id: str = Path(..., description="UUID of the article to retrieve."),
):
    from fastapi import HTTPException
    article = _db.get(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


# =============================================================================
# HEALTH ENDPOINT — Essential for Kubernetes probes
# =============================================================================

@app.get("/healthz", tags=["Health"], summary="Liveness probe")
def healthz():
    return {"status": "ok"}


# =============================================================================
# PROGRAMMATIC SPEC EXPORT — Useful in CI to detect breaking changes
# =============================================================================

def export_openapi_spec(path: str = "openapi.json"):
    """Write the OpenAPI JSON to a file for version-controlled docs."""
    import json
    spec = app.openapi()
    with open(path, "w") as f:
        json.dump(spec, f, indent=2, default=str)
    print(f"OpenAPI spec written to {path}")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys
    if "--export" in sys.argv:
        export_openapi_spec()
    else:
        import uvicorn
        uvicorn.run("10_api_documentation:app", host="127.0.0.1", port=8000, reload=True)
