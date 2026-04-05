#!/usr/bin/env python3
"""Example: API Versioning

Demonstrates multiple API versioning strategies with FastAPI:
- URL path versioning (/api/v1/, /api/v2/)
- Header versioning (Accept-Version header)
- Query parameter versioning (?version=2)
- Sunset header for deprecated versions

Related lesson: 09_API_Versioning.md

Run:
    pip install "fastapi[standard]"
    uvicorn 09_api_versioning:app --reload --port 8000
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# =============================================================================
# SCHEMAS — Version-specific response models
# =============================================================================
# When a new version introduces breaking changes, create separate models.
# This makes the contract explicit and prevents accidental field exposure.

class ProductV1(BaseModel):
    """V1 response — flat structure, price as float."""
    id: str
    name: str
    price: float
    category: str


class PriceDetail(BaseModel):
    amount: float
    currency: str = "USD"
    discount_pct: float = 0.0


class ProductV2(BaseModel):
    """V2 response — nested price object, added 'tags' field."""
    id: str
    name: str
    price: PriceDetail
    category: str
    tags: list[str] = []


# =============================================================================
# IN-MEMORY DATA
# =============================================================================

PRODUCTS = {
    "1": {"id": "1", "name": "Keyboard", "price": 79.99, "category": "electronics",
           "tags": ["peripherals", "input"], "currency": "USD", "discount_pct": 10.0},
    "2": {"id": "2", "name": "Notebook", "price": 12.50, "category": "stationery",
           "tags": ["writing"], "currency": "USD", "discount_pct": 0.0},
}


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(title="API Versioning Demo", version="2.0.0")


# =============================================================================
# STRATEGY 1: URL Path Versioning  (most common, most explicit)
# =============================================================================
# Pros: easy to route, cache-friendly, self-documenting URLs
# Cons: URL proliferation, harder to share links across versions

@app.get("/api/v1/products/{product_id}", response_model=ProductV1, tags=["URL Versioning"])
def get_product_v1(product_id: str):
    """V1 endpoint — returns flat price as a float."""
    p = PRODUCTS.get(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    return ProductV1(id=p["id"], name=p["name"], price=p["price"], category=p["category"])


@app.get("/api/v2/products/{product_id}", response_model=ProductV2, tags=["URL Versioning"])
def get_product_v2(product_id: str):
    """V2 endpoint — returns nested price object with currency and discount."""
    p = PRODUCTS.get(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    return ProductV2(
        id=p["id"], name=p["name"], category=p["category"], tags=p["tags"],
        price=PriceDetail(amount=p["price"], currency=p["currency"], discount_pct=p["discount_pct"]),
    )


# =============================================================================
# STRATEGY 2: Header Versioning  (Accept-Version or custom header)
# =============================================================================
# Pros: clean URLs, content negotiation style
# Cons: harder to test in browser, invisible in access logs

@app.get("/api/products/{product_id}", tags=["Header Versioning"])
def get_product_header(product_id: str, accept_version: Optional[str] = Header(None)):
    """Route based on the Accept-Version header (defaults to latest)."""
    version = accept_version or "2"
    p = PRODUCTS.get(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")

    if version == "1":
        return ProductV1(id=p["id"], name=p["name"], price=p["price"], category=p["category"])
    return ProductV2(
        id=p["id"], name=p["name"], category=p["category"], tags=p["tags"],
        price=PriceDetail(amount=p["price"], currency=p["currency"], discount_pct=p["discount_pct"]),
    )


# =============================================================================
# STRATEGY 3: Query Parameter Versioning
# =============================================================================
# Pros: easy to switch in browser, no header setup
# Cons: clutters query string, less RESTful

@app.get("/api/qp/products/{product_id}", tags=["Query Param Versioning"])
def get_product_query(product_id: str, version: int = Query(2, ge=1, le=2)):
    """Route based on ?version= query parameter."""
    p = PRODUCTS.get(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")

    if version == 1:
        return ProductV1(id=p["id"], name=p["name"], price=p["price"], category=p["category"])
    return ProductV2(
        id=p["id"], name=p["name"], category=p["category"], tags=p["tags"],
        price=PriceDetail(amount=p["price"], currency=p["currency"], discount_pct=p["discount_pct"]),
    )


# =============================================================================
# SUNSET HEADER — Deprecation signaling (RFC 8594)
# =============================================================================
# When retiring a version, add a Sunset header with the retirement date.
# Clients can parse this to plan their migration.

@app.middleware("http")
async def add_sunset_header(request: Request, call_next):
    response = await call_next(request)
    if "/api/v1/" in str(request.url):
        response.headers["Sunset"] = "Sat, 01 Jan 2027 00:00:00 GMT"
        response.headers["Deprecation"] = "true"
        response.headers["Link"] = '</api/v2/>; rel="successor-version"'
    return response


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("09_api_versioning:app", host="127.0.0.1", port=8000, reload=True)
