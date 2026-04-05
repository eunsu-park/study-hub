#!/bin/bash
# Exercises for Lesson 12: HATEOAS and Hypermedia
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: HAL Format Implementation ===
# Problem: Implement a HAL (Hypertext Application Language) response for
# an order resource with embedded sub-resources and link relations.
exercise_1() {
    echo "=== Exercise 1: HAL Format Implementation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, Any

app = FastAPI()


def hal_response(
    data: dict,
    links: dict[str, Any],
    embedded: dict[str, Any] = None,
) -> dict:
    """Build a HAL+JSON response.

    HAL (Hypertext Application Language) format:
    - _links: navigation links (self, related resources, actions)
    - _embedded: sub-resources included in the response
    - Everything else: resource properties

    Content-Type: application/hal+json
    """
    response = {**data, "_links": links}
    if embedded:
        response["_embedded"] = embedded
    return response


@app.get("/api/v1/orders/{order_id}")
def get_order(request: Request, order_id: str):
    """Return an order in HAL format with embedded items and customer."""
    base = str(request.base_url).rstrip("/")

    return hal_response(
        data={
            "id": order_id,
            "status": "shipped",
            "total": 149.97,
            "currency": "USD",
            "created_at": "2025-06-15T10:00:00Z",
        },
        links={
            "self": {"href": f"{base}/api/v1/orders/{order_id}"},
            "customer": {"href": f"{base}/api/v1/customers/cust_1"},
            "items": {"href": f"{base}/api/v1/orders/{order_id}/items"},
            # Templated link — client fills in the tracking number
            "tracking": {
                "href": f"{base}/api/v1/shipments/{{trackingNumber}}",
                "templated": True,
            },
            # Actions available for this order (based on state)
            "cancel": {"href": f"{base}/api/v1/orders/{order_id}/cancel", "method": "POST"},
            "return": {"href": f"{base}/api/v1/orders/{order_id}/return", "method": "POST"},
        },
        embedded={
            # Sub-resources included to avoid extra round-trips
            "items": [
                {
                    "id": "item_1",
                    "product": "Wireless Mouse",
                    "quantity": 2,
                    "price": 29.99,
                    "_links": {
                        "self": {"href": f"{base}/api/v1/products/prod_1"},
                    },
                },
                {
                    "id": "item_2",
                    "product": "USB-C Hub",
                    "quantity": 1,
                    "price": 89.99,
                    "_links": {
                        "self": {"href": f"{base}/api/v1/products/prod_2"},
                    },
                },
            ],
            "customer": {
                "id": "cust_1",
                "name": "Alice Smith",
                "_links": {
                    "self": {"href": f"{base}/api/v1/customers/cust_1"},
                },
            },
        },
    )
SOLUTION
}

# === Exercise 2: JSON:API Format ===
# Problem: Implement a JSON:API compliant response with resource linkage,
# included resources, and sparse fieldsets.
exercise_2() {
    echo "=== Exercise 2: JSON:API Format ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()


@app.get("/api/v1/articles/{article_id}")
def get_article(
    article_id: str,
    include: Optional[str] = Query(None, description="Comma-separated related resources"),
    fields_article: Optional[str] = Query(None, alias="fields[article]"),
    fields_author: Optional[str] = Query(None, alias="fields[author]"),
):
    """Return an article in JSON:API format.

    JSON:API spec: https://jsonapi.org/
    Key features:
    - data: primary resource with type + id + attributes + relationships
    - included: related resources (compound documents)
    - links: pagination and self-referencing
    - Sparse fieldsets via fields[type]=field1,field2
    """
    article = {
        "type": "article",
        "id": article_id,
        "attributes": {
            "title": "Understanding HATEOAS",
            "body": "Hypermedia as the Engine of Application State...",
            "created_at": "2025-06-15T10:00:00Z",
            "view_count": 1234,
        },
        "relationships": {
            "author": {
                "data": {"type": "person", "id": "author_1"},
                "links": {
                    "related": f"/api/v1/people/author_1",
                },
            },
            "comments": {
                "data": [
                    {"type": "comment", "id": "comment_1"},
                    {"type": "comment", "id": "comment_2"},
                ],
                "links": {
                    "related": f"/api/v1/articles/{article_id}/comments",
                },
            },
        },
        "links": {
            "self": f"/api/v1/articles/{article_id}",
        },
    }

    # Apply sparse fieldsets
    if fields_article:
        allowed = set(fields_article.split(","))
        article["attributes"] = {
            k: v for k, v in article["attributes"].items() if k in allowed
        }

    response = {
        "data": article,
        "links": {"self": f"/api/v1/articles/{article_id}"},
    }

    # Include related resources if requested
    if include:
        included_types = set(include.split(","))
        response["included"] = []

        if "author" in included_types:
            author = {
                "type": "person",
                "id": "author_1",
                "attributes": {"name": "Alice", "bio": "API design enthusiast"},
                "links": {"self": "/api/v1/people/author_1"},
            }
            if fields_author:
                allowed = set(fields_author.split(","))
                author["attributes"] = {
                    k: v for k, v in author["attributes"].items() if k in allowed
                }
            response["included"].append(author)

        if "comments" in included_types:
            response["included"].extend([
                {
                    "type": "comment",
                    "id": "comment_1",
                    "attributes": {"body": "Great article!", "author": "Bob"},
                },
                {
                    "type": "comment",
                    "id": "comment_2",
                    "attributes": {"body": "Very helpful.", "author": "Charlie"},
                },
            ])

    return response

# Usage:
# GET /api/v1/articles/1
# GET /api/v1/articles/1?include=author,comments
# GET /api/v1/articles/1?include=author&fields[article]=title&fields[author]=name
SOLUTION
}

# === Exercise 3: Link Relations and State Machines ===
# Problem: Build a state machine-driven API where available actions are
# communicated through hypermedia links.
exercise_3() {
    echo "=== Exercise 3: Link Relations and State Machines ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()

# State machine definition
STATES = {
    "draft": {
        "transitions": {
            "submit":  {"target": "pending_review", "method": "POST"},
            "delete":  {"target": None, "method": "DELETE"},
        },
        "description": "Initial state. Can be edited and submitted.",
    },
    "pending_review": {
        "transitions": {
            "approve": {"target": "approved", "method": "POST"},
            "reject":  {"target": "draft", "method": "POST"},
        },
        "description": "Waiting for reviewer approval.",
    },
    "approved": {
        "transitions": {
            "publish": {"target": "published", "method": "POST"},
            "archive": {"target": "archived", "method": "POST"},
        },
        "description": "Approved and ready for publication.",
    },
    "published": {
        "transitions": {
            "unpublish": {"target": "draft", "method": "POST"},
            "archive":   {"target": "archived", "method": "POST"},
        },
        "description": "Live and visible to users.",
    },
    "archived": {
        "transitions": {
            "restore": {"target": "draft", "method": "POST"},
        },
        "description": "Archived. Can be restored to draft.",
    },
}

documents = {
    "doc_1": {"id": "doc_1", "title": "API Guide", "status": "draft"},
}


def build_state_links(request: Request, doc_id: str, current_status: str) -> dict:
    """Generate links based on the document's current state."""
    base = str(request.base_url).rstrip("/")
    links = {
        "self": {"href": f"{base}/api/v1/documents/{doc_id}", "method": "GET"},
    }

    state = STATES.get(current_status, {})
    for action, config in state.get("transitions", {}).items():
        links[action] = {
            "href": f"{base}/api/v1/documents/{doc_id}/{action}",
            "method": config["method"],
            "description": f"Transition to '{config['target']}'" if config["target"] else "Remove document",
        }

    return links


@app.get("/api/v1/documents/{doc_id}")
def get_document(request: Request, doc_id: str):
    doc = documents.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    state_info = STATES[doc["status"]]
    return {
        **doc,
        "state_description": state_info["description"],
        "_links": build_state_links(request, doc_id, doc["status"]),
    }


@app.post("/api/v1/documents/{doc_id}/{action}")
def transition_document(request: Request, doc_id: str, action: str):
    """Perform a state transition on the document."""
    doc = documents.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    state = STATES.get(doc["status"], {})
    transition = state.get("transitions", {}).get(action)

    if not transition:
        allowed = list(state.get("transitions", {}).keys())
        raise HTTPException(
            status_code=409,
            detail=f"Action '{action}' not allowed in state '{doc['status']}'. "
                   f"Allowed: {allowed}",
        )

    if transition["target"] is None:
        del documents[doc_id]
        return {"message": "Document deleted"}

    doc["status"] = transition["target"]
    return {
        **doc,
        "_links": build_state_links(request, doc_id, doc["status"]),
    }

# The client never needs to hardcode state rules —
# it just follows the links provided in each response.
SOLUTION
}

# === Exercise 4: Hypermedia Collection Navigation ===
# Problem: Build a collection endpoint that uses hypermedia links for
# pagination, filtering, and individual resource access.
exercise_4() {
    echo "=== Exercise 4: Hypermedia Collection Navigation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Query, Request
from typing import Optional

app = FastAPI()

products = [{"id": str(i), "name": f"Product {i}", "price": i * 10.0}
            for i in range(1, 51)]


@app.get("/api/v1/products")
def list_products(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50),
    min_price: Optional[float] = Query(None),
):
    """Collection with full hypermedia navigation."""
    base = str(request.base_url).rstrip("/")
    path = "/api/v1/products"

    # Filter
    items = products
    if min_price is not None:
        items = [p for p in items if p["price"] >= min_price]

    total = len(items)
    total_pages = (total + per_page - 1) // per_page
    offset = (page - 1) * per_page
    page_items = items[offset:offset + per_page]

    # Build query string for filters (preserved across pages)
    filter_qs = f"&min_price={min_price}" if min_price else ""

    # Collection links
    links = {
        "self": {"href": f"{base}{path}?page={page}&per_page={per_page}{filter_qs}"},
        "first": {"href": f"{base}{path}?page=1&per_page={per_page}{filter_qs}"},
        "last": {"href": f"{base}{path}?page={total_pages}&per_page={per_page}{filter_qs}"},
        "create": {"href": f"{base}{path}", "method": "POST"},
    }
    if page < total_pages:
        links["next"] = {"href": f"{base}{path}?page={page+1}&per_page={per_page}{filter_qs}"}
    if page > 1:
        links["prev"] = {"href": f"{base}{path}?page={page-1}&per_page={per_page}{filter_qs}"}

    # Each item gets its own links
    data = []
    for item in page_items:
        data.append({
            **item,
            "_links": {
                "self": {"href": f"{base}{path}/{item['id']}"},
                "update": {"href": f"{base}{path}/{item['id']}", "method": "PATCH"},
                "delete": {"href": f"{base}{path}/{item['id']}", "method": "DELETE"},
            },
        })

    return {
        "data": data,
        "meta": {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
        },
        "_links": links,
    }
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 12: HATEOAS and Hypermedia"
echo "=========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
