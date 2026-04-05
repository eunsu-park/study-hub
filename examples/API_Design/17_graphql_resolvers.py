#!/usr/bin/env python3
"""Example: GraphQL Resolvers

Demonstrates resolver patterns and the N+1 problem with solutions:
- Basic field resolvers
- The N+1 query problem explained
- DataLoader pattern for batched data fetching
- Resolver context for dependency injection
- Error handling in resolvers

Related lesson: 17_GraphQL_Resolvers.md

Run:
    pip install strawberry-graphql[fastapi] "fastapi[standard]"
    uvicorn 17_graphql_resolvers:app --reload --port 8000
"""

import asyncio
import logging
from typing import Optional

import strawberry
from fastapi import FastAPI
from strawberry.dataloader import DataLoader
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resolvers")

# =============================================================================
# SIMULATED DATABASE — Tracks query count to demonstrate N+1
# =============================================================================

_query_count = 0


def _inc_query():
    global _query_count
    _query_count += 1
    logger.info(f"  DB query #{_query_count}")


DEPARTMENTS = {
    "d1": {"id": "d1", "name": "Engineering"},
    "d2": {"id": "d2", "name": "Marketing"},
}

EMPLOYEES = {
    "e1": {"id": "e1", "name": "Alice", "dept_id": "d1"},
    "e2": {"id": "e2", "name": "Bob", "dept_id": "d1"},
    "e3": {"id": "e3", "name": "Carol", "dept_id": "d2"},
    "e4": {"id": "e4", "name": "Dave", "dept_id": "d2"},
    "e5": {"id": "e5", "name": "Eve", "dept_id": "d1"},
}

PROJECTS = {
    "p1": {"id": "p1", "title": "API Redesign", "lead_id": "e1"},
    "p2": {"id": "p2", "title": "Mobile App", "lead_id": "e3"},
}


# =============================================================================
# NAIVE RESOLVERS — Demonstrate the N+1 problem
# =============================================================================
# When listing N employees, each calls get_department individually,
# resulting in 1 (list) + N (department) queries. This is O(N).

def get_department_naive(dept_id: str) -> Optional[dict]:
    """Fetch a single department — called once per employee (N+1 problem)."""
    _inc_query()
    return DEPARTMENTS.get(dept_id)


# =============================================================================
# DATALOADER — Batched fetching to solve N+1
# =============================================================================
# DataLoader collects all IDs requested in one tick of the event loop
# and fetches them in a single batch query. N+1 becomes 1+1.

async def load_departments(keys: list[str]) -> list[Optional[dict]]:
    """Batch loader: fetch multiple departments in one call."""
    _inc_query()
    logger.info(f"  Batch loading departments: {keys}")
    return [DEPARTMENTS.get(k) for k in keys]


async def load_employees_by_dept(keys: list[str]) -> list[list[dict]]:
    """Batch loader: fetch employees grouped by department ID."""
    _inc_query()
    logger.info(f"  Batch loading employees for depts: {keys}")
    result = []
    for dept_id in keys:
        emps = [e for e in EMPLOYEES.values() if e["dept_id"] == dept_id]
        result.append(emps)
    return result


# =============================================================================
# TYPES WITH DATALOADER-BACKED RESOLVERS
# =============================================================================

@strawberry.type
class Department:
    id: str
    name: str

    @strawberry.field
    async def employees(self, info: Info) -> list["Employee"]:
        """Uses DataLoader to batch-fetch employees for this department."""
        loader = info.context["emp_by_dept_loader"]
        emp_dicts = await loader.load(self.id)
        return [Employee(id=e["id"], name=e["name"], dept_id=e["dept_id"])
                for e in emp_dicts]


@strawberry.type
class Employee:
    id: str
    name: str
    dept_id: str

    @strawberry.field
    async def department(self, info: Info) -> Optional[Department]:
        """Uses DataLoader — even if 100 employees are listed, the department
        table is queried only ONCE (all dept_ids batched together)."""
        loader = info.context["dept_loader"]
        dept = await loader.load(self.dept_id)
        return Department(**dept) if dept else None


@strawberry.type
class Project:
    id: str
    title: str
    lead_id: str

    @strawberry.field
    async def lead(self, info: Info) -> Optional[Employee]:
        """Resolve the project lead via the employee store."""
        data = EMPLOYEES.get(self.lead_id)
        if not data:
            return None
        return Employee(id=data["id"], name=data["name"], dept_id=data["dept_id"])


# =============================================================================
# QUERY ROOT
# =============================================================================

@strawberry.type
class Query:
    @strawberry.field
    def employees(self) -> list[Employee]:
        _inc_query()
        return [Employee(id=e["id"], name=e["name"], dept_id=e["dept_id"])
                for e in EMPLOYEES.values()]

    @strawberry.field
    def departments(self) -> list[Department]:
        _inc_query()
        return [Department(**d) for d in DEPARTMENTS.values()]

    @strawberry.field
    def project(self, id: str) -> Optional[Project]:
        data = PROJECTS.get(id)
        return Project(**data) if data else None

    @strawberry.field(description="Return the DB query count (for demonstrating N+1).")
    def query_count(self) -> int:
        return _query_count

    @strawberry.field(description="Reset the query counter.")
    def reset_counter(self) -> int:
        global _query_count
        _query_count = 0
        return 0


# =============================================================================
# CONTEXT — Inject DataLoaders per request
# =============================================================================
# DataLoaders MUST be created per-request to avoid stale caches across requests.

async def get_context():
    return {
        "dept_loader": DataLoader(load_fn=load_departments),
        "emp_by_dept_loader": DataLoader(load_fn=load_employees_by_dept),
    }


# =============================================================================
# SCHEMA & APP
# =============================================================================

schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema, context_getter=get_context)

app = FastAPI(title="GraphQL Resolvers Demo")
app.include_router(graphql_app, prefix="/graphql")

# =============================================================================
# EXAMPLE QUERIES
# =============================================================================

EXAMPLE_QUERIES = """
# === Demonstrate N+1: list employees with their departments ===
# Check the server logs — dept_loader batches all dept_ids into ONE query.
query {
  resetCounter
  employees {
    name
    department { name }
  }
  queryCount
}

# === Nested: departments -> employees -> department (still batched) ===
query {
  departments {
    name
    employees { name }
  }
}
"""

if __name__ == "__main__":
    import uvicorn
    print(EXAMPLE_QUERIES)
    uvicorn.run("17_graphql_resolvers:app", host="127.0.0.1", port=8000, reload=True)
