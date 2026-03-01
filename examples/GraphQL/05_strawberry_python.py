"""
GraphQL Code-First with Python — Strawberry + FastAPI
Demonstrates: type definitions, resolvers, DataLoader, mutations.

Run: pip install strawberry-graphql[fastapi] fastapi uvicorn
     uvicorn 05_strawberry_python:app --reload
Open: http://127.0.0.1:8000/graphql
"""

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.dataloader import DataLoader
from fastapi import FastAPI
from typing import Optional
from datetime import datetime


# --- Data Store ---

users_db = {
    "1": {"id": "1", "name": "Alice", "email": "alice@example.com"},
    "2": {"id": "2", "name": "Bob", "email": "bob@example.com"},
}

posts_db = [
    {"id": "1", "title": "GraphQL with Python", "content": "Strawberry is great!", "author_id": "1", "published": True},
    {"id": "2", "title": "Type Safety", "content": "Python types → GraphQL types", "author_id": "1", "published": False},
    {"id": "3", "title": "FastAPI Integration", "content": "Easy setup...", "author_id": "2", "published": True},
]

next_post_id = 4


# --- Strawberry Types ---

@strawberry.type
class User:
    id: strawberry.ID
    name: str
    email: str

    @strawberry.field
    async def posts(self, info: strawberry.types.Info) -> list["Post"]:
        """Resolve posts for this user using DataLoader."""
        return await info.context["posts_by_author_loader"].load(self.id)


@strawberry.enum
class PostStatus:
    DRAFT = "draft"
    PUBLISHED = "published"


@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    content: str
    published: bool
    author_id: strawberry.Private[str]  # Not exposed in schema

    @strawberry.field
    async def author(self, info: strawberry.types.Info) -> User:
        """Resolve author using DataLoader (avoids N+1)."""
        return await info.context["user_loader"].load(self.author_id)


# --- Input Types ---

@strawberry.input
class CreatePostInput:
    title: str
    content: str
    author_id: strawberry.ID


@strawberry.input
class UpdatePostInput:
    title: Optional[str] = None
    content: Optional[str] = None
    published: Optional[bool] = None


# --- DataLoaders ---

async def load_users(keys: list[str]) -> list[User]:
    """Batch load users by IDs."""
    print(f"  DataLoader: loading users {keys}")
    return [
        User(**users_db[key]) if key in users_db else None
        for key in keys
    ]


async def load_posts_by_author(keys: list[str]) -> list[list[Post]]:
    """Batch load posts grouped by author ID."""
    print(f"  DataLoader: loading posts for authors {keys}")
    result = []
    for author_id in keys:
        author_posts = [
            Post(
                id=p["id"],
                title=p["title"],
                content=p["content"],
                published=p["published"],
                author_id=p["author_id"],
            )
            for p in posts_db
            if p["author_id"] == author_id
        ]
        result.append(author_posts)
    return result


# --- Query ---

@strawberry.type
class Query:
    @strawberry.field
    async def posts(self, published_only: bool = False) -> list[Post]:
        filtered = posts_db
        if published_only:
            filtered = [p for p in posts_db if p["published"]]
        return [
            Post(
                id=p["id"],
                title=p["title"],
                content=p["content"],
                published=p["published"],
                author_id=p["author_id"],
            )
            for p in filtered
        ]

    @strawberry.field
    async def post(self, id: strawberry.ID) -> Optional[Post]:
        for p in posts_db:
            if p["id"] == id:
                return Post(
                    id=p["id"],
                    title=p["title"],
                    content=p["content"],
                    published=p["published"],
                    author_id=p["author_id"],
                )
        return None

    @strawberry.field
    async def users(self) -> list[User]:
        return [User(**u) for u in users_db.values()]


# --- Mutation ---

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_post(self, input: CreatePostInput) -> Post:
        global next_post_id
        post_data = {
            "id": str(next_post_id),
            "title": input.title,
            "content": input.content,
            "author_id": input.author_id,
            "published": False,
        }
        posts_db.append(post_data)
        next_post_id += 1
        return Post(
            id=post_data["id"],
            title=post_data["title"],
            content=post_data["content"],
            published=post_data["published"],
            author_id=post_data["author_id"],
        )

    @strawberry.mutation
    async def update_post(self, id: strawberry.ID, input: UpdatePostInput) -> Optional[Post]:
        for p in posts_db:
            if p["id"] == id:
                if input.title is not None:
                    p["title"] = input.title
                if input.content is not None:
                    p["content"] = input.content
                if input.published is not None:
                    p["published"] = input.published
                return Post(
                    id=p["id"],
                    title=p["title"],
                    content=p["content"],
                    published=p["published"],
                    author_id=p["author_id"],
                )
        return None


# --- Schema and App ---

schema = strawberry.Schema(query=Query, mutation=Mutation)


async def get_context():
    return {
        "user_loader": DataLoader(load_fn=load_users),
        "posts_by_author_loader": DataLoader(load_fn=load_posts_by_author),
    }


graphql_app = GraphQLRouter(schema, context_getter=get_context)

app = FastAPI(title="GraphQL Strawberry Demo")
app.include_router(graphql_app, prefix="/graphql")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
