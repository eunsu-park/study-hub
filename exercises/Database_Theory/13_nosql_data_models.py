"""
Exercises for Lesson 13: NoSQL Data Models
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers CAP theorem analysis, data model selection, Redis rate limiter,
MongoDB schema design, Cassandra data modeling, and consistency models.
"""

import time
import json
from collections import defaultdict


# === Exercise 1: CAP Theorem Analysis ===
# Problem: Classify systems as CP or AP.

def exercise_1():
    """Classify systems under CAP theorem."""
    systems = [
        {
            "system": "1. Banking wire transfer system",
            "classification": "CP (Consistency + Partition tolerance)",
            "justification": "Financial transactions MUST be consistent -- a wire transfer cannot "
                           "result in money appearing in both accounts or neither. During a partition, "
                           "the system becomes unavailable (rejects transactions) rather than risk "
                           "inconsistency. This is why banking systems use ACID databases."
        },
        {
            "system": "2. Social media 'likes' counter",
            "classification": "AP (Availability + Partition tolerance)",
            "justification": "Approximate counts are acceptable. Users expect likes to be 'eventually' "
                           "accurate, not instantly precise. The counter should always be readable "
                           "(available) even during partitions. Off-by-one is fine; being down is not."
        },
        {
            "system": "3. DNS system",
            "classification": "AP (Availability + Partition tolerance)",
            "justification": "DNS prioritizes availability: stale DNS records (eventual consistency) "
                           "are preferable to DNS resolution failures. DNS caches serve old records "
                           "during partitions, with TTL-based eventual consistency."
        },
        {
            "system": "4. Distributed configuration store (etcd/ZooKeeper)",
            "classification": "CP (Consistency + Partition tolerance)",
            "justification": "Configuration data must be consistent -- all nodes must agree on "
                           "the current configuration (e.g., who is the leader, what are the "
                           "connection strings). Uses Raft/ZAB consensus. During partition, "
                           "minority partition becomes unavailable."
        },
        {
            "system": "5. Shopping cart service",
            "classification": "AP (Availability + Partition tolerance)",
            "justification": "Amazon's Dynamo paper showed that shopping carts should always be "
                           "writable (available). If items appear duplicated after a partition heals, "
                           "that's better than losing the cart entirely. Conflicts resolved at checkout."
        }
    ]

    for s in systems:
        print(f"{s['system']}")
        print(f"  Classification: {s['classification']}")
        print(f"  Justification: {s['justification']}")
        print()


# === Exercise 2: Data Model Selection ===
# Problem: Choose NoSQL data model for each scenario.

def exercise_2():
    """Select appropriate NoSQL data model for each scenario."""
    scenarios = [
        {
            "scenario": "1. Real-time multiplayer game (100K players, 60 updates/sec)",
            "model": "Key-Value (Redis)",
            "schema": {
                "player:{player_id}:position": "{'x': 145.3, 'y': 67.8, 'z': 12.0}",
                "player:{player_id}:state": "{'score': 1500, 'health': 85, 'team': 'blue'}",
                "game:{game_id}:state": "{'phase': 'battle', 'time_remaining': 120}"
            },
            "reason": "Sub-millisecond latency required. Simple key lookups. In-memory storage. "
                     "No complex queries needed -- just GET/SET on player state."
        },
        {
            "scenario": "2. Recipe website (variable ingredients, search by ingredient)",
            "model": "Document (MongoDB)",
            "schema": {
                "recipe": {
                    "_id": "ObjectId",
                    "title": "Pasta Carbonara",
                    "ingredients": [
                        {"name": "spaghetti", "amount": "400g"},
                        {"name": "pancetta", "amount": "200g"}
                    ],
                    "steps": ["Boil pasta...", "Fry pancetta..."],
                    "nutrition": {"calories": 650, "protein": 25},
                    "ratings": {"avg": 4.5, "count": 128}
                }
            },
            "reason": "Variable structure (different ingredients per recipe). Embed ingredients for "
                     "single-query reads. Create a multikey index on ingredients.name for search."
        },
        {
            "scenario": "3. Genealogy application (family trees, relationships)",
            "model": "Graph (Neo4j)",
            "schema": {
                "nodes": "(:Person {name, birth_date, death_date, birthplace})",
                "relationships": [
                    "(:Person)-[:PARENT_OF]->(:Person)",
                    "(:Person)-[:MARRIED_TO {date, location}]->(:Person)"
                ]
            },
            "reason": "Naturally graph-structured data. Queries like 'find all ancestors' or "
                     "'find common ancestor' require recursive traversal -- trivial in graph DB, "
                     "painful with JOIN recursion in relational."
        },
        {
            "scenario": "4. IoT fleet management (50K vehicles, GPS every 5 seconds)",
            "model": "Wide-Column (Cassandra) or Time-Series (TimescaleDB)",
            "schema": {
                "table": "vehicle_telemetry",
                "partition_key": "vehicle_id",
                "clustering_key": "timestamp DESC",
                "columns": "latitude, longitude, speed, fuel_level"
            },
            "reason": "High write throughput (~10K writes/sec). Time-ordered data. "
                     "Query pattern: 'latest readings for vehicle X' -> partition by vehicle_id, "
                     "cluster by timestamp DESC. Cassandra excels at this pattern."
        }
    ]

    for s in scenarios:
        print(f"{s['scenario']}")
        print(f"  Data Model: {s['model']}")
        print(f"  Schema: {json.dumps(s['schema'], indent=4, default=str)}")
        print(f"  Reason: {s['reason']}")
        print()


# === Exercise 3: Redis Rate Limiter ===
# Problem: Design a Redis rate limiter (100 requests/user/minute).

def exercise_3():
    """Redis rate limiter implementation."""
    print("Redis Rate Limiter: 100 requests per user per minute")
    print()

    # Simulated Redis store
    redis_store = {}

    def check_rate_limit(user_id, current_time=None):
        """Check rate limit using sliding window counter.

        Uses Redis commands:
          MULTI
          INCR rate_limit:{user_id}:{minute_window}
          EXPIRE rate_limit:{user_id}:{minute_window} 60
          EXEC

        Returns: (allowed: bool, remaining: int)
        """
        if current_time is None:
            current_time = int(time.time())

        # Fixed window: truncate to minute boundary
        window = current_time // 60
        key = f"rate_limit:{user_id}:{window}"

        # Simulate INCR + EXPIRE
        if key not in redis_store:
            redis_store[key] = {"count": 0, "expires": current_time + 60}

        redis_store[key]["count"] += 1
        count = redis_store[key]["count"]
        remaining = max(0, 100 - count)
        allowed = count <= 100

        return allowed, remaining

    # Pseudocode
    print("Pseudocode (Redis commands):")
    print("""
    def check_rate_limit(user_id):
        window = current_time_seconds() // 60
        key = f"rate_limit:{user_id}:{window}"

        # Atomic increment + set expiry
        pipe = redis.pipeline()
        pipe.incr(key)           # Increment counter
        pipe.expire(key, 60)     # Auto-expire after 60 seconds
        count, _ = pipe.execute()

        remaining = max(0, 100 - count)
        allowed = count <= 100

        return {
            'allowed': allowed,
            'remaining': remaining,
            'reset_at': (window + 1) * 60  # Next window start
        }
    """)

    # Simulate
    print("Simulation:")
    current_time = 1000000
    user = "user_42"

    # Normal requests
    for i in range(5):
        allowed, remaining = check_rate_limit(user, current_time + i)
        print(f"  Request {i+1}: allowed={allowed}, remaining={remaining}")

    # Simulate hitting the limit
    redis_store.clear()
    key = f"rate_limit:{user}:{current_time // 60}"
    redis_store[key] = {"count": 99, "expires": current_time + 60}

    print(f"\n  ... (after 99 requests)")
    allowed, remaining = check_rate_limit(user, current_time + 100)
    print(f"  Request 100: allowed={allowed}, remaining={remaining}")
    allowed, remaining = check_rate_limit(user, current_time + 101)
    print(f"  Request 101: allowed={allowed}, remaining={remaining}")

    print()
    print("Key design choices:")
    print("  - Fixed window (minute boundary) for simplicity")
    print("  - INCR is atomic -- no race conditions")
    print("  - EXPIRE auto-cleans old keys (no manual cleanup needed)")
    print("  - O(1) operations -- handles millions of users")


# === Exercise 4: MongoDB Schema Design ===
# Problem: Design a blog platform schema.

def exercise_4():
    """MongoDB schema design for blog platform."""
    print("Blog Platform - MongoDB Schema Design")
    print()

    # Post document
    post_doc = {
        "_id": "ObjectId('...')",
        "title": "Introduction to MongoDB",
        "slug": "introduction-to-mongodb",
        "content": "MongoDB is a document database...",
        "author_id": "ObjectId('author_123')",
        "author_name": "Jane Doe",
        "author_avatar": "/avatars/jane.jpg",
        "tags": ["mongodb", "nosql", "database"],
        "created_at": "2025-01-15T10:00:00Z",
        "updated_at": "2025-01-16T14:30:00Z",
        "comment_count": 42,
        "recent_comments": [
            {"user": "Bob", "text": "Great article!", "created_at": "2025-01-15T12:00:00Z"},
            {"user": "Carol", "text": "Very helpful", "created_at": "2025-01-15T13:00:00Z"}
        ]
    }

    # Comment document (separate collection for full comments)
    comment_doc = {
        "_id": "ObjectId('...')",
        "post_id": "ObjectId('post_456')",
        "user_id": "ObjectId('user_789')",
        "user_name": "Bob",
        "text": "Great article!",
        "created_at": "2025-01-15T12:00:00Z",
        "replies": [
            {
                "user_id": "ObjectId('user_101')",
                "user_name": "Jane Doe",
                "text": "Thank you!",
                "created_at": "2025-01-15T14:00:00Z",
                "replies": []
            }
        ]
    }

    print("Post document (posts collection):")
    print(json.dumps(post_doc, indent=2))
    print()
    print("Comment document (comments collection):")
    print(json.dumps(comment_doc, indent=2))
    print()

    print("Design decisions:")
    decisions = [
        ("Author denormalized in post", "EMBED",
         "Author name/avatar stored directly in post for fast reads. "
         "Update all posts if author changes name (acceptable for rare updates)."),
        ("Full comments", "REFERENCE (separate collection)",
         "Posts can have 0-10,000 comments -- embedding all would exceed 16MB doc limit. "
         "Store in separate 'comments' collection with post_id index."),
        ("Recent comments", "EMBED (top 5 in post)",
         "Embed last 5 comments directly in post for fast 'preview' rendering "
         "without joining. Updated on each new comment."),
        ("Comment replies", "EMBED (up to 3 levels)",
         "Replies nested inside comment doc (up to 3 levels). "
         "Keeps conversation threads together for single-read access."),
        ("comment_count", "EMBED (counter)",
         "Maintained by application (increment on insert). Avoids COUNT queries."),
    ]

    for item, strategy, reason in decisions:
        print(f"  {item}: {strategy}")
        print(f"    Reason: {reason}")
        print()

    print("Indexes:")
    indexes = [
        "db.posts.createIndex({author_id: 1, created_at: -1})  // Q2: posts by author, recent first",
        "db.posts.createIndex({created_at: -1})                 // Q3: recent posts across all authors",
        "db.comments.createIndex({post_id: 1, created_at: 1})   // Q1: comments for a post",
    ]
    for idx in indexes:
        print(f"  {idx}")


# === Exercise 8: CAP Theorem Proof Extension ===
# Problem: Extend Gilbert-Lynch proof to 3-node system.

def exercise_8():
    """CAP theorem proof extension for three nodes."""
    print("CAP Theorem Proof Extension: 3-Node System")
    print()

    steps = [
        ("1. Initial state",
         "All three nodes (N1, N2, N3) agree: X = v0 (same initial value)."),
        ("2. Network partition",
         "Partition: {N1} and {N2, N3} cannot communicate."),
        ("3. Write to minority partition",
         "Client writes X = v1 to N1 (minority partition).\n"
         "   If system is Available: N1 must acknowledge the write.\n"
         "   N1 sets X = v1 locally."),
        ("4. Read from majority partition",
         "Client reads X from N2 (or N3) in the majority partition.\n"
         "   If system is Available: N2 must respond.\n"
         "   N2 cannot contact N1 (partition), so N2 still has X = v0."),
        ("5. Contradiction",
         "If Consistent: read from N2 must return v1 (the latest write).\n"
         "   But N2 returns v0 (cannot see N1's write across partition).\n"
         "   CONTRADICTION: C and A and P cannot all hold simultaneously.\n"
         "   We must sacrifice one:\n"
         "     - Sacrifice C (AP): N2 returns stale v0 (eventually consistent)\n"
         "     - Sacrifice A (CP): N2 refuses to respond until partition heals")
    ]

    for step_name, description in steps:
        print(f"  {step_name}:")
        for line in description.split("\n"):
            print(f"    {line}")
        print()


# === Exercise 9: Consistency Model Classification ===
# Problem: Determine minimum consistency model for each scenario.

def exercise_9():
    """Classify minimum consistency model requirements."""
    scenarios = [
        {
            "scenario": "1. User updates profile picture, views profile immediately",
            "model": "Read-Your-Writes",
            "explanation": "The user who made the change must see it immediately. "
                          "Other users can see the old picture temporarily (eventual consistency for others)."
        },
        {
            "scenario": "2. Group chat: Alice replies to Bob's message, everyone sees Bob first",
            "model": "Causal Consistency",
            "explanation": "The reply is causally dependent on Bob's message. All observers must "
                          "see the causal order (Bob's message before Alice's reply). "
                          "Independent messages can appear in any order."
        },
        {
            "scenario": "3. Inventory system: never oversell a product",
            "model": "Linearizability",
            "explanation": "Stock decrements must appear atomic and in real-time order. "
                          "If stock=1 and two purchases arrive, only one should succeed. "
                          "Requires the strongest guarantee to prevent overselling."
        },
        {
            "scenario": "4. A 'like' counter that must eventually be accurate",
            "model": "Eventual Consistency",
            "explanation": "Approximate counts are fine. Likes will converge to the correct "
                          "count eventually. No ordering or immediacy required."
        },
        {
            "scenario": "5. Email system: user always sees sent messages in Sent folder",
            "model": "Read-Your-Writes",
            "explanation": "The sender must see their own sent email immediately. "
                          "The recipient can experience a slight delay (eventual consistency)."
        }
    ]

    # Sort by strength
    strength_order = ["Linearizability", "Sequential Consistency", "Causal Consistency",
                      "Read-Your-Writes", "Eventual Consistency"]

    for s in scenarios:
        strength = strength_order.index(s["model"]) + 1
        print(f"{s['scenario']}")
        print(f"  Minimum model: {s['model']} (strength: {strength}/5)")
        print(f"  Explanation: {s['explanation']}")
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: CAP Theorem Analysis ===")
    print("=" * 70)
    exercise_1()

    print("=" * 70)
    print("=== Exercise 2: Data Model Selection ===")
    print("=" * 70)
    exercise_2()

    print("=" * 70)
    print("=== Exercise 3: Redis Rate Limiter ===")
    print("=" * 70)
    exercise_3()

    print("=" * 70)
    print("=== Exercise 4: MongoDB Schema Design ===")
    print("=" * 70)
    exercise_4()

    print("=" * 70)
    print("=== Exercise 8: CAP Theorem Proof Extension ===")
    print("=" * 70)
    exercise_8()

    print("=" * 70)
    print("=== Exercise 9: Consistency Model Classification ===")
    print("=" * 70)
    exercise_9()

    print("\nAll exercises completed!")
