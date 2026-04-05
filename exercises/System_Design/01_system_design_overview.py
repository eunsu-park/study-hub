"""
Exercises for Lesson 01: System Design Overview
Topic: System_Design

Solutions to practice problems from the lesson.
Back-of-the-envelope calculations and capacity estimation exercises.
"""

import math


# === Exercise 1: QPS Calculation ===
# Problem: Calculate image upload QPS for an Instagram-like service.
# Conditions:
# - DAU: 500 million
# - Daily image upload rate: 10% of users upload average 2 images

def exercise_1():
    """QPS Calculation for Instagram-like image uploads."""
    dau = 500_000_000
    upload_ratio = 0.10
    images_per_uploader = 2
    seconds_per_day = 86_400

    daily_uploads = dau * upload_ratio * images_per_uploader
    avg_qps = daily_uploads / seconds_per_day
    peak_qps = avg_qps * 3  # Peak is typically 2-3x average

    print(f"DAU: {dau:,}")
    print(f"Uploaders: {dau * upload_ratio:,.0f} ({upload_ratio:.0%} of DAU)")
    print(f"Daily uploads: {daily_uploads:,.0f}")
    print(f"Average QPS: {avg_qps:,.0f}")
    print(f"Peak QPS (3x): {peak_qps:,.0f}")


# === Exercise 2: Storage Estimation ===
# Problem: Estimate 1-year message storage for a chat app.
# Conditions:
# - DAU: 100 million
# - Average daily messages: 50/user
# - Average message size: 100 bytes

def exercise_2():
    """Storage estimation for a chat application."""
    dau = 100_000_000
    messages_per_user_per_day = 50
    avg_message_size_bytes = 100
    days_per_year = 365

    daily_messages = dau * messages_per_user_per_day
    daily_storage_bytes = daily_messages * avg_message_size_bytes
    daily_storage_gb = daily_storage_bytes / (1024 ** 3)
    annual_storage_gb = daily_storage_gb * days_per_year
    annual_storage_tb = annual_storage_gb / 1024

    print(f"DAU: {dau:,}")
    print(f"Daily messages: {daily_messages:,.0f} ({daily_messages / 1e9:.0f} billion)")
    print(f"Daily storage: {daily_storage_gb:,.1f} GB")
    print(f"Annual storage: {annual_storage_gb:,.1f} GB = {annual_storage_tb:,.1f} TB")

    # With replication factor and overhead
    replication_factor = 3
    overhead = 1.2  # 20% overhead for indexes, metadata
    total_annual_tb = annual_storage_tb * replication_factor * overhead
    print(f"\nWith 3x replication + 20% overhead: {total_annual_tb:,.1f} TB")


# === Exercise 3: Server Count Estimation ===
# Problem: Estimate web server count needed to handle 100,000 QPS.
# Conditions:
# - Single server throughput: 1,000 QPS
# - Need 20% overhead for availability

def exercise_3():
    """Server count estimation for 100K QPS."""
    target_qps = 100_000
    single_server_qps = 1_000
    overhead_ratio = 0.20  # 20% availability overhead

    base_servers = target_qps / single_server_qps
    with_overhead = base_servers * (1 + overhead_ratio)
    with_ha = with_overhead * 2  # HA redundancy (active-passive)

    print(f"Target QPS: {target_qps:,}")
    print(f"Single server capacity: {single_server_qps:,} QPS")
    print(f"Base servers needed: {base_servers:.0f}")
    print(f"With 20% overhead: {with_overhead:.0f}")
    print(f"With HA (2x redundancy): {with_ha:.0f}")


# === Exercise 4: Requirements Clarification ===
# Problem: Given "Design a URL shortening service",
# write 5 questions to ask the interviewer.

def exercise_4():
    """Requirements clarification for URL shortener design."""
    questions = [
        ("Scale", "What are expected DAU and MAU? How many URLs are shortened per day?"),
        ("URL Format", "Expected length/format of shortened URLs? Custom aliases allowed?"),
        ("Expiration", "Do URLs expire? If so, what's the default TTL?"),
        ("Analytics", "Do we need analytics (click count, geographic stats, referrer tracking)?"),
        ("Availability", "What availability target? (99.9%? 99.99%?) Which is more important: consistency or availability?"),
    ]

    print("URL Shortener - Requirements Clarification Questions:")
    print("=" * 60)
    for i, (category, question) in enumerate(questions, 1):
        print(f"\n{i}. [{category}]")
        print(f"   {question}")

    # Follow-up: quick capacity estimation based on typical answers
    print("\n" + "=" * 60)
    print("Quick Capacity Estimation (assuming typical answers):")
    dau = 10_000_000
    writes_per_day = 1_000_000
    read_write_ratio = 100
    reads_per_day = writes_per_day * read_write_ratio

    print(f"  DAU: {dau:,}")
    print(f"  New URLs/day: {writes_per_day:,}")
    print(f"  Reads/day: {reads_per_day:,}")
    print(f"  Write QPS: {writes_per_day / 86400:.0f}")
    print(f"  Read QPS: {reads_per_day / 86400:,.0f}")


# === Exercise 5: High-Level Architecture ===
# Problem: Draw high-level architecture for a simple file sharing service.

def exercise_5():
    """High-level architecture for a file sharing service."""
    architecture = """
    File Sharing Service - High-Level Architecture
    ================================================

    ┌─────────┐     ┌─────────────┐     ┌──────────────┐
    │ Client  │────>│ Load        │────>│ Web Server   │
    └─────────┘     │ Balancer    │     └──────┬───────┘
                    └─────────────┘            │
                                         ┌────┴────────────┐
                                         │                 │
                                         v                 v
                                   ┌──────────┐     ┌──────────┐
                                   │ Metadata │     │ Object   │
                                   │ DB       │     │ Storage  │
                                   │ (MySQL)  │     │ (S3)     │
                                   └──────────┘     └──────────┘

    Components:
    -----------
    1. Client: Web browser or desktop/mobile app
    2. Load Balancer: Distributes requests across web servers
    3. Web Server: Handles API requests (upload, download, share)
    4. Metadata DB: Stores file metadata (name, owner, size, permissions)
    5. Object Storage: Stores actual file content (S3-compatible)

    Key APIs:
    ---------
    - POST /files          -> Upload file
    - GET  /files/{id}     -> Download file
    - POST /files/{id}/share -> Generate share link
    - GET  /shared/{token} -> Access shared file
    """

    print(architecture)

    # Capacity estimation
    print("Capacity Estimation:")
    print("-" * 40)
    total_users = 10_000_000
    dau = 1_000_000
    uploads_per_user_day = 2
    avg_file_size_mb = 5

    daily_uploads = dau * uploads_per_user_day
    daily_storage_tb = (daily_uploads * avg_file_size_mb) / (1024 * 1024)
    yearly_storage_pb = daily_storage_tb * 365 / 1024

    print(f"  Total users: {total_users:,}")
    print(f"  DAU: {dau:,}")
    print(f"  Daily uploads: {daily_uploads:,}")
    print(f"  Daily storage: {daily_storage_tb:.2f} TB")
    print(f"  Yearly storage: {yearly_storage_pb:.2f} PB")
    print(f"  Upload QPS: {daily_uploads / 86400:.0f}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: QPS Calculation ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Storage Estimation ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Server Count Estimation ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Requirements Clarification ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: High-Level Architecture ===")
    print("=" * 60)
    exercise_5()

    print("\nAll exercises completed!")
