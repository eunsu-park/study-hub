"""
CLI and SDK Usage Patterns — Authentication, Pagination, Retries

Cloud providers expose two parallel interfaces: a CLI (aws, gcloud, az) and
language SDKs (boto3, google-cloud-python, azure-sdk). The patterns you
must get right in both are the same:

1. How credentials are resolved (chain-of-responsibility, NOT "pass a key in")
2. How large lists are paginated (you must iterate; don't assume page 1 is all)
3. How transient failures are retried (exponential backoff, idempotency)
4. How regions are selected (explicit vs. environment vs. config file)

This script is OFFLINE — it simulates the credential chain and retry logic
against a mock service, so you can see the patterns without needing a real
cloud account.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional


# =============================================================================
# 1. Credential chain — "Default Credential Provider"
# =============================================================================
#
# Real SDKs walk the following sources in order and use the first one that
# returns credentials. We mimic that exactly.


@dataclass
class Credentials:
    access_key: str
    secret_key: str
    source: str


def _from_env() -> Optional[Credentials]:
    ak = os.environ.get("DEMO_ACCESS_KEY")
    sk = os.environ.get("DEMO_SECRET_KEY")
    if ak and sk:
        return Credentials(ak, sk, "environment variables")
    return None


def _from_shared_file() -> Optional[Credentials]:
    # Pretend-check for a config file. Return None so the chain continues.
    return None


def _from_instance_metadata() -> Optional[Credentials]:
    # On a real VM, SDKs fetch short-lived creds from the metadata service.
    # In this script we return a fake "role-based" credential to illustrate.
    if os.environ.get("DEMO_SIMULATE_EC2_ROLE") == "1":
        return Credentials("AKIA-ROLE-FAKE", "secret-fake", "EC2 instance profile")
    return None


DEFAULT_CHAIN: List[Callable[[], Optional[Credentials]]] = [
    _from_env,
    _from_shared_file,
    _from_instance_metadata,
]


def resolve_credentials() -> Optional[Credentials]:
    """
    First non-None source wins. Applications should NEVER hard-code keys —
    relying on the default chain lets credentials rotate without code changes.
    """
    for source in DEFAULT_CHAIN:
        creds = source()
        if creds:
            return creds
    return None


# =============================================================================
# 2. Paginated list — server returns at most page_size; client must iterate
# =============================================================================

@dataclass
class ListResponse:
    items: List[str]
    next_token: Optional[str]


class MockListAPI:
    """Simulated 'list objects' API returning 3 pages of 4 items each."""

    def __init__(self) -> None:
        self.all_items = [f"obj-{i:03d}" for i in range(10)]
        self.page_size = 4

    def list(self, next_token: Optional[str] = None) -> ListResponse:
        start = int(next_token) if next_token else 0
        end = start + self.page_size
        items = self.all_items[start:end]
        nxt = str(end) if end < len(self.all_items) else None
        return ListResponse(items=items, next_token=nxt)


def iterate_all(api: MockListAPI) -> Iterator[str]:
    """Paginator — what every SDK wraps behind a .paginate() helper.
    The rule of thumb: never call .list() once and assume it returned everything."""
    token: Optional[str] = None
    while True:
        resp = api.list(next_token=token)
        yield from resp.items
        if resp.next_token is None:
            return
        token = resp.next_token


# =============================================================================
# 3. Retry with exponential backoff + jitter
# =============================================================================

class TransientError(Exception):
    """Server-side throttling or a blip that a retry will likely fix."""


class FatalError(Exception):
    """Bad request, permission denied — retrying will NOT help."""


def call_with_retry(action: Callable[[], object], *, max_attempts: int = 5) -> object:
    """
    Exponential backoff + jitter. Retries are ONLY for transient errors.
    Fatal errors (400s, 403) are re-raised immediately — retrying a
    malformed request just wastes quota.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return action()
        except TransientError as e:
            if attempt == max_attempts:
                raise RuntimeError(f"max retries exceeded: {e}")
            # Full jitter: sleep in [0, base * 2^attempt]. Prevents synchronized
            # thundering herds when many clients retry simultaneously.
            base = 0.1
            delay = random.uniform(0, base * (2 ** attempt))
            print(f"  attempt {attempt} failed ({e}); retrying in {delay:.2f}s")
            time.sleep(delay)
        except FatalError:
            raise   # no retry — let the caller handle it


# =============================================================================
# 4. Demo — exercise all three patterns
# =============================================================================

def demo_credentials() -> None:
    print("=" * 70)
    print("1. Credential resolution (chain of responsibility)")
    print("=" * 70)
    creds = resolve_credentials()
    if creds:
        print(f"  source: {creds.source}")
        print(f"  access_key: {creds.access_key[:6]}... (redacted)")
    else:
        print("  NO credentials found in the chain.")
        print("  In a real SDK, the API call would fail with NoCredentialsError.")
    print("  Rule: do NOT hard-code keys in source; let the chain resolve them.")


def demo_pagination() -> None:
    print("\n" + "=" * 70)
    print("2. Pagination (server returns 10 items in 3 pages)")
    print("=" * 70)
    api = MockListAPI()
    collected = list(iterate_all(api))
    print(f"  collected {len(collected)} items via paginator: {collected}")
    single = api.list().items
    print(f"  WRONG pattern (single call): {single} — {len(single)} items, MISSING the rest")


def demo_retries() -> None:
    print("\n" + "=" * 70)
    print("3. Retry with exponential backoff + jitter")
    print("=" * 70)

    attempts = {"count": 0}

    def flaky_action() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TransientError(f"throttled (attempt {attempts['count']})")
        return "success"

    random.seed(1)
    result = call_with_retry(flaky_action, max_attempts=5)
    print(f"  final result: {result} after {attempts['count']} attempts")

    def permanent_action() -> str:
        raise FatalError("AccessDenied")

    try:
        call_with_retry(permanent_action, max_attempts=5)
    except FatalError as e:
        print(f"  fatal error NOT retried: {e}")


def main() -> None:
    demo_credentials()
    demo_pagination()
    demo_retries()


if __name__ == "__main__":
    main()
