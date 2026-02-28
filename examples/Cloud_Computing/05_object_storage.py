"""
Object Storage Simulation (S3 / GCS / Azure Blob)

Demonstrates object storage concepts:
- Upload/download with metadata and ETags
- Object versioning (keeping history of all changes)
- Lifecycle policies (automatic storage class transitions)
- Storage class cost comparison (Standard -> IA -> Glacier)
- Eventual consistency behavior simulation

No cloud account required -- all behavior is simulated locally.
"""

import hashlib
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class StorageClass(Enum):
    """Storage classes represent cost/access trade-offs.
    Cheaper classes cost less to store but more to retrieve.
    The right class depends on access frequency."""
    STANDARD = "STANDARD"                # Frequent access, lowest latency
    STANDARD_IA = "STANDARD_IA"          # Infrequent access (~30-day minimum)
    GLACIER = "GLACIER"                  # Archive, retrieval takes minutes-hours
    GLACIER_DEEP = "GLACIER_DEEP_ARCHIVE"  # Coldest tier, retrieval takes 12-48h

    @property
    def storage_cost_gb_month(self) -> float:
        """Per-GB monthly storage cost (approximate S3 us-east-1 pricing)."""
        costs = {
            "STANDARD": 0.023,
            "STANDARD_IA": 0.0125,
            "GLACIER": 0.004,
            "GLACIER_DEEP_ARCHIVE": 0.00099,
        }
        return costs[self.value]

    @property
    def retrieval_cost_gb(self) -> float:
        """Per-GB retrieval cost -- this is where cold tiers get expensive.
        Always analyze access patterns before choosing a storage class."""
        costs = {
            "STANDARD": 0.0,
            "STANDARD_IA": 0.01,
            "GLACIER": 0.03,
            "GLACIER_DEEP_ARCHIVE": 0.05,
        }
        return costs[self.value]

    @property
    def retrieval_time_description(self) -> str:
        times = {
            "STANDARD": "milliseconds",
            "STANDARD_IA": "milliseconds",
            "GLACIER": "1-5 minutes (expedited) to 3-5 hours (standard)",
            "GLACIER_DEEP_ARCHIVE": "12-48 hours",
        }
        return times[self.value]


@dataclass
class ObjectVersion:
    """A single version of an object. Versioning protects against
    accidental deletions and overwrites -- essential for compliance."""
    version_id: str
    size_bytes: int
    etag: str          # MD5 hash for integrity verification
    storage_class: StorageClass
    timestamp: float
    is_delete_marker: bool = False


@dataclass
class StorageObject:
    """Represents an object in a bucket with full version history."""
    key: str           # Object path/name (e.g., "images/photo.jpg")
    versions: List[ObjectVersion] = field(default_factory=list)

    @property
    def current_version(self) -> Optional[ObjectVersion]:
        """Latest non-deleted version."""
        for v in reversed(self.versions):
            if not v.is_delete_marker:
                return v
        return None

    @property
    def total_storage_bytes(self) -> int:
        """All versions consume storage -- this is why lifecycle policies
        should clean up old versions to control costs."""
        return sum(v.size_bytes for v in self.versions if not v.is_delete_marker)


@dataclass
class LifecycleRule:
    """Automates storage class transitions and cleanup.
    Example: move to IA after 30 days, Glacier after 90, delete after 365."""
    prefix: str                # Apply to objects with this key prefix
    transition_days: Dict[StorageClass, int] = field(default_factory=dict)
    expiration_days: Optional[int] = None  # Delete after N days
    noncurrent_expiration_days: Optional[int] = None  # Delete old versions


class ObjectStorageBucket:
    """Simulates an object storage bucket (S3 bucket / GCS bucket)."""

    def __init__(self, name: str, versioning_enabled: bool = False):
        self.name = name
        self.versioning_enabled = versioning_enabled
        self.objects: Dict[str, StorageObject] = {}
        self.lifecycle_rules: List[LifecycleRule] = []
        self._version_counter = 0

    def _generate_etag(self, data: bytes) -> str:
        """ETags use MD5 for single-part uploads. For multipart uploads,
        S3 uses a different algorithm (MD5 of part MD5s + part count)."""
        return hashlib.md5(data).hexdigest()

    def _next_version_id(self) -> str:
        self._version_counter += 1
        return f"v{self._version_counter:06d}"

    def put_object(self, key: str, data: bytes,
                   storage_class: StorageClass = StorageClass.STANDARD) -> dict:
        """Upload an object. If versioning is enabled, previous versions are kept."""
        etag = self._generate_etag(data)

        version = ObjectVersion(
            version_id=self._next_version_id() if self.versioning_enabled else "null",
            size_bytes=len(data),
            etag=etag,
            storage_class=storage_class,
            timestamp=time.time(),
        )

        if key not in self.objects:
            self.objects[key] = StorageObject(key=key)

        obj = self.objects[key]
        if not self.versioning_enabled:
            # Without versioning, overwrite destroys the old version permanently
            obj.versions = [version]
        else:
            obj.versions.append(version)

        return {
            "key": key,
            "version_id": version.version_id,
            "etag": etag,
            "size": len(data),
            "storage_class": storage_class.value,
        }

    def get_object(self, key: str,
                   version_id: Optional[str] = None) -> Optional[dict]:
        """Download an object. Retrieval cost depends on storage class."""
        obj = self.objects.get(key)
        if not obj:
            return None

        if version_id:
            version = next((v for v in obj.versions if v.version_id == version_id), None)
        else:
            version = obj.current_version

        if not version or version.is_delete_marker:
            return None

        retrieval_cost = version.size_bytes / (1024**3) * version.storage_class.retrieval_cost_gb
        return {
            "key": key,
            "version_id": version.version_id,
            "size": version.size_bytes,
            "etag": version.etag,
            "storage_class": version.storage_class.value,
            "retrieval_cost": round(retrieval_cost, 8),
            "retrieval_time": version.storage_class.retrieval_time_description,
        }

    def delete_object(self, key: str) -> dict:
        """With versioning, 'delete' inserts a delete marker rather than
        actually removing data. This allows recovery of accidentally deleted objects."""
        obj = self.objects.get(key)
        if not obj:
            return {"error": "NoSuchKey"}

        if self.versioning_enabled:
            marker = ObjectVersion(
                version_id=self._next_version_id(),
                size_bytes=0, etag="", storage_class=StorageClass.STANDARD,
                timestamp=time.time(), is_delete_marker=True,
            )
            obj.versions.append(marker)
            return {"deleted": key, "delete_marker": True, "version_id": marker.version_id}
        else:
            del self.objects[key]
            return {"deleted": key, "permanently_removed": True}

    def apply_lifecycle(self, current_day: int) -> List[str]:
        """Apply lifecycle rules to transition storage classes and expire objects."""
        actions = []
        for key, obj in list(self.objects.items()):
            for rule in self.lifecycle_rules:
                if not key.startswith(rule.prefix):
                    continue
                for version in obj.versions:
                    if version.is_delete_marker:
                        continue
                    age_days = current_day - int(version.timestamp)
                    # Transition to cheaper storage classes based on age
                    for target_class, days in sorted(rule.transition_days.items(),
                                                     key=lambda x: x[1], reverse=True):
                        if age_days >= days and version.storage_class != target_class:
                            old_class = version.storage_class
                            version.storage_class = target_class
                            actions.append(
                                f"  Transition: {key} [{old_class.value} -> {target_class.value}] "
                                f"(age={age_days}d)")
                            break
        return actions


def demo_basic_operations():
    """Demonstrate upload, download, versioning, and delete."""
    print("=" * 70)
    print("Basic Object Storage Operations")
    print("=" * 70)

    bucket = ObjectStorageBucket("my-app-bucket", versioning_enabled=True)

    # Upload initial version
    result = bucket.put_object("reports/Q1.pdf", b"Q1 report data v1")
    print(f"  PUT: {result}")

    # Overwrite creates a new version (old version preserved)
    result = bucket.put_object("reports/Q1.pdf", b"Q1 report data v2 - corrected")
    print(f"  PUT (update): {result}")

    # Get latest version
    obj = bucket.get_object("reports/Q1.pdf")
    print(f"  GET (latest): version={obj['version_id']}, size={obj['size']}")

    # Get specific old version
    obj_v1 = bucket.get_object("reports/Q1.pdf", version_id="v000001")
    print(f"  GET (v1):     version={obj_v1['version_id']}, size={obj_v1['size']}")

    # Delete (creates marker, does not destroy data)
    del_result = bucket.delete_object("reports/Q1.pdf")
    print(f"  DELETE: {del_result}")

    # Can still access old versions even after delete!
    obj_v1_again = bucket.get_object("reports/Q1.pdf", version_id="v000001")
    print(f"  GET after delete (v1): {obj_v1_again is not None} (version still accessible)")
    print()


def demo_lifecycle_transitions():
    """Simulate lifecycle policy transitions over time."""
    print("=" * 70)
    print("Lifecycle Policy Simulation (365-day timeline)")
    print("=" * 70)

    bucket = ObjectStorageBucket("archive-bucket")
    bucket.lifecycle_rules.append(LifecycleRule(
        prefix="logs/",
        transition_days={
            StorageClass.STANDARD_IA: 30,
            StorageClass.GLACIER: 90,
            StorageClass.GLACIER_DEEP: 180,
        },
        expiration_days=365,
    ))

    # Upload objects at day 0
    for i in range(3):
        bucket.put_object(f"logs/app-{i}.log", b"x" * random.randint(1000, 50000))

    checkpoints = [0, 30, 90, 180, 365]
    for day in checkpoints:
        actions = bucket.apply_lifecycle(current_day=day)
        print(f"\n  Day {day}:")
        if actions:
            for a in actions:
                print(a)
        else:
            print("  No transitions")
    print()


def demo_cost_comparison():
    """Compare total cost of ownership across storage classes."""
    print("=" * 70)
    print("Storage Class Cost Comparison (1 TB, 12 months)")
    print("=" * 70)
    data_tb = 1.0
    data_gb = data_tb * 1024
    monthly_retrievals_gb = 50  # 50 GB retrieved per month

    print(f"  {'Storage Class':<25} {'Storage/mo':>12} {'Retrieval/mo':>14} {'Total/year':>12}")
    print(f"  {'-'*65}")
    for sc in StorageClass:
        storage_monthly = data_gb * sc.storage_cost_gb_month
        retrieval_monthly = monthly_retrievals_gb * sc.retrieval_cost_gb
        total_yearly = (storage_monthly + retrieval_monthly) * 12
        print(f"  {sc.value:<25} ${storage_monthly:>10.2f}   ${retrieval_monthly:>11.2f}  ${total_yearly:>10.2f}")

    print(f"\n  Insight: STANDARD_IA saves ~45% on storage but costs $0.01/GB to retrieve.")
    print(f"  If you access data frequently, STANDARD may actually be cheaper overall.")
    print()


if __name__ == "__main__":
    random.seed(42)
    demo_basic_operations()
    demo_lifecycle_transitions()
    demo_cost_comparison()
