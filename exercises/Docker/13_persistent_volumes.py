"""
Exercises for Lesson 13: Persistent Volumes
Topic: Docker

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# === Exercise 1: Volume Type Selector ===
# Problem: Given a workload, recommend the appropriate Docker storage
#   mechanism (volume, bind mount, or tmpfs) with mount flags.

class MountType(Enum):
    VOLUME = "volume"
    BIND = "bind"
    TMPFS = "tmpfs"


def recommend_mount(
    target: str = "/data",
    volume_name: str = "app_data",
    is_dev: bool = False,
    source_path: Optional[str] = None,
    is_sensitive_temp: bool = False,
) -> tuple[MountType, str, str]:
    """Return (type, reason, docker_flag) for the best storage option."""
    if is_sensitive_temp:
        return (MountType.TMPFS,
                "Sensitive temp data should never touch disk",
                f"--mount type=tmpfs,target={target},tmpfs-size=100m")
    if is_dev and source_path:
        return (MountType.BIND,
                "Development benefits from live host file sync",
                f"--mount type=bind,source={source_path},target={target}")
    return (MountType.VOLUME,
            "Named volumes are Docker-managed, ideal for persistence",
            f"--mount type=volume,source={volume_name},target={target}")


def exercise_1():
    """Demonstrate volume type recommendation."""
    print("=== Exercise 1: Volume Type Selector ===\n")
    cases = [
        ("Production DB", dict(target="/var/lib/postgresql/data",
                               volume_name="pgdata")),
        ("Dev hot-reload", dict(is_dev=True, source_path="./src",
                                target="/app/src")),
        ("Session secrets", dict(is_sensitive_temp=True,
                                 target="/run/secrets")),
    ]
    for label, kw in cases:
        mtype, reason, flag = recommend_mount(**kw)
        print(f"  {label:<18} -> {mtype.value:<8} {flag}")


# === Exercise 2: Volume Lifecycle Manager ===
# Problem: Simulate Docker volume lifecycle (create, attach, detach,
#   inspect, remove, prune) and track container references.

@dataclass
class Volume:
    name: str
    driver: str = "local"
    labels: dict = field(default_factory=dict)
    containers: list = field(default_factory=list)

    @property
    def is_dangling(self) -> bool:
        return len(self.containers) == 0


class VolumeManager:
    """Simulates Docker volume lifecycle operations."""

    def __init__(self):
        self._vols: dict[str, Volume] = {}

    def create(self, name: str, **kw) -> Volume:
        if name in self._vols:
            raise ValueError(f"Volume '{name}' exists")
        v = Volume(name=name, **kw)
        self._vols[name] = v
        return v

    def attach(self, vol: str, container: str):
        v = self._get(vol)
        if container not in v.containers:
            v.containers.append(container)

    def detach(self, vol: str, container: str):
        v = self._get(vol)
        if container in v.containers:
            v.containers.remove(container)

    def remove(self, vol: str, force=False):
        v = self._get(vol)
        if v.containers and not force:
            raise ValueError(f"Volume '{vol}' in use by {v.containers}")
        del self._vols[vol]

    def prune(self) -> list[str]:
        dangling = [n for n, v in self._vols.items() if v.is_dangling]
        for n in dangling:
            del self._vols[n]
        return dangling

    def ls(self, label_filter: Optional[str] = None) -> list[str]:
        results = []
        for n, v in self._vols.items():
            if label_filter:
                k, _, val = label_filter.partition("=")
                if v.labels.get(k) != val:
                    continue
            results.append(n)
        return results

    def _get(self, name: str) -> Volume:
        if name not in self._vols:
            raise KeyError(f"Volume '{name}' not found")
        return self._vols[name]


def exercise_2():
    """Demonstrate volume lifecycle management."""
    print("=== Exercise 2: Volume Lifecycle Manager ===\n")
    vm = VolumeManager()
    vm.create("pgdata", labels={"env": "prod"})
    vm.create("redis_data", labels={"env": "prod"})
    vm.create("temp_cache")
    vm.attach("pgdata", "postgres-1")
    vm.attach("redis_data", "redis-1")

    print(f"  All:     {vm.ls()}")
    print(f"  Prod:    {vm.ls('env=prod')}")
    print(f"  Pruned:  {vm.prune()}")
    vm.detach("pgdata", "postgres-1")
    vm.remove("pgdata")
    print(f"  After:   {vm.ls()}")


# === Exercise 3: Backup Strategy Generator ===
# Problem: Generate backup and restore commands for PostgreSQL,
#   MySQL, and MongoDB with proper volume handling.

def generate_backup_cmds(
    db_type: str, container: str, volume: str,
    db_name: str = "mydb", user: str = "postgres",
) -> dict[str, str]:
    """Return dict with 'logical' and 'volume' backup commands."""
    ts = "$(date +%Y%m%d_%H%M%S)"
    if db_type == "postgresql":
        logical = (f"docker exec {container} "
                   f"pg_dump -U {user} -d {db_name} > {db_name}_{ts}.sql")
    elif db_type == "mysql":
        logical = (f"docker exec {container} "
                   f'mysqldump -u {user} -p"$MYSQL_PW" {db_name} '
                   f"> {db_name}_{ts}.sql")
    elif db_type == "mongodb":
        logical = (f"docker exec {container} "
                   f"mongodump --archive=/tmp/bk.gz --gzip --db {db_name} && "
                   f"docker cp {container}:/tmp/bk.gz ./{db_name}_{ts}.gz")
    else:
        raise ValueError(f"Unsupported: {db_type}")

    vol_cmd = (f"docker run --rm -v {volume}:/src:ro -v $(pwd):/bk "
               f"alpine tar czf /bk/{volume}_{ts}.tar.gz -C /src .")
    return {"logical": logical, "volume": vol_cmd}


def exercise_3():
    """Demonstrate backup strategy generation."""
    print("=== Exercise 3: Backup Strategy Generator ===\n")
    for db in ["postgresql", "mysql", "mongodb"]:
        cmds = generate_backup_cmds(db, f"my_{db}", f"{db}_data")
        print(f"  {db.upper()}")
        print(f"    Logical: {cmds['logical']}")
        print(f"    Volume:  {cmds['volume']}\n")


# === Exercise 4: Volume Sharing Safety Analyzer ===
# Problem: Detect unsafe patterns when multiple containers share
#   a volume (multiple writers, shared DB paths, missing :ro).

@dataclass
class Mount:
    container: str
    volume: str
    path: str
    writable: bool = True


DB_PATHS = {"/var/lib/postgresql/data", "/var/lib/mysql",
            "/data/db", "/data/configdb"}


def analyze_sharing(mounts: list[Mount]) -> list[str]:
    """Return list of warnings for unsafe volume sharing patterns."""
    warnings = []
    by_vol: dict[str, list[Mount]] = {}
    for m in mounts:
        by_vol.setdefault(m.volume, []).append(m)

    for vol, users in by_vol.items():
        writers = [u for u in users if u.writable]
        if len(writers) > 1:
            names = [w.container for w in writers]
            warnings.append(
                f"DANGER: '{vol}' has {len(writers)} writers {names}. "
                f"Risk of data corruption."
            )
        for u in users:
            if u.path in DB_PATHS and len(users) > 1:
                warnings.append(
                    f"DANGER: '{vol}' shares DB path '{u.path}' across "
                    f"{len(users)} containers."
                )
                break
    return warnings


def exercise_4():
    """Demonstrate volume sharing safety analysis."""
    print("=== Exercise 4: Volume Sharing Safety Analyzer ===\n")

    safe = [Mount("writer", "logs", "/logs"),
            Mount("reader", "logs", "/logs", writable=False)]
    print(f"  Safe (1 writer + 1 ro reader): {len(analyze_sharing(safe))} warnings")

    danger = [Mount("pg-1", "pgdata", "/var/lib/postgresql/data"),
              Mount("pg-2", "pgdata", "/var/lib/postgresql/data")]
    for w in analyze_sharing(danger):
        print(f"  {w}")

    multi = [Mount("app-1", "uploads", "/uploads"),
             Mount("app-2", "uploads", "/uploads"),
             Mount("cdn", "uploads", "/uploads", writable=False)]
    for w in analyze_sharing(multi):
        print(f"  {w}")


# === Exercise 5: Compose Volume Config Generator ===
# Problem: Generate docker-compose.yml volumes section from service
#   requirements with proper driver configuration.

@dataclass
class SvcVolume:
    service: str
    volume: str
    path: str
    read_only: bool = False
    labels: dict = field(default_factory=dict)


def generate_compose(specs: list[SvcVolume]) -> str:
    """Generate docker-compose.yml content with volume definitions."""
    svcs: dict[str, list[str]] = {}
    vols: dict[str, dict] = {}
    for s in specs:
        mount = f"{s.volume}:{s.path}" + (":ro" if s.read_only else "")
        svcs.setdefault(s.service, []).append(mount)
        if s.volume not in vols:
            vols[s.volume] = s.labels

    lines = ['version: "3.9"', "", "services:"]
    for svc, mounts in svcs.items():
        lines.append(f"  {svc}:")
        lines.append("    volumes:")
        for m in mounts:
            lines.append(f"      - {m}")

    lines.extend(["", "volumes:"])
    for vol, labels in vols.items():
        lines.append(f"  {vol}:")
        lines.append("    driver: local")
        if labels:
            lines.append("    labels:")
            for k, v in labels.items():
                lines.append(f'      {k}: "{v}"')
    return "\n".join(lines)


def exercise_5():
    """Demonstrate Compose volume configuration generation."""
    print("=== Exercise 5: Compose Volume Config Generator ===\n")
    specs = [
        SvcVolume("postgres", "pgdata", "/var/lib/postgresql/data",
                  labels={"env": "prod", "backup": "daily"}),
        SvcVolume("app", "uploads", "/app/uploads"),
        SvcVolume("nginx", "uploads", "/usr/share/nginx/uploads",
                  read_only=True),
    ]
    print(generate_compose(specs))


# === Run all exercises ===

def main():
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4, exercise_5]
    for i, ex in enumerate(exercises):
        if i > 0:
            print("\n")
        ex()
    print("\n\nAll exercises completed!")


if __name__ == "__main__":
    main()
