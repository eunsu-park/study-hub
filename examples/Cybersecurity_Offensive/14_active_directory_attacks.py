"""
Example: Active Directory Attacks
===================================
AD enumeration helpers, Kerberoasting analysis, attack path modeling,
and SPN/UAC flag parsing.

IMPORTANT: For authorized security testing and CTF only.
"""

from dataclasses import dataclass, field
from collections import deque


# ---------------------------------------------------------------------------
# UserAccountControl Flag Parser
# ---------------------------------------------------------------------------

UAC_FLAGS = {
    0x0001: "SCRIPT",
    0x0002: "ACCOUNTDISABLE",
    0x0008: "HOMEDIR_REQUIRED",
    0x0010: "LOCKOUT",
    0x0020: "PASSWD_NOTREQD",
    0x0040: "PASSWD_CANT_CHANGE",
    0x0080: "ENCRYPTED_TEXT_PWD_ALLOWED",
    0x0200: "NORMAL_ACCOUNT",
    0x0800: "INTERDOMAIN_TRUST_ACCOUNT",
    0x1000: "WORKSTATION_TRUST_ACCOUNT",
    0x2000: "SERVER_TRUST_ACCOUNT",
    0x10000: "DONT_EXPIRE_PASSWORD",
    0x20000: "MNS_LOGON_ACCOUNT",
    0x40000: "SMARTCARD_REQUIRED",
    0x80000: "TRUSTED_FOR_DELEGATION",
    0x100000: "NOT_DELEGATED",
    0x200000: "USE_DES_KEY_ONLY",
    0x400000: "DONT_REQ_PREAUTH",
    0x800000: "PASSWORD_EXPIRED",
    0x1000000: "TRUSTED_TO_AUTH_FOR_DELEGATION",
}


def parse_uac(value: int) -> list[str]:
    """Parse UserAccountControl integer into flag names."""
    return [name for flag, name in UAC_FLAGS.items() if value & flag]


# ---------------------------------------------------------------------------
# Kerberoasting Target Identification
# ---------------------------------------------------------------------------

@dataclass
class ADUser:
    dn: str
    sam_account_name: str
    spns: list[str]
    uac: int
    member_of: list[str]

    @property
    def kerberoastable(self) -> bool:
        return len(self.spns) > 0 and "ACCOUNTDISABLE" not in parse_uac(self.uac)

    @property
    def asrep_roastable(self) -> bool:
        return "DONT_REQ_PREAUTH" in parse_uac(self.uac)

    @property
    def is_admin(self) -> bool:
        admin_groups = {"Domain Admins", "Enterprise Admins", "Administrators"}
        return any(g in self.member_of for g in admin_groups)


# ---------------------------------------------------------------------------
# Attack Path Graph
# ---------------------------------------------------------------------------

@dataclass
class AttackEdge:
    source: str
    relation: str
    target: str


class AttackGraph:
    """Graph for AD attack path analysis."""

    def __init__(self):
        self.edges: list[AttackEdge] = []
        self.adjacency: dict[str, list[tuple[str, str]]] = {}

    def add_edge(self, source: str, relation: str, target: str):
        edge = AttackEdge(source, relation, target)
        self.edges.append(edge)
        self.adjacency.setdefault(source, []).append((target, relation))

    def shortest_path(self, start: str, goal: str) -> list[tuple[str, str]]:
        """BFS shortest path from start to goal node."""
        queue = deque([(start, [(start, "start")])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node == goal:
                return path
            for neighbor, rel in self.adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(neighbor, rel)]))
        return []

    def find_all_paths(self, start: str, goal: str,
                       max_depth: int = 10) -> list[list[tuple[str, str]]]:
        """DFS all paths up to max_depth."""
        results = []

        def dfs(node, path, visited):
            if len(path) > max_depth:
                return
            if node == goal:
                results.append(list(path))
                return
            for neighbor, rel in self.adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append((neighbor, rel))
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.discard(neighbor)

        dfs(start, [(start, "start")], {start})
        return results


# ---------------------------------------------------------------------------
# Kerberos Encryption Type Analysis
# ---------------------------------------------------------------------------

KERBEROS_ETYPES = {
    0x17: ("RC4-HMAC", "weak", "Crackable with hashcat -m 13100"),
    0x12: ("AES256-CTS", "strong", "Very slow to crack"),
    0x11: ("AES128-CTS", "moderate", "Slower to crack than RC4"),
    0x03: ("DES-CBC-MD5", "deprecated", "Trivially crackable"),
}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Active Directory Attacks Examples")
    print("=" * 50)

    # UAC parsing
    print("\nUAC Flag Parsing:")
    test_values = [66048, 4260352, 512]
    for val in test_values:
        flags = parse_uac(val)
        print(f"  UAC={val}: {', '.join(flags)}")

    # Kerberoastable users
    print("\nKerberoasting Target Analysis:")
    users = [
        ADUser("CN=svc-sql,DC=corp", "svc-sql",
               ["MSSQLSvc/db01:1433"], 66048, ["Domain Users"]),
        ADUser("CN=admin01,DC=corp", "admin01",
               [], 66048, ["Domain Admins"]),
        ADUser("CN=svc-backup,DC=corp", "svc-backup",
               [], 4260352, ["Backup Operators"]),
    ]
    for user in users:
        tags = []
        if user.kerberoastable:
            tags.append("KERBEROASTABLE")
        if user.asrep_roastable:
            tags.append("ASREP-ROASTABLE")
        if user.is_admin:
            tags.append("ADMIN")
        print(f"  {user.sam_account_name}: {', '.join(tags) or 'standard user'}")

    # Attack path
    print("\nAttack Path Analysis:")
    graph = AttackGraph()
    graph.add_edge("webuser", "CanRDP", "WEB01")
    graph.add_edge("WEB01", "HasSession", "svc-sql")
    graph.add_edge("svc-sql", "MemberOf", "DBA_Group")
    graph.add_edge("DBA_Group", "AdminTo", "DB01")
    graph.add_edge("DB01", "HasSession", "admin01")
    graph.add_edge("admin01", "MemberOf", "Domain Admins")

    path = graph.shortest_path("webuser", "Domain Admins")
    print("  Path: webuser -> Domain Admins")
    for node, rel in path:
        print(f"    [{rel}] -> {node}")

    # Encryption types
    print("\nKerberos Encryption Types:")
    for etype, (name, strength, note) in KERBEROS_ETYPES.items():
        print(f"  0x{etype:02x} {name:15s} [{strength:10s}] {note}")


if __name__ == "__main__":
    demo()
