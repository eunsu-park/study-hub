"""
Example: Network Scanning
==========================
TCP port scanner, host discovery, banner grabber, scan report generator.

IMPORTANT: For authorized security testing and CTF only.
"""

import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

SERVICES = {
    21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
    80: "HTTP", 110: "POP3", 135: "MSRPC", 139: "NetBIOS",
    143: "IMAP", 443: "HTTPS", 445: "SMB", 993: "IMAPS",
    1433: "MSSQL", 3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
    5900: "VNC", 6379: "Redis", 8080: "HTTP-Proxy", 27017: "MongoDB",
}

@dataclass
class PortResult:
    port: int
    state: str
    service: str = ""
    banner: str = ""

def scan_port(host: str, port: int, timeout: float = 2.0) -> PortResult:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        if sock.connect_ex((host, port)) == 0:
            banner = ""
            try:
                sock.send(b"\r\n")
                banner = sock.recv(1024).decode("utf-8", errors="replace").strip()[:100]
            except (socket.timeout, OSError):
                pass
            sock.close()
            return PortResult(port, "open", SERVICES.get(port, "unknown"), banner)
        sock.close()
        return PortResult(port, "closed")
    except socket.timeout:
        return PortResult(port, "filtered")
    except OSError:
        return PortResult(port, "error")

def scan_host(host: str, ports: list[int], workers: int = 100) -> list[PortResult]:
    open_ports = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(scan_port, host, p): p for p in ports}
        for f in as_completed(futures):
            r = f.result()
            if r.state == "open":
                open_ports.append(r)
    return sorted(open_ports, key=lambda r: r.port)

def report(host: str, results: list[PortResult]) -> str:
    lines = [f"Scan Report: {host}", f"Open ports: {len(results)}", "=" * 50]
    for r in results:
        lines.append(f"  {r.port:>6}/tcp  {r.service:15s}  {r.banner[:40]}")
    return "\n".join(lines)

if __name__ == "__main__":
    print("Network Scanning Example")
    print(f"Known services: {len(SERVICES)}")
    print("Usage: results = scan_host('target', [22, 80, 443])")
