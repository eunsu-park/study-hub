"""
Example: Reconnaissance
========================
OSINT collector, DNS enumerator, subdomain discovery, Google dork generator.

IMPORTANT: For authorized security testing and CTF only.
"""

import socket
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError


@dataclass
class SubdomainResult:
    subdomain: str
    ip_address: Optional[str] = None
    is_alive: bool = False

COMMON_SUBDOMAINS = [
    "www", "mail", "ftp", "admin", "api", "dev", "staging", "test",
    "beta", "git", "jenkins", "vpn", "cdn", "static", "blog", "shop",
    "app", "sso", "docs", "status", "portal", "remote", "db", "backup",
]

def check_subdomain(domain: str, prefix: str) -> Optional[SubdomainResult]:
    sub = f"{prefix}.{domain}"
    try:
        ips = socket.getaddrinfo(sub, None, socket.AF_INET)
        return SubdomainResult(sub, ips[0][4][0], True)
    except socket.gaierror:
        return None

def enumerate_subdomains(domain: str, workers: int = 10) -> list[SubdomainResult]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(check_subdomain, domain, p): p for p in COMMON_SUBDOMAINS}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
    return sorted(results, key=lambda r: r.subdomain)

def generate_email_patterns(domain: str, first: str, last: str) -> list[str]:
    fn, ln = first.lower(), last.lower()
    return list(dict.fromkeys([
        f"{fn}.{ln}@{domain}", f"{fn}{ln}@{domain}", f"{fn[0]}{ln}@{domain}",
        f"{fn}_{ln}@{domain}", f"{ln}.{fn}@{domain}", f"{fn}@{domain}",
    ]))

GOOGLE_DORKS = [
    ('site:{d} filetype:pdf "confidential"', "Find confidential PDFs"),
    ('site:{d} inurl:admin', "Find admin pages"),
    ('site:{d} "error in your SQL syntax"', "Find SQL errors"),
    ('site:{d} intitle:"index of"', "Find directory listings"),
    ('site:{d} filetype:env OR filetype:yml "api_key"', "Find API keys"),
    ('site:github.com "{d}" password OR secret', "Find GitHub leaks"),
]

def generate_dorks(domain: str) -> list[tuple[str, str]]:
    return [(q.format(d=domain), desc) for q, desc in GOOGLE_DORKS]

if __name__ == "__main__":
    print("Reconnaissance Example Module")
    print("=" * 50)
    print(f"Subdomain wordlist: {len(COMMON_SUBDOMAINS)} entries")
    print("\nEmail patterns for Jane Smith @ example.com:")
    for p in generate_email_patterns("example.com", "Jane", "Smith"):
        print(f"  {p}")
    print("\nGoogle dorks for example.com:")
    for query, desc in generate_dorks("example.com"):
        print(f"  [{desc}] {query}")
