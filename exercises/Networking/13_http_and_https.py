"""
Exercises for Lesson 13: HTTP and HTTPS
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem 1: HTTP Methods
    - 3 differences between GET and POST.
    - What is idempotency? List all idempotent methods.
    """
    differences = [
        ("Data location", "GET: parameters in URL query string",
         "POST: parameters in request body"),
        ("Cacheability", "GET: cacheable by browsers/proxies",
         "POST: not cached by default"),
        ("Data length", "GET: limited by URL length (~2KB)",
         "POST: no practical size limit"),
        ("Visibility", "GET: data visible in URL (browser history, logs)",
         "POST: data in body (not visible in URL)"),
        ("Use case", "GET: retrieve data (read-only)",
         "POST: submit data (create/modify)"),
    ]

    print("GET vs POST Differences:")
    for i, (aspect, get_desc, post_desc) in enumerate(differences[:3], 1):
        print(f"\n  {i}. {aspect}:")
        print(f"     {get_desc}")
        print(f"     {post_desc}")

    print("\n\nIdempotency:")
    print("  Definition: An operation is idempotent if performing it multiple")
    print("  times produces the same result as performing it once.")
    print()

    methods = {
        "GET":     {"idempotent": True,  "safe": True,  "description": "Retrieve resource"},
        "HEAD":    {"idempotent": True,  "safe": True,  "description": "GET without body"},
        "PUT":     {"idempotent": True,  "safe": False, "description": "Replace resource"},
        "DELETE":  {"idempotent": True,  "safe": False, "description": "Remove resource"},
        "OPTIONS": {"idempotent": True,  "safe": True,  "description": "Get supported methods"},
        "POST":    {"idempotent": False, "safe": False, "description": "Create resource"},
        "PATCH":   {"idempotent": False, "safe": False, "description": "Partial update"},
    }

    print(f"  {'Method':10s} {'Idempotent':12s} {'Safe':6s} {'Description'}")
    print(f"  {'-'*50}")
    for method, info in methods.items():
        idem = "Yes" if info["idempotent"] else "No"
        safe = "Yes" if info["safe"] else "No"
        print(f"  {method:10s} {idem:12s} {safe:6s} {info['description']}")


def exercise_2():
    """
    Problem 2: Status Codes
    Choose appropriate status codes for each situation.
    """
    scenarios = [
        ("User login failure (authentication)", 401, "Unauthorized",
         "Client must authenticate; 401 means 'who are you?'"),
        ("Page not found", 404, "Not Found",
         "The requested resource doesn't exist at this URL"),
        ("Internal server error", 500, "Internal Server Error",
         "Server encountered an unexpected condition"),
        ("Resource created via POST", 201, "Created",
         "Successfully created the new resource; typically includes Location header"),
    ]

    status_categories = {
        "1xx": "Informational (request received, processing)",
        "2xx": "Success (request accepted and processed)",
        "3xx": "Redirection (further action needed)",
        "4xx": "Client Error (bad request from client)",
        "5xx": "Server Error (server failed to fulfill valid request)",
    }

    print("HTTP Status Code Selection:")
    for scenario, code, name, explanation in scenarios:
        print(f"\n  Scenario: {scenario}")
        print(f"  Code: {code} {name}")
        print(f"  Reason: {explanation}")

    print("\n\nStatus code categories:")
    for category, description in status_categories.items():
        print(f"  {category}: {description}")


def exercise_3():
    """
    Problem 3: Headers
    - Difference between Cache-Control: no-cache and no-store.
    - Purpose of ETag header.
    """
    print("Cache-Control Directives:")
    print("\n  no-cache:")
    print("    - Response CAN be cached")
    print("    - BUT must revalidate with server before each use")
    print("    - Server may respond 304 Not Modified (use cached version)")
    print("    - Use case: frequently updated content")

    print("\n  no-store:")
    print("    - Response must NOT be cached at all")
    print("    - Every request must go to the origin server")
    print("    - Use case: sensitive data (banking, medical records)")

    print("\n\nETag Header:")
    print("  Purpose: Resource version identifier for conditional requests")
    print("  How it works:")
    print("    1. Server sends: ETag: \"abc123\" with response")
    print("    2. Client caches response with ETag")
    print("    3. On next request, client sends: If-None-Match: \"abc123\"")
    print("    4. If resource unchanged: Server returns 304 Not Modified (no body)")
    print("    5. If resource changed: Server returns 200 with new content + new ETag")
    print("  Benefit: Saves bandwidth by avoiding re-downloading unchanged content")


def exercise_4():
    """
    Problem 4: HTTP Versions
    - Explain HTTP/1.1 HOL Blocking.
    - How HTTP/2 solves it.
    """
    print("HTTP/1.1 Head-of-Line (HOL) Blocking:")
    print()
    print("  Problem: HTTP/1.1 processes requests sequentially per connection.")
    print("  If request #1 takes a long time, requests #2 and #3 must wait.")
    print()
    print("  Timeline (HTTP/1.1):")
    print("    Connection 1: [--Req 1 (slow)--][Req 2][Req 3]")
    print("    Connection 2: [Req 4][Req 5]                    <- browser opens 6 connections")
    print("                        ^^^ Req 2,3 blocked by slow Req 1")

    print("\n  HTTP/2 Solution: Multiplexing")
    print("    - Multiple streams over a SINGLE TCP connection")
    print("    - Requests and responses are interleaved as frames")
    print("    - No head-of-line blocking at HTTP level")
    print()
    print("  Timeline (HTTP/2):")
    print("    Stream 1: [Frame1a]      [Frame1b]      [Frame1c]")
    print("    Stream 2:         [Frame2a][Frame2b]")
    print("    Stream 3:                         [Frame3a][Frame3b]")
    print("                All share one TCP connection, no blocking")

    print("\n  Additional HTTP/2 features:")
    print("    - Header compression (HPACK)")
    print("    - Server push")
    print("    - Stream prioritization")
    print("    - Binary framing (more efficient than text)")


def exercise_5():
    """
    Problem 5: HTTPS/TLS
    - 3 security benefits of HTTPS.
    - Handshake RTT difference: TLS 1.2 vs TLS 1.3.
    """
    print("HTTPS Security Benefits:")
    benefits = [
        ("Confidentiality", "Data encrypted in transit; attackers cannot read content"),
        ("Integrity", "Data tampering detected; HMAC ensures no modification"),
        ("Authentication", "Server identity verified via certificate chain"),
    ]
    for i, (benefit, detail) in enumerate(benefits, 1):
        print(f"  {i}. {benefit}: {detail}")

    print("\n\nTLS Handshake Comparison:")
    print("\n  TLS 1.2: 2 RTTs for full handshake")
    print("    RTT 1: ClientHello -> ServerHello + Certificate + ServerKeyExchange")
    print("    RTT 2: ClientKeyExchange -> Finished")
    print("    Total: 2 RTTs before data can flow")

    print("\n  TLS 1.3: 1 RTT for full handshake")
    print("    RTT 1: ClientHello + KeyShare -> ServerHello + Certificate + Finished")
    print("    Data can flow after 1 RTT!")
    print("    0-RTT resumption possible for repeat connections")

    print("\n  Improvement: TLS 1.3 saves 1 RTT, reducing connection latency")
    print("  On a 100ms RTT link, that is 100ms faster to first byte.")


def exercise_6():
    """
    Problem 6: Practical curl analysis.
    """
    print("curl Command Analysis:")

    print("\n  1. curl -v http://example.com")
    print("     Request headers include:")
    print("       Host: example.com")
    print("       User-Agent: curl/x.xx")
    print("       Accept: */*")
    print("     -v flag shows verbose output including headers")

    print("\n  2. curl -I -X DELETE http://api.example.com/users/1")
    print("     -I: fetch headers only (HEAD-like)")
    print("     -X DELETE: use DELETE method")
    print("     Expected success status: 200 OK or 204 No Content")

    print("\n  3. curl -X POST http://api.example.com/users \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"name\": \"test\"}'")
    print("     Content-Type: application/json")
    print("     This tells the server the body is JSON-formatted")
    print("     Expected success status: 201 Created")


def exercise_7():
    """
    Problem 7: Certificate Chain
    Why doesn't the root CA sign server certificates directly?
    """
    print("Certificate Chain Architecture:")
    print()
    print("  Root CA")
    print("    |-- signs --> Intermediate CA")
    print("                    |-- signs --> Server Certificate")
    print()
    print("  Why use intermediate CAs?")
    reasons = [
        ("Root CA Protection", "Root CA private key stays offline in an HSM. "
         "If it signed every cert directly, it would need to be online constantly, "
         "increasing risk of compromise."),
        ("Damage Containment", "If an intermediate CA is compromised, only its "
         "certificates are revoked, not the entire trust chain."),
        ("Operational Flexibility", "Different intermediate CAs can handle different "
         "certificate types (DV, OV, EV) or regions."),
        ("Revocation Scope", "Revoking an intermediate affects a smaller set of "
         "certificates than revoking the root."),
    ]
    for reason, detail in reasons:
        print(f"  - {reason}: {detail}")


def exercise_8():
    """
    Problem 8: Security Headers
    Suggest headers to prevent:
    - Clickjacking, XSS, MIME sniffing
    """
    vulnerabilities = [
        {
            "attack": "Clickjacking",
            "header": "X-Frame-Options: DENY",
            "modern": "Content-Security-Policy: frame-ancestors 'none'",
            "explanation": "Prevents page from being embedded in an iframe",
        },
        {
            "attack": "XSS (Cross-Site Scripting)",
            "header": "Content-Security-Policy: default-src 'self'",
            "modern": "CSP with strict-dynamic or nonce-based policy",
            "explanation": "Controls which scripts can execute on the page",
        },
        {
            "attack": "MIME sniffing",
            "header": "X-Content-Type-Options: nosniff",
            "modern": "Same (still used)",
            "explanation": "Prevents browsers from guessing content type",
        },
    ]

    print("Security Headers for Common Vulnerabilities:")
    for v in vulnerabilities:
        print(f"\n  Attack: {v['attack']}")
        print(f"  Header: {v['header']}")
        print(f"  Modern: {v['modern']}")
        print(f"  Effect: {v['explanation']}")

    print("\n  Additional recommended headers:")
    print("    Strict-Transport-Security: max-age=31536000; includeSubDomains")
    print("    Referrer-Policy: strict-origin-when-cross-origin")
    print("    Permissions-Policy: camera=(), microphone=()")


if __name__ == "__main__":
    exercises = [
        exercise_1, exercise_2, exercise_3, exercise_4,
        exercise_5, exercise_6, exercise_7, exercise_8,
    ]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
