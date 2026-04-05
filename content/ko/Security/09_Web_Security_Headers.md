# Web Security Headers와 CSP

**이전**: [08. Injection 공격과 방어](./08_Injection_Attacks.md) | **다음**: [10_API_Security.md](./10_API_Security.md)

---

HTTP 보안 헤더는 웹 애플리케이션의 첫 번째 방어선입니다. 브라우저가 보안 정책을 적용하도록 지시하여 크로스 사이트 스크립팅과 클릭재킹부터 프로토콜 다운그레이드 공격 및 데이터 유출까지 전체 공격 유형을 완화합니다. 단일 헤더가 누락되면 잘 작성된 애플리케이션도 취약해질 수 있습니다. 이 레슨은 Flask와 Django의 실용적인 구성 예제와 함께 모든 주요 보안 헤더에 대한 포괄적인 가이드를 제공합니다.

## 학습 목표

- Content-Security-Policy (CSP)의 목적과 지시문 이해
- HTTPS 연결을 강제하는 HSTS 구성
- X-Content-Type-Options, X-Frame-Options, Referrer-Policy 헤더 적용
- 브라우저 기능을 제한하는 Permissions-Policy 구현
- Cross-Origin 정책 (CORP, COEP, COOP) 구성
- Subresource Integrity (SRI)를 사용하여 외부 리소스 확인
- Flask 및 Django 애플리케이션에서 보안 헤더 설정
- 명령줄 도구 및 스캐너를 사용하여 헤더 테스트 및 감사

---

## 1. 보안 헤더 개요

### 1.1 보안 헤더가 중요한 이유

```
┌─────────────────────────────────────────────────────────────────────┐
│                  HTTP Response 보안 헤더                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Server ──── HTTP Response ────▶ Browser                             │
│              │                                                       │
│              ├── Content-Security-Policy                              │
│              ├── Strict-Transport-Security                            │
│              ├── X-Content-Type-Options                               │
│              ├── X-Frame-Options                                      │
│              ├── Referrer-Policy                                      │
│              ├── Permissions-Policy                                   │
│              ├── Cross-Origin-Resource-Policy                         │
│              ├── Cross-Origin-Embedder-Policy                         │
│              └── Cross-Origin-Opener-Policy                           │
│                                                                      │
│  이 헤더들은 브라우저에 다음을 지시합니다:                             │
│  • 인라인 스크립트 및 무단 리소스 차단 (CSP)                          │
│  • 항상 HTTPS 사용 (HSTS)                                            │
│  • MIME 타입 스니핑 방지 (X-Content-Type-Options)                    │
│  • 프레이밍 / 클릭재킹 차단 (X-Frame-Options)                        │
│  • Referrer 정보 유출 제어 (Referrer-Policy)                         │
│  • 브라우저 API 접근 제한 (Permissions-Policy)                       │
│  • Cross-origin 리소스 격리 (CORP/COEP/COOP)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 심층 방어

보안 헤더는 안전한 코딩 관행을 대체하는 것이 아니라 추가 계층입니다. 완벽하게 코딩된 애플리케이션이라도 보안 헤더는 다음에 대한 보호를 제공합니다:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    심층 방어 계층                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 5:  보안 헤더             (브라우저 강제 정책)                 │
│  Layer 4:  애플리케이션 로직      (입력 검증, 인증)                   │
│  Layer 3:  프레임워크 보호        (CSRF 토큰, ORM)                    │
│  Layer 2:  네트워크 보안          (TLS, 방화벽, WAF)                  │
│  Layer 1:  인프라                (OS 강화, 패치)                      │
│                                                                      │
│  각 계층은 하위 계층이 놓칠 수 있는 것을 포착합니다.                   │
│  보안 헤더는 애플리케이션 코드가 놓칠 수 있는 것을 포착합니다.         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 빠른 참조 테이블

| 헤더 | 완화 대상 | 일반적인 값 |
|--------|-----------|---------------|
| Content-Security-Policy | XSS, 데이터 주입 | `default-src 'self'` |
| Strict-Transport-Security | 프로토콜 다운그레이드, 쿠키 가로채기 | `max-age=31536000; includeSubDomains` |
| X-Content-Type-Options | MIME 타입 혼동 | `nosniff` |
| X-Frame-Options | 클릭재킹 | `DENY` |
| Referrer-Policy | 정보 유출 | `strict-origin-when-cross-origin` |
| Permissions-Policy | 무단 API 접근 | `camera=(), microphone=()` |
| Cross-Origin-Resource-Policy | Cross-origin 데이터 유출 | `same-origin` |
| Cross-Origin-Embedder-Policy | Spectre 스타일 공격 | `require-corp` |
| Cross-Origin-Opener-Policy | Cross-window 공격 | `same-origin` |

---

## 2. Content-Security-Policy (CSP)

### 2.1 CSP란 무엇인가?

Content-Security-Policy는 가장 강력한 보안 헤더입니다. 브라우저가 신뢰해야 하는 콘텐츠 소스의 허용 목록을 정의하여 무단 스크립트 실행을 차단함으로써 XSS 공격을 효과적으로 방지합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CSP 강제 모델                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  브라우저가 CSP 헤더를 수신:                                          │
│    Content-Security-Policy: default-src 'self'; script-src 'self'    │
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────┐                       │
│  │ <script src=      │     │ <script>          │                      │
│  │  "/app.js">       │     │  alert('XSS')     │                      │
│  │                   │     │ </script>          │                      │
│  │  출처: self       │     │  출처: inline     │                      │
│  │  ✓ 허용           │     │  ✗ 차단           │                      │
│  └──────────────────┘     └──────────────────┘                       │
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────┐                       │
│  │ <script src=      │     │ <img src=          │                      │
│  │  "https://cdn     │     │  "/logo.png">      │                      │
│  │  .example.com     │     │                    │                      │
│  │  /lib.js">        │     │  출처: self        │                      │
│  │  출처: cdn        │     │  ✓ 허용            │                      │
│  │  ✗ 차단           │     │                    │                      │
│  └──────────────────┘     └──────────────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 CSP 지시문

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CSP 지시문 카테고리                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Fetch 지시문 (리소스 로드 위치 제어):                                │
│  ├── default-src     다른 모든 fetch 지시문의 대체              │
│  ├── script-src      JavaScript 소스                                 │
│  ├── style-src       CSS 소스                                        │
│  ├── img-src         이미지 소스                                     │
│  ├── font-src        웹 폰트 소스                                    │
│  ├── connect-src     XHR, fetch, WebSocket, EventSource              │
│  ├── media-src       오디오/비디오 소스                              │
│  ├── object-src      <object>, <embed>, <applet>                     │
│  ├── child-src       웹 워커 및 중첩 컨텍스트                        │
│  ├── worker-src      Worker, SharedWorker, ServiceWorker             │
│  └── manifest-src    앱 매니페스트                                   │
│                                                                      │
│  문서 지시문:                                                         │
│  ├── base-uri        <base> 요소 제한                                │
│  ├── sandbox         샌드박스 제한 적용                              │
│  └── plugin-types    플러그인 MIME 타입 제한 (deprecated)            │
│                                                                      │
│  탐색 지시문:                                                         │
│  ├── form-action     폼 제출 대상 제한                                │
│  ├── frame-ancestors 이 페이지를 임베드할 수 있는 대상 제한          │
│  └── navigate-to     탐색 대상 제한 (제한적 지원)                    │
│                                                                      │
│  보고 지시문:                                                         │
│  ├── report-uri      위반 보고 전송 (deprecated)                     │
│  └── report-to       위반 보고 전송 (최신)                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 소스 값

| 소스 | 의미 | 예제 |
|--------|---------|---------|
| `'self'` | 동일 출처 (스킴 + 호스트 + 포트) | `script-src 'self'` |
| `'none'` | 모든 소스 차단 | `object-src 'none'` |
| `'unsafe-inline'` | 인라인 스크립트/스타일 허용 | `style-src 'unsafe-inline'` |
| `'unsafe-eval'` | eval(), Function() 등 허용 | `script-src 'unsafe-eval'` |
| `'strict-dynamic'` | 신뢰하는 스크립트로 로드된 스크립트 신뢰 | `script-src 'strict-dynamic'` |
| `'nonce-<base64>'` | nonce로 특정 인라인 허용 | `script-src 'nonce-abc123'` |
| `'sha256-<hash>'` | 해시로 특정 인라인 허용 | `script-src 'sha256-...'` |
| `https:` | 모든 HTTPS 소스 | `img-src https:` |
| `data:` | Data URI | `img-src data:` |
| `blob:` | Blob URI | `worker-src blob:` |
| `*.example.com` | 와일드카드 서브도메인 | `script-src *.cdn.com` |

### 2.4 CSP 정책 단계별 구축

```python
"""
CSP 정책을 점진적으로 구축 — 허용적인 것부터 엄격한 것까지.
"""

# ── 레벨 1: 기본 CSP (명백한 XSS 차단) ──────────────────────
# default-src 'self'로 시작하고 필요에 따라 완화
csp_level1 = "default-src 'self'"

# ── 레벨 2: 특정 외부 리소스 허용 ───────────────────
csp_level2 = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: https:; "
    "connect-src 'self' https://api.example.com; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

# ── 레벨 3: Nonce 기반 CSP (최신 앱 권장) ──────
# 서버가 요청당 랜덤 nonce 생성
import secrets

def generate_csp_with_nonce():
    nonce = secrets.token_urlsafe(32)
    csp = (
        f"default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        f"style-src 'self' 'nonce-{nonce}'; "
        f"img-src 'self' data:; "
        f"font-src 'self'; "
        f"connect-src 'self'; "
        f"object-src 'none'; "
        f"base-uri 'self'; "
        f"form-action 'self'; "
        f"frame-ancestors 'none'"
    )
    return csp, nonce

csp_header, nonce = generate_csp_with_nonce()
# HTML에서: <script nonce="<nonce>">...</script>

# ── 레벨 4: 해시 기반 CSP (정적 인라인 스크립트용) ─────────
import hashlib
import base64

def compute_csp_hash(script_content: str) -> str:
    """CSP 인라인 스크립트 허용 목록을 위한 SHA-256 해시 계산."""
    digest = hashlib.sha256(script_content.encode('utf-8')).digest()
    b64 = base64.b64encode(digest).decode('utf-8')
    return f"'sha256-{b64}'"

inline_script = "console.log('Hello, World!');"
script_hash = compute_csp_hash(inline_script)
print(f"CSP 해시: {script_hash}")
# 출력: CSP 해시: 'sha256-TWupyvVdPa1DyFqLnQMqRpuUWdS3nKPnz70IcS/1o3Q='

csp_level4 = f"script-src 'self' {script_hash}"

# ── 레벨 5: strict-dynamic (스크립트를 동적으로 로드하는 앱용)
csp_level5 = (
    "script-src 'strict-dynamic' 'nonce-{nonce}'; "
    "object-src 'none'; "
    "base-uri 'self'"
)
# strict-dynamic을 사용하면 nonce가 있는 스크립트로 로드된 스크립트는
# 출처에 관계없이 자동으로 신뢰됩니다.
```

### 2.5 CSP 보고

```python
"""
CSP 위반 보고 — 차단 없이 정책 위반 감지.
"""

# ── Report-Only 모드 (강제 없이 모니터링) ───────────────
# 먼저 Content-Security-Policy-Report-Only 헤더 사용
csp_report_only = (
    "default-src 'self'; "
    "script-src 'self'; "
    "report-uri /csp-report; "
    "report-to csp-endpoint"
)

# Report-To 헤더 (report-to 지시문의 동반자)
report_to = {
    "group": "csp-endpoint",
    "max_age": 86400,
    "endpoints": [
        {"url": "https://example.com/csp-report"}
    ]
}

# ── CSP 위반 보고를 받을 Flask 엔드포인트 ──────────────
from flask import Flask, request, jsonify
import json
import logging

app = Flask(__name__)
logger = logging.getLogger('csp_reports')

@app.route('/csp-report', methods=['POST'])
def csp_report():
    """CSP 위반 보고 수신 및 로깅."""
    try:
        # CSP 보고는 application/csp-report로 전송됨
        report = request.get_json(force=True)
        violation = report.get('csp-report', {})

        logger.warning(
            "CSP 위반: blocked_uri=%s, "
            "violated_directive=%s, "
            "document_uri=%s, "
            "source_file=%s, "
            "line_number=%s",
            violation.get('blocked-uri', 'N/A'),
            violation.get('violated-directive', 'N/A'),
            violation.get('document-uri', 'N/A'),
            violation.get('source-file', 'N/A'),
            violation.get('line-number', 'N/A'),
        )

        return jsonify({"status": "received"}), 204
    except Exception as e:
        logger.error(f"CSP 보고 처리 오류: {e}")
        return jsonify({"error": "invalid report"}), 400

# ── CSP 위반 보고 JSON 예제 ────────────────────────────
example_report = {
    "csp-report": {
        "document-uri": "https://example.com/page",
        "referrer": "",
        "violated-directive": "script-src 'self'",
        "effective-directive": "script-src",
        "original-policy": "default-src 'self'; script-src 'self'",
        "blocked-uri": "https://evil.com/malicious.js",
        "status-code": 200,
        "source-file": "https://example.com/page",
        "line-number": 15,
        "column-number": 2
    }
}
```

### 2.6 일반적인 CSP 실수

```python
"""
일반적인 CSP 실수와 수정 방법.
"""

# ── 실수 1: script-src에 'unsafe-inline' 사용 ─────────────
# 이것은 XSS 방지를 위한 CSP의 목적을 무효화함
bad_csp = "script-src 'self' 'unsafe-inline'"
# 수정: 대신 nonce나 해시 사용
good_csp = "script-src 'self' 'nonce-{random}'"

# ── 실수 2: script-src에 와일드카드 ────────────────────────────
# 모든 서브도메인에서 스크립트 로드 허용
bad_csp = "script-src 'self' *.googleapis.com"
# 수정: 특정 호스트명 사용
good_csp = "script-src 'self' https://ajax.googleapis.com"

# ── 실수 3: default-src 누락 ───────────────────────────────
# default-src 없으면 나열되지 않은 지시문은 기본적으로 모두 허용
bad_csp = "script-src 'self'"
# 수정: 항상 대체로 default-src 포함
good_csp = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'"

# ── 실수 4: script-src에서 data: 허용 ──────────────────────
# data: URI는 실행 가능한 JavaScript를 포함할 수 있음
bad_csp = "script-src 'self' data:"
# 수정: 필요한 곳에만 이미지/폰트에 data: 사용
good_csp = "script-src 'self'; img-src 'self' data:"

# ── 실수 5: object-src 잊음 ────────────────────────────
# Flash와 Java 애플릿은 스크립트 실행 가능
bad_csp = "default-src 'self'"  # object-src가 'self'로 대체됨
# 수정: object-src를 명시적으로 차단
good_csp = "default-src 'self'; object-src 'none'"

# ── 실수 6: 지나치게 허용적인 connect-src ────────────────────
# 모든 HTTPS 엔드포인트로 데이터 유출 허용
bad_csp = "connect-src https:"
# 수정: 특정 API 엔드포인트 나열
good_csp = "connect-src 'self' https://api.example.com"
```

---

## 3. Strict-Transport-Security (HSTS)

### 3.1 HSTS 작동 방식

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HSTS 보호 흐름                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  HSTS 없이:                                                          │
│  User ──http://──▶ Server ──301 Redirect──▶ https://                │
│        ↑                                                             │
│        └── MITM이 이 평문 HTTP 요청을 가로챌 수 있음                 │
│                                                                      │
│  HSTS와 함께:                                                        │
│  User ──http://──▶ 브라우저가 가로챔 (307 내부 리디렉트)             │
│                    Browser ──https://──▶ Server                      │
│                    (HTTP로 네트워크 요청 없음)                        │
│                                                                      │
│  첫 방문:                                                            │
│  1. 브라우저가 HTTP로 연결                                           │
│  2. 서버가 301 + HSTS 헤더로 응답                                    │
│  3. 브라우저가 도메인에 대한 HSTS 정책 저장                          │
│                                                                      │
│  이후 방문:                                                          │
│  1. 브라우저가 자동으로 HTTPS로 업그레이드                           │
│  2. HTTP 요청이 브라우저를 떠나지 않음                               │
│  3. 잘못된 인증서는 하드 실패 유발 (우회 없음)                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 HSTS 지시문

```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
```

| 지시문 | 목적 | 권장 값 |
|-----------|---------|-------------------|
| `max-age` | HSTS를 기억할 기간 (초) | `31536000` (1년) |
| `includeSubDomains` | 모든 서브도메인에 적용 | 완전한 보호를 위해 포함 |
| `preload` | 브라우저 사전 로드 목록에 포함 요청 | 테스트 후 포함 |

```python
"""
HSTS 구성 고려 사항.
"""

# ── 배포 전략: 점진적 롤아웃 ─────────────────────────
# 짧은 max-age로 시작하고 시간이 지남에 따라 증가

# 1단계: 5분으로 테스트
hsts_test = "max-age=300"

# 2단계: 1주일로 증가
hsts_week = "max-age=604800"

# 3단계: 서브도메인과 함께 1개월로 증가
hsts_month = "max-age=2592000; includeSubDomains"

# 4단계: 전체 배포 (1년 + preload)
hsts_full = "max-age=31536000; includeSubDomains; preload"

# 경고: 모든 서브도메인이 HTTPS를 지원하는지 확인하기 전에
# 긴 max-age를 설정하면 사용자를 잠글 수 있습니다.
# includeSubDomains는 모든 서브도메인이 유효한 TLS를 가져야 함을 의미합니다.

# ── HSTS Preload 목록 ───────────────────────────────────────────
# HSTS preload 목록은 브라우저에 내장되어 있습니다.
# 이 목록의 도메인은 첫 방문에도 항상 HTTPS를 사용합니다.
# 제출: https://hstspreload.org/
#
# Preload 요구사항:
# 1. 유효한 인증서
# 2. 동일 호스트에서 모든 HTTP를 HTTPS로 리디렉트
# 3. 다음을 포함하는 HSTS 헤더:
#    - max-age >= 31536000 (1년)
#    - includeSubDomains
#    - preload
# 4. HTTPS 리디렉트도 HSTS 헤더를 포함해야 함
```

---

## 4. X-Content-Type-Options

### 4.1 MIME 스니핑 공격

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MIME 스니핑 공격                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  공격자가 업로드: malicious.jpg                                      │
│  (실제로는 이미지 데이터가 아닌 JavaScript 포함)                     │
│                                                                      │
│  서버가 제공: Content-Type: image/jpeg                               │
│                                                                      │
│  nosniff 없이:                                                       │
│  브라우저가 콘텐츠 스니핑 ──▶ "이것은 JavaScript처럼 보임"           │
│  브라우저가 파일을 JavaScript로 실행 ──▶ XSS!                        │
│                                                                      │
│  nosniff와 함께:                                                     │
│  브라우저가 Content-Type 신뢰 ──▶ "이것은 image/jpeg"                │
│  브라우저가 이미지로 렌더링 (실패) ──▶ 스크립트 실행 없음             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 구성

```
X-Content-Type-Options: nosniff
```

이것은 가장 간단한 보안 헤더입니다 — 정확히 하나의 유효한 값 `nosniff`만 있습니다. 모든 응답에 포함하지 않을 이유가 없습니다. 다음을 방지합니다:

- 스크립트가 아닌 MIME 타입의 파일에서 스크립트가 로드되는 것
- 스타일시트가 CSS가 아닌 MIME 타입의 파일에서 로드되는 것
- 사용자 업로드 콘텐츠를 제공할 때 MIME 타입 혼동 공격

---

## 5. X-Frame-Options와 frame-ancestors

### 5.1 클릭재킹 방지

```
┌─────────────────────────────────────────────────────────────────────┐
│                    클릭재킹 공격                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  공격자의 페이지:                                                    │
│  ┌─────────────────────────────────────────┐                         │
│  │  "여기를 클릭하면 상품을 받으세요!"     │                         │
│  │                                          │                        │
│  │  ┌─────────────────────────────────┐     │                        │
│  │  │  보이지 않는 iframe (opacity: 0)│     │                        │
│  │  │  ┌────────────────────────┐     │     │                        │
│  │  │  │ 귀하의 은행 앱         │     │     │                        │
│  │  │  │                        │     │     │                        │
│  │  │  │  [1000원 송금] ◄────┼─────┼──── 사용자가 여기를 클릭   │
│  │  │  │                        │     │     │                        │
│  │  │  └────────────────────────┘     │     │                        │
│  │  └─────────────────────────────────┘     │                        │
│  └─────────────────────────────────────────┘                         │
│                                                                      │
│  사용자는 상품 버튼을 클릭한다고 생각하지만,                         │
│  실제로는 은행 웹사이트가 포함된 보이지 않는 iframe의                │
│  "송금" 버튼을 클릭하고 있습니다.                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 X-Frame-Options 값

```
# 모든 프레이밍 거부
X-Frame-Options: DENY

# 동일 출처만 허용
X-Frame-Options: SAMEORIGIN

# 특정 출처 허용 (deprecated, 권장하지 않음)
X-Frame-Options: ALLOW-FROM https://trusted.example.com
```

### 5.3 CSP frame-ancestors (최신 대체)

```python
"""
frame-ancestors는 X-Frame-Options의 CSP 대체입니다.
더 세밀한 제어를 제공하고 여러 출처를 지원합니다.
"""

# ── 모든 프레이밍 차단 (X-Frame-Options: DENY와 동등) ─────
csp_no_frame = "frame-ancestors 'none'"

# ── 동일 출처만 허용 ───────────────────────────────────────
csp_same_origin = "frame-ancestors 'self'"

# ── 특정 출처 허용 ───────────────────────────────────────
csp_specific = "frame-ancestors 'self' https://trusted.example.com https://partner.example.com"

# ── X-Frame-Options와의 주요 차이점 ────────────────────────
#
# | 기능                | X-Frame-Options      | frame-ancestors        |
# |---------------------|----------------------|------------------------|
# | 다중 출처           | 아니오               | 예                     |
# | 와일드카드 서브도메인| 아니오               | 예 (*.example.com)     |
# | 스킴 제한           | 아니오               | 예 (https:)            |
# | CSP 통합            | 별도 헤더            | CSP의 일부             |
# | 브라우저 지원       | 범용                 | 최신 브라우저          |
#
# 권장사항: 최대 호환성을 위해 둘 다 설정
# X-Frame-Options: DENY
# Content-Security-Policy: frame-ancestors 'none'
```

---

## 6. Referrer-Policy

### 6.1 Referrer 정보 유출

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Referrer 유출 시나리오                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  사용자가 다음에 있음: https://bank.com/accounts/12345/transfer?amount=500 │
│                                                                      │
│  링크를 클릭: https://analytics.example.com                          │
│                                                                      │
│  Referrer-Policy 없이:                                               │
│  Referer: https://bank.com/accounts/12345/transfer?amount=500        │
│  ↑ 경로와 쿼리 매개변수를 포함한 전체 URL 유출!                      │
│                                                                      │
│  Referrer-Policy: strict-origin-when-cross-origin과 함께              │
│  Referer: https://bank.com                                           │
│  ↑ 출처만 전송, 민감한 경로나 매개변수 없음                          │
│                                                                      │
│  Referrer-Policy: no-referrer와 함께                                 │
│  Referer: (비어 있음)                                                │
│  ↑ referrer 정보가 전혀 전송되지 않음                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 정책 값

| 정책 | 동일 출처 | Cross-Origin (HTTPS→HTTPS) | 다운그레이드 (HTTPS→HTTP) |
|--------|-------------|---------------------------|----------------------|
| `no-referrer` | 없음 | 없음 | 없음 |
| `no-referrer-when-downgrade` | 전체 URL | 전체 URL | 없음 |
| `origin` | 출처만 | 출처만 | 출처만 |
| `origin-when-cross-origin` | 전체 URL | 출처만 | 출처만 |
| `same-origin` | 전체 URL | 없음 | 없음 |
| `strict-origin` | 출처만 | 출처만 | 없음 |
| `strict-origin-when-cross-origin` | 전체 URL | 출처만 | 없음 |
| `unsafe-url` | 전체 URL | 전체 URL | 전체 URL |

```python
"""
권장 Referrer-Policy 구성.
"""

# ── 기본 권장사항 ──────────────────────────────────────
# strict-origin-when-cross-origin은 최신 브라우저의 기본값이며
# 기능과 개인정보 보호의 좋은 균형을 제공합니다
referrer_policy = "strict-origin-when-cross-origin"

# ── 최대 개인정보 보호 ─────────────────────────────────────────
# no-referrer는 모든 referrer 정보를 제거합니다
# 단점: 분석 및 일부 CSRF 보호를 손상시킵니다
referrer_max_privacy = "no-referrer"

# ── 민감한 URL이 있는 사이트용 ───────────────────────────────
# same-origin은 동일 사이트 내에서만 전체 referrer 전송
referrer_sensitive = "same-origin"

# ── 요소별 재정의 ────────────────────────────────────────
# 개별 요소에도 referrer 정책을 설정할 수 있습니다:
# <a href="..." referrerpolicy="no-referrer">외부 링크</a>
# <img src="..." referrerpolicy="no-referrer">
# <script src="..." referrerpolicy="no-referrer">
```

---

## 7. Permissions-Policy

### 7.1 브라우저 기능 제한

Permissions-Policy (이전 Feature-Policy)는 페이지와 포함된 콘텐츠가 사용할 수 있는 브라우저 기능 및 API를 제어합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Permissions-Policy                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  브라우저 API 접근 제어:                                             │
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │  Camera    │  │ Microphone │  │ Geolocation│  │  Payment   │    │
│  │  camera    │  │ microphone │  │ geolocation│  │  payment   │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ Fullscreen │  │  Autoplay  │  │  USB       │  │  Bluetooth │    │
│  │ fullscreen │  │  autoplay  │  │  usb       │  │  bluetooth │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
│                                                                      │
│  문법:                                                               │
│  Permissions-Policy: feature=(allowlist)                             │
│                                                                      │
│  허용 목록 값:                                                       │
│  *            = 모든 출처 허용                                       │
│  self         = 동일 출처만 허용                                     │
│  (비어 있음)  = ()는 완전히 차단을 의미                              │
│  "origin"     = 특정 출처 허용                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 구성 예제

```python
"""
Permissions-Policy 헤더 구성.
"""

# ── 제한적 정책 (대부분의 사이트에 권장) ─────────────
permissions_policy = (
    "camera=(), "                              # 카메라 차단
    "microphone=(), "                          # 마이크 차단
    "geolocation=(), "                         # 위치정보 차단
    "payment=(), "                             # Payment API 차단
    "usb=(), "                                 # WebUSB 차단
    "bluetooth=(), "                           # Web Bluetooth 차단
    "magnetometer=(), "                        # 자력계 차단
    "gyroscope=(), "                           # 자이로스코프 차단
    "accelerometer=(), "                       # 가속도계 차단
    'autoplay=(self), '                        # 동일 출처에서만 자동재생 허용
    'fullscreen=(self), '                      # 동일 출처에서만 전체화면 허용
    'picture-in-picture=(self)'                # 동일 출처에서만 PiP 허용
)

# ── 화상 회의 앱용 정책 ─────────────────────────────
permissions_video_app = (
    'camera=(self "https://meet.example.com"), '
    'microphone=(self "https://meet.example.com"), '
    "geolocation=(), "
    'fullscreen=(self), '
    'display-capture=(self)'
)

# ── 전자상거래 사이트용 정책 ───────────────────────────────
permissions_ecommerce = (
    "camera=(), "
    "microphone=(), "
    'geolocation=(self), '                     # 매장 위치 찾기용
    'payment=(self), '                         # Payment Request API용
    "usb=(), "
    "bluetooth=()"
)
```

---

## 8. Cross-Origin 정책 (CORP, COEP, COOP)

### 8.1 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Cross-Origin 격리                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  세 개의 헤더가 함께 cross-origin 격리를 위해 작동:                  │
│                                                                      │
│  CORP (Cross-Origin-Resource-Policy)                                 │
│  ├── 리소스(이미지, 스크립트 등)에 설정                              │
│  ├── 이 리소스를 로드할 수 있는 대상 제어                            │
│  └── 값: same-site, same-origin, cross-origin                        │
│                                                                      │
│  COEP (Cross-Origin-Embedder-Policy)                                 │
│  ├── 리소스를 포함하는 문서에 설정                                   │
│  ├── 모든 리소스가 옵트인(CORP 또는 CORS를 통해)해야 함              │
│  └── 값: unsafe-none, require-corp, credentialless                   │
│                                                                      │
│  COOP (Cross-Origin-Opener-Policy)                                   │
│  ├── 문서에 설정                                                     │
│  ├── window.opener 관계 제어                                         │
│  └── 값: unsafe-none, same-origin, same-origin-allow-popups          │
│                                                                      │
│  COEP: require-corp + COOP: same-origin이 모두 설정되면:             │
│  ──▶ 페이지가 "cross-origin isolated"                                │
│  ──▶ SharedArrayBuffer, 고해상도 타이머 활성화                       │
│  ──▶ Spectre 스타일 사이드 채널 공격으로부터 보호                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Cross-Origin-Resource-Policy (CORP)

```python
"""
CORP는 리소스를 로드할 수 있는 출처를 제어합니다.
API 응답, 이미지, 스크립트 등에 설정합니다.
"""

# ── 값 ───────────────────────────────────────────────────────
# same-origin:  동일 출처의 요청만
# same-site:    동일 사이트의 요청 (서브도메인 포함)
# cross-origin: 모든 출처가 이 리소스를 로드 가능

# 비공개 API 엔드포인트용
corp_api = "same-origin"

# 공개 CDN 리소스용
corp_cdn = "cross-origin"

# 서브도메인 간 공유되는 리소스용
corp_shared = "same-site"
```

### 8.3 Cross-Origin-Embedder-Policy (COEP)

```python
"""
COEP는 페이지에 로드된 모든 리소스가
cross-origin으로 로드되는 것에 명시적으로 옵트인했는지 확인합니다.
"""

# ── require-corp: 가장 엄격한 모드 ────────────────────────────────
# 모든 cross-origin 리소스는 다음 중 하나를 수행해야 함:
# 1. CORP: cross-origin 헤더와 함께 제공
# 2. crossorigin 속성으로 로드 (CORS)
coep_strict = "require-corp"

# HTML에서 리소스에 crossorigin 속성 필요:
# <img src="https://cdn.example.com/image.jpg" crossorigin>
# <script src="https://cdn.example.com/lib.js" crossorigin>

# ── credentialless: 더 실용적 ──────────────────────────────
# Cross-origin 요청이 자격 증명(쿠키) 없이 이루어짐
# 리소스에 CORP 헤더 불필요
coep_credentialless = "credentialless"

# ── unsafe-none: 제한 없음 (기본값) ───────────────────────
coep_none = "unsafe-none"
```

### 8.4 Cross-Origin-Opener-Policy (COOP)

```python
"""
COOP는 페이지와 opener 간의 관계를 제어합니다
(window.open 또는 링크를 통해 연 페이지).
"""

# ── same-origin: 완전 격리 ─────────────────────────────────
# cross-origin 페이지에 대한 window.opener 참조 끊음
# cross-origin 페이지가 이 창에 접근하는 것을 방지
coop_strict = "same-origin"

# ── same-origin-allow-popups ────────────────────────────────
# same-origin과 동일하지만, 이 페이지에서 연 팝업은
# opener 참조를 유지할 수 있음
coop_popups = "same-origin-allow-popups"

# ── unsafe-none: 제한 없음 (기본값) ───────────────────────
coop_none = "unsafe-none"

# ── cross-origin 격리 달성 ────────────────────────────────
# 두 헤더 모두 필요:
# Cross-Origin-Embedder-Policy: require-corp
# Cross-Origin-Opener-Policy: same-origin
#
# JavaScript에서 확인:
# if (crossOriginIsolated) {
#   // SharedArrayBuffer 사용 가능
#   // Performance.now()가 완전한 정밀도 제공
# }
```

---

## 9. Subresource Integrity (SRI)

### 9.1 SRI 작동 방식

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Subresource Integrity (SRI)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  문제: CDN 침해                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                       │
│  │ Your App │───▶│   CDN    │───▶│ Browser  │                       │
│  └──────────┘    └──────────┘    └──────────┘                       │
│                       ↑                                              │
│                   공격자가 CDN의                                     │
│                   jquery.min.js를                                    │
│                   수정                                               │
│                                                                      │
│  해결책: SRI 해시 확인                                               │
│  <script src="https://cdn/jquery.js"                                │
│          integrity="sha384-abc123..."                                │
│          crossorigin="anonymous">                                    │
│  </script>                                                           │
│                                                                      │
│  브라우저가 파일 다운로드 ──▶ 해시 계산 ──▶ integrity와 비교         │
│  일치?    ✓ 실행     ✗ 차단 및 오류 보고                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 SRI 해시 생성

```bash
# 로컬 파일에서 SRI 해시 생성
cat jquery-3.7.1.min.js | openssl dgst -sha384 -binary | openssl base64 -A
# 출력: oQVuAfEn...

# shasum을 사용하여 SRI 해시 생성
shasum -b -a 384 jquery-3.7.1.min.js | awk '{ print $1 }' | xxd -r -p | base64

# curl을 사용하여 원격 파일에서 해시 생성
curl -s https://code.jquery.com/jquery-3.7.1.min.js | \
    openssl dgst -sha384 -binary | openssl base64 -A
```

```python
"""
Python에서 SRI 해시 생성 및 확인.
"""
import hashlib
import base64
import requests

def generate_sri_hash(content: bytes, algorithm: str = 'sha384') -> str:
    """주어진 콘텐츠에 대한 SRI 해시 생성."""
    hash_func = getattr(hashlib, algorithm)
    digest = hash_func(content).digest()
    b64 = base64.b64encode(digest).decode('utf-8')
    return f"{algorithm}-{b64}"

def generate_sri_from_url(url: str) -> str:
    """리소스를 다운로드하고 SRI 해시 생성."""
    response = requests.get(url)
    response.raise_for_status()
    return generate_sri_hash(response.content)

# 사용 예제
url = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
sri_hash = generate_sri_from_url(url)
print(f'<link rel="stylesheet" href="{url}" '
      f'integrity="{sri_hash}" crossorigin="anonymous">')

# ── HTML에서 SRI ──────────────────────────────────────────────
# 스크립트용:
# <script src="https://cdn.example.com/lib.js"
#         integrity="sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8wC"
#         crossorigin="anonymous"></script>
#
# 스타일시트용:
# <link rel="stylesheet"
#       href="https://cdn.example.com/style.css"
#       integrity="sha384-abc123..."
#       crossorigin="anonymous">
#
# 다중 해시 (마이그레이션용):
# <script src="https://cdn.example.com/lib.js"
#         integrity="sha384-oldHash... sha384-newHash..."
#         crossorigin="anonymous"></script>
# 브라우저는 어떤 해시든 일치하면 허용

# ── 중요 참고사항 ──────────────────────────────────────────────
# 1. cross-origin SRI에는 crossorigin="anonymous" 필수
# 2. SRI는 <script>와 <link rel="stylesheet">에만 작동
# 3. 해시는 바이트 단위로 일치해야 함 (공백도 중요)
# 4. CDN이 파일을 업데이트하면 해시가 깨짐 (의도된 동작)
```

---

## 10. Flask 보안 헤더 구성

### 10.1 수동 헤더 설정

```python
"""
Flask 애플리케이션에서 보안 헤더 설정.
"""
from flask import Flask, request, make_response, g
import secrets

app = Flask(__name__)

# ── 방법 1: after_request 데코레이터 ────────────────────────────
@app.after_request
def set_security_headers(response):
    """모든 응답에 보안 헤더 추가."""

    # CSP용 nonce 생성
    nonce = getattr(g, 'csp_nonce', secrets.token_urlsafe(32))

    # Content-Security-Policy
    response.headers['Content-Security-Policy'] = (
        f"default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        f"style-src 'self' 'nonce-{nonce}' https://fonts.googleapis.com; "
        f"font-src 'self' https://fonts.gstatic.com; "
        f"img-src 'self' data:; "
        f"connect-src 'self'; "
        f"object-src 'none'; "
        f"base-uri 'self'; "
        f"form-action 'self'; "
        f"frame-ancestors 'none'"
    )

    # HSTS (프로덕션에서 HTTPS를 통해서만 설정)
    response.headers['Strict-Transport-Security'] = (
        'max-age=31536000; includeSubDomains; preload'
    )

    # MIME 스니핑 방지
    response.headers['X-Content-Type-Options'] = 'nosniff'

    # 클릭재킹 보호
    response.headers['X-Frame-Options'] = 'DENY'

    # Referrer 제어
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

    # Permissions 정책
    response.headers['Permissions-Policy'] = (
        'camera=(), microphone=(), geolocation=(), payment=()'
    )

    # Cross-origin 정책
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'

    # 서버 식별 제거
    response.headers.pop('Server', None)

    return response


@app.before_request
def generate_nonce():
    """각 요청에 대한 CSP nonce 생성."""
    g.csp_nonce = secrets.token_urlsafe(32)


# ── 방법 2: Flask-Talisman 사용 ──────────────────────────────
# pip install flask-talisman
from flask_talisman import Talisman

app2 = Flask(__name__)

csp = {
    'default-src': "'self'",
    'script-src': "'self'",
    'style-src': "'self' https://fonts.googleapis.com",
    'font-src': "'self' https://fonts.gstatic.com",
    'img-src': "'self' data:",
    'object-src': "'none'",
}

talisman = Talisman(
    app2,
    content_security_policy=csp,
    content_security_policy_nonce_in=['script-src', 'style-src'],
    force_https=True,
    strict_transport_security=True,
    strict_transport_security_max_age=31536000,
    strict_transport_security_include_subdomains=True,
    strict_transport_security_preload=True,
    frame_options='DENY',
    referrer_policy='strict-origin-when-cross-origin',
    permissions_policy={
        'camera': '()',
        'microphone': '()',
        'geolocation': '()',
    },
    session_cookie_secure=True,
    session_cookie_http_only=True,
)

# Jinja2 템플릿에서 nonce 사용:
# <script nonce="{{ csp_nonce() }}">
#     // 여기에 인라인 JavaScript
# </script>
```

### 10.2 라우트별 헤더 재정의

```python
"""
서로 다른 라우트에 대해 서로 다른 보안 헤더.
"""
from flask import Flask, make_response
from functools import wraps

app = Flask(__name__)

def custom_csp(csp_string):
    """특정 라우트에 대한 CSP를 재정의하는 데코레이터."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = make_response(f(*args, **kwargs))
            response.headers['Content-Security-Policy'] = csp_string
            return response
        return decorated_function
    return decorator


@app.route('/admin')
@custom_csp("default-src 'self'; script-src 'self'; frame-ancestors 'none'")
def admin_panel():
    """엄격한 CSP가 있는 관리자 패널."""
    return "Admin Panel"


@app.route('/embed-widget')
@custom_csp(
    "default-src 'self'; "
    "frame-ancestors 'self' https://partner.example.com"
)
def embeddable_widget():
    """특정 파트너가 임베드할 수 있는 위젯."""
    return "Widget"


@app.route('/public-api')
def public_api():
    """cross-origin 접근을 위해 완화된 CORP가 있는 API 엔드포인트."""
    response = make_response({"data": "value"})
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
```

---

## 11. Django 보안 헤더 구성

### 11.1 Django 설정

```python
"""
settings.py에서 Django 보안 헤더 구성.
"""

# ── 내장 Django 보안 설정 ────────────────────────────

# HSTS
SECURE_HSTS_SECONDS = 31536000           # 1년
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# HTTPS 리디렉트
SECURE_SSL_REDIRECT = True               # HTTP를 HTTPS로 리디렉트
SECURE_REDIRECT_EXEMPT = []              # 리디렉트에서 제외할 경로

# Content-Type 스니핑
SECURE_CONTENT_TYPE_NOSNIFF = True       # X-Content-Type-Options: nosniff

# 클릭재킹 보호
X_FRAME_OPTIONS = 'DENY'                 # 내장 미들웨어

# Referrer 정책
SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'

# Cross-origin opener 정책
SECURE_CROSS_ORIGIN_OPENER_POLICY = 'same-origin'

# 쿠키 보안
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = True

# ── 필요한 미들웨어 ──────────────────────────────────────────
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',       # 반드시 첫 번째
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # ... 다른 미들웨어
]
```

### 11.2 django-csp를 사용한 Django CSP

```python
"""
django-csp 패키지를 사용한 CSP 구성.
pip install django-csp
"""

# settings.py
MIDDLEWARE = [
    'csp.middleware.CSPMiddleware',
    # ... 다른 미들웨어
]

# CSP 지시문
CSP_DEFAULT_SRC = ("'self'",)
CSP_SCRIPT_SRC = ("'self'",)
CSP_STYLE_SRC = ("'self'", "https://fonts.googleapis.com")
CSP_FONT_SRC = ("'self'", "https://fonts.gstatic.com")
CSP_IMG_SRC = ("'self'", "data:")
CSP_CONNECT_SRC = ("'self'",)
CSP_OBJECT_SRC = ("'none'",)
CSP_BASE_URI = ("'self'",)
CSP_FORM_ACTION = ("'self'",)
CSP_FRAME_ANCESTORS = ("'none'",)

# script-src와 style-src에 nonce 활성화
CSP_INCLUDE_NONCE_IN = ['script-src', 'style-src']

# 위반 보고
CSP_REPORT_URI = '/csp-report/'

# Report-only 모드 (테스트용)
# CSP_REPORT_ONLY = True


# ── Django 템플릿에서 ──────────────────────────────────────────
# {% load csp %}
#
# <script nonce="{% csp_nonce %}">
#     // 이 인라인 스크립트는 nonce에 의해 허용됨
#     console.log('Hello from a nonced script');
# </script>
#
# ── 뷰별 CSP 재정의 ────────────────────────────────────────────
# from csp.decorators import csp_update, csp_replace, csp_exempt
#
# @csp_update(SCRIPT_SRC=("'self'", "https://cdn.example.com"))
# def my_view(request):
#     ...
#
# @csp_exempt  # 이 뷰에 대해 CSP 완전히 비활성화
# def legacy_view(request):
#     ...
```

---

## 12. 보안 헤더 테스트

### 12.1 curl로 테스트

```bash
# ── 모든 응답 헤더 보기 ────────────────────────────────────
curl -I https://example.com

# ── 특정 보안 헤더 확인 ─────────────────────────────────
curl -sI https://example.com | grep -iE \
  '(content-security|strict-transport|x-content-type|x-frame|referrer-policy|permissions-policy|cross-origin)'

# ── TLS 세부정보도 보기 위한 상세 출력 ───────────────────────
curl -vI https://example.com 2>&1 | head -40

# ── HSTS 테스트 ───────────────────────────────────────────────
curl -sI https://example.com | grep -i strict-transport

# ── report-only 모드에서 CSP 테스트 ────────────────────────────
curl -sI https://example.com | grep -i content-security-policy

# ── 누락된 헤더 확인 ───────────────────────────────────────────
HEADERS_TO_CHECK=(
    "Content-Security-Policy"
    "Strict-Transport-Security"
    "X-Content-Type-Options"
    "X-Frame-Options"
    "Referrer-Policy"
    "Permissions-Policy"
)

URL="https://example.com"
echo "Checking security headers for $URL"
echo "=================================="

for header in "${HEADERS_TO_CHECK[@]}"; do
    result=$(curl -sI "$URL" | grep -i "^$header:")
    if [ -n "$result" ]; then
        echo "[PASS] $result"
    else
        echo "[FAIL] Missing: $header"
    fi
done
```

### 12.2 Python 보안 헤더 스캐너

```python
"""
간단한 보안 헤더 스캐너.
"""
import requests
from dataclasses import dataclass
from typing import Optional

@dataclass
class HeaderCheck:
    name: str
    present: bool
    value: Optional[str]
    severity: str  # 'critical', 'high', 'medium', 'low'
    recommendation: str

def scan_security_headers(url: str) -> list[HeaderCheck]:
    """URL의 보안 헤더를 스캔하고 결과 반환."""

    response = requests.get(url, allow_redirects=True, timeout=10)
    headers = response.headers
    results = []

    # ── Content-Security-Policy ──────────────────────────────────
    csp = headers.get('Content-Security-Policy')
    results.append(HeaderCheck(
        name='Content-Security-Policy',
        present=csp is not None,
        value=csp,
        severity='critical',
        recommendation=(
            "CSP 헤더를 추가하세요. 다음으로 시작: "
            "Content-Security-Policy: default-src 'self'; "
            "object-src 'none'; base-uri 'self'"
        ) if not csp else _analyze_csp(csp)
    ))

    # ── Strict-Transport-Security ────────────────────────────────
    hsts = headers.get('Strict-Transport-Security')
    results.append(HeaderCheck(
        name='Strict-Transport-Security',
        present=hsts is not None,
        value=hsts,
        severity='critical',
        recommendation=(
            "HSTS 헤더를 추가하세요: "
            "Strict-Transport-Security: max-age=31536000; "
            "includeSubDomains; preload"
        ) if not hsts else _analyze_hsts(hsts)
    ))

    # ── X-Content-Type-Options ───────────────────────────────────
    xcto = headers.get('X-Content-Type-Options')
    results.append(HeaderCheck(
        name='X-Content-Type-Options',
        present=xcto is not None,
        value=xcto,
        severity='high',
        recommendation=(
            "헤더를 추가하세요: X-Content-Type-Options: nosniff"
        ) if not xcto else "OK"
    ))

    # ── X-Frame-Options ─────────────────────────────────────────
    xfo = headers.get('X-Frame-Options')
    results.append(HeaderCheck(
        name='X-Frame-Options',
        present=xfo is not None,
        value=xfo,
        severity='high',
        recommendation=(
            "헤더를 추가하세요: X-Frame-Options: DENY "
            "(또는 프레이밍이 필요한 경우 SAMEORIGIN)"
        ) if not xfo else "OK"
    ))

    # ── Referrer-Policy ──────────────────────────────────────────
    rp = headers.get('Referrer-Policy')
    results.append(HeaderCheck(
        name='Referrer-Policy',
        present=rp is not None,
        value=rp,
        severity='medium',
        recommendation=(
            "헤더를 추가하세요: Referrer-Policy: "
            "strict-origin-when-cross-origin"
        ) if not rp else "OK"
    ))

    # ── Permissions-Policy ───────────────────────────────────────
    pp = headers.get('Permissions-Policy')
    results.append(HeaderCheck(
        name='Permissions-Policy',
        present=pp is not None,
        value=pp,
        severity='medium',
        recommendation=(
            "헤더를 추가하세요: Permissions-Policy: "
            "camera=(), microphone=(), geolocation=()"
        ) if not pp else "OK"
    ))

    # ── 위험한 헤더 확인 ─────────────────────────────────────
    server = headers.get('Server')
    if server:
        results.append(HeaderCheck(
            name='Server',
            present=True,
            value=server,
            severity='low',
            recommendation=(
                f"Server 헤더가 다음을 노출: '{server}'. "
                "제거 또는 난독화를 고려하세요."
            )
        ))

    x_powered = headers.get('X-Powered-By')
    if x_powered:
        results.append(HeaderCheck(
            name='X-Powered-By',
            present=True,
            value=x_powered,
            severity='medium',
            recommendation=(
                f"X-Powered-By가 다음을 노출: '{x_powered}'. "
                "정보 공개를 피하기 위해 이 헤더를 제거하세요."
            )
        ))

    return results


def _analyze_csp(csp: str) -> str:
    """일반적인 약점에 대한 CSP 정책 분석."""
    issues = []
    if "'unsafe-inline'" in csp and 'script-src' in csp:
        issues.append("script-src가 'unsafe-inline' 허용 (XSS 보호 약화)")
    if "'unsafe-eval'" in csp:
        issues.append("정책이 'unsafe-eval' 허용 (eval() 기반 공격 활성화)")
    if "default-src" not in csp:
        issues.append("default-src 대체 지시문 누락")
    if "object-src" not in csp and "default-src 'none'" not in csp:
        issues.append("object-src 지시문 누락 (Flash/플러그인 위험)")
    if "base-uri" not in csp:
        issues.append("base-uri 누락 (dangling markup injection 활성화 가능)")
    return "; ".join(issues) if issues else "OK"


def _analyze_hsts(hsts: str) -> str:
    """약점에 대한 HSTS 헤더 분석."""
    issues = []
    hsts_lower = hsts.lower()
    if 'max-age=' in hsts_lower:
        # max-age 값 추출
        import re
        match = re.search(r'max-age=(\d+)', hsts_lower)
        if match:
            max_age = int(match.group(1))
            if max_age < 31536000:
                issues.append(
                    f"max-age={max_age}가 1년(31536000)보다 작음"
                )
    if 'includesubdomains' not in hsts_lower:
        issues.append("includeSubDomains 누락")
    if 'preload' not in hsts_lower:
        issues.append("preload 누락 (브라우저 preload 목록에 적합하지 않음)")
    return "; ".join(issues) if issues else "OK"


# ── 사용 ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else 'https://example.com'
    print(f"\n스캔 중: {url}\n{'=' * 60}")

    results = scan_security_headers(url)

    for check in results:
        status = "PASS" if check.present and check.recommendation == "OK" else "FAIL"
        icon = "[+]" if status == "PASS" else "[-]"
        print(f"\n{icon} {check.name} [{check.severity.upper()}]")
        print(f"    값: {check.value or '(설정되지 않음)'}")
        if check.recommendation != "OK":
            print(f"    권장사항: {check.recommendation}")
```

### 12.3 온라인 스캐너

```
┌─────────────────────────────────────────────────────────────────────┐
│                    보안 헤더 스캐너                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. SecurityHeaders.com                                              │
│     URL: https://securityheaders.com                                 │
│     등급: A+에서 F까지                                               │
│     모든 주요 보안 헤더 확인                                         │
│                                                                      │
│  2. Mozilla Observatory                                              │
│     URL: https://observatory.mozilla.org                             │
│     CSP 분석을 포함한 포괄적인 스캔                                  │
│     개선 조언 제공                                                   │
│                                                                      │
│  3. CSP Evaluator (Google)                                           │
│     URL: https://csp-evaluator.withgoogle.com                        │
│     특화된 CSP 분석                                                  │
│     우회 및 약점 식별                                                │
│                                                                      │
│  4. Hardenize                                                        │
│     URL: https://www.hardenize.com                                   │
│     헤더 + TLS + DNS + 이메일 보안 테스트                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 13. 완전한 보안 헤더 템플릿

### 13.1 Nginx 구성

```nginx
# /etc/nginx/conf.d/security-headers.conf
# 서버 블록에 이것을 포함

# Content-Security-Policy
add_header Content-Security-Policy
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
    always;

# HSTS (모든 곳에서 HTTPS가 작동하는지 확인한 후에만 활성화)
add_header Strict-Transport-Security
    "max-age=31536000; includeSubDomains; preload"
    always;

# MIME 스니핑 방지
add_header X-Content-Type-Options "nosniff" always;

# 클릭재킹 보호
add_header X-Frame-Options "DENY" always;

# Referrer 정책
add_header Referrer-Policy "strict-origin-when-cross-origin" always;

# Permissions 정책
add_header Permissions-Policy
    "camera=(), microphone=(), geolocation=(), payment=()"
    always;

# Cross-origin 정책
add_header Cross-Origin-Opener-Policy "same-origin" always;
add_header Cross-Origin-Resource-Policy "same-origin" always;
add_header Cross-Origin-Embedder-Policy "require-corp" always;

# 서버 버전 정보 제거
server_tokens off;

# X-Powered-By 제거 (업스트림 앱에서 설정한 경우)
proxy_hide_header X-Powered-By;
```

### 13.2 Apache 구성

```apache
# .htaccess 또는 httpd.conf

# Content-Security-Policy
Header always set Content-Security-Policy "\
default-src 'self'; \
script-src 'self'; \
style-src 'self' 'unsafe-inline'; \
img-src 'self' data:; \
font-src 'self'; \
connect-src 'self'; \
object-src 'none'; \
base-uri 'self'; \
form-action 'self'; \
frame-ancestors 'none'"

# HSTS
Header always set Strict-Transport-Security \
    "max-age=31536000; includeSubDomains; preload"

# 기타 보안 헤더
Header always set X-Content-Type-Options "nosniff"
Header always set X-Frame-Options "DENY"
Header always set Referrer-Policy "strict-origin-when-cross-origin"
Header always set Permissions-Policy \
    "camera=(), microphone=(), geolocation=(), payment=()"
Header always set Cross-Origin-Opener-Policy "same-origin"
Header always set Cross-Origin-Resource-Policy "same-origin"

# 서버 정보 제거
ServerTokens Prod
Header always unset X-Powered-By
```

---

## 14. 연습 문제

### 연습 문제 1: CSP 정책 분석

다음 CSP 정책을 분석하고 모든 보안 약점을 식별하세요:

```
Content-Security-Policy:
    default-src *;
    script-src 'self' 'unsafe-inline' 'unsafe-eval' https:;
    style-src 'self' 'unsafe-inline';
    img-src *;
    connect-src *;
    font-src *
```

질문:
1. 몇 개의 별개의 보안 문제를 식별할 수 있습니까?
2. 각 문제에 대해 활성화되는 공격 벡터를 설명하세요.
3. Google Fonts와 단일 CDN (`cdn.example.com`)을 허용하면서 안전하도록 정책을 다시 작성하세요.

### 연습 문제 2: Flask 보안 헤더 미들웨어

다음을 수행하는 Flask 미들웨어 클래스를 구축하세요:

1. 모든 권장 보안 헤더 설정
2. 요청당 고유한 CSP nonce 생성
3. Jinja2 템플릿에서 nonce 사용 가능하게 만들기
4. 데코레이터를 통한 라우트별 CSP 재정의 허용
5. CSP 테스트를 위한 "report-only" 모드 지원
6. 헤더 구성 오류 로깅

### 연습 문제 3: 헤더 스캐너 향상

섹션 12.2의 Python 보안 헤더 스캐너를 확장하여 다음을 수행하세요:

1. `Cross-Origin-Embedder-Policy`와 `Cross-Origin-Opener-Policy` 확인
2. HSTS max-age가 최소 1년인지 확인
3. `X-Powered-By` 또는 `Server` 헤더가 버전 정보를 노출하는지 감지
4. `unsafe-inline` 및 `unsafe-eval` 사용을 위해 CSP 파싱 및 분석
5. 스캔 결과를 기반으로 등급 생성 (A+에서 F까지)
6. 텍스트 및 JSON 형식으로 결과 출력

### 연습 문제 4: SRI 해시 생성기

다음을 수행하는 Python 스크립트를 작성하세요:

1. CDN URL 목록을 입력으로 받기
2. 각 리소스 다운로드
3. SHA-384 무결성 해시 계산
4. 무결성 속성이 있는 완전한 HTML `<script>` 또는 `<link>` 태그 생성
5. 선택적으로 모든 태그가 있는 HTML 파일 작성
6. 오류를 우아하게 처리 (네트워크 실패, 404 등)

### 연습 문제 5: HSTS Preload 준비 확인

도메인이 HSTS preload 제출 준비가 되었는지 확인하는 도구를 작성하세요:

1. 도메인에 유효한 TLS 인증서가 있는지 확인
2. HTTP가 HTTPS로 리디렉트되는지 확인
3. HSTS 헤더에 `max-age >= 31536000`이 포함되어 있는지 확인
4. `includeSubDomains` 및 `preload` 지시문 확인
5. HTTPS 지원을 위해 일반적인 서브도메인(www, mail, api) 테스트
6. 준비 보고서 생성

### 연습 문제 6: Cross-Origin 격리 감사

주어진 웹 애플리케이션 URL에 대해:

1. `Cross-Origin-Embedder-Policy`가 설정되어 있는지 확인
2. `Cross-Origin-Opener-Policy`가 설정되어 있는지 확인
3. 페이지에서 로드된 모든 cross-origin 리소스 식별
4. 각 cross-origin 리소스에 대해 `Cross-Origin-Resource-Policy`가 설정되어 있는지 확인
5. COEP: require-corp가 활성화된 경우 깨질 리소스 보고

---

## 요약

| 헤더 | 방지하는 공격 | 필수? |
|--------|-----------------|------------|
| Content-Security-Policy | XSS, 주입 | 예 |
| Strict-Transport-Security | 프로토콜 다운그레이드 | 예 (HTTPS 사이트) |
| X-Content-Type-Options | MIME 혼동 | 예 |
| X-Frame-Options | 클릭재킹 | 예 |
| Referrer-Policy | 정보 유출 | 예 |
| Permissions-Policy | 기능 남용 | 권장 |
| CORP/COEP/COOP | Spectre, cross-origin | 권장 |

### 핵심 요점

1. **report-only 모드로 시작** — CSP를 먼저 report-only로 배포하고, 위반을 수정한 후 강제 적용
2. **unsafe-inline보다 nonce 사용** — nonce는 인라인 스크립트에 대한 요청별 인증 제공
3. **HSTS는 점진적 롤아웃 필요** — 짧은 max-age로 시작하고 모든 서브도메인이 HTTPS를 지원하는지 확인한 후 증가
4. **심층 방어** — 보안 헤더는 안전한 코딩 관행을 보완하며 대체하지 않음
5. **정기적으로 테스트** — CI/CD에서 자동화된 스캐너를 사용하여 헤더 회귀 포착

---

**이전**: [08. Injection 공격과 방어](./08_Injection_Attacks.md) | **다음**: [10_API_Security.md](./10_API_Security.md)
