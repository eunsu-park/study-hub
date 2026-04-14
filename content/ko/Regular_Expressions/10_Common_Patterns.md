# 자주 쓰는 패턴

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 이메일 검증 패턴을 구성하고 이해할 수 있다
2. URL을 정규식으로 파싱하고 검증할 수 있다
3. IPv4 주소 형식을 매칭할 수 있다
4. 다양한 날짜 및 시간 형식을 검증할 수 있다
5. 여러 형식의 전화번호를 매칭할 수 있다
6. 비밀번호, 사용자명 등 일반적인 검증 패턴을 만들 수 있다
7. 검증에서 엄격함과 실용성 사이의 균형을 이해할 수 있다
8. 정규식이 적절하지 않은 검증 상황을 판단할 수 있다

---

## 1. 이메일 주소 검증

### 실용적 패턴 (대부분의 경우 충분)

```python
import re

email_pattern = re.compile(r"""
    ^[\w.+-]+           # 로컬 파트
    @                   # @ 기호
    [\w-]+              # 도메인
    (?:\.[\w-]+)*       # 서브도메인
    \.[a-zA-Z]{2,}      # TLD
    $
""", re.VERBOSE)

test_emails = [
    ("user@example.com", True),
    ("first.last@domain.org", True),
    ("user+tag@sub.domain.co.uk", True),
    ("@missing-local.com", False),
    ("missing-domain@", False),
    ("no-tld@domain", False),
]

for email, expected in test_emails:
    result = bool(email_pattern.match(email))
    status = "PASS" if result == expected else "FAIL"
    print(f"[{status}] {email:35s} -> {result}")
```

```
이메일 패턴 분석:

    user.name+tag@sub.domain.com
    ─────────────┬──────────────
    [\w.+-]+     @  [\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}
    로컬 파트        TLD 포함 도메인
```

> **참고**: RFC 5322 호환 이메일 정규식은 수천 자에 달합니다. 위 패턴은 실제 이메일의 99% 이상을 처리합니다. 프로덕션 용도로는 전용 이메일 검증 라이브러리를 고려하세요.

---

## 2. URL 검증

```python
import re

url_pattern = re.compile(r"""
    ^(?:(?P<scheme>https?|ftp)://)   # 스킴
    (?P<host>                         # 호스트
        (?:[\w-]+\.)+                 # 도메인 레이블
        [a-zA-Z]{2,}                  # TLD
        |                             # 또는
        \d{1,3}(?:\.\d{1,3}){3}      # IPv4 주소
    )
    (?::(?P<port>\d{1,5}))?          # 선택적 포트
    (?P<path>/[^\s?#]*)?             # 선택적 경로
    (?:\?(?P<query>[^\s#]*))?        # 선택적 쿼리
    (?:\#(?P<fragment>\S*))?         # 선택적 프래그먼트
    $
""", re.VERBOSE)

urls = [
    "https://www.example.com",
    "http://example.com:8080/path?q=1#section",
    "ftp://files.example.com/pub/data.zip",
    "not-a-url",
]

for url in urls:
    match = url_pattern.match(url)
    if match:
        d = match.groupdict()
        print(f"유효: {url}")
        print(f"  scheme={d['scheme']}, host={d['host']}, port={d['port']}")
    else:
        print(f"무효: {url}")
```

---

## 3. IPv4 주소 검증

```python
import re

# 엄격한 검증: 각 옥텟이 0-255인지 확인
strict_ip = re.compile(r"""
    ^
    (?:
        (?:25[0-5])           # 250-255
        |(?:2[0-4]\d)         # 200-249
        |(?:1\d{2})           # 100-199
        |(?:[1-9]\d)          # 10-99
        |(?:\d)               # 0-9
    )
    (?:\.
        (?:
            (?:25[0-5])
            |(?:2[0-4]\d)
            |(?:1\d{2})
            |(?:[1-9]\d)
            |(?:\d)
        )
    ){3}
    $
""", re.VERBOSE)

test_ips = [
    ("192.168.1.1", True),
    ("255.255.255.255", True),
    ("0.0.0.0", True),
    ("256.1.1.1", False),
    ("192.168.1", False),
]

for ip, expected in test_ips:
    result = bool(strict_ip.match(ip))
    status = "PASS" if result == expected else "FAIL"
    print(f"[{status}] {ip:20s} -> {result}")
```

```
IPv4 옥텟 분석:

    값 범위        패턴
    ───────────    ───────
    0-9            \d
    10-99          [1-9]\d
    100-199        1\d{2}
    200-249        2[0-4]\d
    250-255        25[0-5]
```

---

## 4. 날짜와 시간 검증

```python
import re

# YYYY-MM-DD (ISO 8601)
iso_date = re.compile(r"""
    ^(?P<year>\d{4})          # 년
    -(?P<month>0[1-9]|1[0-2]) # 월 (01-12)
    -(?P<day>0[1-9]|[12]\d|3[01])  # 일 (01-31)
    $
""", re.VERBOSE)

# HH:MM:SS (24시간)
time_24h = re.compile(r"""
    ^(?P<hour>[01]\d|2[0-3])      # 시 (00-23)
    :(?P<minute>[0-5]\d)           # 분 (00-59)
    (?::(?P<second>[0-5]\d))?      # 선택적 초
    $
""", re.VERBOSE)

dates = ["2024-01-15", "2024-13-01", "2024-01-32"]
for d in dates:
    result = "유효" if iso_date.match(d) else "무효"
    print(f"{d} -> {result}")

times = ["08:30:45", "23:59", "25:00"]
for t in times:
    result = "유효" if time_24h.match(t) else "무효"
    print(f"{t} -> {result}")
```

---

## 5. 전화번호 패턴

```python
import re

# 유연한 미국 전화번호
us_phone = re.compile(r"""
    ^
    (?:\+?1[\s.-]?)?          # 선택적 국가 코드
    (?:\(?(\d{3})\)?[\s.-]?)  # 지역 코드 (선택적 괄호)
    (\d{3})                    # 국번
    [\s.-]?                    # 구분자
    (\d{4})                    # 가입자 번호
    $
""", re.VERBOSE)

phones = [
    "555-867-5309",
    "(555) 867-5309",
    "555.867.5309",
    "+1 555 867 5309",
    "5558675309",
]

for phone in phones:
    match = us_phone.match(phone)
    if match:
        area, exchange, subscriber = match.groups()
        print(f"{phone:25s} -> ({area}) {exchange}-{subscriber}")
```

---

## 6. 비밀번호 검증

```python
import re

def validate_password(password):
    """상세한 피드백과 함께 비밀번호 강도 검증."""
    rules = [
        (r'.{8,}', "최소 8자"),
        (r'[A-Z]', "최소 1개 대문자"),
        (r'[a-z]', "최소 1개 소문자"),
        (r'\d', "최소 1개 숫자"),
        (r'[!@#$%^&*(),.?":{}|<>]', "최소 1개 특수문자"),
        (r'^[^\s]+$', "공백 불가"),
    ]

    passed = True
    for pattern, description in rules:
        if not re.search(pattern, password):
            print(f"  실패: {description}")
            passed = False

    return passed

passwords = ["P@ssw0rd!", "password", "SHORT1!", "NoSpecial1"]
for pwd in passwords:
    print(f"\n'{pwd}':")
    result = validate_password(pwd)
    print(f"  결과: {'강함' if result else '약함'}")
```

---

## 7. 사용자명과 식별자 패턴

```python
import re

# 사용자명: 3~20자, 영숫자 + 밑줄, 문자로 시작
username_pattern = re.compile(r'^[a-zA-Z]\w{2,19}$')

# 슬러그: 소문자, 하이픈, 연속 하이픈 불가
slug_pattern = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')

# 시맨틱 버전: MAJOR.MINOR.PATCH
semver_pattern = re.compile(r"""
    ^(?P<major>0|[1-9]\d*)
    \.(?P<minor>0|[1-9]\d*)
    \.(?P<patch>0|[1-9]\d*)
    (?:-(?P<pre>[a-zA-Z0-9.]+))?    # 선택적 프리릴리스
    (?:\+(?P<build>[a-zA-Z0-9.]+))? # 선택적 빌드 메타데이터
    $
""", re.VERBOSE)

for name in ["alice", "Bob_99", "ab", "1alice"]:
    print(f"사용자명 '{name}': {bool(username_pattern.match(name))}")

for slug in ["my-blog-post", "hello", "Bad Slug", "double--dash"]:
    print(f"슬러그 '{slug}': {bool(slug_pattern.match(slug))}")

for ver in ["1.0.0", "2.1.3-beta.1", "0.0.1+build.123", "1.0"]:
    print(f"버전 '{ver}': {bool(semver_pattern.match(ver))}")
```

---

## 8. 데이터 추출 패턴

### 마크다운 링크 추출

```python
import re

markdown = """
Check out [Google](https://www.google.com) and
[Python docs](https://docs.python.org/3/).
Also see [local page](./about.md).
"""

links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', markdown)
for text, url in links:
    print(f"텍스트: {text:15s} URL: {url}")
```

---

## 9. 정규식을 사용하면 안 되는 경우

정규식은 강력하지만 항상 최선의 도구는 아닙니다:

```python
# 나쁜 예: HTML을 정규식으로 파싱
# HTML은 정규 언어가 아님 -- 적절한 파서 사용
# from bs4 import BeautifulSoup

# 나쁜 예: JSON을 정규식으로 검증
# 대안: import json; json.loads(text)

# 나쁜 예: 복잡한 날짜 형식 검증
# 대안: from datetime import datetime; datetime.strptime(text, fmt)

# 좋은 사용 사례:
# - 빠른 텍스트 검색과 추출
# - 로그 파일 파싱
# - 데이터 정제 (공백, 형식)
# - 간단한 검증 (형식 확인)
# - 텍스트 검색 및 치환
# - 토큰화
```

---

## 10. 패턴 참조 카드

```
패턴                       용도                  매칭 예시
───────                   ───────              ─────────────
[\w.+-]+@[\w-]+\.\w+      이메일 (기본)          user@example.com
https?://\S+              URL (기본)             https://example.com/path
\d{1,3}(\.\d{1,3}){3}    IPv4 (형식)            192.168.1.1
\d{4}-\d{2}-\d{2}        날짜 ISO               2024-01-15
\d{2}:\d{2}(:\d{2})?     시간                   08:30:45
#[0-9a-fA-F]{3,6}        16진수 색상             #FF0000
\d+\.\d+\.\d+            시맨틱 버전             1.2.3
\b[A-Z][a-z]+\b          대문자로 시작하는 단어   Hello
```

---

## 요약

| 분류 | 권장 접근법 |
|------|------------|
| 이메일 | 형식 확인에는 간단한 정규식; 프로덕션 검증에는 라이브러리 |
| URL | 추출에는 정규식; 파싱에는 `urllib.parse` |
| IP 주소 | 형식에는 정규식; 검증에는 `ipaddress` 모듈 |
| 날짜/시간 | 추출에는 정규식; 파싱과 검증에는 `datetime` |
| 전화번호 | 유연한 구분자로 정규식; 매칭 후 정규화 |
| 비밀번호 | 규칙별 개별 정규식 검사 (규칙당 하나) |

핵심 원칙:
- 단순하게 시작하고, 필요한 만큼만 복잡도 추가
- 30자를 넘는 패턴에는 `re.VERBOSE` 사용
- 유효한 입력과 무효한 입력 모두로 테스트
- 엣지 케이스 고려 (빈 문자열, 경계 값)
- 적절한 파서를 사용해야 할 때를 판단

---

## 다음 강의

[11_성능과 함정](./11_Performance_and_Pitfalls.md)에서는 정규식 성능 문제와 이를 피하는 방법을 배웁니다.
