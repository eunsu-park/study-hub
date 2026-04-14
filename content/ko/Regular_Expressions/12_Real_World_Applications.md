# 실전 활용

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 정규식을 사용하여 로그 파일을 파싱하고 분석할 수 있다
2. 치환 패턴으로 지저분한 데이터를 정제하고 정규화할 수 있다
3. 비정형 텍스트에서 구조화된 데이터를 추출할 수 있다
4. 정규식 기반 검색-치환으로 코드를 리팩토링할 수 있다
5. 설정 파일을 검증하고 변환할 수 있다
6. 정규식으로 간단한 토크나이저/렉서를 구축할 수 있다
7. 정규식 인식 분할로 CSV 및 TSV 데이터를 처리할 수 있다
8. 여러 정규식 기법을 결합하여 완전한 솔루션을 만들 수 있다

---

## 1. 로그 파일 파싱

### 표준 로그 형식 파싱

```python
import re
from collections import Counter

log_text = """
[2024-01-15 08:30:45] INFO  server.py:42 - Server started on port 8080
[2024-01-15 08:30:46] DEBUG db.py:15 - Connection pool initialized (size=10)
[2024-01-15 08:31:02] WARN  auth.py:88 - Failed login attempt from 192.168.1.50
[2024-01-15 08:31:15] ERROR api.py:203 - Request timeout after 30s: GET /api/users
[2024-01-15 08:31:16] ERROR api.py:203 - Request timeout after 30s: GET /api/orders
[2024-01-15 08:31:20] INFO  api.py:55 - Health check OK
[2024-01-15 08:31:45] ERROR db.py:67 - Connection refused: max pool exhausted
[2024-01-15 08:32:00] INFO  server.py:100 - Graceful shutdown initiated
""".strip()

log_pattern = re.compile(r"""
    ^\[(?P<datetime>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]\s+
    (?P<level>\w+)\s+
    (?P<file>\S+):(?P<line>\d+)\s+-\s+
    (?P<message>.+)$
""", re.VERBOSE | re.MULTILINE)

entries = []
for match in log_pattern.finditer(log_text):
    entries.append(match.groupdict())

# 분석: 레벨별 집계
level_counts = Counter(e['level'] for e in entries)
print("로그 레벨 분포:")
for level, count in level_counts.most_common():
    bar = "#" * (count * 5)
    print(f"  {level:5s}: {count} {bar}")

# 에러 찾기
print("\n에러:")
for e in entries:
    if e['level'] == 'ERROR':
        print(f"  [{e['datetime']}] {e['file']}:{e['line']} - {e['message']}")
```

### Apache/Nginx 접근 로그 파싱

```python
import re

access_log = """
192.168.1.10 - - [15/Jan/2024:08:30:45 +0000] "GET /index.html HTTP/1.1" 200 5432
192.168.1.20 - user1 [15/Jan/2024:08:30:46 +0000] "POST /api/login HTTP/1.1" 401 128
10.0.0.5 - - [15/Jan/2024:08:31:02 +0000] "GET /api/users?page=2 HTTP/1.1" 200 8192
""".strip()

apache_pattern = re.compile(r"""
    (?P<ip>\S+)\s+
    \S+\s+
    (?P<user>\S+)\s+
    \[(?P<time>[^\]]+)\]\s+
    "(?P<method>\w+)\s+
    (?P<path>\S+)\s+
    (?P<protocol>[^"]+)"\s+
    (?P<status>\d{3})\s+
    (?P<size>\d+)
""", re.VERBOSE)

print("접근 로그 분석:")
print(f"{'IP':20s} {'메서드':6s} {'상태':6s} {'경로'}")
print("-" * 60)
for match in apache_pattern.finditer(access_log):
    d = match.groupdict()
    print(f"{d['ip']:20s} {d['method']:6s} {d['status']:6s} {d['path']}")
```

---

## 2. 데이터 정제와 정규화

### 사용자 입력 정제

```python
import re

def clean_text(text):
    """사용자 제출 텍스트를 정제하고 정규화."""
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)

    # 여러 공백/탭을 단일 공백으로 축소
    text = re.sub(r'[ \t]+', ' ', text)

    # 줄바꿈 정규화
    text = re.sub(r'\r\n|\r', '\n', text)

    # 3개 이상 연속 줄바꿈을 2개로 축소
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 각 행의 앞뒤 공백 제거
    text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE)

    return text.strip()

dirty = '''  <p>Hello   &nbsp;  World</p>

    This   has    too    much   space.


And   too   many   blank   lines.   '''

print(clean_text(dirty))
```

### 전화번호 정규화

```python
import re

def normalize_phone(phone):
    """다양한 전화번호 형식을 (XXX) XXX-XXXX로 정규화."""
    digits = re.sub(r'\D', '', phone)

    if len(digits) == 11 and digits[0] == '1':
        digits = digits[1:]

    if len(digits) != 10:
        return None

    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"

phones = [
    "555-867-5309",
    "(555) 867.5309",
    "1-555-867-5309",
    "+1 555 867 5309",
    "5558675309",
]

for phone in phones:
    normalized = normalize_phone(phone)
    print(f"{phone:25s} -> {normalized}")
```

---

## 3. 비정형 텍스트에서 데이터 추출

### 보고서에서 구조화된 데이터 추출

```python
import re

report = """
Quarterly Sales Report - Q4 2024
=================================

Region: North America
  Revenue: $1,234,567.89
  Units Sold: 45,678
  Growth: +12.5%

Region: Europe
  Revenue: $987,654.32
  Units Sold: 32,100
  Growth: -3.2%

Region: Asia Pacific
  Revenue: $2,345,678.90
  Units Sold: 78,900
  Growth: +28.7%
"""

region_pattern = re.compile(r"""
    Region:\s+(?P<region>.+?)\n
    \s+Revenue:\s+\$(?P<revenue>[\d,]+\.\d{2})\n
    \s+Units\s+Sold:\s+(?P<units>[\d,]+)\n
    \s+Growth:\s+(?P<growth>[+-]?\d+\.?\d*%)
""", re.VERBOSE)

print(f"{'지역':20s} {'매출':>15s} {'판매량':>10s} {'성장률':>8s}")
print("-" * 55)
for match in region_pattern.finditer(report):
    d = match.groupdict()
    print(f"{d['region']:20s} ${d['revenue']:>14s} {d['units']:>10s} {d['growth']:>8s}")
```

---

## 4. 정규식을 이용한 코드 리팩토링

### 변수명 변경

```python
import re

code = """
def calculate_total(item_price, item_count):
    item_total = item_price * item_count
    item_tax = item_total * 0.08
    return item_total + item_tax
"""

# "item_" 접두사를 "product_"로 변경
refactored = re.sub(r'\bitem_(\w+)', r'product_\1', code)
print(refactored)
```

### print 문을 로깅으로 변환

```python
import re

code = '''
print("Starting process")
print(f"Processing {filename}")
print("Error: " + str(e))
print(f"Completed in {elapsed:.2f}s")
'''

def convert_print_to_log(match):
    content = match.group(1)
    if re.search(r'(?i)error|fail|exception', content):
        return f'logger.error({content})'
    elif re.search(r'(?i)warn', content):
        return f'logger.warning({content})'
    else:
        return f'logger.info({content})'

result = re.sub(r'print\((.+)\)', convert_print_to_log, code)
print(result)
```

---

## 5. 설정 파일 처리

### INI 파일 파서

```python
import re

ini_text = """
[database]
host = localhost
port = 5432
name = myapp

[server]
host = 0.0.0.0
port = 8080
debug = true

# This is a comment
[logging]
level = INFO
file = /var/log/app.log
"""

def parse_ini(text):
    """INI 파일을 중첩 딕셔너리로 파싱."""
    config = {}
    current_section = None

    section_pattern = re.compile(r'^\[(\w+)\]$')
    kv_pattern = re.compile(r'^(\w+)\s*=\s*(.+?)$')
    comment_pattern = re.compile(r'^\s*[#;]')

    for line in text.strip().split('\n'):
        line = line.strip()

        if not line or comment_pattern.match(line):
            continue

        section_match = section_pattern.match(line)
        if section_match:
            current_section = section_match.group(1)
            config[current_section] = {}
            continue

        kv_match = kv_pattern.match(line)
        if kv_match and current_section:
            key, value = kv_match.groups()
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            config[current_section][key.strip()] = value

    return config

config = parse_ini(ini_text)
for section, values in config.items():
    print(f"[{section}]")
    for key, value in values.items():
        print(f"  {key} = {value} ({type(value).__name__})")
```

---

## 6. 간단한 토크나이저 구축

### 산술 표현식 토크나이저

```python
import re

def tokenize(expression):
    """수학 표현식을 토큰화."""
    token_spec = [
        ('NUMBER',   r'\d+\.?\d*'),      # 정수 또는 실수
        ('PLUS',     r'\+'),              # 덧셈
        ('MINUS',    r'-'),               # 뺄셈
        ('TIMES',    r'\*'),              # 곱셈
        ('DIVIDE',   r'/'),               # 나눗셈
        ('LPAREN',   r'\('),              # 왼쪽 괄호
        ('RPAREN',   r'\)'),              # 오른쪽 괄호
        ('SKIP',     r'[ \t]+'),          # 공백 (건너뜀)
        ('MISMATCH', r'.'),               # 기타 문자 (에러)
    ]

    combined = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)
    master_pattern = re.compile(combined)

    tokens = []
    for match in master_pattern.finditer(expression):
        kind = match.lastgroup
        value = match.group()
        if kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise ValueError(f"예상치 못한 문자: {value!r}")
        elif kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
        tokens.append((kind, value))

    return tokens

expr = "3.14 * (2 + 5) - 10 / 3"
tokens = tokenize(expr)
for token in tokens:
    print(f"  {token[0]:10s} : {token[1]}")
```

---

## 7. 배치 파일 이름 변경 (시뮬레이션)

```python
import re

def simulate_rename(filenames, pattern, replacement):
    """정규식을 이용한 배치 파일 이름 변경 시뮬레이션."""
    results = []
    for name in filenames:
        new_name = re.sub(pattern, replacement, name)
        if new_name != name:
            results.append((name, new_name))
    return results

files = [
    "IMG_20240115_083045.jpg",
    "IMG_20240116_142030.jpg",
    "Screenshot 2024-01-17 at 10.30.00.png",
]

# IMG_YYYYMMDD_HHMMSS를 YYYY-MM-DD_HH-MM-SS로 변환
renames = simulate_rename(
    files,
    r'IMG_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',
    r'\1-\2-\3_\4-\5-\6'
)

print("제안된 이름 변경:")
for old, new in renames:
    print(f"  {old:45s} -> {new}")
```

---

## 8. 로그 분석 파이프라인

```python
import re
from collections import defaultdict

logs = """
2024-01-15T08:30:45Z [api-server] INFO  GET /api/users 200 45ms
2024-01-15T08:30:46Z [api-server] INFO  POST /api/login 401 12ms
2024-01-15T08:30:47Z [api-server] ERROR GET /api/orders 500 3002ms
2024-01-15T08:31:00Z [db-server]  WARN  Connection pool 80% utilized
2024-01-15T08:31:02Z [api-server] INFO  GET /api/users 200 38ms
2024-01-15T08:31:05Z [api-server] INFO  GET /api/products 200 52ms
2024-01-15T08:31:10Z [api-server] ERROR POST /api/orders 500 5001ms
2024-01-15T08:31:15Z [api-server] INFO  GET /health 200 2ms
""".strip()

http_pattern = re.compile(r"""
    (?P<timestamp>\S+)\s+
    \[(?P<server>[^\]]+)\]\s+
    (?P<level>\w+)\s+
    (?P<method>GET|POST|PUT|DELETE|PATCH)\s+
    (?P<path>\S+)\s+
    (?P<status>\d{3})\s+
    (?P<duration>\d+)ms
""", re.VERBOSE)

endpoint_stats = defaultdict(lambda: {'count': 0, 'errors': 0, 'total_ms': 0})
slow_requests = []

for match in http_pattern.finditer(logs):
    d = match.groupdict()
    key = f"{d['method']} {d['path']}"
    duration = int(d['duration'])
    status = int(d['status'])

    endpoint_stats[key]['count'] += 1
    endpoint_stats[key]['total_ms'] += duration
    if status >= 400:
        endpoint_stats[key]['errors'] += 1
    if duration > 1000:
        slow_requests.append(d)

print("엔드포인트 통계:")
print(f"{'엔드포인트':30s} {'요청':>8s} {'에러':>6s} {'평균ms':>8s}")
print("-" * 55)
for endpoint, stats in sorted(endpoint_stats.items()):
    avg = stats['total_ms'] / stats['count']
    print(f"{endpoint:30s} {stats['count']:>8d} {stats['errors']:>6d} {avg:>8.0f}")

if slow_requests:
    print(f"\n느린 요청 (>1초): {len(slow_requests)}건")
    for r in slow_requests:
        print(f"  {r['timestamp']} {r['method']} {r['path']} -> {r['duration']}ms")
```

---

## 9. 완전한 텍스트 처리기 구축

```python
import re

class TextProcessor:
    """정규식 규칙을 사용한 구성 가능한 텍스트 처리기."""

    def __init__(self):
        self.rules = []

    def add_rule(self, name, pattern, replacement, flags=0):
        """처리 규칙 추가."""
        compiled = re.compile(pattern, flags)
        self.rules.append((name, compiled, replacement))

    def process(self, text, verbose=False):
        """모든 규칙을 텍스트에 적용."""
        for name, pattern, replacement in self.rules:
            new_text = pattern.sub(replacement, text)
            if verbose and new_text != text:
                print(f"  적용: {name}")
            text = new_text
        return text

# 처리기 구성
proc = TextProcessor()
proc.add_rule("html_제거", r'<[^>]+>', '')
proc.add_rule("공백_정규화", r'\s+', ' ')
proc.add_rule("구두점_공백_수정", r'\s+([.,!?;:])', r'\1')
proc.add_rule("양끝_정리", r'^\s+|\s+$', '')

messy = """
  <p>hello world .  this is a   test   !</p>
  <br>  multiple    spaces   and  <b>tags</b>  everywhere .
"""

clean = proc.process(messy, verbose=True)
print(f"\n결과: '{clean}'")
```

---

## 요약

| 활용 분야 | 사용된 핵심 기법 |
|-----------|-----------------|
| 로그 파싱 | 명명 그룹, VERBOSE, MULTILINE, finditer |
| 데이터 정제 | 콜백을 사용한 sub, 문자 클래스, 앵커 |
| 데이터 추출 | 캡처 그룹, findall, 명명 그룹 |
| 코드 리팩토링 | 역참조, 단어 경계, sub |
| 설정 파싱 | MULTILINE, 행 단위 매칭 |
| 토큰화 | 명명 그룹, 교대, finditer |
| 배치 이름 변경 | 그룹 참조를 사용한 sub |

최종 조언:
- 항상 동작하는 가장 단순한 패턴부터 시작
- 복잡한 패턴에는 `re.VERBOSE`로 문서화
- 엣지 케이스와 잘못된 입력으로 테스트
- 구조화된 형식(HTML, JSON, CSV)에는 전용 파서 고려
- 최상의 결과를 위해 정규식과 Python 문자열 메서드를 결합
- 대량 데이터 처리 시 성능 프로파일링

---

## 과정 완료!

정규 표현식 과정을 완료한 것을 축하합니다! 이제 패턴 매칭과 텍스트 처리의 탄탄한 기초를 갖추었습니다. 실제 데이터를 가지고 계속 연습하여 유창성을 키워나가세요.

추가 학습 자료:
- [Python `re` 모듈 문서](https://docs.python.org/3/library/re.html)
- [정규 표현식 HOWTO](https://docs.python.org/3/howto/regex.html)
- [regex101.com](https://regex101.com/)에서 Python 모드로 연습
