# 보안 테스팅

**이전**: [컨테이너 보안](./12_Container_Security.md) | **다음**: [사고 대응](./14_Incident_Response.md)

---

보안 테스팅은 공격자보다 먼저 소프트웨어의 취약점을 찾아내는 체계적인 프로세스입니다. 이 레슨에서는 주요 보안 테스팅 카테고리인 정적 애플리케이션 보안 테스팅(SAST), 동적 애플리케이션 보안 테스팅(DAST), 소프트웨어 구성 분석(SCA), 퍼징(fuzzing)과 함께 침투 테스팅 방법론 및 CI/CD 통합을 다룹니다. 이 레슨을 마치면 프로젝트를 위한 포괄적인 보안 테스팅 파이프라인을 구축할 수 있게 됩니다.

## 학습 목표

- SAST, DAST, SCA, 퍼징의 차이점 이해
- Bandit과 Semgrep을 사용하여 Python 코드의 취약점 발견
- 프로젝트별 패턴을 위한 맞춤형 Semgrep 규칙 작성
- CI/CD 파이프라인에 보안 스캐닝 통합
- 침투 테스팅 방법론 적용
- 효과적인 보안 코드 리뷰 수행

---

## 1. 보안 테스팅 개요

### 1.1 보안 테스팅 피라미드

```
┌─────────────────────────────────────────────────────────────────┐
│                  보안 테스팅 피라미드                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                         /\                                       │
│                        /  \        수동 침투 테스팅              │
│                       / PT \       (가장 비용 높음, 로직 결함에  │
│                      /      \       가장 철저함)                 │
│                     /--------\                                   │
│                    /   DAST   \     실행 중인 애플리케이션에 대한│
│                   /            \    동적 테스팅                   │
│                  /--------------\                                 │
│                 /    퍼징        \   크래시 발견을 위한           │
│                /      (Fuzzing)   \  자동화된 입력 변이          │
│               /--------------------\                             │
│              /        SCA           \  의존성 취약점 스캐닝      │
│             /                        \                           │
│            /--------------------------\                          │
│           /           SAST             \ 정적 코드 분석          │
│          /                              \ (가장 저렴, 가장 빠름,│
│         /________________________________\ 가장 자동화 가능)     │
│                                                                  │
│  ◄── 위로 갈수록 비용/노력 증가                                 │
│  ◄── 위로 갈수록 자동화 감소                                    │
│  ◄── 각 계층은 서로 다른 취약점 클래스 탐지                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 각 테스팅 유형을 적용할 시점

```
┌──────────────────────────────────────────────────────────────────┐
│                    SDLC 보안 테스팅 맵                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  코드 작성 ──► 커밋 ──► 빌드 ──► 배포 ──► 프로덕션               │
│      │           │         │        │           │                │
│      ▼           ▼         ▼        ▼           ▼                │
│    IDE        Pre-commit  CI/CD   스테이징    지속적 모니터링    │
│  린팅          훅         파이프라인  테스팅                      │
│                                                                   │
│  ┌──────┐   ┌──────────┐  ┌──────┐  ┌──────┐  ┌──────────┐     │
│  │ SAST │   │SAST + SCA│  │ 전체 │  │ DAST │  │ 런타임   │     │
│  │(IDE) │   │(pre-push)│  │ 유형 │  │  PT  │  │ 스캐닝   │     │
│  └──────┘   └──────────┘  └──────┘  └──────┘  └──────────┘     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.3 테스팅 접근 방식 비교

| 기능 | SAST | DAST | SCA | 퍼징 | PT |
|------|------|------|-----|------|----|
| **실행 중인 앱 필요** | 아니오 | 예 | 아니오 | 경우에 따라 | 예 |
| **언어 의존적** | 예 | 아니오 | 예 | 다양 | 아니오 |
| **위양성률** | 높음 | 중간 | 낮음 | 낮음 | 매우 낮음 |
| **로직 결함 발견** | 드묾 | 때때로 | 아니오 | 드묾 | 예 |
| **자동화 수준** | 완전 | 완전 | 완전 | 완전 | 부분적 |
| **속도** | 빠름 | 느림 | 빠름 | 중간 | 매우 느림 |
| **커버리지** | 코드 경로 | 공격 표면 | 의존성 | 입력 공간 | 타겟 지향 |

---

## 2. 정적 애플리케이션 보안 테스팅 (SAST)

### 2.1 SAST 동작 원리

SAST 도구는 소스 코드(또는 바이트코드)를 실행하지 않고 분석합니다. 이들은 추상 구문 트리(AST) 또는 제어/데이터 흐름 그래프를 구축하고 잠재적 취약점을 나타내는 패턴을 매칭합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      SAST 분석 파이프라인                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  소스 코드                                                       │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────┐                                                    │
│  │  파서     │ ──► 추상 구문 트리 (AST)                          │
│  └──────────┘                                                    │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────┐                                            │
│  │  제어 흐름        │ ──► CFG: 실행 경로                        │
│  │  분석             │                                           │
│  └──────────────────┘                                            │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────┐                                            │
│  │  데이터 흐름      │ ──► 오염 추적: 소스 → 싱크                │
│  │  분석             │                                           │
│  └──────────────────┘                                            │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────┐                                            │
│  │  패턴 매칭        │ ──► 알려진 취약점 패턴                    │
│  └──────────────────┘                                            │
│      │                                                           │
│      ▼                                                           │
│  취약점 보고서                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Bandit: Python 보안 린터

Bandit은 Python을 위한 가장 인기 있는 SAST 도구입니다. 하드코딩된 비밀번호, `eval()` 사용, 안전하지 않은 해시 함수, SQL 인젝션 패턴과 같은 일반적인 보안 문제를 검사합니다.

#### 설치 및 기본 사용법

```bash
# Bandit 설치
pip install bandit

# 단일 파일 스캔
bandit target_file.py

# 전체 디렉토리 스캔
bandit -r ./myproject/

# 특정 심각도 수준으로 스캔
bandit -r ./myproject/ -ll  # Medium 이상
bandit -r ./myproject/ -lll  # High만

# 출력 형식
bandit -r ./myproject/ -f json -o bandit_report.json
bandit -r ./myproject/ -f html -o bandit_report.html
bandit -r ./myproject/ -f csv -o bandit_report.csv

# 특정 테스트 제외
bandit -r ./myproject/ --skip B101,B601

# 특정 테스트만 포함
bandit -r ./myproject/ --tests B301,B302,B303
```

#### Bandit 테스트 카테고리

```python
"""
Bandit 테스트 카테고리 및 탐지 내용.
각 테스트는 B101, B102 등의 ID를 가집니다.
"""

# ─── B1xx: 일반 보안 문제 ───

# B101: assert_used - assert는 -O 플래그로 제거됨
assert user.is_admin  # WARNING: B101

# B102: exec_used
exec(user_input)  # WARNING: B102

# B103: set_bad_file_permissions
import os
os.chmod('/etc/shadow', 0o777)  # WARNING: B103

# B104: hardcoded_bind_all_interfaces
app.run(host='0.0.0.0')  # WARNING: B104

# B105-B107: hardcoded passwords
password = "SuperSecret123"  # WARNING: B105
config = {"password": "admin123"}  # WARNING: B106


# ─── B2xx: 암호화 문제 ───

# B301: pickle 사용 (역직렬화 공격)
import pickle
data = pickle.loads(user_data)  # WARNING: B301

# B303: 안전하지 않은 해시 함수
import hashlib
h = hashlib.md5(password.encode())  # WARNING: B303

# B304-B305: 안전하지 않은 암호
from Crypto.Cipher import DES
cipher = DES.new(key, DES.MODE_ECB)  # WARNING: B304


# ─── B3xx: 인젝션 문제 ───

# B601: paramiko 셸 인젝션
import paramiko
client.exec_command(user_input)  # WARNING: B601

# B602: subprocess with shell=True
import subprocess
subprocess.call(user_input, shell=True)  # WARNING: B602

# B608: 문자열 포매팅을 통한 SQL 인젝션
query = "SELECT * FROM users WHERE id = %s" % user_id  # WARNING: B608


# ─── B5xx: 암호화 및 SSL 문제 ───

# B501: verify=False로 요청
import requests
requests.get(url, verify=False)  # WARNING: B501

# B502: 버전 체크 없는 ssl
import ssl
context = ssl._create_unverified_context()  # WARNING: B502


# ─── B6xx: 인젝션 문제 (계속) ───

# B610-B611: Django SQL 인젝션
Entry.objects.extra(where=[user_input])  # WARNING: B610

# B701: Jinja2 autoescape 비활성화
from jinja2 import Environment
env = Environment(autoescape=False)  # WARNING: B701
```

#### Bandit 구성 파일

```yaml
# .bandit.yaml (또는 setup.cfg [bandit] 섹션)

# 건너뛸 테스트
skips:
  - B101  # assert_used (테스트 파일에서는 허용)
  - B601  # paramiko (입력을 정제함)

# 제외할 경로
exclude_dirs:
  - tests
  - venv
  - .tox
  - migrations

# 심각도 임계값 설정
# 이 심각도 이상만 보고
severity: LOW

# 신뢰도 임계값 설정
confidence: LOW
```

#### Bandit 출력 해석하기

```bash
# 취약한 샘플 파일에 bandit 실행
$ bandit -r vulnerable_app.py

Run started:2025-01-15 10:30:00

Test results:
>> Issue: [B608:hardcoded_sql_expressions] Possible SQL injection vector
   through string-based query construction.
   Severity: Medium   Confidence: Low
   CWE: CWE-89 (https://cwe.mitre.org/data/definitions/89.html)
   Location: vulnerable_app.py:42:0
   More Info: https://bandit.readthedocs.io/en/latest/plugins/b608...
41	    user_id = request.args.get('id')
42	    query = f"SELECT * FROM users WHERE id = '{user_id}'"
43	    cursor.execute(query)

>> Issue: [B105:hardcoded_password_string] Possible hardcoded password
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   Location: vulnerable_app.py:15:0
14
15	    DATABASE_PASSWORD = "admin123"
16


--------------------------------------------------
Code scanned:
    Total lines of code: 156
    Total lines skipped (#nosec): 0

Run metrics:
    Total issues (by severity):
        Undefined: 0
        Low: 1
        Medium: 1
        High: 0
    Total issues (by confidence):
        Undefined: 0
        Low: 1
        Medium: 1
        High: 0
Files skipped (0):
```

#### 위양성 억제하기

```python
# 방법 1: #nosec로 인라인 억제
import hashlib
# 이 MD5는 비보안 체크섬용이며, 비밀번호 해싱이 아님
checksum = hashlib.md5(file_content).hexdigest()  # nosec B303

# 방법 2: 특정 테스트 ID로 인라인
password_hash = hashlib.sha256(salt + password)  # nosec B303

# 방법 3: 베이스라인 파일 사용
# 베이스라인 생성 (현재 문제 캡처)
# bandit -r ./myproject/ -f json -o baseline.json
# 베이스라인에 대해 실행 (새로운 문제만 표시)
# bandit -r ./myproject/ -b baseline.json
```

### 2.3 Semgrep: 다중 언어 정적 분석

Semgrep은 30개 이상의 언어를 지원하는 빠른 오픈소스 SAST 도구로, 이해하고 확장하기 쉬운 패턴 매칭 접근 방식을 사용합니다.

#### 설치 및 기본 사용법

```bash
# Semgrep 설치
pip install semgrep

# 기본 규칙으로 실행
semgrep --config auto .

# 특정 규칙셋으로 실행
semgrep --config p/python .
semgrep --config p/flask .
semgrep --config p/django .
semgrep --config p/owasp-top-ten .
semgrep --config p/security-audit .

# 로컬 규칙 파일로 실행
semgrep --config my_rules.yaml .

# 출력 형식
semgrep --config auto . --json > report.json
semgrep --config auto . --sarif > report.sarif
```

#### 맞춤형 Semgrep 규칙 작성

```yaml
# custom_rules.yaml

rules:
  # 규칙 1: f-string을 통한 SQL 인젝션 탐지
  - id: sql-injection-fstring
    patterns:
      - pattern: |
          $CURSOR.execute(f"...{$VAR}...")
    message: >
      f-string 보간을 통한 잠재적 SQL 인젝션.
      대신 파라미터화된 쿼리 사용:
      cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    languages: [python]
    severity: ERROR
    metadata:
      cwe:
        - CWE-89
      owasp:
        - A03:2021 Injection
      category: security
      technology:
        - python

  # 규칙 2: 하드코딩된 JWT 시크릿 탐지
  - id: hardcoded-jwt-secret
    patterns:
      - pattern: |
          jwt.encode($PAYLOAD, "...", ...)
      - pattern-not: |
          jwt.encode($PAYLOAD, $CONFIG, ...)
    message: >
      JWT 토큰이 하드코딩된 시크릿으로 서명됨.
      환경 변수 또는 시크릿 관리 서비스 사용.
    languages: [python]
    severity: ERROR
    metadata:
      cwe:
        - CWE-798

  # 규칙 3: 로그인에 rate limiting 누락 탐지
  - id: login-no-rate-limit
    patterns:
      - pattern: |
          @$APP.route("/login", ...)
          def $FUNC(...):
              ...
      - pattern-not-inside: |
          @limiter.limit(...)
          @$APP.route("/login", ...)
          def $FUNC(...):
              ...
    message: >
      rate limiting 없는 로그인 엔드포인트. 브루트포스 공격을 방지하기 위해
      @limiter.limit() 추가.
    languages: [python]
    severity: WARNING

  # 규칙 4: 사용자 입력과 함께 eval/exec 탐지
  - id: dangerous-eval-user-input
    patterns:
      - pattern-either:
          - pattern: eval(request.$METHOD.get(...))
          - pattern: exec(request.$METHOD.get(...))
          - pattern: |
              $X = request.$METHOD.get(...)
              ...
              eval($X)
          - pattern: |
              $X = request.$METHOD.get(...)
              ...
              exec($X)
    message: >
      사용자 입력이 eval()/exec()에 전달됨. 이는 임의 코드 실행을 허용함.
      신뢰할 수 없는 입력과 함께 eval/exec 절대 사용 금지.
    languages: [python]
    severity: ERROR
    metadata:
      cwe:
        - CWE-95

  # 규칙 5: Flask 폼에서 CSRF 보호 누락 탐지
  - id: flask-form-no-csrf
    patterns:
      - pattern: |
          @$APP.route("...", methods=[..., "POST", ...])
          def $FUNC(...):
              ...
              $X = request.form[...]
              ...
      - pattern-not-inside: |
          @csrf.exempt
          ...
    message: >
      POST 엔드포인트가 폼 데이터를 처리함. Flask-WTF 또는 수동 토큰 검증을 통해
      CSRF 보호가 활성화되어 있는지 확인.
    languages: [python]
    severity: WARNING
```

#### 맞춤형 규칙 실행

```bash
# 맞춤형 규칙 테스트
semgrep --config custom_rules.yaml ./myproject/

# 맞춤형 규칙과 표준 규칙셋 결합
semgrep --config custom_rules.yaml --config p/python ./myproject/

# 특정 파일에 대해 규칙 테스트
semgrep --config custom_rules.yaml target_file.py

# 규칙 구문 검증
semgrep --validate --config custom_rules.yaml
```

#### 고급 Semgrep 패턴

```yaml
rules:
  # 오염 추적: 소스에서 싱크까지 데이터 추적
  - id: flask-ssrf
    mode: taint
    pattern-sources:
      - patterns:
          - pattern: request.$METHOD.get(...)
    pattern-sinks:
      - patterns:
          - pattern: requests.get($URL, ...)
    message: >
      사용자 입력이 HTTP 요청으로 흐름, SSRF 가능성.
    languages: [python]
    severity: ERROR

  # 메타변수 비교
  - id: weak-rsa-key
    patterns:
      - pattern: rsa.generate_private_key(public_exponent=65537, key_size=$SIZE)
      - metavariable-comparison:
          metavariable: $SIZE
          comparison: $SIZE < 2048
    message: RSA 키 크기 $SIZE가 너무 작음. 최소 2048 비트 사용.
    languages: [python]
    severity: ERROR

  # 특정 메타변수에 초점을 맞춘 패턴
  - id: unvalidated-redirect
    patterns:
      - pattern: redirect($URL)
      - pattern-not: redirect(url_for(...))
      - focus-metavariable: $URL
    message: 잠재적 오픈 리다이렉트. 안전한 리다이렉트를 위해 url_for() 사용.
    languages: [python]
    severity: WARNING
```

### 2.4 SonarQube 개요

SonarQube는 지속적인 코드 품질 및 보안 검사를 위한 엔터프라이즈급 플랫폼입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SonarQube 아키텍처                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  개발자 머신                    SonarQube 서버                    │
│  ┌──────────────┐              ┌──────────────────┐             │
│  │  소스 코드    │    스캔     │  ┌─────────────┐  │             │
│  │      +        │ ────────►  │  │   분석기     │  │             │
│  │ sonar-scanner │             │  │   엔진       │  │             │
│  └──────────────┘              │  └──────┬──────┘  │             │
│                                │         │         │             │
│  CI/CD 파이프라인              │         ▼         │             │
│  ┌──────────────┐              │  ┌─────────────┐  │             │
│  │  빌드 스텝    │   리포트   │  │   데이터베이스│  │             │
│  │  + 스캐너     │ ────────►  │  │ (PostgreSQL) │  │             │
│  └──────────────┘              │  └──────┬──────┘  │             │
│                                │         │         │             │
│                                │         ▼         │             │
│  웹 브라우저                   │  ┌─────────────┐  │             │
│  ┌──────────────┐              │  │  Web UI /    │  │             │
│  │  대시보드     │ ◄────────  │  │  Dashboard   │  │             │
│  │  & 리포트     │             │  └─────────────┘  │             │
│  └──────────────┘              └──────────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```bash
# Docker로 SonarQube 실행
docker run -d --name sonarqube \
  -p 9000:9000 \
  sonarqube:community

# 프로젝트 구성: sonar-project.properties
# sonar.projectKey=my-python-project
# sonar.sources=src
# sonar.python.version=3.11
# sonar.exclusions=**/tests/**,**/migrations/**

# 스캐너 실행
sonar-scanner \
  -Dsonar.projectKey=my-python-project \
  -Dsonar.sources=. \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.token=your_token_here
```

---

## 3. 동적 애플리케이션 보안 테스팅 (DAST)

### 3.1 DAST 동작 원리

DAST 도구는 조작된 요청을 보내고 응답을 분석하여 취약점을 찾는 방식으로 실행 중인 애플리케이션을 테스트합니다. 자동화된 공격자처럼 작동합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                     DAST 테스팅 플로우                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────┐     크롤링     ┌───────────────┐                │
│  │   DAST    │ ─────────────► │   실행 중인    │                │
│  │   도구    │                │   애플리케이션 │                │
│  │           │ ◄───────────── │   (대상)       │                │
│  │           │    응답        │               │                │
│  │           │                └───────────────┘                │
│  │           │                                                  │
│  │  단계 1:  │  엔드포인트 발견을 위한 스파이더/크롤링         │
│  │  발견     │  폼, 파라미터, API 엔드포인트 찾기              │
│  │           │                                                  │
│  │  단계 2:  │  악의적인 페이로드 전송:                        │
│  │  공격     │  - SQL 인젝션 문자열                            │
│  │           │  - XSS 페이로드                                  │
│  │           │  - 경로 순회 시도                                │
│  │           │  - 명령 인젝션                                   │
│  │           │                                                  │
│  │  단계 3:  │  응답 분석:                                      │
│  │  분석     │  - 정보 노출 에러 메시지                        │
│  │           │  - 반사된 입력 (XSS)                            │
│  │           │  - 예상치 못한 동작                              │
│  └───────────┘                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 OWASP ZAP

OWASP ZAP (Zed Attack Proxy)는 가장 널리 사용되는 무료 DAST 도구입니다.

```bash
# Docker로 ZAP 실행
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t http://target-app:8080

# 전체 스캔 (더 철저하지만 느림)
docker run -t owasp/zap2docker-stable zap-full-scan.py \
  -t http://target-app:8080

# API 스캔 (REST API용)
docker run -t owasp/zap2docker-stable zap-api-scan.py \
  -t http://target-app:8080/openapi.json \
  -f openapi

# HTML 리포트 생성
docker run -v $(pwd):/zap/wrk/:rw \
  -t owasp/zap2docker-stable zap-baseline.py \
  -t http://target-app:8080 \
  -r report.html
```

#### ZAP Python API

```python
"""
자동화된 스캐닝을 위한 OWASP ZAP의 Python API 사용.
필요: pip install python-owasp-zap-v2.4
ZAP이 데몬으로 실행 중이어야 함.
"""

from zapv2 import ZAPv2
import time

# ZAP에 연결
zap = ZAPv2(
    apikey='your-api-key',
    proxies={
        'http': 'http://127.0.0.1:8080',
        'https': 'http://127.0.0.1:8080'
    }
)

target = 'http://target-app:5000'

def run_zap_scan(target_url: str) -> dict:
    """대상에 대해 ZAP 스캔 실행 및 결과 반환."""

    print(f"[*] 대상 스파이더링: {target_url}")
    scan_id = zap.spider.scan(target_url)

    # 스파이더 완료 대기
    while int(zap.spider.status(scan_id)) < 100:
        print(f"    스파이더 진행률: {zap.spider.status(scan_id)}%")
        time.sleep(2)

    print(f"[*] 스파이더가 {len(zap.spider.results(scan_id))}개 URL 발견")

    # 능동 스캔 실행
    print(f"[*] 능동 스캔 시작...")
    scan_id = zap.ascan.scan(target_url)

    while int(zap.ascan.status(scan_id)) < 100:
        print(f"    능동 스캔 진행률: {zap.ascan.status(scan_id)}%")
        time.sleep(5)

    # 경고 가져오기
    alerts = zap.core.alerts(baseurl=target_url)

    # 위험 수준별로 분류
    results = {
        'High': [],
        'Medium': [],
        'Low': [],
        'Informational': []
    }

    for alert in alerts:
        risk = alert.get('risk', 'Informational')
        results[risk].append({
            'name': alert.get('name'),
            'url': alert.get('url'),
            'description': alert.get('description'),
            'solution': alert.get('solution'),
            'cweid': alert.get('cweid'),
        })

    return results


def print_results(results: dict) -> None:
    """스캔 결과를 읽기 쉬운 형식으로 출력."""
    for risk_level in ['High', 'Medium', 'Low', 'Informational']:
        alerts = results[risk_level]
        if alerts:
            print(f"\n{'='*60}")
            print(f"  {risk_level} 위험 경고: {len(alerts)}")
            print(f"{'='*60}")
            for alert in alerts:
                print(f"\n  [{alert['cweid']}] {alert['name']}")
                print(f"  URL: {alert['url']}")
                print(f"  해결책: {alert['solution'][:100]}...")


if __name__ == '__main__':
    results = run_zap_scan(target)
    print_results(results)
```

### 3.3 Burp Suite 개념

Burp Suite는 무료 커뮤니티 에디션이 있는 상용 웹 보안 테스팅 플랫폼입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Burp Suite 아키텍처                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  브라우저 ◄──────► Burp 프록시 ◄──────► 대상 서버               │
│                      │                                           │
│           ┌──────────┼──────────────────┐                       │
│           │          │                  │                        │
│           ▼          ▼                  ▼                        │
│      ┌─────────┐ ┌─────────┐    ┌───────────┐                  │
│      │ Spider  │ │Repeater │    │  Scanner   │                  │
│      │(크롤링) │ │(수동     │    │(자동화)    │                  │
│      │         │ │ 테스트)  │    │            │                  │
│      └─────────┘ └─────────┘    └───────────┘                  │
│           │          │                  │                        │
│           ▼          ▼                  ▼                        │
│      ┌─────────┐ ┌─────────┐    ┌───────────┐                  │
│      │Sequencer│ │Intruder │    │  Decoder   │                  │
│      │(토큰    │ │(페이로드│    │(인코드/    │                  │
│      │ 테스트) │ │ 퍼저)   │    │ 디코드)    │                  │
│      └─────────┘ └─────────┘    └───────────┘                  │
│                                                                  │
│  주요 기능:                                                      │
│  - HTTP/HTTPS 트래픽 가로채기 및 수정                           │
│  - 자동화된 취약점 스캐닝                                        │
│  - Repeater 및 Intruder로 수동 테스팅                          │
│  - Sequencer로 세션 토큰 분석                                   │
│  - BApp Store를 통한 확장 가능                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 소프트웨어 구성 분석 (SCA)

### 4.1 SCA가 중요한 이유

대부분의 현대 애플리케이션은 70-90%의 서드파티 코드로 구성됩니다. SCA 도구는 의존성에서 알려진 취약점을 스캔합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                애플리케이션의 코드 구성                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  ██████████████████████████████████████████████████████  │   │
│  │  ██████████████████████████████████████████████████████  │   │
│  │  ██████████████ 서드파티 라이브러리 ████████████████████  │   │
│  │  ██████████████   (코드의 70-90%)  ████████████████████  │   │
│  │  ██████████████████████████████████████████████████████  │   │
│  │  ██████████████████████████████████████████████████████  │   │
│  │  ██████████████████████████████████████████████████████  │   │
│  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │  ░░░░░ 작성한 코드 (10-30%) ░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  각 서드파티 라이브러리는 자체 의존성을 가질 수 있으며           │
│  (전이 의존성), 깊은 의존성 트리를 생성합니다.                  │
│  이 트리의 어느 곳에서든 취약점은 애플리케이션에 영향을 줍니다. │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 pip-audit

```bash
# pip-audit 설치
pip install pip-audit

# 현재 환경 스캔
pip-audit

# requirements 파일 스캔
pip-audit -r requirements.txt

# JSON 형식으로 출력
pip-audit -f json -o audit_report.json

# 취약점 자동 수정 (패키지 업그레이드)
pip-audit --fix

# 특정 취약점 데이터베이스로 스캔
pip-audit --vulnerability-service osv  # Google OSV (기본값)
pip-audit --vulnerability-service pypi  # PyPI Advisory DB

# 엄격 모드: 취약점 발견 시 오류로 종료
pip-audit --strict
```

#### pip-audit 출력 예제

```
$ pip-audit -r requirements.txt

Found 3 known vulnerabilities in 2 packages

Name       Version  ID                  Fix Versions
---------- -------- ------------------- ---------------
flask      2.0.1    PYSEC-2023-62       2.3.2
requests   2.25.1   PYSEC-2023-74       2.31.0
requests   2.25.1   GHSA-j8r2-6x86-q33q 2.32.0
```

### 4.3 Safety

```bash
# safety 설치
pip install safety

# 현재 환경 검사
safety check

# requirements 파일 검사
safety check -r requirements.txt

# JSON 형식으로 출력
safety check --output json

# CI에서 사용 (취약점 발견 시 종료 코드 1)
safety check --full-report
```

### 4.4 Python 스크립트: 의존성 취약점 스캐너

```python
"""
dependency_scanner.py - 포괄적인 의존성 취약점 스캐너.
pip-audit 결과와 추가 검사를 결합합니다.
"""

import subprocess
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Vulnerability:
    """의존성의 단일 취약점을 나타냄."""
    package: str
    version: str
    vuln_id: str
    description: str = ""
    fix_version: Optional[str] = None
    severity: str = "UNKNOWN"
    aliases: list[str] = field(default_factory=list)


@dataclass
class ScanResult:
    """의존성 스캔 결과."""
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    scanned_packages: int = 0
    scan_tool: str = ""
    errors: list[str] = field(default_factory=list)


def run_pip_audit(requirements_file: Optional[str] = None) -> ScanResult:
    """pip-audit 실행 및 결과 파싱."""
    cmd = ["pip-audit", "-f", "json", "--desc"]
    if requirements_file:
        cmd.extend(["-r", requirements_file])

    result = ScanResult(scan_tool="pip-audit")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        data = json.loads(proc.stdout)

        # 의존성 파싱
        for dep in data.get("dependencies", []):
            result.scanned_packages += 1
            for vuln in dep.get("vulns", []):
                v = Vulnerability(
                    package=dep["name"],
                    version=dep["version"],
                    vuln_id=vuln["id"],
                    description=vuln.get("description", ""),
                    fix_version=vuln.get("fix_versions", [None])[0]
                        if vuln.get("fix_versions") else None,
                    aliases=vuln.get("aliases", [])
                )
                result.vulnerabilities.append(v)

    except FileNotFoundError:
        result.errors.append("pip-audit가 설치되지 않음. 실행: pip install pip-audit")
    except subprocess.TimeoutExpired:
        result.errors.append("pip-audit이 120초 후 타임아웃")
    except json.JSONDecodeError as e:
        result.errors.append(f"pip-audit 출력 파싱 실패: {e}")

    return result


def check_requirements_pinning(requirements_file: str) -> list[str]:
    """
    의존성이 정확한 버전으로 올바르게 고정되었는지 검사.
    고정되지 않은 의존성은 취약한 버전을 자동으로 가져올 수 있어
    보안 위험입니다.
    """
    warnings = []
    req_path = Path(requirements_file)

    if not req_path.exists():
        return [f"Requirements 파일을 찾을 수 없음: {requirements_file}"]

    for line_num, line in enumerate(req_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        # 고정되지 않은 의존성 검사
        if "==" not in line:
            if ">=" in line:
                warnings.append(
                    f"라인 {line_num}: '{line}'이 >=를 사용 (상한 미고정). "
                    f"정확한 고정을 위해 == 사용."
                )
            elif line.isidentifier() or "." in line:
                warnings.append(
                    f"라인 {line_num}: '{line}'에 버전 고정 없음. "
                    f"정확한 버전을 고정하기 위해 == 사용."
                )

    return warnings


def check_known_malicious_packages(requirements_file: str) -> list[str]:
    """
    알려진 타이포스쿼팅 / 악성 패키지 이름 검사.
    이는 단순화된 검사 - 실제 스캐너는 더 큰 데이터베이스를 사용.
    """
    # 알려진 타이포스쿼팅 예제 (단순화된 목록)
    SUSPICIOUS_PATTERNS = {
        "python-dateutil": "dateutil",       # 일반적인 혼동
        "beautifulsoup4": "beautifulsoup",   # 구버전
        # 알려진 악성 패키지 예제 (현재 PyPI에서 제거됨)
        "colourama": "colorama",
        "python3-dateutil": "python-dateutil",
        "jeIlyfish": "jellyfish",            # 대문자 I vs 소문자 l
    }

    warnings = []
    req_path = Path(requirements_file)

    if not req_path.exists():
        return []

    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip()

        if pkg_name.lower() in [k.lower() for k in SUSPICIOUS_PATTERNS]:
            correct = SUSPICIOUS_PATTERNS.get(pkg_name, "unknown")
            warnings.append(
                f"경고: '{pkg_name}'은 '{correct}'의 타이포스쿼팅일 수 있음. "
                f"패키지 이름이 올바른지 확인."
            )

    return warnings


def generate_report(scan_result: ScanResult, pinning_warnings: list[str],
                    typosquat_warnings: list[str]) -> str:
    """형식화된 보안 리포트 생성."""
    lines = []
    lines.append("=" * 60)
    lines.append("  의존성 보안 스캔 리포트")
    lines.append("=" * 60)
    lines.append(f"\n도구: {scan_result.scan_tool}")
    lines.append(f"스캔된 패키지: {scan_result.scanned_packages}")
    lines.append(f"발견된 취약점: {len(scan_result.vulnerabilities)}")

    if scan_result.errors:
        lines.append(f"\n오류:")
        for err in scan_result.errors:
            lines.append(f"  [!] {err}")

    if scan_result.vulnerabilities:
        lines.append(f"\n{'─' * 60}")
        lines.append("  취약점")
        lines.append(f"{'─' * 60}")

        for vuln in scan_result.vulnerabilities:
            lines.append(f"\n  패키지: {vuln.package} {vuln.version}")
            lines.append(f"  ID:      {vuln.vuln_id}")
            if vuln.aliases:
                lines.append(f"  별칭: {', '.join(vuln.aliases)}")
            if vuln.fix_version:
                lines.append(f"  수정:     {vuln.fix_version}로 업그레이드")
            if vuln.description:
                desc = vuln.description[:200]
                lines.append(f"  상세:  {desc}")

    if pinning_warnings:
        lines.append(f"\n{'─' * 60}")
        lines.append("  버전 고정 경고")
        lines.append(f"{'─' * 60}")
        for w in pinning_warnings:
            lines.append(f"  [!] {w}")

    if typosquat_warnings:
        lines.append(f"\n{'─' * 60}")
        lines.append("  타이포스쿼팅 경고")
        lines.append(f"{'─' * 60}")
        for w in typosquat_warnings:
            lines.append(f"  [!] {w}")

    lines.append(f"\n{'=' * 60}")

    # 종료 권장사항 결정
    if scan_result.vulnerabilities or typosquat_warnings:
        lines.append("  결과: 실패 - 주의가 필요한 문제 발견")
    elif pinning_warnings:
        lines.append("  결과: 경고 - 고정 문제 수정 고려")
    else:
        lines.append("  결과: 통과 - 문제 없음")

    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    req_file = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"

    print(f"[*] 의존성 스캔 중: {req_file}")

    # 모든 검사 실행
    scan_result = run_pip_audit(req_file)
    pinning_warnings = check_requirements_pinning(req_file)
    typosquat_warnings = check_known_malicious_packages(req_file)

    # 리포트 생성 및 출력
    report = generate_report(scan_result, pinning_warnings, typosquat_warnings)
    print(report)

    # CI를 위한 적절한 코드로 종료
    if scan_result.vulnerabilities or typosquat_warnings:
        sys.exit(1)
    elif scan_result.errors:
        sys.exit(2)
    else:
        sys.exit(0)
```

### 4.5 Dependabot 구성 (GitHub)

```yaml
# .github/dependabot.yml

version: 2
updates:
  # Python pip 의존성
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "America/New_York"
    open-pull-requests-limit: 10
    reviewers:
      - "security-team"
    labels:
      - "dependencies"
      - "security"
    # 마이너/패치 업데이트를 함께 그룹화
    groups:
      minor-and-patch:
        update-types:
          - "minor"
          - "patch"
    # 특정 패키지 무시
    ignore:
      - dependency-name: "boto3"
        update-types: ["version-update:semver-patch"]
    # 보안 업데이트만 (버전 업데이트 없음)
    # 보안만을 위해서는 아래 주석 해제 및 스케줄 제거
    # open-pull-requests-limit: 0

  # Docker 베이스 이미지
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

---

## 5. 퍼징

### 5.1 퍼징이란?

퍼징은 프로그램에 랜덤하거나, 잘못된 형식이거나, 예상치 못한 입력을 공급하여 크래시, 행(hang), 보안 취약점을 찾는 자동화된 테스팅 기법입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      퍼징 피드백 루프                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐     ┌────────────┐     ┌────────────────┐       │
│  │   시드      │     │  변이       │     │    대상         │       │
│  │   코퍼스    │────►│  엔진       │────►│   프로그램      │       │
│  │  (초기      │     │            │     │                │       │
│  │   입력)     │     │ - 비트 플립│     │ 입력 파싱      │       │
│  └────────────┘     │ - 삽입     │     │ 데이터 처리    │       │
│       ▲             │ - 삭제     │     │ 결과 반환      │       │
│       │             │ - 교체     │     └───────┬────────┘       │
│       │             └────────────┘             │                │
│       │                                        │                │
│       │         ┌──────────────┐               │                │
│       │         │  커버리지    │◄──────────────┘                │
│       └─────────│  모니터      │  (코드 커버리지 피드백)        │
│  (새로운 경로를 │              │                                 │
│   찾은 입력     └──────────────┘                                │
│   저장)              │                                           │
│                      ▼                                           │
│                ┌──────────────┐                                  │
│                │  크래시/버그 │                                  │
│                │  탐지        │                                  │
│                └──────────────┘                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 AFL (American Fuzzy Lop)

AFL은 C/C++ 프로그램을 위한 가장 영향력 있는 커버리지 가이드 퍼저입니다.

```bash
# AFL++ 설치
sudo apt-get install afl++  # Debian/Ubuntu

# AFL 계측으로 대상 컴파일
afl-cc -o target_program target_program.c

# 시드 코퍼스 디렉토리 생성
mkdir -p seeds
echo "valid input" > seeds/seed1.txt

# AFL 실행
afl-fuzz -i seeds -o findings ./target_program @@

# @@는 입력 파일 경로로 대체됨
# -i: 시드 디렉토리
# -o: 출력 디렉토리 (크래시, 행, 큐)

# AFL 상태 모니터링
afl-whatsup findings/
```

#### AFL 출력 디렉토리 구조

```
findings/
├── crashes/         # 크래시를 발생시킨 입력
│   ├── id:000000,...  # 크래시 유발 입력
│   └── README.txt
├── hangs/           # 행/타임아웃을 발생시킨 입력
├── queue/           # 흥미로운 입력 (새로운 커버리지)
└── fuzzer_stats     # 현재 퍼징 통계
```

### 5.3 Hypothesis: Python을 위한 속성 기반 테스팅

Hypothesis는 속성 기반 테스팅을 위한 Python 라이브러리입니다. 전통적인 퍼저는 아니지만, 엣지 케이스를 찾기 위해 자동으로 테스트 입력을 생성합니다.

```python
"""
Hypothesis를 이용한 속성 기반 테스팅.
설치: pip install hypothesis
"""

from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st
import json
import re


# ─── 기본 예제: 생성된 입력으로 함수 테스팅 ───

def encode_decode_round_trip(data: str) -> bool:
    """인코딩 후 디코딩하면 원본 데이터를 반환해야 함."""
    encoded = data.encode('utf-8')
    decoded = encoded.decode('utf-8')
    return decoded == data


@given(st.text())
def test_encode_decode_roundtrip(s):
    """UTF-8 인코드/디코드가 완벽한 왕복임을 테스트."""
    assert encode_decode_round_trip(s)


# ─── JSON 파싱 견고성 테스팅 ───

@given(st.text())
def test_json_loads_doesnt_crash(s):
    """
    json.loads는 성공적으로 파싱하거나
    ValueError/JSONDecodeError를 발생시켜야 함 - 절대 크래시나 행 없음.
    """
    try:
        json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass  # 유효하지 않은 JSON에 대해 예상됨


# ─── 입력 검증 함수 테스팅 ───

def validate_email(email: str) -> bool:
    """간단한 이메일 검증 (데모를 위해 의도적으로 버그가 있음)."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


@given(st.emails())
def test_valid_emails_pass_validation(email):
    """모든 유효한 이메일은 검증기를 통과해야 함."""
    # 이것은 실패할 가능성이 높음, 정규식의 갭을 노출함
    assert validate_email(email), f"유효한 이메일 거부됨: {email}"


# ─── 구조화된 데이터로 테스팅 ───

# 사용자 등록 데이터 생성을 위한 전략
user_strategy = st.fixed_dictionaries({
    'username': st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N')),
        min_size=1,
        max_size=50
    ),
    'email': st.emails(),
    'password': st.text(min_size=8, max_size=128),
    'age': st.integers(min_value=0, max_value=200),
})


def process_registration(data: dict) -> dict:
    """사용자 등록 처리. 모든 유효한 입력을 처리해야 함."""
    if len(data['username']) < 1:
        raise ValueError("사용자명이 너무 짧음")
    if len(data['password']) < 8:
        raise ValueError("비밀번호가 너무 짧음")
    if data['age'] < 13:
        raise ValueError("최소 13세 이상이어야 함")

    return {
        'username': data['username'].lower(),
        'email': data['email'].lower(),
        'status': 'registered'
    }


@given(user_strategy)
def test_registration_never_crashes(user_data):
    """등록은 성공하거나 ValueError를 발생시켜야 함."""
    try:
        result = process_registration(user_data)
        assert 'username' in result
        assert 'email' in result
    except ValueError:
        pass  # 유효하지 않은 데이터에 대해 예상됨


# ─── URL 파서 퍼징 ───

from urllib.parse import urlparse, urlunparse

@given(st.from_regex(
    r'https?://[a-z0-9.-]{1,50}(:[0-9]{1,5})?(/[a-z0-9._-]{0,20}){0,5}(\?[a-z0-9=&]{0,50})?',
    fullmatch=True
))
def test_url_parse_roundtrip(url):
    """URL 파싱 후 언파싱은 URL을 보존해야 함."""
    parsed = urlparse(url)
    reconstructed = urlunparse(parsed)
    # 컴포넌트 비교를 위해 둘 다 재파싱 (정규화가 다를 수 있음)
    assert urlparse(reconstructed).netloc == parsed.netloc


# ─── 보안에 민감한 함수 테스팅 ───

def sanitize_filename(filename: str) -> str:
    """파일명에서 위험한 문자 제거."""
    # 경로 구분자 및 널 바이트 제거
    sanitized = filename.replace('/', '').replace('\\', '')
    sanitized = sanitized.replace('\x00', '')
    sanitized = sanitized.replace('..', '')
    # 선행 점 제거 (숨김 파일)
    sanitized = sanitized.lstrip('.')
    return sanitized or 'unnamed'


@given(st.text(min_size=1, max_size=255))
def test_sanitized_filename_is_safe(filename):
    """정제된 파일명은 절대 경로 순회를 포함하지 않아야 함."""
    result = sanitize_filename(filename)
    assert '/' not in result, f"경로 구분자 포함: {result}"
    assert '\\' not in result, f"백슬래시 포함: {result}"
    assert '\x00' not in result, f"널 바이트 포함: {result}"
    assert not result.startswith('.'), f"숨김 파일: {result}"
    assert '..' not in result, f"경로 순회 포함: {result}"
    assert len(result) > 0, "정제 후 빈 파일명"


# ─── 고급: 상태 기반 테스팅 ───

from hypothesis.stateful import RuleBasedStateMachine, rule, initialize

class ShoppingCartStateMachine(RuleBasedStateMachine):
    """
    쇼핑 카트를 위한 상태 기반 테스팅.
    Hypothesis가 작업 시퀀스를 생성하고
    각 단계 후 불변성을 검사함.
    """

    def __init__(self):
        super().__init__()
        self.cart = {}
        self.total = 0.0

    @initialize()
    def init_cart(self):
        self.cart = {}
        self.total = 0.0

    @rule(
        item=st.text(min_size=1, max_size=20),
        price=st.floats(min_value=0.01, max_value=10000, allow_nan=False),
        quantity=st.integers(min_value=1, max_value=100)
    )
    def add_item(self, item, price, quantity):
        """카트에 아이템 추가."""
        if item in self.cart:
            self.cart[item]['quantity'] += quantity
        else:
            self.cart[item] = {'price': price, 'quantity': quantity}
        self._recalculate_total()

    @rule(item=st.text(min_size=1, max_size=20))
    def remove_item(self, item):
        """카트에서 아이템 제거."""
        if item in self.cart:
            del self.cart[item]
            self._recalculate_total()

    def _recalculate_total(self):
        self.total = sum(
            v['price'] * v['quantity']
            for v in self.cart.values()
        )

    def teardown(self):
        """불변성: 합계는 절대 음수가 아니어야 함."""
        assert self.total >= 0, f"음수 합계: {self.total}"
        assert len(self.cart) >= 0


# 상태 머신에서 테스트 생성
TestShoppingCart = ShoppingCartStateMachine.TestCase


# ─── 설정과 함께 Hypothesis 실행 ───

@settings(
    max_examples=1000,        # 생성할 테스트 케이스 수
    deadline=None,            # 테스트당 시간 제한 없음
    suppress_health_check=[   # 특정 건강 검사 억제
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
    ],
)
@given(st.binary(min_size=1, max_size=1024))
def test_binary_processing(data):
    """바이너리 프로세서가 모든 입력을 처리하는지 테스트."""
    # 여기에 바이너리 처리 함수
    try:
        result = data.decode('utf-8', errors='replace')
        assert isinstance(result, str)
    except Exception as e:
        # errors='replace'로는 여기에 절대 도달하지 않아야 함
        raise AssertionError(f"예상치 못한 오류: {e}")
```

### 5.4 네트워크 프로토콜 퍼징

```python
"""
교육 목적을 위한 간단한 프로토콜 퍼저.
프로토콜 테스팅을 위한 잘못된 형식의 입력 생성.
"""

import random
import struct
import socket
from typing import Generator


def mutate_bytes(data: bytes, num_mutations: int = 5) -> bytes:
    """바이트 문자열에 랜덤 변이 적용."""
    data = bytearray(data)

    for _ in range(num_mutations):
        mutation_type = random.choice([
            'bit_flip', 'byte_replace', 'insert', 'delete',
            'duplicate', 'overflow'
        ])

        if len(data) == 0:
            data = bytearray(random.randbytes(10))
            continue

        pos = random.randint(0, max(0, len(data) - 1))

        if mutation_type == 'bit_flip':
            bit = random.randint(0, 7)
            data[pos] ^= (1 << bit)

        elif mutation_type == 'byte_replace':
            # 흥미로운 값으로 교체
            interesting = [0x00, 0x01, 0x7F, 0x80, 0xFF, 0xFE]
            data[pos] = random.choice(interesting)

        elif mutation_type == 'insert':
            insert_data = random.randbytes(random.randint(1, 10))
            data[pos:pos] = insert_data

        elif mutation_type == 'delete':
            del_len = random.randint(1, min(5, len(data) - pos))
            del data[pos:pos + del_len]

        elif mutation_type == 'duplicate':
            chunk = data[pos:pos + random.randint(1, 10)]
            data[pos:pos] = chunk

        elif mutation_type == 'overflow':
            # 매우 긴 문자열 삽입
            overflow = b'A' * random.choice([256, 1024, 4096, 65536])
            data[pos:pos] = overflow

    return bytes(data)


def generate_http_fuzz_requests(host: str, port: int) -> Generator[bytes, None, None]:
    """퍼징된 HTTP 요청 생성."""

    base_requests = [
        f"GET / HTTP/1.1\r\nHost: {host}\r\n\r\n".encode(),
        f"POST /login HTTP/1.1\r\nHost: {host}\r\nContent-Length: 10\r\n\r\nuser=admin".encode(),
        f"GET /{'A' * 5000} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode(),
    ]

    # 원본 요청 양보
    for req in base_requests:
        yield req

    # 변이된 버전 양보
    for _ in range(100):
        base = random.choice(base_requests)
        yield mutate_bytes(base)

    # 특수 케이스
    yield b"\x00" * 1024                        # 널 바이트
    yield b"GET / HTTP/9.9\r\n\r\n"             # 유효하지 않은 버전
    yield b"XYZZY / HTTP/1.1\r\n\r\n"           # 유효하지 않은 메서드
    yield b"GET / HTTP/1.1\r\n" + b"X: Y\r\n" * 10000 + b"\r\n"  # 헤더 폭탄
    yield b"GET / HTTP/1.1\r\nContent-Length: -1\r\n\r\n"         # 음수 길이
    yield b"GET / HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n"  # 거대한 길이


def fuzz_target(host: str, port: int, timeout: float = 2.0) -> None:
    """
    대상 서버에 퍼징된 요청 전송.

    경고: 소유하거나 명시적 테스트 권한이 있는 서버에만 사용.
    무단 테스팅은 불법입니다.
    """
    print(f"[*] 퍼징 중 {host}:{port}")
    crash_count = 0
    total_sent = 0

    for payload in generate_http_fuzz_requests(host, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.sendall(payload)

            try:
                response = sock.recv(4096)
                total_sent += 1
            except socket.timeout:
                print(f"  [!] 페이로드 #{total_sent}에서 타임아웃 "
                      f"(길이={len(payload)})")
                total_sent += 1

        except ConnectionRefusedError:
            crash_count += 1
            print(f"  [!!!] 페이로드 #{total_sent} 후 연결 거부됨. "
                  f"서버가 크래시했을 수 있음!")
            print(f"       페이로드 (처음 100 바이트): {payload[:100]}")
            # 크래시 유발 페이로드 저장
            with open(f"crash_{crash_count}.bin", "wb") as f:
                f.write(payload)

        except Exception as e:
            print(f"  [!] 오류: {e}")

        finally:
            sock.close()

    print(f"\n[*] 퍼징 완료. {total_sent}개 페이로드 전송. "
          f"탐지된 크래시: {crash_count}")
```

---

## 6. 침투 테스팅 방법론

### 6.1 침투 테스팅 프로세스

```
┌─────────────────────────────────────────────────────────────────┐
│              침투 테스팅 방법론                                  │
│              (PTES / OWASP 기반)                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐                                       │
│  │ 1. 계획 및 범위      │  대상, 교전 규칙 정의                 │
│  │    (사전 교전)       │  법적 승인, 경계                      │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │ 2. 정찰              │  수동: OSINT, DNS, WHOIS              │
│  │    (정보             │  능동: 포트 스캔, 서비스 열거         │
│  │     수집)            │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │ 3. 취약점            │  자동화 스캐닝 (Nessus, ZAP)          │
│  │    평가              │  수동 테스팅, 설정 오류               │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │ 4. 익스플로잇        │  취약점 익스플로잇 시도               │
│  │                      │  접근 획득, 권한 상승                 │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │ 5. 익스플로잇 후     │  측면 이동, 데이터 유출 테스트        │
│  │                      │  지속성 메커니즘                      │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │ 6. 보고              │  경영진 요약, 기술 상세               │
│  │                      │  개선 권장사항                        │
│  └──────────────────────┘                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 웹 애플리케이션 침투 테스팅 체크리스트

```
┌──────────────────────────────────────────────────────────────────┐
│              웹 애플리케이션 침투 테스팅 체크리스트               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  인증:                                                            │
│  [ ] 브루트포스 보호 (계정 잠금, 속도 제한)                       │
│  [ ] 비밀번호 복잡도 강제                                         │
│  [ ] 다중 인증                                                    │
│  [ ] 세션 관리 (타임아웃, 로테이션, 보안 플래그)                  │
│  [ ] 비밀번호 재설정 메커니즘                                     │
│  [ ] 기본 인증 정보                                               │
│                                                                   │
│  권한 부여:                                                       │
│  [ ] 수평 권한 상승 (다른 사용자 데이터 접근)                     │
│  [ ] 수직 권한 상승 (관리자 기능)                                 │
│  [ ] IDOR (안전하지 않은 직접 객체 참조)                          │
│  [ ] 기능 수준 접근 제어 누락                                     │
│                                                                   │
│  입력 검증:                                                       │
│  [ ] SQL 인젝션 (모든 입력 지점)                                  │
│  [ ] XSS (반사형, 저장형, DOM 기반)                               │
│  [ ] 명령 인젝션                                                  │
│  [ ] 경로 순회 / LFI / RFI                                        │
│  [ ] XML 외부 엔티티 (XXE)                                        │
│  [ ] 서버 측 요청 위조 (SSRF)                                     │
│  [ ] 템플릿 인젝션 (SSTI)                                         │
│                                                                   │
│  구성:                                                            │
│  [ ] HTTPS 강제                                                   │
│  [ ] 보안 헤더 (CSP, HSTS, X-Frame-Options 등)                   │
│  [ ] CORS 정책                                                    │
│  [ ] 에러 처리 (프로덕션에서 스택 트레이스 없음)                  │
│  [ ] 디렉토리 목록 비활성화                                       │
│  [ ] 불필요한 기능/페이지 제거                                    │
│                                                                   │
│  비즈니스 로직:                                                   │
│  [ ] 경쟁 조건                                                    │
│  [ ] 가격 조작                                                    │
│  [ ] 워크플로우 우회                                              │
│  [ ] 대량 할당                                                    │
│                                                                   │
│  API 특화:                                                        │
│  [ ] API 키 노출                                                  │
│  [ ] 속도 제한                                                    │
│  [ ] 과도한 데이터 노출                                           │
│  [ ] 리소스 제한 부족                                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 Python 침투 테스팅 헬퍼

```python
"""
침투 테스팅 헬퍼 함수.
승인된 테스팅용으로만 사용.
"""

import requests
import urllib.parse
from typing import Optional


# ─── SQL 인젝션 테스팅 ───

SQL_INJECTION_PAYLOADS = [
    "' OR '1'='1",
    "' OR '1'='1' --",
    "' OR '1'='1' /*",
    "1; DROP TABLE users --",
    "' UNION SELECT NULL,NULL,NULL --",
    "' AND 1=1 --",
    "' AND 1=2 --",
    "admin'--",
    "1' ORDER BY 1 --",
    "1' ORDER BY 100 --",
    "-1 OR 1=1",
    "' OR ''='",
    "'; WAITFOR DELAY '0:0:5' --",  # Time-based blind SQLi (MSSQL)
    "' OR SLEEP(5) --",              # Time-based blind SQLi (MySQL)
]


def test_sql_injection(url: str, param_name: str,
                       method: str = "GET") -> list[dict]:
    """
    URL 파라미터의 SQL 인젝션 취약점 테스트.
    잠재적으로 취약한 페이로드 목록 반환.
    """
    results = []

    # 먼저 베이스라인 응답 가져오기
    if method == "GET":
        baseline = requests.get(url, params={param_name: "1"}, timeout=10)
    else:
        baseline = requests.post(url, data={param_name: "1"}, timeout=10)

    baseline_length = len(baseline.text)
    baseline_time = baseline.elapsed.total_seconds()

    for payload in SQL_INJECTION_PAYLOADS:
        try:
            if method == "GET":
                resp = requests.get(
                    url, params={param_name: payload}, timeout=15
                )
            else:
                resp = requests.post(
                    url, data={param_name: payload}, timeout=15
                )

            # SQL 인젝션 징후 검사
            indicators = []

            # 에러 기반: 응답의 SQL 에러 메시지
            sql_errors = [
                "sql syntax", "mysql", "sqlite", "postgresql",
                "ora-", "unclosed quotation", "unterminated string",
                "syntax error"
            ]
            for err in sql_errors:
                if err in resp.text.lower():
                    indicators.append(f"SQL 에러 메시지: '{err}'")

            # 불린 기반: 상당한 길이 차이
            length_diff = abs(len(resp.text) - baseline_length)
            if length_diff > baseline_length * 0.3:
                indicators.append(
                    f"응답 길이 변경: {baseline_length} -> {len(resp.text)}"
                )

            # 시간 기반: 응답이 상당히 더 오래 걸림
            time_diff = resp.elapsed.total_seconds() - baseline_time
            if time_diff > 4.0:
                indicators.append(
                    f"응답 지연: {resp.elapsed.total_seconds():.1f}s "
                    f"(베이스라인: {baseline_time:.1f}s)"
                )

            if indicators:
                results.append({
                    'payload': payload,
                    'status_code': resp.status_code,
                    'indicators': indicators,
                    'response_length': len(resp.text),
                    'response_time': resp.elapsed.total_seconds()
                })

        except requests.exceptions.Timeout:
            results.append({
                'payload': payload,
                'status_code': None,
                'indicators': ['요청 타임아웃 (시간 기반 SQLi 가능성)'],
                'response_length': 0,
                'response_time': 15.0
            })
        except requests.exceptions.RequestException as e:
            pass  # 연결 에러, 건너뛰기

    return results


# ─── XSS 테스팅 ───

XSS_PAYLOADS = [
    '<script>alert(1)</script>',
    '<img src=x onerror=alert(1)>',
    '<svg onload=alert(1)>',
    '"><script>alert(1)</script>',
    "'-alert(1)-'",
    '<body onload=alert(1)>',
    '{{7*7}}',  # 템플릿 인젝션 테스트
    '${7*7}',   # 템플릿 인젝션 테스트
    'javascript:alert(1)',
    '<iframe src="javascript:alert(1)">',
]


def test_reflected_xss(url: str, param_name: str) -> list[dict]:
    """
    페이로드가 이스케이프되지 않고 응답에 나타나는지 검사하여
    반사형 XSS 테스트.
    """
    results = []

    for payload in XSS_PAYLOADS:
        try:
            resp = requests.get(
                url, params={param_name: payload}, timeout=10
            )

            # 페이로드가 인코딩 없이 반사되는지 검사
            if payload in resp.text:
                results.append({
                    'payload': payload,
                    'reflected': True,
                    'encoded': False,
                    'status_code': resp.status_code,
                })
            # HTML 인코딩된 버전 검사 (부분 보호)
            encoded = (payload.replace('<', '&lt;')
                              .replace('>', '&gt;')
                              .replace('"', '&quot;'))
            if encoded in resp.text and payload not in resp.text:
                results.append({
                    'payload': payload,
                    'reflected': True,
                    'encoded': True,
                    'status_code': resp.status_code,
                })

        except requests.exceptions.RequestException:
            pass

    return results


# ─── 보안 헤더 검사기 ───

SECURITY_HEADERS = {
    'Strict-Transport-Security': {
        'description': 'HSTS - HTTPS 강제',
        'recommended': 'max-age=31536000; includeSubDomains',
        'severity': 'HIGH',
    },
    'Content-Security-Policy': {
        'description': 'CSP - XSS 및 인젝션 방지',
        'recommended': "default-src 'self'",
        'severity': 'HIGH',
    },
    'X-Content-Type-Options': {
        'description': 'MIME 스니핑 방지',
        'recommended': 'nosniff',
        'severity': 'MEDIUM',
    },
    'X-Frame-Options': {
        'description': '클릭재킹 방지',
        'recommended': 'DENY',
        'severity': 'MEDIUM',
    },
    'X-XSS-Protection': {
        'description': '레거시 XSS 필터',
        'recommended': '0',  # 현대적 가이드: 비활성화, CSP 사용
        'severity': 'LOW',
    },
    'Referrer-Policy': {
        'description': 'Referrer 정보 제어',
        'recommended': 'strict-origin-when-cross-origin',
        'severity': 'LOW',
    },
    'Permissions-Policy': {
        'description': '브라우저 기능 제어',
        'recommended': 'camera=(), microphone=(), geolocation=()',
        'severity': 'MEDIUM',
    },
}


def check_security_headers(url: str) -> dict:
    """URL의 보안 헤더 검사."""
    resp = requests.get(url, timeout=10, allow_redirects=True)
    results = {
        'url': url,
        'status_code': resp.status_code,
        'headers_present': {},
        'headers_missing': {},
    }

    for header, info in SECURITY_HEADERS.items():
        value = resp.headers.get(header)
        if value:
            results['headers_present'][header] = {
                'value': value,
                'description': info['description'],
            }
        else:
            results['headers_missing'][header] = {
                'recommended': info['recommended'],
                'description': info['description'],
                'severity': info['severity'],
            }

    return results
```

---

## 7. 보안 코드 리뷰 체크리스트

### 7.1 코드 리뷰 프로세스

```
┌──────────────────────────────────────────────────────────────────┐
│              보안 코드 리뷰 프로세스                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  단계 1: 변경사항 이해                                            │
│  ├── 이 코드는 무엇을 하는가?                                     │
│  ├── 어떤 데이터를 처리하는가?                                    │
│  └── 신뢰 경계는 무엇인가?                                        │
│                                                                   │
│  단계 2: 입력/출력 검사                                           │
│  ├── 모든 외부 입력이 검증되었는가?                               │
│  ├── 출력이 적절히 인코딩되었는가?                                │
│  └── 파일 작업이 안전한 경로를 사용하는가?                        │
│                                                                   │
│  단계 3: 인증 및 권한 부여                                        │
│  ├── 모든 보호된 엔드포인트에서 인증 검사?                        │
│  ├── 적절한 세션 관리?                                            │
│  └── 최소 권한 적용?                                              │
│                                                                   │
│  단계 4: 데이터 보호                                              │
│  ├── 민감한 데이터를 저장 시 암호화?                              │
│  ├── 민감한 데이터를 전송 시 암호화?                              │
│  ├── 소스 코드에 비밀 없음?                                       │
│  └── PII를 올바르게 처리?                                         │
│                                                                   │
│  단계 5: 에러 처리                                                │
│  ├── 에러가 정보를 누출하지 않는가?                               │
│  ├── 적절한 예외 처리?                                            │
│  └── 안전하게 실패 (기본적으로 거부)?                             │
│                                                                   │
│  단계 6: 의존성                                                   │
│  ├── 새로운 의존성을 검토했는가?                                  │
│  ├── 버전이 고정되었는가?                                         │
│  └── 알려진 취약점을 확인했는가?                                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 Python 특화 보안 리뷰 체크리스트

```python
"""
예제를 포함한 Python 보안 코드 리뷰 체크리스트.
각 섹션은 취약한 버전과 안전한 버전을 보여줍니다.
"""

# ─── 1. 입력 검증 ───

# 취약: 검증 없음
def get_user_bad(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)

# 안전: 파라미터화된 쿼리 + 타입 검증
def get_user_good(user_id: int):
    if not isinstance(user_id, int) or user_id < 0:
        raise ValueError("유효하지 않은 사용자 ID")
    return db.execute("SELECT * FROM users WHERE id = ?", (user_id,))


# ─── 2. 인증 ───

# 취약: 비밀번호 비교 시 타이밍 공격
def check_password_bad(stored: str, provided: str) -> bool:
    return stored == provided  # 문자열 비교는 단락 평가됨

# 안전: 상수 시간 비교
import hmac
def check_password_good(stored: str, provided: str) -> bool:
    return hmac.compare_digest(stored.encode(), provided.encode())


# ─── 3. 직렬화 ───

# 취약: 신뢰할 수 없는 데이터로 pickle
import pickle
def load_data_bad(data: bytes):
    return pickle.loads(data)  # 임의 코드 실행!

# 안전: JSON 또는 검증된 스키마 사용
import json
def load_data_good(data: str):
    parsed = json.loads(data)
    # 스키마 검증
    if not isinstance(parsed, dict):
        raise ValueError("JSON 객체 예상")
    return parsed


# ─── 4. 파일 작업 ───

# 취약: 경로 순회
import os
def read_file_bad(filename: str):
    with open(f"/uploads/{filename}") as f:
        return f.read()

# 안전: 경로 해석 및 검증
from pathlib import Path
UPLOAD_DIR = Path("/uploads").resolve()

def read_file_good(filename: str):
    file_path = (UPLOAD_DIR / filename).resolve()
    if not file_path.is_relative_to(UPLOAD_DIR):
        raise ValueError("경로 순회 탐지")
    if not file_path.is_file():
        raise FileNotFoundError("파일을 찾을 수 없음")
    return file_path.read_text()


# ─── 5. 암호화 ───

# 취약: 약한 해싱
import hashlib
def hash_password_bad(password: str) -> str:
    return hashlib.md5(password.encode()).hexdigest()

# 안전: 적절한 비밀번호 해싱
from argon2 import PasswordHasher
ph = PasswordHasher()

def hash_password_good(password: str) -> str:
    return ph.hash(password)

def verify_password_good(hash: str, password: str) -> bool:
    try:
        return ph.verify(hash, password)
    except Exception:
        return False


# ─── 6. Subprocess ───

# 취약: 셸 인젝션
import subprocess
def run_command_bad(filename: str):
    subprocess.run(f"cat {filename}", shell=True)

# 안전: 셸 없음, 리스트 사용
def run_command_good(filename: str):
    # 먼저 파일명 검증
    if not Path(filename).name == filename:  # 경로 구분자 없음
        raise ValueError("유효하지 않은 파일명")
    subprocess.run(["cat", filename], shell=False, check=True)


# ─── 7. 로깅 ───

# 취약: 민감한 데이터 로깅
import logging
logger = logging.getLogger(__name__)

def login_bad(username: str, password: str):
    logger.info(f"로그인 시도: {username} / {password}")  # 비밀번호 로깅!

# 안전: 비밀 절대 로깅 금지
def login_good(username: str, password: str):
    logger.info(f"로그인 시도: user={username}")
    # 민감한 필드는 플레이스홀더 사용
    logger.debug("로그인 시도: user=%s password=<REDACTED>", username)


# ─── 8. 정규 표현식 ───

# 취약: ReDoS (정규 표현식 서비스 거부)
import re
def validate_email_bad(email: str) -> bool:
    # 이 패턴은 치명적인 백트래킹에 취약
    pattern = r'^([a-zA-Z0-9]+)*@[a-zA-Z0-9]+\.[a-zA-Z]+$'
    return bool(re.match(pattern, email, re.TIMEOUT))

# 안전: 잘 테스트된 라이브러리 또는 간단한 패턴 사용
def validate_email_good(email: str) -> bool:
    # 간단한, 백트래킹 없는 패턴
    if len(email) > 254:
        return False
    pattern = r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
    return bool(re.match(pattern, email))
```

---

## 8. CI/CD 보안 파이프라인 통합

### 8.1 GitHub Actions 보안 파이프라인

```yaml
# .github/workflows/security.yml

name: Security Scanning Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    # 월요일 오전 9시 UTC에 주간 실행
    - cron: '0 9 * * 1'

permissions:
  contents: read
  security-events: write  # SARIF 업로드용

jobs:
  # ─── 단계 1: SAST ───
  sast-bandit:
    name: "SAST: Bandit"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Bandit
        run: pip install bandit[toml]

      - name: Run Bandit
        run: |
          bandit -r src/ \
            -f sarif \
            -o bandit-results.sarif \
            --severity-level medium \
            --confidence-level medium \
            --exclude tests/
        continue-on-error: true

      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: bandit-results.sarif

  sast-semgrep:
    name: "SAST: Semgrep"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Semgrep
        uses: semgrep/semgrep-action@v1
        with:
          config: >-
            p/python
            p/flask
            p/owasp-top-ten
            .semgrep/custom_rules.yaml
        env:
          SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}

  # ─── 단계 2: SCA ───
  sca-dependencies:
    name: "SCA: Dependency Check"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pip-audit safety

      - name: Run pip-audit
        run: |
          pip-audit \
            -r requirements.txt \
            -f json \
            -o pip-audit-results.json \
            --desc
        continue-on-error: true

      - name: Run Safety
        run: |
          safety check \
            -r requirements.txt \
            --full-report \
            --output json > safety-results.json
        continue-on-error: true

      - name: Check for critical vulnerabilities
        run: |
          python -c "
          import json, sys
          with open('pip-audit-results.json') as f:
              data = json.load(f)
          vulns = []
          for dep in data.get('dependencies', []):
              vulns.extend(dep.get('vulns', []))
          if vulns:
              print(f'Found {len(vulns)} vulnerabilities!')
              for v in vulns:
                  print(f'  - {v[\"id\"]}: {v.get(\"description\", \"\")[:80]}')
              sys.exit(1)
          print('No vulnerabilities found.')
          "

  # ─── 단계 3: 비밀 스캐닝 ───
  secret-scanning:
    name: "Secret Scanning"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 비밀 스캐닝을 위한 전체 히스토리

      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Run TruffleHog
        uses: trufflesecurity/trufflehog@main
        with:
          extra_args: --only-verified

  # ─── 단계 4: 컨테이너 스캐닝 ───
  container-scan:
    name: "Container Scan"
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Run Trivy
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif

  # ─── 단계 5: DAST (스테이징에서) ───
  dast-zap:
    name: "DAST: ZAP Baseline"
    runs-on: ubuntu-latest
    needs: [sast-bandit, sast-semgrep, sca-dependencies]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Start application
        run: |
          docker-compose up -d
          sleep 10  # 앱 시작 대기

      - name: Run ZAP Baseline Scan
        uses: zaproxy/action-baseline@v0.12.0
        with:
          target: 'http://localhost:8080'
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'

      - name: Stop application
        if: always()
        run: docker-compose down

  # ─── 보안 게이트 ───
  security-gate:
    name: "Security Gate"
    runs-on: ubuntu-latest
    needs: [sast-bandit, sast-semgrep, sca-dependencies, secret-scanning]
    steps:
      - name: Check results
        run: |
          echo "모든 보안 검사 통과!"
          echo "상세한 발견사항은 Security 탭을 확인하세요."
```

### 8.2 보안을 위한 Pre-commit 훅

```yaml
# .pre-commit-config.yaml

repos:
  # Bandit - Python 보안 린터
  - repo: https://github.com/PyCQA/bandit
    rev: '1.7.7'
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml', '-ll']
        additional_dependencies: ['bandit[toml]']

  # 비밀 탐지
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  # 개인 키 검사
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: detect-private-key
      - id: check-added-large-files
        args: ['--maxkb=500']

  # Gitleaks
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.1
    hooks:
      - id: gitleaks
```

### 8.3 GitLab CI 보안 파이프라인

```yaml
# .gitlab-ci.yml

stages:
  - test
  - security
  - deploy

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

# ─── SAST 단계 ───
bandit-sast:
  stage: security
  image: python:3.12-slim
  script:
    - pip install bandit
    - bandit -r src/ -f json -o gl-sast-report.json --severity-level medium || true
  artifacts:
    reports:
      sast: gl-sast-report.json

semgrep-sast:
  stage: security
  image: semgrep/semgrep:latest
  script:
    - semgrep --config auto --sarif -o semgrep-results.sarif .
  artifacts:
    reports:
      sast: semgrep-results.sarif

# ─── 의존성 스캐닝 ───
dependency-check:
  stage: security
  image: python:3.12-slim
  script:
    - pip install pip-audit
    - pip-audit -r requirements.txt --strict
  allow_failure: true

# ─── 비밀 탐지 ───
secret-detection:
  stage: security
  image:
    name: zricethezav/gitleaks:latest
    entrypoint: [""]
  script:
    - gitleaks detect --source . --report-path gitleaks-report.json
  artifacts:
    reports:
      secret_detection: gitleaks-report.json
```

---

## 9. 종합 보안 스캐닝 스크립트

```python
"""
security_scanner.py - Unified security scanning orchestrator.
Runs multiple security tools and generates a combined report.

Usage:
    python security_scanner.py --project-dir ./myproject
    python security_scanner.py --project-dir ./myproject --output report.json
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Finding:
    """A single security finding from any tool."""
    tool: str
    severity: str           # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str           # SAST, SCA, SECRET, CONFIG
    title: str
    description: str
    file: Optional[str] = None
    line: Optional[int] = None
    cwe: Optional[str] = None
    fix: Optional[str] = None


@dataclass
class ScanReport:
    """Combined report from all scanners."""
    project: str
    scan_date: str = ""
    scan_duration_seconds: float = 0.0
    findings: list[Finding] = field(default_factory=list)
    tools_run: list[str] = field(default_factory=list)
    tools_failed: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.scan_date:
            self.scan_date = datetime.now().isoformat()


class SecurityScanner:
    """Orchestrates multiple security scanning tools."""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir).resolve()
        self.report = ScanReport(project=str(self.project_dir))
        self.start_time = time.time()

    def run_all(self) -> ScanReport:
        """Run all available security scanners."""
        print(f"[*] Starting security scan of: {self.project_dir}")
        print(f"[*] Scan started at: {self.report.scan_date}")
        print()

        self._run_bandit()
        self._run_pip_audit()
        self._run_secret_check()
        self._check_security_configs()

        self.report.scan_duration_seconds = time.time() - self.start_time
        self._generate_summary()

        return self.report

    def _run_bandit(self) -> None:
        """Run Bandit SAST scanner."""
        print("[*] Running Bandit (SAST)...")
        try:
            result = subprocess.run(
                [
                    "bandit", "-r", str(self.project_dir),
                    "-f", "json",
                    "--severity-level", "low",
                    "-x", "tests,venv,.tox"
                ],
                capture_output=True, text=True, timeout=120
            )

            data = json.loads(result.stdout)
            for issue in data.get("results", []):
                self.report.findings.append(Finding(
                    tool="bandit",
                    severity=issue["issue_severity"].upper(),
                    category="SAST",
                    title=f"[{issue['test_id']}] {issue['test_name']}",
                    description=issue["issue_text"],
                    file=issue["filename"],
                    line=issue["line_number"],
                    cwe=issue.get("issue_cwe", {}).get("id"),
                ))

            self.report.tools_run.append("bandit")
            print(f"    Found {len(data.get('results', []))} issues")

        except FileNotFoundError:
            print("    [!] Bandit not installed")
            self.report.tools_failed.append("bandit")
        except Exception as e:
            print(f"    [!] Bandit failed: {e}")
            self.report.tools_failed.append("bandit")

    def _run_pip_audit(self) -> None:
        """Run pip-audit SCA scanner."""
        print("[*] Running pip-audit (SCA)...")

        req_file = self.project_dir / "requirements.txt"
        if not req_file.exists():
            print("    [!] No requirements.txt found, skipping")
            return

        try:
            result = subprocess.run(
                ["pip-audit", "-r", str(req_file), "-f", "json", "--desc"],
                capture_output=True, text=True, timeout=120
            )

            data = json.loads(result.stdout)
            vuln_count = 0
            for dep in data.get("dependencies", []):
                for vuln in dep.get("vulns", []):
                    vuln_count += 1
                    fix_versions = vuln.get("fix_versions", [])
                    self.report.findings.append(Finding(
                        tool="pip-audit",
                        severity="HIGH",
                        category="SCA",
                        title=f"{dep['name']} {dep['version']}: {vuln['id']}",
                        description=vuln.get("description", ""),
                        fix=f"Upgrade to {fix_versions[0]}"
                            if fix_versions else "No fix available",
                    ))

            self.report.tools_run.append("pip-audit")
            print(f"    Found {vuln_count} vulnerable dependencies")

        except FileNotFoundError:
            print("    [!] pip-audit not installed")
            self.report.tools_failed.append("pip-audit")
        except Exception as e:
            print(f"    [!] pip-audit failed: {e}")
            self.report.tools_failed.append("pip-audit")

    def _run_secret_check(self) -> None:
        """Check for hardcoded secrets in source files."""
        print("[*] Checking for hardcoded secrets...")
        import re

        secret_patterns = [
            (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\'][a-zA-Z0-9]{20,}["\']',
             "Possible API key"),
            (r'(?i)(password|passwd|pwd)\s*[:=]\s*["\'][^"\']{6,}["\']',
             "Possible hardcoded password"),
            (r'(?i)(secret|token)\s*[:=]\s*["\'][a-zA-Z0-9+/=]{20,}["\']',
             "Possible hardcoded secret/token"),
            (r'-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY-----',
             "Private key detected"),
            (r'(?i)aws[_-]?(?:access[_-]?key[_-]?id|secret[_-]?access[_-]?key)\s*[:=]\s*["\']?[A-Z0-9]{16,}',
             "Possible AWS credential"),
        ]

        findings_count = 0
        for py_file in self.project_dir.rglob("*.py"):
            # Skip virtual environments and test fixtures
            rel_path = py_file.relative_to(self.project_dir)
            if any(part in str(rel_path) for part in
                   ['venv', '.tox', 'node_modules', '__pycache__']):
                continue

            try:
                content = py_file.read_text(errors='ignore')
                for line_num, line in enumerate(content.splitlines(), 1):
                    # Skip comments with nosec
                    if '# nosec' in line or '# noqa' in line:
                        continue
                    for pattern, description in secret_patterns:
                        if re.search(pattern, line):
                            findings_count += 1
                            self.report.findings.append(Finding(
                                tool="secret-scanner",
                                severity="HIGH",
                                category="SECRET",
                                title=description,
                                description=f"Potential secret found in source code",
                                file=str(rel_path),
                                line=line_num,
                                cwe="CWE-798",
                                fix="Move secrets to environment variables or "
                                    "a secret management service",
                            ))
            except Exception:
                pass

        self.report.tools_run.append("secret-scanner")
        print(f"    Found {findings_count} potential secrets")

    def _check_security_configs(self) -> None:
        """Check for security-related configuration issues."""
        print("[*] Checking security configurations...")

        findings_count = 0

        # Check for DEBUG mode in Flask/Django settings
        for py_file in self.project_dir.rglob("*.py"):
            rel_path = py_file.relative_to(self.project_dir)
            if any(part in str(rel_path) for part in ['venv', '.tox']):
                continue

            try:
                content = py_file.read_text(errors='ignore')

                # Flask DEBUG
                if 'app.run(debug=True)' in content:
                    findings_count += 1
                    self.report.findings.append(Finding(
                        tool="config-checker",
                        severity="MEDIUM",
                        category="CONFIG",
                        title="Flask debug mode enabled",
                        description="Debug mode should not be enabled in production",
                        file=str(rel_path),
                        fix="Use environment variable: "
                            "app.run(debug=os.getenv('FLASK_DEBUG', False))",
                    ))

                # Django DEBUG
                if 'DEBUG = True' in content and 'settings' in str(rel_path):
                    findings_count += 1
                    self.report.findings.append(Finding(
                        tool="config-checker",
                        severity="MEDIUM",
                        category="CONFIG",
                        title="Django DEBUG mode enabled in settings",
                        description="DEBUG should be False in production",
                        file=str(rel_path),
                        fix="Use: DEBUG = os.getenv('DJANGO_DEBUG', 'False') == 'True'",
                    ))

            except Exception:
                pass

        self.report.tools_run.append("config-checker")
        print(f"    Found {findings_count} configuration issues")

    def _generate_summary(self) -> None:
        """Generate summary statistics."""
        severity_counts = {
            'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'INFO': 0
        }
        category_counts = {
            'SAST': 0, 'SCA': 0, 'SECRET': 0, 'CONFIG': 0
        }

        for f in self.report.findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
            category_counts[f.category] = category_counts.get(f.category, 0) + 1

        self.report.summary = {
            'total_findings': len(self.report.findings),
            'by_severity': severity_counts,
            'by_category': category_counts,
            'tools_run': len(self.report.tools_run),
            'tools_failed': len(self.report.tools_failed),
        }


def print_report(report: ScanReport) -> None:
    """Print a formatted text report."""
    print("\n" + "=" * 65)
    print("  SECURITY SCAN REPORT")
    print("=" * 65)
    print(f"  Project:  {report.project}")
    print(f"  Date:     {report.scan_date}")
    print(f"  Duration: {report.scan_duration_seconds:.1f}s")
    print(f"  Tools:    {', '.join(report.tools_run)}")
    if report.tools_failed:
        print(f"  Failed:   {', '.join(report.tools_failed)}")

    s = report.summary
    print(f"\n  Total findings: {s['total_findings']}")
    print(f"  By severity: "
          f"CRITICAL={s['by_severity']['CRITICAL']} "
          f"HIGH={s['by_severity']['HIGH']} "
          f"MEDIUM={s['by_severity']['MEDIUM']} "
          f"LOW={s['by_severity']['LOW']}")
    print(f"  By category: "
          f"SAST={s['by_category']['SAST']} "
          f"SCA={s['by_category']['SCA']} "
          f"SECRET={s['by_category']['SECRET']} "
          f"CONFIG={s['by_category']['CONFIG']}")

    # Print findings grouped by severity
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
        findings = [f for f in report.findings if f.severity == severity]
        if not findings:
            continue

        print(f"\n{'─' * 65}")
        print(f"  {severity} ({len(findings)} findings)")
        print(f"{'─' * 65}")

        for f in findings:
            print(f"\n  [{f.tool}] {f.title}")
            if f.file:
                loc = f"    File: {f.file}"
                if f.line:
                    loc += f":{f.line}"
                print(loc)
            print(f"    {f.description[:100]}")
            if f.fix:
                print(f"    Fix: {f.fix[:100]}")

    print(f"\n{'=' * 65}")
    if s['by_severity']['CRITICAL'] > 0 or s['by_severity']['HIGH'] > 0:
        print("  RESULT: FAIL - Critical/High issues found")
    elif s['total_findings'] > 0:
        print("  RESULT: WARN - Issues found, review recommended")
    else:
        print("  RESULT: PASS - No issues found")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Unified Security Scanner")
    parser.add_argument("--project-dir", required=True,
                        help="Path to project directory")
    parser.add_argument("--output", help="Output JSON report file")
    parser.add_argument("--fail-on", default="high",
                        choices=["critical", "high", "medium", "low"],
                        help="Severity level that causes non-zero exit")
    args = parser.parse_args()

    scanner = SecurityScanner(args.project_dir)
    report = scanner.run_all()

    # Print text report
    print_report(report)

    # Save JSON report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\nJSON report saved to: {args.output}")

    # Exit code based on findings
    severity_order = ['low', 'medium', 'high', 'critical']
    threshold_idx = severity_order.index(args.fail_on)
    fail_severities = [s.upper() for s in severity_order[threshold_idx:]]

    for finding in report.findings:
        if finding.severity in fail_severities:
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
```

---

## 10. 연습 문제

### 연습 문제 1: Bandit 스캔 분석

다음의 의도적으로 취약한 코드에 대해 Bandit을 실행하고 모든 발견 사항을 수정하세요:

```python
"""vulnerable_app.py - Fix all security issues found by Bandit."""
import os
import pickle
import hashlib
import subprocess
import sqlite3
from flask import Flask, request

app = Flask(__name__)
SECRET_KEY = "my-super-secret-key-12345"

@app.route('/search')
def search():
    query = request.args.get('q')
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM products WHERE name LIKE '%{query}%'")
    return str(cursor.fetchall())

@app.route('/run')
def run_command():
    cmd = request.args.get('cmd')
    result = subprocess.check_output(cmd, shell=True)
    return result

@app.route('/load')
def load_data():
    data = request.get_data()
    obj = pickle.loads(data)
    return str(obj)

@app.route('/hash')
def hash_password():
    password = request.args.get('pw')
    return hashlib.md5(password.encode()).hexdigest()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
```

**과제:**
1. `bandit vulnerable_app.py`를 실행하고 모든 발견 사항을 문서화하세요
2. 각 취약점을 수정하세요
3. Bandit을 다시 실행하여 모든 문제가 해결되었는지 확인하세요

### 연습 문제 2: 커스텀 Semgrep 규칙 작성

다음을 탐지하는 Semgrep 규칙을 작성하세요:
1. 문자열 연결이 포함된 `os.system()` 사용
2. POST를 수락하지만 `Content-Type`을 검증하지 않는 Flask 라우트
3. HTTP 요청을 처리하는 함수 내에서 `eval()` 또는 `exec()` 사용
4. 하드코딩된 데이터베이스 연결 문자열

### 연습 문제 3: 의존성 감사

의도적으로 오래되고 취약한 패키지가 포함된 `requirements.txt`를 생성하세요:
```
flask==2.0.1
requests==2.25.1
django==3.2.0
pyyaml==5.3.1
pillow==8.0.0
```

1. `pip-audit -r requirements.txt`를 실행하고 발견된 모든 CVE를 문서화하세요
2. 각 패키지의 최소 안전 버전을 확인하세요
3. 수정된 버전이 포함된 `requirements-secure.txt`를 생성하세요

### 연습 문제 4: 속성 기반 테스팅

다음에 대한 Hypothesis 테스트를 작성하세요:
1. 비밀번호 강도 검증기 (대문자, 소문자, 숫자, 특수문자 필수, 최소 8자)
2. `javascript:` 및 `data:` 스킴을 방지해야 하는 URL 살균기
3. 모든 HTML을 제거하되 텍스트 내용은 보존해야 하는 HTML 태그 제거기

### 연습 문제 5: CI/CD 보안 파이프라인

다음을 수행하는 GitHub Actions 워크플로우를 설계하고 구현하세요:
1. SARIF 출력으로 Bandit 실행
2. requirements.txt에 대해 pip-audit 실행
3. gitleaks를 사용한 시크릿 검사
4. HIGH 또는 CRITICAL 발견 사항이 있으면 파이프라인 실패
5. 발견 사항 요약을 PR에 코멘트로 게시

### 연습 문제 6: 보안 코드 리뷰

다음 코드를 리뷰하고 모든 보안 문제를 식별하세요:

```python
from flask import Flask, request, jsonify, redirect
import jwt
import sqlite3
import os

app = Flask(__name__)

@app.route('/api/users/<user_id>')
def get_user(user_id):
    db = sqlite3.connect('users.db')
    cursor = db.execute(
        f"SELECT id, name, email, ssn FROM users WHERE id = {user_id}"
    )
    user = cursor.fetchone()
    if user:
        return jsonify({
            'id': user[0], 'name': user[1],
            'email': user[2], 'ssn': user[3]
        })
    return jsonify({'error': f'User {user_id} not found'}), 404

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    db = sqlite3.connect('users.db')
    cursor = db.execute(
        f"SELECT * FROM users WHERE email = '{data['email']}' "
        f"AND password = '{data['password']}'"
    )
    user = cursor.fetchone()
    if user:
        token = jwt.encode(
            {'user_id': user[0], 'role': user[4]},
            'secret123',
            algorithm='HS256'
        )
        return jsonify({'token': token})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/redirect')
def handle_redirect():
    url = request.args.get('url')
    return redirect(url)

@app.route('/api/upload', methods=['POST'])
def upload():
    f = request.files['file']
    f.save(os.path.join('/uploads', f.filename))
    return jsonify({'status': 'uploaded'})
```

최소 10가지의 서로 다른 보안 취약점을 문서화하고 각각에 대한 수정 방법을 제시하세요.

---

## 요약

```
┌─────────────────────────────────────────────────────────────────┐
│              보안 테스팅 핵심 요약                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 방어 계층화: SAST + SCA + DAST + 퍼징 조합 사용             │
│  2. 시프트 레프트: SDLC에서 가능한 빨리 버그 발견               │
│  3. 자동화: 모든 도구를 CI/CD 파이프라인에 통합                  │
│  4. 커스텀 규칙: 프로젝트별 Semgrep 규칙 작성                   │
│  5. 오탐 관리: 베이스라인과 #nosec로 관리                        │
│  6. 의존성: 정기적으로 스캔 및 업데이트 (Dependabot/SCA)        │
│  7. 코드 리뷰: 보안은 사람과 도구의 협업                         │
│  8. 퍼징: 다른 방법으로는 발견하지 못하는 버그 탐지              │
│  9. 침투 테스팅: 다른 모든 발견 사항을 검증                      │
│ 10. 지속적: 보안 테스팅은 일회성이 아닌 지속적 활동              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**이전**: [12. 컨테이너 및 클라우드 보안](./12_Container_Security.md) | **다음**: [14. 사고 대응과 포렌식](14_Incident_Response.md)
