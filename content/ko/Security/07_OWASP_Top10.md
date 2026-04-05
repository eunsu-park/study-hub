# 07. OWASP Top 10 (2021)

**이전**: [06. 인가와 접근 제어](06_Authorization.md) | **다음**: [08. 인젝션 공격과 방어](08_Injection_Attacks.md)

---

OWASP (Open Worldwide Application Security Project) Top 10은 웹 애플리케이션 보안 위험에 대해 가장 널리 인정받는 문서입니다. 실제 취약점 데이터를 기반으로 주기적으로 업데이트되며, 개발자와 보안 전문가를 위한 표준 인식 문서 역할을 합니다. 2021년 판은 위협 환경의 중대한 변화를 반영하여 세 가지 새로운 카테고리와 주요 재편성이 있습니다. 이 레슨에서는 설명, 실제 사례, 취약한 코드, 수정된 코드, 방어 전략과 함께 열 가지 카테고리 각각을 다룹니다.

## 학습 목표

- OWASP Top 10 (2021) 모든 카테고리 이해하기
- 각 카테고리의 취약한 코드 패턴 식별하기
- 각 취약점 클래스를 방지하는 안전한 코드 작성하기
- 개발 및 코드 리뷰 시 보안 체크리스트로 OWASP Top 10 적용하기
- 각 취약점의 실제 영향 인식하기

---

## 1. 개요: 2021 Top 10

```
┌─────────────────────────────────────────────────────────────────┐
│                  OWASP Top 10 - 2021                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  #   카테고리                              2017년 대비 변화     │
│  ─── ──────────────────────────────────── ────────────────────  │
│  A01  취약한 접근 제어                      ↑ #5에서 상승        │
│  A02  암호화 실패                           ↑ #3에서 상승(이름변경)│
│  A03  인젝션                                ↓ #1에서 하락        │
│  A04  안전하지 않은 설계                    ★ 신규               │
│  A05  보안 구성 오류                        ↑ #6에서 상승        │
│  A06  취약하고 오래된 구성 요소             ↑ #9에서 상승(이름변경)│
│  A07  식별 및 인증 실패                     ↓ #2에서 하락(이름변경)│
│  A08  소프트웨어 및 데이터 무결성 실패      ★ 신규               │
│  A09  보안 로깅 및 모니터링 실패            ↑ #10에서 상승(이름변경)│
│  A10  서버 측 요청 위조 (SSRF)              ★ 신규               │
│                                                                  │
│  주요 트렌드:                                                    │
│  - 취약한 접근 제어가 1위 (테스트된 앱의 94%)                   │
│  - 인젝션이 #1에서 #3으로 하락 (프레임워크가 도움)             │
│  - 세 가지 새 카테고리가 현대 위협 반영                         │
│  - 공급망 및 무결성 우려 증가                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. A01: 취약한 접근 제어

### 2.1 설명

접근 제어는 사용자가 의도된 권한을 벗어나 행동할 수 없도록 정책을 강제합니다. 취약한 접근 제어는 **가장 일반적인** 웹 애플리케이션 취약점으로, 테스트된 애플리케이션의 94%에서 발견됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              A01: 취약한 접근 제어                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  일반적인 약점:                                                  │
│  - URL/파라미터/API 수정으로 접근 제어 우회                     │
│  - 다른 사람의 계정 보기 또는 편집 (IDOR)                       │
│  - 접근 제어가 누락된 API 접근 (POST/PUT/DELETE)                │
│  - 권한 상승 (로그인 없이 관리자로 행동)                        │
│  - 메타데이터 조작 (JWT 재생, 쿠키 변조)                        │
│  - 무단 API 접근을 허용하는 CORS 구성 오류                      │
│  - 인증되지 않은/관리자 페이지로 강제 브라우징                  │
│                                                                  │
│  영향: 데이터 도난, 무단 데이터 수정,                            │
│        계정 탈취, 전체 시스템 침해                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 취약한 코드

```python
# 취약: 관리자 엔드포인트에 접근 제어 없음
@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
def delete_user(user_id):
    # 이 URL을 아는 사람은 누구나 사용자를 삭제할 수 있음!
    db.delete_user(user_id)
    return jsonify({"status": "deleted"})


# 취약: IDOR - 소유권 확인 없음
@app.route('/api/orders/<int:order_id>')
@require_auth
def get_order(order_id):
    order = db.get_order(order_id)
    return jsonify(order)  # 사용자 A가 사용자 B의 주문을 볼 수 있음
```

### 2.3 수정된 코드

```python
# 수정: 적절한 인가 확인
@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@require_role('admin')  # 관리자만 접근 가능
def delete_user(user_id):
    db.delete_user(user_id)
    return jsonify({"status": "deleted"})


# 수정: 소유권 검증
@app.route('/api/orders/<int:order_id>')
@require_auth
def get_order(order_id):
    order = db.get_order(order_id)
    if not order:
        return jsonify({"error": "Not found"}), 404

    # 주문이 인증된 사용자에게 속하는지 검증
    if order['user_id'] != g.current_user['id']:
        return jsonify({"error": "Not found"}), 404  # 403이 아닌 404

    return jsonify(order)
```

### 2.4 방어

- 공개 리소스를 제외하고 기본적으로 거부
- 접근 제어 메커니즘을 한 번 구현하고 모든 곳에서 재사용
- 레코드 소유권 강제 (사용자가 제공한 ID에만 의존하지 않기)
- 웹 서버 디렉토리 목록 비활성화
- 접근 제어 실패를 로그에 기록하고 관리자에게 알림
- 자동화된 스캔 피해를 최소화하기 위해 API 접근 속도 제한
- 로그아웃 시 JWT 토큰 무효화 (서버 측 토큰 차단 목록)

---

## 3. A02: 암호화 실패

### 3.1 설명

이전에는 "민감한 데이터 노출"이라고 불렸으며, 이 카테고리는 민감한 데이터 노출로 이어지는 암호화 관련 실패에 초점을 맞춥니다. 보호해야 할 데이터를 암호화하지 않는 것과 약하거나 오래된 암호화 알고리즘 사용을 모두 포함합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              A02: 암호화 실패                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  스스로에게 물어보세요:                                          │
│  1. 어떤 데이터가 평문으로 전송되거나 저장되는가?               │
│  2. 오래되거나 약한 암호화 알고리즘이나 프로토콜이 사용되는가? │
│  3. 기본 암호화 키가 사용되거나 키가 순환되지 않는가?          │
│  4. 암호화가 강제되지 않는가 (HTTPS 리디렉션 누락)?            │
│  5. 적절한 HTTP 보안 헤더가 누락되어 있는가?                   │
│  6. 서버 인증서가 올바르게 검증되는가?                          │
│  7. 비밀번호에 대해 deprecated 해싱이 사용되는가 (MD5, SHA1)?  │
│                                                                  │
│  민감한 데이터 카테고리:                                         │
│  - 비밀번호, 신용카드 번호, 건강 기록                           │
│  - 개인 데이터, 비즈니스 비밀                                   │
│  - 개인정보 보호 규정에 의해 보호되는 데이터 (GDPR, HIPAA, PCI)│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 취약한 코드

```python
# 취약: MD5로 비밀번호 저장
import hashlib

def store_password(password):
    # MD5는 비밀번호 해싱에 깨졌음!
    hash_value = hashlib.md5(password.encode()).hexdigest()
    db.store(hash_value)

# 취약: 하드코딩된 암호화 키
ENCRYPTION_KEY = "my-secret-key-123"  # 소스 제어에 커밋됨!

# 취약: ECB 모드 사용 (패턴 드러남)
from Crypto.Cipher import AES
cipher = AES.new(key, AES.MODE_ECB)  # ECB 모드는 안전하지 않음!
encrypted = cipher.encrypt(data)

# 취약: 민감한 데이터에 HTTP 사용
# HTTP에서 HTTPS로의 리디렉션 없음
# HSTS 헤더 없음
```

### 3.3 수정된 코드

```python
# 수정: 비밀번호에 Argon2 사용
from argon2 import PasswordHasher
ph = PasswordHasher()

def store_password(password):
    hash_value = ph.hash(password)  # 자동 솔팅을 사용한 Argon2id
    db.store(hash_value)

# 수정: 환경에서 키 가져오기, 소스 코드에서 가져오지 않기
import os
ENCRYPTION_KEY = os.environ['ENCRYPTION_KEY']  # 256비트 키

# 수정: GCM 모드 사용 (인증된 암호화)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

key = AESGCM.generate_key(bit_length=256)
aesgcm = AESGCM(key)
nonce = os.urandom(12)  # nonce를 절대 재사용하지 마세요
encrypted = aesgcm.encrypt(nonce, data, associated_data)

# 수정: HTTPS 강제 및 보안 헤더 추가
@app.after_request
def set_security_headers(response):
    response.headers['Strict-Transport-Security'] = \
        'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response
```

### 3.4 방어

- 민감도에 따라 데이터 분류; 분류별로 제어 적용
- 불필요한 민감한 데이터를 저장하지 마세요; 가능한 한 빨리 폐기
- 모든 민감한 데이터를 저장 시 및 전송 시 암호화 (TLS 1.2+)
- 강력하고 표준 알고리즘 사용 (AES-256-GCM, RSA-2048+, Ed25519)
- 인증된 암호화 사용 (GCM, ChaCha20-Poly1305), ECB는 절대 사용 안 함
- Argon2id, bcrypt 또는 scrypt를 사용하여 비밀번호 저장
- 암호학적으로 안전한 난수 생성기를 사용하여 키 생성
- 민감한 데이터가 포함된 페이지의 캐싱 비활성화

---

## 4. A03: 인젝션

### 4.1 설명

인젝션 결함은 신뢰할 수 없는 데이터가 명령 또는 쿼리의 일부로 인터프리터에 전송될 때 발생합니다. 공격자의 악의적인 데이터는 인터프리터를 속여 의도하지 않은 명령을 실행하거나 권한 없이 데이터에 접근하게 할 수 있습니다. SQL 인젝션, NoSQL 인젝션, OS 명령 인젝션, LDAP 인젝션은 모두 이 취약점의 형태입니다.

> **참고**: 인젝션은 레슨 08에서 훨씬 더 자세히 다룹니다. 이 섹션은 요약을 제공합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              A03: 인젝션                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  인젝션 유형:                                                    │
│  ├── SQL 인젝션           (가장 일반적)                          │
│  ├── NoSQL 인젝션         (MongoDB 등)                           │
│  ├── 명령 인젝션          (os.system, subprocess)                │
│  ├── LDAP 인젝션          (디렉토리 서비스)                      │
│  ├── XPath 인젝션         (XML 쿼리)                             │
│  ├── 템플릿 인젝션        (Jinja2, Twig SSTI)                   │
│  └── 표현 언어            (Spring EL, OGNL)                      │
│                                                                  │
│  근본 원인:                                                      │
│  사용자 입력이 매개변수화되거나 적절히 이스케이프되는 대신       │
│  쿼리/명령에 연결됨                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 취약 코드 vs 수정된 코드

```python
# 취약: SQL 인젝션
@app.route('/search')
def search():
    q = request.args.get('q')
    # 문자열 연결 = SQL 인젝션!
    results = db.execute(f"SELECT * FROM products WHERE name LIKE '%{q}%'")
    return jsonify(results)

# 공격: /search?q=' OR '1'='1' --


# 수정: 매개변수화된 쿼리
@app.route('/search')
def search():
    query = request.args.get('q', '')
    results = db.execute(
        "SELECT * FROM products WHERE name LIKE :query",
        {"query": f"%{query}%"}  # 파라미터 바인딩
    )
    return jsonify(results)
```

### 4.3 방어

- 매개변수화된 쿼리 / 준비된 문 사용 (항상)
- 매개변수화를 처리하는 ORM 프레임워크 사용 (SQLAlchemy, Django ORM)
- 모든 입력 검증 및 정제 (화이트리스트 검증)
- 특정 인터프리터에 대한 특수 문자 이스케이프
- 인젝션 시 대량 공개를 방지하기 위해 쿼리에 LIMIT 사용

---

## 5. A04: 안전하지 않은 설계

### 5.1 설명

이것은 2021년의 **새로운 카테고리**로 설계 및 아키텍처 결함과 관련된 위험에 초점을 맞춥니다. 위협 모델링, 보안 설계 패턴 및 참조 아키텍처의 더 많은 사용을 요구합니다. 안전하지 않은 설계는 완벽한 구현으로도 수정할 수 없습니다 - 결함이 있는 설계는 본질적으로 취약합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              A04: 안전하지 않은 설계                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  안전하지 않은 설계 ≠ 안전하지 않은 구현                        │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ 안전하지 않은    │    │ 안전하지 않은    │                   │
│  │ 설계             │    │ 구현             │                   │
│  │                  │    │                  │                    │
│  │ 청사진이 결함이  │    │ 청사진은 좋지만  │                    │
│  │ 있음. 코드 수정  │    │ 코드에 버그가    │                    │
│  │ 으로 도움 안됨.  │    │ 있음.            │                    │
│  │                  │    │                  │                    │
│  │ 예시:            │    │ 예시:            │                    │
│  │ 비밀번호 재설정  │    │ 로그인 폼의      │                    │
│  │ 이메일로 평문    │    │ SQL 인젝션       │                    │
│  │ 비밀번호 전송    │    │                  │                    │
│  │ (설계상)         │    │                  │                    │
│  └──────────────────┘    └──────────────────┘                   │
│                                                                  │
│  안전하지 않은 설계의 예:                                        │
│  - 인증에 속도 제한 없음 (무차별 대입 공격 가능)                │
│  - 비밀번호 복구의 유일한 방법으로 보안 질문 사용               │
│  - 입력 검증 아키텍처 없음 (각 개발자가 자신의 방식으로 구현)  │
│  - 설계상 소스 코드에 비밀 저장                                 │
│  - 다중 테넌트 시스템에서 테넌트 데이터 간 분리 없음            │
│  - 요구사항에서 남용 사례 누락 (행복한 경로만)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 실제 사례: 영화관 예약

```python
# 안전하지 않은 설계: 속도 제한 없는 영화 티켓 예약
# 설계상 봇이 인기 영화의 모든 티켓을 예약할 수 있음

@app.route('/api/book', methods=['POST'])
@require_auth
def book_ticket():
    movie_id = request.json['movie_id']
    seats = request.json['seats']  # 좌석 수 제한 없음!

    # 사용자당 속도 제한 없음
    # 거래당 최대 좌석 수 없음
    # 수요가 많은 이벤트에 CAPTCHA 없음
    # 사기 탐지 없음

    booking = create_booking(g.user.id, movie_id, seats)
    return jsonify(booking)


# 안전한 설계: 적절한 보호 장치 포함

@app.route('/api/book', methods=['POST'])
@require_auth
@rate_limit(max_requests=5, per_minutes=1)  # 속도 제한
def book_ticket():
    movie_id = request.json['movie_id']
    seats = request.json['seats']

    # 설계 수준 제어
    MAX_SEATS_PER_BOOKING = 6
    if len(seats) > MAX_SEATS_PER_BOOKING:
        return jsonify({"error": f"예약당 최대 {MAX_SEATS_PER_BOOKING}석"}), 400

    # 사용자가 이 상영회에 너무 많은 좌석을 이미 예약했는지 확인
    existing = get_user_bookings(g.user.id, movie_id)
    if len(existing) >= 2:  # 영화당 사용자당 최대 2개 예약
        return jsonify({"error": "이 영화에 대한 최대 예약 수에 도달했습니다"}), 400

    # 수요가 많은 이벤트의 경우 추가 검증 필요
    movie = get_movie(movie_id)
    if movie.get('high_demand'):
        if not verify_captcha(request.json.get('captcha_token')):
            return jsonify({"error": "CAPTCHA 검증 필요"}), 400

    booking = create_booking(g.user.id, movie_id, seats)
    return jsonify(booking)
```

### 5.3 방어

- 중요한 인증, 접근 제어 및 비즈니스 로직에 위협 모델링 사용
- 사용자 스토리에 보안 언어 및 제어 통합
- 모든 중요한 흐름이 남용에 저항하는지 검증하는 단위 및 통합 테스트 작성
- 실패를 위한 설계: 사용자/세션당 리소스 소비 제한
- 신뢰 경계를 분리하기 위해 애플리케이션 레이어 계층화
- 보안 설계 패턴 사용 (예: 무상태 세션 관리, 입력 검증 프레임워크)

---

## 6. A05: 보안 구성 오류

### 6.1 설명

보안 구성 오류는 실제로 가장 흔히 볼 수 있는 문제입니다. 잘못 구성된 권한, 불필요한 기능 활성화, 기본 계정/비밀번호 미변경, 지나치게 자세한 오류 메시지, 보안 강화 누락이 포함됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              A05: 보안 구성 오류                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  일반적인 구성 오류:                                             │
│  ┌──────────────────────────────────────────────────┐           │
│  │ 문제                     │ 위험                   │           │
│  ├──────────────────────────┼───────────────────────┤           │
│  │ 프로덕션에서 디버그 모드 │ 스택 추적 노출         │           │
│  │ 기본 admin:admin         │ 즉시 침해              │           │
│  │ 디렉토리 목록 활성화     │ 소스/구성 노출         │           │
│  │ 불필요한 서비스          │ 공격 표면 증가         │           │
│  │ 자세한 오류 메시지       │ 정보 공개              │           │
│  │ 보안 헤더 누락           │ XSS, 클릭재킹          │           │
│  │ CORS: Access-Control-    │ 교차 출처 공격         │           │
│  │   Allow-Origin: *        │                         │           │
│  │ 오래된 소프트웨어        │ 알려진 취약점          │           │
│  │ S3 버킷 공개             │ 데이터 침해            │           │
│  │ .git 폴더 노출           │ 소스 코드 유출         │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 취약한 구성

```python
# 취약: 프로덕션에서 Flask 디버그 모드
app = Flask(__name__)
app.config['DEBUG'] = True  # 대화형 디버거 노출!
app.config['SECRET_KEY'] = 'dev'  # 약한/기본 비밀 키

# 취약: 지나치게 허용적인 CORS
from flask_cors import CORS
CORS(app, origins="*")  # 모든 웹사이트가 요청할 수 있음!

# 취약: 자세한 오류 메시지
@app.errorhandler(500)
def handle_error(error):
    return jsonify({
        "error": str(error),
        "traceback": traceback.format_exc(),  # 내부 정보 유출!
        "database": app.config['DATABASE_URI'],  # 자격 증명 유출!
    }), 500
```

### 6.3 수정된 구성

```python
import os

app = Flask(__name__)

# 수정: 환경 기반 구성
app.config['DEBUG'] = False
app.config['TESTING'] = False
app.config['SECRET_KEY'] = os.environ['SECRET_KEY']  # 강력하고 무작위

# 수정: 제한적인 CORS
from flask_cors import CORS
CORS(app, origins=[
    "https://myapp.com",
    "https://www.myapp.com",
])

# 수정: 프로덕션에서 일반적인 오류 메시지
@app.errorhandler(500)
def handle_error(error):
    # 전체 오류를 내부적으로 로그
    app.logger.error(f"Internal error: {error}", exc_info=True)

    # 사용자에게 일반 메시지 반환
    return jsonify({
        "error": "내부 오류가 발생했습니다",
        "reference": generate_error_reference_id(),  # 지원 티켓용
    }), 500

# 수정: 보안 헤더
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '0'  # 레거시 XSS 필터 비활성화
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers['Permissions-Policy'] = (
        'camera=(), microphone=(), geolocation=()'
    )
    if request.is_secure:
        response.headers['Strict-Transport-Security'] = (
            'max-age=31536000; includeSubDomains; preload'
        )
    return response
```

### 6.4 방어

- 모든 환경에 대해 반복 가능한 강화 프로세스 구현
- 불필요한 기능, 프레임워크 및 구성 요소 제거 또는 설치하지 않기
- 패치 관리 프로세스의 일부로 구성 검토 및 업데이트
- 일관되고 감사 가능한 배포를 위해 인프라를 코드로 사용
- CI/CD에서 자동화된 보안 구성 검증 구현
- 다른 자격 증명으로 환경 분리 (개발, 스테이징, 프로덕션)

---

## 7. A06: 취약하고 오래된 구성 요소

### 7.1 설명

알려진 취약점이 있는 구성 요소 (라이브러리, 프레임워크, OS)를 사용하는 애플리케이션은 악용될 수 있습니다. 이것은 점점 더 중요해지는 공급망 위험입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          A06: 취약하고 오래된 구성 요소                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  다음과 같은 경우 취약합니다:                                    │
│  - 사용된 모든 구성 요소의 버전을 모름                          │
│  - 소프트웨어가 지원되지 않거나 패치되지 않음                   │
│  - 정기적으로 취약점을 스캔하지 않음                            │
│  - 적시에 수정하거나 업그레이드하지 않음                        │
│  - 개발자가 업데이트된 라이브러리의 호환성을 테스트하지 않음    │
│  - 구성 요소 구성이 보안되지 않음 (A05 참조)                    │
│                                                                  │
│  실제 영향:                                                      │
│  - Log4Shell (CVE-2021-44228): Log4j의 치명적 RCE              │
│  - Equifax 침해 (2017): 패치되지 않은 Apache Struts             │
│  - Event-Stream (2018): 악의적인 npm 패키지                     │
│  - ua-parser-js (2021): npm의 공급망 공격                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 종속성 감사

```bash
# Python: 알려진 취약점 확인
pip install pip-audit
pip-audit                        # 설치된 패키지 스캔
pip-audit -r requirements.txt    # requirements 파일 스캔

# Python: safety로 대체
pip install safety
safety check                     # 설치된 패키지 확인

# Node.js: 내장 감사
npm audit
npm audit fix                    # 가능한 경우 자동 수정

# 일반: OWASP Dependency-Check
# 여러 언어 스캔 (Java, .NET, Python, JS 등)
# https://owasp.org/www-project-dependency-check/

# GitHub: Dependabot (취약한 종속성에 대한 자동 PR)
# GitLab: CI/CD 파이프라인의 종속성 스캔

# 해시 검증으로 종속성 고정 (Python)
pip install --require-hashes -r requirements.txt
```

```python
# 고정된 버전과 해시가 있는 requirements.txt
# 생성: pip-compile --generate-hashes requirements.in
flask==3.0.0 \
    --hash=sha256:21128f47e...
werkzeug==3.0.1 \
    --hash=sha256:5a7b12abc...
```

### 7.3 방어

- 사용하지 않는 종속성, 기능, 구성 요소, 파일 및 문서 제거
- 도구를 사용하여 구성 요소 버전을 지속적으로 목록화 (pip-audit, npm audit, OWASP Dependency-Check)
- 취약점 알림을 위해 CVE 및 NVD와 같은 소스 모니터링
- 공식 소스에서만 보안 링크를 통해 구성 요소 얻기
- 유지 관리되지 않는 라이브러리 모니터링 (보안 패치 없음)
- 패치 계획 수립: 업데이트를 신속하게 테스트하고 배포

---

## 8. A07: 식별 및 인증 실패

### 8.1 설명

사용자 신원 확인, 인증 및 세션 관리는 인증 관련 공격으로부터 보호하는 데 중요합니다. 이것은 이전에 "취약한 인증"이라고 불렸습니다.

> **참고**: 포괄적인 인증 내용은 레슨 05를 참조하세요.

```
┌─────────────────────────────────────────────────────────────────┐
│          A07: 식별 및 인증 실패                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  약점:                                                           │
│  - 무차별 대입 또는 크레덴셜 스터핑 허용                        │
│  - 약하거나 잘 알려진 비밀번호 허용                             │
│  - 약한 크레덴셜 복구 사용 ("당신의 애완동물 이름은?")         │
│  - 평문 또는 약하게 해시된 비밀번호 사용                        │
│  - 다중 인증 누락 또는 비효율적                                 │
│  - URL에 세션 ID 노출                                           │
│  - 로그인 후 세션 ID를 순환하지 않음                            │
│  - 로그아웃 시 세션을 적절히 무효화하지 않음                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 취약 코드 vs 수정된 코드

```python
# 취약: 무차별 대입 보호 없음
@app.route('/login', methods=['POST'])
def login_vulnerable():
    username = request.json['username']
    password = request.json['password']

    user = db.find_user(username)
    if user and check_password(password, user.password_hash):
        session['user_id'] = user.id  # 세션 재생성 없음!
        return jsonify({"status": "success"})

    return jsonify({"error": "Invalid credentials"}), 401


# 수정: 속도 제한, 잠금 및 세션 관리 포함
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["200 per day"])

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")  # 로그인 시도 속도 제한
def login_secure():
    username = request.json['username']
    password = request.json['password']

    # 계정 잠금 확인
    if is_account_locked(username):
        return jsonify({"error": "계정이 일시적으로 잠겼습니다"}), 429

    user = db.find_user(username)
    if user and check_password(password, user.password_hash):
        # 실패 시도 재설정
        reset_failed_attempts(username)

        # 세션 재생성 (세션 고정 방지)
        session.clear()
        session['user_id'] = user.id
        session['created_at'] = time.time()
        session.permanent = True

        return jsonify({"status": "success"})

    # 실패 시도 증가
    record_failed_attempt(username)

    # 일반 오류 (사용자 이름 존재 여부 공개 안 함)
    return jsonify({"error": "Invalid credentials"}), 401
```

### 8.3 방어

- 다중 인증 구현 (TOTP 또는 FIDO2)
- 로그인 엔드포인트에 속도 제한 및 계정 잠금 사용
- 유출된 비밀번호 데이터베이스와 비밀번호 확인
- 안전한 비밀번호 저장 사용 (Argon2id, bcrypt)
- 로그인 후 세션 ID 재생성
- 적절한 세션 타임아웃 및 무효화 구현

---

## 9. A08: 소프트웨어 및 데이터 무결성 실패

### 9.1 설명

이 **새로운 카테고리**는 무결성을 검증하지 않고 소프트웨어 업데이트, 중요한 데이터 및 CI/CD 파이프라인에 대한 가정을 하는 데 초점을 맞춥니다. 이전 "안전하지 않은 역직렬화" 카테고리를 포함합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          A08: 소프트웨어 및 데이터 무결성 실패                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  공격 벡터:                                                      │
│                                                                  │
│  1. CI/CD 파이프라인 침해:                                       │
│     공격자가 빌드 파이프라인을 수정하여 악의적인 코드 주입       │
│     ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐              │
│     │ Code │───▶│Build │───▶│ Test │───▶│Deploy│              │
│     │      │    │      │    │      │    │      │              │
│     └──────┘    └──┬───┘    └──────┘    └──────┘              │
│                    │                                             │
│                    ▼ 공격자가 여기에 백도어 주입                 │
│                                                                  │
│  2. 검증 없는 자동 업데이트:                                     │
│     앱이 디지털 서명 검증 없이 업데이트 다운로드                │
│     공격자가 MITM을 수행하여 악의적인 업데이트 제공             │
│                                                                  │
│  3. 안전하지 않은 역직렬화:                                      │
│     앱이 신뢰할 수 없는 데이터를 역직렬화하여 RCE 발생          │
│     pickle.loads(user_input)  ← 원격 코드 실행!                 │
│                                                                  │
│  4. 종속성 혼동:                                                 │
│     공격자가 공개 레지스트리에 내부 패키지와 같은 이름으로       │
│     악의적인 패키지 게시                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 취약한 코드: 안전하지 않은 역직렬화

```python
import pickle
import yaml

# 취약: pickle로 신뢰할 수 없는 데이터 역직렬화
@app.route('/api/import', methods=['POST'])
def import_data_vulnerable():
    data = request.get_data()
    obj = pickle.loads(data)  # RCE! 공격자가 임의 코드 실행 가능
    return jsonify({"status": "imported"})

# 공격 페이로드:
# import pickle, os
# class Exploit:
#     def __reduce__(self):
#         return (os.system, ('rm -rf /',))
# pickle.dumps(Exploit())


# 취약: YAML load (임의 Python 객체 허용)
@app.route('/api/config', methods=['POST'])
def load_config_vulnerable():
    config = yaml.load(request.data)  # 안전하지 않음! 코드 실행 가능
    return jsonify(config)


# 수정: 안전한 대안 사용
import json

@app.route('/api/import', methods=['POST'])
def import_data_secure():
    # 데이터 교환에 pickle 대신 JSON 사용
    data = request.get_json()
    if not validate_schema(data):  # 구조 검증
        return jsonify({"error": "잘못된 데이터 형식"}), 400
    return jsonify({"status": "imported"})


@app.route('/api/config', methods=['POST'])
def load_config_secure():
    config = yaml.safe_load(request.data)  # safe_load는 코드 실행 차단
    return jsonify(config)
```

### 9.3 방어

- 소프트웨어/데이터 무결성을 검증하기 위해 디지털 서명 또는 유사한 방법 사용
- 라이브러리 및 종속성이 신뢰할 수 있는 저장소를 사용하도록 보장
- 소프트웨어 공급망 보안 도구 사용 (SLSA, Sigstore)
- 무단 액세스 또는 수정에 대해 CI/CD 파이프라인 검토
- 신뢰할 수 없는 클라이언트에 서명되지 않거나 암호화되지 않은 직렬화된 데이터 전송 안 함
- 신뢰할 수 없는 데이터에 `pickle`, `marshal` 또는 `yaml.load()` 절대 사용 안 함

---

## 10. A09: 보안 로깅 및 모니터링 실패

### 10.1 설명

충분한 로깅 및 모니터링이 없으면 침해를 제때 감지할 수 없습니다. 대부분의 성공적인 공격은 취약점 탐색으로 시작하며, 이러한 탐색이 계속되도록 허용하면 성공적인 악용 가능성이 높아질 수 있습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│      A09: 보안 로깅 및 모니터링 실패                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  문제:                                                           │
│  - 로그인 실패가 로깅되지 않음                                  │
│  - 경고 및 오류가 로그 메시지를 생성하지 않거나 불명확함        │
│  - 로그가 로컬에만 저장 (서버가 침해되면 손실)                  │
│  - 알림 임계값 또는 효과적인 에스컬레이션 없음                  │
│  - 침투 테스트 및 DAST 스캔이 알림을 트리거하지 않음            │
│  - 애플리케이션이 공격을 탐지, 에스컬레이션 또는 알림 불가      │
│  - 로그 인젝션: 공격자가 가짜 로그 항목 작성                    │
│                                                                  │
│  침해 탐지 평균 시간: 287일 (IBM 2021)                          │
│  <200일 내 탐지된 침해 비용: $3.6M                              │
│  >200일 후 탐지된 침해 비용: $4.9M                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 보안 로깅 구현

```python
"""
security_logging.py - 포괄적인 보안 이벤트 로깅
"""
import logging
import json
import time
from flask import Flask, request, g
from datetime import datetime, timezone

app = Flask(__name__)

# ==============================================================
# 보안 이벤트 로거
# ==============================================================

class SecurityLogger:
    """구조화된 보안 이벤트 로깅."""

    def __init__(self, app_name: str):
        self.logger = logging.getLogger(f"security.{app_name}")
        self.logger.setLevel(logging.INFO)

        # 구조화된 로깅을 위한 JSON 포매터
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def _log_event(self, event_type: str, severity: str, **kwargs):
        """구조화된 보안 이벤트 로그."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "severity": severity,
            "ip_address": request.remote_addr if request else None,
            "user_agent": request.headers.get('User-Agent') if request else None,
            "user_id": getattr(g, 'current_user', {}).get('id'),
            **kwargs
        }
        self.logger.info(json.dumps(event))

    def login_success(self, user_id: str):
        self._log_event("auth.login.success", "INFO", user_id=user_id)

    def login_failure(self, username: str, reason: str):
        self._log_event("auth.login.failure", "WARNING",
                       username=self._sanitize(username),
                       reason=reason)

    def access_denied(self, resource: str, action: str):
        self._log_event("authz.denied", "WARNING",
                       resource=resource, action=action)

    def suspicious_activity(self, description: str, **details):
        self._log_event("security.suspicious", "HIGH",
                       description=description, **details)

    def data_access(self, resource_type: str, resource_id: str,
                    action: str):
        self._log_event("data.access", "INFO",
                       resource_type=resource_type,
                       resource_id=resource_id, action=action)

    def _sanitize(self, value: str) -> str:
        """로그 인젝션을 방지하기 위해 로그 입력 정제."""
        if not isinstance(value, str):
            return str(value)
        # 개행 및 제어 문자 제거
        return value.replace('\n', '\\n').replace('\r', '\\r')


sec_log = SecurityLogger("myapp")


# 라우트에서 사용
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    user = authenticate(username, password)
    if user:
        sec_log.login_success(user.id)
        return jsonify({"status": "success"})
    else:
        sec_log.login_failure(username, "invalid_credentials")
        return jsonify({"error": "Invalid credentials"}), 401


# 무차별 대입 모니터링
@app.before_request
def detect_brute_force():
    """잠재적인 무차별 대입 시도 탐지 및 알림."""
    if request.path == '/login' and request.method == 'POST':
        ip = request.remote_addr
        recent_failures = get_recent_login_failures(ip, minutes=5)

        if recent_failures >= 10:
            sec_log.suspicious_activity(
                "무차별 대입 공격 가능성",
                ip_address=ip,
                failure_count=recent_failures,
                timeframe_minutes=5
            )
            # 선택적: IP 차단, CAPTCHA 요구, SOC 알림
```

### 10.3 로그에 기록할 내용

| 이벤트 | 심각도 | 포함할 세부 정보 |
|-------|----------|-------------------|
| 로그인 성공/실패 | INFO/WARNING | 사용자 이름, IP, 타임스탬프, 사용자 에이전트 |
| 인가 실패 | WARNING | 사용자, 리소스, 시도한 작업 |
| 입력 검증 실패 | WARNING | 엔드포인트, 잘못된 입력 유형 |
| 관리자 작업 | INFO | 관리자 사용자, 작업, 대상 |
| 비밀번호 변경 | INFO | 사용자 ID (비밀번호는 절대 안 됨) |
| 계정 잠금 | WARNING | 사용자 이름, 실패 횟수 |
| 데이터 내보내기/다운로드 | INFO | 사용자, 데이터 유형, 볼륨 |
| API 속도 제한 트리거 | WARNING | 클라이언트, 엔드포인트, 속도 |
| 시스템 오류 | ERROR | 오류 유형, 스택 추적 (클라이언트에는 안 됨) |

### 10.4 방어

- 모든 로그인, 접근 제어 및 서버 측 입력 검증 실패 로그
- 포렌식 분석을 위한 충분한 컨텍스트가 로그에 있는지 확인
- 기계 파싱 가능성을 위해 구조화된 로깅 사용 (JSON)
- 중앙 집중식 로그 관리 구현 (ELK, Splunk, CloudWatch)
- 에스컬레이션이 있는 효과적인 모니터링 및 알림 설정
- 인시던트 대응 계획 수립 및 연습
- 변조로부터 로그 보호 (한 번 쓰기 저장소, 무결성 확인)

---

## 11. A10: 서버 측 요청 위조 (SSRF)

### 11.1 설명

SSRF 결함은 웹 애플리케이션이 사용자가 제공한 URL을 검증하지 않고 원격 리소스를 가져올 때 발생합니다. 이를 통해 공격자는 애플리케이션이 방화벽, VPN 또는 네트워크 ACL로 보호되는 경우에도 예상치 못한 대상으로 조작된 요청을 보내도록 강제할 수 있습니다. 이것은 2021년의 **새로운 카테고리**입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              A10: 서버 측 요청 위조 (SSRF)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  정상:                                                           │
│  User ──▶ Server ──▶ https://api.external.com/data              │
│                                                                  │
│  SSRF 공격:                                                      │
│  User ──▶ Server ──▶ http://169.254.169.254/metadata            │
│                       (AWS 인스턴스 메타데이터!)                 │
│                                                                  │
│  User ──▶ Server ──▶ http://localhost:6379/                     │
│                       (내부 Redis 서버!)                         │
│                                                                  │
│  User ──▶ Server ──▶ http://10.0.0.5:8080/admin                │
│                       (내부 관리자 패널!)                        │
│                                                                  │
│  User ──▶ Server ──▶ file:///etc/passwd                         │
│                       (로컬 파일 읽기!)                          │
│                                                                  │
│  영향:                                                           │
│  - 클라우드 인스턴스 메타데이터 접근 (자격 증명 도용)           │
│  - 내부 네트워크 스캔                                            │
│  - 내부 서비스 접근 (Redis, 데이터베이스, 관리자 패널)          │
│  - 로컬 파일 읽기                                                │
│  - Capital One 침해 (2019): SSRF → 메타데이터 → S3 접근        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 취약한 코드

```python
import requests
from urllib.parse import urlparse

# 취약: 임의 URL 가져오기
@app.route('/api/fetch-url', methods=['POST'])
def fetch_url_vulnerable():
    url = request.json['url']
    # 검증 없음! 사용자가 내부 서비스에 접근 가능
    response = requests.get(url)
    return jsonify({"content": response.text})

# 공격 예시:
# {"url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"}
# {"url": "http://localhost:6379/CONFIG SET dir /tmp"}
# {"url": "file:///etc/passwd"}


# 취약: 검증 없는 이미지 프록시
@app.route('/api/proxy-image')
def proxy_image_vulnerable():
    image_url = request.args.get('url')
    response = requests.get(image_url)
    return response.content, 200, {'Content-Type': response.headers.get('Content-Type')}
```

### 11.3 수정된 코드

```python
import ipaddress
import socket
from urllib.parse import urlparse
import requests

# 허용된 도메인의 화이트리스트
ALLOWED_DOMAINS = {
    "api.example.com",
    "images.example.com",
    "cdn.trusted-partner.com",
}

# 차단된 IP 범위 (내부 네트워크)
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local (AWS 메타데이터)
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),       # Carrier-grade NAT
    ipaddress.ip_network("fd00::/8"),             # IPv6 private
    ipaddress.ip_network("::1/128"),              # IPv6 loopback
]


def is_safe_url(url: str) -> bool:
    """URL이 가져오기에 안전한지 검증."""
    try:
        parsed = urlparse(url)

        # HTTP(S) 스킴만 허용
        if parsed.scheme not in ('http', 'https'):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # 옵션 A: 화이트리스트 접근 (가장 강력)
        if hostname not in ALLOWED_DOMAINS:
            return False

        # 옵션 B: 블랙리스트 접근 (화이트리스트가 불가능한 경우)
        # 호스트 이름을 IP로 해석하고 차단된 범위와 비교
        resolved_ips = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in resolved_ips:
            ip = ipaddress.ip_address(sockaddr[0])
            for blocked_range in BLOCKED_IP_RANGES:
                if ip in blocked_range:
                    return False

        return True

    except (ValueError, socket.gaierror):
        return False


# 수정: 가져오기 전 URL 검증
@app.route('/api/fetch-url', methods=['POST'])
def fetch_url_secure():
    url = request.json.get('url', '')

    if not is_safe_url(url):
        return jsonify({"error": "URL이 허용되지 않습니다"}), 400

    try:
        response = requests.get(
            url,
            timeout=5,
            allow_redirects=False,  # 내부 서비스로의 리디렉션 방지
        )

        # 리디렉션이 있는 경우 대상도 검증
        if response.is_redirect:
            redirect_url = response.headers.get('Location', '')
            if not is_safe_url(redirect_url):
                return jsonify({"error": "차단된 URL로의 리디렉션"}), 400

        return jsonify({"content": response.text[:10000]})  # 응답 크기 제한

    except requests.RequestException as e:
        return jsonify({"error": "URL을 가져오지 못했습니다"}), 400
```

### 11.4 방어

- 모든 클라이언트 제공 입력 URL을 정제하고 검증
- 긍정적 허용 목록으로 URL 스킴, 포트 및 대상 강제
- 원시 응답을 클라이언트에 보내지 않기
- HTTP 리디렉션 비활성화
- 네트워크 수준 세그먼테이션 사용 (서버-내부 트래픽을 방지하는 방화벽 규칙)
- 클라우드 환경의 경우: 인스턴스 메타데이터에 대해 IMDSv1 대신 IMDSv2 사용 (토큰 필요)

---

## 12. 방어 체크리스트

```
┌─────────────────────────────────────────────────────────────────┐
│          OWASP Top 10 방어 체크리스트                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  A01 - 취약한 접근 제어:                                         │
│  [ ] 기본 거부 접근 제어                                        │
│  [ ] 리소스 소유권 검증                                          │
│  [ ] 민감한 엔드포인트의 속도 제한                               │
│  [ ] CORS 적절히 구성                                           │
│                                                                  │
│  A02 - 암호화 실패:                                              │
│  [ ] 민감도별로 데이터 분류                                     │
│  [ ] 모든 전송 중 데이터에 TLS 1.2+                             │
│  [ ] 저장 중 데이터에 AES-256-GCM 또는 ChaCha20                │
│  [ ] 비밀번호에 Argon2id/bcrypt                                 │
│                                                                  │
│  A03 - 인젝션:                                                   │
│  [ ] 모든 곳에서 매개변수화된 쿼리                               │
│  [ ] 데이터베이스 접근에 ORM 사용                               │
│  [ ] 입력 검증 (화이트리스트 방식)                              │
│  [ ] 컨텍스트에 맞는 출력 인코딩                                │
│                                                                  │
│  A04 - 안전하지 않은 설계:                                       │
│  [ ] 위협 모델링 수행                                           │
│  [ ] 요구사항에 남용 사례 포함                                  │
│  [ ] 민감한 플로우에 속도 제한, CAPTCHA                         │
│  [ ] 보안 설계 리뷰                                             │
│                                                                  │
│  A05 - 보안 구성 오류:                                           │
│  [ ] 각 환경에 대한 강화 체크리스트                             │
│  [ ] 프로덕션에서 디버그 모드 OFF                               │
│  [ ] 보안 헤더 구성                                             │
│  [ ] 기본 크레덴셜 없음                                          │
│                                                                  │
│  A06 - 취약한 구성 요소:                                         │
│  [ ] CI/CD에서 종속성 스캔                                      │
│  [ ] 정기적인 업데이트 및 패치                                  │
│  [ ] 신뢰할 수 있는 소스의 구성 요소만                          │
│  [ ] SBOM(소프트웨어 자재 명세서) 유지                          │
│                                                                  │
│  A07 - 인증 실패:                                                │
│  [ ] MFA 활성화 (특히 관리자 계정)                              │
│  [ ] 실패한 시도 후 계정 잠금                                   │
│  [ ] 로그인 후 세션 재생성                                      │
│  [ ] 유출된 비밀번호 확인                                       │
│                                                                  │
│  A08 - 무결성 실패:                                              │
│  [ ] 업데이트/배포에 대한 디지털 서명                           │
│  [ ] 접근 제어로 CI/CD 파이프라인 보안                          │
│  [ ] 신뢰할 수 없는 데이터의 역직렬화 금지                      │
│  [ ] 외부 스크립트에 대한 SRI(하위 리소스 무결성)               │
│                                                                  │
│  A09 - 로깅 및 모니터링:                                         │
│  [ ] 충분한 컨텍스트로 보안 이벤트 로깅                         │
│  [ ] 중앙 집중식 로그 관리                                      │
│  [ ] 의심스러운 패턴에 대한 알림                                │
│  [ ] 인시던트 대응 계획 테스트                                  │
│                                                                  │
│  A10 - SSRF:                                                     │
│  [ ] URL 검증 (화이트리스트 선호)                               │
│  [ ] 네트워크 세그먼테이션                                      │
│  [ ] 불필요한 URL 스킴 비활성화                                 │
│  [ ] 클라우드 메타데이터에 IMDSv2                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. 연습 문제

### 연습 문제 1: 취약점 식별

다음 Flask 애플리케이션을 검토하고 각 취약점에 해당하는 OWASP Top 10 카테고리를 식별하세요.

```python
"""
연습 문제: 번호가 매겨진 각 문제에 대한 OWASP Top 10 카테고리를 식별하세요.
일부 줄에는 여러 문제가 있을 수 있습니다.
"""
from flask import Flask, request, jsonify, send_file
import pickle
import os
import sqlite3
import yaml
import requests

app = Flask(__name__)
app.config['DEBUG'] = True                        # 문제 1: ???
app.config['SECRET_KEY'] = 'development'          # 문제 2: ???

@app.route('/api/search')
def search():
    q = request.args.get('q')
    conn = sqlite3.connect('app.db')
    cursor = conn.execute(
        f"SELECT * FROM products WHERE name LIKE '%{q}%'"  # 문제 3: ???
    )
    return jsonify(cursor.fetchall())

@app.route('/api/user/<int:user_id>')
def get_user(user_id):                            # 문제 4: ???
    user = db.get_user(user_id)
    return jsonify(user)

@app.route('/api/import', methods=['POST'])
def import_data():
    data = pickle.loads(request.data)             # 문제 5: ???
    return jsonify({"status": "imported"})

@app.route('/api/fetch', methods=['POST'])
def fetch():
    url = request.json['url']
    resp = requests.get(url)                      # 문제 6: ???
    return resp.text

@app.route('/api/config', methods=['POST'])
def load_config():
    config = yaml.load(request.data)              # 문제 7: ???
    return jsonify(config)

@app.route('/login', methods=['POST'])
def login():
    username = request.json['username']
    password = request.json['password']
    user = db.find_user(username)
    if user and user.password == password:         # 문제 8: ???
        session['user'] = user.id
        return jsonify({"status": "ok"})
    return jsonify({"error": f"User {username} not found or wrong password"}),  401  # 문제 9: ???

@app.errorhandler(500)
def error(e):
    return jsonify({
        "error": str(e),
        "trace": traceback.format_exc()           # 문제 10: ???
    }), 500
```

### 연습 문제 2: 안전한 애플리케이션 설계

파일 공유 애플리케이션에 대한 보안 제어를 설계하고 구현하세요.

```python
"""
연습 문제: 각 OWASP Top 10 카테고리에 대한 보안 제어를 구현하세요.
애플리케이션은 사용자가 파일을 업로드, 공유 및 다운로드할 수 있습니다.
"""

class SecureFileSharing:
    def upload_file(self, user_id: str, file_data: bytes,
                    filename: str) -> dict:
        """
        안전한 파일 업로드.
        고려 사항: A03 (파일 이름을 통한 인젝션), A04 (파일 크기 제한),
                  A05 (파일 유형 검증), A08 (무결성 확인)
        """
        pass

    def share_file(self, owner_id: str, file_id: str,
                   target_user_id: str, permissions: list) -> bool:
        """
        다른 사용자와 파일 공유.
        고려 사항: A01 (접근 제어), A04 (공유 제한)
        """
        pass

    def download_file(self, user_id: str, file_id: str) -> bytes:
        """
        파일 다운로드.
        고려 사항: A01 (접근 제어), A09 (로깅),
                  A10 (파일이 외부 URL을 참조하는 경우)
        """
        pass

    def fetch_external_file(self, url: str) -> bytes:
        """
        외부 URL에서 파일 가져오기.
        고려 사항: A10 (SSRF), A06 (URL 라이브러리 검증)
        """
        pass
```

### 연습 문제 3: 보안 감사 보고서

시뮬레이션된 보안 감사 수행:

```
연습 문제: 다음과 같은 특성을 가진 웹 애플리케이션이 주어졌을 때:
- Python/Flask 백엔드
- PostgreSQL 데이터베이스
- JWT 인증
- 파일 업로드 기능
- 웹훅 통합 (외부 URL 가져오기)
- 15개의 Python 종속성 사용 (6개월 동안 감사되지 않음)
- 로컬 파일에만 기록된 로그
- 속도 제한 없음
- AWS EC2에서 실행

각 OWASP Top 10 카테고리에 대해:
1. 이 애플리케이션에 대한 특정 위험 식별
2. 위험 등급 평가 (Critical/High/Medium/Low)
3. 구체적인 해결 단계 제안
4. 구현 노력 추정

발견 사항을 구조화된 보안 감사 보고서로 작성하세요.
```

### 연습 문제 4: 취약한 애플리케이션 수정

연습 문제 1의 코드를 가져와서 모든 취약점을 수정하여 다시 작성하세요. 수정된 버전은 모든 OWASP Top 10 카테고리를 다루어야 합니다.

### 연습 문제 5: OWASP Top 10 매핑

다음 실제 침해 사례를 OWASP Top 10 카테고리에 매핑하세요.

```
1. Equifax (2017) - 패치되지 않은 Apache Struts 취약점
   → A0?: ___

2. Capital One (2019) - AWS 메타데이터 접근을 위한 SSRF
   → A0?: ___

3. SolarWinds (2020) - 침해된 빌드 파이프라인
   → A0?: ___

4. Facebook (2019) - 보호되지 않은 S3의 5억 4천만 사용자 레코드
   → A0?: ___

5. Uber (2016) - GitHub 저장소에 하드코딩된 AWS 자격 증명
   → A0?: ___

6. British Airways (2018) - 결제 페이지의 Magecart XSS
   → A0?: ___

7. Marriott (2018) - 4년 동안 감지되지 않은 침해
   → A0?: ___

8. Yahoo (2013-2014) - 사용자 데이터의 약한/암호화 없음
   → A0?: ___
```

---

## 14. 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                OWASP Top 10 (2021) 요약                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  A01: 취약한 접근 제어 - 1위 위협. 기본 거부.                   │
│  A02: 암호화 실패 - 민감한 모든 것을 암호화.                    │
│  A03: 인젝션 - 모든 쿼리를 매개변수화.                          │
│  A04: 안전하지 않은 설계 - 보안은 코드가 아닌 설계에서 시작.   │
│  A05: 보안 구성 오류 - 모든 것을 강화.                          │
│  A06: 취약한 구성 요소 - 종속성을 알고 업데이트.                │
│  A07: 인증 실패 - MFA, 속도 제한, 강력한 비밀번호.              │
│  A08: 무결성 실패 - 모든 것에 서명, pickle 사용 금지.           │
│  A09: 로깅 실패 - 보안 이벤트 로깅, 알림, 대응.                 │
│  A10: SSRF - 모든 URL 검증, 네트워크 세그먼트.                  │
│                                                                  │
│  OWASP Top 10은 시작점이지 완전한 목록이 아닙니다.              │
│  애플리케이션 보안을 위한 최소 기준선으로 사용하세요.            │
│                                                                  │
│  리소스:                                                         │
│  - https://owasp.org/Top10/                                     │
│  - OWASP Application Security Verification Standard (ASVS)     │
│  - OWASP Testing Guide                                          │
│  - OWASP Cheat Sheet Series                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**이전**: [06. 인가와 접근 제어](06_Authorization.md) | **다음**: [08. 인젝션 공격과 방어](08_Injection_Attacks.md)
