# 로깅

**이전**: [디버깅 전략](./05_Debugging_Strategy.md) | **다음**: [테스트 기초](./07_Testing_Basics.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 프로덕션 코드에서 로깅이 print 디버깅보다 우수한 이유 설명하기
2. `logging` 모듈을 적절한 로그 레벨(DEBUG~CRITICAL)로 사용하기
3. 타임스탬프, 소스 위치, 커스텀 필드가 포함된 로그 출력 형식 설정하기
4. 핸들러를 사용하여 파일, 콘솔, 다중 목적지로 로그 출력 보내기
5. 모듈형 애플리케이션을 위해 이름이 있는 로거 생성 및 사용하기
6. `logging.basicConfig()`과 딕셔너리 기반 설정 적용하기
7. 기계 판독 가능한 출력을 위한 구조화된 로깅 패턴 사용하기
8. 흔한 로깅 함정(성능, 보안, 포맷) 피하기

---

print 디버깅은 빠른 조사에 효과적이지만, 프로덕션 코드에는 더 견고한 것이 필요합니다. `logging` 모듈은 심각도별 필터링, 다른 목적지로의 라우팅, 코드에 영구적으로 남길 수 있는 진단 출력을 위한 Python 내장 솔루션입니다. print 문과 달리 로그 호출은 코드 변경 없이 켜고 끌 수 있습니다.

> **경험 법칙:** 프로덕션에서 실행되는 코드의 문제를 진단하기 위해 `print()`를 사용하려 한다면, 대신 `logging`을 사용하세요. print는 개발용, logging은 운영용입니다.

---

## 1. 왜 print가 아니라 Logging인가

| 기능 | `print()` | `logging` |
|------|-----------|-----------|
| 심각도 레벨 | 없음 | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| 코드 변경 없이 활성/비활성 | 불가 | 가능 (설정으로) |
| 타임스탬프 | 수동 | 자동 |
| 출력 목적지 | stdout만 | 파일, stderr, 네트워크, 이메일 등 |
| 소스 위치 | 수동 | 자동 (파일, 줄, 함수) |
| 스레드 안전성 | 없음 | 있음 |
| 성능 제어 | 없음 | 포맷 전 레벨 필터링 |
| 프로덕션 준비 | 아님 | 준비됨 |

---

## 2. 빠른 시작

### 2.1 가장 간단한 예제

```python
import logging

logging.basicConfig(level=logging.DEBUG)

logging.debug("계산 시작")
logging.info("100개 레코드 처리 중")
logging.warning("디스크 공간 10% 미만")
logging.error("데이터베이스 연결 실패")
logging.critical("시스템 메모리 부족, 종료 중")
```

### 2.2 기본 동작

`basicConfig()` 없이는 WARNING 이상만 표시됩니다:

```python
import logging
logging.debug("표시되지 않음")
logging.info("표시되지 않음")
logging.warning("이것은 표시됨")
```

---

## 3. 로그 레벨

```
┌──────────────────────────────────────────────────────┐
│  레벨      값    사용 시점                            │
├──────────────────────────────────────────────────────┤
│  DEBUG      10   상세한 진단 정보                     │
│                  (변수 값, 흐름 추적)                  │
│  INFO       20   정상 동작 확인                       │
│                  ("서버가 8080 포트에서 시작됨")        │
│  WARNING    30   예상치 못한 상황이나 아직 고장 아님    │
│                  ("재시도 3/5회")                      │
│  ERROR      40   실패했으나 프로그램은 계속됨            │
│                  ("DB 쿼리 실패")                      │
│  CRITICAL   50   프로그램 계속 불가                    │
│                  ("메모리 부족, 종료 중")               │
└──────────────────────────────────────────────────────┘

     심각도 증가 →→→→→→→→→→→→→→→→→→→→→
```

### 올바른 레벨 선택

```python
import logging
logger = logging.getLogger(__name__)

def process_order(order):
    logger.debug(f"주문 처리 중: {order}")           # 개발 상세 정보
    
    if order["total"] > 10000:
        logger.info(f"대형 주문: #{order['id']}")    # 비즈니스 이벤트
    
    if order["stock"] < order["quantity"]:
        logger.warning(f"재고 부족: {order['item']}")  # 우려 사항
    
    try:
        charge_payment(order)
    except PaymentError as e:
        logger.error(f"결제 실패 주문 #{order['id']}: {e}")  # 실패
    
    if not db.is_connected():
        logger.critical("데이터베이스 연결 끊김!")    # 시스템 장애
```

---

## 4. 로그 포맷

### 4.1 포맷 문자열

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("서버 시작")
# 2024-01-15 10:30:45 [INFO    ] root:8 - 서버 시작
```

### 4.2 주요 포맷 필드

| 필드 | 설명 | 예시 |
|------|------|------|
| `%(asctime)s` | 타임스탬프 | `2024-01-15 10:30:45,123` |
| `%(levelname)s` | 로그 레벨명 | `INFO` |
| `%(name)s` | 로거 이름 | `my_module` |
| `%(filename)s` | 소스 파일명 | `app.py` |
| `%(lineno)d` | 줄 번호 | `42` |
| `%(funcName)s` | 함수명 | `process_order` |
| `%(message)s` | 로그 메시지 | `처리 완료` |

---

## 5. 핸들러: 로그가 가는 곳

### 5.1 다중 핸들러 (일반적인 패턴)

```python
import logging

def setup_logging():
    logger = logging.getLogger("myapp")
    logger.setLevel(logging.DEBUG)
    
    # 콘솔: INFO 이상
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    
    # 파일: 모든 것 (DEBUG 이상)
    file_handler = logging.FileHandler("debug.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s"
    ))
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger
```

### 5.2 핸들러 아키텍처

```
              Logger (myapp)
              level=DEBUG
                   │
          ┌────────┴────────┐
          ▼                 ▼
    StreamHandler     FileHandler
    level=INFO        level=DEBUG
          │                 │
          ▼                 ▼
    콘솔 (stderr)     app.log
    INFO+ 메시지      모든 메시지
```

### 5.3 순환 파일 핸들러

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "app.log",
    maxBytes=5_000_000,     # 파일당 5 MB
    backupCount=3,          # 백업 파일 3개 유지
)
```

---

## 6. 이름이 있는 로거

### 6.1 모듈 수준 로거

각 모듈은 자체 로거를 가져야 합니다:

```python
# file: database.py
import logging
logger = logging.getLogger(__name__)   # 로거 이름 = "database"

def connect():
    logger.info("데이터베이스 연결 중...")
    ...

# file: api.py
import logging
logger = logging.getLogger(__name__)   # 로거 이름 = "api"

def handle_request():
    logger.debug("요청 수신")
    ...
```

### 6.2 선택적 로깅

```python
# 데이터베이스 관련 디버그 로그만 표시, 나머지는 침묵
logging.getLogger("myapp").setLevel(logging.WARNING)
logging.getLogger("myapp.database").setLevel(logging.DEBUG)
```

---

## 7. 설정 방법

### 7.1 딕셔너리 설정 (애플리케이션용)

```python
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "app.log",
        },
    },
    "loggers": {
        "myapp": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

---

## 8. 로깅 모범 사례

### 8.1 지연 포맷 사용

```python
# 나쁜 예: 레벨이 필터링되어도 문자열이 항상 포맷됨
logger.debug(f"Processing {len(large_list)} items: {large_list}")

# 좋은 예: DEBUG 레벨이 활성화된 경우에만 포맷됨
logger.debug("Processing %d items: %s", len(large_list), large_list)
```

### 8.2 예외를 올바르게 로깅

```python
# 나쁜 예: 스택 트레이스를 잃음
try:
    process()
except Exception as e:
    logger.error(f"실패: {e}")

# 좋은 예: 전체 트레이스백 포함
try:
    process()
except Exception:
    logger.exception("처리 실패")  # 자동으로 트레이스백 추가
```

### 8.3 민감한 데이터 로깅 금지

```python
# 나쁜 예: 비밀번호가 로그에!
logger.info(f"User login: {username}, password: {password}")

# 좋은 예: 민감한 필드 삭제
logger.info(f"User login: {username}")
```

---

## 9. Logging vs Print: 결정 가이드

```
진단 출력이...
│
├─ 임시적 (커밋 전에 제거)?
│   └─ print() 또는 debug_print() 사용
│
├─ 영구적 (코드베이스에 남음)?
│   └─ logging 사용
│
├─ 프로덕션에서 필요?
│   └─ logging 사용
│
├─ 간단한 일회성 스크립트?
│   └─ print()로 충분
│
└─ 프로그램의 실제 출력의 일부?
    └─ print() 사용 (stdout으로)
        진단은 logging으로 (stderr/파일로)
```

---

## 요약

- 빠른 조사를 넘어서는 코드에는 `print()` 대신 `logging` 사용
- 올바른 레벨 선택: 상세 정보는 DEBUG, 마일스톤은 INFO, 우려는 WARNING, 실패는 ERROR, 치명적 장애는 CRITICAL
- 타임스탬프, 소스 위치, 로거 이름으로 로그 포맷 설정
- 핸들러로 콘솔, 파일, 또는 둘 다에 출력 전송
- `logging.getLogger(__name__)`으로 모듈별 이름 있는 로거 생성
- 성능을 위해 지연 포맷(`%s` 스타일) 사용
- `logger.exception()`으로 트레이스백을 캡처하여 예외 로깅
- 민감한 데이터(비밀번호, 토큰, 개인정보)를 절대 로깅하지 말 것

---

## 연습문제

1. 타임스탬프와 줄 번호가 포함된 커스텀 포맷으로 기본 로깅 설정하기
2. 다중 핸들러 설정 만들기: 콘솔(INFO+)과 파일(DEBUG+)
3. 적절한 레벨을 사용하여 데이터 처리 함수에 로깅 추가하기
4. 딕셔너리 설정으로 로깅 구성하기

**이전**: [디버깅 전략](./05_Debugging_Strategy.md) | **다음**: [테스트 기초](./07_Testing_Basics.md)
