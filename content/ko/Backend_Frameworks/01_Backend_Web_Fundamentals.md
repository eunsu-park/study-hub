# 01. 백엔드 웹 기초(Backend Web Fundamentals)

**이전**: [개요](./00_Overview.md) | **다음**: [FastAPI 기초](./02_FastAPI_Basics.md)

**난이도**: ⭐⭐

---

## 학습 목표

- HTTP 요청/응답 사이클을 메서드, 헤더, 상태 코드, 본문 형식을 포함하여 설명할 수 있다
- REST 원칙을 적용하여 적절한 CRUD 매핑이 포함된 리소스 중심 API를 설계할 수 있다
- WSGI와 ASGI 서버 모델을 비교하고 각각이 적합한 상황을 파악할 수 있다
- 변화하는 서비스에 대한 일관성 있는 API 버전 관리 전략을 설계할 수 있다
- 라우팅부터 응답까지 일반적인 웹 프레임워크에서 요청 생명주기를 추적할 수 있다

---

## 목차

1. [HTTP 요청/응답 사이클](#1-http-요청응답-사이클)
2. [REST 원칙](#2-rest-원칙)
3. [WSGI vs ASGI](#3-wsgi-vs-asgi)
4. [API의 공용어로서의 JSON](#4-api의-공용어로서의-json)
5. [API 버전 관리 전략](#5-api-버전-관리-전략)
6. [웹 프레임워크의 요청 생명주기](#6-웹-프레임워크의-요청-생명주기)
7. [연습 문제](#7-연습-문제)
8. [참고 자료](#8-참고-자료)

---

## 1. HTTP 요청/응답 사이클

클라이언트(브라우저, 모바일 앱, CLI 도구)와 백엔드 서버 간의 모든 상호작용은 **HTTP 요청/응답** 모델을 따릅니다. 클라이언트가 요청을 보내면 서버가 이를 처리하고 응답을 반환합니다.

### 이론: HTTP: 와이어 프로토콜

HTTP는 신뢰성 있는 바이트 스트림(TCP, HTTP/3에서는 QUIC) 위에 얹힌 요청/응답 프로토콜입니다. 모든 요청과 응답은 동일한 세 부분 구조를 가집니다: **시작 라인(start line)**, **헤더(headers)**, **본문(body)**. 클라이언트와 서버 사이의 모든 계약은 이 세 영역에 인코딩됩니다.

#### A.1 와이어 포맷의 진화

| 버전 | 전송 | 동시성 모델 | 핵심 변경 |
|------|------|-------------|----------|
| HTTP/1.0 | TCP | 연결당 요청 하나 | 무상태, 요청마다 연결 |
| HTTP/1.1 | TCP | 파이프라이닝(거의 안 씀), keep-alive | 영구 연결, 청크 인코딩, `Host` 헤더 |
| HTTP/2 | TCP + TLS | 단일 연결 위에서 멀티플렉싱 | 바이너리 프레이밍, 헤더 압축(HPACK), 서버 푸시 |
| HTTP/3 | QUIC (UDP) | HoL 블로킹 없는 스트림 멀티플렉싱 | 독립 스트림; 한 스트림 패킷 손실이 다른 스트림을 막지 않음 |

가장 큰 영향을 주는 성질은 **head-of-line(HoL) 블로킹**입니다. HTTP/1.1에서는 단일 TCP 연결이 요청들을 직렬화합니다. 요청 1이 느리면 같은 소켓의 요청 2, 3은 그 뒤에서 대기합니다. 브라우저는 origin당 6개의 병렬 연결을 열어 이 제약을 우회했습니다. HTTP/2는 하나의 TCP 연결에 여러 논리적 *스트림*을 다중화하지만, TCP 계층의 패킷 손실은 여전히 모든 스트림을 막습니다(TCP HoL 블로킹). HTTP/3는 전송 계층을 TCP에서 QUIC로 옮겨, 각 스트림이 자체 손실 복구 상태를 갖도록 합니다. 한 스트림에서 떨어진 패킷이 더 이상 다른 스트림을 멈추지 않습니다.

백엔드 개발자에게 실용적 결과는 **HTTP/2 이상에서는 요청을 묶을 역사적 동기가 사라졌다**는 것입니다. 스프라이트 시트, JS 번들, 리소스 concat은 HTTP/1.1에서 추가 요청마다 연결 설정 비용을 치러야 했기 때문에 존재했습니다. 멀티플렉싱이 가능해지면서 세분화된 리소스가 다시 유효한 설계 선택이 되었습니다.

#### A.2 상태 코드의 의미 — 그저 숫자가 아니다

상태 코드의 첫 자리가 *계급(class)*이며, 계약은 계급 단위에서 성립합니다.

- **1xx — 정보.** `100 Continue`는 클라이언트가 본문을 보내기 전에 서버가 받아줄지 확인할 수 있게 해줍니다. 자주 보이지는 않지만 의도적인 코드입니다.
- **2xx — 성공.** `200 OK`(응답에 본문 있음), `201 Created`(새 리소스의 `Location` 헤더 포함), `204 No Content`(성공이지만 의도적으로 본문 없음 — `DELETE`와 대부분의 `PUT`에 적합).
- **3xx — 리다이렉션.** `301` 영구(영원히 캐시), `302/307` 임시, `304 Not Modified`(`If-None-Match`로 지정된 캐시 버전과 본문이 동일).
- **4xx — 클라이언트 오류.** 요청 자체가 잘못됐거나 권한이 없습니다. `400`(문법 오류), `401`(자격 증명 없음), `403`(자격 증명은 있으나 부족), `404`(리소스 없음), `409`(상태 충돌), `422`(문법은 맞지만 의미가 잘못됨 — Pydantic 스타일 검증 실패), `429`(요청 제한 초과).
- **5xx — 서버 오류.** 유효한 요청이었는데도 서버가 실패했습니다. `500`(처리되지 않은 예외), `502/503/504`(상위/의존성 실패).

주니어와 시니어 API 설계자를 가르는 규율은 **더 정확한 코드가 있을 때 절대 `200`으로 도망치지 않는 것**입니다. 리소스를 만든 `POST`는 `Location: /users/42`와 함께 `201`을 반환해야 합니다. 성공한 `DELETE`는 `204`를 돌려줘야 합니다. 검증 실패는 `400`이 아니라 `422`여야 합니다. 이는 스타일 문제가 아니라, 캐시·모니터링 도구·클라이언트 SDK 모두가 상태 계급을 기준으로 분기하기 때문입니다.

#### A.3 헤더는 확장 가능한 협상 채널

헤더는 본문에 들어가기 부적절한 모든 횡단 관심사를 운반합니다. 인증(`Authorization`), 콘텐츠 협상(`Accept`, `Content-Type`), 캐싱(`ETag`, `If-None-Match`, `Cache-Control`), 트레이싱(`Traceparent`), 요청 제한(`X-RateLimit-Remaining`). 두 가지 원칙을 기억해야 합니다.

1. **본문은 중간자에게 불투명하지만 헤더는 그렇지 않다.** 프록시, CDN, 로드 밸런서는 헤더를 기준으로 라우팅·캐싱·재작성합니다. 네트워크 경로에 영향을 주려는 정보는 반드시 헤더에 두어야 합니다.
2. **커스텀 헤더에 `X-` 접두를 붙이지 마세요.** RFC 6648은 이 관행을 폐기했습니다. 한 헤더가 유용해지면 표준화되는데, 그때 이름을 바꾸면 모든 클라이언트가 깨지기 때문입니다.

### HTTP 요청의 구조

```
POST /api/users HTTP/1.1          <-- 요청 라인: 메서드 경로 버전
Host: api.example.com             <-- 헤더 시작
Content-Type: application/json
Authorization: Bearer eyJhbG...
Content-Length: 56
                                  <-- 빈 줄이 헤더와 본문을 구분
{"name": "Alice", "email": "alice@example.com"}   <-- 본문 (선택 사항)
```

### HTTP 메서드

| 메서드 | 목적 | 멱등성(Idempotent) | 안전성(Safe) | 본문 여부 |
|--------|---------|-----------|------|----------|
| `GET` | 리소스 조회 | 예 | 예 | 아니오 |
| `POST` | 새 리소스 생성 | 아니오 | 아니오 | 예 |
| `PUT` | 리소스 전체 교체 | 예 | 아니오 | 예 |
| `PATCH` | 리소스 부분 업데이트 | 아니오 | 아니오 | 예 |
| `DELETE` | 리소스 삭제 | 예 | 아니오 | 선택 사항 |
| `HEAD` | GET과 동일하지만 본문 없음 | 예 | 예 | 아니오 |
| `OPTIONS` | 허용된 메서드 조회 (CORS) | 예 | 예 | 아니오 |

**멱등성(Idempotent)**이란 동일한 요청을 여러 번 호출해도 결과가 같다는 의미입니다. 같은 본문으로 `PUT /users/42`를 반복 호출하면 항상 동일한 상태가 됩니다. 반면 `POST /users`는 매번 중복 데이터를 생성할 수 있습니다.

### HTTP 응답의 구조

```
HTTP/1.1 201 Created             <-- 상태 라인: 버전 코드 사유
Content-Type: application/json
Location: /api/users/42          <-- 새 리소스의 위치
X-Request-Id: abc-123
                                 <-- 빈 줄
{"id": 42, "name": "Alice", "email": "alice@example.com"}
```

### 상태 코드 패밀리

| 범위 | 분류 | 주요 코드 |
|-------|----------|-------------|
| `1xx` | 정보 | `101 Switching Protocols` (WebSocket 업그레이드) |
| `2xx` | 성공 | `200 OK`, `201 Created`, `204 No Content` |
| `3xx` | 리다이렉션 | `301 Moved Permanently`, `304 Not Modified` |
| `4xx` | 클라이언트 오류 | `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `422 Unprocessable Entity`, `429 Too Many Requests` |
| `5xx` | 서버 오류 | `500 Internal Server Error`, `502 Bad Gateway`, `503 Service Unavailable` |

유용한 규칙: 클라이언트가 잘못된 요청을 보낸 경우 `4xx`를 반환하고, 서버에서 실패한 경우 `5xx`를 반환합니다. 오류 본문과 함께 `200`을 반환하는 것은 클라이언트와 모니터링 도구를 혼란스럽게 만드므로 절대 해서는 안 됩니다.

### 요청 흐름 다이어그램

```
  클라이언트                          서버
    |                                |
    |  ---- HTTP 요청 -----------> |
    |       메서드 + URL             |
    |       헤더                    |
    |       본문 (선택 사항)          |
    |                                |
    |                          [ 처리   ]
    |                          [ 라우팅  ]
    |                          [ 로직   ]
    |                          [ DB 호출 ]
    |                                |
    |  <--- HTTP 응답 ------------ |
    |       상태 코드               |
    |       헤더                    |
    |       본문 (선택 사항)          |
    |                                |
```

---

## 2. REST 원칙

**REST**(Representational State Transfer, 표현 상태 전달)는 프로토콜이 아닌 아키텍처 스타일입니다. 예측 가능하고 확장 가능하며 사용하기 쉬운 웹 API를 설계하기 위한 지침을 제공합니다.

### 이론: REST: 아키텍처 스타일

REST는 "JSON over HTTP"가 아닙니다. Roy Fielding이 2000년 박사 논문에서 HTTP의 설계 자체로부터 추출한 6가지 아키텍처 제약입니다. 이를 위반한 시스템도 동작은 하지만, REST가 아니며, Fielding의 분석은 그것이 어디서 진화하기 어려워질지를 예측합니다.

#### B.1 6가지 제약

1. **클라이언트-서버.** 관심사 분리. 클라이언트는 사용자 상호작용, 서버는 데이터와 규칙을 담당합니다. 양쪽이 독립적으로 진화할 수 있습니다.
2. **무상태(Stateless).** 클라이언트에서 서버로 가는 모든 요청이 그 자체로 이해 가능해야 합니다. 서버에 세션 상태를 두지 않습니다. 이것이 수평 확장을 단순하게 만드는 핵심입니다 — 어떤 워커든 어떤 요청이든 처리할 수 있습니다.
3. **캐시 가능(Cacheable).** 응답은 캐시 가능 여부를 명시해야 합니다. 캐시는 클라이언트, CDN, 리버스 프록시 등 어디에든 있을 수 있습니다. `Cache-Control`과 `ETag`가 그 메커니즘입니다.
4. **계층 시스템(Layered system).** 클라이언트는 자신이 origin 서버, CDN, 로드 밸런서, 서비스 메시 사이드카 중 무엇과 통신하는지 알 수 없어야 합니다. 각 계층은 다음 계층을 수정하지 않고 횡단 정책(인증, 요청 제한, 관측성)을 추가할 수 있습니다.
5. **균일 인터페이스(Uniform interface).** 리소스는 URI로 식별되고, 작은 고정 동사 어휘로 조작되며, `Content-Type`을 통해 자기 기술적입니다. 클라이언트는 인터페이스 하나를 배우고 모든 서비스에서 재사용합니다.
6. **요청 시 코드(Code-on-demand, 선택).** 서버가 실행 가능한 코드(JavaScript)를 보내 클라이언트 기능을 확장할 수 있습니다. 6개 중 유일하게 선택적이며, 대부분의 API는 사용하지 않습니다.

가장 결과가 깊은 제약은 **무상태성**입니다. 서버가 클라이언트별 상태를 보관하지 않으면, 배포는 롤링이 되고, 확장은 워커 추가가 되며, 크래시는 일시적 현상이 됩니다 — 워커 A에서 실패한 요청이 워커 B에서 성공할 수 있습니다. 두 워커가 교환 가능하기 때문입니다.

#### B.2 멱등성 위계

**안전(safe)**은 리소스를 수정하지 않는다는 뜻이고, **멱등(idempotent)**은 같은 호출을 반복해도 한 번 호출한 것과 효과가 같다는 뜻입니다. 둘 다 수학적 성질이며, *프레임워크가 강제할 수 없는 계약*입니다. 핸들러를 작성하는 사람이 지켜야 합니다.

| 메서드 | 안전 | 멱등 | 왜 중요한가 |
|--------|------|------|------------|
| GET, HEAD, OPTIONS | 예 | 예 | 캐시와 프록시가 자유롭게 prefetch할 수 있다 |
| PUT, DELETE | 아니오 | 예 | 네트워크 실패 시 재시도가 안전하다 |
| POST | 아니오 | 아니오 | 재시도가 중복 생성을 만들 수 있다; `Idempotency-Key` 헤더가 필요하다 |
| PATCH | 아니오 | 경우에 따라 | 교체 형태의 PATCH는 멱등; "카운터 증가"는 아니다 |

모바일 클라이언트가 `POST /payments`를 기다리다 타임아웃이 나면, 안전한 선택지는 둘뿐입니다. (1) 서버가 중복을 제거할 수 있도록 idempotency key와 함께 재시도하거나, (2) 서버에 "내 요청을 이미 처리했나요?"라고 묻는 것. 두 경우 모두 POST가 비멱등이라는 사실의 직접적 결과입니다.

#### B.3 리소스 모델링: 동사보다 명사

REST URI는 *동작*(동사)이 아니라 *리소스*(명사)를 명명합니다. 동사는 HTTP 메서드입니다. 비교해 보세요.

```
RPC 형태:    POST /createUser, POST /deleteUser, POST /promoteUser
REST 형태:   POST /users, DELETE /users/42, PATCH /users/42  (본문 {"role": "admin"})
```

REST 형태는 리소스당 N개의 엔드포인트를 단일 엔드포인트 패밀리로 압축합니다. 클라이언트는 "모든 리소스가 표준 동사를 말한다"는 하나의 멘탈 모델을 익히고, 엔드포인트 이름을 외울 필요가 없어집니다.

### 핵심 원칙

1. **행위가 아닌 리소스**: URL은 동사(`/getUser?id=42`)가 아닌 명사(`/users/42`)를 식별합니다
2. **무상태성(Statelessness)**: 각 요청은 처리에 필요한 모든 정보를 담고 있습니다. 서버는 요청 간에 클라이언트 세션 상태를 저장하지 않습니다.
3. **균일한 인터페이스(Uniform interface)**: 모든 리소스에 걸쳐 표준 HTTP 메서드를 일관성 있게 사용합니다.
4. **HATEOAS**(Hypermedia As The Engine Of Application State): 응답에 관련 리소스의 링크가 포함됩니다. 실제로는 많은 API가 이를 생략합니다.

### CRUD 매핑

| 연산 | HTTP 메서드 | URL 패턴 | 응답 코드 |
|-----------|------------|-------------|---------------|
| 생성(Create) | `POST` | `/api/users` | `201 Created` |
| 조회(Read, 목록) | `GET` | `/api/users` | `200 OK` |
| 조회(Read, 상세) | `GET` | `/api/users/42` | `200 OK` |
| 수정(Update, 전체) | `PUT` | `/api/users/42` | `200 OK` |
| 수정(Update, 부분) | `PATCH` | `/api/users/42` | `200 OK` |
| 삭제(Delete) | `DELETE` | `/api/users/42` | `204 No Content` |

### 리소스 명명 규칙

```
# 좋음 - 복수 명사, 계층적 관계
GET  /api/users/42/orders          # 사용자 42에 속한 주문 목록
GET  /api/users/42/orders/7        # 특정 주문
POST /api/users/42/orders          # 사용자 42의 주문 생성

# 나쁨 - URL에 동사 포함, 평면 구조
GET  /api/getUserOrders?userId=42
POST /api/createOrder
```

### 필터링, 정렬, 페이지네이션(Pagination)

```
# 쿼리 파라미터를 사용한 필터링
GET /api/users?role=admin&status=active

# 정렬 (내림차순은 - 접두사 사용)
GET /api/users?sort=-created_at,name

# 페이지네이션 (오프셋 기반)
GET /api/users?page=2&per_page=25

# 페이지네이션 (커서 기반 -- 대용량 데이터셋에 적합)
GET /api/users?cursor=eyJpZCI6NDJ9&limit=25
```

---

## 3. WSGI vs ASGI

Python 웹 서버는 웹 서버와 애플리케이션 사이에 표준 인터페이스가 필요합니다. 두 가지 표준이 존재합니다: **WSGI**(동기) 와 **ASGI**(비동기).

### 이론: 서버 실행 모델

세 번째 축은 앞의 두 축과 직교합니다. 단일 프로세스가 어떻게 수천 개의 동시 HTTP 요청을 처리하는가? 세 가지 동시성 모델이 시장을 지배하며, 모든 Python/Node/Go 웹 프레임워크는 그중 하나 위에 세워져 있습니다.

#### C.1 동기, 요청당 스레드(Django 클래식, Flask)

가장 단순한 모델입니다. 각 요청에 워커 스레드(또는 프로세스)가 할당됩니다. 핸들러는 끝까지 실행되며, I/O(데이터베이스, 다운스트림 HTTP)에서 스레드를 블로킹한 뒤 해제합니다. 처리량은 `워커 수`로 제한되며, 워커 수는 RAM에 의해 제한됩니다(각 Python 스레드가 자체 스택을 갖고, GIL이 두 스레드의 Python 바이트코드 동시 실행을 막기 때문에 워커는 보통 OS 프로세스입니다).

Gunicorn 워커 4개로 띄운 WSGI 배포는 *동시 진행 중*인 요청 4개를 처리할 수 있습니다. 다운스트림 호출이 100ms 걸리면 워커 한 명을 100ms 동안 묶습니다. 산수는 가차없습니다. 모든 요청이 데이터베이스에서 200ms를 기다린다면 최대 처리량은 `4 × (1000ms / 200ms) = 20 RPS`입니다. 해법은 워커를 늘리거나(메모리 추가) 비동기 모델로 옮기는 것입니다.

#### C.2 이벤트 루프, 단일 스레드 비동기(Node.js, FastAPI/Starlette, asyncio)

요청당 스레드가 아니라, 한 스레드가 **이벤트 루프**를 돌리며 수천 개의 진행 중 요청을 다중화합니다. 핸들러가 I/O를 await하면 루프가 그 코루틴을 일시 중지하고 다른 코루틴을 실행합니다. I/O가 완료되면 루프가 원래 코루틴을 재개합니다. 스레드는 절대 블로킹되지 않으며, 항상 *무언가*를 실행하고 있습니다.

이 모델은 핸들러가 **I/O 바운드**일 때 — 즉, 데이터베이스, 캐시, 다운스트림 서비스를 기다리는 대부분의 웹 API에서 — 정확히 승리합니다. 핸들러가 **CPU 바운드**일 때는 패배합니다. CPU 무거운 코루틴 하나가 전체 루프를 굶깁니다. 스케줄링할 다른 일이 없기 때문입니다. 완화책은 CPU 작업을 스레드 풀로 떠넘기는 것입니다(`asyncio.run_in_executor`, Node `worker_threads`).

Python에서 인터페이스 계약은 **ASGI**입니다. WSGI의 비동기 인지 후속입니다. Starlette, FastAPI, Quart 모두 이를 구현합니다.

#### C.3 경량 "그린" 스레드(Go 고루틴, JVM 가상 스레드, Erlang)

고루틴은 런타임이 OS 스레드 위에 다중화하는 사용자 공간 스레드입니다. 각 고루틴은 시작 시 약 2KB의 스택을 차지하며 필요에 따라 늘어납니다. Go 서버는 백만 개의 고루틴이 I/O에서 블로킹되어 있어도 멀쩡합니다. 런타임이 이벤트 루프 작업을 *투명하게* 수행합니다. 코드는 동기처럼 보이지만 비동기처럼 실행됩니다.

많은 워크로드에서 두 모델의 장점을 모두 갖습니다 — 동기처럼 보이는 코드와 비동기 수준의 확장성, 게다가 코어 간 진정한 병렬성(GIL 없음)까지. 비용은 언어 차원입니다. 런타임을 바꾸지 않고는 Python에 이 모델을 끼워 넣을 수 없습니다.

#### C.4 백 프레셔(back-pressure): 세 모델이 공유하는 성질

서버가 처리할 수 있는 속도보다 빨리 요청을 받아들이면 결국 메모리, 파일 디스크립터, 요청 큐가 고갈됩니다. 방어책은 **백 프레셔**입니다. 어떤 부하 수준에서 서버는 새 작업을 거부해야 합니다(`503` 반환 또는 연결 종료) — 호출자가 속도를 늦추도록 말입니다. 구체적으로는 연결 수락 큐 상한, 진행 중 요청 수 상한, 데이터베이스 연결 풀 상한이 있습니다. 세 실행 모델은 백 프레셔를 *어디에* 적용하는지가 다를 뿐, *필요한지 여부*는 동일합니다.

### WSGI (Web Server Gateway Interface)

WSGI는 PEP 3333 (2010)에서 정의되었습니다. 워커 프로세스당 한 번에 하나의 요청을 처리합니다.

```python
# 최소한의 WSGI 애플리케이션
# 각 호출은 응답이 준비될 때까지 블로킹됩니다
def application(environ: dict, start_response):
    """WSGI는 요청 환경(environ)과 응답 시작을 위한 콜백을
    받는 callable을 기대합니다."""
    status = "200 OK"
    headers = [("Content-Type", "text/plain")]
    start_response(status, headers)
    return [b"Hello, WSGI!"]
```

**WSGI 서버**: Gunicorn, uWSGI, mod_wsgi
**WSGI 프레임워크**: Flask, Django (전통 모드)

### ASGI (Asynchronous Server Gateway Interface)

ASGI는 async/await, WebSocket, 장기 연결을 지원하기 위해 도입되었습니다. 워커 하나가 많은 동시 연결을 처리할 수 있습니다.

```python
# 최소한의 ASGI 애플리케이션
# async/await 사용 -- 수천 개의 동시 연결 처리 가능
async def application(scope: dict, receive, send):
    """ASGI는 세 가지 인자를 받는 callable을 사용합니다:
    scope = 연결 메타데이터, receive = 수신 메시지,
    send = 송신 메시지."""
    if scope["type"] == "http":
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        })
        await send({
            "type": "http.response.body",
            "body": b"Hello, ASGI!",
        })
```

**ASGI 서버**: Uvicorn, Hypercorn, Daphne
**ASGI 프레임워크**: FastAPI, Starlette, Django (ASGI 모드)

### 비교

```
WSGI (동기)                           ASGI (비동기)
┌──────────────────┐                  ┌──────────────────┐
│  워커 프로세스    │                  │  이벤트 루프       │
│                   │                  │                   │
│  요청 1 ████      │ (블로킹)          │  요청 1 ██  ██    │ (논블로킹)
│  요청 2   ████    │ (대기)           │  요청 2  ██  ██   │ (인터리빙)
│  요청 3     ████  │                  │  요청 3 ██  ██    │
└──────────────────┘                  │  요청 4  ██  ██   │
                                      └──────────────────┘
N개의 동시 요청을 위해 N개의           워커 하나가 많은 연결을
워커가 필요. CPU 집약적 작업에 적합.   처리. I/O 집약적 작업에 적합.
```

### 각각의 사용 시기

| 시나리오 | 권장 사항 |
|----------|---------------|
| 단순 CRUD API | 둘 다 가능; ASGI가 미래 지향적 |
| 고동시성 I/O (DB, 외부 API) | ASGI |
| WebSocket, SSE, 롱 폴링 | ASGI (WSGI는 이를 지원할 수 없음) |
| CPU 집약적 연산 | 다중 워커 WSGI, 또는 스레드 풀을 사용하는 ASGI |
| 레거시 애플리케이션 | Flask/Django로 이미 구축된 경우 WSGI |

---

## 4. API의 공용어로서의 JSON

**JSON**(JavaScript Object Notation)은 웹 API의 지배적인 데이터 형식입니다. 사람이 읽을 수 있고, 언어 독립적이며, JavaScript에서 기본으로 지원됩니다.

### JSON 데이터 타입

```json
{
    "string": "hello",
    "integer": 42,
    "float": 3.14,
    "boolean": true,
    "null_value": null,
    "array": [1, 2, 3],
    "nested_object": {
        "key": "value"
    }
}
```

### Python JSON 처리

```python
import json
from datetime import datetime, date
from decimal import Decimal

# Python의 json 모듈은 기본 타입을 처리하지만, datetime과 Decimal은
# JSON 기본형이 아니므로 커스텀 직렬화가 필요합니다
class APIEncoder(json.JSONEncoder):
    """json.dumps()가 기본으로 처리하지 못하는 타입을 위한 커스텀 인코더."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()  # ISO 8601이 API 날짜의 표준
        if isinstance(obj, Decimal):
            return float(obj)  # JSON 호환성을 위해 정밀도를 타협
        return super().default(obj)

data = {"created_at": datetime.now(), "price": Decimal("19.99")}
json_string = json.dumps(data, cls=APIEncoder, indent=2)
print(json_string)
# {"created_at": "2025-01-15T14:30:00", "price": 19.99}
```

### 콘텐츠 협상(Content Negotiation)

클라이언트는 `Accept` 헤더를 통해 서버에 원하는 형식을 알립니다. 서버는 선택한 형식을 `Content-Type`으로 응답합니다:

```
# 클라이언트가 JSON 요청
GET /api/users/42 HTTP/1.1
Accept: application/json

# 서버가 JSON으로 응답
HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8
```

### JSON의 대안

| 형식 | 장점 | 단점 | 사용 사례 |
|--------|------|------|----------|
| JSON | 범용, 사람이 읽을 수 있음 | 장황함, 스키마 없음 | 일반 API |
| MessagePack | 압축된 바이너리 JSON | 사람이 읽을 수 없음 | 고처리량 내부 통신 |
| Protocol Buffers | 강타입, 빠름 | `.proto` 파일 필요 | gRPC 마이크로서비스 |
| XML | 자기 서술적, 스키마 | 매우 장황함 | 레거시 SOAP, 설정 파일 |

---

## 5. API 버전 관리 전략

API는 진화합니다. 파괴적인 변경(Breaking changes)은 불가피합니다. 버전 관리를 통해 기존 클라이언트를 손상시키지 않고 변경 사항을 도입할 수 있습니다.

### 전략 1: URL 경로 버전 관리

```
GET /api/v1/users/42
GET /api/v2/users/42
```

**장점**: 명시적, 이해하기 쉬움, 라우팅 용이
**단점**: URL 오염, 폐기(deprecation) 어려움
**사용 예**: GitHub, Stripe, Twitter

### 전략 2: 헤더 버전 관리

```
GET /api/users/42
Accept: application/vnd.myapi.v2+json
```

**장점**: 깔끔한 URL, HTTP 시맨틱스 준수
**단점**: 검색 어려움, 브라우저에서 테스트 불편
**사용 예**: GitHub (이 방식도 지원)

### 전략 3: 쿼리 파라미터 버전 관리

```
GET /api/users/42?version=2
```

**장점**: 추가 용이, 선택적 (기본값은 최신 버전)
**단점**: 잊기 쉬움, 쿼리 문자열 오염
**사용 예**: Google APIs, Amazon

### 실용적인 권장 사항

대부분의 프로젝트에서 **URL 경로 버전 관리** (`/api/v1/`)가 최선의 기본 선택입니다. 명시적이고 단순하며 모든 HTTP 클라이언트와 캐싱 레이어에서 작동합니다. 파괴적인 변경이 있을 때만 새 버전을 도입하세요.

```python
# FastAPI 예제: 라우터를 사용한 버전 구성
from fastapi import APIRouter, FastAPI

app = FastAPI()

# 각 버전은 독립적인 로직을 가진 자체 라우터를 가집니다
v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

@v1_router.get("/users/{user_id}")
async def get_user_v1(user_id: int):
    return {"id": user_id, "name": "Alice"}  # v1: 플랫 응답

@v2_router.get("/users/{user_id}")
async def get_user_v2(user_id: int):
    return {"data": {"id": user_id, "name": "Alice"}, "meta": {"version": 2}}

app.include_router(v1_router)
app.include_router(v2_router)
```

---

## 6. 웹 프레임워크의 요청 생명주기

요청이 도착하면 비즈니스 로직에 도달하기 전에 여러 컴포넌트로 구성된 파이프라인을 통과합니다.

### 파이프라인

```
클라이언트 요청
     │
     ▼
┌─────────────────┐
│  웹 서버         │  Uvicorn / Gunicorn이 원시 HTTP 수신
│  (ASGI/WSGI)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  미들웨어         │  핸들러 전에 실행
│  스택            │  - CORS 헤더
│                   │  - 인증 검사
│                   │  - 요청 로깅
│                   │  - 속도 제한
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  라우터           │  URL 패턴을 핸들러 함수에 매핑
│                   │  /api/users/{id} → get_user(id)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  의존성 주입      │  핸들러가 선언한 의존성 해결
│  (Dependency     │  - 데이터베이스 세션
│   Injection)     │  - 현재 인증된 사용자
│                   │  - 설정 값
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  요청 검증        │  파싱 및 검증:
│  (Validation)    │  - 경로 파라미터 (타입 변환)
│                   │  - 쿼리 파라미터
│                   │  - 요청 본문 (JSON → Pydantic 모델)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  핸들러           │  비즈니스 로직이 실행되는 곳
│  (뷰/엔드포인트)  │  - 데이터베이스 쿼리
│                   │  - 데이터 처리
│                   │  - 응답 객체 반환
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  응답 직렬화      │  반환값을 JSON으로 직렬화
│  (Serialization) │  응답 모델 필터링 적용
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  미들웨어         │  핸들러 후에 실행
│  (후처리)        │  - 응답 헤더 추가
│                   │  - 응답 본문 압축
│                   │  - 응답 상태 + 타이밍 로그
└────────┬────────┘
         │
         ▼
    HTTP 응답
```

### 미들웨어 예제

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """각 요청이 얼마나 걸리는지 측정하여 헤더로 추가합니다.
    모든 엔드포인트를 수정하지 않고 성능 모니터링에 유용합니다."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    # X- 접두사는 커스텀 헤더의 관례입니다
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response
```

### 파이프라인의 오류 처리

어느 단계에서든 예외가 발생하면 프레임워크가 이를 잡아 HTTP 오류 응답으로 변환합니다:

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    user = await find_user(user_id)
    if user is None:
        # HTTPException은 파이프라인을 단락시키고
        # 추가 처리 없이 즉시 오류 응답을 반환합니다
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found"
        )
    return user
```

---

## 7. 연습 문제

### 문제 1: REST API 설계

**블로그** 애플리케이션을 위한 URL 구조와 HTTP 메서드를 설계하세요. 다음 리소스를 포함해야 합니다: 게시물(posts), 댓글(comments), 태그(tags). 다음 사항을 포함하세요:
- 게시물에 대한 CRUD 연산
- 게시물 아래 중첩된 댓글
- 게시물에 태그 할당
- 태그로 게시물 필터링
- 게시물 목록 페이지네이션

각 엔드포인트를 `메서드 /경로 -> 상태코드` 형식으로 작성하세요.

### 문제 2: 상태 코드 선택

각 시나리오에 대해 가장 적절한 HTTP 상태 코드를 선택하고 이유를 설명하세요:

1. 사용자가 잘못된 이메일 형식으로 회원가입 폼을 제출한 경우
2. 서버의 데이터베이스 연결 풀이 고갈된 경우
3. 사용자가 `DELETE /api/users/99`를 요청했지만 사용자 99가 존재하지 않는 경우
4. 사용자가 리소스를 요청했지만 JWT 토큰이 만료된 경우
5. `POST` 요청은 성공했지만 생성된 리소스는 나중에 사용 가능할 경우 (비동기 처리)

### 문제 3: WSGI vs ASGI 분석

다음 기능을 가진 채팅 애플리케이션을 구축하고 있다고 가정합니다:
- 사용자 프로필과 채팅 기록을 위한 REST API 제공
- 실시간 메시징을 위한 WebSocket 연결 유지
- 다국어 지원을 위한 외부 번역 API 호출

어떤 서버 모델(WSGI 또는 ASGI)을 선택하겠습니까? 위의 세 가지 요구사항 각각을 다루어 선택 이유를 정당화하세요.

### 문제 4: 미들웨어 파이프라인

다음을 수행하는 Python 함수(의사 미들웨어)를 작성하세요:
1. `X-API-Key` 헤더 확인
2. 키가 없으면 `401 Unauthorized` 반환
3. 키가 유효하지 않으면 (미리 정의된 집합에 없으면) `403 Forbidden` 반환
4. 요청 메서드, 경로, 응답 상태 코드 로깅
5. 모든 응답에 `X-Request-Id` 헤더 (UUID) 추가

### 문제 5: JSON 직렬화 엣지 케이스

다음 Python 데이터가 주어졌을 때, 모든 타입을 올바르게 처리하는 커스텀 JSON 인코더를 작성하세요:
```python
data = {
    "id": uuid.UUID("12345678-1234-5678-1234-567812345678"),
    "amount": Decimal("99.95"),
    "created_at": datetime(2025, 6, 15, 10, 30),
    "tags": frozenset({"python", "api"}),
    "metadata": None
}
```

---

## 8. 참고 자료

- [MDN HTTP 레퍼런스](https://developer.mozilla.org/en-US/docs/Web/HTTP)
- [RFC 7231 - HTTP/1.1 시맨틱스와 콘텐츠](https://datatracker.ietf.org/doc/html/rfc7231)
- [Roy Fielding의 REST 논문](https://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm)
- [PEP 3333 - Python 웹 서버 게이트웨이 인터페이스](https://peps.python.org/pep-3333/)
- [ASGI 명세](https://asgi.readthedocs.io/en/latest/specs/main.html)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [JSON 명세 (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)

---

**이전**: [개요](./00_Overview.md) | **다음**: [FastAPI 기초](./02_FastAPI_Basics.md)
