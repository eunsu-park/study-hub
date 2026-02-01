# 시스템 디자인 학습 가이드

## 소개

이 폴더는 시스템 디자인(System Design)을 체계적으로 학습하기 위한 자료를 담고 있습니다. 대규모 시스템을 설계하는 데 필요한 핵심 개념부터 실전 패턴까지 단계별로 학습할 수 있습니다. 기술 면접 준비와 실무 아키텍처 설계 역량 향상에 도움이 됩니다.

**대상 독자**: 백엔드 개발자, 시스템 아키텍트, 기술 면접 준비자

---

## 학습 로드맵

```
[기초]                    [중급]                    [고급]
  │                         │                         │
  ▼                         ▼                         ▼
시스템 설계 개요 ────▶ 로드 밸런싱 ─────▶ 분산 캐시 시스템
  │                         │                         │
  ▼                         ▼                         ▼
확장성 기초 ─────────▶ 리버스 프록시 ───▶ 데이터베이스 확장
  │                         │                         │
  ▼                         ▼                         ▼
네트워크 기초 복습 ──▶ 캐싱 전략 ───────▶ 데이터베이스 복제
  │                         │                         │
  ▼                         ▼                         ▼
                      API 게이트웨이 ───▶ 메시지 큐
                                             │
                                             ▼
                                        마이크로서비스
                                             │
                                             ▼
                                        분산 시스템
                                             │
                                             ▼
                                        실전 설계
```

---

## 선수 지식

- **필수**
  - 네트워크 기초 (HTTP, DNS, TCP/IP) → [Networking/](../Networking/00_Overview.md)
  - 데이터베이스 기초 (SQL, 트랜잭션) → [PostgreSQL/](../PostgreSQL/00_Overview.md)
  - 프로그래밍 언어 1개 이상

- **권장**
  - Linux 기본 명령어 → [Linux/](../Linux/00_Overview.md)
  - Docker 기초 → [Docker/](../Docker/00_Overview.md)
  - REST API 개념 → [Web_Development/](../Web_Development/00_Overview.md)

---

## 파일 목록

### 기초 (01-03)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_System_Design_Overview.md](./01_System_Design_Overview.md) | ⭐ | 시스템 설계란, 면접 평가 기준, 문제 접근 프레임워크 |
| [02_Scalability_Basics.md](./02_Scalability_Basics.md) | ⭐⭐ | 수직/수평 확장, CAP 정리, PACELC |
| [03_Network_Fundamentals_Review.md](./03_Network_Fundamentals_Review.md) | ⭐⭐ | DNS, CDN, HTTP/2/3, REST vs gRPC |

### 로드밸런싱과 프록시 (04-05)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [04_Load_Balancing.md](./04_Load_Balancing.md) | ⭐⭐⭐ | L4/L7 로드 밸런서, 분배 알고리즘, 헬스 체크 |
| [05_Reverse_Proxy_API_Gateway.md](./05_Reverse_Proxy_API_Gateway.md) | ⭐⭐⭐ | 리버스 프록시, API Gateway, Rate Limiting |

### 캐싱 (06-07)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [06_Caching_Strategies.md](./06_Caching_Strategies.md) | ⭐⭐⭐ | Cache-Aside, Write-Through, 캐시 무효화 |
| [07_Distributed_Cache_Systems.md](./07_Distributed_Cache_Systems.md) | ⭐⭐⭐ | Redis, Memcached, 일관성 해싱 |

### 데이터베이스 확장 (08-10)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [08_Database_Scaling.md](./08_Database_Scaling.md) | ⭐⭐⭐ | 파티셔닝, 샤딩 전략, 리밸런싱 |
| [09_Database_Replication.md](./09_Database_Replication.md) | ⭐⭐⭐ | 리더 복제, Quorum, 장애 복구 |
| [10_Data_Consistency_Patterns.md](./10_Data_Consistency_Patterns.md) | ⭐⭐⭐ | 일관성 모델, 최종 일관성, 강한 일관성 |

### 메시지 큐 (11-12)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [11_Message_Queue_Basics.md](./11_Message_Queue_Basics.md) | ⭐⭐⭐ | 비동기 처리, Kafka, RabbitMQ |
| [12_Message_System_Comparison.md](./12_Message_System_Comparison.md) | ⭐⭐⭐⭐ | Kafka vs RabbitMQ, 사용 사례 |

### 마이크로서비스 (13-14)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [13_Microservices_Basics.md](./13_Microservices_Basics.md) | ⭐⭐⭐⭐ | 모놀리스 vs MSA, 서비스 분리 |
| [14_Microservices_Patterns.md](./14_Microservices_Patterns.md) | ⭐⭐⭐⭐ | 서비스 메시, Circuit Breaker, Saga |

### 분산 시스템 (15-16)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [15_Distributed_Systems_Concepts.md](./15_Distributed_Systems_Concepts.md) | ⭐⭐⭐⭐ | 분산 시스템 특성, 장애 모델 |
| [16_Consensus_Algorithms.md](./16_Consensus_Algorithms.md) | ⭐⭐⭐⭐⭐ | Raft, Paxos, 리더 선출 |

### 실전 설계 (17-18)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [17_Design_Example_1.md](./17_Design_Example_1.md) | ⭐⭐⭐ | URL 단축기, 페이스트빈 설계 |
| [18_Design_Example_2.md](./18_Design_Example_2.md) | ⭐⭐⭐⭐ | 채팅 시스템, 알림 시스템 설계 |

---

## 추천 학습 순서

### 1단계: 기초 다지기 (1주)
```
01_System_Design_Overview → 02_Scalability_Basics → 03_Network_Fundamentals_Review
```
시스템 설계의 핵심 개념과 면접 접근법을 익힙니다.

### 2단계: 트래픽 처리 (1주)
```
04_Load_Balancing → 05_Reverse_Proxy_API_Gateway
```
트래픽 분산과 API 관리 방법을 학습합니다.

### 3단계: 캐싱 마스터 (1주)
```
06_Caching_Strategies → 07_Distributed_Cache_Systems
```
성능 최적화의 핵심인 캐싱을 깊이 있게 다룹니다.

### 4단계: 데이터베이스 확장 (1~2주)
```
08_Database_Scaling → 09_Database_Replication → 10_Data_Consistency_Patterns
```
대용량 데이터 처리를 위한 DB 확장 전략을 익힙니다.

### 5단계: 메시지 큐와 마이크로서비스 (2주)
```
11_Message_Queue_Basics → 12_Message_System_Comparison → 13_Microservices_Basics → 14_Microservices_Patterns
```
비동기 처리와 분산 아키텍처 패턴을 학습합니다.

### 6단계: 분산 시스템과 실전 설계 (2~3주)
```
15_Distributed_Systems_Concepts → 16_Consensus_Algorithms → 17_Design_Example_1 → 18_Design_Example_2
```
분산 시스템 이론과 실전 설계 문제를 다룹니다.

---

## 면접 대비 팁

### 시스템 설계 면접 접근법

```
┌─────────────────────────────────────────────────────────────────┐
│                    시스템 설계 면접 4단계                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 요구사항 명확화 (5분)                                        │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • 기능 요구사항 확인                                    │  │
│     │ • 비기능 요구사항 (성능, 가용성, 확장성)                │  │
│     │ • 규모 추정 (사용자 수, 트래픽)                         │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  2. Back-of-the-envelope 계산 (5분)                             │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • QPS (초당 쿼리 수)                                    │  │
│     │ • 저장 용량                                             │  │
│     │ • 대역폭                                                │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  3. 고수준 설계 (15-20분)                                        │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • 주요 컴포넌트 다이어그램                              │  │
│     │ • 데이터 흐름                                           │  │
│     │ • API 설계                                              │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  4. 상세 설계 (15-20분)                                          │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • 데이터베이스 스키마                                   │  │
│     │ • 확장 전략                                             │  │
│     │ • 트레이드오프 논의                                     │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 자주 출제되는 주제

| 주제 | 핵심 개념 | 관련 파일 |
|------|----------|----------|
| URL 단축기 | 해시 함수, Base62 | 08, 07 |
| 채팅 시스템 | WebSocket, 메시지 큐 | 03, 11 |
| 뉴스피드 | 팬아웃, 캐싱 | 06, 07 |
| 검색 엔진 | 역색인, 샤딩 | 08 |
| 알림 시스템 | 메시지 큐, 우선순위 | 11, 12 |
| 파일 저장소 | 분산 스토리지, 청크 | 08, 09 |

### 면접 체크리스트

- [ ] 요구사항을 명확히 했는가?
- [ ] 규모를 추정했는가?
- [ ] 고수준 아키텍처를 그렸는가?
- [ ] 데이터 모델을 설계했는가?
- [ ] 병목 지점을 식별했는가?
- [ ] 확장 전략을 제시했는가?
- [ ] 트레이드오프를 논의했는가?

---

## 관련 자료

### 다른 폴더와의 연계

| 폴더 | 관련 내용 |
|------|----------|
| [Networking/](../Networking/00_Overview.md) | DNS, HTTP, TCP/IP, 네트워크 보안 |
| [PostgreSQL/](../PostgreSQL/00_Overview.md) | 데이터베이스, 트랜잭션, 복제 |
| [Docker/](../Docker/00_Overview.md) | 컨테이너화, 마이크로서비스 배포 |
| [Linux/](../Linux/00_Overview.md) | 서버 관리, 성능 모니터링 |

### 추천 도서

- **Designing Data-Intensive Applications** - Martin Kleppmann
- **System Design Interview** - Alex Xu
- **Building Microservices** - Sam Newman
- **Web Scalability for Startup Engineers** - Artur Ejsmont

### 온라인 자료

- [System Design Primer (GitHub)](https://github.com/donnemartin/system-design-primer)
- [Grokking System Design (Educative)](https://www.educative.io/courses/grokking-the-system-design-interview)
- [High Scalability Blog](http://highscalability.com/)
- [ByteByteGo Blog](https://bytebytego.com/)

---

## 학습 팁

1. **그림 그리기**: 시스템 아키텍처를 직접 그려보세요
2. **숫자 감각**: QPS, 저장 용량 계산에 익숙해지세요
3. **트레이드오프**: 모든 결정에는 장단점이 있습니다
4. **실제 사례**: 대형 서비스의 아키텍처를 분석해보세요
5. **면접 연습**: 소리 내어 설명하는 연습을 하세요

---

**문서 정보**
- 최종 수정: 2026년 1월
- 전체 학습 시간: 약 6-8주
