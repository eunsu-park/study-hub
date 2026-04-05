# 사고 대응 및 포렌식

**이전**: [보안 테스팅](./13_Security_Testing.md) | **다음**: [프로젝트: 보안 API](./15_Project_Secure_API.md)

---

보안 사고는 불가피합니다. 회복력 있는 조직과 취약한 조직을 구분하는 것은 침해를 당하느냐 여부가 아니라, 얼마나 효과적으로 대응하느냐입니다. 이 레슨은 NIST 프레임워크를 기반으로 한 완전한 사고 대응 생명주기, 디지털 포렌식 기초, 로그 분석 기법, 침해 지표(IOC) 탐지를 위한 실용적인 Python 스크립트를 다룹니다. 이 레슨을 마치면 사고 대응 플레이북을 작성하고 보안 이벤트를 체계적으로 분석할 수 있습니다.

## 학습 목표

- NIST 사고 대응 생명주기 단계 이해
- 사고 대응 계획 수립 및 유지
- 침해 지표를 찾기 위한 로그 분석
- 디지털 포렌식 기초 및 증거 보관 연속성 이해
- 로그 파싱 및 IOC 탐지를 위한 Python 스크립트 작성
- 일반적인 시나리오에 대한 사고 대응 플레이북 작성
- 효과적인 사고 후 검토 수행

---

## 1. 사고 대응 개요

### 1.1 보안 사고란 무엇인가?

보안 사고는 정보나 시스템의 기밀성, 무결성, 가용성을 손상시키는 모든 이벤트입니다. 모든 보안 이벤트가 사고인 것은 아니며, 트리아지를 통해 심각도를 판단합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              보안 이벤트 → 사고 분류                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  보안 이벤트 (수백만 건/일)                                      │
│  ├── 방화벽 차단                                                 │
│  ├── 로그인 실패 시도                                            │
│  ├── 포트 스캔                                                   │
│  ├── 악성코드 탐지 (격리됨)                                      │
│  └── IDS/IPS 경고                                                │
│       │                                                          │
│       ▼  트리아지 및 상관 분석                                   │
│                                                                  │
│  보안 사고 (몇 건/월)                                            │
│  ├── 성공적인 무단 접근                                          │
│  ├── 데이터 침해 / 유출                                          │
│  ├── 악성코드 감염 (활성)                                        │
│  ├── 서비스 거부 공격                                            │
│  ├── 내부자 위협 활동                                            │
│  └── 랜섬웨어 배포                                               │
│       │                                                          │
│       ▼  심각도 분류                                             │
│                                                                  │
│  ┌───────────┬──────────────────────────────────────────────┐   │
│  │ 심각도    │ 설명                                         │   │
│  ├───────────┼──────────────────────────────────────────────┤   │
│  │ Critical  │ 활성 데이터 침해, 랜섬웨어, 시스템 전체      │   │
│  │ (P1)      │ 침해. 즉시 대응 필요.                        │   │
│  ├───────────┼──────────────────────────────────────────────┤   │
│  │ High      │ 확인된 침입, 단일 시스템 침해,               │   │
│  │ (P2)      │ 활성 악성코드. 몇 시간 내.                   │   │
│  ├───────────┼──────────────────────────────────────────────┤   │
│  │ Medium    │ 의심스러운 활동, 정책 위반,                  │   │
│  │ (P3)      │ 취약점 적극 악용. 며칠 내.                   │   │
│  ├───────────┼──────────────────────────────────────────────┤   │
│  │ Low       │ 사소한 정책 위반, 실패한 공격 시도,          │   │
│  │ (P4)      │ 정보성. 몇 주 내.                            │   │
│  └───────────┴──────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 일반적인 사고 유형

| 사고 유형 | 예시 | 일반적인 지표 |
|---|---|---|
| **악성코드** | 랜섬웨어, 트로이목마, 웜 | 비정상적인 프로세스, 파일 암호화, C2 트래픽 |
| **무단 접근** | 자격 증명 침해, 무차별 대입 | 로그인 실패, 비정상 시간 접근, 비정상 지리적 위치 |
| **데이터 침해** | 유출, 우발적 노출 | 대량 데이터 전송, DB 덤프, 비정상 쿼리 |
| **DoS/DDoS** | 볼륨형, 애플리케이션 레이어 | 트래픽 급증, 서비스 저하, CPU 소진 |
| **내부자 위협** | 데이터 절도, 사보타주 | 과도한 접근, 대용량 다운로드, 정책 위반 |
| **웹 애플리케이션** | SQLi, XSS, RCE | 의심스러운 요청 패턴, WAF 경고, 오류 급증 |
| **공급망** | 침해된 종속성, 업데이트 | 예상치 못한 패키지 변경, 의심스러운 빌드 아티팩트 |
| **피싱** | 자격 증명 수집, BEC | 신고된 이메일, 비정상 로그인 위치, 송금 요청 |

---

## 2. NIST 사고 대응 생명주기

### 2.1 4단계

NIST 컴퓨터 보안 사고 처리 가이드(SP 800-61)는 4가지 주요 단계를 정의합니다. 실제로 이러한 단계는 중첩되고 순환합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              NIST 사고 대응 생명주기                             │
│              (SP 800-61 Rev. 2)                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────┐                                           │
│   │  1. 준비         │◄──────────────────────────────┐          │
│   │                  │                               │          │
│   │  - IR 계획       │                               │          │
│   │  - 팀 교육       │                               │          │
│   │  - 도구 준비     │                               │          │
│   │  - 커뮤니케이션  │                               │          │
│   └────────┬─────────┘                               │          │
│            │                                          │          │
│            ▼                                          │          │
│   ┌─────────────────────────┐                        │          │
│   │  2. 탐지 및              │                        │          │
│   │     분석                 │                        │          │
│   │                          │                        │          │
│   │  - 경고 모니터링         │                        │          │
│   │  - 이벤트 트리아지       │                        │          │
│   │  - 범위 결정             │                        │          │
│   │  - 심각도 분류           │                        │          │
│   └────────┬────────────────┘                        │          │
│            │                                          │          │
│            ▼                                          │          │
│   ┌─────────────────────────┐                        │          │
│   │  3. 격리,                │    ◄── 이 하위 단계들 │          │
│   │     제거 및              │        간에 순환 가능  │          │
│   │     복구                 │                        │          │
│   │                          │                        │          │
│   │  - 영향받은 시스템 격리  │                        │          │
│   │  - 위협 제거             │                        │          │
│   │  - 시스템 복구           │                        │          │
│   │  - 복구 검증             │                        │          │
│   └────────┬────────────────┘                        │          │
│            │                                          │          │
│            ▼                                          │          │
│   ┌─────────────────────────┐                        │          │
│   │  4. 사고 후              │────────────────────────┘          │
│   │     활동                 │    교훈이 준비 단계로             │
│   │                          │    피드백됨                       │
│   │  - 교훈 학습             │                                    │
│   │  - 보고서 작성           │                                    │
│   │  - 프로세스 개선         │                                    │
│   └─────────────────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 1단계: 준비

준비는 가장 중요한 단계입니다. 이것이 없으면 나머지는 모두 압박 속에서의 즉흥적인 대응이 됩니다.

```
┌──────────────────────────────────────────────────────────────────┐
│                    준비 체크리스트                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  인력:                                                            │
│  [x] 명확한 역할을 가진 IR 팀 식별                               │
│  [x] 연락처 목록 (팀, 경영진, 법무, PR, 벤더)                    │
│  [x] 당직 근무 일정                                               │
│  [x] 정기적인 교육 및 탁상 연습                                   │
│  [x] 외부 IR 리테이너 (선택사항)                                  │
│                                                                   │
│  프로세스:                                                        │
│  [x] 경영진이 승인한 서면 IR 계획                                │
│  [x] 일반적인 사고 유형에 대한 플레이북                          │
│  [x] 에스컬레이션 절차                                            │
│  [x] 커뮤니케이션 템플릿 (내부, 외부, 법무)                      │
│  [x] 증거 처리 절차                                               │
│  [x] 규제 알림 요구사항 문서화                                    │
│                                                                   │
│  기술:                                                            │
│  [x] 로깅 인프라 (중앙화됨, 보관됨)                              │
│  [x] SIEM 또는 로그 분석 도구                                    │
│  [x] 포렌식 워크스테이션 / 툴킷                                  │
│  [x] 네트워크 모니터링 / IDS                                     │
│  [x] 엔드포인트 탐지 및 대응 (EDR)                               │
│  [x] 백업 및 복구 시스템 테스트 완료                             │
│  [x] 클린 OS 이미지 / 골든 이미지                                │
│  [x] 점프 백: 휴대용 포렌식 도구, 케이블, 스토리지               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

#### IR 팀 역할

```
┌──────────────────────────────────────────────────────────────────┐
│                   사고 대응 팀 역할                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────┐                                         │
│  │   사고 관리자        │  전체 조정, 의사결정,                   │
│  │   (팀 리더)          │  경영진과의 커뮤니케이션                │
│  └─────────┬───────────┘                                         │
│            │                                                      │
│  ┌─────────┴────────────────────────────────┐                    │
│  │         │              │                  │                    │
│  ▼         ▼              ▼                  ▼                    │
│ ┌────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐            │
│ │보안     │ │  포렌식     │ │  시스템  │ │ 커뮤니케 │            │
│ │분석가   │ │  분석가     │ │  관리자  │ │ 이션/법무│            │
│ │        │ │            │ │          │ │          │            │
│ │모니터링│ │증거        │ │격리 및   │ │알림 및   │            │
│ │트리아지│ │수집        │ │복구      │ │문서화    │            │
│ │분석    │ │분석        │ │패치      │ │규제 대응 │            │
│ └────────┘ └────────────┘ └──────────┘ └──────────┘            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 2단계: 탐지 및 분석

#### 탐지 소스

```
┌──────────────────────────────────────────────────────────────────┐
│                    탐지 소스                                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  자동화:                                                          │
│  ├── SIEM 경고 (상관된 로그 이벤트)                              │
│  ├── IDS/IPS (Snort, Suricata)                                   │
│  ├── EDR 경고 (CrowdStrike, Carbon Black 등)                     │
│  ├── WAF 경고 (ModSecurity, AWS WAF)                             │
│  ├── 안티바이러스 / 안티멀웨어                                    │
│  ├── 파일 무결성 모니터링 (OSSEC, Tripwire)                      │
│  ├── 네트워크 트래픽 이상 (NetFlow, Zeek)                        │
│  └── 클라우드 보안 경고 (GuardDuty, Security Center)             │
│                                                                   │
│  사람:                                                            │
│  ├── 사용자 신고 ("뭔가 이상합니다")                             │
│  ├── 헬프데스크 티켓                                              │
│  ├── 외부 알림 (파트너, 벤더, 연구원)                            │
│  ├── 법 집행기관 알림                                             │
│  ├── 언론 보도                                                    │
│  └── 위협 인텔리전스 피드                                         │
│                                                                   │
│  능동적:                                                          │
│  ├── 위협 헌팅                                                    │
│  ├── 침투 테스트 결과                                             │
│  ├── 취약점 스캔                                                  │
│  └── 로그 검토 / 감사                                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

#### 초기 분석 질문

경고가 발생하면 다음 질문들에 체계적으로 답변하십시오:

```
1. 무엇이 일어났는가?
   - 어떤 시스템/데이터가 영향을 받았는가?
   - 이것은 어떤 유형의 사고인가?
   - 초기 증거는 무엇인가?

2. 언제 일어났는가?
   - 첫 번째 지표는 언제였는가?
   - 언제 탐지되었는가?
   - 현재 진행 중인가?

3. 어디에 영향이 있는가?
   - 어떤 호스트/네트워크?
   - 어떤 애플리케이션/서비스?
   - 어떤 데이터/사용자?

4. 누가 관련되어 있는가?
   - 소스 IP/계정?
   - 타겟팅된 사용자/시스템?
   - 내부 또는 외부 행위자?

5. 어떻게 일어났는가?
   - 공격 벡터 (피싱, 익스플로잇, 내부자 등)?
   - 악용된 취약점?
   - 사용된 도구/기법?

6. 얼마나 심각한가?
   - 범위: 몇 개의 시스템이 영향을 받았는가?
   - 영향: 데이터 손실, 서비스 중단?
   - 심각도 분류 (P1-P4)?
```

### 2.4 3단계: 격리, 제거 및 복구

```
┌──────────────────────────────────────────────────────────────────┐
│          격리 → 제거 → 복구                                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────┐                 │
│  │  격리 (출혈 멈추기)                          │                 │
│  │                                              │                 │
│  │  단기:                                       │                 │
│  │  ├── 영향받은 시스템 격리 (네트워크)        │                 │
│  │  ├── 악성 IP/도메인 차단                    │                 │
│  │  ├── 침해된 계정 비활성화                   │                 │
│  │  ├── 필요시 DNS 리디렉션                    │                 │
│  │  └── 변경 전 증거 보존                      │                 │
│  │                                              │                 │
│  │  장기:                                       │                 │
│  │  ├── 임시 패치/우회 방법 적용               │                 │
│  │  ├── 영향받은 영역의 모니터링 증가          │                 │
│  │  ├── 추가 접근 제어 구현                    │                 │
│  │  └── 적절한 경우 허니팟/카나리 설정         │                 │
│  └─────────────────────────────────────────────┘                 │
│            │                                                      │
│            ▼                                                      │
│  ┌─────────────────────────────────────────────┐                 │
│  │  제거 (위협 제거)                            │                 │
│  │                                              │                 │
│  │  ├── 악성코드 / 백도어 제거                 │                 │
│  │  ├── 악용된 취약점 패치                     │                 │
│  │  ├── 침해된 자격 증명 재설정                │                 │
│  │  ├── 필요시 영향받은 시스템 재구축          │                 │
│  │  ├── 방화벽/IDS 규칙 업데이트               │                 │
│  │  └── 지속성 메커니즘 스캔                   │                 │
│  └─────────────────────────────────────────────┘                 │
│            │                                                      │
│            ▼                                                      │
│  ┌─────────────────────────────────────────────┐                 │
│  │  복구 (정상으로 복귀)                        │                 │
│  │                                              │                 │
│  │  ├── 클린 백업에서 복원                     │                 │
│  │  ├── 골든 이미지에서 시스템 재구축          │                 │
│  │  ├── 서비스 점진적 복원                     │                 │
│  │  ├── 시스템 무결성 검증                     │                 │
│  │  ├── 재침해에 대한 면밀한 모니터링          │                 │
│  │  └── 사고 해결 선언                         │                 │
│  └─────────────────────────────────────────────┘                 │
│                                                                   │
│  중요: 모든 조치를 타임스탬프와 함께 문서화하십시오!             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.5 4단계: 사고 후 활동 (교훈 학습)

```
┌──────────────────────────────────────────────────────────────────┐
│              사고 후 검토 회의 안건                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  일정: 사고 종료 후 1-2주 이내                                    │
│  참석자: 모든 관련자 (비난 없는 환경)                             │
│                                                                   │
│  1. 타임라인 검토 (30분)                                          │
│     - 완전한 타임라인 살펴보기                                    │
│     - 첫 지표는 언제? 탐지는 언제? 해결은 언제?                  │
│                                                                   │
│  2. 잘된 점 (15분)                                                │
│     - 효과적인 탐지 메커니즘                                      │
│     - 신속한 격리 조치                                            │
│     - 좋은 커뮤니케이션                                           │
│                                                                   │
│  3. 개선할 수 있는 점 (30분)                                      │
│     - 탐지 공백                                                   │
│     - 대응 지연                                                   │
│     - 커뮤니케이션 단절                                           │
│     - 도구/프로세스 공백                                          │
│                                                                   │
│  4. 근본 원인 분석 (30분)                                         │
│     - 근본 원인은 무엇이었는가?                                   │
│     - 기존 통제가 실패한 이유는?                                  │
│     - "5 Whys" 기법 사용                                          │
│                                                                   │
│  5. 액션 아이템 (15분)                                            │
│     - 담당자와 마감일이 있는 구체적인 개선사항                    │
│     - 프로세스 변경                                               │
│     - 기술 변경                                                   │
│     - 교육 필요사항                                               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 로그 분석 및 SIEM 개념

### 3.1 로깅 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                  중앙 집중식 로깅 아키텍처                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  로그 소스                    수집            스토리지/분석       │
│  ┌──────────┐                                                     │
│  │ 웹       │──┐                                                  │
│  │ 서버     │  │    ┌───────────────┐    ┌───────────────────┐   │
│  └──────────┘  ├──► │  로그 전송     │──► │                   │   │
│  ┌──────────┐  │    │  (Filebeat,   │    │  SIEM / 로그      │   │
│  │ 앱       │──┤    │   Fluentd,    │    │  관리             │   │
│  │ 서버     │  │    │   rsyslog)    │    │                   │   │
│  └──────────┘  │    └───────────────┘    │  - Elasticsearch  │   │
│  ┌──────────┐  │                         │  - Splunk         │   │
│  │ 데이터   │──┤    ┌───────────────┐    │  - Graylog        │   │
│  │ 베이스   │  ├──► │  메시지       │──► │  - Wazuh          │   │
│  └──────────┘  │    │  큐           │    │  - QRadar         │   │
│  ┌──────────┐  │    │  (Kafka,      │    │                   │   │
│  │ 방화벽   │──┤    │   Redis)      │    │  기능:            │   │
│  │ / IDS    │  │    └───────────────┘    │  - 검색           │   │
│  └──────────┘  │                         │  - 상관 분석      │   │
│  ┌──────────┐  │                         │  - 경고           │   │
│  │ 엔드포   │──┘                         │  - 대시보드       │   │
│  │ 인트에이│                             │  - 보관           │   │
│  │ 전트     │                            └───────────────────┘   │
│  └──────────┘                                                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 중요한 로그 소스

| 로그 소스 | 캡처 대상 | 보안 가치 |
|---|---|---|
| **웹 서버** (nginx, Apache) | 접근 로그, 오류 로그 | 공격 탐지, 이상 탐지 |
| **애플리케이션** | 인증 이벤트, 오류, API 호출 | 비즈니스 로직 공격, 악용 |
| **데이터베이스** | 쿼리, 연결, 오류 | SQL 인젝션, 데이터 유출 |
| **OS / 시스템** | 인증 로그, 프로세스 실행, 파일 변경 | 권한 상승, 지속성 |
| **방화벽** | 허용/거부, 연결 | 네트워크 공격, 측면 이동 |
| **DNS** | 쿼리, 응답 | C2 통신, 데이터 유출 |
| **이메일** | 송수신, 첨부 파일 | 피싱, 데이터 유출 |
| **클라우드** | API 호출, 설정 변경 | 잘못된 설정, 무단 접근 |

### 3.3 SIEM 상관 규칙

```
┌──────────────────────────────────────────────────────────────────┐
│                   일반적인 SIEM 상관 규칙                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  규칙 1: 무차별 대입 탐지                                         │
│  ┌────────────────────────────────────────────────────┐          │
│  │ IF: 동일 IP에서 5분 내에 10회 이상 로그인 실패    │          │
│  │ THEN: "가능한 무차별 대입 공격" 경고               │          │
│  │ 심각도: Medium                                     │          │
│  │ 조치: IP 임시 차단, SOC 알림                       │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                   │
│  규칙 2: 불가능한 이동                                            │
│  ┌────────────────────────────────────────────────────┐          │
│  │ IF: 동일 사용자가 30분 내에 500마일 이상          │          │
│  │     떨어진 2개 지역에서 로그인                     │          │
│  │ THEN: "불가능한 이동 탐지" 경고                    │          │
│  │ 심각도: High                                       │          │
│  │ 조치: 재인증 강제, 사용자 알림                     │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                   │
│  규칙 3: 데이터 유출                                              │
│  ┌────────────────────────────────────────────────────┐          │
│  │ IF: 외부 IP로 100MB 이상 데이터 전송               │          │
│  │     평소 1MB/시간 미만 전송하는 서버에서           │          │
│  │ THEN: "가능한 데이터 유출" 경고                    │          │
│  │ 심각도: Critical                                   │          │
│  │ 조치: 전송 차단, 호스트 격리, IR 경고              │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                   │
│  규칙 4: 권한 상승                                                │
│  ┌────────────────────────────────────────────────────┐          │
│  │ IF: 사용자가 관리자/루트 그룹에 추가됨             │          │
│  │     AND 승인된 변경 시스템에서 온 것이 아님        │          │
│  │ THEN: "무단 권한 상승" 경고                        │          │
│  │ 심각도: Critical                                   │          │
│  │ 조치: 변경 되돌림, 계정 비활성화, IR 경고          │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                   │
│  규칙 5: 웹 애플리케이션 공격                                     │
│  ┌────────────────────────────────────────────────────┐          │
│  │ IF: 동일 IP에서 1분 내에 WAF 차단 5회 이상        │          │
│  │     AND 동일 앱에서 HTTP 500 오류 증가             │          │
│  │ THEN: "활성 웹 애플리케이션 공격" 경고             │          │
│  │ 심각도: High                                       │          │
│  │ 조치: IP 차단, 로깅 증가, IR 경고                  │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Python: 로그 파싱 및 분석

### 4.1 웹 서버 로그 파서

```python
"""
log_parser.py - 웹 서버 접근 로그를 파싱하고 분석합니다.
Apache/Nginx combined 로그 형식 지원.

예제 로그 라인:
192.168.1.100 - admin [15/Jan/2025:10:30:45 +0000] "GET /admin HTTP/1.1" 200 1234 "http://example.com" "Mozilla/5.0"
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator


# Combined log format regex
LOG_PATTERN = re.compile(
    r'(?P<ip>\S+)\s+'              # IP address
    r'\S+\s+'                       # ident (usually -)
    r'(?P<user>\S+)\s+'            # authenticated user
    r'\[(?P<time>[^\]]+)\]\s+'     # timestamp
    r'"(?P<method>\S+)\s+'         # HTTP method
    r'(?P<path>\S+)\s+'            # request path
    r'(?P<protocol>\S+)"\s+'       # HTTP version
    r'(?P<status>\d+)\s+'          # status code
    r'(?P<size>\S+)\s+'            # response size
    r'"(?P<referer>[^"]*)"\s+'     # referer
    r'"(?P<agent>[^"]*)"'          # user agent
)


@dataclass
class LogEntry:
    """파싱된 로그 엔트리."""
    ip: str
    user: str
    timestamp: datetime
    method: str
    path: str
    protocol: str
    status: int
    size: int
    referer: str
    user_agent: str
    raw: str = ""


@dataclass
class SecurityAlert:
    """로그 분석에서 생성된 보안 경고."""
    alert_type: str
    severity: str           # CRITICAL, HIGH, MEDIUM, LOW
    description: str
    source_ip: Optional[str] = None
    count: int = 0
    sample_entries: list = field(default_factory=list)
    timestamp: str = ""

    def __str__(self):
        return (f"[{self.severity}] {self.alert_type}: {self.description} "
                f"(IP: {self.source_ip}, count: {self.count})")


def parse_log_line(line: str) -> Optional[LogEntry]:
    """단일 로그 라인을 LogEntry로 파싱합니다."""
    match = LOG_PATTERN.match(line.strip())
    if not match:
        return None

    data = match.groupdict()

    # Parse timestamp
    try:
        ts = datetime.strptime(data['time'], '%d/%b/%Y:%H:%M:%S %z')
    except ValueError:
        ts = datetime.now()

    # Parse size (may be '-')
    try:
        size = int(data['size'])
    except ValueError:
        size = 0

    return LogEntry(
        ip=data['ip'],
        user=data['user'],
        timestamp=ts,
        method=data['method'],
        path=data['path'],
        protocol=data['protocol'],
        status=int(data['status']),
        size=size,
        referer=data['referer'],
        user_agent=data['agent'],
        raw=line.strip()
    )


def parse_log_file(filepath: str) -> Iterator[LogEntry]:
    """로그 파일의 모든 엔트리를 파싱합니다."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")

    with path.open('r', errors='ignore') as f:
        for line in f:
            entry = parse_log_line(line)
            if entry:
                yield entry


class LogAnalyzer:
    """파싱된 로그 엔트리에서 보안 지표를 분석합니다."""

    def __init__(self):
        self.entries: list[LogEntry] = []
        self.alerts: list[SecurityAlert] = []

    def load(self, filepath: str) -> int:
        """로그 파일을 로드하고 파싱합니다. 엔트리 개수를 반환합니다."""
        self.entries = list(parse_log_file(filepath))
        return len(self.entries)

    def analyze_all(self) -> list[SecurityAlert]:
        """모든 분석 규칙을 실행합니다."""
        self.alerts = []
        self.detect_brute_force()
        self.detect_directory_traversal()
        self.detect_sql_injection()
        self.detect_scanner_activity()
        self.detect_error_spikes()
        self.detect_suspicious_user_agents()
        self.detect_admin_access()
        return self.alerts

    def detect_brute_force(self, threshold: int = 10,
                            window_minutes: int = 5) -> None:
        """무차별 대입 로그인 시도를 탐지합니다."""
        # Group 401/403 responses by IP
        failed_logins = defaultdict(list)

        for entry in self.entries:
            if entry.status in (401, 403):
                failed_logins[entry.ip].append(entry)

        for ip, entries in failed_logins.items():
            if len(entries) >= threshold:
                # Check if they occur within the time window
                entries.sort(key=lambda e: e.timestamp)
                for i in range(len(entries) - threshold + 1):
                    window = entries[i:i + threshold]
                    time_diff = (window[-1].timestamp -
                                 window[0].timestamp).total_seconds()
                    if time_diff <= window_minutes * 60:
                        self.alerts.append(SecurityAlert(
                            alert_type="BRUTE_FORCE",
                            severity="HIGH",
                            description=(
                                f"{len(entries)} failed auth attempts from {ip} "
                                f"({threshold}+ within {window_minutes} min)"
                            ),
                            source_ip=ip,
                            count=len(entries),
                            sample_entries=[e.raw for e in entries[:3]],
                        ))
                        break  # One alert per IP

    def detect_directory_traversal(self) -> None:
        """경로 순회 시도를 탐지합니다."""
        traversal_patterns = [
            '../', '..\\', '%2e%2e', '%252e%252e',
            '/etc/passwd', '/etc/shadow', '/windows/system32',
            'boot.ini', 'web.config',
        ]

        traversal_attempts = defaultdict(list)

        for entry in self.entries:
            path_lower = entry.path.lower()
            for pattern in traversal_patterns:
                if pattern in path_lower:
                    traversal_attempts[entry.ip].append(entry)
                    break

        for ip, entries in traversal_attempts.items():
            self.alerts.append(SecurityAlert(
                alert_type="DIRECTORY_TRAVERSAL",
                severity="HIGH",
                description=(
                    f"Path traversal attempt from {ip}: "
                    f"{entries[0].path}"
                ),
                source_ip=ip,
                count=len(entries),
                sample_entries=[e.raw for e in entries[:3]],
            ))

    def detect_sql_injection(self) -> None:
        """요청 경로에서 SQL 인젝션 시도를 탐지합니다."""
        sqli_patterns = [
            "' or ", "' and ", "union select", "order by",
            "1=1", "' --", "'; drop", "sleep(",
            "benchmark(", "waitfor delay", "pg_sleep",
            "%27", "char(", "concat(",
        ]

        sqli_attempts = defaultdict(list)

        for entry in self.entries:
            path_lower = entry.path.lower()
            for pattern in sqli_patterns:
                if pattern in path_lower:
                    sqli_attempts[entry.ip].append(entry)
                    break

        for ip, entries in sqli_attempts.items():
            self.alerts.append(SecurityAlert(
                alert_type="SQL_INJECTION",
                severity="CRITICAL",
                description=(
                    f"SQL injection attempt from {ip}: "
                    f"{entries[0].path[:100]}"
                ),
                source_ip=ip,
                count=len(entries),
                sample_entries=[e.raw for e in entries[:3]],
            ))

    def detect_scanner_activity(self) -> None:
        """자동화된 취약점 스캐너 활동을 탐지합니다."""
        scanner_paths = [
            '/.env', '/wp-admin', '/wp-login.php',
            '/phpmyadmin', '/admin', '/administrator',
            '/.git/config', '/.svn/entries',
            '/robots.txt', '/sitemap.xml',
            '/backup', '/database', '/db',
            '/server-status', '/server-info',
            '/.htaccess', '/web.config',
            '/xmlrpc.php', '/api/v1',
        ]

        ip_scanner_hits = defaultdict(set)

        for entry in self.entries:
            for scanner_path in scanner_paths:
                if entry.path.lower().startswith(scanner_path):
                    ip_scanner_hits[entry.ip].add(entry.path)

        for ip, paths in ip_scanner_hits.items():
            if len(paths) >= 5:  # Hit 5+ scanner paths
                self.alerts.append(SecurityAlert(
                    alert_type="VULNERABILITY_SCANNER",
                    severity="MEDIUM",
                    description=(
                        f"Scanner activity from {ip}: "
                        f"probed {len(paths)} common paths"
                    ),
                    source_ip=ip,
                    count=len(paths),
                    sample_entries=list(paths)[:5],
                ))

    def detect_error_spikes(self, threshold: int = 50) -> None:
        """오류 응답의 비정상적인 급증을 탐지합니다."""
        # Count 5xx errors per IP
        error_counts = Counter()
        for entry in self.entries:
            if 500 <= entry.status < 600:
                error_counts[entry.ip] += 1

        for ip, count in error_counts.most_common():
            if count >= threshold:
                self.alerts.append(SecurityAlert(
                    alert_type="ERROR_SPIKE",
                    severity="MEDIUM",
                    description=(
                        f"High error rate from {ip}: "
                        f"{count} server errors (5xx)"
                    ),
                    source_ip=ip,
                    count=count,
                ))

    def detect_suspicious_user_agents(self) -> None:
        """의심스러운 user agent를 가진 요청을 탐지합니다."""
        suspicious_agents = [
            'sqlmap', 'nikto', 'nmap', 'masscan',
            'dirbuster', 'gobuster', 'wfuzz',
            'burpsuite', 'acunetix', 'nessus',
            'python-requests',  # May be legitimate, but flag
            'curl/',            # May be legitimate, but flag
        ]

        for entry in self.entries:
            agent_lower = entry.user_agent.lower()
            for sus_agent in suspicious_agents:
                if sus_agent in agent_lower:
                    self.alerts.append(SecurityAlert(
                        alert_type="SUSPICIOUS_USER_AGENT",
                        severity="MEDIUM",
                        description=(
                            f"Suspicious user agent from {entry.ip}: "
                            f"{entry.user_agent[:80]}"
                        ),
                        source_ip=entry.ip,
                        count=1,
                    ))
                    break  # One alert per entry

    def detect_admin_access(self) -> None:
        """관리 엔드포인트에 대한 접근을 탐지합니다."""
        admin_paths = ['/admin', '/dashboard', '/manage', '/api/admin']
        admin_access = defaultdict(list)

        for entry in self.entries:
            for admin_path in admin_paths:
                if entry.path.lower().startswith(admin_path):
                    if entry.status == 200:
                        admin_access[entry.ip].append(entry)

        for ip, entries in admin_access.items():
            self.alerts.append(SecurityAlert(
                alert_type="ADMIN_ACCESS",
                severity="LOW",
                description=(
                    f"Successful admin access from {ip}: "
                    f"{entries[0].path}"
                ),
                source_ip=ip,
                count=len(entries),
            ))

    def print_report(self) -> None:
        """형식화된 분석 보고서를 출력합니다."""
        print("=" * 65)
        print("  LOG ANALYSIS SECURITY REPORT")
        print("=" * 65)
        print(f"  Total entries analyzed: {len(self.entries)}")
        print(f"  Total alerts: {len(self.alerts)}")

        severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        for severity in severity_order:
            alerts = [a for a in self.alerts if a.severity == severity]
            if not alerts:
                continue

            print(f"\n{'─' * 65}")
            print(f"  {severity} ({len(alerts)})")
            print(f"{'─' * 65}")

            for alert in alerts:
                print(f"\n  [{alert.alert_type}]")
                print(f"  {alert.description}")
                if alert.sample_entries:
                    print(f"  Samples:")
                    for sample in alert.sample_entries[:2]:
                        print(f"    {str(sample)[:100]}")

        # Summary statistics
        print(f"\n{'─' * 65}")
        print("  TOP SOURCE IPs")
        print(f"{'─' * 65}")
        ip_alert_count = Counter()
        for alert in self.alerts:
            if alert.source_ip:
                ip_alert_count[alert.source_ip] += 1
        for ip, count in ip_alert_count.most_common(10):
            print(f"  {ip:20s} {count} alerts")

        print(f"\n{'=' * 65}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python log_parser.py <access.log>")
        sys.exit(1)

    analyzer = LogAnalyzer()
    count = analyzer.load(sys.argv[1])
    print(f"[*] Loaded {count} log entries")

    alerts = analyzer.analyze_all()
    analyzer.print_report()
```

### 4.2 IOC 탐지 스크립트

```python
"""
ioc_detector.py - 침해 지표(IOC) 탐지.
파일, 네트워크 연결, 시스템 상태에서 알려진 IOC를 확인합니다.
"""

import hashlib
import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class IOC:
    """침해 지표."""
    ioc_type: str       # IP, DOMAIN, HASH_MD5, HASH_SHA256, FILENAME, REGEX
    value: str
    description: str = ""
    source: str = ""    # Where this IOC came from
    severity: str = "MEDIUM"


@dataclass
class IOCMatch:
    """스캔 중 발견된 매칭."""
    ioc: IOC
    location: str       # Where the match was found
    context: str = ""   # Additional context
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class IOCDatabase:
    """
    간단한 IOC 데이터베이스.
    프로덕션에서는 STIX/TAXII 또는 적절한 위협 인텔리전스 플랫폼을 사용하십시오.
    """

    def __init__(self):
        self.iocs: list[IOC] = []

    def load_from_json(self, filepath: str) -> int:
        """JSON 파일에서 IOC를 로드합니다."""
        with open(filepath) as f:
            data = json.load(f)

        for item in data.get('iocs', []):
            self.iocs.append(IOC(
                ioc_type=item['type'],
                value=item['value'],
                description=item.get('description', ''),
                source=item.get('source', ''),
                severity=item.get('severity', 'MEDIUM'),
            ))

        return len(self.iocs)

    def load_sample_iocs(self) -> None:
        """데모용 샘플 IOC를 로드합니다."""
        # These are FAKE IOCs for educational purposes
        sample_iocs = [
            IOC("IP", "198.51.100.1", "Known C2 server", "sample", "HIGH"),
            IOC("IP", "203.0.113.50", "Phishing infrastructure", "sample", "HIGH"),
            IOC("DOMAIN", "evil-malware.example.com", "Malware C2", "sample", "CRITICAL"),
            IOC("DOMAIN", "phish.example.net", "Phishing domain", "sample", "HIGH"),
            IOC("HASH_SHA256",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "Known malware hash (empty file SHA256)", "sample", "MEDIUM"),
            IOC("FILENAME", "mimikatz.exe", "Credential dumping tool", "sample", "CRITICAL"),
            IOC("FILENAME", "nc.exe", "Netcat - possible backdoor", "sample", "HIGH"),
            IOC("REGEX", r"(?:eval|exec)\s*\(\s*base64", "Obfuscated code execution",
                "sample", "HIGH"),
        ]
        self.iocs.extend(sample_iocs)

    def get_by_type(self, ioc_type: str) -> list[IOC]:
        """특정 유형의 모든 IOC를 가져옵니다."""
        return [ioc for ioc in self.iocs if ioc.ioc_type == ioc_type]


class IOCScanner:
    """시스템에서 침해 지표를 스캔합니다."""

    def __init__(self, ioc_db: IOCDatabase):
        self.ioc_db = ioc_db
        self.matches: list[IOCMatch] = []

    def scan_all(self, scan_dir: Optional[str] = None) -> list[IOCMatch]:
        """모든 IOC 스캔을 실행합니다."""
        self.matches = []
        print("[*] Starting IOC scan...")

        self.scan_file_hashes(scan_dir or "/tmp")
        self.scan_file_names(scan_dir or "/tmp")
        self.scan_network_connections()
        self.scan_dns_cache()
        self.scan_file_contents(scan_dir or "/tmp")

        return self.matches

    def scan_file_hashes(self, directory: str) -> None:
        """알려진 악성코드 해시와 파일 해시를 비교합니다."""
        print(f"[*] Scanning file hashes in {directory}...")
        hash_iocs = {
            ioc.value.lower(): ioc
            for ioc in self.ioc_db.get_by_type("HASH_SHA256")
        }
        md5_iocs = {
            ioc.value.lower(): ioc
            for ioc in self.ioc_db.get_by_type("HASH_MD5")
        }

        if not hash_iocs and not md5_iocs:
            return

        scan_path = Path(directory)
        for filepath in scan_path.rglob("*"):
            if not filepath.is_file():
                continue
            # Skip very large files (>100MB)
            try:
                if filepath.stat().st_size > 100 * 1024 * 1024:
                    continue
            except OSError:
                continue

            try:
                content = filepath.read_bytes()
                sha256 = hashlib.sha256(content).hexdigest().lower()
                md5 = hashlib.md5(content).hexdigest().lower()

                if sha256 in hash_iocs:
                    self.matches.append(IOCMatch(
                        ioc=hash_iocs[sha256],
                        location=str(filepath),
                        context=f"SHA256: {sha256}",
                    ))

                if md5 in md5_iocs:
                    self.matches.append(IOCMatch(
                        ioc=md5_iocs[md5],
                        location=str(filepath),
                        context=f"MD5: {md5}",
                    ))

            except (PermissionError, OSError):
                pass

    def scan_file_names(self, directory: str) -> None:
        """알려진 악성 이름을 가진 파일을 확인합니다."""
        print(f"[*] Scanning file names in {directory}...")
        name_iocs = {
            ioc.value.lower(): ioc
            for ioc in self.ioc_db.get_by_type("FILENAME")
        }

        if not name_iocs:
            return

        scan_path = Path(directory)
        for filepath in scan_path.rglob("*"):
            if filepath.name.lower() in name_iocs:
                self.matches.append(IOCMatch(
                    ioc=name_iocs[filepath.name.lower()],
                    location=str(filepath),
                    context=f"Filename match: {filepath.name}",
                ))

    def scan_network_connections(self) -> None:
        """알려진 악성 IP와 활성 네트워크 연결을 비교합니다."""
        print("[*] Scanning network connections...")
        ip_iocs = {
            ioc.value: ioc
            for ioc in self.ioc_db.get_by_type("IP")
        }

        if not ip_iocs:
            return

        try:
            # Use netstat or ss to get connections
            result = subprocess.run(
                ["netstat", "-an"],
                capture_output=True, text=True, timeout=10
            )

            for line in result.stdout.splitlines():
                for bad_ip in ip_iocs:
                    if bad_ip in line:
                        self.matches.append(IOCMatch(
                            ioc=ip_iocs[bad_ip],
                            location="Active network connection",
                            context=line.strip(),
                        ))

        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Try ss as fallback
            try:
                result = subprocess.run(
                    ["ss", "-an"],
                    capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.splitlines():
                    for bad_ip in ip_iocs:
                        if bad_ip in line:
                            self.matches.append(IOCMatch(
                                ioc=ip_iocs[bad_ip],
                                location="Active network connection",
                                context=line.strip(),
                            ))
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print("    [!] Could not check network connections")

    def scan_dns_cache(self) -> None:
        """알려진 악성 도메인에 대한 DNS 해석을 확인합니다."""
        print("[*] Checking known malicious domains...")
        domain_iocs = self.ioc_db.get_by_type("DOMAIN")

        for ioc in domain_iocs:
            try:
                # Try to resolve the domain - if it resolves from cache
                # the system may have contacted it
                result = socket.getaddrinfo(
                    ioc.value, None, socket.AF_INET,
                    socket.SOCK_STREAM
                )
                if result:
                    ip = result[0][4][0]
                    self.matches.append(IOCMatch(
                        ioc=ioc,
                        location="DNS resolution",
                        context=f"{ioc.value} resolves to {ip}",
                    ))
            except (socket.gaierror, socket.timeout, OSError):
                pass  # Domain doesn't resolve - good

    def scan_file_contents(self, directory: str,
                           extensions: tuple = ('.py', '.js', '.sh', '.php',
                                                '.rb', '.pl', '.ps1')) -> None:
        """IOC 패턴(정규식)에 대해 파일 내용을 스캔합니다."""
        print(f"[*] Scanning file contents in {directory}...")
        regex_iocs = self.ioc_db.get_by_type("REGEX")

        if not regex_iocs:
            return

        compiled_patterns = []
        for ioc in regex_iocs:
            try:
                compiled_patterns.append((re.compile(ioc.value, re.IGNORECASE), ioc))
            except re.error:
                print(f"    [!] Invalid regex pattern: {ioc.value}")

        scan_path = Path(directory)
        for filepath in scan_path.rglob("*"):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in extensions:
                continue
            try:
                if filepath.stat().st_size > 10 * 1024 * 1024:  # Skip >10MB
                    continue
                content = filepath.read_text(errors='ignore')

                for pattern, ioc in compiled_patterns:
                    matches = pattern.findall(content)
                    if matches:
                        self.matches.append(IOCMatch(
                            ioc=ioc,
                            location=str(filepath),
                            context=f"Pattern '{ioc.value}' found {len(matches)} times",
                        ))

            except (PermissionError, OSError):
                pass

    def print_report(self) -> None:
        """IOC 스캔 결과를 출력합니다."""
        print("\n" + "=" * 65)
        print("  IOC SCAN REPORT")
        print("=" * 65)
        print(f"  IOCs in database: {len(self.ioc_db.iocs)}")
        print(f"  Matches found: {len(self.matches)}")

        if not self.matches:
            print("\n  No IOC matches found. System appears clean.")
            print("=" * 65)
            return

        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            matches = [m for m in self.matches if m.ioc.severity == severity]
            if not matches:
                continue

            print(f"\n{'─' * 65}")
            print(f"  {severity} ({len(matches)} matches)")
            print(f"{'─' * 65}")

            for match in matches:
                print(f"\n  Type: {match.ioc.ioc_type}")
                print(f"  IOC:  {match.ioc.value}")
                print(f"  Desc: {match.ioc.description}")
                print(f"  Found: {match.location}")
                print(f"  Context: {match.context}")

        print(f"\n{'=' * 65}")
        print("  RECOMMENDED ACTIONS:")
        critical = [m for m in self.matches if m.ioc.severity == 'CRITICAL']
        if critical:
            print("  [!] CRITICAL matches found - initiate incident response")
            print("  [!] Isolate affected system immediately")
        elif self.matches:
            print("  [*] Review matches and determine if they are true positives")
            print("  [*] Escalate confirmed matches to security team")
        print("=" * 65)


# ─── Example IOC JSON format ───
SAMPLE_IOC_JSON = """
{
  "iocs": [
    {
      "type": "IP",
      "value": "198.51.100.1",
      "description": "Known C2 server for BotnetX",
      "source": "ThreatIntel Feed Alpha",
      "severity": "HIGH"
    },
    {
      "type": "DOMAIN",
      "value": "evil-malware.example.com",
      "description": "Malware distribution domain",
      "source": "OSINT",
      "severity": "CRITICAL"
    },
    {
      "type": "HASH_SHA256",
      "value": "a1b2c3d4e5f6...",
      "description": "Ransomware binary",
      "source": "VirusTotal",
      "severity": "CRITICAL"
    },
    {
      "type": "FILENAME",
      "value": "mimikatz.exe",
      "description": "Credential dumping tool",
      "source": "MITRE ATT&CK",
      "severity": "CRITICAL"
    },
    {
      "type": "REGEX",
      "value": "(?:eval|exec)\\\\s*\\\\(\\\\s*base64",
      "description": "Obfuscated code execution pattern",
      "source": "Custom rule",
      "severity": "HIGH"
    }
  ]
}
"""


if __name__ == "__main__":
    import sys

    # Initialize IOC database
    ioc_db = IOCDatabase()

    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        count = ioc_db.load_from_json(sys.argv[1])
        print(f"[*] Loaded {count} IOCs from {sys.argv[1]}")
        scan_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp"
    else:
        ioc_db.load_sample_iocs()
        print(f"[*] Loaded {len(ioc_db.iocs)} sample IOCs")
        scan_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"

    # Run scanner
    scanner = IOCScanner(ioc_db)
    scanner.scan_all(scan_dir)
    scanner.print_report()
```

---

## 5. 디지털 포렌식 기초

### 5.1 포렌식 원칙

```
┌──────────────────────────────────────────────────────────────────┐
│                  디지털 포렌식 원칙                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. 증거 보존                                                     │
│     - 원본 증거로는 절대 작업하지 않음                            │
│     - 포렌식 이미지 생성 (비트 단위 복사본)                       │
│     - 디스크 액세스 시 쓰기 차단기 사용                           │
│     - 모든 것을 타임스탬프와 함께 문서화                          │
│                                                                   │
│  2. 보관 연속성 문서화                                            │
│     - 누가 증거를 수집했는가?                                     │
│     - 언제 수집되었는가?                                          │
│     - 어떻게 보관되었는가?                                        │
│     - 누가 접근했는가?                                            │
│                                                                   │
│  3. 무결성 검증                                                   │
│     - 모든 증거를 즉시 해싱 (SHA-256)                            │
│     - 분석 전후 해시 검증                                         │
│     - 모든 변경은 증거를 무효화함                                 │
│                                                                   │
│  4. 복사본으로 분석                                               │
│     - 포렌식 복사본으로 작업, 원본은 절대 안됨                    │
│     - 증거를 수정하지 않는 포렌식 도구 사용                       │
│     - 모든 분석 단계의 상세 메모 유지                             │
│                                                                   │
│  5. 결과 보고                                                     │
│     - 사실적이고 객관적인 보고                                    │
│     - 재현 가능한 방법론                                          │
│     - 증거에서 결론까지 명확한 연결                               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 휘발성 순서

증거를 수집할 때는 가장 휘발성이 높은(수명이 짧은) 데이터부터 시작합니다.

```
┌──────────────────────────────────────────────────────────────────┐
│               휘발성 순서 (높음 → 낮음)                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  가장 휘발성 높음 (먼저 수집)                                     │
│  │                                                                │
│  ├── 1. CPU 레지스터, 캐시                                        │
│  │      수명: 나노초                                              │
│  │                                                                │
│  ├── 2. 메모리 (RAM)                                             │
│  │      수명: 전원 사이클                                         │
│  │      포함: 실행 중인 프로세스, 네트워크 연결,                  │
│  │      복호화된 데이터, 패스워드, 암호화 키                      │
│  │                                                                │
│  ├── 3. 네트워크 상태                                             │
│  │      수명: 초-분                                               │
│  │      포함: 활성 연결, 라우팅 테이블, ARP 캐시                  │
│  │                                                                │
│  ├── 4. 실행 중인 프로세스                                        │
│  │      수명: 프로세스가 종료될 때까지                            │
│  │      포함: 프로세스 목록, 열린 파일, 로드된 라이브러리         │
│  │                                                                │
│  ├── 5. 디스크 (파일 시스템)                                      │
│  │      수명: 덮어쓰기될 때까지                                   │
│  │      포함: 파일, 로그, 스왑, 임시 파일, 슬랙 공간              │
│  │                                                                │
│  ├── 6. 원격 로깅 / 모니터링                                      │
│  │      수명: 보관 정책                                           │
│  │                                                                │
│  └── 7. 아카이브 미디어 (백업, 테이프)                            │
│         수명: 년                                                  │
│                                                                   │
│  가장 휘발성 낮음 (나중에 수집)                                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.3 보관 연속성

```
┌──────────────────────────────────────────────────────────────────┐
│                    보관 연속성 양식                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  사건 번호: IR-2025-0042                                          │
│  증거 ID: EVD-001                                                │
│  설명: Dell Latitude 7420 노트북, S/N: ABC123DEF                 │
│  발견 위치: 건물 A, 사무실 302                                    │
│                                                                   │
│  증거 해시 (수집 시):                                             │
│  SHA-256: a1b2c3d4e5f6789...                                     │
│                                                                   │
│  ┌──────────┬───────────────────┬────────────┬────────────────┐  │
│  │   날짜   │  인계자           │ 인수자     │    목적        │  │
│  ├──────────┼───────────────────┼────────────┼────────────────┤  │
│  │ 01/15/25 │ Officer Smith     │ Analyst Lee│ 최초           │  │
│  │ 10:30 AM │ (배지 #1234)      │ (IR팀)     │ 수집           │  │
│  ├──────────┼───────────────────┼────────────┼────────────────┤  │
│  │ 01/15/25 │ Analyst Lee       │ 증거       │ 안전           │  │
│  │ 11:45 AM │ (IR팀)            │ 보관함     │ 보관           │  │
│  ├──────────┼───────────────────┼────────────┼────────────────┤  │
│  │ 01/16/25 │ 증거 보관함       │ Forensic   │ 디스크         │  │
│  │ 09:00 AM │                   │ Analyst Kim│ 이미징         │  │
│  ├──────────┼───────────────────┼────────────┼────────────────┤  │
│  │ 01/16/25 │ Forensic Analyst  │ 증거       │ 이미징 후      │  │
│  │ 05:00 PM │ Kim               │ 보관함     │ 반납           │  │
│  └──────────┴───────────────────┴────────────┴────────────────┘  │
│                                                                   │
│  참고사항:                                                        │
│  - 수집 시 노트북이 꺼져 있었음                                   │
│  - 우발적인 부팅을 방지하기 위해 배터리 제거                      │
│  - FTK Imager를 사용하여 디스크 이미징, 해시 검증 완료           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.4 포렌식 디스크 이미징

```bash
# dd를 사용한 포렌식 이미지 생성
# 경고: dd는 매우 주의해서 사용 - 잘못된 매개변수는 데이터 파괴 가능

# 1단계: 타겟 디스크 식별
lsblk
# or
fdisk -l

# 2단계: 포렌식 이미지 생성 (비트 단위 복사)
# /dev/sdb = 소스 (증거 드라이브, 쓰기 차단기를 통해)
# evidence.dd = 대상 이미지 파일
sudo dd if=/dev/sdb of=evidence.dd bs=4096 conv=noerror,sync status=progress

# 3단계: 원본과 이미지의 해시 계산
sha256sum /dev/sdb > original_hash.txt
sha256sum evidence.dd > image_hash.txt

# 4단계: 해시 일치 검증
diff original_hash.txt image_hash.txt

# 더 나은 대안: dc3dd (포렌식에 특화된 dd)
sudo dc3dd if=/dev/sdb of=evidence.dd hash=sha256 log=imaging.log

# 대안: FTK Imager (크로스 플랫폼, GUI)
# 내장 해싱이 포함된 E01 (Expert Witness) 형식 생성
```

---

## 6. 메모리 포렌식 개념

### 6.1 메모리 포렌식이 필요한 이유

```
┌──────────────────────────────────────────────────────────────────┐
│                메모리에만 존재하는 것들                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────┐                 │
│  │  RAM에서만 찾을 수 있는 것들:               │                 │
│  │                                              │                 │
│  │  - 실행 중인 프로세스 (숨겨진 것 포함)      │                 │
│  │  - 네트워크 연결                             │                 │
│  │  - 복호화 키                                 │                 │
│  │  - 패스워드 (메모리 내 평문)                │                 │
│  │  - 주입된 코드 (파일리스 악성코드)          │                 │
│  │  - 클립보드 내용                             │                 │
│  │  - 채팅 메시지 (디스크에 저장되기 전)       │                 │
│  │  - 암호화 키 (전체 디스크 암호화)           │                 │
│  │  - 명령 기록                                 │                 │
│  │  - 언팩/복호화된 악성코드                   │                 │
│  └─────────────────────────────────────────────┘                 │
│                                                                   │
│  최신 악성코드는 디스크 기반 탐지를 피하기 위해                   │
│  전적으로 메모리에서 작동하는 경우가 많습니다                     │
│  ("파일리스 악성코드").                                           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 메모리 수집

```bash
# Linux 메모리 수집

# 방법 1: /proc/kcore (루트 권한 필요)
sudo dd if=/proc/kcore of=memory.raw bs=1M

# 방법 2: LiME (Linux Memory Extractor) - 권장
# LiME 커널 모듈 로드
sudo insmod lime-$(uname -r).ko "path=memory.lime format=lime"

# 방법 3: AVML (Microsoft의 Linux 메모리 수집 도구)
sudo ./avml memory.lime

# Windows 메모리 수집
# - FTK Imager (무료, GUI)
# - WinPmem (커맨드라인)
# - Belkasoft RAM Capturer (무료)

# macOS 메모리 수집
# - osxpmem
sudo ./osxpmem -o memory.aff4
```

### 6.3 Volatility 프레임워크 (개요)

```bash
# Volatility 3 - 메모리 포렌식 프레임워크
# 설치: pip install volatility3

# OS 프로파일 식별
vol -f memory.raw windows.info

# 실행 중인 프로세스 목록
vol -f memory.raw windows.pslist
vol -f memory.raw windows.pstree  # 트리 뷰

# 숨겨진 프로세스 찾기
vol -f memory.raw windows.psscan

# 네트워크 연결
vol -f memory.raw windows.netscan

# 명령 기록
vol -f memory.raw windows.cmdline

# 특정 프로세스의 DLL 목록
vol -f memory.raw windows.dlllist --pid 1234

# 특정 프로세스 덤프
vol -f memory.raw windows.memmap --pid 1234 --dump

# 레지스트리 분석
vol -f memory.raw windows.registry.hivelist

# Linux 메모리 분석
vol -f memory.raw linux.pslist
vol -f memory.raw linux.bash  # Bash 기록
vol -f memory.raw linux.netstat
```

---

## 7. 네트워크 포렌식

### 7.1 패킷 캡처 분석

```python
"""
pcap_analyzer.py - 패킷 캡처 분석을 사용한 기본 네트워크 포렌식.
필요: pip install scapy

경고: 모니터링 권한이 있는 네트워크의 캡처만 분석하십시오.
"""

try:
    from scapy.all import rdpcap, IP, TCP, UDP, DNS, Raw
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Scapy가 설치되지 않았습니다. pip install scapy로 설치하세요")

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class PcapAnalysis:
    """PCAP 파일 분석 결과."""
    total_packets: int = 0
    protocols: dict = field(default_factory=dict)
    top_talkers: list = field(default_factory=list)
    dns_queries: list = field(default_factory=list)
    suspicious_connections: list = field(default_factory=list)
    http_requests: list = field(default_factory=list)
    large_transfers: list = field(default_factory=list)


def analyze_pcap(filepath: str) -> PcapAnalysis:
    """
    보안 관련 정보를 찾기 위해 PCAP 파일을 분석합니다.

    Args:
        filepath: .pcap 또는 .pcapng 파일 경로

    Returns:
        발견 사항이 포함된 PcapAnalysis
    """
    if not SCAPY_AVAILABLE:
        raise ImportError("PCAP 분석을 위해 Scapy가 필요합니다")

    packets = rdpcap(filepath)
    analysis = PcapAnalysis(total_packets=len(packets))

    # 통계 추적
    ip_src_counter = Counter()
    ip_dst_counter = Counter()
    protocol_counter = Counter()
    connection_sizes = defaultdict(int)
    dns_queries = []
    http_requests = []

    for pkt in packets:
        # 프로토콜 분석
        if pkt.haslayer(TCP):
            protocol_counter['TCP'] += 1
        elif pkt.haslayer(UDP):
            protocol_counter['UDP'] += 1

        # IP 레이어 분석
        if pkt.haslayer(IP):
            src = pkt[IP].src
            dst = pkt[IP].dst
            ip_src_counter[src] += 1
            ip_dst_counter[dst] += 1

            # 연결 데이터 볼륨 추적
            if pkt.haslayer(Raw):
                key = f"{src} -> {dst}"
                connection_sizes[key] += len(pkt[Raw].load)

        # DNS 분석
        if pkt.haslayer(DNS) and pkt[DNS].qr == 0:  # Query
            try:
                query_name = pkt[DNS].qd.qname.decode('utf-8', errors='ignore')
                dns_queries.append({
                    'query': query_name,
                    'src': pkt[IP].src if pkt.haslayer(IP) else 'unknown',
                    'type': pkt[DNS].qd.qtype,
                })
            except (AttributeError, IndexError):
                pass

        # HTTP 요청 탐지 (기본)
        if pkt.haslayer(TCP) and pkt.haslayer(Raw):
            payload = pkt[Raw].load
            try:
                text = payload.decode('utf-8', errors='ignore')
                if text.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ')):
                    lines = text.split('\r\n')
                    http_requests.append({
                        'method': lines[0].split(' ')[0],
                        'path': lines[0].split(' ')[1] if len(lines[0].split(' ')) > 1 else '',
                        'src': pkt[IP].src if pkt.haslayer(IP) else 'unknown',
                        'dst': pkt[IP].dst if pkt.haslayer(IP) else 'unknown',
                    })
            except (UnicodeDecodeError, IndexError):
                pass

    # 결과 컴파일
    analysis.protocols = dict(protocol_counter)

    # 주요 대화자 (패킷 수 기준)
    analysis.top_talkers = [
        {'ip': ip, 'packets_sent': count}
        for ip, count in ip_src_counter.most_common(10)
    ]

    # DNS 쿼리 (중복 제거)
    seen_queries = set()
    for q in dns_queries:
        if q['query'] not in seen_queries:
            analysis.dns_queries.append(q)
            seen_queries.add(q['query'])

    # 대용량 데이터 전송 (잠재적 유출)
    analysis.large_transfers = [
        {'connection': conn, 'bytes': size}
        for conn, size in sorted(
            connection_sizes.items(), key=lambda x: x[1], reverse=True
        )[:10]
    ]

    analysis.http_requests = http_requests[:50]

    # 의심스러운 패턴 탐지
    analysis.suspicious_connections = detect_suspicious_patterns(
        packets, ip_src_counter, dns_queries
    )

    return analysis


def detect_suspicious_patterns(packets, ip_counter, dns_queries):
    """의심스러운 네트워크 패턴을 탐지합니다."""
    suspicious = []

    # 1. 비컨 탐지 (정기적인 간격 연결)
    # 단순화: 매우 규칙적인 패킷 간격을 가진 IP 확인
    # (실제 비컨 탐지는 통계적 분석 필요)

    # 2. DNS 터널링 지표
    long_queries = [q for q in dns_queries if len(q['query']) > 50]
    if long_queries:
        suspicious.append({
            'type': 'DNS_TUNNELING_POSSIBLE',
            'description': f"{len(long_queries)}개의 비정상적으로 긴 DNS 쿼리 발견",
            'samples': [q['query'][:80] for q in long_queries[:3]],
        })

    # 3. 포트 스캔 탐지
    dst_ports = defaultdict(set)
    for pkt in packets:
        if pkt.haslayer(TCP) and pkt.haslayer(IP):
            src = pkt[IP].src
            dst_port = pkt[TCP].dport
            dst_ports[src].add(dst_port)

    for ip, ports in dst_ports.items():
        if len(ports) > 20:  # 많은 다른 포트 접근
            suspicious.append({
                'type': 'PORT_SCAN_POSSIBLE',
                'description': f"{ip}가 {len(ports)}개의 다른 포트에 접촉",
                'samples': sorted(list(ports))[:10],
            })

    return suspicious


def print_pcap_report(analysis: PcapAnalysis) -> None:
    """형식화된 PCAP 분석 보고서를 출력합니다."""
    print("=" * 65)
    print("  네트워크 포렌식 보고서")
    print("=" * 65)
    print(f"  총 패킷 수: {analysis.total_packets}")
    print(f"  프로토콜: {analysis.protocols}")

    print(f"\n{'─' * 65}")
    print("  주요 대화자 (전송된 패킷 기준)")
    print(f"{'─' * 65}")
    for t in analysis.top_talkers:
        print(f"  {t['ip']:20s} {t['packets_sent']} 패킷")

    if analysis.dns_queries:
        print(f"\n{'─' * 65}")
        print(f"  DNS 쿼리 ({len(analysis.dns_queries)}개 고유)")
        print(f"{'─' * 65}")
        for q in analysis.dns_queries[:20]:
            print(f"  {q['src']:20s} -> {q['query']}")

    if analysis.large_transfers:
        print(f"\n{'─' * 65}")
        print("  최대 데이터 전송")
        print(f"{'─' * 65}")
        for t in analysis.large_transfers:
            size_kb = t['bytes'] / 1024
            print(f"  {t['connection']:40s} {size_kb:.1f} KB")

    if analysis.suspicious_connections:
        print(f"\n{'─' * 65}")
        print("  의심스러운 패턴")
        print(f"{'─' * 65}")
        for s in analysis.suspicious_connections:
            print(f"  [{s['type']}] {s['description']}")
            if s.get('samples'):
                for sample in s['samples']:
                    print(f"    - {sample}")

    print(f"\n{'=' * 65}")
```

### 7.2 유용한 커맨드라인 도구

```bash
# tcpdump - 패킷 캡처
# eth0에서 모든 트래픽 캡처
sudo tcpdump -i eth0 -w capture.pcap

# 특정 IP와의 트래픽만 캡처
sudo tcpdump -i eth0 host 192.168.1.100 -w suspicious.pcap

# HTTP 트래픽만 캡처
sudo tcpdump -i eth0 port 80 -w http_traffic.pcap

# DNS 트래픽 캡처
sudo tcpdump -i eth0 port 53 -w dns_traffic.pcap

# tshark (커맨드라인 Wireshark)
# HTTP 요청 추출
tshark -r capture.pcap -Y "http.request" -T fields \
  -e ip.src -e http.request.method -e http.request.uri

# DNS 쿼리 추출
tshark -r capture.pcap -Y "dns.qr == 0" -T fields \
  -e ip.src -e dns.qry.name

# 파일 전송 추출
tshark -r capture.pcap --export-objects http,exported_files/

# 대화 통계 표시
tshark -r capture.pcap -z conv,ip -q
```

---

## 8. 사고 대응 플레이북

### 8.1 플레이북 템플릿

```
┌──────────────────────────────────────────────────────────────────┐
│                  IR 플레이북 템플릿                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  플레이북: [사고 유형]                                            │
│  버전: 1.0                                                        │
│  최종 업데이트: [날짜]                                            │
│  담당자: [팀/담당자]                                              │
│  심각도: [P1-P4]                                                  │
│                                                                   │
│  트리거:                                                          │
│  [이 플레이북을 활성화하는 경고/조건]                             │
│                                                                   │
│  초기 트리아지 (첫 15분):                                         │
│  [ ] 1단계: ...                                                   │
│  [ ] 2단계: ...                                                   │
│  [ ] 3단계: 심각도 분류                                           │
│  [ ] 4단계: 사고 관리자에게 알림                                  │
│                                                                   │
│  격리 (첫 1-4시간):                                               │
│  [ ] 1단계: ...                                                   │
│  [ ] 2단계: ...                                                   │
│  [ ] 3단계: 증거 보존                                             │
│                                                                   │
│  제거:                                                            │
│  [ ] 1단계: ...                                                   │
│  [ ] 2단계: ...                                                   │
│                                                                   │
│  복구:                                                            │
│  [ ] 1단계: ...                                                   │
│  [ ] 2단계: 정상 작동 검증                                        │
│                                                                   │
│  커뮤니케이션:                                                    │
│  - 내부: [누구에게 언제 알릴지]                                  │
│  - 외부: [고객, 규제기관, 법 집행기관]                            │
│                                                                   │
│  에스컬레이션:                                                    │
│  - 조건 → 조치                                                    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 플레이북: 랜섬웨어 사고

```
┌──────────────────────────────────────────────────────────────────┐
│            플레이북: 랜섬웨어 사고                                 │
├──────────────────────────────────────────────────────────────────┤
│  심각도: P1 (Critical)                                            │
│                                                                   │
│  트리거:                                                          │
│  - 엔드포인트에 랜섬 메모 표시됨                                  │
│  - 대량 파일 암호화 탐지                                          │
│  - EDR 경고: 랜섬웨어 행위                                        │
│                                                                   │
│  초기 트리아지 (첫 15분):                                         │
│  [ ] 1. 영향받은 시스템 전원 끄지 말 것                           │
│  [ ] 2. 영향받은 시스템을 네트워크에서 분리                       │
│         (이더넷 뽑기, WiFi 비활성화 - 종료하지 말 것)            │
│  [ ] 3. 랜섬 메모 문서화 (사진/스크린샷)                          │
│  [ ] 4. 가능하면 랜섬웨어 변종 식별                               │
│  [ ] 5. 범위 결정: 몇 개의 시스템이 영향을 받았는가?             │
│  [ ] 6. 사고 관리자에게 알림 → IR 팀 활성화                      │
│  [ ] 7. CISO / 경영진에게 알림                                   │
│                                                                   │
│  격리 (첫 1-4시간):                                               │
│  [ ] 1. 영향받은 네트워크 세그먼트 격리                           │
│  [ ] 2. 알려진 랜섬웨어 C2 IP/도메인 차단                        │
│  [ ] 3. 확산 방지를 위해 네트워크 공유 비활성화                   │
│  [ ] 4. 잠재적으로 침해된 모든 자격 증명 재설정                   │
│  [ ] 5. 영향받은 시스템의 메모리 덤프 캡처                        │
│  [ ] 6. 로그 보존 (SIEM, 방화벽, 엔드포인트)                     │
│  [ ] 7. 백업 무결성 확인 (백업이 영향받았는가?)                  │
│                                                                   │
│  제거:                                                            │
│  [ ] 1. 초기 감염 벡터 식별 (이메일, 익스플로잇 등)              │
│  [ ] 2. NoMoreRansom.org에서 복호화 도구 확인                    │
│  [ ] 3. 영향받은 모든 시스템에서 악성코드 제거                    │
│  [ ] 4. 감염을 허용한 취약점 패치                                 │
│  [ ] 5. 지속성 메커니즘에 대해 모든 시스템 스캔                   │
│                                                                   │
│  복구:                                                            │
│  [ ] 1. 클린하고 검증된 백업에서 복원                             │
│  [ ] 2. 정리할 수 없는 시스템 재구축                              │
│  [ ] 3. 재감염 모니터링하며 단계적으로 복원                       │
│  [ ] 4. 조직 전체 모든 패스워드 재설정                            │
│  [ ] 5. 복구 후 30일간 모니터링 강화                              │
│                                                                   │
│  커뮤니케이션:                                                    │
│  - 내부: 2시간 내 전체 알림                                       │
│  - 법무: 즉시 법률 고문 참여                                      │
│  - 보험: 사이버 보험사에 알림                                     │
│  - 법 집행기관: FBI IC3에 신고 제출                              │
│  - 규제기관: 규제 요구사항에 따라 (GDPR: 72시간)                 │
│  - 고객: 데이터 침해 확인 시                                      │
│                                                                   │
│  하지 말 것:                                                      │
│  - 법무 및 법 집행기관과 상담 없이 몸값 지불하지 말 것            │
│  - 법적 지침 없이 공격자와 통신하지 말 것                         │
│  - 증거 파괴하지 말 것                                            │
│  - 클린한지 확인하기 전에 백업에서 복원하지 말 것                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 8.3 플레이북: 자격 증명 침해

```
┌──────────────────────────────────────────────────────────────────┐
│         플레이북: 자격 증명 침해                                   │
├──────────────────────────────────────────────────────────────────┤
│  심각도: P2 (High)                                                │
│                                                                   │
│  트리거:                                                          │
│  - 다크웹 / 페이스트 사이트에서 자격 증명 발견                    │
│  - 사용자가 피싱 / 자격 증명 절도 신고                            │
│  - 불가능한 이동 경고                                             │
│  - MFA 우회 탐지                                                  │
│                                                                   │
│  초기 트리아지:                                                   │
│  [ ] 1. 영향받은 계정 식별                                        │
│  [ ] 2. 자격 증명 유형 결정 (패스워드, API 키, 토큰)             │
│  [ ] 3. 감사 로그에서 무단 접근 확인                              │
│  [ ] 4. MFA가 활성화되어 있었는지 확인                            │
│                                                                   │
│  격리:                                                            │
│  [ ] 1. 영향받은 계정의 패스워드 강제 재설정                      │
│  [ ] 2. 모든 활성 세션 / 토큰 취소                               │
│  [ ] 3. 해당하는 경우 API 키 교체                                 │
│  [ ] 4. 아직 활성화되지 않았다면 MFA 활성화                       │
│  [ ] 5. 의심스러운 소스 IP 차단                                   │
│  [ ] 6. 메일박스 규칙 확인 (전달, 삭제)                           │
│                                                                   │
│  조사:                                                            │
│  [ ] 1. 침해된 자격 증명으로 수행된 모든 조치 검토                │
│  [ ] 2. 측면 이동 확인 (다른 시스템 접근)                         │
│  [ ] 3. 데이터 접근 / 유출 확인                                   │
│  [ ] 4. 자격 증명이 어떻게 침해되었는지 식별                      │
│  [ ] 5. 자격 증명이 다른 서비스에서 재사용되었는지 확인           │
│                                                                   │
│  복구:                                                            │
│  [ ] 1. 계정이 보안되었는지 검증 (새 패스워드 + MFA)              │
│  [ ] 2. 무단 변경 사항 되돌림                                     │
│  [ ] 3. 사고를 사용자에게 알리고 보안 교육 요구                   │
│  [ ] 4. 30일간 계정 모니터링                                      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. 사고 후 검토 템플릿

```python
"""
incident_report.py - 사고 후 검토 보고서를 생성합니다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TimelineEvent:
    """사고 타임라인의 단일 이벤트."""
    timestamp: str
    description: str
    actor: str = ""        # 조치를 수행한 사람
    evidence: str = ""     # 뒷받침하는 증거


@dataclass
class ActionItem:
    """사고 검토의 후속 조치."""
    description: str
    owner: str
    due_date: str
    priority: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    status: str = "OPEN"      # OPEN, IN_PROGRESS, DONE


@dataclass
class IncidentReport:
    """완전한 사고 후 검토 보고서."""
    # 메타데이터
    incident_id: str
    title: str
    severity: str
    status: str = "CLOSED"
    report_date: str = ""
    report_author: str = ""

    # 타임라인
    detected_at: str = ""
    contained_at: str = ""
    eradicated_at: str = ""
    recovered_at: str = ""
    closed_at: str = ""

    # 상세
    summary: str = ""
    root_cause: str = ""
    impact: str = ""
    affected_systems: list[str] = field(default_factory=list)
    affected_users: int = 0
    data_compromised: str = ""

    # 분석
    attack_vector: str = ""
    attacker_info: str = ""
    timeline: list[TimelineEvent] = field(default_factory=list)

    # 교훈
    what_went_well: list[str] = field(default_factory=list)
    what_went_wrong: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)

    # 메트릭
    time_to_detect: str = ""       # 침해부터 탐지까지의 시간
    time_to_contain: str = ""      # 탐지부터 격리까지의 시간
    time_to_recover: str = ""      # 격리부터 복구까지의 시간
    total_duration: str = ""       # 총 사고 지속 시간

    def __post_init__(self):
        if not self.report_date:
            self.report_date = datetime.now().strftime("%Y-%m-%d")

    def generate_markdown(self) -> str:
        """Markdown 보고서를 생성합니다."""
        lines = []

        lines.append(f"# 사고 보고서: {self.incident_id}")
        lines.append(f"\n**제목**: {self.title}")
        lines.append(f"**심각도**: {self.severity}")
        lines.append(f"**상태**: {self.status}")
        lines.append(f"**보고서 작성일**: {self.report_date}")
        lines.append(f"**작성자**: {self.report_author}")

        # 요약
        lines.append("\n## 요약\n")
        lines.append(self.summary)

        # 영향
        lines.append("\n## 영향\n")
        lines.append(self.impact)
        if self.affected_systems:
            lines.append(f"\n**영향받은 시스템**: {', '.join(self.affected_systems)}")
        lines.append(f"**영향받은 사용자**: {self.affected_users}")
        if self.data_compromised:
            lines.append(f"**침해된 데이터**: {self.data_compromised}")

        # 타임라인
        lines.append("\n## 타임라인\n")
        lines.append("| 시간 | 이벤트 | 담당자 |")
        lines.append("|------|-------|-------|")
        for event in self.timeline:
            lines.append(
                f"| {event.timestamp} | {event.description} | {event.actor} |"
            )

        # 주요 메트릭
        lines.append("\n## 주요 메트릭\n")
        lines.append(f"- **탐지 시간**: {self.time_to_detect}")
        lines.append(f"- **격리 시간**: {self.time_to_contain}")
        lines.append(f"- **복구 시간**: {self.time_to_recover}")
        lines.append(f"- **총 지속 시간**: {self.total_duration}")

        # 근본 원인
        lines.append("\n## 근본 원인 분석\n")
        lines.append(self.root_cause)
        lines.append(f"\n**공격 벡터**: {self.attack_vector}")

        # 교훈
        lines.append("\n## 교훈\n")
        lines.append("### 잘된 점\n")
        for item in self.what_went_well:
            lines.append(f"- {item}")
        lines.append("\n### 개선이 필요한 점\n")
        for item in self.what_went_wrong:
            lines.append(f"- {item}")

        # 액션 아이템
        lines.append("\n## 액션 아이템\n")
        lines.append("| # | 조치 | 담당자 | 마감일 | 우선순위 | 상태 |")
        lines.append("|---|--------|-------|----------|----------|--------|")
        for i, action in enumerate(self.action_items, 1):
            lines.append(
                f"| {i} | {action.description} | {action.owner} | "
                f"{action.due_date} | {action.priority} | {action.status} |"
            )

        return "\n".join(lines)


# ─── 사용 예제 ───

def create_sample_report() -> IncidentReport:
    """데모를 위한 샘플 사고 보고서를 생성합니다."""
    report = IncidentReport(
        incident_id="IR-2025-0042",
        title="침해된 API 키를 통한 무단 접근",
        severity="P2 - High",
        report_author="보안팀",
        summary=(
            "2025년 1월 15일, 무단 당사자가 침해된 API 키를 사용하여 "
            "프로덕션 API에 접근했습니다. 키가 공개 GitHub 저장소에 "
            "실수로 커밋되었습니다. 공격자는 탐지되기 전 약 2시간 동안 "
            "고객 주문 데이터에 접근했습니다."
        ),
        root_cause=(
            "개발자가 1월 10일 공개 GitHub 저장소에 API 키를 커밋했습니다. "
            "키가 자동화된 봇에 의해 스크랩되어 1월 15일 프로덕션 API "
            "접근에 사용되었습니다. 개발자 머신에 비밀 탐지를 위한 "
            "pre-commit hook이 구성되지 않았습니다."
        ),
        impact=(
            "약 1,200명의 고객에 대한 고객 주문 데이터(이름, 주소, 주문 기록)가 "
            "잠재적으로 접근되었습니다. 결제 카드 데이터는 노출되지 않았습니다 "
            "(별도 저장). 데이터 수정의 증거는 없습니다."
        ),
        attack_vector="공개 Git 저장소에서 침해된 API 키",
        affected_systems=["api-prod-01", "api-prod-02", "orders-db"],
        affected_users=1200,
        data_compromised="고객 이름, 주소, 주문 기록",
        detected_at="2025-01-15 14:30 UTC",
        contained_at="2025-01-15 14:45 UTC",
        eradicated_at="2025-01-15 16:00 UTC",
        recovered_at="2025-01-15 18:00 UTC",
        closed_at="2025-01-20 09:00 UTC",
        time_to_detect="5일 (키 커밋부터 탐지까지)",
        time_to_contain="15분",
        time_to_recover="3.5시간",
        total_duration="5일",
        timeline=[
            TimelineEvent(
                "2025-01-10 09:15", "API 키가 공개 저장소에 커밋됨",
                "개발자 A"
            ),
            TimelineEvent(
                "2025-01-15 12:30", "첫 무단 API 접근",
                "미상의 공격자"
            ),
            TimelineEvent(
                "2025-01-15 14:30", "비정상적인 API 사용 경고 발생",
                "SIEM"
            ),
            TimelineEvent(
                "2025-01-15 14:35", "SOC 분석가가 무단 접근 확인",
                "분석가 B"
            ),
            TimelineEvent(
                "2025-01-15 14:45", "API 키 취소, 공격자 차단",
                "분석가 B"
            ),
            TimelineEvent(
                "2025-01-15 15:00", "사고 관리자에게 알림, IR 활성화",
                "IR 리드 C"
            ),
            TimelineEvent(
                "2025-01-15 16:00", "노출된 모든 API 키 교체",
                "DevOps 팀"
            ),
            TimelineEvent(
                "2025-01-15 18:00", "모니터링 결과 추가 접근 없음 확인",
                "SOC 팀"
            ),
        ],
        what_went_well=[
            "비정상 패턴이 탐지되면 SIEM 경고가 빠르게 발생",
            "API 키 취소가 신속했음 (경고부터 격리까지 15분)",
            "IR 팀이 플레이북을 효과적으로 따름",
            "SOC와 개발팀 간 원활한 커뮤니케이션",
        ],
        what_went_wrong=[
            "API 키가 탐지되기 전 5일간 공개 저장소에 있었음",
            "GitHub 저장소에 자동화된 비밀 스캔 없음",
            "모든 개발자 머신에 pre-commit hook이 강제되지 않음",
            "API 키가 지나치게 광범위한 권한을 가짐 (모든 주문 읽기)",
            "API 키에 IP 기반 접근 제한 없음",
        ],
        action_items=[
            ActionItem(
                "모든 저장소에 Gitleaks 배포",
                "DevOps 팀", "2025-02-01", "HIGH"
            ),
            ActionItem(
                "detect-secrets로 pre-commit hook 강제",
                "개발 리드", "2025-02-15", "HIGH"
            ),
            ActionItem(
                "API 키 범위 제한 구현 (최소 권한)",
                "API 팀", "2025-03-01", "HIGH"
            ),
            ActionItem(
                "프로덕션 API 키에 IP 허용 목록 추가",
                "인프라", "2025-03-01", "MEDIUM"
            ),
            ActionItem(
                "개발자 보안 교육 실시 (비밀 관리)",
                "보안팀", "2025-02-28", "MEDIUM"
            ),
            ActionItem(
                "자동 키 교체 구현 (최대 90일)",
                "DevOps 팀", "2025-04-01", "MEDIUM"
            ),
        ],
    )

    return report


if __name__ == "__main__":
    report = create_sample_report()
    markdown = report.generate_markdown()
    print(markdown)

    # 파일로 저장
    with open("incident_report_IR-2025-0042.md", "w") as f:
        f.write(markdown)
    print("\n보고서가 저장되었습니다: incident_report_IR-2025-0042.md")
```

---

## 10. 연습 문제

### 연습 문제 1: 로그 분석

다음 샘플 로그 엔트리가 주어졌을 때, 모든 보안 사고를 식별하십시오:

```
192.168.1.50 - - [15/Jan/2025:10:00:01 +0000] "GET /login HTTP/1.1" 200 1234
192.168.1.50 - - [15/Jan/2025:10:00:02 +0000] "POST /login HTTP/1.1" 401 89
192.168.1.50 - - [15/Jan/2025:10:00:03 +0000] "POST /login HTTP/1.1" 401 89
192.168.1.50 - - [15/Jan/2025:10:00:04 +0000] "POST /login HTTP/1.1" 401 89
10.0.0.5 - admin [15/Jan/2025:10:05:00 +0000] "GET /admin/users HTTP/1.1" 200 5678
10.0.0.5 - admin [15/Jan/2025:10:05:01 +0000] "GET /admin/export?table=users HTTP/1.1" 200 890123
10.0.0.5 - admin [15/Jan/2025:10:05:02 +0000] "GET /admin/export?table=payments HTTP/1.1" 200 1234567
203.0.113.10 - - [15/Jan/2025:10:10:00 +0000] "GET /../../etc/passwd HTTP/1.1" 403 0
203.0.113.10 - - [15/Jan/2025:10:10:01 +0000] "GET /search?q=' OR 1=1 -- HTTP/1.1" 500 0
203.0.113.10 - - [15/Jan/2025:10:10:02 +0000] "GET /search?q=<script>alert(1)</script> HTTP/1.1" 200 456
```

**과제:**
1. 각 의심스러운 패턴 분류 (무차별 대입, 순회, SQLi, XSS, 데이터 유출)
2. 각 발견 사항의 심각도 결정
3. 각 사항에 대한 간단한 사고 요약 작성

### 연습 문제 2: IOC 데이터베이스

다음을 포함하는 최소 20개의 IOC가 있는 JSON 파일을 생성하십시오:
- 5개의 악성 IP 주소
- 5개의 악성 도메인
- 5개의 악성코드 파일 해시
- 5개의 의심스러운 파일명

일치하는 항목이 있는 테스트 디렉토리를 생성하고 IOC 스캐너를 실행하십시오.

### 연습 문제 3: 사고 대응 플레이북

**SQL 인젝션 공격**에 대한 완전한 사고 대응 플레이북을 작성하십시오:
1. 트리거 조건 정의 (SQLi를 나타내는 경고)
2. NIST의 4단계 모두 포함
3. 각 단계에서 사용할 구체적인 명령/도구 포함
4. 커뮤니케이션 및 에스컬레이션 절차 정의
5. 사고 후 체크리스트 포함

### 연습 문제 4: 메모리 포렌식 분석

Volatility 3를 연구하고 다음을 위한 메모리 덤프 분석 단계별 가이드를 작성하십시오:
1. 실행 중인 모든 프로세스 나열 및 의심스러운 프로세스 식별
2. 활성 네트워크 연결 찾기
3. 각 프로세스의 명령줄 인수 추출
4. 주입된 DLL 또는 코드 식별
5. 메모리에서 암호화 키 또는 패스워드 복구

### 연습 문제 5: 사고 후 보고서

이 레슨의 `IncidentReport` 클래스를 사용하여 다음 시나리오에 대한 완전한 사고 후 보고서를 생성하십시오:

> 회사의 웹 애플리케이션이 일요일 오전 3시에 변조되었습니다. 공격자가 패치되지 않은 WordPress 플러그인의 알려진 CVE를 악용했습니다. 홈페이지를 정치적 메시지로 교체했습니다. 모니터링 시스템이 오전 3:15에 변경을 탐지했습니다. 당직 엔지니어가 오전 4:00에 백업에서 복원했습니다. 조사 결과 공격자가 백도어 관리자 계정도 생성한 것으로 밝혀졌습니다.

### 연습 문제 6: PCAP 분석

CTF 또는 보안 교육 리소스(예: malware-traffic-analysis.net)에서 샘플 PCAP 파일을 다운로드하십시오. 이 레슨의 도구를 사용하여 분석하고 다음을 포함하는 네트워크 포렌식 보고서를 작성하십시오:
1. 주요 대화자 (가장 활발한 IP 주소)
2. DNS 쿼리 (특히 의심스러운 것)
3. HTTP 요청 (악성코드 다운로드, C2 통신 찾기)
4. 데이터 유출 지표

---

## 요약

```
┌──────────────────────────────────────────────────────────────────┐
│           사고 대응 핵심 요점                                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. 준비가 전부입니다: 사고가 발생하기 전에 계획, 도구,          │
│     훈련된 인력을 갖추십시오                                      │
│                                                                   │
│  2. NIST 생명주기를 따르십시오: 준비 → 탐지 →                    │
│     격리 → 제거 → 복구 → 교훈 학습                               │
│                                                                   │
│  3. 증거를 보존하십시오: 모든 것을 문서화하고, 보관 연속성을     │
│     유지하고, 모든 증거를 해싱하고, 복사본으로 작업하십시오      │
│                                                                   │
│  4. 로깅을 중앙화하십시오: 로그하지 않은 것은 조사할 수          │
│     없습니다. 로깅 인프라에 투자하십시오                          │
│                                                                   │
│  5. 탐지를 자동화하십시오: SIEM 상관 규칙과 IOC 스캔을           │
│     사용하여 탐지 시간을 줄이십시오                               │
│                                                                   │
│  6. 플레이북으로 연습하십시오: 작성되고 테스트된 플레이북은      │
│     대응 시간을 줄이고 일관성을 보장합니다                        │
│                                                                   │
│  7. 사고로부터 배우십시오: 사고 후 검토는 보안 개선을 위한       │
│     가장 가치 있는 소스입니다                                     │
│                                                                   │
│  8. 시간이 중요합니다: 활성 사고 중에는 분 단위가 중요합니다.    │
│     평균 탐지 시간(MTTD)과 평균 대응 시간(MTTR)이                │
│     핵심 메트릭입니다                                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

**이전**: [13. 보안 테스팅](13_Security_Testing.md) | **다음**: [15. 프로젝트: 보안 REST API 구축](15_Project_Secure_API.md)
