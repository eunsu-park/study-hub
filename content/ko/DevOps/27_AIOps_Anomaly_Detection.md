# 27. AIOps와 이상 탐지(Anomaly Detection)

**이전**: [인시던트 대응](./26_Incident_Response.md) | **다음**: [캡스톤: 풀스택 관측 가능성](./28_Capstone_Full_Stack_Observability.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. AIOps를 정의하고 머신러닝(machine learning)이 관측 가능성과 운영을 어떻게 향상시키는지 설명할 수 있습니다
2. 시계열(time-series) 메트릭에 대한 통계적 이상 탐지 기법을 구현할 수 있습니다
3. ML 기반 임계값을 통해 알림 피로(alert fatigue)를 줄이는 지능형 알림 시스템을 설계할 수 있습니다
4. AIOps 플랫폼과 그 핵심 기능(이상 탐지, 상관 관계, 근본 원인 분석)을 평가할 수 있습니다
5. 적절한 안전 장치(safeguard)를 갖춘 자동 복구(automated remediation) 패턴을 적용할 수 있습니다
6. AIOps 도구에서 과대 광고(hype)와 실용적 가치를 구별할 수 있습니다

---

정적 알림 임계값("CPU > 80%이면 알림")은 단순하고 예측 가능한 시스템에서는 작동합니다. 그러나 현대 분산 시스템은 복잡하고 동적인 동작을 보입니다 -- 계절적 트래픽 패턴, 점진적 성능 저하, 서비스 간 상관된 장애. AIOps는 머신러닝을 관측 가능성 데이터에 적용하여 정적 규칙이 놓치는 이상을 감지하고, 관련 알림을 상관시켜 노이즈를 줄이며, 일상적인 복구를 자동화합니다.

> **비유 -- 기상 예보**: 정적 알림은 "습도 > 90%이면 비 경고"와 같은 고정 규칙입니다. 습하지만 맑은 날에 거짓 경보를 발생시키고 건조하지만 폭풍이 오는 날에 비를 놓칩니다. ML 기반 이상 탐지는 현대 기상 예보와 같습니다: 기압, 바람, 온도 추세, 과거 패턴, 위성 이미지를 고려하여 훨씬 높은 정확도로 비를 예측합니다. 모델은 각 컨텍스트에서 "정상"이 무엇인지 학습하고 진정한 이상에 대해서만 알림합니다.

## 1. AIOps 개요

### 1.1 AIOps가 다루는 범위

```
┌─────────────────────────────────────────────────┐
│                   AIOps 스택                      │
├─────────────────────────────────────────────────┤
│ 계층 4: 자동 복구(Automated Remediation)          │
│   - 예측된 수요에 기반한 자동 스케일링              │
│   - 자가 치유 (재시작, 롤백, 페일오버)             │
│   - 이상에 의해 트리거되는 런북 자동화              │
├─────────────────────────────────────────────────┤
│ 계층 3: 근본 원인 분석(Root Cause Analysis)       │
│   - 상관된 이상에서 인과 추론                      │
│   - 토폴로지 인식 장애 위치 파악                   │
│   - 변경-영향 상관 관계                           │
├─────────────────────────────────────────────────┤
│ 계층 2: 알림 인텔리전스(Alert Intelligence)       │
│   - 알림 상관 관계 및 그룹핑                       │
│   - 노이즈 감소 (비실행 가능 항목 억제)             │
│   - 심각도 예측                                   │
├─────────────────────────────────────────────────┤
│ 계층 1: 이상 탐지(Anomaly Detection)             │
│   - 기준선 학습 ("정상"이란 무엇인가?)             │
│   - 통계적 이상 탐지                              │
│   - 추세 감지 및 예측                             │
├─────────────────────────────────────────────────┤
│ 기반: 관측 가능성 데이터                           │
│   - 메트릭, 로그, 트레이스, 이벤트, 변경사항        │
└─────────────────────────────────────────────────┘
```

### 1.2 AIOps vs 전통적 모니터링

| 측면 | 전통적 모니터링 | AIOps |
|------|--------------|-------|
| **임계값** | 정적 (수동 구성) | 동적 (데이터에서 학습) |
| **기준선** | 없음 또는 수동 | 자동, 계절성 인식 |
| **알림 볼륨** | 높음 (많은 오탐) | 감소 (상관, 중복 제거) |
| **근본 원인** | 수동 조사 | 상관 관계와 토폴로지에 의해 보조 |
| **복구** | 수동 (런북 따르기) | 알려진 패턴에 대해 자동화 |
| **확장** | 100+ 서비스에서 한계 | 대규모 시스템용으로 설계 |

---

## 2. 이상 탐지 기법(Anomaly Detection Techniques)

### 2.1 통계적 방법

**이동 평균과 표준 편차(Moving Average with Standard Deviation):**

```python
"""이동 평균과 표준 편차를 사용한 간단한 이상 탐지."""
import numpy as np
from dataclasses import dataclass

@dataclass
class AnomalyResult:
    timestamp: float
    value: float
    expected: float
    lower_bound: float
    upper_bound: float
    is_anomaly: bool
    z_score: float

def detect_anomalies_zscore(
    values: list[float],
    window_size: int = 60,
    threshold_sigma: float = 3.0,
) -> list[AnomalyResult]:
    """롤링 통계를 사용한 z-스코어 기반 이상 탐지."""
    results = []
    for i in range(window_size, len(values)):
        window = values[i - window_size:i]
        mean = np.mean(window)
        std = np.std(window)

        if std == 0:
            std = 1e-10  # 0으로 나누기 방지

        z_score = (values[i] - mean) / std
        is_anomaly = abs(z_score) > threshold_sigma

        results.append(AnomalyResult(
            timestamp=i,
            value=values[i],
            expected=mean,
            lower_bound=mean - threshold_sigma * std,
            upper_bound=mean + threshold_sigma * std,
            is_anomaly=is_anomaly,
            z_score=z_score,
        ))
    return results

# 예시: 지연 시간 이상 탐지
latency_samples = [100, 102, 98, 105, 99, 101, 97, 103, 100, 98,
                   # ... 정상 트래픽 ...
                   500, 480, 520,  # ← 이상 (지연 급증)
                   101, 99, 102]  # ← 정상으로 복귀
```

### 2.2 계절 분해(Seasonal Decomposition)

많은 메트릭은 일별, 주별, 월별 패턴을 가집니다:

```python
"""STL 분해를 사용한 계절적 이상 탐지."""
from statsmodels.tsa.seasonal import STL

def detect_seasonal_anomalies(
    values: np.ndarray,
    period: int = 1440,      # 분 해상도 데이터에서 1440분 = 1일
    threshold_sigma: float = 3.0,
) -> np.ndarray:
    """계절 패턴을 고려한 이상 탐지."""
    # STL 분해: value = trend + seasonal + residual
    stl = STL(values, period=period, robust=True)
    result = stl.fit()

    # 이상은 잔차(residual)에 있음 (추세 + 계절 제거 후 남은 것)
    residual = result.resid
    residual_mean = np.mean(residual)
    residual_std = np.std(residual)

    # 잔차가 임계값을 초과하는 포인트가 이상
    z_scores = (residual - residual_mean) / residual_std
    is_anomaly = np.abs(z_scores) > threshold_sigma

    return is_anomaly
```

### 2.3 EWMA (지수 가중 이동 평균)

```python
"""EWMA 기반 이상 탐지: 최근 변화에 더 민감."""

def detect_anomalies_ewma(
    values: list[float],
    alpha: float = 0.1,        # 평활 계수 (0.01=느림, 0.5=빠름)
    threshold_sigma: float = 3.0,
) -> list[bool]:
    """EWMA를 사용한 이상 탐지."""
    ewma = values[0]
    ewma_var = 0.0
    anomalies = []

    for value in values:
        # EWMA 업데이트
        diff = value - ewma
        ewma = alpha * value + (1 - alpha) * ewma
        ewma_var = (1 - alpha) * (ewma_var + alpha * diff * diff)
        ewma_std = np.sqrt(ewma_var)

        # 이상 확인
        is_anomaly = abs(value - ewma) > threshold_sigma * ewma_std if ewma_std > 0 else False
        anomalies.append(is_anomaly)

    return anomalies
```

### 2.4 기법 비교

| 기법 | 장점 | 단점 | 최적 용도 |
|------|------|------|---------|
| **Z-스코어** | 단순, 빠름, 해석 가능 | 계절성 미인식 | 정상(stationary) 메트릭 |
| **EWMA** | 추세에 적응, 경량 | 급격한 변화에 느린 반응 | 점진적으로 변화하는 메트릭 |
| **STL 분해** | 계절성 처리 | 충분한 히스토리 필요 | 일별/주별 패턴 |
| **Isolation Forest** | 다변량 데이터 처리 | 해석성 낮음 | 다중 메트릭 이상 |
| **Prophet** | 추세 + 계절성 + 휴일 처리 | 더 무거움, 학습 필요 | 용량 계획 |

---

## 3. 지능형 알림(Intelligent Alerting)

### 3.1 Prometheus 동적 임계값

```yaml
# 정적 대신: expr: cpu_usage > 80
# 동적 기준선 사용: 평균 대비 3 표준 편차 초과 시 알림

groups:
  - name: dynamic_alerts
    rules:
      # 7일 기준선 통계 사전 계산
      - record: job:http_request_duration:avg_over_7d
        expr: avg_over_time(job:http_request_duration_seconds:p99[7d])

      - record: job:http_request_duration:stddev_over_7d
        expr: stddev_over_time(job:http_request_duration_seconds:p99[7d])

      # 현재 값 > 기준선 + 3 표준 편차일 때 알림
      - alert: LatencyAnomaly
        expr: |
          job:http_request_duration_seconds:p99
          > (job:http_request_duration:avg_over_7d + 3 * job:http_request_duration:stddev_over_7d)
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "지연 시간 이상 감지 (7일 기준선 대비 3 시그마 초과)"
          description: |
            현재 p99: {{ $value }}s
            기준선 평균: {{ with query "job:http_request_duration:avg_over_7d" }}{{ . | first | value }}{{ end }}s
```

### 3.2 알림 상관 관계(Alert Correlation)

관련 알림을 그룹핑하여 알림 폭풍을 줄입니다:

```yaml
# Alertmanager: 관련 알림 그룹핑
route:
  receiver: "default"
  group_by: ["cluster", "service"]   # 클러스터와 서비스별 알림 그룹핑
  group_wait: 30s                      # 관련 알림 수집을 위해 30초 대기
  group_interval: 5m                   # 그룹화된 알림 간 간격
  repeat_interval: 4h

  routes:
    # SEV1: 즉시, 개별 알림
    - match:
        severity: critical
      receiver: "pagerduty-critical"
      group_wait: 10s

    # SEV2+: 서비스별 그룹핑
    - match:
        severity: warning
      receiver: "slack-warnings"
      group_by: ["service", "alertname"]
      group_wait: 60s

# 억제(Inhibition): 높은 심각도 알림 발동 시 낮은 심각도 알림 억제
inhibit_rules:
  # 서비스에 대해 critical 알림 발동 시 같은 서비스의 warning 억제
  - source_matchers:
      - severity="critical"
    target_matchers:
      - severity="warning"
    equal: ["service"]

  # 전체 클러스터 다운 시 개별 서비스 알림 억제
  - source_matchers:
      - alertname="ClusterDown"
    target_matchers:
      - severity=~"warning|critical"
    equal: ["cluster"]
```

### 3.3 노이즈 감소 메트릭

| 메트릭 | AIOps 전 | AIOps 후 | 개선 |
|--------|---------|---------|------|
| 주간 알림 수 | 500 | 50 | 90% 감소 |
| 실행 가능 알림 | 25% | 85% | 3.4배 개선 |
| MTTD (평균 감지 시간) | 15분 | 3분 | 5배 빠름 |
| 오탐률(false positive) | 60% | 10% | 6배 감소 |

---

## 4. 변경-영향 상관 관계(Change-Impact Correlation)

### 4.1 배포와 이상 연결

```python
"""배포와 메트릭 이상을 상관시킴."""
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class Deployment:
    service: str
    version: str
    deployed_at: datetime
    deployer: str
    changed_files: list[str]

@dataclass
class Anomaly:
    metric: str
    service: str
    detected_at: datetime
    severity: str
    value: float
    baseline: float

def correlate_deployments_with_anomalies(
    deployments: list[Deployment],
    anomalies: list[Anomaly],
    correlation_window: timedelta = timedelta(minutes=30),
) -> list[tuple[Anomaly, list[Deployment]]]:
    """이상 직전에 발생한 배포를 찾음."""
    correlated = []

    for anomaly in anomalies:
        related_deployments = [
            d for d in deployments
            if d.deployed_at <= anomaly.detected_at
            and anomaly.detected_at - d.deployed_at <= correlation_window
            and (d.service == anomaly.service
                 or d.service in get_dependencies(anomaly.service))
        ]

        if related_deployments:
            correlated.append((anomaly, related_deployments))

    return correlated

# 사용 예:
# "14:15의 order-service 지연 이상은
#  14:02의 payment-service 배포와 상관됨 (13분 전)"
```

### 4.2 변경 상관 관계를 위한 Grafana 주석

```yaml
# Grafana 대시보드에 배포 이벤트 자동 주석
# deployment-annotator (Kubernetes admission webhook 또는 CI 단계로 실행)

annotations:
  - datasource: prometheus
    expr: |
      changes(kube_deployment_status_observed_generation{
        namespace="production"
      }[5m]) > 0
    name: "배포"
    color: "blue"
    tags: ["deployment"]

  - datasource: prometheus
    expr: |
      ALERTS{alertstate="firing", severity="critical"}
    name: "알림"
    color: "red"
    tags: ["alert"]
```

---

## 5. 자동 복구(Automated Remediation)

### 5.1 복구 안전 수준

| 수준 | 자동화 | 인간 역할 | 예시 |
|------|--------|---------|------|
| **L0: 수동** | 알림 발생 | 인간이 조사 및 수정 | 런북 수동 수행 |
| **L1: 보조** | 진단 명령 사전 실행 | 인간이 검토 및 승인 | 로그 자동 수집, 인간이 결정 |
| **L2: 감독** | 복구 준비, 인간 승인 | 인간이 "승인" 클릭 | 롤백 자동 준비, 인간이 승인 |
| **L3: 자동** | 가드레일 포함 완전 자동 | 사후 인간 통지 | 크래시된 파드 자동 재시작 (Kubernetes 기본) |
| **L4: 예측** | 문제 발생 전 조치 | 인간이 추세 모니터링 | 트래픽 급증 전 자동 스케일 |

### 5.2 안전한 자동 복구 패턴

```python
"""안전 가드레일을 갖춘 자동 복구."""
from datetime import datetime, timedelta

class RemediationGuardrails:
    """폭주 자동화 방지."""

    def __init__(self):
        self.actions_taken: list[dict] = []
        self.max_actions_per_hour = 3
        self.max_concurrent_remediations = 1
        self.cooldown_minutes = 15
        self.active_remediations = 0

    def can_remediate(self, action: str, service: str) -> tuple[bool, str]:
        """복구 실행이 안전한지 확인."""
        # 가드 1: 비율 제한
        recent = [a for a in self.actions_taken
                  if a["time"] > datetime.utcnow() - timedelta(hours=1)]
        if len(recent) >= self.max_actions_per_hour:
            return False, f"비율 제한: 지난 1시간 {len(recent)}/{self.max_actions_per_hour} 조치"

        # 가드 2: 같은 서비스에 대한 이전 조치 후 쿨다운
        same_service = [a for a in self.actions_taken
                        if a["service"] == service
                        and a["time"] > datetime.utcnow() - timedelta(minutes=self.cooldown_minutes)]
        if same_service:
            return False, f"쿨다운: {service}에 대한 마지막 조치 {same_service[-1]['time']}"

        # 가드 3: 동시 복구 없음
        if self.active_remediations >= self.max_concurrent_remediations:
            return False, f"동시 실행 제한: {self.active_remediations} 활성"

        # 가드 4: 업무 시간 확인 (피크 중 자동 복구 없음)
        hour = datetime.utcnow().hour
        if action == "rollback" and 9 <= hour <= 17:
            return False, "업무 시간 중 자동 롤백 비활성화 (수동 승인 필요)"

        return True, "OK"

    def execute(self, action: str, service: str, details: str):
        can, reason = self.can_remediate(action, service)
        if not can:
            notify_oncall(f"자동 복구 차단: {reason}. 수동 개입 필요.")
            return

        self.active_remediations += 1
        try:
            # 복구 실행
            result = run_remediation(action, service, details)
            self.actions_taken.append({
                "action": action, "service": service,
                "time": datetime.utcnow(), "result": result
            })
            # 통지
            notify_oncall(f"자동 복구 실행: {service}에 {action}. 결과: {result}")
        finally:
            self.active_remediations -= 1
```

### 5.3 일반적인 자동 복구 조치

| 트리거 | 조치 | 안전 확인 |
|--------|------|----------|
| 파드 크래시 루프 | 메모리 증가하여 파드 재시작 | 시간당 최대 3회 재시작 |
| 배포 후 높은 오류율 | 이전 버전으로 자동 롤백 | 배포 후 10분 이내일 때만 |
| 연결 풀 소진 | 애플리케이션 파드 재시작 (롤링) | 15분당 최대 1회 재시작 |
| 디스크 공간 > 90% | 오래된 로그와 임시 파일 삭제 | 데이터 디렉토리는 절대 삭제 금지 |
| 인증서 7일 내 만료 | cert-manager를 통해 자동 갱신 | 적용 전 새 인증서 확인 |
| 트래픽 급증 감지 | 레플리카 스케일 업 | 최대 스케일 팩터 3배; 초과 시 인간 승인 |

---

## 6. AIOps 플랫폼

### 6.1 플랫폼 현황

| 플랫폼 | 유형 | 핵심 기능 | 최적 대상 |
|--------|------|---------|---------|
| **Datadog** | SaaS | Watchdog 이상 탐지, 상관 관계 | 풀스택 SaaS 관측 가능성 |
| **Dynatrace** | SaaS | Davis AI 엔진, 자동 토폴로지 | 엔터프라이즈, Java 중심 환경 |
| **New Relic** | SaaS | Applied Intelligence, 이상 탐지 | APM 포함 풀스택 |
| **Grafana ML** | OSS/SaaS | 메트릭 예측, 이상 알림 | Prometheus/Grafana 사용자 |
| **Elastic** | OSS/SaaS | 로그와 메트릭의 ML 이상 탐지 | 로그 중심 환경 |

### 6.2 AIOps 주장 평가

| 주장 | 현실 확인 |
|------|---------|
| "우리 AI가 모든 이상을 감지합니다" | 모든 것을 잡는 시스템은 없음; 오탐률을 질문하세요 |
| "제로 구성 ML" | 모델은 여전히 조정 필요 (민감도, 학습 윈도우) |
| "자동 근본 원인 분석" | 보통 상관 관계이지 진정한 인과 분석은 아님; 인간 검증 필요 |
| "90% 알림 감소" | 종종 달성 가능하나 중복 제거와 그룹핑 포함, ML만은 아님 |
| "자가 치유 인프라" | 알려진 패턴(재시작, 스케일)에는 작동; 새로운 장애에는 여전히 인간 필요 |

---

## 7. 실용적 구현(Practical Implementation)

### 7.1 AIOps 시작하기 (단계별 접근)

```
Phase 1 (1-2개월): 기반
  - 깨끗하고 신뢰할 수 있는 메트릭 데이터 확보 (갭 수정, 명명 표준화)
  - SLO 기반 알림 구현 (레슨 20) -- 알림 볼륨 60-80% 감소
  - 대시보드에 배포 주석 추가
  → ML 없이도 가장 큰 영향

Phase 2 (3-4개월): 통계적 이상 탐지
  - 주요 SLI에 Grafana ML 예측 활성화
  - 정적 임계값을 동적 기준선으로 교체 (z-스코어 또는 EWMA)
  - Alertmanager에서 알림 상관 관계 구현
  → 오탐률 40-60% 감소

Phase 3 (5-6개월): 상관 관계 및 보조 복구
  - 변경-영향 상관 관계 배포 (배포 → 이상 연결)
  - L1-L2 자동 진단 구현 (로그, 트레이스 사전 수집)
  - 알려진 패턴에 대한 자동 복구 구현 (재시작, 스케일, 롤백)
  → MTTR 30-50% 감소

Phase 4 (7개월+): 고급 ML
  - 토폴로지 인식 RCA를 위한 AIOps 플랫폼 평가
  - 예측 스케일링 구현
  - 인시던트 히스토리에 대한 커스텀 모델 학습
  → 추가 20-30% MTTR 감소
```

---

## 8. 다음 단계

- [28_Capstone_Full_Stack_Observability.md](./28_Capstone_Full_Stack_Observability.md) -- 종합 관측 가능성 플랫폼 설계

---

## 연습 문제

### 연습 1: 이상 탐지 알고리즘

다음 지연 시간 데이터(밀리초)에 EWMA 기반 이상 탐지기를 구현하세요. alpha=0.1, threshold=3 sigma를 사용합니다. 어떤 데이터 포인트가 이상인지 식별하고 이유를 설명하세요.

```python
latency_data = [
    100, 102, 98, 105, 99, 101, 97, 103, 100, 98,  # 정상 (1-10분)
    102, 99, 101, 100, 103, 97, 105, 98, 100, 101,  # 정상 (11-20분)
    250, 280, 260,                                     # 급증 (21-23분)
    102, 100, 98, 101, 99,                             # 회복 (24-28분)
    100, 101, 99, 102, 100, 98, 103, 101, 100, 99,   # 정상 (29-38분)
    115, 118, 120, 122, 125, 128, 130, 133, 135, 138, # 점진적 증가 (39-48분)
]
```

<details>
<summary>정답 보기</summary>

```python
import numpy as np

latency_data = [
    100, 102, 98, 105, 99, 101, 97, 103, 100, 98,
    102, 99, 101, 100, 103, 97, 105, 98, 100, 101,
    250, 280, 260,
    102, 100, 98, 101, 99,
    100, 101, 99, 102, 100, 98, 103, 101, 100, 99,
    115, 118, 120, 122, 125, 128, 130, 133, 135, 138,
]

alpha = 0.1
threshold_sigma = 3.0

ewma = latency_data[0]
ewma_var = 0.0
anomalies = []

for i, value in enumerate(latency_data):
    diff = value - ewma
    ewma = alpha * value + (1 - alpha) * ewma
    ewma_var = (1 - alpha) * (ewma_var + alpha * diff * diff)
    ewma_std = np.sqrt(ewma_var)

    is_anomaly = abs(value - ewma) > threshold_sigma * ewma_std if ewma_std > 0 else False

    if is_anomaly:
        anomalies.append((i, value, ewma, ewma_std))
        print(f"분 {i+1}: value={value}, ewma={ewma:.1f}, std={ewma_std:.1f}, 이상")
```

**예상 이상:**
- **21분 (value=250)**: ~100에서 250으로 급격한 급증. EWMA는 ~101, std는 ~2.5. Z-스코어 ≈ (250-101)/2.5 = 59.6 >> 3. 명확한 이상.
- **22분 (value=280)**: 더 높음. EWMA가 약간 상향 조정(~116)되었지만 여전히 280보다 훨씬 아래. 이상.
- **23분 (value=260)**: 여전히 EWMA 위에 있음. 이상.

**점진적 증가 (39-48분)**: 이상으로 표시되지 않을 가능성이 높습니다. 이유:
- EWMA가 점진적 증가를 추적 (각 포인트가 이전 대비 2-3ms만 증가)
- 표준 편차가 추세를 수용하기 위해 상향 조정
- 단일 포인트가 EWMA 대비 3 시그마를 초과하지 않음

이는 EWMA의 한계를 보여줍니다: 점진적 변화에 적응하여 느린 성능 저하를 놓칠 수 있습니다. 점진적 추세에는 추세 감지(예: EWMA 기울기의 선형 회귀) 또는 더 긴 기준선과 비교를 사용하세요.

</details>

### 연습 2: 알림 상관 관계 설계

50개 서비스를 관리하는 마이크로서비스 플랫폼에서 인시던트 중 5분 이내에 47개 알림을 수신했습니다. 관리 가능한 수의 실행 가능 알림으로 그룹핑하는 알림 상관 관계 전략을 설계하세요. 그룹핑 키, 시간 윈도우, 억제 규칙, 온콜 엔지니어에게 상관된 알림을 제시하는 방법을 지정하세요.

<details>
<summary>정답 보기</summary>

**알림 상관 관계 전략:**

**1. 그룹핑 구성:**
```yaml
route:
  group_by: ["cluster", "namespace", "service"]
  group_wait: 60s       # 전송 전 관련 알림 수집을 위해 60초 대기
  group_interval: 5m
```
47개 알림이 ~5-10개 그룹으로 축소 (영향받는 서비스/네임스페이스당 하나).

**2. 억제 규칙:**
```yaml
inhibit_rules:
  # 인프라 다운 시 애플리케이션 알림 억제
  - source_matchers:
      - alertname=~"NodeDown|ClusterUnreachable"
    target_matchers:
      - severity=~"warning|critical"
    equal: ["cluster"]

  # 데이터베이스 다운 시 의존 서비스 알림 억제
  - source_matchers:
      - alertname="DatabaseDown"
    target_matchers:
      - severity=~"warning|critical"
    equal: ["database_dependency"]

  # 같은 서비스에서 Critical이 warning 억제
  - source_matchers:
      - severity="critical"
    target_matchers:
      - severity="warning"
    equal: ["service"]
```

**3. 토폴로지 인식 그룹핑:**
서비스 의존성 그래프를 사용하여 루트 서비스를 식별합니다. `payment-service`가 먼저 알림을 발생시키고 `order-service`, `checkout-service`, `api-gateway`가 이후에 알림을 발생시키면, "payment-service 의존성 장애"로 그룹핑합니다.

**4. 온콜에게 제시:**

```
🚨 인시던트 알림 요약 (47개 알림이 3개 그룹으로 통합)

그룹 1: CRITICAL -- payment-service (근본 원인 후보)
  - PaymentServiceHighErrorRate (14:00:00)     ← 첫 번째 알림
  - PaymentServiceHighLatency (14:00:15)
  - PaymentServiceSLOBurnRate (14:00:30)
  관련 하위 알림: order-service (5개), checkout-service (3개)
  [대시보드 보기] [트레이스 보기] [런북]

그룹 2: WARNING -- 그룹 1에서 연쇄
  - OrderServiceHighErrorRate (14:01:00)
  - OrderServiceTimeouts (14:01:10)
  - CheckoutServiceUnavailable (14:01:30)
  ... 외 25개 (억제됨, 그룹 1이 원인)
  payment-service 장애의 영향. 그룹 1이 수정되면 해결될 것.

그룹 3: INFO -- 무관
  - DiskSpaceWarning on monitoring-node-3 (14:02:00)
  별도 문제, 그룹 1-2와 상관 관계 없음.
```

**결과: 47개 알림 → 3개 실행 가능 그룹. 온콜은 그룹 1에 집중.**

</details>

---

## 참고 자료

- [Moogsoft -- AIOps Platform](https://www.moogsoft.com/)
- [Datadog Watchdog](https://docs.datadoghq.com/watchdog/)
- [Grafana Machine Learning](https://grafana.com/docs/grafana-cloud/alerting-and-irm/machine-learning/)
- [Google SRE Book -- Practical Alerting](https://sre.google/sre-book/practical-alerting/)
- [Statistical Anomaly Detection (Netflix Blog)](https://netflixtechblog.com/rad-outlier-detection-on-big-data-d6b0494371cc)
- [Chaos Engineering + AIOps (Gremlin)](https://www.gremlin.com/blog/aiops-and-chaos-engineering/)
