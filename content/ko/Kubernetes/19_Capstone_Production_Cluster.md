# 19. 캡스톤: 프로덕션 클러스터(Capstone: Production Cluster)

**이전**: [18. ML을 위한 Kubernetes](./18_Kubernetes_for_ML.md) | **다음**: [00. 개요](./00_Overview.md)

## 학습 목표

- 요구사항을 수집하고 Kubernetes 아키텍처 결정으로 변환
- 적절한 노드 풀, 네트워킹, 스토리지를 갖춘 고가용성(HA) 프로덕션 클러스터 설계
- 보안 강화, 옵저버빌리티, CI/CD 통합 구현
- 자동화된 백업과 테스트된 복원 절차로 재해 복구 설정
- 안정성과 성능 SLO를 유지하면서 비용 최적화

---

이 캡스톤 프로젝트는 이전 18개 레슨의 모든 내용을 하나의 통합된 연습으로 결합합니다: 프로덕션 수준의 Kubernetes 클러스터 설계, 배포, 검증. 요구사항 수집, 아키텍처 설계, 보안 강화, 옵저버빌리티, CI/CD 통합, 재해 복구, 비용 최적화를 순서대로 진행합니다. 각 섹션은 이전 섹션을 기반으로 하며, 완전한 프로덕션 플랫폼으로 마무리됩니다.

## 목차

- [이론과 원리](#이론과-원리)
- [1. 요구사항 수집](#1-요구사항-수집)
- [2. 아키텍처 설계](#2-아키텍처-설계)
- [3. HA 컨트롤 플레인](#3-ha-컨트롤-플레인)
- [4. 노드 풀 설계](#4-노드-풀-설계)
- [5. 네트워킹 아키텍처](#5-네트워킹-아키텍처)
- [6. 보안 강화](#6-보안-강화)
- [7. 옵저버빌리티 스택](#7-옵저버빌리티-스택)
- [8. CI/CD 파이프라인 통합](#8-cicd-파이프라인-통합)
- [9. 재해 복구 설정](#9-재해-복구-설정)
- [10. 비용 최적화](#10-비용-최적화)
- [연습문제](#연습문제)

---

## 1. 요구사항 수집

### 이론: 상호 타협 — 모든 선택은 숨은 비용을 가진다

각 기술 결정은 어딘가에 상호 요구를 만듭니다:

- **Istio service mesh 선택** → mTLS, 트래픽 시프팅, 관측 가능성 획득 → 추가 지연(홉당 약 1ms), 사이드카 메모리(파드당 약 100MB), 그리고 운영할 새 컨트롤 플레인으로 지불.
- **VM 워크로드용 KubeVirt 선택** → 통합 컴퓨트 플랫폼 획득 → 사소하지 않은 KubeVirt operator 복잡성과 vanilla 쿠버네티스와 다른 런타임 모델로 지불.
- **Hub-spoke 멀티 클러스터 선택** (15강) → 중앙화된 플랫폼 서비스 획득 → 자체 HA 이야기가 필요한 critical hub 클러스터로 지불.
- **엄격한 Pod Security restricted 프로파일 선택** → 컴플라이언스 + 공격 표면 감소 획득 → 예외가 필요한 워크로드와 그것들을 승인하는 운영 오버헤드로 지불.
- **불변 인프라(업그레이드를 위해 노드 재빌드) 선택** → 깨끗한 업그레이드와 재현성 획득 → 더 긴 업그레이드 윈도우와 전환 동안 더 많은 클라우드 지출로 지불.

"공짜 점심 없음" 속성은 모든 설계 리뷰가 — *이 선택이 우리에게 다른 곳에서 무엇을 비용으로 부담시키는가?* — 를 물어야 함을 의미합니다. 답이 "아무것도"라면, 충분히 깊이 보지 않은 것입니다. 프로덕션 품질 설계는 비용에 대해 정직하고 *이 조직에 대해* 선택된 트레이드오프가 왜 옳은지에 대해 명시적입니다.

### 이론: 진단 렌즈 — 결정하기 전에 스트레스 테스트

좋은 설계는 질문에서 살아남습니다. 아키텍처에 사인오프하기 전에, 이 시나리오들을 걸어보세요:

- **etcd가 손상되면?** 테스트된 복원이 있는가? RPO/RTO? (17강 §B.)
- **한 지역이 오프라인이 되면?** 트래픽이 자동으로 우회되는가? RPO/RTO가 비즈니스 요구를 만족하는가?
- **보안 정책이 클러스터 전역으로 변경되어야 하면?** 단일 적용 소스(GitOps, 15강)가 있는가, 아니면 50개 클러스터에 로그인하는가?
- **개발자가 실수로 privileged 파드를 배포하면?** 어드미션(12강)이 그것을 강제하는가, 아니면 audit만 되는가?
- **트래픽이 10×되면?** HPA가 워크로드를 스케일하는가? Cluster Autoscaler가 노드를 스케일하는가? 메트릭 파이프라인(13강 §B)이 따라잡는가?
- **핵심 엔지니어가 휴가 중이면?** Runbook이 문서화되어 있는가? 그 사람 없이 온콜이 실행할 수 있는가?
- **AWS가 인스턴스 유형을 deprecate하면?** 노드가 swap할 만큼 충분히 불변인가? 특정 하드웨어에 hard 결합이 있는가?
- **파드가 popped(RCE)되면?** Blast radius가 NetworkPolicy, ServiceAccount 권한, Pod Security로 제한되는가? (6, 8강.)

각 "what if"는 한 가정을 스트레스 테스트합니다. 이 모든 것에 — 불완전하더라도 — 답을 가진 설계는 프로덕션급입니다. "나중에 알아낼 것"이라 말하는 설계는 가장 나쁜 순간에 표면화될 숨은 위험을 가집니다.

캡스톤 연습을 위한 렌즈 — 설계의 모든 섹션(HA 컨트롤 플레인, 노드 풀, 네트워킹, 보안, 관측 가능성, CI/CD, DR, 비용)은 이런 종류의 질문에 대해 방어 가능해야 합니다.

### 1.1 이해관계자 질문

클러스터를 설계하기 전에 워크로드, 제약 조건, 기대사항을 이해해야 합니다:

```
Requirements Matrix:
┌───────────────────────────────────────────────────────────┐
│  카테고리            │ 답변할 질문                          │
├──────────────────────┼────────────────────────────────────┤
│  워크로드            │ 애플리케이션 수?                     │
│                      │ 스테이트리스 대 스테이트풀 비율?      │
│                      │ CPU 바운드 대 메모리 바운드 대 GPU?   │
│                      │ 예상 파드 수?                        │
│                      │ 배치 작업 대 장기 실행?               │
├──────────────────────┼────────────────────────────────────┤
│  규모                │ 예상 요청률 (RPS)?                   │
│                      │ 피크 대 평균 트래픽 비율?             │
│                      │ 성장률 (6/12/24개월)?                │
│                      │ 테넌트 팀 수?                        │
├──────────────────────┼────────────────────────────────────┤
│  가용성              │ 목표 업타임 SLO?                     │
│                      │ 허용 가능한 RTO와 RPO?               │
│                      │ 멀티 리전 요구사항?                   │
│                      │ 유지보수 윈도우 정책?                 │
├──────────────────────┼────────────────────────────────────┤
│  규정 준수           │ 데이터 상주 요구사항?                 │
│                      │ 저장 시 및 전송 중 암호화?            │
│                      │ 감사 로깅 요구사항?                   │
│                      │ 네트워크 세그먼테이션 필요?            │
├──────────────────────┼────────────────────────────────────┤
│  예산                │ 월간 인프라 예산?                     │
│                      │ 환경별 비용 제한?                     │
│                      │ 스팟/선점형 허용도?                   │
│                      │ 예약 인스턴스 약정?                   │
└──────────────────────┴────────────────────────────────────┘
```

### 1.2 참조 시나리오

이 캡스톤에서는 다음 시나리오를 사용합니다:

```yaml
# capstone-requirements.yaml
company: "TechCorp"
platform: "e-commerce + ML recommendation engine"

workloads:
  applications: 25
  microservices: 18
  stateful_services: 4          # PostgreSQL, Redis, Elasticsearch, Kafka
  ml_workloads: 3               # training, serving, pipelines
  total_pods_peak: 500
  gpu_requirement: true         # ML용 8x A100

scale:
  peak_rps: 10000
  average_rps: 3000
  peak_to_average_ratio: 3.3
  growth_rate_annual: 40%
  teams: 5                      # platform, backend, frontend, data, ML

availability:
  slo: "99.95%"
  rto: "30 minutes"
  rpo: "1 hour"
  multi_region: false           # 단일 리전, 멀티 AZ
  maintenance_window: "Sunday 02:00-06:00 UTC"

compliance:
  data_residency: "US"
  encryption_at_rest: true
  encryption_in_transit: true
  audit_logging: true
  network_segmentation: true    # 팀별 네임스페이스 격리

budget:
  monthly_limit: "$25,000"
  spot_tolerance: "training jobs only"
  reserved_instances: "control plane + system nodes"
```

---

## 2. 아키텍처 설계

### 이론: 계층 설계 모델

프로덕션 클러스터는 각각 아래 위에 빌드되는 네 계층으로 분해됩니다:

**계층 1 — 기반.** 클러스터 자체 — 컨트롤 플레인 HA(17강), 노드 풀, 네트워킹 모델(8강), storage class(4강), DNS, ingress 컨트롤러(7강). 이는 당신의 "OS"입니다 — 롤아웃 후 거의 변경하지 않아야 하며, 여기서의 변경은 가장 넓은 blast radius를 가집니다.

**계층 2 — 플랫폼 서비스.** 모든 워크로드 *를 위해* 실행되는 것 — 관측 가능성(14강), GitOps 컨트롤러(15강), 시크릿 관리(5강), 정책 강제(12강), 백업/복원 도구(17강). 이들은 워크로드 팀이 소비하지만 운영하지 않는 inner-platform입니다.

**계층 3 — 워크로드.** Service와 Ingress(3, 7강)를 통해 노출되는 애플리케이션 Deployment, StatefulSet, Job(2강). 클러스터의 사용자 가시 가치가 여기에 삽니다.

**계층 4 — Day-2 운영.** SLO 정의, runbook, 온콜 로테이션, 용량 계획, 변경 관리(17강). YAML에 있지 않지만 클러스터가 "프로덕션"인지에 그만큼 중요합니다.

이 계층화가 중요한 이유 — **하위 계층의 변경은 더 큰 blast radius를 가집니다** — 그에 따라 설계해야 합니다. 워크로드 배포가 잘못되면 한 앱을 다운시킵니다 — CNI 업그레이드가 잘못되면 모든 것을 다운시킵니다. 멀티 클러스터 전략(15강)으로 계층-1 변경을 계획하고, 더 낮은 환경에서 먼저 테스트하고, 안전을 위해 더 느린 반복을 수용하세요.

### 이론: 트레이드오프 삼각형 — 비용, 가용성, 변경 속도

모든 설계 결정은 3-방향 트레이드오프에 앉습니다:

- **비용.** 월간 클러스터 지출 — 노드 시간, 스토리지, 관측 가능성 ingest, 관리형 서비스 요금.
- **가용성.** 효과적 가동 시간 — 다중 AZ, 다중 지역, 모든 계층의 중복.
- **변경 속도.** 안전하게 출시할 수 있는 비율 — CI/CD 처리량, commit-to-prod 시간, 테스트 커버리지.

어떤 두 개든 최적화할 수 있지만 셋 모두는 안 됩니다:

- **비용 + 가용성** without 속도 — 모든 변경을 수동으로 승인하는 작은 hyper-stable 플랫폼 팀. 은행. 느리지만 저렴하고 신뢰할 수 있음.
- **비용 + 속도** without 가용성 — 최소 중복, 빠르게 출시, 사고 수용. 초기 단계 스타트업.
- **가용성 + 속도** without 낮은 비용 — 완전 다중 지역 active-active, 모든 것의 자동화된 canary, 견고한 관측 가능성. 현대 SaaS.

이 삼각형을 인식하면 논쟁을 막습니다. "왜 그냥 다중 지역으로 배포하지 않나?" → "비용을 두 배로 만들고 우리는 속도를 우선시했기 때문". "왜 배포 파이프라인이 그렇게 느린가?" → "가용성을 우선시하고 승인을 추가했기 때문". 트레이드오프를 명시적으로 만들고 — 리더십이 코너를 고르게 하세요.

삼각형은 구체적으로 펼쳐집니다:

| 결정 | 비용 ↑ | 가용성 ↑ | 속도 ↑ |
|------|--------|---------|--------|
| 다중 지역 클러스터 | + + + | + + + | – |
| Spot 전용 노드 | – – | – | 0 |
| Service mesh | + | + + | – |
| 엄격한 어드미션 정책 | 0 | + | – – |
| 모든 곳에서의 오토스케일링 | – | + + | + |

보편적으로 옳은 셀은 없습니다 — *당신의* 제약에 맞는 셀이 있습니다.

### 2.1 상위 수준 아키텍처

```
Production Cluster Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                         VPC (10.0.0.0/16)                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              Public Subnets (3 AZs)                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │    │
│  │  │ NAT GW   │  │ NAT GW   │  │ NAT GW   │               │    │
│  │  │ AZ-a     │  │ AZ-b     │  │ AZ-c     │               │    │
│  │  └──────────┘  └──────────┘  └──────────┘               │    │
│  │  ┌──────────────────────────────────────┐                │    │
│  │  │        NLB (API Server endpoint)     │                │    │
│  │  └──────────────────────────────────────┘                │    │
│  │  ┌──────────────────────────────────────┐                │    │
│  │  │        ALB (Ingress traffic)         │                │    │
│  │  └──────────────────────────────────────┘                │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              Private Subnets (3 AZs)                      │    │
│  │                                                           │    │
│  │  Control Plane (HA):                                      │    │
│  │  ┌────────┐  ┌────────┐  ┌────────┐                      │    │
│  │  │ CP-1   │  │ CP-2   │  │ CP-3   │                      │    │
│  │  │ AZ-a   │  │ AZ-b   │  │ AZ-c   │                      │    │
│  │  │ etcd   │  │ etcd   │  │ etcd   │                      │    │
│  │  └────────┘  └────────┘  └────────┘                      │    │
│  │                                                           │    │
│  │  Worker Nodes:                                            │    │
│  │  ┌──────────────────────────────────────────────────┐     │    │
│  │  │ System Pool (3x m5.xlarge, reserved)             │     │    │
│  │  │ General Pool (5-20x m5.2xlarge, on-demand)       │     │    │
│  │  │ Stateful Pool (3x r5.2xlarge, on-demand)         │     │    │
│  │  │ GPU Pool (2-8x p4d.24xlarge, spot + on-demand)   │     │    │
│  │  └──────────────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Supporting Services                                      │    │
│  │  ├── ECR (container registry)                             │    │
│  │  ├── S3 (backups, ML models, logs)                        │    │
│  │  ├── Route 53 (DNS)                                       │    │
│  │  ├── ACM (TLS certificates)                               │    │
│  │  └── CloudWatch (audit logs)                              │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 코드로서의 인프라(Infrastructure as Code)

```bash
# Terraform 프로젝트 구조
terraform/
├── modules/
│   ├── vpc/
│   ├── eks/
│   ├── observability/
│   └── security/
├── environments/
│   ├── production/
│   └── staging/
└── README.md
```

---

## 3. HA 컨트롤 플레인

### 3.1 컨트롤 플레인 구성

```yaml
# kubeadm HA 컨트롤 플레인 구성
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
kubernetesVersion: v1.29.2
controlPlaneEndpoint: "api.prod.example.com:6443"    # 로드 밸런서
networking:
  podSubnet: "10.244.0.0/16"
  serviceSubnet: "10.96.0.0/12"
  dnsDomain: "cluster.local"
apiServer:
  extraArgs:
    audit-policy-file: /etc/kubernetes/audit-policy.yaml
    audit-log-path: /var/log/kubernetes/audit.log
    encryption-provider-config: /etc/kubernetes/encryption-config.yaml
    enable-admission-plugins: >-
      NodeRestriction,
      PodSecurity,
      ResourceQuota,
      LimitRanger,
      ServiceAccount
    max-requests-inflight: "800"
    max-mutating-requests-inflight: "400"
etcd:
  local:
    extraArgs:
      quota-backend-bytes: "8589934592"       # 8GB
      auto-compaction-mode: periodic
      auto-compaction-retention: "8h"
```

### 3.2 저장 시 etcd 암호화

```yaml
# /etc/kubernetes/encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
      - configmaps
    providers:
      - aescbc:
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-key>
      - identity: {}    # 암호화되지 않은 데이터 읽기를 위한 폴백
```

### 3.3 API 서버 감사 정책

```yaml
# /etc/kubernetes/audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  # 특정 비민감 엔드포인트에 대한 요청은 기록하지 않음
  - level: None
    nonResourceURLs:
      - "/healthz*"
      - "/readyz*"
      - "/livez*"
      - "/metrics"

  # 감시(watch) 요청은 기록하지 않음 (너무 많은 로그)
  - level: None
    verbs: ["watch"]

  # 시크릿 접근을 메타데이터 레벨로 기록 (시크릿 내용은 기록하지 않음)
  - level: Metadata
    resources:
      - group: ""
        resources: ["secrets"]
    omitStages:
      - RequestReceived

  # 다른 모든 요청은 RequestResponse 레벨로 기록
  - level: RequestResponse
    resources:
      - group: ""
        resources: ["pods", "services", "configmaps"]
      - group: "apps"
        resources: ["deployments", "statefulsets"]
      - group: "rbac.authorization.k8s.io"
        resources: ["roles", "rolebindings", "clusterroles", "clusterrolebindings"]
    omitStages:
      - RequestReceived

  # 모두 포함하는 메타데이터 레벨
  - level: Metadata
    omitStages:
      - RequestReceived
```

---

## 4. 노드 풀 설계

### 4.1 노드 풀 스펙

```yaml
# EKS 관리형 노드 그룹
node_pools:
  # 시스템 컴포넌트 (모니터링, 인그레스, DNS)
  system:
    instance_types: ["m5.xlarge"]          # 4 vCPU, 16 GB
    capacity_type: ON_DEMAND
    desired: 3
    min: 3
    max: 3
    labels:
      node-role: system
    taints:
      - key: node-role
        value: system
        effect: NoSchedule

  # 범용 워크로드
  general:
    instance_types: ["m5.2xlarge"]         # 8 vCPU, 32 GB
    capacity_type: ON_DEMAND
    desired: 5
    min: 3
    max: 20
    labels:
      node-role: general

  # 스테이트풀 워크로드 (데이터베이스, 캐시)
  stateful:
    instance_types: ["r5.2xlarge"]         # 8 vCPU, 64 GB
    capacity_type: ON_DEMAND
    desired: 3
    min: 3
    max: 6
    labels:
      node-role: stateful
    taints:
      - key: workload-type
        value: stateful
        effect: NoSchedule

  # ML용 GPU 노드
  gpu:
    instance_types: ["p4d.24xlarge"]       # 8x A100, 96 vCPU, 1.1 TB
    capacity_type: ON_DEMAND
    desired: 1
    min: 0
    max: 4
    labels:
      node-role: gpu
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule

  # 학습용 GPU 스팟 노드
  gpu_spot:
    instance_types: ["p3.8xlarge", "p3.16xlarge"]
    capacity_type: SPOT
    desired: 0
    min: 0
    max: 8
    labels:
      node-role: gpu-spot
      instance-lifecycle: spot
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
      - key: instance-lifecycle
        value: spot
        effect: NoSchedule
```

### 4.2 클러스터 오토스케일러 구성

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      nodeSelector:
        node-role: system
      tolerations:
        - key: node-role
          value: system
          effect: NoSchedule
      containers:
        - name: cluster-autoscaler
          image: registry.k8s.io/autoscaling/cluster-autoscaler:v1.29.0
          command:
            - ./cluster-autoscaler
            - --v=4
            - --cloud-provider=aws
            - --skip-nodes-with-local-storage=false
            - --expander=priority          # 우선순위 기반 확장기 사용
            - --scale-down-delay-after-add=10m
            - --scale-down-unneeded-time=10m
            - --scale-down-utilization-threshold=0.5
            - --max-graceful-termination-sec=600
            - --balance-similar-node-groups=true
            - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/prod-cluster
          resources:
            requests:
              cpu: 100m
              memory: 300Mi
            limits:
              cpu: 500m
              memory: 600Mi
```

---

## 5. 네트워킹 아키텍처

### 5.1 CNI와 네트워크 정책

```yaml
# Cilium CNI 구성 (Helm 값)
cilium:
  ipam:
    mode: eni                    # AWS ENI 모드
  eni:
    enabled: true
    awsEnablePrefixDelegation: true
  tunnel: disabled               # 네이티브 라우팅 (오버레이 없음)
  enableIPv4Masquerade: true
  policyEnforcementMode: default
  hubble:
    enabled: true
    relay:
      enabled: true
    ui:
      enabled: true
  encryption:
    enabled: true
    type: wireguard
```

### 5.2 인그레스 아키텍처

```yaml
# AWS Load Balancer Controller + 인그레스
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: production-ingress
  namespace: production
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-east-1:123456:certificate/abc-123
    alb.ingress.kubernetes.io/ssl-policy: ELBSecurityPolicy-TLS13-1-2-2021-06
    alb.ingress.kubernetes.io/wafv2-acl-arn: arn:aws:wafv2:us-east-1:123456:regional/webacl/prod
spec:
  ingressClassName: alb
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-server
                port:
                  number: 80
```

### 5.3 네임스페이스 격리를 위한 네트워크 정책

```yaml
# production 네임스페이스에서 기본 모든 인그레스 차단
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
---
# 인그레스 컨트롤러에서만 인그레스 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-ingress
  namespace: production
spec:
  podSelector:
    matchLabels:
      exposure: external
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
          podSelector:
            matchLabels:
              app.kubernetes.io/name: aws-load-balancer-controller
---
# 같은 네임스페이스 내 서비스 간 통신 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-same-namespace
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector: {}
---
# 모니터링 네임스페이스가 메트릭을 스크랩할 수 있도록 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-monitoring-scrape
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
      ports:
        - port: metrics
          protocol: TCP
```

---

## 6. 보안 강화

### 6.1 보안 체크리스트

```
프로덕션 보안 강화 체크리스트:
┌──────────────────────────────────────────────────────────────┐
│  ✓  컨트롤 플레인                                            │
│  ├── [ ] API 서버는 프라이빗 엔드포인트를 통해서만 접근 가능 │
│  ├── [ ] 감사 로깅 활성화 및 SIEM으로 전송                   │
│  ├── [ ] etcd 저장 시 암호화 (secrets, configmaps)           │
│  ├── [ ] 최소 권한 ServiceAccount으로 RBAC 설정              │
│  ├── [ ] 어드미션 컨트롤러: PodSecurity, OPA/Kyverno        │
│  └── [ ] 인증서 자동 갱신 활성화                             │
│                                                              │
│  ✓  워크로드                                                 │
│  ├── [ ] Pod Security Standards 적용 (restricted)            │
│  ├── [ ] 권한 있는 컨테이너 없음                             │
│  ├── [ ] 읽기 전용 루트 파일시스템                           │
│  ├── [ ] 모든 컨테이너에서 비루트 사용자                     │
│  ├── [ ] 모든 파드에 리소스 제한 설정                        │
│  ├── [ ] 프라이빗 레지스트리에서만 이미지 풀                 │
│  └── [ ] CI/CD에서 이미지 취약점 스캔                        │
│                                                              │
│  ✓  네트워크                                                 │
│  ├── [ ] 네임스페이스별 기본 거부 NetworkPolicy              │
│  ├── [ ] 전송 중 암호화 (서비스 메시를 통한 mTLS)            │
│  ├── [ ] WAF 및 DDoS 보호가 있는 인그레스                    │
│  ├── [ ] 민감한 네임스페이스에 대한 이그레스 제어            │
│  └── [ ] DNS 정책 (DB 파드에 외부 해석 없음)                 │
│                                                              │
│  ✓  공급망                                                   │
│  ├── [ ] 서명된 컨테이너 이미지 (Sigstore/cosign)            │
│  ├── [ ] 모든 이미지에 SBOM 생성                             │
│  ├── [ ] 이미지 서명 검증을 위한 어드미션 웹훅               │
│  └── [ ] 베이스 이미지를 태그가 아닌 다이제스트로 고정       │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 파드 보안 표준(Pod Security Standards)

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.29
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### 6.3 네트워크 정책

```yaml
# 기본 모든 인그레스 거부
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  egress:
    - to: []
      ports:
        - port: 53
          protocol: UDP
        - port: 53
          protocol: TCP
    - to:
        - podSelector: {}
```

### 6.4 RBAC 구성

```yaml
# 팀 수준 RBAC: 백엔드 팀이 자신의 네임스페이스를 관리할 수 있음
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: backend-developer
  namespace: backend
rules:
  - apiGroups: ["", "apps", "batch"]
    resources: ["pods", "deployments", "services", "configmaps", "jobs", "cronjobs"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["pods/log", "pods/exec"]
    verbs: ["get", "create"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get", "list"]         # 읽기는 가능하지만 생성/수정은 불가
  - apiGroups: ["networking.k8s.io"]
    resources: ["ingresses"]
    verbs: ["get", "list", "watch", "create", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: backend-developer-binding
  namespace: backend
subjects:
  - kind: Group
    name: "backend-developers"
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: backend-developer
  apiGroup: rbac.authorization.k8s.io
---
# 온콜 엔지니어를 위한 읽기 전용 클러스터 역할
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: oncall-readonly
rules:
  - apiGroups: ["", "apps", "batch", "networking.k8s.io"]
    resources: ["*"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
```

### 6.5 Kyverno를 사용한 이미지 보안

```yaml
# Kyverno 정책: 프라이빗 레지스트리의 이미지 필요
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-private-registry
spec:
  validationFailureAction: Enforce
  background: true
  rules:
    - name: validate-image-registry
      match:
        any:
          - resources:
              kinds:
                - Pod
              namespaces:
                - production
                - staging
      validate:
        message: "Images must come from the private registry"
        pattern:
          spec:
            containers:
              - image: "registry.example.com/*"
            initContainers:
              - image: "registry.example.com/*"
---
# 이미지 다이제스트 필요 (태그만이 아닌)
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-image-digest
spec:
  validationFailureAction: Enforce
  rules:
    - name: check-digest
      match:
        any:
          - resources:
              kinds:
                - Pod
              namespaces:
                - production
      validate:
        message: "Production images must reference a digest (@sha256:...)"
        pattern:
          spec:
            containers:
              - image: "*@sha256:*"
```

---

## 7. 옵저버빌리티 스택

### 7.1 Prometheus + Thanos

```yaml
# Thanos 사이드카가 있는 Prometheus
prometheus:
  prometheusSpec:
    retention: 15d
    replicas: 2
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          resources:
            requests:
              storage: 100Gi
    thanos:
      image: quay.io/thanos/thanos:v0.34.0
      objectStorageConfig:
        existingSecret:
          name: thanos-objstore-config
          key: objstore.yml
```

### 7.2 Prometheus 스택 배포

```yaml
# kube-prometheus-stack Helm 값
prometheus:
  prometheusSpec:
    retention: 15d
    retentionSize: 50GB
    resources:
      requests:
        cpu: "1"
        memory: 4Gi
      limits:
        cpu: "2"
        memory: 8Gi
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          resources:
            requests:
              storage: 100Gi
    nodeSelector:
      node-role: system
    tolerations:
      - key: node-role
        value: system
        effect: NoSchedule
    # 장기 스토리지를 위한 Thanos 사이드카
    thanos:
      objectStorageConfig:
        existingSecret:
          name: thanos-s3-config
          key: objstore.yml

alertmanager:
  alertmanagerSpec:
    resources:
      requests:
        cpu: 100m
        memory: 256Mi
  config:
    route:
      receiver: "null"
      group_by: ["alertname", "namespace"]
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 4h
      routes:
        - receiver: pagerduty-critical
          match:
            severity: critical
          continue: true
        - receiver: slack-warnings
          match:
            severity: warning
    receivers:
      - name: "null"
      - name: pagerduty-critical
        pagerduty_configs:
          - service_key_file: /etc/alertmanager/secrets/pagerduty-key
      - name: slack-warnings
        slack_configs:
          - api_url_file: /etc/alertmanager/secrets/slack-webhook
            channel: "#k8s-alerts"
            title: "{{ .GroupLabels.alertname }}"
            text: "{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}"

grafana:
  adminPassword: <from-secret>
  persistence:
    enabled: true
    size: 10Gi
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
        - name: default
          folder: Kubernetes
          type: file
          options:
            path: /var/lib/grafana/dashboards
```

### 7.3 Fluent Bit와 Loki를 사용한 로깅

```yaml
# Fluent Bit DaemonSet 구성
fluent-bit:
  config:
    inputs: |
      [INPUT]
          Name              tail
          Tag               kube.*
          Path              /var/log/containers/*.log
          Parser            cri
          DB                /var/log/flb_kube.db
          Mem_Buf_Limit     50MB
          Skip_Long_Lines   On
          Refresh_Interval  10

    filters: |
      [FILTER]
          Name                kubernetes
          Match               kube.*
          Merge_Log           On
          Keep_Log            Off
          K8S-Logging.Parser  On
          K8S-Logging.Exclude On

    outputs: |
      [OUTPUT]
          Name          loki
          Match         kube.*
          Host          loki-gateway.monitoring.svc
          Port          80
          Labels        job=fluent-bit
          Auto_Kubernetes_Labels On

  tolerations:
    - operator: Exists    # GPU를 포함한 모든 노드에서 실행
```

---

## 8. CI/CD 파이프라인 통합

### 8.1 ArgoCD를 사용한 GitOps

```yaml
# 프로덕션 배포를 위한 ArgoCD Application
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: production-apps
  namespace: argocd
spec:
  project: production
  source:
    repoURL: https://github.com/techcorp/k8s-manifests.git
    targetRevision: main
    path: production/
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

### 8.2 이미지 빌드가 포함된 CI 파이프라인

```yaml
# GitHub Actions CI 파이프라인
# .github/workflows/ci.yaml
name: CI Pipeline
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write          # AWS와의 OIDC용
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456:role/github-actions
          aws-region: us-east-1

      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and scan image
        run: |
          IMAGE_TAG="${{ github.sha }}"
          docker build -t $ECR_REGISTRY/app:$IMAGE_TAG .

          # Trivy로 취약점 스캔
          trivy image --exit-code 1 --severity HIGH,CRITICAL \
            $ECR_REGISTRY/app:$IMAGE_TAG

      - name: Push image
        run: |
          IMAGE_TAG="${{ github.sha }}"
          docker push $ECR_REGISTRY/app:$IMAGE_TAG

          # cosign으로 이미지 서명
          cosign sign --yes $ECR_REGISTRY/app:$IMAGE_TAG

      - name: Update Kubernetes manifest
        run: |
          IMAGE_TAG="${{ github.sha }}"
          cd k8s-manifests
          kustomize edit set image app=$ECR_REGISTRY/app:$IMAGE_TAG
          git add .
          git commit -m "Deploy app:$IMAGE_TAG"
          git push
```

### 8.3 Argo Rollouts를 사용한 점진적 배포

```yaml
# 카나리 배포
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: api-server
  namespace: production
spec:
  replicas: 10
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
        - name: api
          image: registry.example.com/api:v2.0@sha256:abc123
          ports:
            - containerPort: 8080
  strategy:
    canary:
      steps:
        - setWeight: 5
        - pause: {duration: 5m}
        - setWeight: 20
        - pause: {duration: 10m}
        - setWeight: 50
        - pause: {duration: 10m}
        - setWeight: 80
        - pause: {duration: 5m}
      analysis:
        templates:
          - templateName: success-rate
        startingStep: 1
```

---

## 9. 재해 복구 설정

### 9.1 자동화된 백업 시스템

```yaml
# 프로덕션용 Velero 백업 스케줄
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: production-backup
  namespace: velero
spec:
  schedule: "0 */4 * * *"       # 4시간마다
  template:
    includedNamespaces:
      - production
      - backend
      - data-platform
    snapshotVolumes: true
    ttl: 720h                    # 30일 보존
```

### 9.2 DR 검증 테스트

```bash
#!/usr/bin/env bash
# dr-drill.sh - 분기별 DR 검증 드릴

set -euo pipefail

DRILL_NS="dr-drill-$(date +%s)"
REPORT_FILE="/tmp/dr-drill-report-$(date +%Y%m%d).md"

echo "# DR 드릴 보고서 - $(date)" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "## 1. 백업 검증" >> "$REPORT_FILE"

# Velero 백업이 존재하고 최신인지 확인
LATEST_BACKUP=$(velero backup get --output json | \
  jq -r '.items | sort_by(.status.completionTimestamp) | last | .metadata.name')
BACKUP_AGE=$(velero backup get "$LATEST_BACKUP" -o json | \
  jq -r '.status.completionTimestamp')
echo "- 최신 백업: $LATEST_BACKUP ($BACKUP_AGE)" >> "$REPORT_FILE"

# S3의 etcd 백업 확인
ETCD_BACKUP=$(aws s3 ls s3://prod-backups/etcd/ --recursive | tail -1)
echo "- 최신 etcd 백업: $ETCD_BACKUP" >> "$REPORT_FILE"

echo "## 2. 복원 테스트" >> "$REPORT_FILE"

# 격리된 네임스페이스 생성 및 복원
kubectl create namespace "$DRILL_NS"
velero restore create "dr-drill-$(date +%s)" \
  --from-backup "$LATEST_BACKUP" \
  --namespace-mappings "production:$DRILL_NS" \
  --wait

# 복원된 리소스 수 계산
PODS=$(kubectl get pods -n "$DRILL_NS" --no-headers | wc -l)
SVCS=$(kubectl get services -n "$DRILL_NS" --no-headers | wc -l)
echo "- 복원된 파드: $PODS" >> "$REPORT_FILE"
echo "- 복원된 서비스: $SVCS" >> "$REPORT_FILE"

echo "## 3. 헬스 검증" >> "$REPORT_FILE"

# 대기 후 파드 헬스 확인
sleep 60
HEALTHY=$(kubectl get pods -n "$DRILL_NS" --field-selector=status.phase=Running --no-headers | wc -l)
TOTAL=$(kubectl get pods -n "$DRILL_NS" --no-headers | wc -l)
echo "- 헬시 파드: $HEALTHY / $TOTAL" >> "$REPORT_FILE"

echo "## 4. 정리" >> "$REPORT_FILE"
kubectl delete namespace "$DRILL_NS" --wait=false
echo "- 드릴 네임스페이스 $DRILL_NS 삭제 예약됨" >> "$REPORT_FILE"

echo "DR 드릴 완료. 보고서: $REPORT_FILE"
cat "$REPORT_FILE"
```

---

## 10. 비용 최적화

### 10.1 비용 분석 프레임워크

```
Cost Breakdown (Monthly Estimate):
┌────────────────────────────────────────────────────────────┐
│  컴포넌트                      │ 월간 비용  │ 총계 비율    │
├───────────────────────────────┼──────────────┼─────────────┤
│  Control Plane (EKS)          │    $219      │    0.9%     │
│  System Nodes (3x m5.xl, RI)  │    $750      │    3.0%     │
│  General Nodes (avg 10x m5.2xl│  $5,550      │   22.2%     │
│  Stateful Nodes (3x r5.2xl)   │  $2,280      │    9.1%     │
│  GPU On-Demand (1x p4d.24xl)  │ $10,080      │   40.3%     │
│  GPU Spot (avg 2x p3.8xl)     │  $2,640      │   10.6%     │
│  Storage (EBS + S3)           │  $1,200      │    4.8%     │
│  Networking (NAT GW, ALB)     │  $1,500      │    6.0%     │
│  Monitoring                    │    $300      │    1.2%     │
│  Backups                       │    $200      │    0.8%     │
│  Misc                          │    $280      │    1.1%     │
├───────────────────────────────┼──────────────┼─────────────┤
│  합계                          │ $24,999      │  100.0%     │
└───────────────────────────────┴──────────────┴─────────────┘
```

### 10.2 비용 최적화 전략

```yaml
# 전략 1: VPA 권장 사항으로 적정 규모 산정
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: api-server-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  updatePolicy:
    updateMode: "Off"    # 권장 사항만
---
# 전략 2: 개발/스테이징을 0으로 스케일
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: staging-api
  namespace: staging
spec:
  scaleTargetRef:
    name: api-server
  minReplicaCount: 0             # 트래픽 없을 때 0으로 스케일
  maxReplicaCount: 5
  cooldownPeriod: 300
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        metricName: http_requests_per_second
        query: sum(rate(http_requests_total{namespace="staging"}[2m]))
        threshold: "1"
```

### 10.3 Kubecost를 사용한 비용 모니터링

```bash
# 비용 가시성을 위한 Kubecost 설치
helm install kubecost cost-analyzer \
  --repo https://kubecost.github.io/cost-analyzer/ \
  --namespace kubecost \
  --create-namespace \
  --set kubecostToken="<token>" \
  --set prometheus.nodeExporter.enabled=false \
  --set prometheus.kube-state-metrics.disabled=true \
  --set global.prometheus.enabled=false \
  --set global.prometheus.fqdn=http://prometheus.monitoring:9090

# 네임스페이스별 비용 쿼리
kubectl port-forward -n kubecost svc/kubecost-cost-analyzer 9090:9090
# 대시보드: http://localhost:9090

# 비용 데이터를 위한 API 쿼리
curl -s "http://localhost:9090/model/allocation?window=30d&aggregate=namespace" | \
  jq '.data[0] | to_entries[] | {namespace: .key, cost: .value.totalCost}'
```

### 10.4 클러스터 통합 보고서

```go
// 클러스터 활용도를 분석하고 통합을 제안하는 Go 도구
package main

import (
    "context"
    "fmt"
    "os"

    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/tools/clientcmd"
    metricsv "k8s.io/metrics/pkg/client/clientset/versioned"
)

func main() {
    config, _ := clientcmd.BuildConfigFromFlags("",
        os.Getenv("HOME")+"/.kube/config")

    clientset, _ := kubernetes.NewForConfig(config)
    metricsClient, _ := metricsv.NewForConfig(config)

    ctx := context.TODO()

    // 노드 할당 가능 리소스 가져오기
    nodes, _ := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
    // 실제 노드 메트릭 가져오기
    nodeMetrics, _ := metricsClient.MetricsV1beta1().NodeMetricses().List(
        ctx, metav1.ListOptions{})

    fmt.Println("=== 클러스터 활용도 보고서 ===")
    fmt.Println()

    for _, node := range nodes.Items {
        allocCPU := node.Status.Allocatable.Cpu().MilliValue()
        allocMem := node.Status.Allocatable.Memory().Value() / (1024 * 1024 * 1024)

        // 일치하는 메트릭 찾기
        for _, nm := range nodeMetrics.Items {
            if nm.Name == node.Name {
                usedCPU := nm.Usage.Cpu().MilliValue()
                usedMem := nm.Usage.Memory().Value() / (1024 * 1024 * 1024)

                cpuPct := float64(usedCPU) / float64(allocCPU) * 100
                memPct := float64(usedMem) / float64(allocMem) * 100

                status := "정상"
                if cpuPct < 20 && memPct < 20 {
                    status = "저활용 - 통합 고려"
                }

                fmt.Printf("노드: %s\n", node.Name)
                fmt.Printf("  CPU: %dm / %dm (%.1f%%)\n", usedCPU, allocCPU, cpuPct)
                fmt.Printf("  메모리: %dGi / %dGi (%.1f%%)\n", usedMem, allocMem, memPct)
                fmt.Printf("  상태: %s\n\n", status)
            }
        }
    }
}
```

---

## 연습문제

### 연습문제 1: 아키텍처 설계 문서

섹션 1.2의 TechCorp 요구사항을 기반으로, 다음을 다루는 아키텍처 결정 기록(ADR)을 작성하세요: (a) 자체 관리 대 관리형 Kubernetes 선택, (b) 선택한 CNI 플러그인과 정당화, (c) 스테이트풀 워크로드를 위한 스토리지 전략, (d) 5개 팀을 위한 멀티 테넌시 모델.

<details><summary>정답 보기</summary>

```
# ADR-001: Kubernetes 플랫폼 아키텍처

## 상태: 승인됨

## 날짜: 2025-01-15

## (a) 결정: 관리형 Kubernetes (AWS EKS)

근거:
- AWS가 컨트롤 플레인 관리 (HA, 업그레이드, 패칭) 처리
- 자체 관리 대비 운영 부담 감소 (~1 FTE 절감)
- AWS 서비스와 네이티브 통합 (ALB, EBS, S3, IAM)
- 비용: EKS 컨트롤 플레인 $219/월 대 자체 관리 3개 노드 ~$2,000/월
- AWS가 EKS 컨트롤 플레인에 99.95% 업타임 SLA 제공, SLO 목표와 일치

## (b) 결정: Cilium CNI

근거:
- eBPF 기반 네트워킹으로 iptables 대안보다 우수한 성능
- 네이티브 AWS ENI 통합 (오버레이 없음, 지연 감소)
- L7 HTTP 인식 정책으로 고급 네트워크 정책
- Hubble을 통한 내장 옵저버빌리티
- 파드 간 암호화를 위한 WireGuard 암호화

## (c) 결정: 스토리지 전략

- PostgreSQL: EBS io2 볼륨 (10,000 IOPS), Patroni를 사용한 3 AZ StatefulSet
- Redis: EBS gp3 + Redis Sentinel HA
- Elasticsearch: 성능을 위한 로컬 NVMe (i3en 인스턴스)
- Kafka: 복제 팩터 3의 EBS gp3

## (d) 결정: 멀티 테넌시 모델

RBAC과 NetworkPolicy 격리를 갖춘 팀별 네임스페이스:
- platform-team: kube-system, monitoring, argocd
- backend-team: backend, backend-jobs
- frontend-team: frontend
- data-team: data-platform, kafka
- ml-team: ml-training, ml-serving, ml-notebooks

격리 메커니즘:
- RBAC: OIDC 그룹에 바인딩된 팀별 Role
- NetworkPolicy: 네임스페이스별 기본 거부, 명시적 허용 규칙
- ResourceQuota: 네임스페이스별 CPU, 메모리, GPU 제한
- PodSecurity: 모든 팀 네임스페이스에 Restricted 프로파일
```

</details>

### 연습문제 2: 보안 강화 구현

프로덕션 네임스페이스에 다음 보안 제어를 구현하세요: (a) 특권 컨테이너를 방지하는 파드 보안 표준, (b) 모든 파드에 리소스 제한을 요구하는 Kyverno 정책, (c) 인그레스 컨트롤러와 모니터링 네임스페이스에서의 인그레스만 허용하는 네트워크 정책, (d) 백엔드 팀에 디플로이먼트 읽기-쓰기, 시크릿 읽기 전용 접근 권한을 부여하는 RBAC 구성.

<details><summary>정답 보기</summary>

```yaml
# (a) Pod Security Standard - Restricted
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.29
---
# (b) 리소스 제한을 요구하는 Kyverno 정책
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-resource-limits
spec:
  validationFailureAction: Enforce
  rules:
    - name: check-limits
      match:
        any:
          - resources:
              kinds:
                - Pod
              namespaces:
                - production
      validate:
        message: "All containers must specify resource requests and limits"
        pattern:
          spec:
            containers:
              - resources:
                  requests:
                    cpu: "?*"
                    memory: "?*"
                  limits:
                    cpu: "?*"
                    memory: "?*"
---
# (c) 기본 모든 인그레스 거부
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
---
# 인그레스 컨트롤러에서 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ingress-controller
  namespace: production
spec:
  podSelector:
    matchLabels:
      exposure: external
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-system
---
# (d) 백엔드 팀 RBAC
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: backend-developer
  namespace: production
rules:
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["services", "configmaps", "pods"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: backend-team-binding
  namespace: production
subjects:
  - kind: Group
    name: backend-developers
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: backend-developer
  apiGroup: rbac.authorization.k8s.io
```

</details>

### 연습문제 3: 옵저버빌리티 스택 배포

완전한 옵저버빌리티 스택을 배포하기 위한 Helm 값과 구성을 작성하세요: (a) 15일 보존과 Thanos 사이드카가 있는 Prometheus, (b) SSO(OIDC)가 있는 Grafana, (c) 로그 수집을 위한 Loki, (d) 3개 가장 중요한 SLO(API 서버 가용성, 스케줄링 지연, 노드 헬스)에 대한 알림 규칙.

<details><summary>정답 보기</summary>

```yaml
# (d) 중요 SLO 알림 규칙
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: critical-slo-alerts
  namespace: monitoring
spec:
  groups:
    # SLO 1: API 서버 가용성 (99.95%)
    - name: api-server-slo
      rules:
        - record: slo:apiserver:availability:5m
          expr: |
            1 - (
              sum(rate(apiserver_request_total{code=~"5.."}[5m]))
              /
              sum(rate(apiserver_request_total[5m]))
            )
        - alert: APIServerAvailabilitySLOBreach
          expr: slo:apiserver:availability:5m < 0.999
          for: 5m
          labels:
            severity: critical

    # SLO 2: 스케줄링 지연 (p99 < 5s)
    - name: scheduling-slo
      rules:
        - record: slo:scheduling:latency_p99:5m
          expr: |
            histogram_quantile(0.99,
              sum(rate(scheduler_scheduling_attempt_duration_seconds_bucket[5m])) by (le)
            )
        - alert: SchedulingLatencySLOBreach
          expr: slo:scheduling:latency_p99:5m > 5
          for: 10m
          labels:
            severity: critical

    # SLO 3: 노드 헬스 (99.9% Ready)
    - name: node-health-slo
      rules:
        - record: slo:node:ready_ratio
          expr: |
            sum(kube_node_status_condition{condition="Ready",status="true"})
            /
            sum(kube_node_status_condition{condition="Ready"})
        - alert: NodeHealthSLOBreach
          expr: slo:node:ready_ratio < 0.999
          for: 5m
          labels:
            severity: critical
```

</details>

### 연습문제 4: 재해 복구 드릴

다음을 검증하는 완전한 DR 드릴 스크립트를 작성하세요: (a) etcd 백업이 존재하며 6시간 미만, (b) Velero가 격리된 테스트 네임스페이스로 네임스페이스를 복원 가능, (c) 복원된 파드가 5분 내에 Running 상태 도달, (d) 각 검사의 합격/불합격 상태를 포함한 보고서 생성. 모든 검사가 통과할 때만 코드 0으로 종료.

<details><summary>정답 보기</summary>

```bash
#!/usr/bin/env bash
set -euo pipefail

DRILL_ID="drill-$(date +%s)"
TEST_NS="dr-test-${DRILL_ID}"
PASS=0
FAIL=0

check() {
    local name="$1" result="$2" detail="$3"
    [ "$result" = "PASS" ] && PASS=$((PASS + 1)) || FAIL=$((FAIL + 1))
    echo "[$result] $name: $detail"
}

echo "=== DR Drill $DRILL_ID ==="

# (a) etcd 백업 확인
LATEST_ETCD=$(aws s3 ls s3://prod-backups/etcd/ --recursive | sort | tail -1)
if [ -z "$LATEST_ETCD" ]; then
    check "etcd-backup" "FAIL" "S3에 etcd 백업 없음"
else
    check "etcd-backup" "PASS" "최근 백업 발견: $LATEST_ETCD"
fi

# (b) Velero 복원 테스트
LATEST_VELERO=$(velero backup get -o json | \
    jq -r '[.items[] | select(.status.phase=="Completed")] |
    sort_by(.status.completionTimestamp) | last | .metadata.name // empty')

if [ -n "$LATEST_VELERO" ]; then
    kubectl create namespace "$TEST_NS" 2>/dev/null || true
    velero restore create "dr-restore-${DRILL_ID}" \
        --from-backup "$LATEST_VELERO" \
        --include-namespaces production \
        --namespace-mappings "production:${TEST_NS}" --wait 2>/dev/null

    RESTORE_STATUS=$(velero restore get "dr-restore-${DRILL_ID}" -o json | jq -r '.status.phase')
    [ "$RESTORE_STATUS" = "Completed" ] && \
        check "velero-restore" "PASS" "복원 완료" || \
        check "velero-restore" "FAIL" "복원 상태: $RESTORE_STATUS"

    # (c) 파드 헬스 확인
    sleep 60
    TOTAL=$(kubectl get pods -n "$TEST_NS" --no-headers 2>/dev/null | wc -l | tr -d ' ')
    RUNNING=$(kubectl get pods -n "$TEST_NS" --no-headers \
        --field-selector=status.phase=Running 2>/dev/null | wc -l | tr -d ' ')

    [ "$TOTAL" -gt 0 ] && [ "$RUNNING" -eq "$TOTAL" ] && \
        check "pod-health" "PASS" "${RUNNING}/${TOTAL} 파드 Running" || \
        check "pod-health" "FAIL" "${RUNNING}/${TOTAL} 파드 Running"

    # 정리
    kubectl delete namespace "$TEST_NS" --wait=false 2>/dev/null || true
else
    check "velero-backup" "FAIL" "완료된 Velero 백업 없음"
fi

# (d) 보고서
echo ""
echo "=== 보고서 ==="
echo "통과: $PASS, 실패: $FAIL"
echo "전체: $([ $FAIL -eq 0 ] && echo "PASS" || echo "FAIL")"

[ $FAIL -eq 0 ] && exit 0 || exit 1
```

</details>

### 연습문제 5: 비용 최적화 계획

섹션 10.1의 비용 분석을 기반으로, 5가지 구체적인 비용 최적화 조치를 식별하세요. 각 조치에 대해: (a) 현재 비용, (b) 예상 절감액, (c) 구현 단계, (d) 위험 또는 트레이드오프를 제시하세요. 전체 20% 비용 절감을 목표로 하세요.

<details><summary>정답 보기</summary>

```
비용 최적화 계획 - 목표: 20% 절감 ($5,000/월 절감)

현재 총액: $24,999/월
목표 총액: $19,999/월

=== 조치 1: 범용 풀에 스팟 인스턴스 (야간+주말) ===
(a) 현재: 10x m5.2xlarge 온디맨드 = $5,550/월
(b) 절감: 비피크 시 50% 스팟 혼합 = ~$1,665/월 절감
(c) 단계: 스팟 노드 그룹 생성, 오토스케일러 구성
(d) 위험: 스팟 중단 → 스테이트리스 워크로드만 적용, 온디맨드 최소 3개 유지

=== 조치 2: GPU 예약 인스턴스 + 스팟 학습 ===
(a) 현재: 1x p4d.24xlarge 온디맨드 = $10,080/월
(b) 절감: 1년 RI (선불 없음) 36% 절감 = ~$4,000/월 절감
(c) 단계: GPU 활용도 분석, 서빙용 RI 구매, 학습은 스팟으로 전환
(d) 위험: RI 약정 (1년) → 변환형 RI로 시작, 1개월 모니터링 후 결정

=== 조치 3: 범용 풀 인스턴스 적정 규모 산정 ===
(a) 현재: m5.2xlarge (8 vCPU, 32GB)
(b) 절감: VPA 권장에 따라 노드 4개 감소 = ~$1,100/월 절감
(c) 단계: VPA 배포 (2주), 권장사항 분석, 리소스 요청 조정
(d) 위험: 과밀 배치 → 20% 여유 유지, 변경 후 지연 모니터링

=== 조치 4: 스토리지 비용 최적화 ===
(a) 현재: $1,200/월
(b) 절감: ~$350/월 (S3 수명주기 정책, 미사용 PVC 삭제)
(c) 단계: PVC 감사, S3 수명주기 정책 추가, 고아 볼륨 삭제
(d) 위험: Elasticsearch gp3 전환 시 성능 → 벤치마크 후 마이그레이션

=== 조치 5: 스테이징/개발을 0으로 스케일 ===
(a) 현재: 스테이징 24/7 운영 = ~$3,000/월
(b) 절감: 비업무 시간 0으로 스케일 = ~$1,500/월 절감
(c) 단계: KEDA 설치, ScaledObject 구성, minReplicas=0
(d) 위험: 콜드 스타트 지연 → 테스트 실행 전 크론 트리거로 스케일업

=== 요약 ===
| 조치                      | 월간 절감    | 구현 기간    |
|---------------------------|-------------|-------------|
| 1. 범용 풀 스팟            | $1,665      | 1주          |
| 2. GPU RI + 스팟 학습      | $4,000      | 2주          |
| 3. 인스턴스 적정 규모      | $1,100      | 3주          |
| 4. 스토리지 최적화         | $350        | 1주          |
| 5. 스테이징 0으로 스케일   | $1,500      | 2주          |
| 총 절감                    | $8,615 (34.5%) |           |

20% 목표를 초과하여 성장을 위한 버퍼 제공.
우선순위 순서: 2 → 1 → 5 → 3 → 4 (높은 영향 순)
```

</details>

---

**이전**: [18. ML을 위한 Kubernetes](./18_Kubernetes_for_ML.md) | **다음**: [00. 개요](./00_Overview.md)
