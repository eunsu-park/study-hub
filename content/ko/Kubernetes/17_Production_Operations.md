# 17. 프로덕션 운영(Production Operations)

**이전**: [16. Kubernetes API 프로그래밍](./16_Kubernetes_API_Programming.md) | **다음**: [18. ML을 위한 Kubernetes](./18_Kubernetes_for_ML.md)

## 학습 목표

- 인플레이스(In-place)와 블루-그린(Blue-Green) 전략을 사용한 Kubernetes 클러스터 업그레이드 계획 및 실행
- 백업, 복원, 조각 모음(Defragmentation)을 포함한 etcd 운영 수행
- Kubernetes 클러스터를 위한 재해 복구(Disaster Recovery) 계획 설계
- 용량 계획(Capacity Planning), 노드 유지보수, 인증서 순환(Certificate Rotation) 수행
- 프로덕션 이슈 트러블슈팅 및 클러스터 성능 튜닝

---

프로덕션에서 Kubernetes를 운영하려면 워크로드 배포를 넘어서는 운영 작업에 대한 숙달이 필요합니다. 클러스터 업그레이드는 다운타임 없이 수행되어야 합니다. etcd -- 유일한 진실의 원천(Single Source of Truth) -- 는 세심한 백업과 유지보수가 필요합니다. 노드는 장애가 발생하고, 인증서는 만료되며, 용량은 수요에 맞춰 증가해야 합니다. 이 레슨에서는 프로덕션 Kubernetes 클러스터를 건강하고, 안정적이며, 성능이 좋은 상태로 유지하는 운영 원칙을 다룹니다.

runbook에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 업그레이드 진행 방식을 제한하는 version-skew 규칙, etcd 백업과 복원이 유일한 진정한 disaster recovery인 이유, 조용히 만료되어 클러스터를 깨뜨리는 인증서 생태계, 그리고 "클러스터가 건강한가?"를 vibes 질문에서 측정 가능한 것으로 바꾸는 SLO 규율을 다룹니다.

## 목차

- [이론과 원리](#이론과-원리)
- [1. 클러스터 업그레이드 전략](#1-클러스터-업그레이드-전략)
- [2. etcd 운영](#2-etcd-운영)
- [3. 재해 복구 계획](#3-재해-복구-계획)
- [4. 용량 계획과 적정 규모 산정](#4-용량-계획과-적정-규모-산정)
- [5. 노드 유지보수](#5-노드-유지보수)
- [6. 인증서 관리와 순환](#6-인증서-관리와-순환)
- [7. 프로덕션 이슈 트러블슈팅](#7-프로덕션-이슈-트러블슈팅)
- [8. 성능 튜닝](#8-성능-튜닝)
- [9. Kubernetes의 SLA와 SLO](#9-kubernetes의-sla와-slo)
- [연습문제](#연습문제)

---

## 이론과 원리

아키텍처, 네트워킹, 워크로드 챕터는 *모든 것이 정상일 때* 쿠버네티스가 어떻게 동작하는지 알려줬습니다. 프로덕션 운영은 그렇지 않을 때 무엇을 하는지입니다. 클러스터는 다운타임 없이 메이저 버전을 가로질러 업그레이드되어야 하고, etcd는 실제로 복원 가능한 백업이 필요하며, 인증서는 조용히 만료되고, 노드는 실패하거나 drain이 필요하며, 용량은 트래픽 스파이크 하에서 고갈됩니다. 동작하는 클러스터와 동작하는 *프로덕션* 클러스터를 구분하는 규율은 이 각각에 대해 — 필요해지기 전에 — 테스트된 절차를 갖는 것입니다. 이 섹션은 업그레이드를 제한하는 version-skew 제약, 실제 RPO/RTO를 정의하는 etcd 백업과 복원, 모든 TLS 연결의 조용한 기반인 인증서 라이프사이클, 그리고 "클러스터가 건강한가?"를 답할 수 있게 만드는 SLO/SLI 프레이밍을 설명합니다.

### A. 업그레이드와 Version-Skew 정책

etcd, 그다음 API 서버, 그다음 노드를 임의 순서로 업그레이드할 수는 없습니다. 쿠버네티스는 컴포넌트 간 버전 차이를 제한하는 엄격한 **version skew 정책**을 정의합니다:

- **최신 API 서버** (HA — 모든 API 서버 레플리카는 롤링 업그레이드 동안 서로 1 minor 버전 이내여야 함).
- **kube-controller-manager, kube-scheduler, cloud-controller-manager** — API 서버보다 최대 1 minor 버전 *오래된*.
- **kubelet** — API 서버보다 최대 3 minor 버전 오래된(따라서 N-3 kubelet은 N API 서버에 대해 동작).
- **kube-proxy** — 최대 3 minor 버전 오래된.
- **kubectl** — API 서버와 최대 1 minor 버전 다른(더 오래되거나 더 새로운).

함의 — **etcd → API 서버 → 다른 컨트롤 플레인 → kubelet → kube-proxy → kubectl/클라이언트** 순서로 업그레이드합니다. 버전 건너뛰기(1.27 → 1.30 직접)는 지원되지 않습니다 — in-tree 마이그레이션 코드는 인접 버전에 대해서만 작성되어 있어, 한 번에 한 minor씩 버전 사다리를 걸어야 합니다.

두 업그레이드 전략이 지배적입니다:

**In-place** (kubeadm upgrade, 관리형 클라우드 업그레이드) — 노드를 drain, kubelet+컨테이너 런타임을 in-place 업그레이드, uncordon. 노드별 반복. 장점 — 추가 용량 불필요; 클러스터 내 IP와 노드 이름 안정 유지. 단점 — 진행 중 업그레이드 상태가 관찰 가능하고 부분 실패 가능. drain이 가용성을 위반하지 않도록 신중한 PDB(2강 §7) 필요.

**Blue-green** (새 버전으로 새 클러스터 프로비저닝, 워크로드 마이그레이션, 옛 것 폐기) — 설계상 zero-downtime; 어떤 워크로드 이동 전에 새 클러스터를 테스트할 수 있게 함. 단점 — 전체 두 번째 클러스터 용량 필요; 클러스터 간 네트워킹과 stateful 워크로드 마이그레이션이 사소하지 않음. GitOps + 멀티 클러스터 service mesh를 사용하는 클라우드 설정에서 흔함.

결정은 주로 stateful 워크로드가 클러스터 내 재시작을 견딜 수 있는지(in-place 선호) 또는 그것들에 대해서도 진정한 zero-downtime이 필요한지(blue-green 선호)에 관한 것입니다.

### B. etcd — 가진 유일한 진정한 백업

etcd를 잃으면 클러스터를 잃습니다 — 모든 쿠버네티스 객체가 거기 삽니다(1강 §B). 백업은 선택이 아니며, **복원 테스트된 백업만이 백업으로 카운트됩니다.** 드릴에서 한 번도 복원되지 않은 백업은 희망이지 백업이 아닙니다.

메커니즘은 단순합니다. etcd는 내장 snapshot 명령을 가집니다:

```bash
ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  snapshot save /backup/etcd-$(date +%Y%m%d-%H%M%S).db
```

이를 정기 스케줄(CronJob, systemd timer, 클라우드 백업 서비스)로 실행하고, etcd가 도달할 수 없는 어딘가(off-cluster S3, off-region 스토리지)로 스냅샷을 보냅니다. 분기마다 sandbox 클러스터로 복원 테스트 — `etcdctl snapshot status`만이 아니라 실제로 스냅샷 복원.

복원은 etcd 데이터 디렉토리를 교체하고 etcd를 새 클러스터로 재시작합니다(스냅샷의 옛 멤버 ID는 라이브 멤버와 다르므로, 신선한 클러스터가 필수). HA — 한 멤버에서 복원, 그다음 다른 것들을 신선하게 추가 — 그것들은 복원된 멤버에서 동기화될 것입니다.

백업 품질에 영향을 미치는 두 운영 노브:

- **Defragmentation.** etcd는 키가 쓰이고 삭제됨에 따라 시간이 지나며 단편화를 누적합니다 — 주기적 `etcdctl defrag`가 공간을 회수합니다. 그것 없이는 etcd가 예상치 못하게 OOM될 수 있습니다. 월간 스케줄.
- **Auto-compaction.** etcd는 revision 이력을 유지합니다. `--auto-compaction-retention=8h` 플래그가 8시간보다 오래된 revision을 prune하여, 스토리지 성장을 제한합니다. 그것 없이는 etcd가 가득 찰 때까지 무한히 성장합니다.

실제 **RPO**(최대 데이터 손실)는 백업 간격 — **RTO**(최대 복원 시간)는 스냅샷 복사 + 복원 + 클러스터 재합류 시간입니다. 잘 운영되는 대부분의 클러스터는 RPO ≤ 1시간, RTO ≤ 30분을 목표로 하며, 둘 다 시간별 스냅샷과 자동화된 복원 도구로 달성 가능합니다.

### C. 인증서 — 조용한 만료

쿠버네티스 클러스터는 기본적으로 약 20개의 인증서를 실행합니다 — API 서버 인증서, etcd peer/client 인증서, kubelet client 인증서, kubelet server 인증서, controller-manager 인증서, scheduler 인증서, front-proxy 인증서, ServiceAccount 서명 키, 그리고 ingress와 웹훅을 위한 모든 인증서. 대부분은 1년 유효성(kubeadm 기본)으로 클러스터 생성 시 발급됩니다. 366일째에 만료되고, 클러스터의 일부가 서로 통신을 멈춥니다.

복구는 잘 알려져 있지만 파괴적입니다:

```bash
kubeadm certs renew all
systemctl restart kubelet
```

그러나 훨씬 더 나은 규율은 *그것이 일어나지 않게 하는 것*입니다:

- **만료 모니터링.** `kubeadm certs check-expiration`이 모든 인증서와 그 남은 유효성을 나열합니다. 이를 CronJob으로 실행 — 어떤 인증서든 < 30일이면 알림. 대부분의 프로덕션 클러스터는 이를 자체 관측 가능성 스택에 통합합니다.
- **kubelet 인증서 자동 회전.** kubelet의 `--rotate-certificates`와 `--rotate-server-certificates` 플래그 + `RotateKubeletClientCertificate`와 `RotateKubeletServerCertificate` 기능 게이트가 kubelet 인증서를 수동 액션 없이 회전하게 합니다.
- **생성 시점에 유효성 증가.** kubeadm은 커스텀 유효성을 허용(`--cert-validity-period`); 일부 팀은 보안 트레이드오프를 수용하고 회전 빈도를 줄이기 위해 5년 인증서를 사용합니다.
- **애플리케이션 TLS에 cert-manager 사용.** 모든 사용자 대면(Ingress 인증서, 웹훅 server 인증서) 자동화 — 클러스터 내부 인증서만 kubeadm/수동 경로 필요.

인증서 만료는 쿠버네티스에서 가장 예방 가능한 프로덕션 사고입니다. 그것이 일어나는 이유는 인증서가 깨질 때까지 보이지 않기 때문입니다 — 남은 수명에 알림하면 그것들이 보이게 됩니다.

### D. SLO, SLI, 그리고 건강 측정의 규율

운영적 정의 없이 "클러스터가 건강한가?"는 답할 수 없습니다. SRE 관행 — **SLI**(Service Level Indicator), **SLO**(Service Level Objective), **error budget** — 이 그것을 관리할 수 있는 것으로 바꿉니다.

전형적 쿠버네티스 플랫폼 SLO 집합:

| SLI | SLO | 왜 중요한가 |
|-----|-----|-----------|
| API 서버 p99 GET 지연 | < 1초 | etcd/apiserver 건강 표시 |
| API 서버 가용성 | > 99.95% | 컨트롤러가 의존하는 plane |
| Pod 스케줄링 지연 p95 (Pending → Bound) | < 10초 | 스케줄러 + 어드미션 건강 표시 |
| 컨테이너 재시작 비율 | 시간당 < 0.1% | 워크로드 건강 표시 |
| 성공적 업그레이드 비율 | 롤아웃 윈도우 내 100% | 변경 위험 표시 |
| etcd 읽기 p99 지연 | < 100ms | etcd 디스크 + 네트워크 표시 |

**Error budget** = `1 - SLO`. API 가용성 SLO가 99.95%이면, error budget은 0.05% = 월 약 22분입니다. 예산이 고갈되면(이미 이번 달 22분 손실), 예산이 회복될 때까지 변경 배포를 중단합니다 — 예산을 태우면서 더 많은 변경을 푸시하면 다음 사고를 더 나쁘게 만듭니다.

이것이 강제하는 규율 — 모든 변경은 위험하며, 정량적 한계로 변경 속도와 신뢰성의 균형을 맞춥니다. SLO를 채택한 플랫폼 팀은 사고 수가 떨어집니다 — error budget이 "배포 중단"을 비정치적으로 만들기 때문입니다.

### 이론에서 아래의 runbook으로

이제 레슨은 이 추상을 적용합니다:

- **섹션 1 (클러스터 업그레이드 전략)**은 §A입니다 — version skew, in-place vs blue-green, kubeadm 업그레이드 흐름, 관리형 클라우드 업그레이드.
- **섹션 2 (etcd 운영)**은 §B입니다 — 백업, 복원, defragmentation, compaction.
- **섹션 3 (재해 복구 계획)**은 더 넓은 RPO/RTO + 테스트된 복원 + 다중 지역 narrative입니다.
- **섹션 4 (용량 계획과 적정 규모 산정)**은 과도 프로비저닝 낭비와 부족 프로비저닝 사고를 모두 피하는 방법입니다.
- **섹션 5 (노드 유지보수)**는 cordon/drain/uncordon dance — 종종 cluster-autoscaler나 업그레이드 도구로 자동화됨.
- **섹션 6 (인증서 관리와 순환)**은 운영 세부의 §C입니다.
- **섹션 7 (프로덕션 이슈 트러블슈팅)**은 실용 playbook — pod 로그, 이벤트, kubelet 로그, 컨트롤 플레인 로그.
- **섹션 8 (성능 튜닝)**은 SLO(§D)가 미끄러지기 시작할 때의 lever입니다.
- **섹션 9 (Kubernetes의 SLA와 SLO)**는 코드화된 §D입니다 — 무엇을 측정할지, 어떻게 표현할지, error budget을 어떻게 관리할지.

운영을 "측정된 SLO를 가진 테스트된 절차"로 보고 나면, 규율은 "무엇이 잘못될 수 있는가, 절차가 무엇인가, 마지막으로 언제 테스트했나?"가 됩니다. 그 사고방식이 운으로 가동 시간을 가지는 클러스터와 설계로 가동 시간을 가지는 클러스터의 차이를 만듭니다.

---

## 1. 클러스터 업그레이드 전략

### 1.1 Kubernetes 버전 정책

Kubernetes는 시맨틱 버전 관리(`MAJOR.MINOR.PATCH`)를 따릅니다. 프로젝트는 가장 최근 3개 마이너 버전에 대한 릴리스 브랜치를 유지합니다. 버전 스큐(Version Skew) 정책은 호환 가능한 컴포넌트 버전을 규정합니다:

```
Version Skew Policy:
┌─────────────────────────────────────────────────────────────┐
│  kube-apiserver     v1.29    (reference version)            │
│  kube-controller-manager     v1.29 or v1.28                 │
│  kube-scheduler              v1.29 or v1.28                 │
│  kubelet                     v1.29, v1.28, or v1.27         │
│  kube-proxy                  same minor as kubelet           │
│  kubectl                     v1.30, v1.29, or v1.28 (+/- 1) │
└─────────────────────────────────────────────────────────────┘

Upgrade Order:
  1. etcd (if separate)
  2. kube-apiserver (all instances)
  3. kube-controller-manager
  4. kube-scheduler
  5. cloud-controller-manager
  6. kubelet + kube-proxy (node by node)
```

### 1.2 인플레이스 업그레이드(In-Place Upgrade)

인플레이스 전략은 기존 클러스터 컴포넌트를 하나씩 업그레이드합니다. kubeadm으로 관리되는 클러스터의 기본 접근 방식입니다.

```bash
# 단계 1: 현재 버전 확인
kubectl get nodes -o wide
kubeadm version

# 단계 2: 첫 번째 컨트롤 플레인 노드에서 kubeadm 업그레이드
sudo apt-mark unhold kubeadm
sudo apt-get update && sudo apt-get install -y kubeadm=1.29.0-1.1
sudo apt-mark hold kubeadm

# 단계 3: 업그레이드 계획 (드라이 런)
sudo kubeadm upgrade plan

# 단계 4: 컨트롤 플레인에 업그레이드 적용
sudo kubeadm upgrade apply v1.29.0

# 단계 5: 컨트롤 플레인 노드에서 kubelet과 kubectl 업그레이드
sudo apt-mark unhold kubelet kubectl
sudo apt-get install -y kubelet=1.29.0-1.1 kubectl=1.29.0-1.1
sudo apt-mark hold kubelet kubectl
sudo systemctl daemon-reload
sudo systemctl restart kubelet

# 단계 6: 추가 컨트롤 플레인 노드는 'apply' 대신 'node' 사용
sudo kubeadm upgrade node

# 단계 7: 워커 노드를 하나씩 업그레이드
# (노드 유지보수 섹션의 드레인 절차 참조)
```

### 1.3 블루-그린 업그레이드(Blue-Green Upgrade)

블루-그린 전략은 기존 클러스터(블루) 옆에 완전히 새로운 클러스터(그린)를 프로비저닝한 후 트래픽을 전환합니다:

```
Blue-Green Cluster Upgrade:
┌────────────────────────────────────────────────┐
│                Load Balancer / DNS              │
│                                                 │
│    ┌─────────────────┐  ┌─────────────────┐    │
│    │  Blue Cluster    │  │  Green Cluster   │    │
│    │  v1.28 (current) │  │  v1.29 (new)     │    │
│    │                  │  │                  │    │
│    │  ┌──────────┐   │  │  ┌──────────┐   │    │
│    │  │ Workloads │   │  │  │ Workloads │   │    │
│    │  └──────────┘   │  │  │ (migrated) │   │    │
│    │                  │  │  └──────────┘   │    │
│    └─────────────────┘  └─────────────────┘    │
│                                                 │
│  Phase 1: Deploy green, migrate workloads       │
│  Phase 2: Shift traffic to green                │
│  Phase 3: Validate, then decommission blue      │
└────────────────────────────────────────────────┘
```

인프라스트럭처 코드(Infrastructure-as-Code)를 사용한 블루-그린 업그레이드 워크플로우:

```bash
# 단계 1: 업데이트된 버전으로 새 클러스터 프로비저닝
terraform apply -var="cluster_version=1.29" -var="cluster_name=prod-green"

# 단계 2: 공유 인프라 배포 (인그레스, 모니터링 등)
kubectl --context=prod-green apply -k infrastructure/

# 단계 3: GitOps를 통한 워크로드 마이그레이션 (ArgoCD가 새 클러스터 대상)
argocd cluster add prod-green
argocd app set my-app --dest-server https://prod-green-api:6443

# 단계 4: 검증 테스트 실행
kubectl --context=prod-green run smoke-test --image=curlimages/curl \
  --rm -it -- curl -s http://my-app.default.svc/health

# 단계 5: DNS / 로드 밸런서를 그린으로 전환
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456 \
  --change-batch file://switch-to-green.json

# 단계 6: 에러율 모니터링 후 블루 해체
terraform destroy -var="cluster_name=prod-blue"
```

### 1.4 카나리 업그레이드 (노드 풀)

관리형 Kubernetes(EKS, GKE, AKS)에서는 노드 풀을 점진적으로 업그레이드합니다:

```bash
# GKE: 업데이트된 버전으로 새 노드 풀 생성
gcloud container node-pools create pool-v129 \
  --cluster=prod-cluster \
  --node-version=1.29.0 \
  --num-nodes=3 \
  --machine-type=e2-standard-4

# 이전 노드에 새 스케줄링 방지를 위한 코든(cordon)
kubectl cordon -l cloud.google.com/gke-nodepool=pool-v128

# 이전 노드를 하나씩 드레인
kubectl drain node-old-1 --ignore-daemonsets --delete-emptydir-data

# 새 풀에서 워크로드가 정상인지 확인
kubectl get pods -o wide | grep pool-v129

# 이전 노드 풀 삭제
gcloud container node-pools delete pool-v128 --cluster=prod-cluster
```

### 1.5 API 지원 중단 관리

각 Kubernetes 마이너 릴리스는 API 버전을 제거하거나 지원 중단할 수 있습니다. 업그레이드 전에 마이그레이션하지 않으면 기존 매니페스트와 CI 파이프라인이 손상됩니다. 컨트롤 플레인을 업그레이드하기 **전에** API 지원 중단을 처리하세요.

#### 1단계: 지원 중단된 API 사용 발견

```bash
# API 서버 메트릭에서 지원 중단된 API 사용 쿼리 (Kubernetes 1.22+)
kubectl get --raw /metrics | grep apiserver_requested_deprecated_apis

# 예시 출력 (레이블을 디코딩하여 문제 리소스 찾기):
# apiserver_requested_deprecated_apis{group="extensions",removed_release="1.25",
#   resource="ingresses",subresource="",version="v1beta1"} 1

# pluto(정적 분석 도구)를 사용하여 라이브 클러스터와 로컬 매니페스트 스캔
# https://github.com/FairwindsOps/pluto
pluto detect-all-in-cluster
pluto detect -f ./manifests/

# kubectl-convert를 사용하여 리소스의 현재 API 버전 나열
kubectl convert -f ingress.yaml --output-version networking.k8s.io/v1
```

#### 2단계: 매니페스트 변환

`kubectl convert`는 매니페스트를 지원 중단된 버전에서 현재 API 버전으로 마이그레이션합니다:

```bash
# kubectl-convert 플러그인 설치
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl-convert"
chmod +x kubectl-convert && mv kubectl-convert /usr/local/bin/

# 단일 파일 변환
kubectl convert -f old-ingress.yaml --output-version networking.k8s.io/v1 > new-ingress.yaml

# 디렉토리의 모든 파일 변환
for f in ./manifests/*.yaml; do
  kubectl convert -f "$f" --output-version networking.k8s.io/v1 > "new-${f##*/}" 2>/dev/null || cp "$f" "new-${f##*/}"
done
```

#### 3단계: 계획된 제거 추적

주요 API 버전별 제거 목록 (업그레이드 계획용):

| 제거 버전 | API | 대체 |
|-----------|-----|------|
| 1.16 | `extensions/v1beta1` Deployments, DaemonSets 등 | `apps/v1` |
| 1.22 | `networking.k8s.io/v1beta1` Ingress | `networking.k8s.io/v1` |
| 1.25 | `policy/v1beta1` PodSecurityPolicy | 제거됨 (PSA 또는 OPA 사용) |
| 1.25 | `batch/v1beta1` CronJob | `batch/v1` |
| 1.26 | `autoscaling/v2beta2` HPA | `autoscaling/v2` |
| 1.27 | `storage.k8s.io/v1beta1` CSIStorageCapacity | `storage.k8s.io/v1` |
| 1.29 | `flowcontrol.apiserver.k8s.io/v1beta2` | `v1beta3` / `v1` |

각 마이너 버전 업그레이드 전에 항상 [Kubernetes API 지원 중단 가이드](https://kubernetes.io/docs/reference/using-api/deprecation-guide/)를 확인하세요.

---

## 2. etcd 운영

### 2.1 Kubernetes에서의 etcd 아키텍처

etcd는 모든 클러스터 상태를 저장합니다: 모든 객체, 모든 워치 리비전, 전체 RBAC 구성. Raft 합의 프로토콜을 사용하며 기능하려면 `(n/2)+1`개 멤버의 쿼럼(Quorum)이 필요합니다.

```
etcd Cluster (3-member):
┌──────────┐     ┌──────────┐     ┌──────────┐
│  etcd-0   │────│  etcd-1   │────│  etcd-2   │
│  (Leader) │    │ (Follower)│    │ (Follower)│
│           │    │           │    │           │
│  Raft Log │    │  Raft Log │    │  Raft Log │
│  WAL      │    │  WAL      │    │  WAL      │
│  Snapshot │    │  Snapshot │    │  Snapshot │
└──────────┘     └──────────┘     └──────────┘

Quorum: 2 of 3 members must agree
Failure tolerance: 1 member
```

### 2.2 백업

etcd 스냅샷은 전체 데이터베이스 상태를 캡처합니다:

```bash
# etcd 클러스터 헬스 확인
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  endpoint health

# 스냅샷 백업 생성
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  snapshot save /backup/etcd-$(date +%Y%m%d-%H%M%S).db

# 스냅샷 검증
ETCDCTL_API=3 etcdctl snapshot status /backup/etcd-20250115-100000.db --write-table
```

CronJob을 사용한 자동 백업:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: etcd-backup
  namespace: kube-system
spec:
  schedule: "0 */6 * * *"    # 6시간마다
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          hostNetwork: true
          nodeSelector:
            node-role.kubernetes.io/control-plane: ""
          tolerations:
            - key: node-role.kubernetes.io/control-plane
              effect: NoSchedule
          containers:
            - name: backup
              image: bitnami/etcd:3.5
              command:
                - /bin/sh
                - -c
                - |
                  ETCDCTL_API=3 etcdctl \
                    --endpoints=https://127.0.0.1:2379 \
                    --cacert=/etc/kubernetes/pki/etcd/ca.crt \
                    --cert=/etc/kubernetes/pki/etcd/server.crt \
                    --key=/etc/kubernetes/pki/etcd/server.key \
                    snapshot save /backup/etcd-$(date +%Y%m%d-%H%M%S).db

                  # S3에 업로드
                  aws s3 cp /backup/etcd-*.db s3://my-etcd-backups/

                  # 7일 이상 된 로컬 백업 정리
                  find /backup -name "etcd-*.db" -mtime +7 -delete
              volumeMounts:
                - name: etcd-certs
                  mountPath: /etc/kubernetes/pki/etcd
                  readOnly: true
                - name: backup-volume
                  mountPath: /backup
          volumes:
            - name: etcd-certs
              hostPath:
                path: /etc/kubernetes/pki/etcd
            - name: backup-volume
              hostPath:
                path: /var/backup/etcd
          restartPolicy: OnFailure
```

### 2.3 복원

```bash
# 모든 API 서버 인스턴스 중지 (복원된 데이터와 충돌 방지)
sudo mv /etc/kubernetes/manifests/kube-apiserver.yaml /tmp/

# 스냅샷에서 새 데이터 디렉토리로 복원
ETCDCTL_API=3 etcdctl snapshot restore /backup/etcd-20250115-100000.db \
  --data-dir=/var/lib/etcd-restored \
  --name=etcd-0 \
  --initial-cluster=etcd-0=https://10.0.1.10:2380 \
  --initial-advertise-peer-urls=https://10.0.1.10:2380

# 이전 데이터 디렉토리 교체
sudo mv /var/lib/etcd /var/lib/etcd-old
sudo mv /var/lib/etcd-restored /var/lib/etcd
sudo chown -R etcd:etcd /var/lib/etcd

# etcd 재시작
sudo systemctl restart etcd

# API 서버 매니페스트 복원
sudo mv /tmp/kube-apiserver.yaml /etc/kubernetes/manifests/

# 클러스터 헬스 확인
kubectl get nodes
kubectl get pods --all-namespaces
```

### 2.4 조각 모음(Defragmentation)

시간이 지남에 따라 etcd는 삭제 및 압축된 리비전에서 단편화된 여유 공간을 축적합니다:

```bash
# 데이터베이스 크기 대 사용 중인 크기 확인
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  endpoint status --write-table

# 이전 리비전 압축 (최근 것만 유지)
# 현재 리비전 가져오기
rev=$(ETCDCTL_API=3 etcdctl endpoint status --write-table \
  | awk -F'|' 'NR==2{print $4}' | tr -d ' ')
ETCDCTL_API=3 etcdctl compact $rev

# 각 멤버를 조각 모음 (쿼럼 유지를 위해 한 번에 하나씩)
ETCDCTL_API=3 etcdctl \
  --endpoints=https://etcd-0:2379 \
  defrag

ETCDCTL_API=3 etcdctl \
  --endpoints=https://etcd-1:2379 \
  defrag

ETCDCTL_API=3 etcdctl \
  --endpoints=https://etcd-2:2379 \
  defrag
```

### 2.5 etcd 성능 모니터링

```bash
# 지연시간 메트릭 확인
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  check perf --load="s" --prefix="/registry"

# etcd의 주요 Prometheus 메트릭
# etcd_disk_wal_fsync_duration_seconds    - WAL fsync 지연 (< 10ms이어야 함)
# etcd_disk_backend_commit_duration_seconds - DB 커밋 지연
# etcd_server_leader_changes_seen_total   - 리더 선출 (드물어야 함)
# etcd_mvcc_db_total_size_in_bytes        - 전체 DB 크기
# etcd_mvcc_db_total_size_in_use_in_bytes - 사용 중인 크기 (전체와 비교)
```

---

## 3. 재해 복구 계획

### 3.1 복구 목표

```
Disaster Recovery Metrics:
┌─────────────────────────────────────────────────────┐
│  RPO (Recovery Point Objective)                     │
│  = 최대 허용 데이터 손실                              │
│  = 마지막 백업 이후 시간                              │
│  예: RPO = 1시간 (매시간 백업)                        │
│                                                     │
│  RTO (Recovery Time Objective)                      │
│  = 최대 허용 다운타임                                 │
│  = 서비스 복원까지의 시간                              │
│  예: RTO = 30분                                     │
│                                                     │
│  복구 계층:                                          │
│  Tier 1: 전체 클러스터 손실 → 백업에서 복원             │
│  Tier 2: 컨트롤 플레인 장애 → HA 페일오버              │
│  Tier 3: 노드 장애 → 재스케줄링                       │
│  Tier 4: 파드 장애 → 재시작/교체                      │
└─────────────────────────────────────────────────────┘
```

### 3.2 DR 런북 구성 요소

완전한 재해 복구 계획은 다음을 포함해야 합니다:

```yaml
# dr-runbook.yaml
metadata:
  name: kubernetes-dr-plan
  last_tested: "2025-01-15"
  owner: platform-team

recovery_scenarios:
  - name: complete-cluster-loss
    rto: 2h
    rpo: 6h
    steps:
      - provision new infrastructure (terraform)
      - restore etcd from S3 backup
      - bootstrap control plane with kubeadm
      - verify node registration
      - restore persistent volumes from snapshots
      - validate application health
    prerequisites:
      - etcd snapshots in S3 (every 6 hours)
      - infrastructure-as-code in Git
      - PV snapshots enabled (CSI driver)
      - DNS TTL set to 300s

  - name: control-plane-failure
    rto: 5m
    rpo: 0
    steps:
      - HA control plane auto-failover (3 replicas)
      - load balancer health check removes failed node
      - remaining nodes form etcd quorum
    prerequisites:
      - 3+ control plane nodes across AZs
      - external load balancer for API server
      - etcd quorum maintained (2 of 3)

  - name: single-node-failure
    rto: 2m
    rpo: 0
    steps:
      - node marked NotReady after 40s
      - pods evicted after 5m (default)
      - scheduler places pods on healthy nodes
      - cluster autoscaler provisions replacement
    prerequisites:
      - pod disruption budgets defined
      - anti-affinity rules for HA workloads
      - cluster autoscaler configured

backup_locations:
  etcd: s3://prod-backups/etcd/
  velero: s3://prod-backups/velero/
  terraform_state: s3://prod-tfstate/
```

### 3.3 애플리케이션 수준 백업을 위한 Velero

```bash
# AWS 프로바이더로 Velero 설치
velero install \
  --provider aws \
  --bucket prod-velero-backups \
  --secret-file ./credentials-velero \
  --backup-location-config region=us-east-1 \
  --snapshot-location-config region=us-east-1

# 네임스페이스 백업 생성
velero backup create app-backup \
  --include-namespaces production \
  --include-resources deployments,services,configmaps,secrets,pvc

# 정기 백업 스케줄
velero schedule create daily-backup \
  --schedule="0 2 * * *" \
  --include-namespaces production,staging \
  --ttl 720h

# 다른 네임스페이스로 복원
velero restore create --from-backup app-backup \
  --namespace-mappings production:production-restored
```

---

## 4. 용량 계획과 적정 규모 산정

### 4.1 리소스 분석

```bash
# 모든 노드의 실제 리소스 사용량 대 요청량 확인
kubectl top nodes

# 파드 리소스 사용량 확인
kubectl top pods --all-namespaces --sort-by=memory

# 요청량 대 실제 사용량 비교
kubectl get pods -o custom-columns=\
"NAME:.metadata.name,\
REQ_CPU:.spec.containers[*].resources.requests.cpu,\
REQ_MEM:.spec.containers[*].resources.requests.memory,\
LIM_CPU:.spec.containers[*].resources.limits.cpu,\
LIM_MEM:.spec.containers[*].resources.limits.memory"
```

### 4.2 VPA 권장 사항을 통한 적정 규모 산정(Right-Sizing)

```yaml
# 권장 사항 전용 모드로 VPA 설치
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: web-app-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  updatePolicy:
    updateMode: "Off"    # 권장 사항만, 자동 업데이트 없음
  resourcePolicy:
    containerPolicies:
      - containerName: web
        minAllowed:
          cpu: 50m
          memory: 64Mi
        maxAllowed:
          cpu: "2"
          memory: 4Gi
```

```bash
# VPA 권장 사항 확인
kubectl get vpa web-app-vpa -o jsonpath='{.status.recommendation.containerRecommendations}' | jq .

# 출력 예시:
# [
#   {
#     "containerName": "web",
#     "lowerBound": {"cpu": "100m", "memory": "128Mi"},
#     "target": {"cpu": "250m", "memory": "256Mi"},
#     "upperBound": {"cpu": "500m", "memory": "512Mi"}
#   }
# ]
```

### 4.3 클러스터 용량 대시보드

용량 계획의 핵심 메트릭:

```
Capacity Planning Metrics:
┌──────────────────────────────────────────────────────────┐
│  메트릭                        │ 경고      │ 위험          │
├──────────────────────────────┼───────────┼───────────────┤
│  노드 CPU 할당                │ > 70%     │ > 85%         │
│  노드 메모리 할당              │ > 75%     │ > 90%         │
│  노드 수 대 오토스케일러 최대   │ > 80%     │ > 95%         │
│  PVC 사용률                    │ > 75%     │ > 90%         │
│  노드당 파드 수                │ > 80/110  │ > 100/110     │
│  etcd 데이터베이스 크기         │ > 4GB     │ > 6GB         │
│  API 서버 지연(p99)            │ > 1s      │ > 3s          │
└──────────────────────────────┴───────────┴───────────────┘
```

```yaml
# 용량에 대한 Prometheus 알림 규칙
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: capacity-alerts
  namespace: monitoring
spec:
  groups:
    - name: capacity
      rules:
        - alert: NodeCPUAllocationHigh
          expr: |
            sum(kube_pod_container_resource_requests{resource="cpu"}) by (node)
            /
            sum(kube_node_status_allocatable{resource="cpu"}) by (node)
            > 0.85
          for: 15m
          labels:
            severity: warning
          annotations:
            summary: "Node {{ $labels.node }} CPU allocation above 85%"

        - alert: ClusterMemoryPressure
          expr: |
            sum(kube_pod_container_resource_requests{resource="memory"})
            /
            sum(kube_node_status_allocatable{resource="memory"})
            > 0.80
          for: 30m
          labels:
            severity: warning
          annotations:
            summary: "Cluster memory allocation above 80%"
```

### 4.4 노드 풀 전략

```
Node Pool Design:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  System Pool (전용)                                      │
│  ├── 3x m5.xlarge (4 vCPU, 16 GB)                       │
│  ├── Taints: node-role=system:NoSchedule                 │
│  └── 실행: 모니터링, 인그레스, DNS, cert-manager          │
│                                                          │
│  General Pool (오토스케일링)                               │
│  ├── 3-20x m5.2xlarge (8 vCPU, 32 GB)                   │
│  ├── Taints 없음                                         │
│  └── 실행: 대부분의 애플리케이션 워크로드                   │
│                                                          │
│  High-Memory Pool (오토스케일링)                           │
│  ├── 1-10x r5.2xlarge (8 vCPU, 64 GB)                   │
│  ├── Taints: workload=highmem:NoSchedule                 │
│  └── 실행: 데이터베이스, 캐시, JVM 애플리케이션            │
│                                                          │
│  Spot Pool (비용 최적화)                                  │
│  ├── 0-30x m5.2xlarge (스팟 인스턴스)                     │
│  ├── Taints: workload=spot:NoSchedule                    │
│  └── 실행: 배치 작업, 개발/테스트 워크로드                  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 5. 노드 유지보수

### 5.1 코든, 드레인, 언코든(Cordon, Drain, Uncordon)

```bash
# 단계 1: 노드를 스케줄 불가능으로 표시 (새 파드가 여기에 배치되지 않음)
kubectl cordon worker-node-3
# node/worker-node-3 cordoned

# 단계 2: 노드가 코든되었는지 확인
kubectl get node worker-node-3
# NAME            STATUS                     ROLES    AGE   VERSION
# worker-node-3   Ready,SchedulingDisabled   <none>   90d   v1.29.0

# 단계 3: 모든 파드 드레인 (DaemonSets 제외)
kubectl drain worker-node-3 \
  --ignore-daemonsets \
  --delete-emptydir-data \
  --grace-period=120 \
  --timeout=300s

# 유지보수 수행 (커널 업데이트, 하드웨어 등)
ssh worker-node-3 "sudo apt-get update && sudo apt-get upgrade -y && sudo reboot"

# 단계 4: 유지보수 후 언코든
kubectl uncordon worker-node-3
# node/worker-node-3 uncordoned
```

### 5.2 파드 중단 예산(Pod Disruption Budgets)

PDB는 자발적 중단(드레인, 업그레이드) 동안 워크로드 가용성을 보호합니다:

```yaml
# 드레인 중 항상 최소 2개의 레플리카가 사용 가능하도록 보장
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: web-app-pdb
  namespace: production
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: web-app
---
# 또는 최대 1개 파드만 비가용 허용
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: api-pdb
  namespace: production
spec:
  maxUnavailable: 1
  selector:
    matchLabels:
      app: api-server
---
# StatefulSet용: 비율 기반
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: database-pdb
  namespace: production
spec:
  maxUnavailable: "33%"
  selector:
    matchLabels:
      app: database
```

### 5.3 자동화된 노드 교정(Remediation)

```yaml
# Node Problem Detector + 교정
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-problem-detector
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: node-problem-detector
  template:
    metadata:
      labels:
        app: node-problem-detector
    spec:
      tolerations:
        - operator: Exists
      containers:
        - name: node-problem-detector
          image: registry.k8s.io/node-problem-detector/node-problem-detector:v0.8.14
          command:
            - /node-problem-detector
            - --logtostderr
            - --config.system-log-monitor=/config/kernel-monitor.json
          volumeMounts:
            - name: log
              mountPath: /var/log
              readOnly: true
            - name: kmsg
              mountPath: /dev/kmsg
              readOnly: true
          resources:
            requests:
              cpu: 20m
              memory: 32Mi
            limits:
              cpu: 100m
              memory: 128Mi
      volumes:
        - name: log
          hostPath:
            path: /var/log/
        - name: kmsg
          hostPath:
            path: /dev/kmsg
```

---

## 6. 인증서 관리와 순환

### 6.1 Kubernetes PKI 개요

```
Kubernetes Certificate Architecture:
┌─────────────────────────────────────────────────────────────┐
│  /etc/kubernetes/pki/                                       │
│  ├── ca.crt / ca.key                 (클러스터 CA, 10년)     │
│  ├── apiserver.crt / apiserver.key   (API 서버, 1년)         │
│  ├── apiserver-kubelet-client.crt    (API→kubelet, 1년)      │
│  ├── front-proxy-ca.crt / key       (프론트 프록시 CA)        │
│  ├── front-proxy-client.crt / key   (집계 계층)              │
│  ├── sa.pub / sa.key                (ServiceAccount 서명)    │
│  └── etcd/                                                   │
│      ├── ca.crt / ca.key            (etcd CA, 10년)          │
│      ├── server.crt / server.key    (etcd 서버, 1년)         │
│      ├── peer.crt / peer.key        (etcd 피어, 1년)         │
│      └── healthcheck-client.crt/key (etcd 헬스, 1년)         │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 인증서 만료 확인

```bash
# kubeadm이 관리하는 모든 인증서 확인
sudo kubeadm certs check-expiration

# OpenSSL로 특정 인증서 확인
openssl x509 -in /etc/kubernetes/pki/apiserver.crt -noout -dates -subject -issuer
```

### 6.3 인증서 순환

```bash
# kubeadm이 관리하는 모든 인증서 갱신
sudo kubeadm certs renew all

# 특정 인증서 갱신
sudo kubeadm certs renew apiserver

# 갱신 후 컨트롤 플레인 컴포넌트 재시작
# (정적 파드는 매니페스트가 변경되면 자동 재시작)
sudo crictl pods --name kube-apiserver -q | xargs sudo crictl stopp
sudo crictl pods --name kube-controller-manager -q | xargs sudo crictl stopp
sudo crictl pods --name kube-scheduler -q | xargs sudo crictl stopp

# kubeconfig 파일 업데이트
sudo kubeadm kubeconfig user --client-name=admin --org=system:masters \
  > /etc/kubernetes/admin.conf

# 자동 kubelet 인증서 순환 활성화
# /var/lib/kubelet/config.yaml에서:
# rotateCertificates: true
# serverTLSBootstrap: true
```

### 6.4 인증서 만료 모니터링

```yaml
# 인증서 만료에 대한 Prometheus 알림
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: cert-expiration-alerts
  namespace: monitoring
spec:
  groups:
    - name: certificates
      rules:
        - alert: KubernetesCertExpiringSoon
          expr: |
            apiserver_client_certificate_expiration_seconds_count > 0
            and
            histogram_quantile(0.01,
              rate(apiserver_client_certificate_expiration_seconds_bucket[5m])
            ) < 604800
          for: 10m
          labels:
            severity: critical
          annotations:
            summary: "Kubernetes client certificate expires within 7 days"

        - alert: KubeletCertExpiring
          expr: |
            kubelet_certificate_manager_server_expiration_renew_errors > 0
          for: 15m
          labels:
            severity: warning
          annotations:
            summary: "Kubelet certificate renewal failing on {{ $labels.node }}"
```

---

## 7. 프로덕션 이슈 트러블슈팅

### 7.1 체계적 트러블슈팅 흐름

```
Troubleshooting Decision Tree:
                    ┌─────────────┐
                    │  Pod Issue?  │
                    └──────┬──────┘
              ┌────────────┼────────────┐
         Pending      CrashLoop     Evicted
              │            │            │
      ┌───────┴──┐   ┌────┴────┐  ┌───┴────┐
      │Scheduling│   │ OOM?    │  │Resource │
      │ Failure  │   │ Config? │  │Pressure?│
      └──────────┘   │ Deps?   │  └────────┘
                      └─────────┘
```

### 7.2 일반 트러블슈팅 명령어

```bash
# Pending 상태의 파드: 이벤트와 노드 리소스 확인
kubectl describe pod <pod-name> -n <namespace>
kubectl get events --sort-by=.lastTimestamp -n <namespace>
kubectl describe nodes | grep -A 5 "Allocated resources"

# CrashLoopBackOff 상태의 파드: 로그 확인
kubectl logs <pod-name> -n <namespace> --previous
kubectl logs <pod-name> -n <namespace> -c <container-name>

# 네트워크 연결 이슈
kubectl run debug --image=nicolaka/netshoot --rm -it -- \
  bash -c "nslookup kubernetes.default && curl -v http://my-service:8080/health"

# 노드 디버그
kubectl debug node/worker-node-1 -it --image=ubuntu -- bash

# API 서버 응답성 확인
kubectl get --raw /healthz
kubectl get --raw /readyz?verbose

# etcd 헬스 체크
kubectl -n kube-system exec etcd-master-0 -- \
  etcdctl --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  endpoint health
```

### 7.3 OOMKilled 조사

```bash
# OOMKilled 파드 찾기
kubectl get pods --all-namespaces -o json | \
  jq -r '.items[] |
    select(.status.containerStatuses[]?.lastState.terminated.reason == "OOMKilled") |
    "\(.metadata.namespace)/\(.metadata.name)"'

# 메모리 제한 대 실제 사용량 확인
kubectl top pod <pod-name> -n <namespace>
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.spec.containers[0].resources.limits.memory}'

# 노드 수준 OOM 이벤트 확인
kubectl get events --all-namespaces --field-selector reason=OOMKilling
```

### 7.4 DNS 트러블슈팅

```bash
# CoreDNS 실행 확인
kubectl get pods -n kube-system -l k8s-app=kube-dns

# 파드에서 DNS 해석 테스트
kubectl run dns-test --image=busybox:1.36 --rm -it -- \
  nslookup kubernetes.default.svc.cluster.local

# CoreDNS 로그 확인
kubectl logs -n kube-system -l k8s-app=kube-dns --tail=100

# 파드 내부의 DNS 구성 확인
kubectl exec <pod-name> -- cat /etc/resolv.conf

# 외부 DNS 해석 테스트
kubectl run dns-test --image=busybox:1.36 --rm -it -- \
  nslookup google.com
```

---

## 8. 성능 튜닝

### 8.1 API 서버 튜닝

```yaml
# /etc/kubernetes/manifests/kube-apiserver.yaml (주요 플래그)
spec:
  containers:
    - command:
        - kube-apiserver
        # 동시 요청 제한 증가
        - --max-requests-inflight=800          # 기본값: 400
        - --max-mutating-requests-inflight=400 # 기본값: 200
        # 감시 캐시 크기 (리소스 타입별)
        - --watch-cache-sizes=pods#1000,nodes#100,services#100
        # API 우선순위 및 공정성 활성화
        - --enable-priority-and-fairness=true
        # 감사 로깅 (성능 영향 — 선택적으로)
        - --audit-policy-file=/etc/kubernetes/audit-policy.yaml
        - --audit-log-path=/var/log/kubernetes/audit.log
        - --audit-log-maxage=7
        - --audit-log-maxbackup=3
        - --audit-log-maxsize=100
```

### 8.2 스케줄러 튜닝

```yaml
# 스케줄러 구성
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
  - schedulerName: default-scheduler
    plugins:
      score:
        enabled:
          - name: NodeResourcesBalancedAllocation
            weight: 1
          - name: NodeResourcesFit
            weight: 2
    pluginConfig:
      - name: NodeResourcesFit
        args:
          scoringStrategy:
            type: MostAllocated    # 노드를 빽빽이 채움 (비용 절감)
            # type: LeastAllocated # 노드에 분산 (성능)
            resources:
              - name: cpu
                weight: 1
              - name: memory
                weight: 1
# 점수를 매길 노드 비율 (대규모 클러스터)
percentageOfNodesToScore: 50    # 기본값: 0 (자동)
```

### 8.3 Kubelet 튜닝

```yaml
# /var/lib/kubelet/config.yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
# 파드 퇴거 임계값
evictionHard:
  memory.available: "200Mi"
  nodefs.available: "10%"
  imagefs.available: "15%"
evictionSoft:
  memory.available: "500Mi"
  nodefs.available: "15%"
evictionSoftGracePeriod:
  memory.available: "1m30s"
  nodefs.available: "1m"
# 노드당 최대 파드 수 (기본값 110)
maxPods: 110
# 이미지 가비지 컬렉션
imageGCHighThresholdPercent: 85
imageGCLowThresholdPercent: 80
# 레지스트리 풀 동시성
serializeImagePulls: false
maxParallelImagePulls: 5
# 파드 시작 최적화
podsPerCore: 0    # 0 = 제한 없음
```

### 8.4 etcd 성능 튜닝

```bash
# 주요 etcd 튜닝 매개변수
# --heartbeat-interval=100      (기본값, 하트비트 간 ms)
# --election-timeout=1000       (기본값, 새 선출 전 ms)
# --snapshot-count=10000        (기본값, 스냅샷 간 작업 수)
# --quota-backend-bytes=8589934592  (8GB 최대 DB 크기)
# --auto-compaction-mode=periodic
# --auto-compaction-retention=8h

# 디스크 I/O 성능 확인 (etcd는 저지연 스토리지 필요)
# WAL fsync는 10ms 미만이어야 함
fio --name=etcd-test --ioengine=sync --rw=write \
  --bs=2300 --numjobs=1 --size=22m --runtime=60 \
  --directory=/var/lib/etcd \
  --fdatasync=1
```

---

## 9. Kubernetes의 SLA와 SLO

### 9.1 플랫폼 SLO 정의

```
Kubernetes Platform SLOs:
┌──────────────────────────────────────────────────────────────┐
│  SLI (지표)               │ SLO (목표)         │ 측정 방법    │
├───────────────────────────┼────────────────────┼─────────────┤
│  API 서버 가용성           │ 99.95% / 월       │ /readyz     │
│  API 서버 지연(p99)       │ < 1s (비목록)      │ Prometheus  │
│  스케줄링 지연(p99)       │ < 5s (파드→바인딩) │ Prometheus  │
│  노드 준비 상태            │ 99.9% of nodes    │ node cond.  │
│  etcd 리더 안정성          │ < 2 elections/day │ etcd metrics│
│  DNS 해석 성공률           │ 99.99%            │ CoreDNS     │
│  인그레스 에러율            │ < 0.1% 5xx       │ Ingress logs│
│  디플로이먼트 롤아웃 시간   │ < 5 min for 100  │ events      │
│                            │   replica deploy  │             │
└───────────────────────────┴────────────────────┴─────────────┘
```

### 9.2 에러 예산(Error Budget)

```
Error Budget Calculation:
  월간 SLO: 99.95% 가용성
  30일 총 분: 43,200
  허용 다운타임: 43,200 * 0.05% = 21.6분/월

  인시던트로 15분 소비 시:
  남은 예산: 6.6분
  예산 소비: 69.4%
  조치: 비중요 변경 동결, 안정성에 집중
```

### 9.3 SLO 기반 알림

```yaml
# 다중 윈도우, 다중 번 레이트 SLO 알림
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: api-server-slo
  namespace: monitoring
spec:
  groups:
    - name: api-server-slo
      rules:
        # 에러 비율 기록 규칙
        - record: apiserver:error_ratio:rate5m
          expr: |
            sum(rate(apiserver_request_total{code=~"5.."}[5m]))
            /
            sum(rate(apiserver_request_total[5m]))

        # 빠른 번 (1시간에 월간 예산의 2%)
        - alert: APIServerHighErrorRate
          expr: |
            apiserver:error_ratio:rate5m > (14.4 * 0.0005)
          for: 2m
          labels:
            severity: critical
            slo: api-server-availability
          annotations:
            summary: "API server burning error budget 14.4x faster than allowed"

        # 느린 번 (3일에 월간 예산의 10%)
        - alert: APIServerElevatedErrorRate
          expr: |
            apiserver:error_ratio:rate5m > (1.0 * 0.0005)
          for: 1h
          labels:
            severity: warning
            slo: api-server-availability
          annotations:
            summary: "API server error rate elevated above SLO threshold"
```

### 9.4 플랫폼 옵저버빌리티 대시보드

Kubernetes 플랫폼 SLO 대시보드의 필수 패널:

```
Dashboard Layout:
┌────────────────────────────┬────────────────────────────┐
│  API 서버 가용성            │  API 서버 지연(p99)         │
│  [99.97% 현재]             │  [243ms 현재]               │
│  SLO: 99.95%               │  SLO: < 1000ms             │
├────────────────────────────┼────────────────────────────┤
│  에러 예산 잔여             │  스케줄링 지연(p99)         │
│  [월간 예산의 72%]          │  [1.2s 현재]               │
│  ████████░░ 72%            │  SLO: < 5000ms             │
├────────────────────────────┼────────────────────────────┤
│  etcd 리더 변경             │  노드 준비 상태            │
│  [24시간 내 0회]           │  [100% 노드 Ready]         │
│  SLO: < 2/day             │  SLO: 99.9%                │
├────────────────────────────┼────────────────────────────┤
│  DNS 해석 성공률            │  인그레스 5xx 비율          │
│  [99.998%]                 │  [0.02%]                   │
│  SLO: 99.99%              │  SLO: < 0.1%              │
└────────────────────────────┴────────────────────────────┘
```

---

## 연습문제

### 연습문제 1: 클러스터 업그레이드 계획

3개의 컨트롤 플레인 노드와 15개의 워커 노드를 가진 Kubernetes v1.27.8 프로덕션 클러스터가 있습니다. 목표 버전은 v1.29.2입니다. 다음을 포함하는 상세한 업그레이드 계획을 작성하세요: (a) 필요한 중간 버전, (b) 작업 순서, (c) 워커 노드 업그레이드 시 PDB 처리 방법, (d) 롤백 전략.

<details><summary>정답 보기</summary>

**업그레이드 계획: v1.27.8에서 v1.29.2로**

**(a) 필요한 중간 버전:**
Kubernetes는 한 번에 하나의 마이너 버전만 업그레이드할 수 있습니다. 경로:
- v1.27.8 → v1.28.latest (예: v1.28.6)
- v1.28.6 → v1.29.2

**(b) 각 마이너 버전 업그레이드의 작업 순서:**

```bash
# 단계 1: v1.27.8 → v1.28.6
# 1. etcd 백업
ETCDCTL_API=3 etcdctl snapshot save /backup/pre-upgrade-v1.27.8.db \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key

# 2. 폐기된 API 확인
kubectl get --raw /metrics | grep apiserver_requested_deprecated_apis

# 3. 첫 번째 컨트롤 플레인 노드 업그레이드
sudo apt-get install -y kubeadm=1.28.6-1.1
sudo kubeadm upgrade plan
sudo kubeadm upgrade apply v1.28.6
sudo apt-get install -y kubelet=1.28.6-1.1 kubectl=1.28.6-1.1
sudo systemctl daemon-reload && sudo systemctl restart kubelet

# 4. 나머지 컨트롤 플레인 노드 업그레이드
for node in cp-2 cp-3; do
  ssh $node "sudo apt-get install -y kubeadm=1.28.6-1.1"
  ssh $node "sudo kubeadm upgrade node"
  ssh $node "sudo apt-get install -y kubelet=1.28.6-1.1 kubectl=1.28.6-1.1"
  ssh $node "sudo systemctl daemon-reload && sudo systemctl restart kubelet"
done

# 5. 워커 노드 업그레이드 (PDB를 준수하며 한 번에 2개씩)
for node in worker-{1..15}; do
  kubectl drain $node --ignore-daemonsets --delete-emptydir-data --timeout=300s
  ssh $node "sudo apt-get install -y kubeadm=1.28.6-1.1 kubelet=1.28.6-1.1"
  ssh $node "sudo kubeadm upgrade node"
  ssh $node "sudo systemctl daemon-reload && sudo systemctl restart kubelet"
  kubectl uncordon $node
  # 노드가 Ready 상태가 되고 파드가 안정될 때까지 대기
  kubectl wait --for=condition=Ready node/$node --timeout=120s
done

# 단계 2: v1.28.6 → v1.29.2 반복
```

**(c) PDB 처리:**
- 드레인은 PDB를 자동으로 준수합니다. PDB가 드레인을 차단하면 `--timeout`까지 대기합니다.
- 워커를 한 번에 2개씩 업그레이드 (15개 중 2개 = 한 번에 13%).
- minAvailable=2, replicas=3인 중요 워크로드의 경우 한 번에 하나의 워커만 드레인 가능합니다.
- 시작 전 `kubectl get pdb --all-namespaces`로 제약 조건을 확인하세요.

**(d) 롤백 전략:**
- 컨트롤 플레인 업그레이드 실패 시: 업그레이드 전 스냅샷에서 etcd를 복원하고 이전 kubeadm/kubelet 버전을 재설치합니다.
- 워커 업그레이드 실패 시: 이전 버전의 노드를 언코든합니다; 버전 스큐 정책에 따라 API 서버 v1.28과 kubelet v1.27이 허용됩니다.
- 성공적인 업그레이드 후 최소 48시간 동안 etcd 스냅샷을 보관합니다.

</details>

### 연습문제 2: etcd 백업 및 복원 드릴

다음을 수행하는 완전한 드릴 스크립트를 작성하세요: (a) 테스트 네임스페이스와 ConfigMap 생성, (b) etcd 스냅샷 생성, (c) 네임스페이스 삭제, (d) 스냅샷 복원, (e) 네임스페이스와 ConfigMap이 다시 존재하는지 확인.

<details><summary>정답 보기</summary>

```bash
#!/usr/bin/env bash
set -euo pipefail

ETCD_OPTS="--endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key"

BACKUP_FILE="/tmp/etcd-drill-$(date +%s).db"

echo "=== 단계 1: 테스트 리소스 생성 ==="
kubectl create namespace dr-drill-test
kubectl create configmap drill-config \
  -n dr-drill-test \
  --from-literal=key1=value1 \
  --from-literal=key2=value2
kubectl get configmap drill-config -n dr-drill-test -o yaml
echo "테스트 리소스 생성 완료."

echo "=== 단계 2: etcd 스냅샷 생성 ==="
ETCDCTL_API=3 etcdctl $ETCD_OPTS snapshot save "$BACKUP_FILE"
ETCDCTL_API=3 etcdctl snapshot status "$BACKUP_FILE" --write-table
echo "스냅샷 저장: $BACKUP_FILE"

echo "=== 단계 3: 테스트 리소스 삭제 ==="
kubectl delete namespace dr-drill-test --wait=true
echo "네임스페이스 삭제. 확인 중..."
kubectl get namespace dr-drill-test 2>&1 && echo "오류: 네임스페이스가 아직 존재" || echo "확인: 네임스페이스 삭제됨."

echo "=== 단계 4: etcd 스냅샷 복원 ==="
sudo mv /etc/kubernetes/manifests/kube-apiserver.yaml /tmp/kube-apiserver.yaml.bak
sleep 10

sudo rm -rf /var/lib/etcd-restored
ETCDCTL_API=3 etcdctl snapshot restore "$BACKUP_FILE" \
  --data-dir=/var/lib/etcd-restored \
  --name=$(hostname) \
  --initial-cluster="$(hostname)=https://127.0.0.1:2380" \
  --initial-advertise-peer-urls=https://127.0.0.1:2380

sudo mv /var/lib/etcd /var/lib/etcd-old
sudo mv /var/lib/etcd-restored /var/lib/etcd
sudo chown -R etcd:etcd /var/lib/etcd 2>/dev/null || true

sudo systemctl restart etcd 2>/dev/null || true

sudo mv /tmp/kube-apiserver.yaml.bak /etc/kubernetes/manifests/kube-apiserver.yaml
echo "API 서버 복구 대기 중..."
until kubectl get nodes &>/dev/null; do sleep 2; done
echo "API 서버 복구됨."

echo "=== 단계 5: 복원된 리소스 확인 ==="
kubectl get namespace dr-drill-test
kubectl get configmap drill-config -n dr-drill-test -o yaml

VALUE=$(kubectl get configmap drill-config -n dr-drill-test \
  -o jsonpath='{.data.key1}')
if [ "$VALUE" = "value1" ]; then
  echo "성공: 복원 후 ConfigMap 데이터 확인됨."
else
  echo "실패: ConfigMap 데이터 불일치. 값: $VALUE"
  exit 1
fi

echo "=== 정리 ==="
kubectl delete namespace dr-drill-test
rm -f "$BACKUP_FILE"
sudo rm -rf /var/lib/etcd-old
echo "DR 드릴 완료."
```

</details>

### 연습문제 3: PDB 제약 조건이 있는 노드 드레인

3개의 워커 노드가 있고, 각각 `web-app`의 2개 레플리카를 실행합니다 (총 6개, `minAvailable: 4`인 PDB). 3개 노드를 하나씩 드레인하는 절차를 작성하고 각 단계에서 무엇이 일어나는지 설명하세요. 2개 노드를 동시에 드레인하면 어떻게 됩니까?

<details><summary>정답 보기</summary>

**설정:**
- 3개 워커 노드: node-1, node-2, node-3
- 6개의 web-app 레플리카가 노드에 분산 (노드당 2개)
- PDB: minAvailable=4 (최대 2개 파드가 비가용 가능)

**단계별 드레인 절차:**

```bash
# 단계 1: node-1 드레인
kubectl drain node-1 --ignore-daemonsets --delete-emptydir-data

# 일어나는 일:
# - node-1이 코든됨 (새 파드 스케줄링 안됨)
# - node-1의 web-app 파드 2개 퇴거
# - PDB 확인: 6 - 2 = 4 가용 >= minAvailable(4) -> 허용
# - 스케줄러가 퇴거된 2개 파드를 node-2와 node-3에 배치
# - 분배: node-1=0, node-2=3, node-3=3
# 파드가 Running 상태가 될 때까지 대기
kubectl wait --for=condition=Ready pod -l app=web-app --timeout=120s

# 단계 2: node-2 드레인
kubectl drain node-2 --ignore-daemonsets --delete-emptydir-data

# 일어나는 일:
# - node-2의 web-app 파드 3개 퇴거 필요
# - 처음 2개 PDB 확인: 6 - 2 = 4 >= 4 -> 허용
# - 2개 퇴거 후: 4개 남음, 3번째 퇴거 시도
# - PDB 확인: 4 - 1 = 3 < 4 -> 차단
# - 드레인이 퇴거된 파드가 node-3에서 Ready가 될 때까지 대기
# - Ready 되면: 총 6개 실행, 3번째 퇴거
# - 분배: node-1=0, node-2=0, node-3=6

# 단계 3: node-3 드레인
kubectl drain node-3 --ignore-daemonsets --delete-emptydir-data

# 일어나는 일:
# - node-3의 web-app 파드 6개 퇴거 필요
# - PDB 차단: 6 - 1 = 5, 6 - 2 = 4 (2개까지 OK)
# - 2개 퇴거 후, 스케줄 가능한 노드 없음 (모두 코든됨)
# - 파드가 Pending 상태로 유지
# - PDB가 추가 퇴거 차단: 4 - 1 = 3 < 4
# - 드레인이 타임아웃까지 무기한 중단

# 해결: 먼저 노드를 언코든
kubectl uncordon node-1
# 이제 퇴거된 파드가 node-1에 스케줄되어 드레인 진행 가능
```

**2개 노드를 동시에 드레인하면?**

```bash
# node-1과 node-2 동시 드레인
kubectl drain node-1 --ignore-daemonsets &
kubectl drain node-2 --ignore-daemonsets &

# 경합 조건:
# - 양쪽 모두 동시에 2개 파드 퇴거 시도
# - 총 퇴거 시도: 4개 동시
# - PDB: 6 - 4 = 2 < minAvailable(4) -> 일부 퇴거 차단
# - API 서버의 퇴거 API가 파드별로 원자적으로 PDB 확인
# - 결과: 일부 퇴거 성공, 나머지 재시도
# - 결국 양쪽 드레인 완료되지만 직렬 PDB 준수 퇴거와
#   재스케줄링 대기로 인해 더 오래 걸림
# - 스케줄링이 촉박하면 한쪽 드레인이 타임아웃될 수 있음
```

</details>

### 연습문제 4: 인증서 만료 대응

모니터링 및 교정 절차를 작성하세요: (a) Kubernetes 컴포넌트 인증서가 30일 이내에 만료될 때 발동되는 Prometheus 알림 규칙, (b) 모든 인증서를 갱신하고 영향받는 컴포넌트를 재시작하는 셸 스크립트, (c) 갱신 후 검증 체크리스트.

<details><summary>정답 보기</summary>

**(a) Prometheus 알림 규칙:**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: cert-expiry-alerts
  namespace: monitoring
spec:
  groups:
    - name: certificate-expiry
      rules:
        - alert: KubeCertExpiry30Days
          expr: |
            apiserver_client_certificate_expiration_seconds_count > 0
            and on()
            (
              apiserver_client_certificate_expiration_seconds_bucket{le="2592000"}
              /
              ignoring(le) apiserver_client_certificate_expiration_seconds_count
            ) > 0
          for: 1h
          labels:
            severity: warning
          annotations:
            summary: "A Kubernetes client certificate expires within 30 days"
            runbook: "https://wiki.internal/runbooks/cert-renewal"

        - alert: KubeCertExpiry7Days
          expr: |
            apiserver_client_certificate_expiration_seconds_count > 0
            and on()
            (
              apiserver_client_certificate_expiration_seconds_bucket{le="604800"}
              /
              ignoring(le) apiserver_client_certificate_expiration_seconds_count
            ) > 0
          for: 10m
          labels:
            severity: critical
          annotations:
            summary: "A Kubernetes client certificate expires within 7 days"
```

**(b) 갱신 스크립트:**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== 갱신 전: 현재 만료 확인 ==="
sudo kubeadm certs check-expiration

echo "=== 현재 인증서 백업 ==="
BACKUP_DIR="/etc/kubernetes/pki-backup-$(date +%Y%m%d)"
sudo cp -r /etc/kubernetes/pki "$BACKUP_DIR"
echo "백업 저장: $BACKUP_DIR"

echo "=== 모든 인증서 갱신 ==="
sudo kubeadm certs renew all

echo "=== kubeconfig 파일 재생성 ==="
sudo kubeadm kubeconfig user --client-name=admin \
  --org=system:masters > /tmp/admin.conf
sudo cp /tmp/admin.conf /etc/kubernetes/admin.conf
sudo cp /etc/kubernetes/admin.conf ~/.kube/config

echo "=== 컨트롤 플레인 컴포넌트 재시작 ==="
sudo crictl pods --name kube-apiserver -q | \
  xargs -r sudo crictl stopp
sudo crictl pods --name kube-controller-manager -q | \
  xargs -r sudo crictl stopp
sudo crictl pods --name kube-scheduler -q | \
  xargs -r sudo crictl stopp
sudo crictl pods --name etcd -q | \
  xargs -r sudo crictl stopp

echo "=== 컴포넌트 재시작 대기 ==="
sleep 15
until kubectl get nodes &>/dev/null; do
  echo "API 서버 대기 중..."
  sleep 5
done

echo "=== 갱신 후: 새 만료 확인 ==="
sudo kubeadm certs check-expiration
echo "인증서 갱신 완료."
```

**(c) 갱신 후 검증 체크리스트:**

```bash
# 1. 모든 인증서가 갱신되었는지 확인
sudo kubeadm certs check-expiration | grep -c "364d"

# 2. API 서버 응답 확인
kubectl get --raw /healthz && echo "API server healthy"

# 3. 모든 노드가 Ready인지 확인
kubectl get nodes -o wide

# 4. controller-manager와 scheduler 실행 확인
kubectl get pods -n kube-system -l tier=control-plane

# 5. etcd 클러스터 헬스 확인
kubectl -n kube-system exec etcd-$(hostname) -- \
  etcdctl endpoint health $ETCD_OPTS

# 6. 워크로드 작업 테스트
kubectl run cert-test --image=busybox:1.36 --rm -it -- echo "Workloads OK"

# 7. 서비스 어카운트 토큰 생성 확인
kubectl create serviceaccount cert-test-sa --dry-run=server -o yaml

# 8. 경고 이벤트 확인
kubectl get events --all-namespaces --field-selector type=Warning \
  --sort-by=.lastTimestamp | head -20
```

</details>

### 연습문제 5: SLO 정의와 에러 예산

Kubernetes 플랫폼 팀을 위한 완전한 SLO 프레임워크를 정의하세요. 포함 항목: (a) 30일 기간에 대한 SLO 대상이 있는 3개 SLI, (b) 각각의 에러 예산 계산, (c) 하나의 SLI에 대한 Prometheus 기록 규칙, (d) 에러 예산 정책 (50%, 75%, 100% 예산 소비 시 조치 설명).

<details><summary>정답 보기</summary>

**(a) SLI와 SLO:**

| SLI | 측정 방법 | SLO 대상 (30일) |
|-----|-------------|---------------------|
| API 서버 가용성 | 성공(비-5xx) /readyz 응답 비율 | 99.95% |
| 파드 스케줄링 지연 | 파드 생성부터 노드 바인딩까지 p99 시간 | < 5초 |
| 노드 헬스 | 임의 시점에서 Ready 상태인 노드 % | 99.9% |

**(b) 에러 예산 (30일 = 43,200분):**

```
API 서버 가용성:
  예산 = 100% - 99.95% = 0.05%
  허용 다운타임 = 43,200 * 0.0005 = 21.6분/월

파드 스케줄링 지연:
  예산 = p99 > 5s인 요청
  월 10,000개 파드 생성 시, 50개가 5s 초과 허용 (0.5% 예산)

노드 헬스:
  예산 = 100% - 99.9% = 0.1%
  노드당 (10개 노드, 43,200분): 노드당 43.2분의 NotReady
```

**(c) API 서버 가용성에 대한 Prometheus 기록 규칙:**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: platform-slo-recording
  namespace: monitoring
spec:
  groups:
    - name: platform-slo
      interval: 30s
      rules:
        # 5분 에러율
        - record: platform:apiserver:error_rate:5m
          expr: |
            1 - (
              sum(rate(apiserver_request_total{code!~"5.."}[5m]))
              /
              sum(rate(apiserver_request_total[5m]))
            )

        # 1시간 에러율 (번 레이트 계산용)
        - record: platform:apiserver:error_rate:1h
          expr: |
            1 - (
              sum(rate(apiserver_request_total{code!~"5.."}[1h]))
              /
              sum(rate(apiserver_request_total[1h]))
            )

        # 30일 가용성 (롤링)
        - record: platform:apiserver:availability:30d
          expr: |
            1 - (
              sum(increase(apiserver_request_total{code=~"5.."}[30d]))
              /
              sum(increase(apiserver_request_total[30d]))
            )

        # 에러 예산 잔여 (백분율)
        - record: platform:apiserver:error_budget_remaining:30d
          expr: |
            1 - (
              (1 - platform:apiserver:availability:30d)
              /
              (1 - 0.9995)
            )
```

**(d) 에러 예산 정책:**

```
예산 소비    | 상태    | 조치
-----------+---------+------------------------------------------
< 50%       | 초록    | 정상 운영. 기능을 자유롭게 배포.
            |         | 주간 스탠드업에서 SLO 리뷰.
            |         |
50-75%      | 노랑    | 변경에 대한 리뷰 엄격화.
            |         | 모든 배포에 롤백 계획 필수.
            |         | 상위 에러 기여 요인 조사.
            |         | 스탠드업에서 매일 SLO 확인.
            |         |
75-100%     | 주황    | 비중요 변경 동결.
            |         | 스프린트의 50%를 안정성에 투자.
            |         | 모든 새 에러에 대해 포스트모템 리뷰.
            |         | 엔지니어링 리더십에 에스컬레이션.
            |         |
> 100%      | 빨강    | 전면 변경 동결 (긴급만 허용).
            |         | 엔지니어링 100%를 안정성에 투입.
            |         | 경영진 인시던트 리뷰.
            |         | 예산이 회복될 때까지 새 기능 없음.
            |         | 아키텍처 변경 검토.
```

</details>

---

**이전**: [16. Kubernetes API 프로그래밍](./16_Kubernetes_API_Programming.md) | **다음**: [18. ML을 위한 Kubernetes](./18_Kubernetes_for_ML.md)
