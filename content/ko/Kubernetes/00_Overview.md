# Kubernetes

## 토픽 개요

쿠버네티스(Kubernetes, K8s)는 컨테이너화된 애플리케이션의 배포, 스케일링, 관리를 자동화하는 업계 표준 컨테이너 오케스트레이션(Container Orchestration) 플랫폼입니다. 원래 구글(Google)에서 설계하고 현재는 클라우드 네이티브 컴퓨팅 재단(Cloud Native Computing Foundation, CNCF)에서 관리하는 쿠버네티스는 소규모 스타트업부터 대기업까지 모든 규모의 현대 클라우드 네이티브 인프라의 기반이 되었습니다.

이 토픽은 기본적인 `kubectl` 명령어를 훨씬 넘어섭니다. 쿠버네티스 아키텍처 내부 구조, CNI 플러그인을 활용한 고급 네트워킹, 스토리지 시스템, 보안 강화, 커스텀 리소스 정의(Custom Resource Definitions), 오퍼레이터(Operators), 어드미션 컨트롤러(Admission Controllers), 멀티 클러스터 관리, client-go를 활용한 API 프로그래밍, 프로덕션 운영, 쿠버네티스 위의 ML 워크로드를 다룹니다. 목표는 프로덕션 환경에서 쿠버네티스 클러스터를 설계, 운영, 확장할 수 있는 깊고 전이 가능한 지식을 구축하는 것입니다.

이 토픽은 컨테이너(Docker), 리눅스(Linux) 기초, 기본 네트워킹에 대한 이해를 전제로 합니다. Docker, Linux, Networking 토픽을 먼저 완료하는 것을 권장합니다.

## 학습 경로

```
Architecture & Core              Networking & Storage             Security
─────────────────                ─────────────────                ─────────────────
01 Architecture        ★★       03 Networking          ★★★      06 RBAC & Security   ★★★
02 Workloads           ★★       04 Storage             ★★       07 Ingress/Gateway   ★★★
                                 05 Config & Secrets    ★★       08 CNI Advanced      ★★★★

Package & Extend                 Operations                       Project
─────────────────                ─────────────────                ─────────────────
09 Helm & Kustomize    ★★★      13 Autoscaling         ★★★      19 Capstone          ★★★★
10 CRDs                ★★★★     14 Observability       ★★★
11 Operators           ★★★★     15 Multi-Cluster       ★★★★     ML & API
12 Admission Ctrl      ★★★★     16 API Programming     ★★★★     ─────────────────
                                 17 Production Ops      ★★★      18 K8s for ML        ★★★
```

## 레슨 목록

| # | 레슨 | 난이도 | 핵심 개념 |
|---|--------|------------|--------------|
| 01 | [아키텍처 심층 분석](./01_Architecture_Deep_Dive.md) | ⭐⭐ | 컨트롤 플레인(Control Plane), etcd, API 서버, 스케줄러(Scheduler), kubelet |
| 02 | [워크로드 리소스](./02_Workload_Resources.md) | ⭐⭐ | 디플로이먼트(Deployments), 스테이트풀셋(StatefulSets), 데몬셋(DaemonSets), 잡(Jobs), 크론잡(CronJobs) |
| 03 | [네트워킹 기초](./03_Networking_Fundamentals.md) | ⭐⭐⭐ | 서비스(Services), DNS, kube-proxy, iptables vs IPVS |
| 04 | [스토리지와 영속성](./04_Storage_and_Persistence.md) | ⭐⭐ | PV, PVC, 스토리지클래스(StorageClasses), CSI 드라이버, 동적 프로비저닝(Dynamic Provisioning) |
| 05 | [구성과 시크릿](./05_Configuration_and_Secrets.md) | ⭐⭐ | 컨피그맵(ConfigMaps), 시크릿(Secrets), external-secrets-operator, sealed-secrets |
| 06 | [RBAC과 보안](./06_RBAC_and_Security.md) | ⭐⭐⭐ | RBAC, 파드 보안 표준(Pod Security Standards), OPA/Gatekeeper |
| 07 | [인그레스와 게이트웨이 API](./07_Ingress_and_Gateway_API.md) | ⭐⭐⭐ | 인그레스 컨트롤러(Ingress Controllers), 게이트웨이 API(Gateway API), TLS 종료(TLS Termination) |
| 08 | [CNI와 고급 네트워킹](./08_CNI_and_Advanced_Networking.md) | ⭐⭐⭐⭐ | Calico, Cilium, eBPF 네트워킹, 고급 네트워크폴리시(NetworkPolicy) |
| 09 | [Helm과 Kustomize](./09_Helm_and_Kustomize.md) | ⭐⭐⭐ | 패키지 관리, 오버레이(Overlays), Helm 차트 개발 |
| 10 | [커스텀 리소스 정의](./10_Custom_Resource_Definitions.md) | ⭐⭐⭐⭐ | CRD 설계, 검증, 버전 관리, 변환 웹훅(Conversion Webhooks) |
| 11 | [오퍼레이터](./11_Operators.md) | ⭐⭐⭐⭐ | 오퍼레이터 패턴(Operator Pattern), operator-sdk, kubebuilder, 라이프사이클 관리 |
| 12 | [어드미션 컨트롤러](./12_Admission_Controllers.md) | ⭐⭐⭐⭐ | 검증/변형 웹훅(Validating/Mutating Webhooks), OPA Gatekeeper 정책 |
| 13 | [오토스케일링](./13_Autoscaling.md) | ⭐⭐⭐ | HPA, VPA, 클러스터 오토스케일러(Cluster Autoscaler), KEDA 이벤트 기반 스케일링 |
| 14 | [옵저버빌리티](./14_Observability.md) | ⭐⭐⭐ | Prometheus 스택, EFK/Loki 로깅, 분산 트레이싱 |
| 15 | [멀티 클러스터](./15_Multi_Cluster.md) | ⭐⭐⭐⭐ | 페더레이션(Federation), 멀티 클러스터 서비스 메시, Liqo, Submariner |
| 16 | [쿠버네티스 API 프로그래밍](./16_Kubernetes_API_Programming.md) | ⭐⭐⭐⭐ | client-go, 인포머(Informers), 동적 클라이언트(Dynamic Client), controller-runtime |
| 17 | [프로덕션 운영](./17_Production_Operations.md) | ⭐⭐⭐ | 업그레이드, etcd 백업/복구, 재해 복구, 용량 계획 |
| 18 | [ML을 위한 쿠버네티스](./18_Kubernetes_for_ML.md) | ⭐⭐⭐ | GPU 스케줄링, Kubeflow, K8s에서의 모델 서빙 |
| 19 | [캡스톤: 프로덕션 클러스터](./19_Capstone_Production_Cluster.md) | ⭐⭐⭐⭐ | 모든 패턴을 적용한 프로덕션급 클러스터 설계 및 배포 |

## 선수 과목

- 컨테이너 기초 (Docker 이미지, 컨테이너, Dockerfile)
- 리눅스(Linux) 명령줄 활용 능력
- 기본 네트워킹 개념 (TCP/IP, DNS, HTTP)
- 권장: [Docker](../Docker/00_Overview.md), [Linux](../Linux/00_Overview.md), [Networking](../Networking/00_Overview.md)

## 환경 설정

```bash
# kubectl 설치
# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# minikube 설치 (로컬 개발 클러스터)
# macOS
brew install minikube

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Helm 설치
brew install helm  # macOS
# 또는
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 로컬 클러스터 시작
minikube start --cpus=4 --memory=8192

# 확인
kubectl cluster-info
kubectl get nodes
helm version
```

## 추천 자료

- [Kubernetes 공식 문서](https://kubernetes.io/docs/) — 공식 레퍼런스
- [Kubernetes in Action, 2nd Edition](https://www.manning.com/books/kubernetes-in-action-second-edition) — Marko Luksa
- [Programming Kubernetes](https://www.oreilly.com/library/view/programming-kubernetes/9781492047094/) — Michael Hausenblas & Stefan Schimanski
- [Kubernetes Patterns, 2nd Edition](https://www.oreilly.com/library/view/kubernetes-patterns-2nd/9781098131678/) — Ibryam & Huß
- [CNCF Landscape](https://landscape.cncf.io/) — 클라우드 네이티브 에코시스템 개요
