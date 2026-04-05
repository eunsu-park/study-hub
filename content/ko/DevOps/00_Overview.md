# DevOps

## 소개

이 폴더는 DevOps 엔지니어링 학습 자료를 포함합니다. DevOps 문화와 기초 개념부터 CI/CD 파이프라인, Infrastructure as Code, 구성 관리, 컨테이너 오케스트레이션, 서비스 메시 네트워킹까지 단계별로 학습합니다.

**대상 독자**: 소프트웨어 엔지니어, 시스템 관리자, 플랫폼 엔지니어, 프로덕션 시스템을 구축하거나 운영하는 모든 분을 대상으로 합니다.

---

## 학습 로드맵

```
[기초]                    [CI/CD]                  [인프라]
     |                        |                           |
     v                        v                           v
DevOps 기초 ──────────> CI 기초 ──────────────> Infrastructure as Code
     |                        |                           |
     v                        v                           v
버전 관리 워크플로우 ──> GitHub Actions            Terraform 고급
                              |                           |
                              v                           v
                    컨테이너 오케스트레이션        구성 관리
                        운영                              |
                              |                           v
                              +──────> 서비스 메시 & 네트워킹
```

---

## 사전 요구사항

- [Linux](../Linux/00_Overview.md) -- 명령줄 및 시스템 관리 기초
- [Git](../Git/00_Overview.md) -- 버전 관리 기초
- [Docker](../Docker/00_Overview.md) -- 컨테이너화 개념 및 사용법
- [Cloud Computing](../Cloud_Computing/00_Overview.md) -- 클라우드 플랫폼 기본 지식

---

## 파일 목록

| 파일 | 난이도 | 주요 내용 |
|------|--------|-----------|
| [01_DevOps_Fundamentals.md](./01_DevOps_Fundamentals.md) | ⭐ | DevOps 문화, CALMS 프레임워크, DORA 메트릭, 라이프사이클 |
| [02_Version_Control_Workflows.md](./02_Version_Control_Workflows.md) | ⭐⭐ | GitFlow, 트렁크 기반 개발, 브랜칭 전략, 모노레포 vs 폴리레포 |
| [03_CI_Fundamentals.md](./03_CI_Fundamentals.md) | ⭐⭐ | CI 파이프라인 개념, 빌드/테스트/배포 단계, 아티팩트 관리 |
| [04_GitHub_Actions_Deep_Dive.md](./04_GitHub_Actions_Deep_Dive.md) | ⭐⭐⭐ | 워크플로우 문법, 러너, 매트릭스 전략, 재사용 가능한 워크플로우 |
| [05_Infrastructure_as_Code.md](./05_Infrastructure_as_Code.md) | ⭐⭐⭐ | IaC 원칙, Terraform 기초, HCL, 상태 관리 |
| [06_Terraform_Advanced.md](./06_Terraform_Advanced.md) | ⭐⭐⭐⭐ | 모듈, 워크스페이스, 데이터 소스, Terragrunt, 테스팅 |
| [07_Configuration_Management.md](./07_Configuration_Management.md) | ⭐⭐⭐ | Ansible 플레이북, 역할, 인벤토리, Jinja2, Vault |
| [08_Container_Orchestration_Operations.md](./08_Container_Orchestration_Operations.md) | ⭐⭐⭐⭐ | K8s Deployment, Service, HPA, 리소스 관리, 롤링 업데이트 |
| [09_Service_Mesh_and_Networking.md](./09_Service_Mesh_and_Networking.md) | ⭐⭐⭐⭐ | Istio, Envoy 사이드카, 트래픽 관리, mTLS, 관측 가능성 |
| [10_Monitoring_and_Observability.md](./10_Monitoring_and_Observability.md) | ⭐⭐⭐ | Prometheus, Grafana, ELK/EFK 스택, 분산 트레이싱 |
| [11_Log_Management.md](./11_Log_Management.md) | ⭐⭐⭐ | 구조화된 로깅, 로그 집계, Fluentd, Loki |
| [12_GitOps.md](./12_GitOps.md) | ⭐⭐⭐⭐ | ArgoCD, Flux, GitOps 원칙, 선언적 전달 |
| [13_Secrets_Management.md](./13_Secrets_Management.md) | ⭐⭐⭐ | HashiCorp Vault, Sealed Secrets, SOPS, 순환 전략 |
| [14_Cloud_Native_CI_CD.md](./14_Cloud_Native_CI_CD.md) | ⭐⭐⭐⭐ | Tekton, 클라우드 네이티브 파이프라인, Argo Workflows |
| [15_Reliability_Engineering.md](./15_Reliability_Engineering.md) | ⭐⭐⭐⭐ | SRE 원칙, SLO/SLI/SLA, 오류 예산, 인시던트 대응 |
| [16_Chaos_Engineering.md](./16_Chaos_Engineering.md) | ⭐⭐⭐⭐ | Chaos Monkey, Litmus, 결함 주입, 게임 데이 |
| [17_Platform_Engineering.md](./17_Platform_Engineering.md) | ⭐⭐⭐⭐ | 내부 개발자 플랫폼, Backstage, 골든 패스 |
| [18_DevSecOps.md](./18_DevSecOps.md) | ⭐⭐⭐⭐ | Shift-left 보안, SAST/DAST, 공급망 보안, SBOM |
| [19_Observability_Engineering.md](./19_Observability_Engineering.md) | ⭐⭐⭐⭐ | 관측 가능성 vs 모니터링, 텔레메트리 유형, OpenTelemetry, 계측 |
| [20_SLO_Engineering.md](./20_SLO_Engineering.md) | ⭐⭐⭐⭐ | SLI/SLO/SLA, 오류 예산, 번 레이트 알림 |
| [21_Signal_Correlation.md](./21_Signal_Correlation.md) | ⭐⭐⭐⭐ | 메트릭/로그/트레이스 상관 관계, exemplar, 트레이스-로그 연결 |
| [22_Advanced_Metrics_Architecture.md](./22_Advanced_Metrics_Architecture.md) | ⭐⭐⭐⭐⭐ | Prometheus 페더레이션, Thanos/Mimir, 카디널리티 관리 |
| [23_OpenTelemetry_Pipelines.md](./23_OpenTelemetry_Pipelines.md) | ⭐⭐⭐⭐⭐ | OTel Collector 아키텍처, 프로세서, 익스포터, 테일 샘플링 |
| [24_eBPF_Observability.md](./24_eBPF_Observability.md) | ⭐⭐⭐⭐⭐ | eBPF 기초, bpftrace, Cilium Hubble, 커널 수준 관측 가능성 |
| [25_Continuous_Profiling.md](./25_Continuous_Profiling.md) | ⭐⭐⭐⭐ | CPU/메모리 프로파일링, Pyroscope, pprof, 플레임 그래프 |
| [26_Incident_Response.md](./26_Incident_Response.md) | ⭐⭐⭐⭐ | 온콜 실무, 인시던트 관리, 포스트모템, 런북 |
| [27_AIOps_Anomaly_Detection.md](./27_AIOps_Anomaly_Detection.md) | ⭐⭐⭐⭐⭐ | ML 기반 이상 탐지, 지능형 알림, 자동 복구 |
| [28_Capstone_Full_Stack_Observability.md](./28_Capstone_Full_Stack_Observability.md) | ⭐⭐⭐⭐⭐ | 종합 관측 가능성 플랫폼 설계, 도구 선택, 비용 관리 |

---

## 권장 학습 순서

### 1단계: DevOps 기초
1. DevOps 기초 -- 문화, 원칙, 메트릭
2. 버전 관리 워크플로우 -- 팀 협업을 위한 브랜칭 전략

### 2단계: CI/CD 파이프라인
3. CI 기초 -- 파이프라인 개념과 모범 사례
4. GitHub Actions 심화 -- CI/CD 실습 구현

### 3단계: Infrastructure as Code
5. Infrastructure as Code -- Terraform 기초
6. Terraform 고급 -- 프로덕션 수준의 IaC 패턴

### 4단계: 구성 및 오케스트레이션
7. 구성 관리 -- Ansible을 활용한 자동화
8. 컨테이너 오케스트레이션 운영 -- 프로덕션 환경의 Kubernetes

### 5단계: 고급 네트워킹 및 운영
9. 서비스 메시와 네트워킹 -- Istio와 트래픽 관리
10. 모니터링 및 알림 -- Prometheus, Grafana, 알림

### 6단계: 관측 가능성 및 신뢰성
19. 관측 가능성 엔지니어링 -- 관측 가능성 vs 모니터링, OpenTelemetry
20. SLO 엔지니어링 -- SLI/SLO/SLA, 오류 예산, 번 레이트 알림
21. 신호 상관 관계 -- 메트릭, 로그, 트레이스 상관
22. 고급 메트릭 아키텍처 -- Prometheus 페더레이션, Thanos, Mimir
23. OpenTelemetry 파이프라인 -- OTel Collector, 테일 샘플링
24. eBPF 관측 가능성 -- 커널 수준 관측, bpftrace, Hubble
25. 지속적 프로파일링 -- 플레임 그래프, pprof, Pyroscope
26. 인시던트 대응 -- 온콜, 포스트모템, 런북
27. AIOps와 이상 탐지 -- ML 기반 알림, 자동 복구
28. 캡스톤: 풀스택 관측 가능성 -- 종합 플랫폼 설계

---

## 실습 환경

### 필수 도구

```bash
# Install Terraform
brew install terraform    # macOS
# or download from https://www.terraform.io/downloads

# Install Ansible
pip install ansible

# Install kubectl
brew install kubectl      # macOS

# Install GitHub CLI
brew install gh

# Verify installations
terraform --version
ansible --version
kubectl version --client
gh --version
```

### 로컬 Kubernetes

```bash
# minikube for local K8s
brew install minikube
minikube start

# Verify
kubectl cluster-info
```

---

## 관련 자료

- [Docker](../Docker/00_Overview.md) -- 컨테이너화 기초
- [Cloud Computing](../Cloud_Computing/00_Overview.md) -- 클라우드 플랫폼 및 서비스
- [Git](../Git/00_Overview.md) -- 버전 관리 시스템
- [Security](../Security/00_Overview.md) -- 보안 원칙 및 실무
- [Software Engineering](../Software_Engineering/00_Overview.md) -- 소프트웨어 개발 방법론

---

**License**: CC BY-NC 4.0
