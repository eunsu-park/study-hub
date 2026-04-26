# 06. RBAC와 보안(RBAC and Security)

**이전**: [구성 관리와 시크릿](./05_Configuration_and_Secrets.md) | **다음**: [인그레스와 Gateway API](./07_Ingress_and_Gateway_API.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Kubernetes 인증 방법과 API 서버가 신원을 확인하는 방법을 설명할 수 있다
2. Role, ClusterRole, Binding을 사용하여 역할 기반 접근 제어(RBAC, Role-Based Access Control)를 구성할 수 있다
3. 워크로드 격리를 강제하기 위해 파드 보안 표준(Pod Security Standards)과 파드 보안 어드미션(Pod Security Admission)을 적용할 수 있다
4. 사용자 정의 어드미션 제어를 위한 OPA/Gatekeeper 정책을 구현할 수 있다
5. 보안 컨텍스트(security context), seccomp 프로파일, 네트워크 정책으로 파드를 강화할 수 있다

---

Kubernetes 클러스터는 다양한 팀, 서비스, 신뢰 수준의 워크로드를 실행하는 멀티테넌트(multi-tenant) 플랫폼입니다. 적절한 보안 없이는 단일 손상된 파드가 권한을 상승시키고, 시크릿을 유출하고, 전체 클러스터에 걸쳐 횡적 이동(lateral movement)할 수 있습니다. 이 레슨에서는 RBAC로 사용자 인증 및 API 요청 권한 부여부터, 파드 보안 표준으로 워크로드 강화, OPA/Gatekeeper로 조직 정책 적용까지 전체 Kubernetes 보안 스택을 다룹니다.

> **심층 방어(Defense in Depth):** Kubernetes 보안은 단일 기능이 아니라 계층화된 접근 방식입니다. 인증(Authentication)은 신원을 확인하고, 인가(Authorization)는 접근을 제어하며, 어드미션 제어(Admission Control)는 정책을 적용하고, 런타임 보안은 파드가 할 수 있는 일을 제한합니다. 각 계층은 다른 계층의 실패를 보완합니다.

매니페스트에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 모든 API 요청이 통과하는 4단계 게이트(authn → authz → admission → schema validation), RBAC가 순수 가산형(additive)인 이유, Role/ClusterRole + Binding 조합 모델, 그리고 Pod Security Standards가 RBAC가 아닌 admission에 위치하는 이유를 다룹니다.

## 목차

- [이론과 원리](#이론과-원리)
- [1. 인증 방법](#1-인증-방법)
  - [1.1 X.509 클라이언트 인증서](#11-x509-클라이언트-인증서)
  - [1.2 베어러 토큰(Bearer Token)](#12-베어러-토큰bearer-token)
  - [1.3 OpenID Connect (OIDC)](#13-openid-connect-oidc)
  - [1.4 서비스 어카운트 토큰](#14-서비스-어카운트-토큰)
- [2. 인가 모드](#2-인가-모드)
  - [2.1 RBAC (역할 기반 접근 제어)](#21-rbac-역할-기반-접근-제어)
  - [2.2 ABAC (속성 기반 접근 제어)](#22-abac-속성-기반-접근-제어)
  - [2.3 웹훅 인가(Webhook Authorization)](#23-웹훅-인가webhook-authorization)
- [3. RBAC 심층 분석](#3-rbac-심층-분석)
  - [3.1 Role과 ClusterRole](#31-role과-clusterrole)
  - [3.2 RoleBinding과 ClusterRoleBinding](#32-rolebinding과-clusterrolebinding)
  - [3.3 집계된 ClusterRole(Aggregated ClusterRoles)](#33-집계된-clusterroleaggregated-clusterroles)
  - [3.4 일반적인 RBAC 패턴](#34-일반적인-rbac-패턴)
- [4. 서비스 어카운트(Service Accounts)](#4-서비스-어카운트service-accounts)
  - [4.1 서비스 어카운트 기초](#41-서비스-어카운트-기초)
  - [4.2 바인딩된 서비스 어카운트 토큰](#42-바인딩된-서비스-어카운트-토큰)
  - [4.3 자동 마운트 비활성화](#43-자동-마운트-비활성화)
- [5. 파드 보안 표준(Pod Security Standards)](#5-파드-보안-표준pod-security-standards)
  - [5.1 세 가지 프로파일](#51-세-가지-프로파일)
  - [5.2 파드 보안 어드미션(Pod Security Admission)](#52-파드-보안-어드미션pod-security-admission)
  - [5.3 네임스페이스 수준 적용](#53-네임스페이스-수준-적용)
- [6. 보안 컨텍스트(Security Contexts)](#6-보안-컨텍스트security-contexts)
  - [6.1 파드 수준 보안 컨텍스트](#61-파드-수준-보안-컨텍스트)
  - [6.2 컨테이너 수준 보안 컨텍스트](#62-컨테이너-수준-보안-컨텍스트)
- [7. Seccomp과 AppArmor](#7-seccomp과-apparmor)
  - [7.1 Seccomp 프로파일](#71-seccomp-프로파일)
  - [7.2 AppArmor 프로파일](#72-apparmor-프로파일)
- [8. OPA/Gatekeeper 정책](#8-opagatekeeper-정책)
  - [8.1 아키텍처](#81-아키텍처)
  - [8.2 ConstraintTemplate과 Constraint](#82-constrainttemplate과-constraint)
- [9. 보안을 위한 네트워크 정책(Network Policies)](#9-보안을-위한-네트워크-정책network-policies)
  - [9.1 기본 거부(Default Deny)](#91-기본-거부default-deny)
  - [9.2 특정 트래픽 허용](#92-특정-트래픽-허용)
- [연습문제](#연습문제)

---

## 이론과 원리

쿠버네티스 보안은 **독립적 게이트의 파이프라인**으로 이해하는 것이 가장 정확합니다 — 요청은 인증(authn), 인가(authz), 어드미션, 스키마 검증을 차례로 통과해야 비로소 etcd에 저장됩니다. 각 게이트는 다른 일, 다른 실패 모드, 다른 확장 지점을 가집니다. 이를 혼동하는 것("RBAC가 내 파드를 거부했다")은 가장 흔한 디버깅 실수입니다. 이 섹션은 4-게이트 파이프라인, RBAC의 가산형 모델, 그리고 더 깊은 보안 통제(Pod Security Standards, OPA, network policy)가 인가 계층이 아니라 어드미션과 런타임에 위치하는 이유를 설명합니다.

### A. 4단계 요청 파이프라인

모든 API 요청 — `kubectl apply`, 컨트롤러 호출, 사이드카 GET, 대시보드 클릭 — 은 API 서버 내부에서 동일한 체인을 통과합니다:

1. **인증(authn) — "당신은 누구인가?"** 클라이언트 자격 증명(X.509 인증서, bearer 토큰, OIDC ID 토큰, ServiceAccount JWT)을 검증합니다. 출력은 `user.Info` 구조체(username, groups, extra). 어떤 인증자도 자격 증명을 인식하지 못하면 요청은 익명(또는 익명 비활성 시 거부)이 됩니다.
2. **인가(authz) — "허용된 동작인가?"** `(user, verb, resource, namespace, name)`이 주어졌을 때, 활성화된 각 인가자(RBAC, ABAC, Webhook, Node)에 묻습니다 — Allow, Deny, NoOpinion 중 하나를 반환. 첫 명시적 응답이 우선합니다 — **모두 NoOpinion이면 요청은 거부됩니다**(기본 거부).
3. **어드미션 제어 — "이 요청이, 작성된 그대로, 정말 일어나야 하는가?"** Mutating 웹훅은 객체를 재작성할 수 있습니다(사이드카 주입, 기본값 설정). Validating 웹훅은 거부할 수 있습니다(privileged 파드 차단, 레이블 강제). 내장 어드미션 컨트롤러는 quota, 기본값, namespace 존재 검사, Pod Security Standards를 처리합니다.
4. **스키마 검증과 영속화.** 객체에 대한 OpenAPI/CEL 스키마 검사 후 etcd에 쓰기.

네 게이트를 모두 통과해야 객체가 존재합니다. 어느 게이트의 실패든 요청이 영속화되지 않음을 의미합니다. 어느 게이트가 거부 중인지 아는 것("forbidden" → authz, "denied by webhook" → admission, "invalid schema" → 4단계)이 디버깅을 다루기 쉽게 만듭니다.

이 파이프라인은 또한 "사용자에게 모든 권한을 줬는데도 privileged 파드를 만들 수 없다"가 버그가 아닌 이유입니다 — RBAC는 yes라 했지만 Pod Security Admission이 no라 했기 때문입니다.

### B. 순수 허용 목록으로서의 RBAC

RBAC는 범위별로 네 가지 객체 종류를 가집니다:

| Kind | 정의 | 범위 |
|------|------|------|
| `Role` | `(verb, resource)` 권한 집합 | 네임스페이스 |
| `ClusterRole` | 동일하지만 클러스터 전역 | 클러스터 |
| `RoleBinding` | `Role`(또는 `ClusterRole`)을 한 네임스페이스의 주체에 부여 | 네임스페이스 |
| `ClusterRoleBinding` | `ClusterRole`을 클러스터 전역 주체에 부여 | 클러스터 |

주체는 user, group, ServiceAccount입니다.

이 모델에는 세 가지 핵심 속성이 있습니다:

- **가산형(additive only).** RBAC에는 "거부" 규칙이 없습니다. role을 바인딩하여 권한을 누적합니다. 권한을 빼앗으려면 바인딩을 제거합니다 — 사용자 X에게 시크릿 읽기를 부여하는 role이 있는 상태에서 "사용자 X는 시크릿을 읽을 수 없다"라고 쓸 방법은 없습니다. 이는 시스템을 추론하기 쉽게 만들지만(규칙 우선순위 퍼즐 없음) 최소 권한은 패치가 아니라 신중한 role 구성을 필요로 합니다.
- **verb-resource 단위.** 권한은 `(verb, resource)` 수준입니다 — `get pods`, `list deployments`, `create configmaps/myconfig`(리소스 이름 선택). 하위 리소스는 별개(`pods/exec`, `pods/log`). 와일드카드 존재(`verbs: ["*"]`, `resources: ["*"]`)하지만 프로덕션 role에서는 피해야 합니다.
- **데이터 필터링 없음.** RBAC는 네임스페이스에서 *시크릿을 list할 수 있는지*를 통제하지, *어느 시크릿을 보는지*를 통제하지 않습니다. list할 수 있다면 모두 list할 수 있습니다. 행 단위 필터링은 어드미션 웹훅이나 외부 정책 엔진을 필요로 합니다.

Role/ClusterRole + Binding 분리는 동일한 `ClusterRole`("view")을 재사용할 수 있게 합니다 — 사용자 A에게 `dev` 네임스페이스에서, 사용자 B에게 `prod` 네임스페이스에서, 그룹 `oncall`에게 클러스터 전역으로 바인딩 — 세 바인딩, 하나의 role 정의.

### C. 워크로드를 위한 식별 — ServiceAccount와 토큰 투영

인간 사용자는 인증서나 OIDC로 인증합니다. 파드는 **ServiceAccount 토큰**으로 인증합니다 — API 서버가 서명한 JWT가 자동으로 `/var/run/secrets/kubernetes.io/serviceaccount/token`에 마운트됩니다. 모든 네임스페이스의 `default` ServiceAccount가 구성되지 않은 파드가 받는 것입니다.

현대 클러스터는 **bound ServiceAccount 토큰**(BoundServiceAccountTokenVolume, 1.21부터 기본 활성)을 사용합니다:

- 토큰은 파드별로 생성됩니다(Secret 객체에 저장되지 않음).
- 특정 파드의 UID와 audience에 바인딩됩니다 — 파드가 삭제되면 토큰은 무효화됩니다.
- kubelet이 만료 전에 토큰을 회전합니다(기본 1시간).

이는 "etcd에 영구 ServiceAccount 토큰이 앉아 있어 Secret read 권한이 있는 누구나 가져갈 수 있다"는 옛 실패 모드를 제거합니다. 외부 시스템에 대해서는 TokenRequest API로 자체 토큰을 발급하고 특정 audience(예: 내부 Vault에 대해서만 유효한 토큰)로 범위를 지정할 수 있습니다.

**자동 마운트(auto-mounting)**는 기본 활성이지만, API가 필요 없는 파드는 끄는 것이 좋습니다(`automountServiceAccountToken: false`를 Pod 또는 SA 수준에서). API 서버와 통신할 필요 없는 파드가 API 자격 증명을 가지고 다닐 이유가 없습니다.

### D. 심층 방어 — Pod Security Standards, NetworkPolicy, OPA

RBAC는 API 접근을 통제합니다 — 실행 중인 컨테이너가 무엇을 할 수 있는지는 *통제하지 않습니다*. `privileged: true`와 `hostNetwork: true`인 컨테이너는 노드의 모든 시크릿을 읽고 클러스터를 피벗할 수 있습니다 — ServiceAccount의 RBAC 권한이 0이라도. 그래서 쿠버네티스는 추가 통제를 계층화합니다:

**Pod Security Standards (PSS)**는 **Pod Security Admission** 컨트롤러가 강제하는 세 프로파일을 정의합니다:

| 프로파일 | 허용 사항 | 사용 시기 |
|---------|---------|---------|
| `privileged` | 모든 것 | 신뢰 시스템 워크로드(CNI, 스토리지 드라이버) |
| `baseline` | 최소 컨테이너 위생; `privileged`, hostNetwork, hostPath 차단 | 대부분의 사용자 워크로드 |
| `restricted` | 엄격한 강화; non-root, 모든 capability drop, seccomp RuntimeDefault | 특별 요구 없는 앱 |

강제는 레이블을 통한 네임스페이스 단위:

```yaml
metadata:
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.29
```

이는 **어드미션 시점** 강제입니다 — 위반하는 파드는 §A의 게이트 3에서 거부됩니다.

**NetworkPolicy**는 L3/L4 수준에서 파드-파드 트래픽을 제한합니다. 기본 쿠버네티스 네트워킹은 "어떤 파드든 어떤 파드와도 통신 가능"입니다(3강 §A). NetworkPolicy로 "`tier=backend` 레이블의 파드는 `tier=frontend` 레이블의 파드로부터 8080 포트 트래픽만 허용"이라 말할 수 있습니다. 구현은 CNI 플러그인에 위임됩니다(8강).

**OPA Gatekeeper / Kyverno**는 웹훅을 통해 어드미션(게이트 3)에 플러그인하는 정책 엔진입니다. "모든 네임스페이스는 cost-center 레이블을 가져야 한다"나 "컨테이너 이미지는 내부 레지스트리에서 와야 한다" 같은 규칙을 도메인 특화 언어(OPA는 Rego, Kyverno는 YAML)로 작성하게 해줍니다. Pod Security Standards로 부족할 때의 다음 단계입니다.

멘탈 모델: **RBAC는 API 접근을 통제하고, PSS + NetworkPolicy + OPA는 워크로드 동작을 통제합니다.** 둘 다 필요합니다. 엄격한 RBAC를 가졌지만 `privileged` 어드미션이 허용된 잘못 구성된 클러스터는 컨테이너 하나만에 전체 침해됩니다.

### 이론에서 아래의 YAML으로

이제 레슨은 이 추상을 구체화합니다:

- **섹션 1 (인증)**은 §A의 게이트 1입니다 — 네 가지 자격 증명 유형과 API 서버가 각각을 검증하는 방법.
- **섹션 2 (인가 모드)**는 게이트 2의 플러그인을 소개합니다 — 보통 RBAC(§B)와 고급 케이스에는 Webhook을 사용합니다.
- **섹션 3 (RBAC 심층 분석)**은 §B의 4객체 모델을 구체적 YAML과 role 합성을 가능하게 하는 aggregation 패턴으로 다룹니다.
- **섹션 4 (서비스 어카운트)**는 §C입니다 — 파드가 식별을 얻는 방법과 자동 마운트를 잠그는 방법.
- **섹션 5 (Pod Security Standards)**는 어드미션 레이블로 강제되는 §D의 PSS 계층입니다.
- **섹션 6–7 (Security Contexts, Seccomp/AppArmor)**는 PSS가 강제하는 워크로드 측 강화입니다.
- **섹션 8 (OPA Gatekeeper)**는 §D의 어드미션 제어 정책 엔진 확장입니다.
- **섹션 9 (Network Policies)**는 §D의 L3/L4 계층입니다.

§A의 4-게이트 파이프라인을 보고 나면, "왜 거부됐나?" 질문은 특정 게이트와 특정 확장 지점으로 매핑됩니다.

---

## 1. 인증 방법

Kubernetes API 서버에 대한 모든 요청은 인증되어야 합니다. Kubernetes에는 내장 사용자 데이터베이스가 없으며, 대신 플러그인 아키텍처를 통해 외부 시스템에 인증을 위임합니다.

```
┌──────────┐     ┌─────────────────────────┐     ┌───────────────┐
│  kubectl  │────▶│    API Server            │────▶│ Authorization │
│  or SDK   │     │  Authentication Plugins: │     │   (RBAC)      │
└──────────┘     │  - X.509 Certs          │     └───────────────┘
                  │  - Bearer Tokens         │
                  │  - OIDC                  │
                  │  - Service Accounts      │
                  └─────────────────────────┘
```

### 1.1 X.509 클라이언트 인증서

클러스터 관리자에게 가장 일반적인 방법입니다. API 서버는 구성된 인증 기관(CA, Certificate Authority)에 대해 클라이언트 인증서를 검증합니다.

```bash
# 새 사용자를 위한 개인 키 생성
openssl genrsa -out developer.key 2048

# 인증서 서명 요청(CSR, Certificate Signing Request) 생성
# CN(Common Name)은 사용자명이 됨
# O(Organization)는 그룹이 됨
openssl req -new -key developer.key \
  -out developer.csr \
  -subj "/CN=jane/O=dev-team"

# Kubernetes CertificateSigningRequest 생성
cat <<EOF | kubectl apply -f -
apiVersion: certificates.k8s.io/v1
kind: CertificateSigningRequest
metadata:
  name: jane-csr
spec:
  request: $(cat developer.csr | base64 | tr -d '\n')
  signerName: kubernetes.io/kube-apiserver-client
  usages:
    - client auth
EOF

# CSR 승인
kubectl certificate approve jane-csr

# 서명된 인증서 추출
kubectl get csr jane-csr -o jsonpath='{.status.certificate}' | base64 -d > developer.crt

# 새 사용자를 위한 kubectl 컨텍스트 구성
kubectl config set-credentials jane \
  --client-certificate=developer.crt \
  --client-key=developer.key

kubectl config set-context jane-context \
  --cluster=minikube \
  --user=jane \
  --namespace=dev

kubectl config use-context jane-context
```

### 1.2 베어러 토큰(Bearer Token)

베어러 토큰은 `Authorization` 헤더로 전송됩니다. Kubernetes는 정적 토큰 파일과 부트스트랩 토큰을 지원하지만, 이는 주로 자동화된 설정에서 사용됩니다.

```bash
# 정적 토큰 파일 형식 (줄당 하나의 토큰)
# token,user,uid,"group1,group2"
# 프로덕션에서는 권장하지 않음 -- 대신 OIDC 사용

# kubectl로 베어러 토큰 사용
kubectl --token="eyJhbGciOiJSUzI1NiIs..." get pods

# curl로 베어러 토큰 사용
curl -k https://API_SERVER:6443/api/v1/pods \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIs..."
```

### 1.3 OpenID Connect (OIDC)

OIDC는 프로덕션 클러스터에서 권장되는 인증 방법입니다. 외부 ID 제공자(Dex, Keycloak, Google, Azure AD)에 인증을 위임합니다.

```
┌───────┐     ┌──────────────┐     ┌───────────────┐
│ User  │────▶│ OIDC Provider│────▶│   API Server  │
│       │◀────│ (Keycloak)   │     │ validates JWT │
│       │     │              │     │ id_token      │
│ gets  │     └──────────────┘     └───────────────┘
│ token │
└───────┘
```

OIDC를 위한 API 서버 구성:

```yaml
# kube-apiserver 플래그 (정적 파드 매니페스트 또는 kubeadm 구성)
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
apiServer:
  extraArgs:
    oidc-issuer-url: "https://keycloak.example.com/realms/k8s"
    oidc-client-id: "kubernetes"
    oidc-username-claim: "email"
    oidc-groups-claim: "groups"
    oidc-username-prefix: "oidc:"
    oidc-groups-prefix: "oidc:"
```

### 1.4 서비스 어카운트 토큰

서비스 어카운트(Service Account)는 클러스터 내에서 실행되는 워크로드를 인증합니다. Kubernetes 1.24부터 바인딩된 서비스 어카운트 토큰(프로젝티드 볼륨)이 기본값입니다 -- 시간 제한이 있고, 대상이 지정되며, 자동으로 교체됩니다.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-app
  namespace: production
automountServiceAccountToken: false  # 필요하지 않으면 명시적으로 비활성화
```

```bash
# 서비스 어카운트에 대한 단기 토큰 생성
kubectl create token my-app --duration=1h --namespace=production

# 토큰 검사 (JWT임)
kubectl create token my-app | jwt decode -
```

---

## 2. 인가 모드

인증 후, API 서버는 인증된 신원이 요청한 작업을 수행할 수 있는지 확인합니다. Kubernetes는 순서대로 평가되는 여러 인가 모드를 지원합니다.

### 2.1 RBAC (역할 기반 접근 제어)

RBAC는 모든 프로덕션 클러스터에서 표준 인가 모드입니다. 사용자, 그룹 또는 서비스 어카운트에 할당된 역할을 기반으로 권한을 부여합니다.

```bash
# 활성화된 인가 모드 확인
kubectl api-versions | grep rbac
# rbac.authorization.k8s.io/v1

# 사용자가 작업을 수행할 수 있는지 테스트
kubectl auth can-i create deployments --namespace=dev --as=jane
# yes

kubectl auth can-i delete nodes --as=jane
# no
```

### 2.2 ABAC (속성 기반 접근 제어)

ABAC는 정적 정책 파일을 사용하며 업데이트하려면 API 서버를 재시작해야 합니다. 현대 클러스터에서는 거의 사용되지 않습니다.

```json
{
  "apiVersion": "abac.authorization.kubernetes.io/v1beta1",
  "kind": "Policy",
  "spec": {
    "user": "jane",
    "namespace": "dev",
    "resource": "pods",
    "readonly": true
  }
}
```

### 2.3 웹훅 인가(Webhook Authorization)

웹훅 인가는 인가 결정을 외부 HTTP 서비스에 위임합니다. 기존 엔터프라이즈 인가 시스템과 통합할 때 유용합니다.

```yaml
# 웹훅 인가 구성
apiVersion: v1
kind: Config
clusters:
  - name: authz-webhook
    cluster:
      server: https://authz.example.com/authorize
      certificate-authority: /etc/kubernetes/pki/authz-ca.pem
users:
  - name: kube-apiserver
    user:
      client-certificate: /etc/kubernetes/pki/authz-client.pem
      client-key: /etc/kubernetes/pki/authz-client-key.pem
current-context: webhook
contexts:
  - context:
      cluster: authz-webhook
      user: kube-apiserver
    name: webhook
```

---

## 3. RBAC 심층 분석

RBAC에는 함께 작동하는 네 가지 리소스 유형이 있습니다:

```
                 네임스페이스 범위              클러스터 범위
               ┌──────────┐                 ┌──────────────┐
  권한         │   Role   │                 │ ClusterRole  │
               └────┬─────┘                 └──────┬───────┘
                    │ 바인딩                        │ 바인딩
               ┌────▼──────────┐             ┌─────▼────────────┐
  바인딩       │ RoleBinding   │             │ClusterRoleBinding│
               └───────────────┘             └──────────────────┘
```

### 3.1 Role과 ClusterRole

**Role**은 특정 네임스페이스 내에서 권한을 정의합니다. **ClusterRole**은 클러스터 전체 또는 모든 네임스페이스에 걸쳐 권한을 정의합니다.

```yaml
# Role: "dev" 네임스페이스에서 파드 읽기 허용
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: dev
rules:
  - apiGroups: [""]           # core API 그룹
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods/log"]   # 하위 리소스
    verbs: ["get"]
```

```yaml
# ClusterRole: 모든 네임스페이스에서 디플로이먼트 관리 허용
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: deployment-manager
rules:
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["apps"]
    resources: ["deployments/scale"]
    verbs: ["update", "patch"]
  - apiGroups: ["apps"]
    resources: ["deployments/status"]
    verbs: ["get"]
```

```yaml
# 네임스페이스가 없는 리소스(노드)를 위한 ClusterRole
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: node-viewer
rules:
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["metrics.k8s.io"]
    resources: ["nodes"]
    verbs: ["get", "list"]
```

### 3.2 RoleBinding과 ClusterRoleBinding

**RoleBinding**은 네임스페이스 내에서 Role(또는 ClusterRole)을 주체(subject)에 부여합니다. **ClusterRoleBinding**은 전체 클러스터에 걸쳐 ClusterRole을 부여합니다.

```yaml
# RoleBinding: "dev" 네임스페이스에서 사용자 "jane"에게 pod-reader 역할 부여
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods-dev
  namespace: dev
subjects:
  - kind: User
    name: jane
    apiGroup: rbac.authorization.k8s.io
  - kind: Group
    name: dev-team
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

```yaml
# ClusterRoleBinding: 그룹에 클러스터 전체 관리자 권한 부여
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: platform-admins
subjects:
  - kind: Group
    name: oidc:platform-admins    # 접두사가 붙은 OIDC 그룹
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: cluster-admin
  apiGroup: rbac.authorization.k8s.io
```

```yaml
# ClusterRole을 참조하는 RoleBinding (네임스페이스 범위로 제한)
# 강력한 패턴: 권한을 한 번 정의하고 네임스페이스별로 바인딩
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: deploy-manager-dev
  namespace: dev
subjects:
  - kind: ServiceAccount
    name: ci-pipeline
    namespace: ci
roleRef:
  kind: ClusterRole          # Role이 아닌 ClusterRole
  name: deployment-manager
  apiGroup: rbac.authorization.k8s.io
```

### 3.3 집계된 ClusterRole(Aggregated ClusterRoles)

집계된 ClusterRole은 레이블 셀렉터(label selector)를 사용하여 여러 ClusterRole을 결합합니다. 내장된 `admin`, `edit`, `view` 역할은 집계됩니다.

```yaml
# "admin" 역할에 집계되는 사용자 정의 ClusterRole
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: custom-resource-admin
  labels:
    rbac.authorization.k8s.io/aggregate-to-admin: "true"
rules:
  - apiGroups: ["example.com"]
    resources: ["widgets"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
```

```bash
# 집계 확인 -- admin 역할에 widget 권한이 포함되어야 함
kubectl describe clusterrole admin | grep widgets
```

### 3.4 일반적인 RBAC 패턴

```bash
# RBAC 디버깅 -- 사용자/서비스 어카운트가 할 수 있는 작업 확인
kubectl auth can-i --list --as=system:serviceaccount:dev:my-app

# 네임스페이스에 바인딩된 역할 확인
kubectl get rolebindings -n dev -o wide

# 클러스터 수준 바인딩 확인
kubectl get clusterrolebindings -o wide | grep dev-team

# 사용자를 가장하여 권한 테스트
kubectl get pods -n dev --as=jane --as-group=dev-team
```

**최소 권한 원칙(Principle of Least Privilege) 체크리스트:**

| 가이드라인 | 예시 |
|-----------|------|
| ClusterRole보다 네임스페이스 범위 Role 사용 | 모든 곳의 `ClusterRole` 대신 `dev`의 `Role` |
| 와일드카드 동사(`*`) 피하기 | 정확한 동사 지정: `get`, `list`, `watch` |
| 와일드카드 리소스(`*`) 피하기 | 특정 리소스 나열: `pods`, `services` |
| User 바인딩보다 Group 바인딩 선호 | 개별 사용자 대신 `dev-team` 그룹에 바인딩 |
| 정기적으로 감사 | `kubectl auth can-i --list --as=...` |

---

## 4. 서비스 어카운트(Service Accounts)

서비스 어카운트는 파드의 ID 메커니즘입니다. 모든 네임스페이스에는 `default` 서비스 어카운트가 있으며, 별도로 지정하지 않으면 파드가 이를 사용합니다.

### 4.1 서비스 어카운트 기초

```yaml
# 전용 서비스 어카운트 생성
apiVersion: v1
kind: ServiceAccount
metadata:
  name: log-collector
  namespace: monitoring
```

```yaml
# 파드에 서비스 어카운트 할당
apiVersion: v1
kind: Pod
metadata:
  name: log-agent
  namespace: monitoring
spec:
  serviceAccountName: log-collector
  containers:
    - name: agent
      image: fluent/fluent-bit:latest
```

### 4.2 바인딩된 서비스 어카운트 토큰

Kubernetes 1.24부터 서비스 어카운트 토큰은 만료가 있는 프로젝티드 볼륨(projected volume)입니다.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: api-client
spec:
  serviceAccountName: my-app
  containers:
    - name: client
      image: my-api-client:v1
      volumeMounts:
        - name: token
          mountPath: /var/run/secrets/tokens
          readOnly: true
  volumes:
    - name: token
      projected:
        sources:
          - serviceAccountToken:
              path: api-token
              expirationSeconds: 3600      # 1시간
              audience: "https://my-api.example.com"
```

### 4.3 자동 마운트 비활성화

Kubernetes API를 호출할 필요가 없는 파드의 경우, 자동 토큰 마운트를 비활성화하여 공격 표면을 줄입니다.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: static-web
  namespace: frontend
automountServiceAccountToken: false
---
apiVersion: v1
kind: Pod
metadata:
  name: web-server
  namespace: frontend
spec:
  serviceAccountName: static-web
  automountServiceAccountToken: false   # 명확성을 위해 파드 수준에서도 설정
  containers:
    - name: nginx
      image: nginx:1.27
```

---

## 5. 파드 보안 표준(Pod Security Standards)

파드 보안 표준(PSS, Pod Security Standards)은 파드가 할 수 있는 일을 제어하는 세 가지 점진적 프로파일을 정의합니다. 더 이상 사용되지 않는 PodSecurityPolicy를 대체합니다.

### 5.1 세 가지 프로파일

```
Privileged          Baseline              Restricted
─────────           ────────              ──────────
제한 없음           알려진 권한            강력하게 제한
                    상승 방지             모범 사례 적용

예시:               차단 대상:             요구사항:
- 시스템 데몬       - hostNetwork         - 비루트 사용자
- 노드 에이전트     - hostPID             - 읽기 전용 루트 FS
- CNI 플러그인      - privileged          - 모든 기능 삭제
                    - hostPath            - Seccomp 프로파일
                                          - 권한 상승 불가
```

### 5.2 파드 보안 어드미션(Pod Security Admission)

파드 보안 어드미션(PSA, Pod Security Admission)은 네임스페이스 수준에서 파드 보안 표준을 적용하는 내장 어드미션 컨트롤러(admission controller)입니다.

세 가지 적용 모드:

| 모드 | 동작 |
|------|------|
| `enforce` | 정책을 위반하는 파드 거부 |
| `audit` | 허용하되 감사 로그에 위반 사항 기록 |
| `warn` | 허용하되 사용자에게 경고 반환 |

### 5.3 네임스페이스 수준 적용

```yaml
# 레이블을 통해 네임스페이스에 파드 보안 표준 적용
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    # restricted 프로파일 적용
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.30
    # 동일한 프로파일로 감사 및 경고
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: v1.30
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: v1.30
```

```yaml
# restricted 프로파일을 준수하는 파드
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
  namespace: production
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: my-app:v1
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
      resources:
        limits:
          memory: "256Mi"
          cpu: "500m"
        requests:
          memory: "128Mi"
          cpu: "250m"
```

```bash
# 정책을 위반할 때 어떻게 되는지 테스트
kubectl run test-privileged \
  --image=nginx \
  --namespace=production \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "nginx",
        "image": "nginx",
        "securityContext": {"privileged": true}
      }]
    }
  }'
# Error: pods "test-privileged" is forbidden: violates PodSecurity "restricted:v1.30"

# 파드를 생성하지 않고 준수 여부 확인을 위한 dry-run
kubectl label namespace staging \
  pod-security.kubernetes.io/enforce=restricted \
  --dry-run=server --overwrite
```

---

## 6. 보안 컨텍스트(Security Contexts)

보안 컨텍스트는 파드와 컨테이너에 대한 권한 및 접근 제어 설정을 구성합니다.

### 6.1 파드 수준 보안 컨텍스트

파드 수준 설정은 init 컨테이너를 포함하여 파드의 모든 컨테이너에 적용됩니다.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 10000
    runAsGroup: 10000
    fsGroup: 20000           # 볼륨 마운트용 GID
    fsGroupChangePolicy: "OnRootMismatch"  # 더 빠른 볼륨 마운트
    supplementalGroups: [30000]
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: my-app:v2
      # 컨테이너 수준 설정이 파드 수준을 오버라이드
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
          add: ["NET_BIND_SERVICE"]  # 1024 미만 포트에 바인딩할 때만 필요
```

### 6.2 컨테이너 수준 보안 컨텍스트

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hardened-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hardened
  template:
    metadata:
      labels:
        app: hardened
    spec:
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: app
          image: my-app:v2
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          # 쓰기 가능한 디렉토리는 tmpfs여야 함
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: cache
              mountPath: /app/cache
      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 100Mi
        - name: cache
          emptyDir:
            sizeLimit: 500Mi
```

**Linux 기능(capability) 참조:**

| 기능 | 용도 | 필요한 경우 |
|------|------|------------|
| `NET_BIND_SERVICE` | 1024 미만 포트에 바인딩 | 포트 80/443의 웹 서버 |
| `NET_RAW` | 원시 소켓(raw socket) | 네트워크 진단 |
| `SYS_PTRACE` | 프로세스 추적 | 디버거, 프로파일러 |
| `DAC_OVERRIDE` | 파일 권한 검사 우회 | 레거시 앱 |
| `SETUID` / `SETGID` | UID/GID 변경 | su, sudo |

---

## 7. Seccomp과 AppArmor

### 7.1 Seccomp 프로파일

Seccomp(secure computing)은 컨테이너가 수행할 수 있는 시스템 콜을 제한합니다. `RuntimeDefault` 프로파일은 `reboot`, `mount`, `ptrace`와 같은 위험한 시스콜을 차단합니다.

```yaml
# RuntimeDefault seccomp 프로파일 사용 (권장)
apiVersion: v1
kind: Pod
metadata:
  name: seccomp-default
spec:
  securityContext:
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: my-app:v1
```

```yaml
# 사용자 정의 seccomp 프로파일 (노드 파일시스템에서 로드)
apiVersion: v1
kind: Pod
metadata:
  name: seccomp-custom
spec:
  securityContext:
    seccompProfile:
      type: Localhost
      localhostProfile: profiles/custom-audit.json
  containers:
    - name: app
      image: my-app:v1
```

사용자 정의 seccomp 프로파일 예시 (`/var/lib/kubelet/seccomp/profiles/`에 배치):

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": [
        "read", "write", "open", "close", "stat", "fstat",
        "mmap", "mprotect", "munmap", "brk",
        "rt_sigaction", "rt_sigprocmask", "ioctl",
        "access", "pipe", "select", "sched_yield",
        "clone", "execve", "exit_group", "futex",
        "epoll_ctl", "epoll_wait", "accept4",
        "socket", "connect", "bind", "listen",
        "getpid", "gettid", "nanosleep"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

### 7.2 AppArmor 프로파일

AppArmor는 프로그램별 보안 프로파일로 프로그램을 제한합니다. 파드가 참조하기 전에 프로파일이 노드에 로드되어 있어야 합니다.

```yaml
# AppArmor 프로파일이 있는 파드 (Kubernetes 1.30+ 어노테이션 불필요)
apiVersion: v1
kind: Pod
metadata:
  name: apparmor-pod
spec:
  securityContext:
    appArmorProfile:
      type: Localhost
      localhostProfile: k8s-custom-deny-write
  containers:
    - name: app
      image: my-app:v1
```

```bash
# 노드에서 로드된 AppArmor 프로파일 확인
ssh node01 "sudo aa-status"

# 사용자 정의 프로파일 로드
ssh node01 "sudo apparmor_parser -r /etc/apparmor.d/k8s-custom-deny-write"
```

---

## 8. OPA/Gatekeeper 정책

Open Policy Agent(OPA) Gatekeeper는 검증 어드미션 웹훅(validating admission webhook)으로 작동하는 정책 엔진입니다. 파드 보안 표준이 제공하는 것 이상의 사용자 정의 정책 적용을 가능하게 합니다.

### 8.1 아키텍처

```
                    ┌────────────────┐
  kubectl apply ──▶ │   API Server   │
                    │                │
                    │  Admission     │
                    │  Webhooks:     │
                    │  ┌───────────┐ │
                    │  │Gatekeeper │ │──▶ Rego 정책을
                    │  │ Webhook   │ │    제약조건에 대해 평가
                    │  └───────────┘ │
                    └────────────────┘
```

```bash
# minikube에 Gatekeeper 설치
helm repo add gatekeeper https://open-policy-agent.github.io/gatekeeper/charts
helm install gatekeeper gatekeeper/gatekeeper \
  --namespace gatekeeper-system \
  --create-namespace
```

### 8.2 ConstraintTemplate과 Constraint

**ConstraintTemplate**은 Rego로 재사용 가능한 정책 로직을 정의합니다. **Constraint**는 특정 매개변수로 해당 템플릿을 적용합니다.

```yaml
# ConstraintTemplate: 모든 리소스에 특정 레이블 요구
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8srequiredlabels
spec:
  crd:
    spec:
      names:
        kind: K8sRequiredLabels
      validation:
        openAPIV3Schema:
          type: object
          properties:
            labels:
              type: array
              items:
                type: string
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8srequiredlabels

        violation[{"msg": msg}] {
          provided := {label | input.review.object.metadata.labels[label]}
          required := {label | label := input.parameters.labels[_]}
          missing := required - provided
          count(missing) > 0
          msg := sprintf("Missing required labels: %v", [missing])
        }
```

```yaml
# Constraint: 모든 Deployment에 "team"과 "env" 레이블 요구
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: deployment-must-have-labels
spec:
  match:
    kinds:
      - apiGroups: ["apps"]
        kinds: ["Deployment"]
  parameters:
    labels:
      - "team"
      - "env"
```

```yaml
# ConstraintTemplate: latest 태그 사용 차단
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8sblocklatesttag
spec:
  crd:
    spec:
      names:
        kind: K8sBlockLatestTag
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8sblocklatesttag

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          endswith(container.image, ":latest")
          msg := sprintf("Container '%v' uses ':latest' tag. Use a specific version.", [container.name])
        }

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not contains(container.image, ":")
          msg := sprintf("Container '%v' has no tag. Use a specific version.", [container.name])
        }
```

```bash
# Gatekeeper 감사 결과 확인
kubectl get k8srequiredlabels deployment-must-have-labels -o yaml

# 모든 위반 사항 나열
kubectl get constraints -o wide
```

---

## 9. 보안을 위한 네트워크 정책(Network Policies)

네트워크 정책은 파드 간 통신을 제어하는 핵심 보안 계층입니다. CNI 플러그인(Calico, Cilium 등)에 의해 적용됩니다.

### 9.1 기본 거부(Default Deny)

기본 거부 정책으로 시작하고 필요한 트래픽을 명시적으로 허용합니다.

```yaml
# 네임스페이스의 모든 인그레스 트래픽 기본 거부
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: production
spec:
  podSelector: {}       # 네임스페이스의 모든 파드 선택
  policyTypes:
    - Ingress
---
# 네임스페이스의 모든 이그레스 트래픽 기본 거부
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-egress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Egress
```

### 9.2 특정 트래픽 허용

```yaml
# 프론트엔드 파드가 포트 8080에서 백엔드 파드에 도달하도록 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      role: backend
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              role: frontend
      ports:
        - protocol: TCP
          port: 8080
---
# 모든 파드가 DNS(kube-dns)에 도달하도록 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
```

```bash
# 네트워크 정책 적용 테스트
# 테스트 파드 배포
kubectl run test-client --image=busybox --rm -it --restart=Never \
  -n production -- wget -qO- --timeout=2 http://backend:8080/health

# 테스트 파드에 프론트엔드 레이블을 추가하여 접근 허용
kubectl run test-client --image=busybox --rm -it --restart=Never \
  -n production --labels="role=frontend" \
  -- wget -qO- --timeout=2 http://backend:8080/health
```

---

## 연습문제

### 연습문제 1: 읽기 전용 RBAC 정책 생성

`staging` 네임스페이스에서 사용자 `alice`에게 파드, 서비스, 디플로이먼트에 대한 읽기 전용 접근 권한을 부여하는 Role과 RoleBinding을 생성하세요. 그런 다음 `kubectl auth can-i`를 사용하여 권한을 확인하세요.

<details><summary>정답 보기</summary>

```yaml
# role-readonly.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: staging
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: readonly
  namespace: staging
rules:
  - apiGroups: [""]
    resources: ["pods", "services"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: alice-readonly
  namespace: staging
subjects:
  - kind: User
    name: alice
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: readonly
  apiGroup: rbac.authorization.k8s.io
```

```bash
kubectl apply -f role-readonly.yaml

# 권한 확인
kubectl auth can-i get pods -n staging --as=alice
# yes

kubectl auth can-i create pods -n staging --as=alice
# no

kubectl auth can-i delete deployments -n staging --as=alice
# no

kubectl auth can-i list services -n staging --as=alice
# yes
```

</details>

### 연습문제 2: 파드 보안 표준 적용

`restricted` 파드 보안 표준이 적용된 `secure-ns`라는 네임스페이스를 생성하세요. 그런 다음 특권(privileged) 파드 배포를 시도하여 거부되는지 확인하세요. 마지막으로 준수하는 파드를 배포하세요.

<details><summary>정답 보기</summary>

```yaml
# secure-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: secure-ns
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.30
    pod-security.kubernetes.io/warn: restricted
```

```bash
kubectl apply -f secure-namespace.yaml

# 이것은 거부됨
kubectl run bad-pod --image=nginx --namespace=secure-ns \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "nginx",
        "image": "nginx",
        "securityContext": {"privileged": true}
      }]
    }
  }'
# Error: violates PodSecurity "restricted:v1.30"
```

```yaml
# compliant-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: good-pod
  namespace: secure-ns
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: busybox:1.36
      command: ["sleep", "3600"]
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
```

```bash
kubectl apply -f compliant-pod.yaml
kubectl get pod good-pod -n secure-ns
# good-pod   1/1     Running   0          5s
```

</details>

### 연습문제 3: Gatekeeper 정책 생성

모든 파드에 리소스 제한(memory와 cpu)이 정의되어 있어야 하는 Gatekeeper ConstraintTemplate과 Constraint를 작성하세요. 리소스 제한이 없는 파드로 테스트하세요.

<details><summary>정답 보기</summary>

```yaml
# constraint-template.yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8srequireresourcelimits
spec:
  crd:
    spec:
      names:
        kind: K8sRequireResourceLimits
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8srequireresourcelimits

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not container.resources.limits.memory
          msg := sprintf("Container '%v' must have memory limits", [container.name])
        }

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not container.resources.limits.cpu
          msg := sprintf("Container '%v' must have CPU limits", [container.name])
        }
---
# constraint.yaml
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequireResourceLimits
metadata:
  name: must-have-limits
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    excludedNamespaces:
      - kube-system
      - gatekeeper-system
```

```bash
kubectl apply -f constraint-template.yaml
# 템플릿이 준비될 때까지 대기
kubectl apply -f constraint.yaml

# 이것은 거부되어야 함
kubectl run no-limits --image=nginx
# Error: Container 'no-limits' must have memory limits

# 이것은 성공해야 함
kubectl run with-limits --image=nginx \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "with-limits",
        "image": "nginx",
        "resources": {
          "limits": {"memory": "128Mi", "cpu": "500m"},
          "requests": {"memory": "64Mi", "cpu": "250m"}
        }
      }]
    }
  }'
```

</details>

### 연습문제 4: 네트워크 정책 마이크로세그멘테이션(Microsegmentation)

`microservices` 네임스페이스에서 다음 규칙을 구현하는 네트워크 정책을 생성하세요:
- `frontend` 파드는 포트 8080에서 `api` 파드하고만 통신 가능
- `api` 파드는 포트 5432에서 `database` 파드하고만 통신 가능
- `database` 파드는 `api` 파드로부터의 트래픽만 수락
- 모든 파드가 DNS에 도달 가능

<details><summary>정답 보기</summary>

```yaml
# network-policies.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: microservices
---
# 모든 트래픽 기본 거부
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: microservices
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
---
# 모든 파드에 DNS 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: microservices
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
---
# 프론트엔드에서 API로의 이그레스 (포트 8080)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: frontend-to-api
  namespace: microservices
spec:
  podSelector:
    matchLabels:
      tier: frontend
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              tier: api
      ports:
        - protocol: TCP
          port: 8080
---
# 프론트엔드로부터 API의 인그레스 (포트 8080)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-ingress-from-frontend
  namespace: microservices
spec:
  podSelector:
    matchLabels:
      tier: api
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              tier: frontend
      ports:
        - protocol: TCP
          port: 8080
---
# API에서 데이터베이스로의 이그레스 (포트 5432)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-to-database
  namespace: microservices
spec:
  podSelector:
    matchLabels:
      tier: api
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              tier: database
      ports:
        - protocol: TCP
          port: 5432
---
# API로부터 데이터베이스의 인그레스 (포트 5432)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-ingress-from-api
  namespace: microservices
spec:
  podSelector:
    matchLabels:
      tier: database
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              tier: api
      ports:
        - protocol: TCP
          port: 5432
```

```bash
kubectl apply -f network-policies.yaml

# 정책 확인
kubectl get networkpolicies -n microservices
# NAME                        POD-SELECTOR     AGE
# default-deny-all            <none>           5s
# allow-dns                   <none>           5s
# frontend-to-api             tier=frontend    5s
# api-ingress-from-frontend   tier=api         5s
# api-to-database             tier=api         5s
# database-ingress-from-api   tier=database    5s
```

</details>

### 연습문제 5: 최소 권한의 서비스 어카운트

`monitoring` 네임스페이스에 모든 네임스페이스에서 파드 로그만 읽을 수 있는 서비스 어카운트 `log-reader`를 생성하세요. 이 서비스 어카운트를 사용하는 파드를 배포하고 로그를 읽을 수 있지만 시크릿을 나열할 수 없는지 확인하세요.

<details><summary>정답 보기</summary>

```yaml
# log-reader-rbac.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: log-reader
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: pod-log-reader
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: log-reader-binding
subjects:
  - kind: ServiceAccount
    name: log-reader
    namespace: monitoring
roleRef:
  kind: ClusterRole
  name: pod-log-reader
  apiGroup: rbac.authorization.k8s.io
```

```yaml
# log-reader-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: log-reader-pod
  namespace: monitoring
spec:
  serviceAccountName: log-reader
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: kubectl
      image: bitnami/kubectl:latest
      command: ["sleep", "3600"]
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
```

```bash
kubectl apply -f log-reader-rbac.yaml
kubectl apply -f log-reader-pod.yaml

# 권한 확인
kubectl auth can-i get pods/log --all-namespaces \
  --as=system:serviceaccount:monitoring:log-reader
# yes

kubectl auth can-i list secrets --all-namespaces \
  --as=system:serviceaccount:monitoring:log-reader
# no

kubectl auth can-i create pods \
  --as=system:serviceaccount:monitoring:log-reader
# no

# 파드 내부에서 테스트
kubectl exec -it log-reader-pod -n monitoring -- \
  kubectl logs -n kube-system kube-apiserver-minikube --tail=5

kubectl exec -it log-reader-pod -n monitoring -- \
  kubectl get secrets -n kube-system
# Error from server (Forbidden)
```

</details>

---

**이전**: [구성 관리와 시크릿](./05_Configuration_and_Secrets.md) | **다음**: [인그레스와 Gateway API](./07_Ingress_and_Gateway_API.md)
