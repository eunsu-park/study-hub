# 07. Kubernetes 보안

## 학습 목표
- Kubernetes 보안 아키텍처 이해
- RBAC을 통한 접근 제어 구현
- NetworkPolicy로 네트워크 격리
- Secrets 및 민감 정보 관리
- Pod 보안 정책 적용

## 목차
1. [Kubernetes 보안 개요](#1-kubernetes-보안-개요)
2. [RBAC (역할 기반 접근 제어)](#2-rbac-역할-기반-접근-제어)
3. [ServiceAccount](#3-serviceaccount)
4. [NetworkPolicy](#4-networkpolicy)
5. [Secrets 관리](#5-secrets-관리)
6. [Pod 보안](#6-pod-보안)
7. [연습 문제](#7-연습-문제)

---

## 1. Kubernetes 보안 개요

### 1.1 4C 보안 모델

```
┌─────────────────────────────────────────────────────────────┐
│                     Cloud (클라우드)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Cluster (클러스터)                    │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │              Container (컨테이너)             │   │   │
│  │  │  ┌─────────────────────────────────────┐   │   │   │
│  │  │  │            Code (코드)               │   │   │   │
│  │  │  │  - 취약점 스캐닝                      │   │   │   │
│  │  │  │  - 의존성 관리                        │   │   │   │
│  │  │  │  - 보안 코딩                          │   │   │   │
│  │  │  └─────────────────────────────────────┘   │   │   │
│  │  │  - 이미지 보안                              │   │   │
│  │  │  - 런타임 보안                              │   │   │
│  │  │  - 리소스 제한                              │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │  - RBAC, NetworkPolicy                            │   │
│  │  - Secrets 관리                                    │   │
│  │  - Pod 보안                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│  - 네트워크 보안                                           │
│  - IAM, 방화벽                                            │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 인증과 인가

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│    User     │────▶│   API 서버   │────▶│   리소스    │
│  (kubectl)  │     │              │     │   (Pods)    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │   인증   │ │   인가   │ │ Admission│
        │(AuthN)   │ │(AuthZ)   │ │ Control  │
        ├──────────┤ ├──────────┤ ├──────────┤
        │• 인증서  │ │• RBAC    │ │• 검증    │
        │• 토큰    │ │• ABAC    │ │• 변환    │
        │• OIDC    │ │• Webhook │ │• 정책    │
        └──────────┘ └──────────┘ └──────────┘
```

### 1.3 보안 구성 요소

```yaml
# 현재 클러스터 보안 상태 확인
# API 서버 설정 확인
kubectl describe pod kube-apiserver-<master-node> -n kube-system

# 인증 모드 확인
kubectl api-versions | grep rbac
# rbac.authorization.k8s.io/v1

# 클러스터 권한 확인
kubectl auth can-i --list
```

---

## 2. RBAC (역할 기반 접근 제어)

### 2.1 RBAC 핵심 개념

```
┌─────────────────────────────────────────────────────────────┐
│                      RBAC 구성 요소                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐                  ┌───────────────┐      │
│  │     Role      │                  │  ClusterRole  │      │
│  │  (네임스페이스)│                  │   (클러스터)   │      │
│  └───────┬───────┘                  └───────┬───────┘      │
│          │                                  │               │
│          │ 바인딩                           │ 바인딩        │
│          ▼                                  ▼               │
│  ┌───────────────┐                  ┌───────────────┐      │
│  │ RoleBinding   │                  │ClusterRole    │      │
│  │               │                  │   Binding     │      │
│  └───────┬───────┘                  └───────┬───────┘      │
│          │                                  │               │
│          └──────────────┬───────────────────┘               │
│                         ▼                                   │
│                 ┌───────────────┐                           │
│                 │   Subjects    │                           │
│                 │ • User        │                           │
│                 │ • Group       │                           │
│                 │ • ServiceAcc  │                           │
│                 └───────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Role 정의

```yaml
# role-pod-reader.yaml
# 특정 네임스페이스에서 Pod 읽기 권한
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: development
  name: pod-reader
rules:
- apiGroups: [""]          # "" = core API group
  resources: ["pods"]
  verbs: ["get", "watch", "list"]

---
# role-deployment-manager.yaml
# Deployment 관리 권한
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: development
  name: deployment-manager
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]

---
# role-secret-reader.yaml
# 특정 Secret만 읽기 (resourceNames 사용)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: specific-secret-reader
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["app-config", "db-credentials"]  # 특정 리소스만
  verbs: ["get"]
```

### 2.3 ClusterRole 정의

```yaml
# clusterrole-node-reader.yaml
# 클러스터 전체에서 노드 정보 읽기
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: node-reader
rules:
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["get", "watch", "list"]

---
# clusterrole-pv-manager.yaml
# PersistentVolume 관리 (클러스터 범위 리소스)
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: pv-manager
rules:
- apiGroups: [""]
  resources: ["persistentvolumes"]
  verbs: ["get", "list", "watch", "create", "delete"]
- apiGroups: [""]
  resources: ["persistentvolumeclaims"]
  verbs: ["get", "list", "watch"]

---
# clusterrole-namespace-admin.yaml
# 모든 네임스페이스에서 관리자 역할
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: namespace-admin
rules:
- apiGroups: [""]
  resources: ["namespaces"]
  verbs: ["get", "list", "watch", "create", "delete"]
- apiGroups: [""]
  resources: ["*"]
  verbs: ["*"]

---
# 집계된 ClusterRole (Aggregation)
# clusterrole-monitoring.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: monitoring
  labels:
    rbac.example.com/aggregate-to-monitoring: "true"
aggregationRule:
  clusterRoleSelectors:
  - matchLabels:
      rbac.example.com/aggregate-to-monitoring: "true"
rules: []  # 규칙은 자동으로 집계됨
```

### 2.4 RoleBinding & ClusterRoleBinding

```yaml
# rolebinding-pod-reader.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: development
subjects:
- kind: User
  name: jane
  apiGroup: rbac.authorization.k8s.io
- kind: Group
  name: developers
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io

---
# rolebinding-sa.yaml
# ServiceAccount에 역할 바인딩
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: app-deployment-binding
  namespace: development
subjects:
- kind: ServiceAccount
  name: app-deployer
  namespace: development
roleRef:
  kind: Role
  name: deployment-manager
  apiGroup: rbac.authorization.k8s.io

---
# clusterrolebinding.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: node-reader-binding
subjects:
- kind: Group
  name: ops-team
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: node-reader
  apiGroup: rbac.authorization.k8s.io

---
# ClusterRole을 특정 네임스페이스에 바인딩
# (ClusterRole 재사용)
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: admin-binding
  namespace: staging
subjects:
- kind: User
  name: admin-user
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole      # ClusterRole이지만
  name: admin            # RoleBinding으로 범위 제한
  apiGroup: rbac.authorization.k8s.io
```

### 2.5 RBAC 테스트 및 디버깅

```bash
# 권한 확인
kubectl auth can-i create pods --namespace development
# yes

kubectl auth can-i delete pods --namespace production --as jane
# no

kubectl auth can-i '*' '*' --all-namespaces --as system:serviceaccount:default:admin
# yes

# 특정 사용자의 모든 권한 확인
kubectl auth can-i --list --as jane --namespace development

# RBAC 리소스 조회
kubectl get roles -n development
kubectl get rolebindings -n development
kubectl get clusterroles
kubectl get clusterrolebindings

# 상세 정보
kubectl describe role pod-reader -n development
kubectl describe rolebinding read-pods -n development
```

---

## 3. ServiceAccount

### 3.1 ServiceAccount 기본

```yaml
# serviceaccount.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-service-account
  namespace: production
  annotations:
    description: "Application service account for production"
# Kubernetes 1.24+에서는 토큰이 자동 생성되지 않음

---
# 토큰 생성 (Kubernetes 1.24+)
apiVersion: v1
kind: Secret
metadata:
  name: app-sa-token
  namespace: production
  annotations:
    kubernetes.io/service-account.name: app-service-account
type: kubernetes.io/service-account-token
```

### 3.2 Pod에서 ServiceAccount 사용

```yaml
# pod-with-sa.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
  namespace: production
spec:
  serviceAccountName: app-service-account
  automountServiceAccountToken: true  # 토큰 자동 마운트
  containers:
  - name: app
    image: myapp:latest
    # 토큰은 /var/run/secrets/kubernetes.io/serviceaccount/에 마운트됨

---
# 토큰 마운트 비활성화 (보안 강화)
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  serviceAccountName: restricted-sa
  automountServiceAccountToken: false  # 토큰 비마운트
  containers:
  - name: app
    image: myapp:latest
```

### 3.3 ServiceAccount를 위한 RBAC

```yaml
# CI/CD 파이프라인용 ServiceAccount 예시
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cicd-deployer
  namespace: cicd

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cicd-deployer-role
rules:
# Deployment 관리
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Service 관리
- apiGroups: [""]
  resources: ["services"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# ConfigMap, Secret 읽기
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
# Pod 상태 확인
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: cicd-deployer-binding
subjects:
- kind: ServiceAccount
  name: cicd-deployer
  namespace: cicd
roleRef:
  kind: ClusterRole
  name: cicd-deployer-role
  apiGroup: rbac.authorization.k8s.io
```

### 3.4 ServiceAccount 토큰 사용

```bash
# ServiceAccount 토큰 가져오기
TOKEN=$(kubectl create token app-service-account -n production)

# 또는 Secret에서 가져오기
TOKEN=$(kubectl get secret app-sa-token -n production -o jsonpath='{.data.token}' | base64 -d)

# 토큰으로 API 호출
curl -k -H "Authorization: Bearer $TOKEN" \
  https://kubernetes.default.svc/api/v1/namespaces/production/pods

# kubeconfig 생성
kubectl config set-credentials sa-user --token=$TOKEN
kubectl config set-context sa-context --cluster=my-cluster --user=sa-user
```

---

## 4. NetworkPolicy

### 4.1 NetworkPolicy 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    NetworkPolicy 동작                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NetworkPolicy 없음:                                        │
│  ┌─────┐     ┌─────┐     ┌─────┐                           │
│  │Pod A│◀───▶│Pod B│◀───▶│Pod C│  모든 트래픽 허용          │
│  └─────┘     └─────┘     └─────┘                           │
│                                                             │
│  NetworkPolicy 적용:                                        │
│  ┌─────┐     ┌─────┐     ┌─────┐                           │
│  │Pod A│────▶│Pod B│  ✗  │Pod C│  정책에 따라 제한          │
│  └─────┘     └─────┘     └─────┘                           │
│                                                             │
│  ⚠️  주의: CNI 플러그인이 NetworkPolicy를 지원해야 함        │
│      (Calico, Cilium, Weave Net 등)                        │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 기본 NetworkPolicy

```yaml
# deny-all-ingress.yaml
# 기본적으로 모든 인바운드 트래픽 거부
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
  namespace: production
spec:
  podSelector: {}  # 모든 Pod에 적용
  policyTypes:
  - Ingress
  # ingress 규칙 없음 = 모든 인바운드 거부

---
# deny-all-egress.yaml
# 모든 아웃바운드 트래픽 거부
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-egress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Egress
  # egress 규칙 없음 = 모든 아웃바운드 거부

---
# default-deny-all.yaml
# 모든 트래픽 거부 (가장 제한적)
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
```

### 4.3 허용 정책

```yaml
# allow-frontend-to-backend.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080

---
# allow-backend-to-database.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-backend-to-database
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: database
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: backend
    ports:
    - protocol: TCP
      port: 5432

---
# 다른 네임스페이스에서의 접근 허용
# allow-from-monitoring.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-monitoring
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
      podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 9090
```

### 4.4 복합 정책

```yaml
# comprehensive-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-server-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: api-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # 1. 같은 네임스페이스의 frontend에서 허용
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 443
  # 2. Ingress Controller에서 허용
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 443
  # 3. 특정 IP 대역에서 허용
  - from:
    - ipBlock:
        cidr: 10.0.0.0/8
        except:
        - 10.0.1.0/24  # 이 대역은 제외
    ports:
    - protocol: TCP
      port: 443
  egress:
  # 1. 데이터베이스로 아웃바운드
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
  # 2. 캐시 서버로 아웃바운드
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  # 3. DNS 허용 (필수!)
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
```

### 4.5 NetworkPolicy 디버깅

```bash
# NetworkPolicy 조회
kubectl get networkpolicy -n production
kubectl describe networkpolicy api-server-policy -n production

# Pod 레이블 확인
kubectl get pods -n production --show-labels

# 연결 테스트
kubectl run test-pod --rm -it --image=busybox -n production -- /bin/sh
# Pod 내부에서
wget -qO- --timeout=2 http://backend-service:8080
nc -zv database-service 5432

# CNI 플러그인 확인
kubectl get pods -n kube-system | grep -E "calico|cilium|weave"
```

---

## 5. Secrets 관리

### 5.1 Secret 유형

```yaml
# 1. Opaque (일반 데이터)
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: production
type: Opaque
data:
  # base64 인코딩 필요
  username: YWRtaW4=         # admin
  password: cGFzc3dvcmQxMjM=  # password123
stringData:
  # stringData는 인코딩 불필요
  api-key: my-secret-api-key

---
# 2. kubernetes.io/dockerconfigjson (컨테이너 레지스트리)
apiVersion: v1
kind: Secret
metadata:
  name: docker-registry-secret
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: eyJhdXRocyI6eyJodHRwczovL2luZGV4LmRvY2tlci5pby92MS8iOnsidXNlcm5hbWUiOiJ1c2VyIiwicGFzc3dvcmQiOiJwYXNzIiwiYXV0aCI6ImRYTmxjanB3WVhOeiJ9fX0=

---
# 3. kubernetes.io/tls (TLS 인증서)
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTi...
  tls.key: LS0tLS1CRUdJTi...

---
# 4. kubernetes.io/basic-auth
apiVersion: v1
kind: Secret
metadata:
  name: basic-auth
type: kubernetes.io/basic-auth
stringData:
  username: admin
  password: t0p-Secret
```

### 5.2 Secret 생성 명령어

```bash
# Opaque Secret (literal)
kubectl create secret generic db-credentials \
  --from-literal=username=admin \
  --from-literal=password=secret123 \
  -n production

# 파일에서 생성
kubectl create secret generic ssh-key \
  --from-file=ssh-privatekey=~/.ssh/id_rsa \
  --from-file=ssh-publickey=~/.ssh/id_rsa.pub

# Docker 레지스트리 시크릿
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=myuser \
  --docker-password=mytoken \
  --docker-email=user@example.com

# TLS 시크릿
kubectl create secret tls app-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem
```

### 5.3 Secret 사용

```yaml
# 환경 변수로 사용
apiVersion: v1
kind: Pod
metadata:
  name: app-with-secrets
spec:
  containers:
  - name: app
    image: myapp:latest
    env:
    # 특정 키만 사용
    - name: DB_USERNAME
      valueFrom:
        secretKeyRef:
          name: db-credentials
          key: username
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-credentials
          key: password
    # 전체 Secret을 환경변수로
    envFrom:
    - secretRef:
        name: app-secrets

---
# 볼륨으로 마운트
apiVersion: v1
kind: Pod
metadata:
  name: app-with-secret-volume
spec:
  containers:
  - name: app
    image: myapp:latest
    volumeMounts:
    - name: secret-volume
      mountPath: /etc/secrets
      readOnly: true
    - name: tls-volume
      mountPath: /etc/tls
      readOnly: true
  volumes:
  - name: secret-volume
    secret:
      secretName: app-secrets
      # 특정 키만 마운트
      items:
      - key: api-key
        path: api-key.txt
        mode: 0400  # 파일 권한
  - name: tls-volume
    secret:
      secretName: tls-secret

---
# 이미지 Pull Secret
apiVersion: v1
kind: Pod
metadata:
  name: private-image-pod
spec:
  containers:
  - name: app
    image: ghcr.io/myorg/private-app:latest
  imagePullSecrets:
  - name: regcred
```

### 5.4 Secret 보안 강화

```yaml
# Secret 암호화 설정 (kube-apiserver)
# /etc/kubernetes/encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      - aescbc:
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-key>
      - identity: {}  # 폴백 (암호화 안 됨)

---
# RBAC으로 Secret 접근 제한
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-reader
  namespace: production
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["app-secrets"]  # 특정 Secret만
  verbs: ["get"]
```

### 5.5 외부 Secret 관리 도구

```yaml
# External Secrets Operator 예시
# AWS Secrets Manager에서 가져오기
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: aws-secret
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager
    kind: SecretStore
  target:
    name: db-credentials  # 생성될 K8s Secret 이름
  data:
  - secretKey: username
    remoteRef:
      key: production/db-credentials
      property: username
  - secretKey: password
    remoteRef:
      key: production/db-credentials
      property: password

---
# Sealed Secrets (GitOps용)
# kubeseal로 암호화
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: mysecret
  namespace: production
spec:
  encryptedData:
    password: AgBy8hCi...암호화된데이터...
```

---

## 6. Pod 보안

### 6.1 Pod Security Standards

```
┌─────────────────────────────────────────────────────────────┐
│              Pod Security Standards (PSS)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Privileged (권한)                                          │
│  ├── 제한 없음                                              │
│  └── 시스템 Pod용                                           │
│                                                             │
│  Baseline (기준)                                            │
│  ├── 알려진 권한 상승 방지                                   │
│  ├── hostNetwork, hostPID 금지                              │
│  └── 대부분의 워크로드에 적합                                │
│                                                             │
│  Restricted (제한)                                          │
│  ├── 강력한 보안 정책                                       │
│  ├── 비root 실행 필수                                       │
│  ├── 읽기 전용 루트 파일시스템                              │
│  └── 보안 민감 워크로드용                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Pod Security Admission

```yaml
# 네임스페이스에 보안 레벨 적용
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    # enforce: 위반 시 거부
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest
    # audit: 감사 로그에 기록
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: latest
    # warn: 경고 메시지 표시
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: latest

---
# baseline 레벨 네임스페이스
apiVersion: v1
kind: Namespace
metadata:
  name: staging
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/warn: restricted
```

### 6.3 보안 컨텍스트

```yaml
# secure-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  # Pod 레벨 보안 컨텍스트
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 3000
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault

  containers:
  - name: app
    image: myapp:latest
    # 컨테이너 레벨 보안 컨텍스트
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
          - ALL
        # 필요한 capability만 추가
        # add:
        #   - NET_BIND_SERVICE

    # 리소스 제한
    resources:
      limits:
        cpu: "500m"
        memory: "128Mi"
      requests:
        cpu: "250m"
        memory: "64Mi"

    # 임시 볼륨 (읽기 전용 루트에서 쓰기 필요 시)
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: cache
      mountPath: /app/cache

  volumes:
  - name: tmp
    emptyDir: {}
  - name: cache
    emptyDir:
      sizeLimit: 100Mi
```

### 6.4 고급 보안 설정

```yaml
# highly-secure-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: secure-app
  template:
    metadata:
      labels:
        app: secure-app
    spec:
      # ServiceAccount 토큰 비마운트
      automountServiceAccountToken: false

      # Pod 보안 컨텍스트
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534  # nobody
        runAsGroup: 65534
        fsGroup: 65534
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: app
        image: myapp:latest
        imagePullPolicy: Always

        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
              - ALL

        # 포트
        ports:
        - containerPort: 8080
          protocol: TCP

        # 리소스 제한
        resources:
          limits:
            cpu: "1"
            memory: "512Mi"
          requests:
            cpu: "100m"
            memory: "128Mi"

        # 헬스 체크
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: config
          mountPath: /etc/app
          readOnly: true

      volumes:
      - name: tmp
        emptyDir:
          medium: Memory
          sizeLimit: 64Mi
      - name: config
        configMap:
          name: app-config

      # 호스트 네트워크/PID 사용 금지
      hostNetwork: false
      hostPID: false
      hostIPC: false

      # DNS 정책
      dnsPolicy: ClusterFirst
```

### 6.5 보안 스캐닝

```bash
# 이미지 취약점 스캐닝 (Trivy)
trivy image myapp:latest

# 클러스터 보안 스캔 (kubescape)
kubescape scan framework nsa --exclude-namespaces kube-system

# Pod 보안 검사 (kube-bench)
kubectl apply -f https://raw.githubusercontent.com/aquasecurity/kube-bench/main/job.yaml
kubectl logs job/kube-bench

# OPA/Gatekeeper 정책 확인
kubectl get constrainttemplates
kubectl get constraints
```

---

## 7. 연습 문제

### 연습 1: 개발팀 RBAC 구성
```yaml
# 요구사항:
# - development 네임스페이스에서 개발자는 Pod, Deployment, Service 관리 가능
# - production 네임스페이스에서는 Pod 조회만 가능
# - Secret 접근 불가

# Role 및 RoleBinding 작성
```

### 연습 2: 마이크로서비스 NetworkPolicy
```yaml
# 요구사항:
# - frontend -> api-gateway -> backend -> database 순서로만 통신
# - monitoring 네임스페이스에서 모든 Pod의 /metrics 접근 허용
# - 외부에서 frontend만 접근 가능

# NetworkPolicy 작성
```

### 연습 3: 안전한 애플리케이션 배포
```yaml
# 요구사항:
# - 비root 사용자로 실행
# - 읽기 전용 루트 파일시스템
# - 모든 capability 제거
# - 리소스 제한 설정
# - Secret을 환경변수와 볼륨으로 마운트

# Deployment 작성
```

### 연습 4: 보안 감사
```bash
# 다음 항목 점검:
# 1. 클러스터에서 privileged Pod 찾기
# 2. default ServiceAccount 사용하는 Pod 찾기
# 3. Secret이 환경변수로 노출된 Pod 찾기
# 4. NetworkPolicy가 없는 네임스페이스 찾기

# 명령어 작성
```

---

## 다음 단계

- [08_Kubernetes_심화](08_Kubernetes_심화.md) - Ingress, StatefulSet, PV/PVC
- [09_Helm_패키지관리](09_Helm_패키지관리.md) - Helm 차트 관리
- [10_CI_CD_파이프라인](10_CI_CD_파이프라인.md) - 자동화 배포

## 참고 자료

- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [RBAC Documentation](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)

---

[← 이전: Docker Compose](06_Docker_Compose.md) | [다음: Kubernetes 심화 →](08_Kubernetes_심화.md) | [목차](00_Overview.md)
