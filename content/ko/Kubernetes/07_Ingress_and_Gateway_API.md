# 07. 인그레스와 Gateway API(Ingress and Gateway API)

**이전**: [RBAC와 보안](./06_RBAC_and_Security.md) | **다음**: [CNI와 고급 네트워킹](./08_CNI_and_Advanced_Networking.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 호스트 기반 및 경로 기반 라우팅 규칙으로 Ingress 리소스를 구성할 수 있다
2. Ingress 컨트롤러(NGINX, Traefik)를 배포하고 비교할 수 있다
3. cert-manager를 사용하여 TLS 종료 및 자동 인증서 관리를 설정할 수 있다
4. Gateway API(Gateway, HTTPRoute, GRPCRoute)를 사용하여 트래픽 라우팅을 설계할 수 있다
5. 속도 제한(rate limiting), 인증, 트래픽 분할(traffic splitting)을 포함한 고급 패턴을 구현할 수 있다

---

Kubernetes Service는 클러스터 내에서 애플리케이션을 노출하지만, 외부에서 접근하려면 추가 계층이 필요합니다. Ingress 리소스와 최신 Gateway API는 외부 URL을 내부 서비스에 매핑하는 선언적, HTTP 인식 라우팅을 제공합니다. 이 레슨에서는 성숙한 Ingress API와 차세대 Gateway API를 모두 다루며, TLS 관리, 트래픽 분할, 게이트웨이 수준 인증을 포함합니다.

> **Ingress vs Gateway API:** Ingress는 Kubernetes 1.19부터 안정적이지만 잘 알려진 한계가 있습니다 -- 벤더별 어노테이션, TCP/UDP 라우팅 미지원, 평면적 권한 모델. Gateway API(Kubernetes 1.27부터 코어 리소스 GA)는 역할 중심적이고, 표현력이 풍부하며, 이식 가능한 API로 이러한 문제를 해결합니다. 새 프로젝트는 Gateway API를 선호해야 하지만, Ingress는 여전히 널리 배포되어 있습니다.

## 목차

- [1. Ingress 기본](#1-ingress-기본)
  - [1.1 Ingress 리소스 구조](#11-ingress-리소스-구조)
  - [1.2 호스트 기반 라우팅(Host-Based Routing)](#12-호스트-기반-라우팅host-based-routing)
  - [1.3 경로 기반 라우팅(Path-Based Routing)](#13-경로-기반-라우팅path-based-routing)
  - [1.4 기본 백엔드(Default Backend)](#14-기본-백엔드default-backend)
- [2. Ingress 컨트롤러](#2-ingress-컨트롤러)
  - [2.1 NGINX Ingress 컨트롤러](#21-nginx-ingress-컨트롤러)
  - [2.2 Traefik Ingress 컨트롤러](#22-traefik-ingress-컨트롤러)
  - [2.3 컨트롤러 비교](#23-컨트롤러-비교)
- [3. TLS 종료(TLS Termination)](#3-tls-종료tls-termination)
  - [3.1 수동 TLS 구성](#31-수동-tls-구성)
  - [3.2 cert-manager를 사용한 자동 인증서](#32-cert-manager를-사용한-자동-인증서)
- [4. Gateway API](#4-gateway-api)
  - [4.1 아키텍처와 리소스 모델](#41-아키텍처와-리소스-모델)
  - [4.2 GatewayClass와 Gateway](#42-gatewayclass와-gateway)
  - [4.3 HTTPRoute](#43-httproute)
  - [4.4 GRPCRoute](#44-grpcroute)
  - [4.5 Gateway API의 TLS](#45-gateway-api의-tls)
- [5. Gateway API vs Ingress](#5-gateway-api-vs-ingress)
- [6. 고급 패턴](#6-고급-패턴)
  - [6.1 속도 제한(Rate Limiting)](#61-속도-제한rate-limiting)
  - [6.2 게이트웨이에서의 인증](#62-게이트웨이에서의-인증)
  - [6.3 트래픽 분할(Traffic Splitting)](#63-트래픽-분할traffic-splitting)
  - [6.4 URL 재작성과 리다이렉트](#64-url-재작성과-리다이렉트)
- [연습문제](#연습문제)

---

## 1. Ingress 기본

Ingress 리소스는 외부 HTTP(S) 트래픽을 클러스터 내 서비스로 라우팅하는 규칙을 정의합니다. 작동하려면 Ingress 컨트롤러가 필요합니다.

```
                    ┌─────────────────────────────────────────┐
                    │             Ingress Controller           │
  Internet ──────▶ │  (NGINX / Traefik / HAProxy / Envoy)    │
                    │                                         │
                    │  Rules:                                  │
                    │  app.example.com ──▶ app-service:80     │
                    │  api.example.com ──▶ api-service:8080   │
                    │  example.com/docs ──▶ docs-service:80   │
                    └─────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌──────────┐   ┌──────────┐   ┌──────────┐
              │app-service│  │api-service│  │docs-service│
              └──────────┘   └──────────┘   └──────────┘
```

### 1.1 Ingress 리소스 구조

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
  namespace: production
  annotations:
    # 컨트롤러별 어노테이션
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx          # 이 리소스를 처리할 컨트롤러 지정
  defaultBackend:                  # 모든 요청의 기본 백엔드
    service:
      name: default-service
      port:
        number: 80
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: app-service
                port:
                  number: 80
  tls:
    - hosts:
        - app.example.com
      secretName: app-tls-cert
```

**경로 유형(Path Types):**

| PathType | 동작 | 예시 |
|----------|------|------|
| `Prefix` | 경로 요소별 URL 경로 접두사 매칭 | `/api`는 `/api`, `/api/`, `/api/v1`과 매칭 |
| `Exact` | URL 경로와 정확히 매칭 | `/api`는 `/api`만 매칭, `/api/`는 아님 |
| `ImplementationSpecific` | IngressClass에 따라 매칭 결정 | 컨트롤러 정의 동작 |

### 1.2 호스트 기반 라우팅(Host-Based Routing)

`Host` 헤더를 기반으로 다른 서비스로 트래픽을 라우팅합니다.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: multi-host-ingress
  namespace: production
spec:
  ingressClassName: nginx
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend-service
                port:
                  number: 80
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 8080
    - host: grafana.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: grafana-service
                port:
                  number: 3000
```

### 1.3 경로 기반 라우팅(Path-Based Routing)

URL 경로를 기반으로 다른 서비스로 트래픽을 라우팅합니다.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: path-based-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/use-regex: "true"
spec:
  ingressClassName: nginx
  rules:
    - host: example.com
      http:
        paths:
          - path: /api/v1
            pathType: Prefix
            backend:
              service:
                name: api-v1-service
                port:
                  number: 8080
          - path: /api/v2
            pathType: Prefix
            backend:
              service:
                name: api-v2-service
                port:
                  number: 8080
          - path: /static
            pathType: Prefix
            backend:
              service:
                name: cdn-service
                port:
                  number: 80
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend-service
                port:
                  number: 80
```

### 1.4 기본 백엔드(Default Backend)

기본 백엔드는 어떤 규칙에도 매칭되지 않는 요청을 처리합니다. 사용자 정의 404 페이지에 유용합니다.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: with-default-backend
spec:
  ingressClassName: nginx
  defaultBackend:
    service:
      name: custom-404-service
      port:
        number: 80
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: app-service
                port:
                  number: 80
```

---

## 2. Ingress 컨트롤러

Ingress 리소스는 단지 구성 객체입니다. 그 자체로는 아무것도 하지 않습니다 -- Ingress 컨트롤러가 Ingress 리소스를 감시하고 기반 로드 밸런서를 구성합니다.

### 2.1 NGINX Ingress 컨트롤러

가장 널리 배포된 Ingress 컨트롤러입니다. Ingress 리소스에서 NGINX 구성을 생성합니다.

```bash
# minikube에 NGINX Ingress 컨트롤러 설치
minikube addons enable ingress

# 설치 확인
kubectl get pods -n ingress-nginx
# NAME                                       READY   STATUS
# ingress-nginx-controller-xxxxx-xxxxx       1/1     Running

# 또는 Helm으로 설치 (프로덕션)
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.replicaCount=2 \
  --set controller.metrics.enabled=true
```

주요 NGINX Ingress 어노테이션:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nginx-features
  annotations:
    # 속도 제한
    nginx.ingress.kubernetes.io/limit-rps: "10"
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "5"

    # 타임아웃
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "10"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"

    # 본문 크기
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"

    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.example.com"

    # HTTP를 HTTPS로 리다이렉트
    nginx.ingress.kubernetes.io/ssl-redirect: "true"

    # WebSocket 지원
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"

    # 사용자 정의 NGINX 구성 스니펫
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
spec:
  ingressClassName: nginx
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: app-service
                port:
                  number: 80
```

### 2.2 Traefik Ingress 컨트롤러

Traefik은 자동 서비스 디스커버리와 내장 Let's Encrypt 지원을 갖춘 클라우드 네이티브 리버스 프록시(reverse proxy)입니다.

```bash
# Helm으로 Traefik 설치
helm repo add traefik https://traefik.github.io/charts
helm install traefik traefik/traefik \
  --namespace traefik \
  --create-namespace \
  --set ports.web.port=8000 \
  --set ports.websecure.port=8443
```

```yaml
# Traefik은 고급 기능을 위해 자체 IngressRoute CRD를 사용
apiVersion: traefik.io/v1alpha1
kind: IngressRoute
metadata:
  name: app-route
  namespace: production
spec:
  entryPoints:
    - websecure
  routes:
    - match: Host(`app.example.com`) && PathPrefix(`/api`)
      kind: Rule
      services:
        - name: api-service
          port: 8080
      middlewares:
        - name: rate-limit
        - name: strip-prefix
    - match: Host(`app.example.com`)
      kind: Rule
      services:
        - name: frontend-service
          port: 80
  tls:
    certResolver: letsencrypt
---
# 속도 제한을 위한 Traefik 미들웨어
apiVersion: traefik.io/v1alpha1
kind: Middleware
metadata:
  name: rate-limit
  namespace: production
spec:
  rateLimit:
    average: 100
    burst: 50
    period: 1m
---
apiVersion: traefik.io/v1alpha1
kind: Middleware
metadata:
  name: strip-prefix
  namespace: production
spec:
  stripPrefix:
    prefixes:
      - /api
```

### 2.3 컨트롤러 비교

| 기능 | NGINX Ingress | Traefik | Envoy (Contour) |
|------|--------------|---------|------------------|
| 구성 | 어노테이션 | CRD + 어노테이션 | CRD (HTTPProxy) |
| 핫 리로드(Hot reload) | 예 (lua) | 예 (네이티브) | 예 (xDS) |
| Let's Encrypt | cert-manager 경유 | 내장 | cert-manager 경유 |
| TCP/UDP | ConfigMap | 네이티브 CRD | 네이티브 CRD |
| 대시보드 | 없음 (Prometheus 사용) | 내장 | 없음 |
| Gateway API | 예 (v1.0+) | 예 (v3.0+) | 예 (v1.28+) |
| 시장 점유율 | 최고 | 2위 | 성장 중 |

---

## 3. TLS 종료(TLS Termination)

Ingress 또는 Gateway 수준에서 TLS를 종료하면 애플리케이션 서비스에서 암호화를 오프로드합니다.

### 3.1 수동 TLS 구성

```bash
# 자체 서명 인증서 생성 (개발 전용)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=app.example.com"

# Kubernetes TLS 시크릿 생성
kubectl create secret tls app-tls-cert \
  --cert=tls.crt \
  --key=tls.key \
  --namespace=production
```

```yaml
# TLS가 있는 Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tls-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - app.example.com
        - api.example.com
      secretName: app-tls-cert
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: app-service
                port:
                  number: 80
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 8080
```

### 3.2 cert-manager를 사용한 자동 인증서

cert-manager는 Let's Encrypt 및 기타 CA로부터 TLS 인증서의 프로비저닝과 갱신을 자동화합니다.

```bash
# cert-manager 설치
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set crds.enabled=true
```

```yaml
# Let's Encrypt 프로덕션용 ClusterIssuer
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod-account-key
    solvers:
      - http01:
          ingress:
            ingressClassName: nginx
```

```yaml
# Let's Encrypt 스테이징용 ClusterIssuer (테스트용)
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-staging
spec:
  acme:
    server: https://acme-staging-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-staging-account-key
    solvers:
      - http01:
          ingress:
            ingressClassName: nginx
```

```yaml
# 자동 인증서 프로비저닝이 있는 Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: auto-tls-ingress
  namespace: production
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - app.example.com
      secretName: app-tls-auto   # cert-manager가 이 시크릿을 생성
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: app-service
                port:
                  number: 80
```

```bash
# 인증서 상태 모니터링
kubectl get certificates -n production
# NAME           READY   SECRET         AGE
# app-tls-auto   True    app-tls-auto   5m

kubectl describe certificate app-tls-auto -n production

# 인증서 주문 및 챌린지 확인
kubectl get orders -n production
kubectl get challenges -n production
```

---

## 4. Gateway API

Gateway API는 Kubernetes에서 인그레스 트래픽을 관리하기 위한 차세대 API입니다. Ingress보다 더 표현력이 풍부하고, 역할 중심적이며, 이식 가능한 대안을 제공합니다.

### 4.1 아키텍처와 리소스 모델

```
┌─────────────────────────────────────────────────────────────┐
│                    역할 중심 설계                              │
│                                                              │
│  인프라 제공자          클러스터 운영자      앱 개발자          │
│  ┌──────────────┐          ┌──────────┐       ┌───────────┐ │
│  │ GatewayClass │──────────│ Gateway  │───────│ HTTPRoute │ │
│  │              │          │          │       │ GRPCRoute │ │
│  │ "어떤        │          │ "어디서  │       │ TCPRoute  │ │
│  │  컨트롤러"   │          │  수신"   │       │ TLSRoute  │ │
│  │              │          │          │       │           │ │
│  └──────────────┘          └──────────┘       └───────────┘ │
└─────────────────────────────────────────────────────────────┘
```

```bash
# Gateway API CRD 설치
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml

# CRD 확인
kubectl get crd | grep gateway
# gatewayclasses.gateway.networking.k8s.io
# gateways.gateway.networking.k8s.io
# httproutes.gateway.networking.k8s.io
# grpcroutes.gateway.networking.k8s.io
# referencegrants.gateway.networking.k8s.io
```

### 4.2 GatewayClass와 Gateway

```yaml
# GatewayClass: 사용할 컨트롤러 구현체 정의
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: nginx
spec:
  controllerName: gateway.nginx.org/nginx-gateway-controller
```

```yaml
# Gateway: 리스너가 있는 실제 로드 밸런서 생성
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: main-gateway
  namespace: gateway-infra
spec:
  gatewayClassName: nginx
  listeners:
    - name: http
      protocol: HTTP
      port: 80
      allowedRoutes:
        namespaces:
          from: All           # 모든 네임스페이스에서 라우트 수락
    - name: https
      protocol: HTTPS
      port: 443
      tls:
        mode: Terminate
        certificateRefs:
          - name: wildcard-tls
            kind: Secret
      allowedRoutes:
        namespaces:
          from: Selector
          selector:
            matchLabels:
              gateway-access: "true"   # 레이블된 네임스페이스만
    - name: api
      protocol: HTTPS
      port: 443
      hostname: "api.example.com"     # 리스너 수준 호스트명 필터
      tls:
        mode: Terminate
        certificateRefs:
          - name: api-tls
            kind: Secret
      allowedRoutes:
        namespaces:
          from: Same           # 이 네임스페이스의 라우트만
```

### 4.3 HTTPRoute

HTTPRoute는 HTTP 트래픽을 위한 핵심 라우팅 리소스입니다.

```yaml
# HTTPRoute: 호스트 기반 및 경로 기반 라우팅
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: app-routes
  namespace: production
spec:
  parentRefs:
    - name: main-gateway
      namespace: gateway-infra
      sectionName: https          # https 리스너에 연결
  hostnames:
    - "app.example.com"
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /api/v2
      backendRefs:
        - name: api-v2-service
          port: 8080
    - matches:
        - path:
            type: PathPrefix
            value: /api
      backendRefs:
        - name: api-v1-service
          port: 8080
    - matches:
        - path:
            type: PathPrefix
            value: /
      backendRefs:
        - name: frontend-service
          port: 80
```

```yaml
# HTTPRoute: 헤더 기반 라우팅
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: header-routing
  namespace: production
spec:
  parentRefs:
    - name: main-gateway
      namespace: gateway-infra
  hostnames:
    - "api.example.com"
  rules:
    - matches:
        - headers:
            - name: X-API-Version
              value: "2"
      backendRefs:
        - name: api-v2-service
          port: 8080
    - matches:
        - headers:
            - name: X-API-Version
              value: "1"
      backendRefs:
        - name: api-v1-service
          port: 8080
    # 기본 라우트 (헤더 매칭 없음)
    - backendRefs:
        - name: api-v1-service
          port: 8080
```

### 4.4 GRPCRoute

```yaml
# gRPC 서비스를 위한 GRPCRoute
apiVersion: gateway.networking.k8s.io/v1
kind: GRPCRoute
metadata:
  name: grpc-routes
  namespace: production
spec:
  parentRefs:
    - name: main-gateway
      namespace: gateway-infra
      sectionName: https
  hostnames:
    - "grpc.example.com"
  rules:
    - matches:
        - method:
            service: myapp.UserService
            method: GetUser
      backendRefs:
        - name: user-grpc-service
          port: 50051
    - matches:
        - method:
            service: myapp.OrderService
      backendRefs:
        - name: order-grpc-service
          port: 50051
```

### 4.5 Gateway API의 TLS

```yaml
# TLS 패스스루(passthrough)가 있는 Gateway (백엔드에서 종료)
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: passthrough-gateway
  namespace: gateway-infra
spec:
  gatewayClassName: nginx
  listeners:
    - name: tls-passthrough
      protocol: TLS
      port: 443
      tls:
        mode: Passthrough       # TLS를 종료하지 않음
      allowedRoutes:
        namespaces:
          from: All
```

```yaml
# Gateway API와 cert-manager 통합
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: wildcard-tls
  namespace: gateway-infra
spec:
  secretName: wildcard-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - "*.example.com"
    - "example.com"
```

---

## 5. Gateway API vs Ingress

| 기능 | Ingress | Gateway API |
|------|---------|-------------|
| API 성숙도 | 안정 (v1, 1.19+) | 안정 코어 (v1, 1.27+) |
| 라우팅 표현력 | 기본 (호스트, 경로) | 풍부 (헤더, 쿼리, 메서드) |
| 프로토콜 지원 | HTTP(S)만 | HTTP, HTTPS, gRPC, TCP, TLS |
| 구성 | 어노테이션 (이식 불가) | 네이티브 스펙 필드 (이식 가능) |
| 역할 분리 | 없음 (단일 리소스) | GatewayClass / Gateway / Route |
| 크로스 네임스페이스 | 미지원 | ReferenceGrant |
| 트래픽 분할 | 어노테이션 (컨트롤러별) | 네이티브 `backendRefs` (가중치) |
| 헤더 수정 | 어노테이션 | 네이티브 `RequestHeaderModifier` |
| 리다이렉트 / 재작성 | 어노테이션 | 네이티브 `RequestRedirect` / `URLRewrite` |

**마이그레이션 전략:** Ingress와 Gateway API를 동시에 실행할 수 있습니다. 기존 Ingress 리소스와 함께 Gateway API를 배포하고, 점진적으로 라우트를 마이그레이션하세요.

---

## 6. 고급 패턴

### 6.1 속도 제한(Rate Limiting)

```yaml
# NGINX Ingress: 어노테이션 기반 속도 제한
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rate-limited-api
  annotations:
    nginx.ingress.kubernetes.io/limit-rps: "10"
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "3"
    nginx.ingress.kubernetes.io/limit-connections: "5"
spec:
  ingressClassName: nginx
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 8080
```

```yaml
# Gateway API: 정책 첨부를 통한 속도 제한 (구현체별)
# NGINX Gateway Fabric 사용 예시
apiVersion: gateway.nginx.org/v1alpha1
kind: NginxProxy
metadata:
  name: rate-limit-policy
spec:
  rateLimiting:
    rate: 10
    burst: 30
    key: "${remote_addr}"
    rejectStatusCode: 429
```

### 6.2 게이트웨이에서의 인증

```yaml
# NGINX Ingress: 외부 인증
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: auth-ingress
  annotations:
    # 외부 인증 서비스 (OAuth2 Proxy, Authelia 등)
    nginx.ingress.kubernetes.io/auth-url: "https://auth.example.com/oauth2/auth"
    nginx.ingress.kubernetes.io/auth-signin: "https://auth.example.com/oauth2/start?rd=$scheme://$host$request_uri"
    nginx.ingress.kubernetes.io/auth-response-headers: "X-Auth-User,X-Auth-Email"
spec:
  ingressClassName: nginx
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: app-service
                port:
                  number: 80
```

```yaml
# NGINX Ingress의 Basic 인증
# 먼저 htpasswd 시크릿 생성
# htpasswd -c auth admin
# kubectl create secret generic basic-auth --from-file=auth -n production

apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: basic-auth-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: basic-auth
    nginx.ingress.kubernetes.io/auth-realm: "Authentication Required"
spec:
  ingressClassName: nginx
  rules:
    - host: admin.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: admin-service
                port:
                  number: 80
```

### 6.3 트래픽 분할(Traffic Splitting)

트래픽 분할은 카나리 배포(canary deployment), A/B 테스트, 점진적 롤아웃을 가능하게 합니다.

```yaml
# Gateway API: 가중 트래픽 분할 (카나리 배포)
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: canary-route
  namespace: production
spec:
  parentRefs:
    - name: main-gateway
      namespace: gateway-infra
  hostnames:
    - "app.example.com"
  rules:
    - backendRefs:
        - name: app-stable
          port: 80
          weight: 90            # 90%를 안정 버전으로
        - name: app-canary
          port: 80
          weight: 10            # 10%를 카나리로
```

```yaml
# NGINX Ingress: 어노테이션을 사용한 카나리 배포
# 메인 Ingress (기본적으로 모든 트래픽 수신)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-main
  namespace: production
spec:
  ingressClassName: nginx
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: app-stable
                port:
                  number: 80
---
# 카나리 Ingress (트래픽의 일정 비율 수신)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-canary
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"
spec:
  ingressClassName: nginx
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: app-canary
                port:
                  number: 80
```

```yaml
# 헤더 기반 카나리 (특정 사용자를 카나리로 라우팅)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-canary-header
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-by-header: "X-Canary"
    nginx.ingress.kubernetes.io/canary-by-header-value: "true"
spec:
  ingressClassName: nginx
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: app-canary
                port:
                  number: 80
```

### 6.4 URL 재작성과 리다이렉트

```yaml
# Gateway API: URL 재작성
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: rewrite-route
  namespace: production
spec:
  parentRefs:
    - name: main-gateway
      namespace: gateway-infra
  hostnames:
    - "app.example.com"
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /old-api
      filters:
        - type: URLRewrite
          urlRewrite:
            path:
              type: ReplacePrefixMatch
              replacePrefixMatch: /api/v2
      backendRefs:
        - name: api-v2-service
          port: 8080
```

```yaml
# Gateway API: HTTP에서 HTTPS로 리다이렉트
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: redirect-to-https
  namespace: production
spec:
  parentRefs:
    - name: main-gateway
      namespace: gateway-infra
      sectionName: http          # HTTP 리스너에 연결
  hostnames:
    - "app.example.com"
  rules:
    - filters:
        - type: RequestRedirect
          requestRedirect:
            scheme: https
            statusCode: 301
```

```yaml
# Gateway API: ReferenceGrant를 사용한 크로스 네임스페이스 라우팅
# "production" 네임스페이스의 라우트가 "backend" 네임스페이스의 서비스를 참조하도록 허용
apiVersion: gateway.networking.k8s.io/v1beta1
kind: ReferenceGrant
metadata:
  name: allow-production-routes
  namespace: backend               # 참조되는 네임스페이스
spec:
  from:
    - group: gateway.networking.k8s.io
      kind: HTTPRoute
      namespace: production        # 참조를 만드는 네임스페이스
  to:
    - group: ""
      kind: Service
```

---

## 연습문제

### 연습문제 1: 다중 서비스 Ingress

경로 기반으로 세 개의 서비스로 트래픽을 라우팅하는 Ingress 리소스를 생성하세요:
- `/`는 `frontend` 서비스(포트 80)로
- `/api`는 `api-server` 서비스(포트 8080)로
- `/ws`는 `websocket` 서비스(포트 9090)로 WebSocket 지원과 함께

모든 서비스는 `webapp` 네임스페이스에 있습니다.

<details><summary>정답 보기</summary>

```yaml
# multi-service-ingress.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: webapp
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: webapp-ingress
  namespace: webapp
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
spec:
  ingressClassName: nginx
  rules:
    - host: webapp.local
      http:
        paths:
          - path: /ws
            pathType: Prefix
            backend:
              service:
                name: websocket
                port:
                  number: 9090
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: api-server
                port:
                  number: 8080
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend
                port:
                  number: 80
```

```bash
kubectl apply -f multi-service-ingress.yaml

# minikube에서 Ingress IP 가져오기
minikube ip
# /etc/hosts에 추가: <minikube-ip> webapp.local

# 각 경로 테스트
curl http://webapp.local/
curl http://webapp.local/api/health
# WebSocket 테스트 (wscat 사용)
# wscat -c ws://webapp.local/ws
```

</details>

### 연습문제 2: cert-manager를 사용한 TLS

자체 서명 ClusterIssuer(minikube용)로 cert-manager를 설정한 후, `secure.local`에 대한 자동 TLS 인증서 프로비저닝이 있는 Ingress를 생성하세요.

<details><summary>정답 보기</summary>

```bash
# cert-manager 설치
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set crds.enabled=true

# cert-manager 파드 대기
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager \
  -n cert-manager --timeout=120s
```

```yaml
# self-signed-issuer.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: selfsigned-issuer
spec:
  selfSigned: {}
---
# tls-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: secure-ingress
  namespace: webapp
  annotations:
    cert-manager.io/cluster-issuer: "selfsigned-issuer"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - secure.local
      secretName: secure-local-tls
  rules:
    - host: secure.local
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend
                port:
                  number: 80
```

```bash
kubectl apply -f self-signed-issuer.yaml
kubectl apply -f tls-ingress.yaml

# 인증서가 발급되었는지 확인
kubectl get certificate -n webapp
# NAME               READY   SECRET             AGE
# secure-local-tls   True    secure-local-tls   30s

kubectl describe certificate secure-local-tls -n webapp

# HTTPS 테스트 (자체 서명이므로 -k 사용)
curl -k https://secure.local/
```

</details>

### 연습문제 3: Gateway API HTTPRoute

Gateway API를 사용하여 다음과 같은 Gateway와 HTTPRoute를 생성하세요:
- 포트 80(HTTP)과 포트 443(HTTPS)에서 수신
- `app.example.com`을 `app-service:80`으로 라우팅
- `api.example.com/v1/*`을 `api-v1:8080`으로, `api.example.com/v2/*`를 `api-v2:8080`으로 라우팅

<details><summary>정답 보기</summary>

```yaml
# gateway-setup.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: nginx
spec:
  controllerName: gateway.nginx.org/nginx-gateway-controller
---
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: main-gw
  namespace: gateway-infra
spec:
  gatewayClassName: nginx
  listeners:
    - name: http
      protocol: HTTP
      port: 80
      allowedRoutes:
        namespaces:
          from: All
    - name: https
      protocol: HTTPS
      port: 443
      tls:
        mode: Terminate
        certificateRefs:
          - name: wildcard-cert
      allowedRoutes:
        namespaces:
          from: All
---
# App HTTPRoute
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: app-route
  namespace: production
spec:
  parentRefs:
    - name: main-gw
      namespace: gateway-infra
  hostnames:
    - "app.example.com"
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /
      backendRefs:
        - name: app-service
          port: 80
---
# 버전별 경로가 있는 API HTTPRoute
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: api-route
  namespace: production
spec:
  parentRefs:
    - name: main-gw
      namespace: gateway-infra
  hostnames:
    - "api.example.com"
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /v2
      backendRefs:
        - name: api-v2
          port: 8080
    - matches:
        - path:
            type: PathPrefix
            value: /v1
      backendRefs:
        - name: api-v1
          port: 8080
    - backendRefs:
        - name: api-v1
          port: 8080
```

```bash
kubectl apply -f gateway-setup.yaml

# Gateway 상태 확인
kubectl get gateway main-gw -n gateway-infra
kubectl describe gateway main-gw -n gateway-infra

# HTTPRoute 확인
kubectl get httproutes -A
```

</details>

### 연습문제 4: 트래픽 분할을 사용한 카나리 배포

Gateway API HTTPRoute를 사용하여 트래픽의 95%가 `app-v1`으로, 5%가 `app-v2`로 가는 카나리 배포를 구현하세요. `X-Canary: true` 헤더가 있는 요청은 항상 `app-v2`로 가도록 헤더 기반 오버라이드를 포함하세요.

<details><summary>정답 보기</summary>

```yaml
# canary-httproute.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: canary-deployment
  namespace: production
spec:
  parentRefs:
    - name: main-gateway
      namespace: gateway-infra
  hostnames:
    - "app.example.com"
  rules:
    # 규칙 1: 헤더 기반 오버라이드 (최고 우선순위)
    - matches:
        - headers:
            - name: X-Canary
              value: "true"
      backendRefs:
        - name: app-v2
          port: 80
    # 규칙 2: 가중 트래픽 분할
    - matches:
        - path:
            type: PathPrefix
            value: /
      backendRefs:
        - name: app-v1
          port: 80
          weight: 95
        - name: app-v2
          port: 80
          weight: 5
```

```bash
kubectl apply -f canary-httproute.yaml

# 라우트 확인
kubectl describe httproute canary-deployment -n production

# 일반 트래픽 테스트 (95/5 분할)
for i in $(seq 1 100); do
  curl -s http://app.example.com/ | grep -o "v[12]"
done | sort | uniq -c
# 예상: ~95 v1, ~5 v2

# 헤더 오버라이드 테스트 (항상 v2)
curl -H "X-Canary: true" http://app.example.com/
# 항상 v2 응답 반환
```

</details>

### 연습문제 5: ReferenceGrant를 사용한 다중 네임스페이스 Gateway

`gateway-infra` 네임스페이스에 Gateway를 설정하세요. `team-a`와 `team-b` 네임스페이스 모두에 HTTPRoute를 생성하고, 각각 자체 서브도메인을 라우팅하세요. ReferenceGrant를 사용하여 `team-a` 라우트가 `shared-services` 네임스페이스의 공유 인증 서비스를 참조할 수 있도록 하세요.

<details><summary>정답 보기</summary>

```yaml
# multi-namespace-gateway.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: team-a
  labels:
    gateway-access: "true"
---
apiVersion: v1
kind: Namespace
metadata:
  name: team-b
  labels:
    gateway-access: "true"
---
apiVersion: v1
kind: Namespace
metadata:
  name: shared-services
---
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: shared-gateway
  namespace: gateway-infra
spec:
  gatewayClassName: nginx
  listeners:
    - name: https
      protocol: HTTPS
      port: 443
      tls:
        mode: Terminate
        certificateRefs:
          - name: wildcard-tls
      allowedRoutes:
        namespaces:
          from: Selector
          selector:
            matchLabels:
              gateway-access: "true"
---
# Team A 라우트
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: team-a-route
  namespace: team-a
spec:
  parentRefs:
    - name: shared-gateway
      namespace: gateway-infra
  hostnames:
    - "team-a.example.com"
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /auth
      backendRefs:
        - name: auth-service
          namespace: shared-services     # 크로스 네임스페이스 참조
          port: 8080
    - matches:
        - path:
            type: PathPrefix
            value: /
      backendRefs:
        - name: team-a-app
          port: 80
---
# Team B 라우트
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: team-b-route
  namespace: team-b
spec:
  parentRefs:
    - name: shared-gateway
      namespace: gateway-infra
  hostnames:
    - "team-b.example.com"
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /
      backendRefs:
        - name: team-b-app
          port: 80
---
# ReferenceGrant: team-a HTTPRoute가 shared-services의 서비스를 참조하도록 허용
apiVersion: gateway.networking.k8s.io/v1beta1
kind: ReferenceGrant
metadata:
  name: allow-team-a-auth
  namespace: shared-services
spec:
  from:
    - group: gateway.networking.k8s.io
      kind: HTTPRoute
      namespace: team-a
  to:
    - group: ""
      kind: Service
```

```bash
kubectl apply -f multi-namespace-gateway.yaml

# 라우트가 수락되었는지 확인
kubectl get httproutes -A
# NAMESPACE   NAME           HOSTNAMES              AGE
# team-a      team-a-route   ["team-a.example.com"] 10s
# team-b      team-b-route   ["team-b.example.com"] 10s

# ReferenceGrant 확인
kubectl get referencegrants -n shared-services

# 라우팅 테스트
curl https://team-a.example.com/
curl https://team-a.example.com/auth/login
curl https://team-b.example.com/
```

</details>

---

**이전**: [RBAC와 보안](./06_RBAC_and_Security.md) | **다음**: [CNI와 고급 네트워킹](./08_CNI_and_Advanced_Networking.md)
