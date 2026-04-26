# 07. Ingress and Gateway API

**Previous**: [RBAC and Security](./06_RBAC_and_Security.md) | **Next**: [CNI and Advanced Networking](./08_CNI_and_Advanced_Networking.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Configure Ingress resources with host-based and path-based routing rules
2. Deploy and compare Ingress controllers (NGINX, Traefik)
3. Set up TLS termination and automated certificate management with cert-manager
4. Design traffic routing using the Gateway API (Gateway, HTTPRoute, GRPCRoute)
5. Implement advanced patterns including rate limiting, authentication, and traffic splitting

---

Kubernetes Services expose applications within the cluster, but reaching them from the outside world requires an additional layer. Ingress resources and the newer Gateway API provide declarative, HTTP-aware routing that maps external URLs to internal services. This lesson covers both the mature Ingress API and the next-generation Gateway API, including TLS management, traffic splitting, and gateway-level authentication.

> **Ingress vs Gateway API:** Ingress has been stable since Kubernetes 1.19 but has well-known limitations -- vendor-specific annotations, no support for TCP/UDP routing, and a flat permission model. The Gateway API (GA since Kubernetes 1.27 for core resources) solves these issues with a role-oriented, expressive, and portable API. New projects should prefer Gateway API, but Ingress remains widely deployed.

Before the YAML, read [**Theory & Principles**](#theory--principles) — why an Ingress is "just an annotated config the controller compiles into nginx/envoy/whatever," the L4 vs L7 distinction that motivated the move from Service-LoadBalancer to Ingress, the role-oriented model that motivated Gateway API, and the cert-manager reconciliation loop that makes TLS automatic.

## Table of Contents

- [Theory & Principles](#theory--principles)
- [1. Ingress Fundamentals](#1-ingress-fundamentals)
  - [1.1 Ingress Resource Structure](#11-ingress-resource-structure)
  - [1.2 Host-Based Routing](#12-host-based-routing)
  - [1.3 Path-Based Routing](#13-path-based-routing)
  - [1.4 Default Backend](#14-default-backend)
- [2. Ingress Controllers](#2-ingress-controllers)
  - [2.1 NGINX Ingress Controller](#21-nginx-ingress-controller)
  - [2.2 Traefik Ingress Controller](#22-traefik-ingress-controller)
  - [2.3 Controller Comparison](#23-controller-comparison)
- [3. TLS Termination](#3-tls-termination)
  - [3.1 Manual TLS Configuration](#31-manual-tls-configuration)
  - [3.2 Automated Certificates with cert-manager](#32-automated-certificates-with-cert-manager)
- [4. Gateway API](#4-gateway-api)
  - [4.1 Architecture and Resource Model](#41-architecture-and-resource-model)
  - [4.2 GatewayClass and Gateway](#42-gatewayclass-and-gateway)
  - [4.3 HTTPRoute](#43-httproute)
  - [4.4 GRPCRoute](#44-grpcroute)
  - [4.5 TLS with Gateway API](#45-tls-with-gateway-api)
- [5. Gateway API vs Ingress](#5-gateway-api-vs-ingress)
- [6. Advanced Patterns](#6-advanced-patterns)
  - [6.1 Rate Limiting](#61-rate-limiting)
  - [6.2 Authentication at the Gateway](#62-authentication-at-the-gateway)
  - [6.3 Traffic Splitting](#63-traffic-splitting)
  - [6.4 URL Rewriting and Redirects](#64-url-rewriting-and-redirects)
- [Exercises](#exercises)

---

## Theory & Principles

External traffic into a Kubernetes cluster is the most heterogeneous part of the platform — there are at least four different objects that can put a public IP in front of your Pods (`Service: LoadBalancer`, `Service: NodePort`, `Ingress`, `Gateway`), each with different capabilities and ownership models. The reason this complexity exists is that L4 load balancing (TCP/UDP, what Service does) and L7 routing (HTTP host/path/header matching) are different problems, and the original Service object solved only the first. This section explains the L4-vs-L7 split, the controller-as-compiler pattern that makes Ingress work, the role-oriented redesign that became Gateway API, and how cert-manager closes the TLS loop.

### A. L4 vs L7: Why Service Is Not Enough

A `Service` of type `LoadBalancer` gives you an external L4 load balancer pointing at your Pods. That works perfectly for any TCP/UDP workload — Postgres, Kafka, gRPC streaming — because at L4 the load balancer just shuffles packets without inspecting them. But for HTTP, L4 is not enough:

- You typically want **one external IP for many services**, distinguished by host (`api.example.com` vs `www.example.com`) or path (`/api/*` vs `/`). A pure L4 LB cannot read the HTTP `Host` header.
- You want **TLS termination** in one place so individual services don't each need a cert. L4 cannot decrypt.
- You want **HTTP-aware features**: rewrites, redirects, header manipulation, rate limiting, request logging.

L4 cannot do any of these because it operates below HTTP. So Kubernetes added a higher layer: an **Ingress** is a declarative description of how external HTTP traffic should be routed to in-cluster Services, and an **Ingress controller** is the actual reverse proxy (nginx, Traefik, HAProxy, Istio, AWS ALB, ...) that implements the rules.

The mental model: `Service` is the *destination* abstraction (one stable VIP for a set of Pods), `Ingress` is the *routing* abstraction (which incoming HTTP request goes to which Service). They compose; you do not pick one or the other.

### B. The Ingress Controller as a Compiler

The Ingress object is just a typed config file:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
spec:
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /v1
            pathType: Prefix
            backend:
              service: { name: api-v1, port: { number: 8080 } }
```

The Ingress object on its own does *nothing*. There is no kubelet that reads it, no built-in proxy that implements it. What makes it work is an **Ingress controller** — a Pod (or DaemonSet) running in the cluster that:

1. Watches all Ingress objects.
2. Compiles them into the native config of its underlying proxy (nginx.conf, Traefik dynamic config, Envoy xDS).
3. Reloads or hot-swaps the proxy with the new config.
4. Watches Services and EndpointSlices to keep upstream pools current.

This is why your first Ingress did nothing until you installed an ingress-nginx Helm chart. It is also why every Ingress controller has slightly different behavior — they all "compile to" different proxies. Some HTTP features (rate limiting, custom auth) require vendor-specific annotations because the Ingress spec doesn't standardize them.

The controller-as-compiler pattern shows up everywhere in Kubernetes (cert-manager, Argo CD, the Deployment controller itself), but Ingress is one of the cleanest examples — *the data-plane behavior is entirely a function of the controller you chose, even though the config is portable*.

### C. Gateway API: Role-Oriented Redesign

Ingress works but has accumulated debt:

- **Vendor-specific annotations** for everything beyond basic routing — your Ingress YAML is portable in name only.
- **No L4 support** — you can't route TCP/UDP through Ingress, so non-HTTP traffic still uses Service: LoadBalancer.
- **Flat permission model** — anyone who can edit Ingresses in their namespace can claim arbitrary hostnames, hijacking traffic from other tenants.

Gateway API (GA since K8s 1.27 for core resources) redesigns the same problem with three role-separated objects:

- **GatewayClass** (cluster admin): "this controller implements Gateways of class `aws-alb`/`istio`/`envoy`."
- **Gateway** (infra/platform team): "I have a Gateway named `prod-gateway`, listening on port 443 with this TLS cert, attached to GatewayClass `aws-alb`."
- **HTTPRoute / GRPCRoute / TCPRoute / TLSRoute** (app team): "route requests for `api.example.com/v1/*` from the `prod-gateway` to my Service."

The split lets the platform team own infrastructure (which load balancer, which certs) and app teams own routing — without app teams being able to spin up new external IPs or steal hostnames. Cross-namespace references use **ReferenceGrants** so the producer of a hostname must explicitly allow consumers.

`HTTPRoute` standardizes header matching, weighted backends (10% to v2 for canary), redirects, rewrites, request mirroring — features that previously required vendor annotations. New projects should default to Gateway API; Ingress remains widely deployed and gets bug fixes but no new features.

### D. TLS Automation: cert-manager as a Reconciler

Manual TLS is operationally painful — certificates expire, renewal at 3am leaves you debugging at 6am. **cert-manager** is the standard solution and is itself a controller-as-compiler:

1. You create a `Certificate` CR: "I want a cert for `api.example.com`, valid for 90 days, issued by `letsencrypt-prod`, stored in Secret `api-tls`."
2. cert-manager creates a `CertificateRequest` and asks the named `Issuer` (an ACME, Vault, internal CA, etc.) to fulfill it.
3. ACME requires proof of domain control — cert-manager performs an HTTP-01 or DNS-01 challenge (creating temporary Ingress paths or DNS records) until the CA issues the cert.
4. cert-manager writes the cert + key into the target Secret. Ingress / Gateway picks it up automatically.
5. Before expiry (default 1/3 of lifetime remaining), cert-manager renews — same loop, no human in the path.

The key insight: cert-manager runs the same reconciliation loop pattern as every other controller. Desired state: "a non-expired cert exists in Secret X." Observed state: "current cert expires in N days." Action: "if N < threshold, request renewal." This is what makes TLS at scale tractable — it's just another controller.

### From Theory to the YAML Below

The lesson now applies these abstractions:

- **Section 1 (Ingress Fundamentals)** is §B with concrete Ingress objects, host and path matching.
- **Section 2 (Ingress Controllers)** shows the §B controller-as-compiler — installing nginx, Traefik, comparing what each compiles to.
- **Section 3 (TLS Termination)** is §D's manual baseline followed by cert-manager automation.
- **Section 4 (Gateway API)** is §C — `Gateway`, `HTTPRoute`, `GRPCRoute` introduced in role-separated form.
- **Section 5 (Gateway API vs Ingress)** is the migration guide between the two abstractions.
- **Section 6 (Advanced Patterns)** — rate limiting, gateway auth, traffic splitting — is what Ingress requires vendor annotations for and Gateway API standardizes.

Once you see Ingress/Gateway as "controllers compiling YAML into proxy config," every "why doesn't this annotation work in nginx?" reduces to "your controller doesn't speak that dialect, switch controller or switch to Gateway API."

---

## 1. Ingress Fundamentals

An Ingress resource defines rules for routing external HTTP(S) traffic to services within the cluster. It requires an Ingress controller to function.

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

### 1.1 Ingress Resource Structure

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
  namespace: production
  annotations:
    # Controller-specific annotations
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx          # Specifies which controller handles this
  defaultBackend:                  # Catch-all backend
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

**Path types:**

| PathType | Behavior | Example |
|----------|----------|---------|
| `Prefix` | Matches URL path prefix by path element | `/api` matches `/api`, `/api/`, `/api/v1` |
| `Exact` | Matches the URL path exactly | `/api` matches only `/api`, not `/api/` |
| `ImplementationSpecific` | Matching depends on IngressClass | Controller-defined behavior |

### 1.2 Host-Based Routing

Route traffic to different services based on the `Host` header.

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

### 1.3 Path-Based Routing

Route traffic to different services based on the URL path.

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

### 1.4 Default Backend

A default backend handles requests that do not match any rule. It is useful for custom 404 pages.

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

## 2. Ingress Controllers

An Ingress resource is just a configuration object. It does nothing by itself -- an Ingress controller watches for Ingress resources and configures the underlying load balancer.

### 2.1 NGINX Ingress Controller

The most widely deployed Ingress controller. It generates NGINX configuration from Ingress resources.

```bash
# Install NGINX Ingress Controller on minikube
minikube addons enable ingress

# Verify installation
kubectl get pods -n ingress-nginx
# NAME                                       READY   STATUS
# ingress-nginx-controller-xxxxx-xxxxx       1/1     Running

# Or install via Helm (production)
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.replicaCount=2 \
  --set controller.metrics.enabled=true
```

Common NGINX Ingress annotations:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nginx-features
  annotations:
    # Rate limiting
    nginx.ingress.kubernetes.io/limit-rps: "10"
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "5"

    # Timeouts
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "10"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"

    # Body size
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"

    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.example.com"

    # Redirect HTTP to HTTPS
    nginx.ingress.kubernetes.io/ssl-redirect: "true"

    # WebSocket support
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"

    # Custom NGINX configuration snippet
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

### 2.2 Traefik Ingress Controller

Traefik is a cloud-native reverse proxy with automatic service discovery and built-in Let's Encrypt support.

```bash
# Install Traefik via Helm
helm repo add traefik https://traefik.github.io/charts
helm install traefik traefik/traefik \
  --namespace traefik \
  --create-namespace \
  --set ports.web.port=8000 \
  --set ports.websecure.port=8443
```

```yaml
# Traefik uses its own IngressRoute CRD for advanced features
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
# Traefik middleware for rate limiting
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

### 2.3 Controller Comparison

| Feature | NGINX Ingress | Traefik | Envoy (Contour) |
|---------|--------------|---------|------------------|
| Configuration | Annotations | CRD + annotations | CRD (HTTPProxy) |
| Hot reload | Yes (lua) | Yes (native) | Yes (xDS) |
| Let's Encrypt | Via cert-manager | Built-in | Via cert-manager |
| TCP/UDP | ConfigMap | Native CRD | Native CRD |
| Dashboard | No (use Prometheus) | Built-in | No |
| Gateway API | Yes (v1.0+) | Yes (v3.0+) | Yes (v1.28+) |
| Market share | Highest | Second | Growing |

---

## 3. TLS Termination

TLS termination at the Ingress or Gateway level offloads encryption from application services.

### 3.1 Manual TLS Configuration

```bash
# Create a self-signed certificate (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=app.example.com"

# Create a Kubernetes TLS secret
kubectl create secret tls app-tls-cert \
  --cert=tls.crt \
  --key=tls.key \
  --namespace=production
```

```yaml
# Ingress with TLS
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

### 3.2 Automated Certificates with cert-manager

cert-manager automates the provisioning and renewal of TLS certificates from Let's Encrypt and other CAs.

```bash
# Install cert-manager
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set crds.enabled=true
```

```yaml
# ClusterIssuer for Let's Encrypt production
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
# ClusterIssuer for Let's Encrypt staging (for testing)
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
# Ingress with automatic certificate provisioning
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
      secretName: app-tls-auto   # cert-manager creates this secret
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
# Monitor certificate status
kubectl get certificates -n production
# NAME           READY   SECRET         AGE
# app-tls-auto   True    app-tls-auto   5m

kubectl describe certificate app-tls-auto -n production

# Check certificate orders and challenges
kubectl get orders -n production
kubectl get challenges -n production
```

---

## 4. Gateway API

The Gateway API is the next-generation API for managing ingress traffic in Kubernetes. It provides a more expressive, role-oriented, and portable alternative to Ingress.

### 4.1 Architecture and Resource Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Role-Oriented Design                      │
│                                                              │
│  Infrastructure Provider    Cluster Operator    App Developer│
│  ┌──────────────┐          ┌──────────┐       ┌───────────┐ │
│  │ GatewayClass │──────────│ Gateway  │───────│ HTTPRoute │ │
│  │              │          │          │       │ GRPCRoute │ │
│  │ "Which       │          │ "Where   │       │ TCPRoute  │ │
│  │  controller" │          │  to      │       │ TLSRoute  │ │
│  │              │          │  listen" │       │           │ │
│  └──────────────┘          └──────────┘       └───────────┘ │
└─────────────────────────────────────────────────────────────┘
```

```bash
# Install Gateway API CRDs
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml

# Verify CRDs
kubectl get crd | grep gateway
# gatewayclasses.gateway.networking.k8s.io
# gateways.gateway.networking.k8s.io
# httproutes.gateway.networking.k8s.io
# grpcroutes.gateway.networking.k8s.io
# referencegrants.gateway.networking.k8s.io
```

### 4.2 GatewayClass and Gateway

```yaml
# GatewayClass: defines which controller implementation to use
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: nginx
spec:
  controllerName: gateway.nginx.org/nginx-gateway-controller
```

```yaml
# Gateway: creates the actual load balancer with listeners
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
          from: All           # Accept routes from any namespace
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
              gateway-access: "true"   # Only labeled namespaces
    - name: api
      protocol: HTTPS
      port: 443
      hostname: "api.example.com"     # Listener-level hostname filter
      tls:
        mode: Terminate
        certificateRefs:
          - name: api-tls
            kind: Secret
      allowedRoutes:
        namespaces:
          from: Same           # Only routes in this namespace
```

### 4.3 HTTPRoute

HTTPRoute is the core routing resource for HTTP traffic.

```yaml
# HTTPRoute: host-based and path-based routing
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: app-routes
  namespace: production
spec:
  parentRefs:
    - name: main-gateway
      namespace: gateway-infra
      sectionName: https          # Attach to the https listener
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
# HTTPRoute: header-based routing
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
    # Default route (no header match)
    - backendRefs:
        - name: api-v1-service
          port: 8080
```

### 4.4 GRPCRoute

```yaml
# GRPCRoute for gRPC services
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

### 4.5 TLS with Gateway API

```yaml
# Gateway with TLS passthrough (terminate at the backend)
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
        mode: Passthrough       # Do not terminate TLS
      allowedRoutes:
        namespaces:
          from: All
```

```yaml
# cert-manager integration with Gateway API
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

| Feature | Ingress | Gateway API |
|---------|---------|-------------|
| API maturity | Stable (v1, 1.19+) | Stable core (v1, 1.27+) |
| Routing expressiveness | Basic (host, path) | Rich (headers, query, method) |
| Protocol support | HTTP(S) only | HTTP, HTTPS, gRPC, TCP, TLS |
| Configuration | Annotations (not portable) | Native spec fields (portable) |
| Role separation | None (one resource) | GatewayClass / Gateway / Route |
| Cross-namespace | Not supported | ReferenceGrant |
| Traffic splitting | Annotations (controller-specific) | Native `backendRefs` with weights |
| Header modification | Annotations | Native `RequestHeaderModifier` |
| Redirect / Rewrite | Annotations | Native `RequestRedirect` / `URLRewrite` |

**Migration strategy:** You can run both Ingress and Gateway API simultaneously. Start by deploying Gateway API alongside existing Ingress resources, then gradually migrate routes.

---

## 6. Advanced Patterns

### 6.1 Rate Limiting

```yaml
# NGINX Ingress: annotation-based rate limiting
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
# Gateway API: rate limiting via policy attachment (implementation-specific)
# Example using NGINX Gateway Fabric
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

### 6.2 Authentication at the Gateway

```yaml
# NGINX Ingress: external authentication
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: auth-ingress
  annotations:
    # External auth service (OAuth2 Proxy, Authelia, etc.)
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
# Basic auth with NGINX Ingress
# First create the htpasswd secret
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

### 6.3 Traffic Splitting

Traffic splitting enables canary deployments, A/B testing, and gradual rollouts.

```yaml
# Gateway API: weighted traffic splitting (canary deployment)
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
          weight: 90            # 90% to stable
        - name: app-canary
          port: 80
          weight: 10            # 10% to canary
```

```yaml
# NGINX Ingress: canary deployment with annotations
# Main Ingress (receives all traffic by default)
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
# Canary Ingress (receives a percentage of traffic)
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
# Header-based canary (route specific users to canary)
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

### 6.4 URL Rewriting and Redirects

```yaml
# Gateway API: URL rewrite
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
# Gateway API: HTTP to HTTPS redirect
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: redirect-to-https
  namespace: production
spec:
  parentRefs:
    - name: main-gateway
      namespace: gateway-infra
      sectionName: http          # Attach to HTTP listener
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
# Gateway API: cross-namespace routing with ReferenceGrant
# Allow routes in "production" namespace to reference services in "backend" namespace
apiVersion: gateway.networking.k8s.io/v1beta1
kind: ReferenceGrant
metadata:
  name: allow-production-routes
  namespace: backend               # The namespace being referenced
spec:
  from:
    - group: gateway.networking.k8s.io
      kind: HTTPRoute
      namespace: production        # The namespace making the reference
  to:
    - group: ""
      kind: Service
```

---

## Exercises

### Exercise 1: Multi-Service Ingress

Create an Ingress resource that routes traffic to three services based on path:
- `/` goes to `frontend` service (port 80)
- `/api` goes to `api-server` service (port 8080)
- `/ws` goes to `websocket` service (port 9090) with WebSocket support

All services are in the `webapp` namespace.

<details><summary>Show Answer</summary>

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

# On minikube, get the Ingress IP
minikube ip
# Add to /etc/hosts: <minikube-ip> webapp.local

# Test each path
curl http://webapp.local/
curl http://webapp.local/api/health
# WebSocket test with wscat
# wscat -c ws://webapp.local/ws
```

</details>

### Exercise 2: TLS with cert-manager

Set up cert-manager with a self-signed ClusterIssuer (for minikube), then create an Ingress with automatic TLS certificate provisioning for `secure.local`.

<details><summary>Show Answer</summary>

```bash
# Install cert-manager
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set crds.enabled=true

# Wait for cert-manager pods
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

# Verify certificate was issued
kubectl get certificate -n webapp
# NAME               READY   SECRET             AGE
# secure-local-tls   True    secure-local-tls   30s

kubectl describe certificate secure-local-tls -n webapp

# Test HTTPS (self-signed, so use -k)
curl -k https://secure.local/
```

</details>

### Exercise 3: Gateway API HTTPRoute

Using Gateway API, create a Gateway and HTTPRoute that:
- Listens on port 80 (HTTP) and port 443 (HTTPS)
- Routes `app.example.com` to `app-service:80`
- Routes `api.example.com/v1/*` to `api-v1:8080` and `api.example.com/v2/*` to `api-v2:8080`

<details><summary>Show Answer</summary>

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
# API HTTPRoute with versioned paths
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

# Verify Gateway status
kubectl get gateway main-gw -n gateway-infra
kubectl describe gateway main-gw -n gateway-infra

# Verify HTTPRoutes
kubectl get httproutes -A
```

</details>

### Exercise 4: Canary Deployment with Traffic Splitting

Implement a canary deployment where 95% of traffic goes to `app-v1` and 5% goes to `app-v2` using Gateway API HTTPRoute. Include a header-based override so requests with `X-Canary: true` always go to `app-v2`.

<details><summary>Show Answer</summary>

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
    # Rule 1: Header-based override (highest priority)
    - matches:
        - headers:
            - name: X-Canary
              value: "true"
      backendRefs:
        - name: app-v2
          port: 80
    # Rule 2: Weighted traffic splitting
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

# Verify the route
kubectl describe httproute canary-deployment -n production

# Test normal traffic (95/5 split)
for i in $(seq 1 100); do
  curl -s http://app.example.com/ | grep -o "v[12]"
done | sort | uniq -c
# Expected: ~95 v1, ~5 v2

# Test header override (always v2)
curl -H "X-Canary: true" http://app.example.com/
# Always returns v2 response
```

</details>

### Exercise 5: Multi-Namespace Gateway with ReferenceGrant

Set up a Gateway in the `gateway-infra` namespace. Create HTTPRoutes in both `team-a` and `team-b` namespaces, each routing their own subdomain. Use ReferenceGrant to allow `team-a` routes to reference a shared authentication service in the `shared-services` namespace.

<details><summary>Show Answer</summary>

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
# Team A route
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
          namespace: shared-services     # Cross-namespace reference
          port: 8080
    - matches:
        - path:
            type: PathPrefix
            value: /
      backendRefs:
        - name: team-a-app
          port: 80
---
# Team B route
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
# ReferenceGrant: allow team-a HTTPRoutes to reference services in shared-services
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

# Verify routes are accepted
kubectl get httproutes -A
# NAMESPACE   NAME           HOSTNAMES              AGE
# team-a      team-a-route   ["team-a.example.com"] 10s
# team-b      team-b-route   ["team-b.example.com"] 10s

# Verify ReferenceGrant
kubectl get referencegrants -n shared-services

# Test routing
curl https://team-a.example.com/
curl https://team-a.example.com/auth/login
curl https://team-b.example.com/
```

</details>

---

**Previous**: [RBAC and Security](./06_RBAC_and_Security.md) | **Next**: [CNI and Advanced Networking](./08_CNI_and_Advanced_Networking.md)
