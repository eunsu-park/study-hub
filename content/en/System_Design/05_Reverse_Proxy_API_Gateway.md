# Reverse Proxy and API Gateway

**Previous**: [Load Balancing](./04_Load_Balancing.md) | **Next**: [Caching Strategies](./06_Caching_Strategies.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a reverse proxy is and how it differs from a forward proxy
2. Describe core reverse proxy features including SSL termination, compression, and response caching
3. Define the API Gateway pattern and explain how it simplifies client-to-microservice communication
4. Compare rate limiting algorithms (token bucket, leaky bucket, fixed window, sliding window) and choose one for a given use case
5. Design a reverse proxy configuration that provides SSL offloading, caching, and request routing
6. Evaluate when to use a standalone reverse proxy versus a full API gateway solution

**Difficulty**: ⭐⭐⭐
**Estimated Learning Time**: 2-3 hours
**Prerequisites**: [04_Load_Balancing.md](./04_Load_Balancing.md)

---

In a microservices world, clients should never talk directly to dozens of backend services -- it would be like asking hotel guests to find the kitchen, laundry, and maintenance rooms themselves. A reverse proxy and API gateway serve as the single front door that handles security, routing, and cross-cutting concerns so that backend teams can focus on business logic. Mastering these patterns is essential for designing clean, secure, and performant architectures.

## Table of Contents

1. [What is a Reverse Proxy?](#1-what-is-a-reverse-proxy)
2. [Core Reverse Proxy Features](#2-core-reverse-proxy-features)
3. [API Gateway Pattern](#3-api-gateway-pattern)
4. [Rate Limiting](#4-rate-limiting)
5. [Practice Problems](#5-practice-problems)
6. [References](#6-references)

---

## 1. What is a Reverse Proxy?

### 1.1 Forward Proxy vs Reverse Proxy

```
┌─────────────────────────────────────────────────────────────────┐
│              Forward Proxy vs Reverse Proxy                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Forward Proxy                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  "Makes requests to servers on behalf of clients"         │ │
│  │                                                            │ │
│  │  ┌──────┐    ┌──────────┐    ┌──────────┐                  │ │
│  │  │Client│───▶│ Forward  │───▶│  Server  │                  │ │
│  │  │      │    │ Proxy    │    │  (Web)   │                  │ │
│  │  └──────┘    └──────────┘    └──────────┘                  │ │
│  │                                                            │ │
│  │  Use cases: Anonymity, access control, caching            │ │
│  │  Examples: Corporate firewall, VPN, Squid                 │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Reverse Proxy                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  "Handles client requests on behalf of servers"           │ │
│  │                                                            │ │
│  │  ┌──────┐    ┌──────────┐    ┌──────────┐                  │ │
│  │  │Client│───▶│ Reverse  │───▶│ Backend  │                  │ │
│  │  │      │    │ Proxy    │    │ Servers  │                  │ │
│  │  └──────┘    └──────────┘    └──────────┘                  │ │
│  │                                                            │ │
│  │  Clients don't know the actual servers!                   │ │
│  │                                                            │ │
│  │  Use cases: Load balancing, SSL termination, caching, security│ │
│  │  Examples: Nginx, HAProxy, AWS ALB                        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Reverse Proxy Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Reverse Proxy Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                         Internet                                 │
│                            │                                     │
│                            ▼                                     │
│                    ┌──────────────┐                              │
│                    │   Firewall   │                              │
│                    └──────┬───────┘                              │
│                           │                                      │
│                           ▼                                      │
│                    ┌──────────────┐                              │
│                    │   Reverse    │ ◀── SSL termination, caching│
│                    │   Proxy      │     compression, security    │
│                    │   (Nginx)    │                              │
│                    └──────┬───────┘                              │
│                           │                                      │
│              ┌────────────┼────────────┐                         │
│              │            │            │                         │
│              ▼            ▼            ▼                         │
│         ┌────────┐   ┌────────┐   ┌────────┐                    │
│         │ App    │   │ App    │   │ App    │                    │
│         │ Server │   │ Server │   │ Server │                    │
│         │   1    │   │   2    │   │   3    │                    │
│         └────────┘   └────────┘   └────────┘                    │
│                                                                  │
│  Clients → Only know proxy IP, not actual server IPs            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Reverse Proxy Features

### 2.1 SSL/TLS Termination

```
┌─────────────────────────────────────────────────────────────────┐
│                     SSL Termination                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Without SSL Termination (End-to-End Encryption):               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Client ═══HTTPS═══▶ Proxy ═══HTTPS═══▶ Server            │ │
│  │                                                            │ │
│  │  • Certificates needed on all servers                     │ │
│  │  • Increased server load (encryption/decryption)          │ │
│  │  • Complex certificate management                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  With SSL Termination (Proxy handles SSL):                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Client ═══HTTPS═══▶ Proxy ───HTTP───▶ Server             │ │
│  │                        │                                   │ │
│  │                   SSL Termination                          │ │
│  │                                                            │ │
│  │  • Certificate only on proxy                              │ │
│  │  • Reduced server load                                    │ │
│  │  • Centralized certificate management                     │ │
│  │                                                            │ │
│  │  Nginx configuration:                                      │ │
│  │  server {                                                  │ │
│  │      listen 443 ssl;                                       │ │
│  │      ssl_certificate /path/to/cert.pem;                    │ │
│  │      ssl_certificate_key /path/to/key.pem;                 │ │
│  │                                                            │ │
│  │      location / {                                          │ │
│  │          proxy_pass http://backend;  # HTTP to backend    │ │
│  │      }                                                     │ │
│  │  }                                                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Note: If internal network is untrusted, consider SSL Passthrough│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Response Compression

```
┌─────────────────────────────────────────────────────────────────┐
│                      Response Compression                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Save network bandwidth, improve response speed"               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Before compression: 100KB HTML                            │ │
│  │  After compression: 20KB (gzip) → 80% saved!              │ │
│  │                                                            │ │
│  │  Client                Proxy              Server           │ │
│  │    │                     │                  │              │ │
│  │    │──Request ──────────▶│                  │              │ │
│  │    │  Accept-Encoding:   │──────────────────▶              │ │
│  │    │  gzip, deflate      │                  │              │ │
│  │    │                     │◀─────────────────│              │ │
│  │    │                     │  Original response│             │ │
│  │    │                     │  (100KB)         │              │ │
│  │    │                     │                  │              │ │
│  │    │◀── Compressed ──────│                  │              │ │
│  │    │  Content-Encoding:  │  (Proxy does    │              │ │
│  │    │  gzip               │   gzip compression)             │ │
│  │    │  (20KB)             │                  │              │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Nginx configuration:                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  gzip on;                                                  │ │
│  │  gzip_types text/plain text/css application/json          │ │
│  │             application/javascript text/xml;               │ │
│  │  gzip_min_length 1000;  # Compress only >= 1KB            │ │
│  │  gzip_comp_level 6;     # Compression level (1-9)         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Compression algorithm comparison:                               │
│  • gzip: Widely supported, good compression ratio              │
│  • Brotli: Better compression, modern browser support          │
│  • zstd: Fast compression/decompression, latest                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Caching

```
┌─────────────────────────────────────────────────────────────────┐
│                    Proxy Caching                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Return cached responses for repeated requests"                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  First request (Cache Miss):                              │ │
│  │                                                            │ │
│  │  Client ──▶ Proxy ──▶ Server                               │ │
│  │         ◀────────◀────  (Store response)                   │ │
│  │                  ▼                                         │ │
│  │              [Cache]                                       │ │
│  │                                                            │ │
│  │  Second request (Cache Hit):                              │ │
│  │                                                            │ │
│  │  Client ──▶ Proxy                                          │ │
│  │         ◀────  (Serve from cache, no server hit!)         │ │
│  │              ▲                                             │ │
│  │          [Cache]                                           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Nginx caching configuration:                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  # Define cache path                                       │ │
│  │  proxy_cache_path /var/cache/nginx levels=1:2              │ │
│  │                   keys_zone=my_cache:10m                   │ │
│  │                   max_size=10g inactive=60m;               │ │
│  │                                                            │ │
│  │  server {                                                  │ │
│  │      location / {                                          │ │
│  │          proxy_cache my_cache;                             │ │
│  │          proxy_cache_valid 200 60m;  # Cache 200 for 60min│ │
│  │          proxy_cache_valid 404 1m;   # Cache 404 for 1min │ │
│  │          proxy_cache_use_stale error timeout;              │ │
│  │          add_header X-Cache-Status $upstream_cache_status; │ │
│  │          proxy_pass http://backend;                        │ │
│  │      }                                                     │ │
│  │  }                                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Cache status headers:                                           │
│  • X-Cache-Status: HIT (cache hit)                              │
│  • X-Cache-Status: MISS (cache miss)                            │
│  • X-Cache-Status: EXPIRED (expired)                            │
│  • X-Cache-Status: STALE (using stale cache)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Security Features

```
┌─────────────────────────────────────────────────────────────────┐
│                   Reverse Proxy Security Features                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. IP-based Access Control                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  location /admin {                                         │ │
│  │      allow 10.0.0.0/8;       # Allow internal network     │ │
│  │      allow 192.168.1.100;    # Allow specific IP          │ │
│  │      deny all;               # Deny the rest              │ │
│  │      proxy_pass http://backend;                            │ │
│  │  }                                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Header Security                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  # Hide sensitive headers                                  │ │
│  │  proxy_hide_header X-Powered-By;                           │ │
│  │  proxy_hide_header Server;                                 │ │
│  │                                                            │ │
│  │  # Add security headers                                    │ │
│  │  add_header X-Frame-Options "SAMEORIGIN";                  │ │
│  │  add_header X-Content-Type-Options "nosniff";              │ │
│  │  add_header X-XSS-Protection "1; mode=block";              │ │
│  │  add_header Strict-Transport-Security "max-age=31536000";  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Request Size Limits                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  client_max_body_size 10m;    # Max upload size           │ │
│  │  client_body_timeout 60s;     # Request body timeout      │ │
│  │  client_header_timeout 60s;   # Header timeout            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  4. Hide Backend Servers                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Clients see: proxy.example.com                           │ │
│  │  Actual backend: 10.0.1.5:8080 (not exposed)              │ │
│  │                                                            │ │
│  │  → Prevents direct attacks                                │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. API Gateway Pattern

### 3.1 What is an API Gateway?

```
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Single Entry Point for microservices"                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Without API Gateway:                                      │ │
│  │                                                            │ │
│  │  Client ──▶ User Service (users.example.com)               │ │
│  │         ──▶ Order Service (orders.example.com)             │ │
│  │         ──▶ Product Service (products.example.com)         │ │
│  │                                                            │ │
│  │  Problems:                                                 │ │
│  │  • Client must know multiple service URLs                 │ │
│  │  • Duplicated authentication logic in each service        │ │
│  │  • Tight coupling between client and services             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  With API Gateway:                                         │ │
│  │                                                            │ │
│  │            ┌───────────────────────────────┐               │ │
│  │            │         API Gateway           │               │ │
│  │  Client ──▶│  api.example.com              │               │ │
│  │            │  ┌─────────────────────────┐  │               │ │
│  │            │  │ • Authentication/       │  │               │ │
│  │            │  │   Authorization         │  │               │ │
│  │            │  │ • Routing               │  │               │ │
│  │            │  │ • Rate Limiting         │  │               │ │
│  │            │  │ • Request/Response      │  │               │ │
│  │            │  │   Transformation        │  │               │ │
│  │            │  │ • Logging/Monitoring    │  │               │ │
│  │            │  └─────────────────────────┘  │               │ │
│  │            └───────────────┬───────────────┘               │ │
│  │                            │                               │ │
│  │              ┌─────────────┼─────────────┐                 │ │
│  │              ▼             ▼             ▼                 │ │
│  │         User Service  Order Service  Product Service      │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Core API Gateway Features

```
┌─────────────────────────────────────────────────────────────────┐
│                 API Gateway Core Features                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Request Routing                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  /api/users/*    ──────────▶ User Service                  │ │
│  │  /api/orders/*   ──────────▶ Order Service                 │ │
│  │  /api/products/* ──────────▶ Product Service               │ │
│  │  /api/v2/*       ──────────▶ New Service (version control) │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Authentication/Authorization                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Client ──▶ Gateway                                        │ │
│  │              │                                             │ │
│  │              ├─ Verify JWT token                           │ │
│  │              ├─ Check API Key                              │ │
│  │              ├─ Handle OAuth2                              │ │
│  │              └─ Verify permissions then ──▶ Backend Service│ │
│  │                                                            │ │
│  │  Advantage: Remove authentication logic from backend      │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Request/Response Transformation                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Protocol transformation:                                  │ │
│  │  REST (external) ←─────────▶ gRPC (internal)               │ │
│  │                                                            │ │
│  │  Data format transformation:                               │ │
│  │  JSON ←─────────▶ Protobuf                                 │ │
│  │                                                            │ │
│  │  Response aggregation:                                     │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │  GET /api/dashboard                                │   │ │
│  │  │       │                                            │   │ │
│  │  │       ├──▶ User Service (user info)                │   │ │
│  │  │       ├──▶ Order Service (recent orders)           │   │ │
│  │  │       └──▶ Stats Service (statistics)              │   │ │
│  │  │                                                    │   │ │
│  │  │       ◀── Combine all responses and return ──     │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  4. Logging and Monitoring                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  • Log all API calls                                       │ │
│  │  • Measure response times                                  │ │
│  │  • Monitor error rates                                     │ │
│  │  • Distributed tracing (propagate Trace IDs)               │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 API Gateway Product Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                API Gateway Product Comparison                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Kong                                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Built on Nginx + Lua                                     │ │
│  │ • Rich plugin ecosystem                                    │ │
│  │ • Open source / Enterprise versions                        │ │
│  │ • High performance                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  AWS API Gateway                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Fully managed                                            │ │
│  │ • Lambda, DynamoDB integration                             │ │
│  │ • WebSocket support                                        │ │
│  │ • Pay-per-use pricing                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Envoy (Istio)                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Cloud native                                             │ │
│  │ • Service mesh sidecar                                     │ │
│  │ • Native gRPC support                                      │ │
│  │ • Advanced traffic management                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Nginx Plus / Nginx                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Lightweight, high performance                            │ │
│  │ • Configuration-based (no code needed)                     │ │
│  │ • Wide range of use cases                                  │ │
│  │ • Plus: Commercial features (health checks, monitoring)    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Selection criteria:                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Cloud environment: AWS API Gateway, GCP Cloud Endpoints  │ │
│  │ • On-premises: Kong, Nginx                                 │ │
│  │ • Kubernetes: Envoy, Istio, NGINX Ingress                  │ │
│  │ • Simple requirements: Nginx                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Rate Limiting

### 4.1 What is Rate Limiting?

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rate Limiting                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Limit number of requests per time unit"                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Why is it needed?                                         │ │
│  │                                                            │ │
│  │  • DDoS protection                                         │ │
│  │  • Service stability protection                            │ │
│  │  • Fair resource distribution                              │ │
│  │  • Cost control                                            │ │
│  │  • Prevent API abuse                                       │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Limiting criteria:                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  • By IP address: 100 req/min per IP                       │ │
│  │  • By user: 1000 req/hour per user                         │ │
│  │  • By API key: 10000 req/day per API key                   │ │
│  │  • By endpoint: /login is 5 req/min                        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Response headers:                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  X-RateLimit-Limit: 100        # Limit                     │ │
│  │  X-RateLimit-Remaining: 45     # Remaining requests        │ │
│  │  X-RateLimit-Reset: 1640000000 # Reset time (Unix)         │ │
│  │                                                            │ │
│  │  When limit exceeded: 429 Too Many Requests                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Token Bucket

```
┌─────────────────────────────────────────────────────────────────┐
│                    Token Bucket Algorithm                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Allow requests if bucket has tokens, deny otherwise"          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │           ┌──────────┐                                     │ │
│  │           │  Token   │  ◀── Add tokens at constant rate   │ │
│  │           │  Bucket  │      (e.g., 10/sec)                │ │
│  │           │  ●●●●●   │                                     │ │
│  │           │  ●●●     │  ◀── Bucket capacity (e.g., 100)   │ │
│  │           └────┬─────┘                                     │ │
│  │                │                                           │ │
│  │                ▼                                           │ │
│  │  Request ──▶ Has token? ──Yes──▶ Allow (consume token)    │ │
│  │                │                                           │ │
│  │               No                                           │ │
│  │                │                                           │ │
│  │                ▼                                           │ │
│  │              Deny (429)                                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Characteristics:                                                │
│  • Allows bursts (if tokens accumulated in bucket)              │
│  • Average rate limiting                                         │
│  • Used by AWS, Stripe, etc.                                     │
│                                                                  │
│  Example:                                                        │
│  Bucket size: 100, Refill rate: 10/sec                          │
│  → Can handle burst of 100 requests                              │
│  → Then 10 requests per second                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Leaky Bucket

```
┌─────────────────────────────────────────────────────────────────┐
│                    Leaky Bucket Algorithm                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Process requests at constant rate only (no bursts)"           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Requests ──▶  ┌──────────┐                                │ │
│  │                │  Queue   │  ◀── Store requests in queue   │ │
│  │                │  ○○○○○   │                                │ │
│  │                │  ○○○     │  ◀── Queue size limit          │ │
│  │                └────┬─────┘                                │ │
│  │                     │                                      │ │
│  │              ●      │                                      │ │
│  │              ●  ◀───┘  "Leak" at constant rate            │ │
│  │              ●         (e.g., 10/sec)                      │ │
│  │              ▼                                             │ │
│  │           Processed                                        │ │
│  │                                                            │ │
│  │  When queue is full:                                       │ │
│  │  New request ──▶ Deny (429)                                │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Characteristics:                                                │
│  • Constant output rate (traffic shaping)                       │
│  • No bursts allowed                                             │
│  • Suitable for network traffic control                         │
│                                                                  │
│  Token Bucket vs Leaky Bucket:                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Token Bucket: Allows bursts, average rate limiting        │ │
│  │  Leaky Bucket: No bursts, constant rate output             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Fixed Window

```
┌─────────────────────────────────────────────────────────────────┐
│                  Fixed Window Algorithm                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Reset counter every time window"                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  1-minute window, limit 100:                               │ │
│  │                                                            │ │
│  │  00:00 ~ 00:59                 01:00 ~ 01:59              │ │
│  │  ┌────────────────────┐       ┌────────────────────┐      │ │
│  │  │ Count: 0 → 100     │       │ Count: 0 (reset)  │      │ │
│  │  │ ●●●●●●●●●●●●●●●●●  │       │ ●●●●●●●●●         │      │ │
│  │  └────────────────────┘       └────────────────────┘      │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Problem (boundary issue):                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  00:00          00:30          01:00         01:30        │ │
│  │    │              │              │              │          │ │
│  │    └──────────────┴──────────────┴──────────────┘          │ │
│  │           │               │                                │ │
│  │           ▼               ▼                                │ │
│  │     00:30~00:59     01:00~01:29                            │ │
│  │     100 requests    100 requests                           │ │
│  │                                                            │ │
│  │  → 200 requests possible within 1 minute!                 │ │
│  │     (00:30 ~ 01:30)                                        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros: Simple implementation, memory efficient                  │
│  Cons: Burst possible at window boundaries                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 Sliding Window

```
┌─────────────────────────────────────────────────────────────────┐
│                 Sliding Window Algorithm                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Window moves with time (solves boundary issue)"               │
│                                                                  │
│  Sliding Window Log:                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Store timestamp of each request                          │ │
│  │  [00:30:15, 00:30:20, 00:30:45, 00:31:00, ...]             │ │
│  │                                                            │ │
│  │  Current time: 01:30:00                                    │ │
│  │  Window: 00:30:00 ~ 01:30:00 (1 minute)                    │ │
│  │                                                            │ │
│  │  Count requests within window                              │ │
│  │                                                            │ │
│  │  Cons: High memory usage (store all timestamps)           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Sliding Window Counter (optimized):                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Weighted average of previous and current window          │ │
│  │                                                            │ │
│  │  Current: 01:15 (1-minute window)                          │ │
│  │                                                            │ │
│  │  ├── Previous window (00:00~01:00): 80 requests ──┤        │ │
│  │  ├── Current window (01:00~02:00): 40 requests ──┤         │ │
│  │                                                            │ │
│  │  01:15 is 25% into current window                         │ │
│  │  Weights: previous 75%, current 25%                        │ │
│  │                                                            │ │
│  │  Estimated requests = 80 * 0.75 + 40 * 0.25 = 70          │ │
│  │                                                            │ │
│  │  Pros: Memory efficient (only 2 counters)                 │ │
│  │        Mitigates boundary issue                            │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.6 Algorithm Comparison

| Algorithm | Burst | Memory | Accuracy | Use Case |
|----------|--------|--------|--------|----------|
| Token Bucket | Allow | Small | High | API limiting |
| Leaky Bucket | No | Small | High | Traffic shaping |
| Fixed Window | Boundary burst | Small | Low | Simple implementation |
| Sliding Window | Allow | Large/Medium | High | Accurate limiting |

---

## 5. Practice Problems

### Problem 1: Reverse Proxy Design

Write an Nginx configuration that satisfies the following requirements:

- HTTPS reception (port 443)
- HTTP → HTTPS redirect
- Backend: http://localhost:8080
- Gzip compression enabled
- Static file caching (1 day)

### Problem 2: API Gateway Design

Design an API Gateway for a microservices environment.

Services:
- User Service: /api/users/*
- Order Service: /api/orders/*
- Auth Service: /api/auth/*

Requirements:
- JWT authentication (except Auth)
- Rate Limiting: 1000 req/min per user
- Response logging

### Problem 3: Rate Limiting Selection

Choose an appropriate Rate Limiting algorithm for the following scenarios:

a) Public API (allow bursts, average limiting)
b) Real-time streaming service
c) Login endpoint (prevent brute force)
d) Simple requirements, quick implementation

### Problem 4: Rate Limiting Implementation

Implement the Token Bucket algorithm in pseudocode.

Conditions:
- Bucket size: 100
- Refill rate: 10 tokens/sec

---

## Answers

### Problem 1 Answer

```nginx
server {
    listen 80;
    server_name example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json
               application/javascript text/xml;
    gzip_min_length 1000;

    # Static file caching
    location /static/ {
        expires 1d;
        add_header Cache-Control "public, immutable";
    }

    # Backend proxy
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Problem 2 Answer

```
API Gateway Design:

┌─────────────────────────────────────────────────┐
│                  API Gateway                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. Receive request                             │
│       │                                         │
│  2. Rate Limiting check (Redis-based)           │
│       │ key: user_id, limit: 1000/min           │
│       │                                         │
│  3. Route check                                 │
│       ├── /api/auth/* → Auth Service (skip auth)│
│       ├── /api/users/* → Verify JWT → User Service│
│       └── /api/orders/*→ Verify JWT → Order Service│
│                                                 │
│  4. Response logging (ELK Stack)                │
│       - timestamp, user_id, path, status,       │
│         response_time                           │
│                                                 │
└─────────────────────────────────────────────────┘

Tech stack:
- Kong Gateway + JWT Plugin + Rate Limiting Plugin
- Or AWS API Gateway + Lambda Authorizer
```

### Problem 3 Answer

```
a) Public API: Token Bucket
   - Allows bursts for better user experience
   - Average rate limiting prevents abuse

b) Real-time streaming: Leaky Bucket
   - Constant output rate maintains quality
   - No bursts for stable transmission

c) Login: Fixed Window or Sliding Window
   - Strict limiting (e.g., 5 times/min)
   - Simplicity over boundary issues

d) Simple requirements: Fixed Window
   - Simplest implementation
   - Only needs counter + timestamp
```

### Problem 4 Answer

```python
class TokenBucket:
    def __init__(self, capacity=100, refill_rate=10):
        self.capacity = capacity        # Max bucket size
        self.refill_rate = refill_rate  # Tokens added per second
        self.tokens = capacity          # Current token count
        self.last_refill = current_time()

    def allow_request(self):
        self.refill()

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        else:
            return False

    def refill(self):
        now = current_time()
        elapsed = now - self.last_refill

        # Add tokens proportional to elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

# Usage example
bucket = TokenBucket(capacity=100, refill_rate=10)

if bucket.allow_request():
    process_request()
else:
    return 429  # Too Many Requests
```

---

## Hands-On Exercises

### Exercise 1: Rate Limiter Comparison

Use `examples/System_Design/05_rate_limiter.py` to explore rate limiting behavior.

**Tasks:**
1. Run all demos and compare token bucket vs. sliding window behavior
2. Create a new traffic pattern: 5 requests/second steady state with bursts of 20 requests every 10 seconds
3. Configure both algorithms to allow 10 req/s average. Compare which one handles the bursty pattern better
4. Implement a **per-client** rate limiter that tracks separate limits for different client IPs

### Exercise 2: API Gateway Router

Build a simple API gateway that routes requests to different backend services.

**Tasks:**
1. Implement a `Gateway` class with route registration: `gateway.route("/api/users/*", user_service)`
2. Add path parameter extraction: `/api/users/123` → `service=user_service, params={"id": "123"}`
3. Add middleware chain support: authentication → rate limiting → logging → forward to service
4. Implement request/response transformation: strip internal headers, add CORS headers
5. Add a circuit breaker per backend service (reference `14_circuit_breaker.py`)

### Exercise 3: Reverse Proxy Cache

Implement a caching reverse proxy that sits in front of backend services.

**Tasks:**
1. Build a `CachingProxy` that caches GET responses with configurable TTL
2. Implement `Cache-Control` header parsing: respect `max-age`, `no-cache`, `no-store`
3. Add cache invalidation on POST/PUT/DELETE to the same path
4. Measure cache hit rate under different workloads: read-heavy (90% GET) vs. write-heavy (50% POST)

---

## 6. References

### Tools
- [Nginx](https://nginx.org/)
- [Kong Gateway](https://konghq.com/)
- [AWS API Gateway](https://aws.amazon.com/api-gateway/)
- [Envoy Proxy](https://www.envoyproxy.io/)

### Documentation
- [Nginx Reverse Proxy](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/)
- [Rate Limiting Best Practices](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)

### Algorithms
- [Token Bucket vs Leaky Bucket](https://www.cloudflare.com/learning/bots/what-is-rate-limiting/)

---

**Previous**: [Load Balancing](./04_Load_Balancing.md) | **Next**: [Caching Strategies](./06_Caching_Strategies.md)

---

**Document Information**
- Last Modified: 2024
- Difficulty: ⭐⭐⭐
- Estimated Learning Time: 2-3 hours
