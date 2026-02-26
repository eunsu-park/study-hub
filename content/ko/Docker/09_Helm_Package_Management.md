# 09. Helm 패키지 관리

**이전**: [Kubernetes 고급](./08_Kubernetes_Advanced.md) | **다음**: [CI/CD 파이프라인](./10_CI_CD_Pipelines.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Helm이 무엇인지, 그리고 Kubernetes 애플리케이션 패키징과 배포를 어떻게 단순화하는지 설명할 수 있습니다
2. 적절한 디렉토리 구조와 메타데이터로 Helm 차트를 생성할 수 있습니다
3. Go 템플릿 문법, 내장 함수, 조건문을 사용하여 Helm 템플릿을 작성할 수 있습니다
4. values.yaml과 커맨드라인 오버라이드를 통해 배포를 커스터마이징할 수 있습니다
5. install, upgrade, rollback, uninstall 작업으로 차트 릴리스를 관리할 수 있습니다
6. 재사용 가능한 컴포넌트를 위해 차트 저장소와 의존성 관리를 활용할 수 있습니다

---

Kubernetes 애플리케이션을 배포하려면 일반적으로 Deployment, Service, ConfigMap, Secret 등 여러 YAML 매니페스트가 필요합니다. 애플리케이션이 성장함에 따라 이 매니페스트들을 관리하는 것은 복잡하고 오류가 발생하기 쉽습니다. Helm은 Kubernetes의 사실상 표준 패키지 매니저로, 관련 매니페스트를 설정 가능한 템플릿이 있는 재사용 가능하고 버전이 지정된 차트로 패키징합니다. Helm을 익히면 배포 복잡성이 크게 줄어들고 환경 전반에 걸쳐 일관성 있고 반복 가능한 배포가 가능해집니다.

## 목차
1. [Helm 개요](#1-helm-개요)
2. [Helm 설치 및 설정](#2-helm-설치-및-설정)
3. [차트 구조](#3-차트-구조)
4. [템플릿 작성](#4-템플릿-작성)
5. [Values와 설정](#5-values와-설정)
6. [차트 관리](#6-차트-관리)
7. [연습 문제](#7-연습-문제)

---

## 1. Helm 개요

### 1.1 Helm이란?

```
┌─────────────────────────────────────────────────────────────┐
│                     Helm Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────┐              │
│  │              Helm CLI                     │              │
│  │  • Install/upgrade/delete charts          │              │
│  │  • Release management                     │              │
│  │  • Repository management                  │              │
│  └──────────────────────┬───────────────────┘              │
│                         │                                   │
│          ┌──────────────┼──────────────┐                   │
│          ▼              ▼              ▼                    │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│    │ Chart    │  │ Values   │  │ K8s API  │               │
│    │Repository│  │(Config)  │  │ Server   │               │
│    └──────────┘  └──────────┘  └──────────┘               │
│                                                             │
│  Key Concepts:                                              │
│  • Chart: Package (YAML template bundle)                   │
│  • Release: Chart instance (deployed application)          │
│  • Repository: Chart storage                               │
│  • Values: Chart configuration                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Helm의 장점

```
Traditional Method (Multiple YAML files):
├── deployment.yaml
├── service.yaml
├── configmap.yaml
├── secret.yaml
├── ingress.yaml
├── pvc.yaml
└── ...

Problems:
• Difficult to manage per-environment configuration
• Complex version management
• Difficult rollback
• Not reusable

Using Helm:
├── myapp-chart/
│   ├── Chart.yaml          # Metadata
│   ├── values.yaml         # Default configuration
│   ├── values-prod.yaml    # Production configuration
│   └── templates/          # Templates
│       ├── deployment.yaml
│       ├── service.yaml
│       └── ...

Benefits:
• Single command for install/upgrade
• Separate configuration per environment with values files
• Release history and rollback support
• Chart reuse and sharing
```

---

## 2. Helm 설치 및 설정

### 2.1 Helm 설치

```bash
# macOS
brew install helm

# Linux (script)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Linux (apt)
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm

# Check version
helm version
```

### 2.2 저장소 설정

```bash
# Add official repositories
helm repo add stable https://charts.helm.sh/stable
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

# List repositories
helm repo list

# Update repositories
helm repo update

# Remove repository
helm repo remove stable

# Search charts
helm search repo nginx
helm search repo bitnami/postgresql --versions

# Show chart information
helm show chart bitnami/nginx
helm show values bitnami/nginx
helm show readme bitnami/nginx
```

### 2.3 기본 명령어

```bash
# Install chart
helm install my-release bitnami/nginx

# Specify namespace
helm install my-release bitnami/nginx -n production --create-namespace

# Use values file
helm install my-release bitnami/nginx -f custom-values.yaml

# Set inline values
helm install my-release bitnami/nginx --set replicaCount=3

# Dry-run (test)
helm install my-release bitnami/nginx --dry-run --debug

# List releases
helm list
helm list -n production
helm list --all-namespaces

# Release status
helm status my-release

# Upgrade
helm upgrade my-release bitnami/nginx --set replicaCount=5

# Install or upgrade (install if not exists, upgrade if exists)
helm upgrade --install my-release bitnami/nginx

# Rollback
helm rollback my-release 1

# History
helm history my-release

# Uninstall
helm uninstall my-release
helm uninstall my-release --keep-history  # Keep history
```

---

## 3. 차트 구조

### 3.1 차트 디렉토리 구조

```
myapp/
├── Chart.yaml              # Chart metadata (required)
├── Chart.lock              # Dependency version lock
├── values.yaml             # Default configuration (required)
├── values.schema.json      # Values schema (optional)
├── .helmignore             # Files to exclude from packaging
├── README.md               # Chart documentation
├── LICENSE                 # License
├── charts/                 # Dependency charts
│   └── subchart/
├── crds/                   # CustomResourceDefinition
│   └── myresource.yaml
└── templates/              # Kubernetes manifest templates
    ├── NOTES.txt           # Post-install message
    ├── _helpers.tpl        # Template helper functions
    ├── deployment.yaml
    ├── service.yaml
    ├── configmap.yaml
    ├── secret.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    └── tests/              # Tests
        └── test-connection.yaml
```

### 3.2 Chart.yaml

```yaml
# Chart.yaml
apiVersion: v2                    # For Helm 3 (v1 for Helm 2)
name: myapp                       # Chart name
version: 1.2.3                    # Chart version (SemVer)
appVersion: "2.0.0"               # Application version
description: My awesome application
type: application                 # application or library
keywords:
  - web
  - backend
home: https://example.com
sources:
  - https://github.com/example/myapp
maintainers:
  - name: John Doe
    email: john@example.com
    url: https://johndoe.com
icon: https://example.com/icon.png
kubeVersion: ">=1.22.0-0"         # Supported K8s version
deprecated: false

# Dependencies
dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled  # Conditionally include — skip the DB in dev if an external service is used
    tags:
      - database
  - name: redis
    version: "17.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
    alias: cache  # Alias — lets you reference redis values under `.Values.cache` for clarity

# Annotations
annotations:
  category: Backend
  licenses: Apache-2.0
```

### 3.3 차트 생성

```bash
# Create new chart
helm create myapp

# Check structure
tree myapp/

# Update dependencies
helm dependency update myapp/
helm dependency build myapp/

# Validate chart
helm lint myapp/

# Package chart
helm package myapp/
# Result: myapp-1.2.3.tgz

# Render template (debug)
helm template my-release myapp/ --debug
helm template my-release myapp/ -f custom-values.yaml
```

---

## 4. 템플릿 작성

### 4.1 기본 템플릿 문법

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  # Use template variables
  name: {{ .Release.Name }}-{{ .Chart.Name }}
  labels:
    # Call helper function with include
    {{- include "myapp.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "myapp.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "myapp.selectorLabels" . | nindent 8 }}
      annotations:
        # Trigger Pod restart on config change — without this, ConfigMap updates won't reach running pods
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - name: http
          containerPort: {{ .Values.service.port }}
          protocol: TCP
        {{- if .Values.resources }}
        resources:
          {{- toYaml .Values.resources | nindent 10 }}
        {{- end }}
```

### 4.2 내장 객체

```yaml
# Release information
{{ .Release.Name }}       # Release name
{{ .Release.Namespace }}  # Namespace
{{ .Release.IsUpgrade }}  # Is upgrade?
{{ .Release.IsInstall }}  # Is new install?
{{ .Release.Revision }}   # Release revision

# Chart information
{{ .Chart.Name }}         # Chart name
{{ .Chart.Version }}      # Chart version
{{ .Chart.AppVersion }}   # App version

# Values
{{ .Values.key }}         # values.yaml value

# Files
{{ .Files.Get "config.ini" }}           # File contents
{{ .Files.GetBytes "binary.dat" }}      # Binary file
{{ .Files.Glob "files/*" }}             # Pattern matching

# Template
{{ .Template.Name }}      # Current template path
{{ .Template.BasePath }}  # templates directory path

# Capabilities (cluster information)
{{ .Capabilities.KubeVersion.Major }}   # K8s major version
{{ .Capabilities.APIVersions.Has "apps/v1" }}  # Check API support
```

### 4.3 헬퍼 함수 (_helpers.tpl)

```yaml
# templates/_helpers.tpl
{{/*
Chart name (short)
*/}}
{{- define "myapp.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Full name generation
Use release name as-is if it contains chart name
*/}}
{{- define "myapp.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "myapp.labels" -}}
helm.sh/chart: {{ include "myapp.chart" . }}
{{ include "myapp.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "myapp.selectorLabels" -}}
app.kubernetes.io/name: {{ include "myapp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Chart name:version
*/}}
{{- define "myapp.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
ServiceAccount name
*/}}
{{- define "myapp.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "myapp.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
```

### 4.4 제어문과 함수

```yaml
# Conditionals
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
# ...
{{- end }}

# if-else
{{- if .Values.persistence.enabled }}
  volumeClaimTemplates:
  # ...
{{- else }}
  volumes:
  - name: data
    emptyDir: {}
{{- end }}

# Conditional operators
{{- if and .Values.ingress.enabled .Values.ingress.tls }}
{{- if or .Values.env.dev .Values.env.staging }}
{{- if not .Values.disabled }}
{{- if eq .Values.type "ClusterIP" }}
{{- if ne .Values.env "production" }}
{{- if gt .Values.replicas 1 }}

# Loops (range)
{{- range .Values.hosts }}
- host: {{ .name }}
  paths:
  {{- range .paths }}
  - path: {{ .path }}
    pathType: {{ .pathType }}
  {{- end }}
{{- end }}

# Loop (with index)
{{- range $index, $host := .Values.hosts }}
- name: host-{{ $index }}
  value: {{ $host }}
{{- end }}

# with (change scope)
{{- with .Values.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}

# Variable assignment
{{- $fullName := include "myapp.fullname" . -}}
{{- $svcPort := .Values.service.port -}}

# default (default value)
{{ .Values.image.tag | default .Chart.AppVersion }}

# String functions
{{ .Values.name | upper }}
{{ .Values.name | lower }}
{{ .Values.name | title }}
{{ .Values.name | trim }}
{{ .Values.name | quote }}          # "value"
{{ .Values.name | squote }}         # 'value'
{{ printf "%s-%s" .Release.Name .Chart.Name }}

# Indentation
{{ toYaml .Values.resources | indent 2 }}
{{ toYaml .Values.resources | nindent 2 }}  # Newline + indent

# List/map functions
{{ list "a" "b" "c" | join "," }}
{{ dict "key1" "value1" "key2" "value2" | toYaml }}
{{ .Values.list | first }}
{{ .Values.list | last }}
{{ .Values.list | rest }}           # Exclude first
{{ .Values.list | initial }}        # Exclude last

# lookup (query cluster)
{{- $secret := lookup "v1" "Secret" .Release.Namespace "my-secret" -}}
{{- if $secret }}
  # Secret exists
{{- end }}
```

### 4.5 실전 템플릿 예제

```yaml
# templates/service.yaml
{{- if .Values.service.enabled -}}
apiVersion: v1
kind: Service
metadata:
  name: {{ include "myapp.fullname" . }}
  labels:
    {{- include "myapp.labels" . | nindent 4 }}
  {{- with .Values.service.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  type: {{ .Values.service.type }}
  {{- if and (eq .Values.service.type "LoadBalancer") .Values.service.loadBalancerIP }}
  loadBalancerIP: {{ .Values.service.loadBalancerIP }}
  {{- end }}
  {{- if and (eq .Values.service.type "LoadBalancer") .Values.service.loadBalancerSourceRanges }}
  loadBalancerSourceRanges:
    {{- toYaml .Values.service.loadBalancerSourceRanges | nindent 4 }}
  {{- end }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
      {{- if and (or (eq .Values.service.type "NodePort") (eq .Values.service.type "LoadBalancer")) .Values.service.nodePort }}
      nodePort: {{ .Values.service.nodePort }}
      {{- end }}
  selector:
    {{- include "myapp.selectorLabels" . | nindent 4 }}
{{- end }}

---
# templates/ingress.yaml
{{- if .Values.ingress.enabled -}}
{{- $fullName := include "myapp.fullname" . -}}
{{- $svcPort := .Values.service.port -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ $fullName }}
  labels:
    {{- include "myapp.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType }}
            backend:
              service:
                name: {{ $fullName }}
                port:
                  number: {{ $svcPort }}
          {{- end }}
    {{- end }}
{{- end }}
```

---

## 5. Values와 설정

### 5.1 values.yaml 구조

```yaml
# values.yaml
# Externalize config so the same chart works across dev/staging/prod
# Default configuration

# Replica count
replicaCount: 1

# Image configuration
image:
  repository: myapp/myapp
  pullPolicy: IfNotPresent  # Avoids unnecessary pulls in dev; override to Always in production for security
  tag: ""  # Uses Chart.AppVersion if empty — keeps image version in sync with chart version by default

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

# ServiceAccount
serviceAccount:
  create: true
  annotations: {}
  name: ""

# Pod security
podAnnotations: {}
podSecurityContext:
  fsGroup: 1000

securityContext:
  runAsNonRoot: true  # Prevents container from running as UID 0 even if the image defaults to root
  runAsUser: 1000
  capabilities:
    drop:
    - ALL  # Drop all Linux capabilities — add back only what the app truly needs
  readOnlyRootFilesystem: true  # Immutable filesystem: an attacker cannot install tools or drop malware

# Service configuration
service:
  enabled: true
  type: ClusterIP
  port: 80
  annotations: {}

# Ingress configuration
ingress:
  enabled: false
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: myapp.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: myapp-tls
      hosts:
        - myapp.example.com

# Resource limits
resources:
  limits:
    cpu: 500m
    memory: 512Mi  # limits prevent one pod from starving others on the node
  requests:
    cpu: 100m
    memory: 128Mi  # requests guarantee scheduling — the scheduler reserves this much capacity

# Autoscaling
autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80

# Node selection
nodeSelector: {}
tolerations: []
affinity: {}

# Environment variables
env:
  LOG_LEVEL: info
  DATABASE_HOST: localhost

# Environment variables loaded from ConfigMap
envFrom: []

# Additional volumes
extraVolumes: []
extraVolumeMounts: []

# Persistence
persistence:
  enabled: false
  storageClass: ""
  accessMode: ReadWriteOnce
  size: 10Gi
  existingClaim: ""

# Probes
livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: http
  initialDelaySeconds: 5
  periodSeconds: 5

# Dependency chart configuration
postgresql:
  enabled: false
  auth:
    database: myapp
    username: myapp

redis:
  enabled: false
  architecture: standalone
```

### 5.2 환경별 values 파일

```yaml
# values-dev.yaml
replicaCount: 1

image:
  tag: "dev"

env:
  LOG_LEVEL: debug
  ENV: development

resources:
  limits:
    cpu: 200m
    memory: 256Mi
  requests:
    cpu: 50m
    memory: 64Mi

ingress:
  enabled: true
  hosts:
    - host: dev.myapp.example.com
      paths:
        - path: /
          pathType: Prefix

---
# values-staging.yaml
replicaCount: 2

image:
  tag: "staging"

env:
  LOG_LEVEL: info
  ENV: staging

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 100m
    memory: 128Mi

ingress:
  enabled: true
  hosts:
    - host: staging.myapp.example.com
      paths:
        - path: /
          pathType: Prefix

---
# values-prod.yaml
replicaCount: 3  # Multiple replicas for high availability — if one pod crashes, others continue serving

image:
  tag: "1.0.0"  # Fixed version — never use "latest" in production; pinned tags enable deterministic rollbacks

env:
  LOG_LEVEL: warn
  ENV: production

resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20

ingress:
  enabled: true
  hosts:
    - host: myapp.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: myapp-tls
      hosts:
        - myapp.example.com

postgresql:
  enabled: true
  auth:
    existingSecret: postgres-credentials
```

### 5.3 values 사용

```bash
# Use default values
helm install myapp ./myapp

# Specify values file
helm install myapp ./myapp -f values-prod.yaml

# Multiple values files (later files take precedence)
helm install myapp ./myapp -f values.yaml -f values-prod.yaml -f values-secret.yaml

# Inline configuration
helm install myapp ./myapp --set replicaCount=3

# Set complex values
helm install myapp ./myapp \
  --set image.tag=v1.0.0 \
  --set 'ingress.hosts[0].host=app.example.com' \
  --set 'env.API_KEY=secret123'

# File contents as value
helm install myapp ./myapp --set-file config=./app.conf

# Check merged values (dry-run)
helm install myapp ./myapp -f values-prod.yaml --dry-run --debug
```

---

## 6. 차트 관리

### 6.1 차트 테스트

```yaml
# templates/tests/test-connection.yaml
apiVersion: v1
kind: Pod
metadata:
  name: "{{ include "myapp.fullname" . }}-test-connection"
  labels:
    {{- include "myapp.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": test
    "helm.sh/hook-delete-policy": before-hook-creation,hook-succeeded
spec:
  containers:
    - name: wget
      image: busybox
      command: ['wget']
      args: ['{{ include "myapp.fullname" . }}:{{ .Values.service.port }}']
  restartPolicy: Never
```

```bash
# Run test
helm test my-release

# Check test results
kubectl logs my-release-myapp-test-connection
```

### 6.2 Hook (훅)

```yaml
# templates/hooks/pre-install-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: "{{ include "myapp.fullname" . }}-db-init"
  labels:
    {{- include "myapp.labels" . | nindent 4 }}
  annotations:
    # Hook type — run DB migration before app starts so the schema is ready when pods boot
    "helm.sh/hook": pre-install,pre-upgrade
    # Hook priority (lower number first) — ensures this runs before other hooks like config seeding
    "helm.sh/hook-weight": "-5"
    # Delete policy — clean up the Job to avoid accumulating completed pods in the namespace
    "helm.sh/hook-delete-policy": before-hook-creation,hook-succeeded
spec:
  template:
    spec:
      containers:
      - name: db-init
        image: postgres:15
        command: ["psql", "-c", "CREATE DATABASE myapp;"]
      restartPolicy: Never
  backoffLimit: 1
```

```
Hook Types:
• pre-install   : Before installation
• post-install  : After installation
• pre-delete    : Before deletion
• post-delete   : After deletion
• pre-upgrade   : Before upgrade
• post-upgrade  : After upgrade
• pre-rollback  : Before rollback
• post-rollback : After rollback
• test          : On helm test execution

Delete Policies:
• before-hook-creation : Delete previous hook before creating new one
• hook-succeeded       : Delete on success
• hook-failed          : Delete on failure
```

### 6.3 차트 저장소 관리

```bash
# Run ChartMuseum (local repository)
docker run -d \
  -p 8080:8080 \
  -e DEBUG=1 \
  -e STORAGE=local \
  -e STORAGE_LOCAL_ROOTDIR=/charts \
  -v $(pwd)/charts:/charts \
  ghcr.io/helm/chartmuseum:v0.16.0

# Add repository
helm repo add myrepo http://localhost:8080

# Upload chart
curl --data-binary "@myapp-1.0.0.tgz" http://localhost:8080/api/charts

# Or use Helm plugin
helm plugin install https://github.com/chartmuseum/helm-push
helm cm-push myapp-1.0.0.tgz myrepo

# Use OCI registry (Helm 3.8+) — OCI avoids running a separate chart server; reuses your existing container registry
helm push myapp-1.0.0.tgz oci://ghcr.io/myorg/charts

# Install from OCI
helm install myapp oci://ghcr.io/myorg/charts/myapp --version 1.0.0
```

### 6.4 의존성 관리

```yaml
# Chart.yaml
dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: "17.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
    alias: cache
```

```bash
# Download dependencies
helm dependency update ./myapp

# Check dependencies
helm dependency list ./myapp

# Downloaded to charts/ directory
ls ./myapp/charts/
```

### 6.5 릴리스 관리

```bash
# List releases
helm list -A

# Release status
helm status myapp

# Release history
helm history myapp

# Check values of specific revision
helm get values myapp --revision 2

# Check manifest
helm get manifest myapp

# Rollback
helm rollback myapp 2

# Uninstall (keep history)
helm uninstall myapp --keep-history

# Check uninstalled releases
helm list --uninstalled

# Complete deletion
helm uninstall myapp
```

---

## 7. 연습 문제

### 연습 1: 웹 애플리케이션 차트 생성
```bash
# Requirements:
# 1. Create new chart (webapp)
# 2. Deployment, Service, Ingress templates
# 3. Manage configuration with ConfigMap
# 4. Set default values in values.yaml
# 5. Set production configuration in values-prod.yaml

# Execution commands
helm create webapp
# Modify necessary files
```

### 연습 2: 의존성이 있는 차트
```yaml
# Requirements:
# 1. Add PostgreSQL dependency
# 2. Add Redis dependency (optional with condition)
# 3. Add dependency chart configuration to values.yaml

# Write Chart.yaml
```

### 연습 3: Helm Hook 구현
```yaml
# Requirements:
# 1. pre-install: Database migration
# 2. post-install: Send notification
# 3. pre-upgrade: Create backup

# Write Hook Job template
```

### 연습 4: 차트 배포 자동화
```bash
# Requirements:
# 1. Update Chart.yaml version
# 2. Package chart
# 3. Push to OCI registry
# 4. Deploy to staging/production

# Write script or CI/CD pipeline
```

---

## 다음 단계

- [10_CI_CD_파이프라인](10_CI_CD_Pipelines.md) - GitHub Actions와 배포 자동화
- [07_Kubernetes_보안](07_Kubernetes_Security.md) - 보안 복습
- [08_Kubernetes_고급](08_Kubernetes_Advanced.md) - 고급 K8s 기능

## 참고 자료

- [Helm 공식 문서](https://helm.sh/docs/)
- [Helm 차트 모범 사례](https://helm.sh/docs/chart_best_practices/)
- [Helm 템플릿 가이드](https://helm.sh/docs/chart_template_guide/)
- [Artifact Hub](https://artifacthub.io/) - 차트 검색

---

## 연습 문제

### 연습 1: 첫 번째 Helm 차트(Chart) 생성 및 설치

차트를 스캐폴드(scaffold)하고, 커스터마이징한 후 클러스터에 설치합니다.

1. 새 차트를 생성합니다: `helm create myapp`
2. 생성된 구조를 탐색합니다 (`Chart.yaml`, `values.yaml`, `templates/`)
3. `values.yaml`을 열어 `replicaCount`를 2로, `image.tag`를 `alpine`으로 변경합니다
4. 차트 오류를 검사합니다: `helm lint myapp`
5. 설치 없이 템플릿을 렌더링합니다: `helm template myapp ./myapp`
6. 차트를 설치합니다: `helm install myapp-release ./myapp`
7. 확인합니다: `helm list` 및 `kubectl get pods`
8. 제거합니다: `helm uninstall myapp-release`

### 연습 2: Values(값)로 배포 커스터마이징

동일한 차트를 다른 환경에 배포하기 위해 값 오버라이드(values override)를 사용합니다.

1. 연습 1의 `myapp` 차트를 사용합니다
2. `values-dev.yaml` 파일을 생성합니다:
   ```yaml
   replicaCount: 1
   service:
     type: NodePort
   ```
3. `values-prod.yaml` 파일을 생성합니다:
   ```yaml
   replicaCount: 3
   service:
     type: LoadBalancer
   ```
4. `dev` 네임스페이스에 설치합니다: `helm install myapp-dev ./myapp -f values-dev.yaml -n dev --create-namespace`
5. `prod` 네임스페이스에 설치합니다: `helm install myapp-prod ./myapp -f values-prod.yaml -n prod --create-namespace`
6. 두 릴리즈를 비교합니다: `helm list -A` 및 `kubectl get svc -A`

### 연습 3: 릴리즈(Release) 업그레이드 및 롤백(Rollback)

업그레이드와 롤백을 통해 Helm 릴리즈 라이프사이클을 실습합니다.

1. 복제본 1개로 차트를 설치합니다: `helm install myapp ./myapp --set replicaCount=1`
2. 릴리즈 상태를 확인합니다: `helm status myapp`
3. 복제본 3개로 업그레이드합니다: `helm upgrade myapp ./myapp --set replicaCount=3`
4. 변경 사항을 확인합니다: `kubectl get pods`
5. 릴리즈 히스토리(history)를 확인합니다: `helm history myapp`
6. 리비전(revision) 1로 롤백합니다: `helm rollback myapp 1`
7. 복제본 수가 1로 돌아갔는지 확인합니다: `kubectl get pods`

### 연습 4: Helm 저장소(Repository) 사용

공개 저장소에서 차트를 설치하고 값을 커스터마이징합니다.

1. Bitnami 저장소를 추가합니다: `helm repo add bitnami https://charts.bitnami.com/bitnami`
2. 저장소를 업데이트합니다: `helm repo update`
3. nginx 차트를 검색합니다: `helm search repo bitnami/nginx`
4. 기본 값을 확인합니다: `helm show values bitnami/nginx`
5. 사용자 정의 복제본 수로 nginx를 설치합니다:
   ```bash
   helm install my-nginx bitnami/nginx \
     --set replicaCount=2 \
     --set service.type=NodePort
   ```
6. minikube를 통해 서비스에 접근합니다: `minikube service my-nginx --url`
7. 완료 후 제거합니다: `helm uninstall my-nginx`

### 연습 5: 차트 의존성(Dependency) 추가

차트 의존성을 사용하여 멀티 컴포넌트 애플리케이션을 구성합니다.

1. 새 차트를 생성합니다: `helm create webapp`
2. `Chart.yaml`에 Redis 의존성을 추가합니다:
   ```yaml
   dependencies:
     - name: redis
       version: "19.x.x"
       repository: "https://charts.bitnami.com/bitnami"
   ```
3. 의존성을 다운로드합니다: `helm dependency update webapp`
4. `webapp` 차트 디렉토리 내에 `charts/redis-*.tgz`가 생성되었는지 확인합니다
5. `values.yaml`에서 앱이 `redis-master`를 호스트로 사용하여 Redis에 연결하도록 설정합니다
6. 통합 차트를 설치합니다: `helm install webapp-release ./webapp`
7. webapp과 Redis Pod 모두 실행 중인지 확인합니다: `kubectl get pods`

---

[← 이전: Kubernetes 고급](08_Kubernetes_Advanced.md) | [다음: CI/CD 파이프라인 →](10_CI_CD_Pipelines.md) | [목차](00_Overview.md)
