# 09. Helm 패키지 관리

## 학습 목표
- Helm의 개념과 구조 이해
- Helm 차트 생성 및 관리
- values.yaml을 통한 설정 커스터마이징
- 템플릿 함수와 조건문 활용
- 차트 저장소 관리 및 배포

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
│                     Helm 아키텍처                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────┐              │
│  │              Helm CLI                     │              │
│  │  • 차트 설치/업그레이드/삭제              │              │
│  │  • 릴리스 관리                            │              │
│  │  • 저장소 관리                            │              │
│  └──────────────────────┬───────────────────┘              │
│                         │                                   │
│          ┌──────────────┼──────────────┐                   │
│          ▼              ▼              ▼                    │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│    │ Chart    │  │ Values   │  │ K8s API  │               │
│    │ Repository│ │ (설정)    │  │ Server   │               │
│    └──────────┘  └──────────┘  └──────────┘               │
│                                                             │
│  핵심 개념:                                                 │
│  • Chart: 패키지 (YAML 템플릿 묶음)                        │
│  • Release: 차트의 인스턴스 (설치된 애플리케이션)          │
│  • Repository: 차트 저장소                                  │
│  • Values: 차트 설정값                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Helm의 장점

```
기존 방식 (다수의 YAML 파일):
├── deployment.yaml
├── service.yaml
├── configmap.yaml
├── secret.yaml
├── ingress.yaml
├── pvc.yaml
└── ...

문제점:
• 환경별 설정 관리 어려움
• 버전 관리 복잡
• 롤백 어려움
• 재사용 불가

Helm 사용:
├── myapp-chart/
│   ├── Chart.yaml          # 메타데이터
│   ├── values.yaml         # 기본 설정
│   ├── values-prod.yaml    # 프로덕션 설정
│   └── templates/          # 템플릿
│       ├── deployment.yaml
│       ├── service.yaml
│       └── ...

장점:
• 단일 명령으로 설치/업그레이드
• 환경별 values 파일로 설정 분리
• 릴리스 이력 및 롤백 지원
• 차트 재사용 및 공유
```

---

## 2. Helm 설치 및 설정

### 2.1 Helm 설치

```bash
# macOS
brew install helm

# Linux (스크립트)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Linux (apt)
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm

# 버전 확인
helm version
```

### 2.2 저장소 설정

```bash
# 공식 저장소 추가
helm repo add stable https://charts.helm.sh/stable
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

# 저장소 목록
helm repo list

# 저장소 업데이트
helm repo update

# 저장소 삭제
helm repo remove stable

# 차트 검색
helm search repo nginx
helm search repo bitnami/postgresql --versions

# 차트 정보 확인
helm show chart bitnami/nginx
helm show values bitnami/nginx
helm show readme bitnami/nginx
```

### 2.3 기본 명령어

```bash
# 차트 설치
helm install my-release bitnami/nginx

# 네임스페이스 지정
helm install my-release bitnami/nginx -n production --create-namespace

# values 파일 사용
helm install my-release bitnami/nginx -f custom-values.yaml

# 인라인 values 설정
helm install my-release bitnami/nginx --set replicaCount=3

# Dry-run (테스트)
helm install my-release bitnami/nginx --dry-run --debug

# 릴리스 목록
helm list
helm list -n production
helm list --all-namespaces

# 릴리스 상태
helm status my-release

# 업그레이드
helm upgrade my-release bitnami/nginx --set replicaCount=5

# 설치 또는 업그레이드 (없으면 설치, 있으면 업그레이드)
helm upgrade --install my-release bitnami/nginx

# 롤백
helm rollback my-release 1

# 히스토리
helm history my-release

# 삭제
helm uninstall my-release
helm uninstall my-release --keep-history  # 히스토리 유지
```

---

## 3. 차트 구조

### 3.1 차트 디렉토리 구조

```
myapp/
├── Chart.yaml              # 차트 메타데이터 (필수)
├── Chart.lock              # 의존성 버전 잠금
├── values.yaml             # 기본 설정값 (필수)
├── values.schema.json      # values 스키마 (선택)
├── .helmignore             # 패키징 제외 파일
├── README.md               # 차트 문서
├── LICENSE                 # 라이선스
├── charts/                 # 의존성 차트
│   └── subchart/
├── crds/                   # CustomResourceDefinition
│   └── myresource.yaml
└── templates/              # Kubernetes 매니페스트 템플릿
    ├── NOTES.txt           # 설치 후 메시지
    ├── _helpers.tpl        # 템플릿 헬퍼 함수
    ├── deployment.yaml
    ├── service.yaml
    ├── configmap.yaml
    ├── secret.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    └── tests/              # 테스트
        └── test-connection.yaml
```

### 3.2 Chart.yaml

```yaml
# Chart.yaml
apiVersion: v2                    # Helm 3용 (v1은 Helm 2)
name: myapp                       # 차트 이름
version: 1.2.3                    # 차트 버전 (SemVer)
appVersion: "2.0.0"               # 애플리케이션 버전
description: My awesome application
type: application                 # application 또는 library
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
kubeVersion: ">=1.22.0-0"         # 지원 K8s 버전
deprecated: false

# 의존성
dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
    tags:
      - database
  - name: redis
    version: "17.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
    alias: cache  # 별칭

# 어노테이션
annotations:
  category: Backend
  licenses: Apache-2.0
```

### 3.3 차트 생성

```bash
# 새 차트 생성
helm create myapp

# 구조 확인
tree myapp/

# 의존성 업데이트
helm dependency update myapp/
helm dependency build myapp/

# 차트 검증
helm lint myapp/

# 차트 패키징
helm package myapp/
# 결과: myapp-1.2.3.tgz

# 템플릿 렌더링 (디버그)
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
  # 템플릿 변수 사용
  name: {{ .Release.Name }}-{{ .Chart.Name }}
  labels:
    # include로 헬퍼 함수 호출
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
        # 설정 변경 시 Pod 재시작 트리거
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
# 릴리스 정보
{{ .Release.Name }}       # 릴리스 이름
{{ .Release.Namespace }}  # 네임스페이스
{{ .Release.IsUpgrade }}  # 업그레이드 여부
{{ .Release.IsInstall }}  # 신규 설치 여부
{{ .Release.Revision }}   # 릴리스 리비전

# 차트 정보
{{ .Chart.Name }}         # 차트 이름
{{ .Chart.Version }}      # 차트 버전
{{ .Chart.AppVersion }}   # 앱 버전

# Values
{{ .Values.key }}         # values.yaml 값

# 파일
{{ .Files.Get "config.ini" }}           # 파일 내용
{{ .Files.GetBytes "binary.dat" }}      # 바이너리 파일
{{ .Files.Glob "files/*" }}             # 패턴 매칭

# 템플릿
{{ .Template.Name }}      # 현재 템플릿 경로
{{ .Template.BasePath }}  # templates 디렉토리 경로

# Capabilities (클러스터 정보)
{{ .Capabilities.KubeVersion.Major }}   # K8s 메이저 버전
{{ .Capabilities.APIVersions.Has "apps/v1" }}  # API 지원 확인
```

### 4.3 헬퍼 함수 (_helpers.tpl)

```yaml
# templates/_helpers.tpl
{{/*
차트 이름 (단축)
*/}}
{{- define "myapp.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
전체 이름 생성
릴리스 이름이 차트 이름을 포함하면 그대로 사용
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
공통 레이블
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
셀렉터 레이블
*/}}
{{- define "myapp.selectorLabels" -}}
app.kubernetes.io/name: {{ include "myapp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
차트 이름:버전
*/}}
{{- define "myapp.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
ServiceAccount 이름
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
# 조건문
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

# 조건 연산자
{{- if and .Values.ingress.enabled .Values.ingress.tls }}
{{- if or .Values.env.dev .Values.env.staging }}
{{- if not .Values.disabled }}
{{- if eq .Values.type "ClusterIP" }}
{{- if ne .Values.env "production" }}
{{- if gt .Values.replicas 1 }}

# 반복문 (range)
{{- range .Values.hosts }}
- host: {{ .name }}
  paths:
  {{- range .paths }}
  - path: {{ .path }}
    pathType: {{ .pathType }}
  {{- end }}
{{- end }}

# 반복 (인덱스 포함)
{{- range $index, $host := .Values.hosts }}
- name: host-{{ $index }}
  value: {{ $host }}
{{- end }}

# with (스코프 변경)
{{- with .Values.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}

# 변수 할당
{{- $fullName := include "myapp.fullname" . -}}
{{- $svcPort := .Values.service.port -}}

# default (기본값)
{{ .Values.image.tag | default .Chart.AppVersion }}

# 문자열 함수
{{ .Values.name | upper }}
{{ .Values.name | lower }}
{{ .Values.name | title }}
{{ .Values.name | trim }}
{{ .Values.name | quote }}          # "value"
{{ .Values.name | squote }}         # 'value'
{{ printf "%s-%s" .Release.Name .Chart.Name }}

# 인덴트
{{ toYaml .Values.resources | indent 2 }}
{{ toYaml .Values.resources | nindent 2 }}  # 줄바꿈 + 인덴트

# 리스트/맵 함수
{{ list "a" "b" "c" | join "," }}
{{ dict "key1" "value1" "key2" "value2" | toYaml }}
{{ .Values.list | first }}
{{ .Values.list | last }}
{{ .Values.list | rest }}           # 첫 번째 제외
{{ .Values.list | initial }}        # 마지막 제외

# lookup (클러스터에서 조회)
{{- $secret := lookup "v1" "Secret" .Release.Namespace "my-secret" -}}
{{- if $secret }}
  # Secret 존재
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
# 기본 설정

# 리플리카 수
replicaCount: 1

# 이미지 설정
image:
  repository: myapp/myapp
  pullPolicy: IfNotPresent
  tag: ""  # 비어있으면 Chart.AppVersion 사용

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

# ServiceAccount
serviceAccount:
  create: true
  annotations: {}
  name: ""

# Pod 보안
podAnnotations: {}
podSecurityContext:
  fsGroup: 1000

securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true

# Service 설정
service:
  enabled: true
  type: ClusterIP
  port: 80
  annotations: {}

# Ingress 설정
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

# 리소스 제한
resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 100m
    memory: 128Mi

# 오토스케일링
autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80

# 노드 선택
nodeSelector: {}
tolerations: []
affinity: {}

# 환경 변수
env:
  LOG_LEVEL: info
  DATABASE_HOST: localhost

# ConfigMap에서 로드할 환경 변수
envFrom: []

# 추가 볼륨
extraVolumes: []
extraVolumeMounts: []

# 영속성
persistence:
  enabled: false
  storageClass: ""
  accessMode: ReadWriteOnce
  size: 10Gi
  existingClaim: ""

# 프로브
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

# 의존성 차트 설정
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
replicaCount: 3

image:
  tag: "1.0.0"  # 고정 버전

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
# 기본 values 사용
helm install myapp ./myapp

# values 파일 지정
helm install myapp ./myapp -f values-prod.yaml

# 여러 values 파일 (나중 파일이 우선)
helm install myapp ./myapp -f values.yaml -f values-prod.yaml -f values-secret.yaml

# 인라인 설정
helm install myapp ./myapp --set replicaCount=3

# 복잡한 값 설정
helm install myapp ./myapp \
  --set image.tag=v1.0.0 \
  --set 'ingress.hosts[0].host=app.example.com' \
  --set 'env.API_KEY=secret123'

# 파일 내용을 값으로
helm install myapp ./myapp --set-file config=./app.conf

# values 병합 확인 (dry-run)
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
# 테스트 실행
helm test my-release

# 테스트 결과 확인
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
    # 훅 타입
    "helm.sh/hook": pre-install,pre-upgrade
    # 훅 우선순위 (낮은 숫자 먼저)
    "helm.sh/hook-weight": "-5"
    # 삭제 정책
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
훅 타입:
• pre-install   : 설치 전
• post-install  : 설치 후
• pre-delete    : 삭제 전
• post-delete   : 삭제 후
• pre-upgrade   : 업그레이드 전
• post-upgrade  : 업그레이드 후
• pre-rollback  : 롤백 전
• post-rollback : 롤백 후
• test          : helm test 실행 시

삭제 정책:
• before-hook-creation : 새 훅 생성 전 이전 훅 삭제
• hook-succeeded       : 훅 성공 시 삭제
• hook-failed          : 훅 실패 시 삭제
```

### 6.3 차트 저장소 관리

```bash
# ChartMuseum 실행 (로컬 저장소)
docker run -d \
  -p 8080:8080 \
  -e DEBUG=1 \
  -e STORAGE=local \
  -e STORAGE_LOCAL_ROOTDIR=/charts \
  -v $(pwd)/charts:/charts \
  ghcr.io/helm/chartmuseum:v0.16.0

# 저장소 추가
helm repo add myrepo http://localhost:8080

# 차트 업로드
curl --data-binary "@myapp-1.0.0.tgz" http://localhost:8080/api/charts

# 또는 Helm 플러그인 사용
helm plugin install https://github.com/chartmuseum/helm-push
helm cm-push myapp-1.0.0.tgz myrepo

# OCI 레지스트리 사용 (Helm 3.8+)
helm push myapp-1.0.0.tgz oci://ghcr.io/myorg/charts

# OCI에서 설치
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
# 의존성 다운로드
helm dependency update ./myapp

# 의존성 확인
helm dependency list ./myapp

# charts/ 디렉토리에 다운로드됨
ls ./myapp/charts/
```

### 6.5 릴리스 관리

```bash
# 릴리스 목록
helm list -A

# 릴리스 상태
helm status myapp

# 릴리스 히스토리
helm history myapp

# 특정 리비전의 values 확인
helm get values myapp --revision 2

# 매니페스트 확인
helm get manifest myapp

# 롤백
helm rollback myapp 2

# 삭제 (히스토리 유지)
helm uninstall myapp --keep-history

# 삭제된 릴리스 확인
helm list --uninstalled

# 완전 삭제
helm uninstall myapp
```

---

## 7. 연습 문제

### 연습 1: 웹 애플리케이션 차트 생성
```bash
# 요구사항:
# 1. 새 차트 생성 (webapp)
# 2. Deployment, Service, Ingress 템플릿
# 3. ConfigMap으로 설정 관리
# 4. values.yaml에 기본값 설정
# 5. values-prod.yaml에 프로덕션 설정

# 실행 명령
helm create webapp
# 필요한 파일 수정
```

### 연습 2: 의존성이 있는 차트
```yaml
# 요구사항:
# 1. PostgreSQL 의존성 추가
# 2. Redis 의존성 추가 (condition으로 선택적)
# 3. 의존성 차트 설정을 values.yaml에 추가

# Chart.yaml 작성
```

### 연습 3: Helm Hook 구현
```yaml
# 요구사항:
# 1. pre-install: 데이터베이스 마이그레이션
# 2. post-install: 알림 전송
# 3. pre-upgrade: 백업 생성

# Hook Job 템플릿 작성
```

### 연습 4: 차트 배포 자동화
```bash
# 요구사항:
# 1. Chart.yaml 버전 업데이트
# 2. 차트 패키징
# 3. OCI 레지스트리에 푸시
# 4. 스테이징/프로덕션 배포

# 스크립트 또는 CI/CD 파이프라인 작성
```

---

## 다음 단계

- [10_CI_CD_파이프라인](10_CI_CD_파이프라인.md) - GitHub Actions와 배포 자동화
- [07_Kubernetes_보안](07_Kubernetes_보안.md) - 보안 복습
- [08_Kubernetes_심화](08_Kubernetes_심화.md) - 고급 K8s 기능

## 참고 자료

- [Helm 공식 문서](https://helm.sh/docs/)
- [Helm 차트 모범 사례](https://helm.sh/docs/chart_best_practices/)
- [Helm 템플릿 가이드](https://helm.sh/docs/chart_template_guide/)
- [Artifact Hub](https://artifacthub.io/) - 차트 검색

---

[← 이전: Kubernetes 심화](08_Kubernetes_심화.md) | [다음: CI/CD 파이프라인 →](10_CI_CD_파이프라인.md) | [목차](00_Overview.md)
