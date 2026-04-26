# 09. Helm과 Kustomize(Helm and Kustomize)

**이전**: [CNI와 고급 네트워킹](./08_CNI_and_Advanced_Networking.md) | **다음**: [커스텀 리소스 정의](./10_Custom_Resource_Definitions.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. values 파일과 의존성 관리를 사용하여 Helm 차트를 생성, 템플릿화, 배포할 수 있다
2. Helm 훅(hook)과 테스트를 사용하여 릴리스 수명주기 이벤트를 관리할 수 있다
3. 전략적 병합 패치(strategic merge patch)와 JSON 패치로 Kustomize 오버레이를 구축할 수 있다
4. Helm과 Kustomize를 비교하여 각 시나리오에 적합한 도구를 선택할 수 있다
5. Helmfile로 다중 차트 배포를 관리할 수 있다

---

Kubernetes 클러스터가 복잡해짐에 따라 원시 YAML 매니페스트 관리는 감당하기 어려워집니다. 단일 마이크로서비스에도 Deployment, Service, ConfigMap, HPA, NetworkPolicy, ServiceAccount가 필요할 수 있으며, 이는 수십 개의 서비스와 여러 환경에 걸쳐 곱해집니다. Helm과 Kustomize는 근본적으로 다른 방식으로 이 문제를 해결하는 두 가지 주요 도구입니다: Helm은 Go 템플릿을 사용한 클라이언트 측 템플릿화를, Kustomize는 템플릿 없이 오버레이 기반 패칭을 사용합니다. 이 레슨에서는 기본 사용법부터 고급 패턴까지 두 도구를 깊이 다루며, 사용 사례에 적합한 접근 방식을 선택하는 데 도움을 줍니다.

> **템플릿화 vs 패칭:** Helm은 템플릿과 values에서 YAML을 생성합니다 -- 유연하지만 읽기 어려운 템플릿을 만들 수 있습니다. Kustomize는 유효한 YAML을 오버레이로 패치합니다 -- 단순하지만 매개변수화에서 덜 유연합니다. 많은 팀이 둘 다 사용합니다: 서드파티 차트에는 Helm을, 애플리케이션별 오버레이에는 Kustomize를.

차트 구문에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 대규모 YAML이 templating과 patching 중 하나를 강제하는 이유, Helm의 텍스트 치환 모델 vs Kustomize의 구조적 병합 모델의 트레이드오프, Helm의 안전한 롤백을 가능하게 하는 release-라이프사이클 상태 머신, 그리고 의존성 해결의 실제 동작을 다룹니다.

## 목차

- [이론과 원리](#이론과-원리)
- [1. Helm 개념](#1-helm-개념)
  - [1.1 차트, 릴리스, 리포지토리](#11-차트-릴리스-리포지토리)
  - [1.2 Helm 아키텍처](#12-helm-아키텍처)
- [2. Helm 차트 구조](#2-helm-차트-구조)
- [3. Helm 템플릿 심층 분석](#3-helm-템플릿-심층-분석)
  - [3.1 내장 객체](#31-내장-객체)
  - [3.2 템플릿 함수와 파이프라인](#32-템플릿-함수와-파이프라인)
  - [3.3 흐름 제어(Flow Control)](#33-흐름-제어flow-control)
  - [3.4 명명된 템플릿과 헬퍼](#34-명명된-템플릿과-헬퍼)
  - [3.5 서브차트와 의존성](#35-서브차트와-의존성)
- [4. Helm 훅과 테스트](#4-helm-훅과-테스트)
  - [4.1 훅 유형](#41-훅-유형)
  - [4.2 데이터베이스 마이그레이션 훅](#42-데이터베이스-마이그레이션-훅)
  - [4.3 Helm 테스트](#43-helm-테스트)
- [5. 차트 개발 모범 사례](#5-차트-개발-모범-사례)
- [6. Kustomize 기초](#6-kustomize-기초)
  - [6.1 Base와 Overlay](#61-base와-overlay)
  - [6.2 Kustomization 파일](#62-kustomization-파일)
- [7. Kustomize 패치](#7-kustomize-패치)
  - [7.1 전략적 병합 패치(Strategic Merge Patches)](#71-전략적-병합-패치strategic-merge-patches)
  - [7.2 JSON 패치](#72-json-패치)
- [8. Kustomize 생성기와 변환기](#8-kustomize-생성기와-변환기)
  - [8.1 ConfigMap과 Secret 생성기](#81-configmap과-secret-생성기)
  - [8.2 변환기(Transformers)](#82-변환기transformers)
- [9. Helm vs Kustomize](#9-helm-vs-kustomize)
- [10. Helmfile](#10-helmfile)
- [연습문제](#연습문제)

---

## 이론과 원리

쿠버네티스 service가 몇 개의 리소스 이상이 되면 — Deployment + Service + ConfigMap + HPA + NetworkPolicy + ServiceAccount + 어쩌면 Ingress — 그리고 dev/stage/prod에 다른 레플리카 수, 이미지 태그, DB 호스트명으로 배포해야 하면, 평면 YAML은 무너집니다. 이를 해결하는 두 길에 부딪힙니다 — **templating**(템플릿에 변수를 치환하여 YAML 생성)과 **patching**(유효한 YAML에서 시작해 구조적 변경을 오버레이). Helm은 templating 길을, Kustomize는 patching 길을 갔습니다. 각각 다른 쪽이 근본적으로 복제할 수 없는 속성을 가집니다. 이 섹션은 두 철학, Helm 롤백을 동작하게 만드는 release-as-state-machine 모델, 그리고 Kustomize를 합성 가능하게 만드는 구조적 병합 알고리즘을 설명합니다.

### A. Templating vs Patching — 두 철학

같은 문제 — "환경별 변형으로 50개 매니페스트를 배포해야 한다" — 에 두 가지 정반대 답이 있습니다:

**Templating (Helm).** YAML을 텍스트로 취급. 변수와 제어 흐름(`{{ .Values.image.tag }}`, `{{- if .Values.ingress.enabled }}`)을 소스에 직접 임베드. 렌더러(`helm template`)가 값을 치환하고 최종 YAML을 방출. 장점 — 최대 표현력 — 어떤 문자열이든 매개변수화 가능, 조건 섹션, 리스트 루프, Sprig 함수로 계산 값. 단점 — 템플릿 자체는 유효한 YAML이 아님(YAML로 lint 불가); 공백이 깨지기 쉬움; 복잡한 차트는 읽기 어려워짐; IDE에서 파일을 열어 무엇이 배포되는지 그냥 볼 수 없음.

**Patching (Kustomize).** YAML을 구조적 데이터로 취급. 베이스는 유효한 매니페스트. 오버레이는 베이스에 병합되는 유효 YAML 패치. 렌더러(`kustomize build`)가 패치를 적용하고 최종 YAML을 방출. 장점 — 모든 파일이 유효 YAML — IDE 하이라이팅, 스키마 검증, 베이스만으로 kubectl apply 모두 동작; 배울 템플릿 구문 없음; 예측 가능한 합성. 단점 — 표현력이 적음 — "X를 가진 모든 컨테이너에 이 레이블을 추가" 같은 것을 쉽게 말할 수 없음; 복잡한 변환은 자체 학습 곡선을 가진 JSON Patch 필요; 조건적 포함이 어색.

어느 쪽도 보편적으로 더 낫지 않습니다. 가장 좋은 팀은 둘 다 사용합니다 — **서드파티 차트에는 Helm**(벤더 유지 패키지를 소비하고 그저 값만 오버라이드하고 싶을 때), **내부 매니페스트에는 Kustomize**(템플릿 계층 없이 정확히 무엇이 배포되는지 보고 싶을 때).

미묘한 속성 — Helm의 출력은 Helm 버전 간에 다르게 동작할 수 있는 Go 템플릿 엔진에 의존합니다 — Kustomize의 출력은 스펙으로부터 결정적입니다. 따라서 Kustomize는 렌더링된 YAML이 진실의 원천인 GitOps에서 우세한 경향이 있습니다.

### B. Helm — 상태 머신을 가진 패키지 관리자

Helm은 templating 도구 이상입니다 — **패키지 관리자**입니다. Helm `Chart`는 정의된 구조의 디렉토리(`Chart.yaml` 메타데이터, `values.yaml` 기본값, `templates/` 템플릿 디렉토리, 서브차트용 선택적 `charts/`)입니다. `helm install`을 하면 세 가지가 일어납니다:

1. **Render** — 템플릿과 값을 결합하여 최종 YAML 생성.
2. **Apply** — 클러스터에 kubectl 스타일 적용.
3. **Release 기록** — 렌더링된 매니페스트, 버전 번호, 메타데이터를 대상 네임스페이스의 Secret으로 저장.

세 번째 부분이 Helm을 templating 도구와 차별화합니다. **Release**는 버전 이력을 가진 명명된 설치입니다:

```
$ helm history my-app
REVISION  STATUS      CHART       APP VERSION  DESCRIPTION
1         superseded  my-app-1.0  v1.0         Install complete
2         superseded  my-app-1.1  v1.1         Upgrade complete
3         deployed    my-app-1.2  v1.2         Upgrade complete
```

각 업그레이드는 전체 렌더링된 매니페스트를 새 Secret으로 저장합니다. `helm rollback my-app 1`은 revision 1의 매니페스트를 다시 적용합니다. 이것이 Helm 롤백이 안전하고 원자적인 이유입니다 — 변했을 수도 있는 템플릿 시간 재계산이 아니라 저장된 이전 상태를 사용합니다.

훅(`helm.sh/hook: pre-install`, `post-upgrade` 등)이 이 상태 머신을 확장합니다 — 훅 어노테이션은 Helm에게 다음 단계 전에 Job을 실행하라고 지시합니다(예: "설치 전에 db-migrate 실행"). 실패한 훅은 release 전이를 차단합니다.

### C. Helm 템플릿 — Sprig, 파이프라인, 명명된 템플릿

Helm은 Go 템플릿과 [Sprig](http://masterminds.github.io/sprig/) 함수 라이브러리를 사용합니다. 결과는 YAML 안에 사는 작은 함수형 언어입니다:

```yaml
{{- $fullName := include "myapp.fullname" . }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ $fullName }}
spec:
  replicas: {{ .Values.replicas | default 3 }}
  template:
    spec:
      containers:
        - name: app
          image: "{{ .Values.image.repo }}:{{ .Values.image.tag | required "image.tag is required" }}"
          {{- if .Values.resources }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          {{- end }}
```

내재화할 세 패턴:

- **파이프라인(`|`)**은 변환을 체이닝합니다 — `{{ .Values.foo | upper | quote }}`는 `"BAR"`를 생성. Sprig는 수백 개 함수(문자열 조작, 수학, 리스트, dict, 날짜, 암호)를 제공합니다.
- **명명된 템플릿(`define` / `include`)**은 `_helpers.tpl`에 저장된 재사용 가능 스니펫입니다. 관습은 여러 곳에서 표준 레이블 블록을 방출하기 위해 `{{ include "myapp.labels" . }}`. `include`(vs `template`)가 선호되는 이유는 추가로 파이프 가능하기 때문입니다.
- **`required`, `default`, `tpl`**은 안전망입니다 — `default`는 fallback 제공, `required`는 값이 없을 때 큰 소리로 에러, `tpl`은 문자열을 템플릿으로 평가(값 자체에 템플릿이 있을 때 유용).

공백은 악명 높게 까다롭습니다. `{{- ... -}}`는 주변 공백을 트림합니다 — `nindent N`은 다중 줄 블록을 N 칸 들여쓰기. 대부분의 Helm 차트 버그는 유효하지 않은 YAML을 생성하는 공백 버그입니다.

`charts/`의 서브차트는 합성을 허용합니다(예: 당신의 앱 차트가 `redis`와 `postgres` 차트에 의존). 부모로부터의 값은 차트 이름을 키로 흘러 내려갑니다(`redis.enabled: false`는 redis 서브차트를 비활성). 이것이 의존성 관리에 대한 Helm의 답입니다.

### D. Kustomize — 구조적 병합과 패치 대수

Kustomize는 유효한 YAML에서 시작합니다 — 당신의 `base/` 디렉토리에는 실제 배포 가능 매니페스트가 있습니다. 오버레이가 구조적 변환을 적용합니다:

```
base/
  deployment.yaml      # replicas: 1, image: nginx:1.25
  service.yaml
  kustomization.yaml   # 리소스 나열

overlays/prod/
  kustomization.yaml   # 패치 — 5로 스케일, 이미지 태그 변경, prod 레이블 추가
  deployment-patch.yaml
```

세 가지 변환 메커니즘:

**1. Strategic Merge Patch.** 부분 쿠버네티스 매니페스트처럼 보입니다 — Kustomize가 스키마를 알고 지능적으로 병합합니다 — 예: `containers: [{ name: app, image: x }]`를 패치하면, 새 컨테이너를 추가하는 대신 `app`이라는 기존 컨테이너를 수정. 스키마 인식, 흔한 케이스에 직관적.

**2. JSON Patch (RFC 6902).** JSON 경로에 대한 명시적 작업 — `{op: replace, path: /spec/replicas, value: 5}`. 장황하지만 정확 — strategic merge가 변경을 표현할 수 없을 때 필요(예: 인덱스로 배열 요소 편집).

**3. Generator (configMapGenerator, secretGenerator).** 파일, 리터럴, env 파일에서 ConfigMap과 Secret을 빌드. 콘텐츠 해시 접미사(`my-config-h7f4d8`)를 가진 불변 객체 생성 — 콘텐츠가 바뀌면 해시가 바뀌어, 새 config를 가져오는 Pod 재시작을 강제합니다. 이는 "ConfigMap 업데이트가 파드를 재시작하지 않는" 문제를 우아하게 해결합니다.

Kustomize는 잘 합성됩니다 — `base → overlays/staging → overlays/prod-east-1`을 가질 수 있고 각 계층이 더 많은 구체를 추가합니다. 컴포넌트(최신 기능)는 "모니터링 추가"나 "네트워크 정책 추가" 같은 횡단 관심사를 여러 오버레이에 mix-in 할 수 있게 합니다.

멘탈 모델 — **Helm은 매개변수로 템플릿을 렌더링하고, Kustomize는 패치 계층을 합성합니다.** 같은 최종 결과, 매우 다른 사용성.

### 이론에서 아래의 YAML으로

이제 레슨은 이 추상을 적용합니다:

- **섹션 1–2 (Helm 개념, 차트 구조)**는 §B입니다 — 구체적 `Chart.yaml`, `values.yaml`, `templates/`을 가진 패키지와 release 모델.
- **섹션 3 (Helm 템플릿 심층 분석)**은 §C입니다 — Sprig 함수, 파이프라인, 명명된 템플릿, 서브차트.
- **섹션 4 (Helm 훅과 테스트)**는 release 라이프사이클 이벤트를 위한 §B의 상태 머신 확장입니다.
- **섹션 5 (차트 모범 사례)**는 §B와 §C로부터의 운영 가이드라인입니다 — 명명, 레이블, 불변성 우려.
- **섹션 6–7 (Kustomize 기초, 패치)**는 §D입니다 — 베이스 + 오버레이 + 두 패치 스타일.
- **섹션 8 (생성기와 변환기)**는 §D의 콘텐츠 해시 트릭입니다.
- **섹션 9 (Helm vs Kustomize)**는 §A를 운영적으로 만든 것입니다 — 언제 어느 쪽을 고를지.
- **섹션 10 (Helmfile)**은 함께 관리할 Helm release가 많을 때 하는 일입니다.

templating-vs-patching을 근본적 철학적 분리로 보고 나면, 모든 "Helm을 써야 하나 Kustomize를 써야 하나?" 논쟁은 "매개변수화된 패키지(Helm)를 원하나, 합성 가능한 유효 YAML 계층(Kustomize)을 원하나?"로 환원됩니다.

---

## 1. Helm 개념

### 1.1 차트, 릴리스, 리포지토리

| 개념 | 설명 | 비유 |
|------|------|------|
| **차트(Chart)** | Kubernetes 리소스 템플릿의 패키지 | 레시피 |
| **릴리스(Release)** | 차트의 설치된 인스턴스 | 레시피로 만든 요리 |
| **리포지토리(Repository)** | 차트의 모음 | 요리책 |
| **Values** | 차트의 구성 매개변수 | 재료 대체 |

```bash
# Helm 핵심 워크플로우
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# 차트 검색
helm search repo nginx
helm search hub wordpress    # Artifact Hub 검색

# 차트 설치 (릴리스 생성)
helm install my-nginx bitnami/nginx --namespace web --create-namespace

# 릴리스 목록
helm list -A

# 새 values로 릴리스 업그레이드
helm upgrade my-nginx bitnami/nginx --set replicaCount=3

# 이전 리비전으로 롤백
helm rollback my-nginx 1

# 릴리스 제거
helm uninstall my-nginx --namespace web

# 릴리스 히스토리 보기
helm history my-nginx
```

### 1.2 Helm 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│  Helm 3 아키텍처 (Tiller 없음)                             │
│                                                          │
│  ┌──────────┐   helm install   ┌──────────────────┐     │
│  │ helm CLI │─────────────────▶│ Kubernetes API    │     │
│  │          │                  │ Server            │     │
│  │ 1. Read  │                  │                   │     │
│  │ chart    │                  │ 2. 리소스 생성     │     │
│  │          │                  │                   │     │
│  │ 3. Store │                  │                   │     │
│  │ release  │                  │                   │     │
│  │ (Secret) │                  │                   │     │
│  └──────────┘                  └──────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Helm 차트 구조

### 2.1 디렉토리 레이아웃

```
my-app/
├── Chart.yaml              # 차트 메타데이터 (이름, 버전, 의존성)
├── Chart.lock              # 잠긴 의존성 버전
├── values.yaml             # 기본 구성 값
├── values.schema.json      # 값 검증을 위한 JSON 스키마 (선택)
├── charts/                 # 의존성 차트 (서브차트)
├── crds/                   # CustomResourceDefinitions
├── templates/              # Kubernetes 매니페스트 템플릿
│   ├── _helpers.tpl        # 명명된 템플릿 헬퍼
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   ├── serviceaccount.yaml
│   ├── NOTES.txt           # 설치 후 사용자에게 표시
│   ├── hooks/              # 수명주기 훅
│   │   └── db-migrate.yaml
│   └── tests/              # helm test로 실행되는 테스트 파드
│       └── test-connection.yaml
└── .helmignore             # 패키징 시 무시할 파일
```

### 2.2 Chart.yaml

```yaml
# Chart.yaml
apiVersion: v2
name: my-app
description: 내 애플리케이션 Helm 차트
type: application
version: 0.1.0              # 차트 버전 (차트 변경 시 증가)
appVersion: "1.0.0"         # 앱 버전 (앱 변경 시 증가)

dependencies:
  - name: postgresql
    version: "15.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: "18.x.x"
    repository: https://charts.bitnami.com/bitnami
    alias: cache
    condition: redis.enabled
```

### 2.3 Values와 Templates

```yaml
# values.yaml
replicaCount: 2

image:
  repository: my-org/my-app
  tag: ""                    # 비어있으면 appVersion 사용
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: false
  className: nginx
  hosts:
    - host: app.example.com
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 500m
    memory: 256Mi
  requests:
    cpu: 250m
    memory: 128Mi

autoscaling:
  enabled: false
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    database: myapp
    username: myapp

redis:
  enabled: false
```

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-app.fullname" . }}
  labels:
    {{- include "my-app.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "my-app.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "my-app.selectorLabels" . | nindent 8 }}
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
    spec:
      serviceAccountName: {{ include "my-app.serviceAccountName" . }}
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            initialDelaySeconds: 15
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {{ include "my-app.fullname" . }}-db
                  key: url
```

---

## 3. Helm 템플릿 심층 분석

### 3.1 내장 객체

| 객체 | 설명 | 예시 |
|------|------|------|
| `.Values` | `values.yaml`과 `--set` 플래그의 값 | `.Values.image.tag` |
| `.Chart` | `Chart.yaml`의 내용 | `.Chart.Name`, `.Chart.Version` |
| `.Release` | 릴리스 메타데이터 | `.Release.Name`, `.Release.Namespace` |
| `.Template` | 현재 템플릿 정보 | `.Template.Name`, `.Template.BasePath` |
| `.Capabilities` | 클러스터 기능 | `.Capabilities.APIVersions.Has "batch/v1"` |
| `.Files` | 차트의 비템플릿 파일 | `.Files.Get "config.ini"` |

### 3.2 템플릿 함수와 파이프라인

```yaml
# templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "my-app.fullname" . }}
data:
  # 문자열 함수
  app-name: {{ .Chart.Name | upper | quote }}
  description: {{ .Chart.Description | trunc 63 | trimSuffix "-" }}

  # 기본값
  log-level: {{ .Values.logLevel | default "info" | quote }}

  # 유형 변환
  max-connections: {{ .Values.maxConnections | int | quote }}

  # 삼항 연산자
  debug-mode: {{ ternary "true" "false" .Values.debug | quote }}

  # 파일 내용 포함
  nginx.conf: |-
    {{ .Files.Get "files/nginx.conf" | indent 4 | trim }}

  # 맵에 대한 반복
  {{- range $key, $value := .Values.env }}
  {{ $key }}: {{ $value | quote }}
  {{- end }}
```

### 3.3 흐름 제어(Flow Control)

```yaml
# templates/ingress.yaml
{{- if .Values.ingress.enabled -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "my-app.fullname" . }}
  labels:
    {{- include "my-app.labels" . | nindent 4 }}
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
                name: {{ include "my-app.fullname" $ }}
                port:
                  number: {{ $.Values.service.port }}
          {{- end }}
    {{- end }}
{{- end }}
```

### 3.4 명명된 템플릿과 헬퍼

```yaml
# templates/_helpers.tpl

{{/*
차트 이름을 확장합니다.
*/}}
{{- define "my-app.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
정규화된 앱 이름을 생성합니다.
일부 Kubernetes 필드가 제한되므로 63자에서 잘라냅니다.
*/}}
{{- define "my-app.fullname" -}}
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
{{- define "my-app.labels" -}}
helm.sh/chart: {{ include "my-app.chart" . }}
{{ include "my-app.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
셀렉터 레이블
*/}}
{{- define "my-app.selectorLabels" -}}
app.kubernetes.io/name: {{ include "my-app.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
서비스 어카운트 이름
*/}}
{{- define "my-app.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "my-app.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
```

### 3.5 서브차트와 의존성

```bash
# 차트 의존성 다운로드 (Chart.yaml에 정의됨)
helm dependency update ./my-app

# my-app/charts/에 차트 다운로드됨
ls my-app/charts/
# postgresql-15.2.0.tgz  redis-18.1.0.tgz

# 부모 values.yaml을 통해 서브차트 값 오버라이드
```

```yaml
# values.yaml -- 서브차트 값은 의존성 이름 아래에 중첩
postgresql:
  enabled: true
  auth:
    database: myapp
    username: myapp
    password: secret    # 프로덕션에서는 --set 사용, 평문 아님
  primary:
    persistence:
      size: 10Gi

# 별칭 "cache"는 redis 서브차트에 매핑
cache:
  enabled: true
  architecture: standalone
  auth:
    enabled: true
```

---

## 4. Helm 훅과 테스트

### 4.1 훅 유형

Helm 훅은 릴리스 수명주기의 특정 시점에서 실행됩니다.

| 훅 | 시점 | 사용 사례 |
|-----|------|----------|
| `pre-install` | 차트 리소스 설치 전 | 시크릿 생성, 사전 조건 확인 |
| `post-install` | 모든 차트 리소스 설치 후 | 알림, 초기 데이터 로드 |
| `pre-upgrade` | 업그레이드 시작 전 | 데이터베이스 백업 |
| `post-upgrade` | 업그레이드 완료 후 | 마이그레이션 실행 |
| `pre-delete` | 삭제 시작 전 | 데이터 백업 |
| `post-delete` | 삭제 완료 후 | 외부 리소스 정리 |
| `pre-rollback` | 롤백 시작 전 | 현재 상태 스냅샷 |
| `post-rollback` | 롤백 완료 후 | 롤백 성공 확인 |
| `test` | `helm test` 실행 시 | 연결 및 헬스 체크 |

### 4.2 데이터베이스 마이그레이션 훅

```yaml
# templates/hooks/db-migrate.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "my-app.fullname" . }}-migrate
  labels:
    {{- include "my-app.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": post-install,post-upgrade
    "helm.sh/hook-weight": "-5"          # 낮은 가중치가 먼저 실행
    "helm.sh/hook-delete-policy": before-hook-creation
spec:
  backoffLimit: 3
  template:
    metadata:
      labels:
        {{- include "my-app.selectorLabels" . | nindent 8 }}
    spec:
      restartPolicy: Never
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: migrate
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          command: ["./migrate", "--direction=up"]
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {{ include "my-app.fullname" . }}-db
                  key: url
```

```yaml
# templates/hooks/pre-upgrade-backup.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "my-app.fullname" . }}-backup
  annotations:
    "helm.sh/hook": pre-upgrade
    "helm.sh/hook-weight": "-10"
    "helm.sh/hook-delete-policy": hook-succeeded
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: backup
          image: postgres:16
          command:
            - sh
            - -c
            - |
              pg_dump $DATABASE_URL > /backup/dump-$(date +%Y%m%d%H%M%S).sql
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {{ include "my-app.fullname" . }}-db
                  key: url
          volumeMounts:
            - name: backup
              mountPath: /backup
      volumes:
        - name: backup
          persistentVolumeClaim:
            claimName: {{ include "my-app.fullname" . }}-backup
```

### 4.3 Helm 테스트

```yaml
# templates/tests/test-connection.yaml
apiVersion: v1
kind: Pod
metadata:
  name: {{ include "my-app.fullname" . }}-test
  labels:
    {{- include "my-app.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": test
    "helm.sh/hook-delete-policy": before-hook-creation
spec:
  restartPolicy: Never
  containers:
    - name: wget
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "Testing HTTP endpoint..."
          wget -qO- --timeout=5 http://{{ include "my-app.fullname" . }}:{{ .Values.service.port }}/healthz
          echo "Testing database connectivity..."
          wget -qO- --timeout=5 http://{{ include "my-app.fullname" . }}:{{ .Values.service.port }}/readyz
          echo "All tests passed!"
```

```bash
# Helm 테스트 실행
helm test my-release --namespace production
# NAME: my-release
# STATUS: deployed
# TEST SUITE:     my-release-my-app-test
# Last Started:   Mon Jan 15 10:00:00 2024
# Last Completed: Mon Jan 15 10:00:05 2024
# Phase:          Succeeded
```

---

## 5. 차트 개발 모범 사례

```bash
# 기본 스캐폴드에서 새 차트 생성
helm create my-new-chart

# 차트 린트
helm lint ./my-new-chart

# 로컬에서 템플릿 렌더링 (설치 없이)
helm template my-release ./my-new-chart --values custom-values.yaml

# 디버그 출력으로 렌더링
helm template my-release ./my-new-chart --debug

# 클러스터에 대한 dry-run (API 서버에 대해 검증)
helm install my-release ./my-new-chart --dry-run --debug

# 차트 패키징
helm package ./my-new-chart
# my-new-chart-0.1.0.tgz

# OCI 레지스트리에 푸시
helm push my-new-chart-0.1.0.tgz oci://registry.example.com/charts
```

**모범 사례 요약:**

| 사례 | 이유 |
|------|------|
| 모든 레이블/이름 계산에 `_helpers.tpl` 사용 | 단일 진실 소스, DRY |
| `values.schema.json` 추가 | 렌더링 전 값 검증 |
| `NOTES.txt` 포함 | 사용자에게 앱 접근 방법 표시 |
| `values.yaml`에 합리적인 기본값 설정 | 차트가 설정 없이 작동해야 함 |
| `checksum/config` 어노테이션 사용 | ConfigMap 변경 시 파드 자동 재시작 |
| 네임스페이스 하드코딩 금지 | `.Release.Namespace` 사용 |
| `latest`가 아닌 이미지 태그 고정 | 재현 가능한 배포 |
| `nameOverride`와 `fullnameOverride` 지원 | 표준 Helm 규칙 |

---

## 6. Kustomize 기초

Kustomize는 `kubectl`에 내장되어 있으며 Helm과 근본적으로 다른 접근 방식을 취합니다: 템플릿화 대신 유효한 YAML을 오버레이로 패치합니다.

### 6.1 Base와 Overlay

```
my-app/
├── base/                     # 공유 기본 구성
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
└── overlays/
    ├── dev/                  # 개발 오버라이드
    │   ├── kustomization.yaml
    │   ├── replica-patch.yaml
    │   └── env-configmap.yaml
    ├── staging/              # 스테이징 오버라이드
    │   ├── kustomization.yaml
    │   └── replica-patch.yaml
    └── production/           # 프로덕션 오버라이드
        ├── kustomization.yaml
        ├── replica-patch.yaml
        ├── hpa.yaml
        └── resource-patch.yaml
```

### 6.2 Kustomization 파일

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml
  - configmap.yaml

commonLabels:
  app.kubernetes.io/name: my-app
  app.kubernetes.io/managed-by: kustomize
```

```yaml
# base/deployment.yaml (유효하고 배포 가능한 YAML -- 템플릿이 아님)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: app
          image: my-app:latest
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 200m
              memory: 256Mi
```

```yaml
# base/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app
spec:
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 8080
```

```yaml
# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base
  - hpa.yaml                # 프로덕션용 추가 리소스

namespace: production        # 모든 리소스에 네임스페이스 설정

namePrefix: prod-            # 모든 리소스 이름에 접두사

commonLabels:
  env: production

images:
  - name: my-app
    newName: registry.example.com/my-app
    newTag: "3.5.1"

patches:
  - path: replica-patch.yaml
  - path: resource-patch.yaml
```

```bash
# 렌더링된 출력 미리보기
kubectl kustomize overlays/production

# 직접 적용
kubectl apply -k overlays/production

# 또는 kustomize CLI 사용
kustomize build overlays/production | kubectl apply -f -

# 실행 중인 클러스터와 비교
kubectl diff -k overlays/production
```

---

## 7. Kustomize 패치

### 7.1 전략적 병합 패치(Strategic Merge Patches)

전략적 병합 패치는 Kubernetes 인식 병합 전략을 사용하여 패치를 base에 병합합니다.

```yaml
# overlays/production/replica-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app            # base 리소스 이름과 일치해야 함
spec:
  replicas: 5             # 레플리카 오버라이드

# overlays/production/resource-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    spec:
      containers:
        - name: app       # 컨테이너 이름과 일치해야 함
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: "1"
              memory: 1Gi
          env:
            - name: LOG_LEVEL
              value: "warn"
            - name: DB_POOL_SIZE
              value: "20"
```

```yaml
# kustomization.yaml 내 인라인 패치 (별도 파일 불필요)
# overlays/staging/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

namespace: staging

patches:
  - target:
      kind: Deployment
      name: my-app
    patch: |-
      - op: replace
        path: /spec/replicas
        value: 3
```

### 7.2 JSON 패치

JSON 패치(RFC 6902)는 정밀한 작업 기반 수정을 제공합니다.

```yaml
# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

patches:
  # JSON 패치: 사이드카 컨테이너 추가
  - target:
      kind: Deployment
      name: my-app
    patch: |-
      - op: add
        path: /spec/template/spec/containers/-
        value:
          name: log-shipper
          image: fluent/fluent-bit:latest
          resources:
            requests:
              cpu: 50m
              memory: 64Mi
            limits:
              cpu: 100m
              memory: 128Mi

  # JSON 패치: 이미지 풀 정책 교체
  - target:
      kind: Deployment
      name: my-app
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/imagePullPolicy
        value: Always

  # JSON 패치: 필드 제거
  - target:
      kind: Service
      name: my-app
    patch: |-
      - op: remove
        path: /spec/type
```

---

## 8. Kustomize 생성기와 변환기

### 8.1 ConfigMap과 Secret 생성기

생성기는 콘텐츠 기반 해시 접미사가 있는 ConfigMap과 Secret을 생성하여, 구성이 변경될 때 파드가 재시작되도록 합니다.

```yaml
# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

configMapGenerator:
  - name: app-config
    literals:
      - DATABASE_HOST=db.production.svc
      - LOG_LEVEL=warn
      - MAX_CONNECTIONS=100
  - name: nginx-config
    files:
      - files/nginx.conf
    options:
      disableNameSuffixHash: true    # 해시 접미사 추가하지 않음

secretGenerator:
  - name: db-credentials
    literals:
      - username=admin
      - password=secret123     # 프로덕션에서는 SOPS 또는 sealed-secrets 사용
    type: Opaque
  - name: tls-cert
    files:
      - tls.crt=certs/server.crt
      - tls.key=certs/server.key
    type: kubernetes.io/tls
```

```bash
# 생성된 ConfigMap 이름에 해시 접미사 포함
kubectl kustomize overlays/production | grep "name: app-config"
# name: app-config-7h8g9k    <-- 콘텐츠 변경 시 해시 접미사 변경

# Deployment 참조가 자동으로 업데이트됨
# containers:
#   env:
#     - name: DATABASE_HOST
#       valueFrom:
#         configMapKeyRef:
#           name: app-config-7h8g9k    <-- 자동 업데이트
```

### 8.2 변환기(Transformers)

```yaml
# 변환기가 있는 kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

# 모든 리소스에 공통 레이블 추가
commonLabels:
  team: platform
  env: production

# 모든 리소스에 공통 어노테이션 추가
commonAnnotations:
  note: "Managed by Kustomize"

# 모든 리소스에 네임스페이스 설정
namespace: production

# 모든 리소스에 이름 접두사/접미사 추가
namePrefix: prod-
nameSuffix: -v2

# 이미지 참조 변환
images:
  - name: my-app
    newName: gcr.io/my-project/my-app
    newTag: v3.5.1
  - name: nginx
    newName: nginx
    newTag: "1.27"
    digest: sha256:abc123...    # 최대 재현성을 위해 다이제스트로 고정

# 패치를 통한 리소스 제약 추가
patches:
  - target:
      kind: Deployment
    patch: |-
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: not-important
      spec:
        template:
          spec:
            securityContext:
              runAsNonRoot: true
              seccompProfile:
                type: RuntimeDefault
```

---

## 9. Helm vs Kustomize

| 측면 | Helm | Kustomize |
|------|------|-----------|
| **접근 방식** | 템플릿화 (Go 템플릿) | 패칭 (오버레이) |
| **기본 파일** | 템플릿 (유효한 YAML 아님) | 유효한 YAML (그대로 배포 가능) |
| **매개변수화** | `values.yaml`, `--set` 플래그 | 패치, 생성기, 변환기 |
| **패키지 관리** | 차트, 리포지토리, OCI | 패키징 개념 없음 |
| **릴리스 추적** | 내장 (클러스터의 Secret) | 없음 (GitOps 도구 사용) |
| **롤백** | 내장 (`helm rollback`) | 없음 (Git 사용) |
| **수명주기 훅** | Pre/post install/upgrade/delete | 없음 |
| **테스트** | `helm test` | 없음 |
| **학습 곡선** | 높음 (Go 템플릿) | 낮음 (YAML 패칭) |
| **생태계** | 거대 (ArtifactHub, Bitnami) | 작음 |
| **서드파티 소프트웨어** | 최선의 선택 (사전 제작 차트) | 어려움 (base YAML 직접 작성) |
| **GitOps** | 지원 (ArgoCD, Flux) | 네이티브 적합 (ArgoCD, Flux) |
| **kubectl 통합** | 별도 바이너리 | 내장 (`kubectl -k`) |

**Helm을 사용할 때:**
- 서드파티 소프트웨어 설치 (데이터베이스, 모니터링 스택)
- 여러 환경에 걸친 복잡한 매개변수화
- 롤백 기능이 있는 릴리스 관리
- 여러 팀을 위한 공유 차트

**Kustomize를 사용할 때:**
- 애플리케이션별 Kubernetes 매니페스트
- 단순한 환경 변형 (dev/staging/prod)
- Git이 진실의 소스인 GitOps 워크플로우
- 유효한 YAML 작업을 선호하는 팀

**둘 다 함께 사용:**

```bash
# Helm 차트를 렌더링한 후 Kustomize로 커스터마이즈
helm template my-release bitnami/postgresql \
  --values values.yaml \
  --namespace production > base/postgresql.yaml

# 그런 다음 환경별 패치에 Kustomize 오버레이 사용
kubectl apply -k overlays/production
```

---

## 10. Helmfile

Helmfile은 여러 Helm 릴리스를 선언적으로 관리합니다.

```yaml
# helmfile.yaml
repositories:
  - name: bitnami
    url: https://charts.bitnami.com/bitnami
  - name: ingress-nginx
    url: https://kubernetes.github.io/ingress-nginx
  - name: jetstack
    url: https://charts.jetstack.io

environments:
  dev:
    values:
      - environments/dev.yaml
  staging:
    values:
      - environments/staging.yaml
  production:
    values:
      - environments/production.yaml

releases:
  - name: ingress-nginx
    namespace: ingress-nginx
    chart: ingress-nginx/ingress-nginx
    version: 4.10.0
    values:
      - ingress-nginx/values.yaml
    set:
      - name: controller.replicaCount
        value: {{ .Values | get "ingressReplicas" 2 }}

  - name: cert-manager
    namespace: cert-manager
    chart: jetstack/cert-manager
    version: 1.14.0
    values:
      - cert-manager/values.yaml

  - name: postgresql
    namespace: database
    chart: bitnami/postgresql
    version: 15.2.0
    values:
      - postgresql/values.yaml
      - postgresql/{{ .Environment.Name }}.yaml
    secrets:
      - postgresql/secrets.yaml    # SOPS로 암호화

  - name: my-app
    namespace: production
    chart: ./charts/my-app
    values:
      - my-app/values.yaml
      - my-app/{{ .Environment.Name }}.yaml
    needs:
      - database/postgresql         # postgresql을 먼저 설치
      - ingress-nginx/ingress-nginx
```

```bash
# Helmfile 설치
brew install helmfile

# 환경에 대한 모든 릴리스 동기화
helmfile -e production sync

# 적용 전 비교
helmfile -e production diff

# 변경사항 적용
helmfile -e production apply

# 모든 릴리스 삭제
helmfile -e production destroy

# 모든 차트 린트
helmfile -e production lint

# 릴리스 목록
helmfile -e production list
```

---

## 연습문제

### 연습문제 1: Helm 차트 생성

Deployment, Service, ConfigMap, 선택적 Ingress를 포함하는 `web-api`라는 Helm 차트를 생성하세요. 차트는 레플리카 수, 이미지 태그, 환경 변수, Ingress 구성에 대한 값을 받아야 합니다. `helm template`과 `helm lint`로 테스트하세요.

<details><summary>정답 보기</summary>

```bash
# 차트 스캐폴드 생성
helm create web-api

# 불필요한 기본값 제거
rm -rf web-api/templates/tests/test-connection.yaml
```

```yaml
# web-api/values.yaml
replicaCount: 2

image:
  repository: my-org/web-api
  tag: "1.0.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: false
  className: nginx
  hosts:
    - host: api.example.com
      paths:
        - path: /
          pathType: Prefix
  tls: []

config:
  LOG_LEVEL: "info"
  PORT: "8080"
  CORS_ORIGIN: "*"

resources:
  limits:
    cpu: 500m
    memory: 256Mi
  requests:
    cpu: 100m
    memory: 128Mi
```

```yaml
# web-api/templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "web-api.fullname" . }}
  labels:
    {{- include "web-api.labels" . | nindent 4 }}
data:
  {{- range $key, $value := .Values.config }}
  {{ $key }}: {{ $value | quote }}
  {{- end }}
```

```bash
# 차트 린트
helm lint ./web-api
# ==> Linting ./web-api
# [INFO] Chart.yaml: icon is recommended
# 1 chart(s) linted, 0 chart(s) failed

# 템플릿 렌더링
helm template my-api ./web-api --set replicaCount=3 --set ingress.enabled=true

# dry-run 설치
helm install my-api ./web-api --dry-run --debug --namespace api --create-namespace

# 실제 설치
helm install my-api ./web-api --namespace api --create-namespace

# 확인
helm list -n api
kubectl get all -n api
```

</details>

### 연습문제 2: 데이터베이스 마이그레이션을 위한 Helm 훅

`web-api` 차트에 데이터베이스 마이그레이션을 실행하는 pre-upgrade 훅을 추가하세요. 훅은 메인 애플리케이션과 동일한 이미지를 사용하되 다른 명령을 실행해야 합니다. 적절한 hook-delete-policy를 포함하세요.

<details><summary>정답 보기</summary>

```yaml
# web-api/templates/hooks/db-migrate.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "web-api.fullname" . }}-db-migrate
  labels:
    {{- include "web-api.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": pre-upgrade,pre-install
    "helm.sh/hook-weight": "0"
    "helm.sh/hook-delete-policy": before-hook-creation,hook-succeeded
spec:
  backoffLimit: 3
  activeDeadlineSeconds: 300
  template:
    metadata:
      labels:
        {{- include "web-api.selectorLabels" . | nindent 8 }}
        hook: db-migrate
    spec:
      restartPolicy: Never
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: migrate
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          command: ["./web-api", "migrate", "--direction=up"]
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          envFrom:
            - configMapRef:
                name: {{ include "web-api.fullname" . }}
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {{ include "web-api.fullname" . }}-db
                  key: url
                  optional: true
```

```bash
# 업그레이드가 훅을 트리거
helm upgrade my-api ./web-api --set image.tag=1.1.0

# 마이그레이션 잡 상태 확인
kubectl get jobs -n api -l hook=db-migrate
# NAME                        COMPLETIONS   DURATION   AGE
# my-api-web-api-db-migrate   1/1           5s         10s

# 마이그레이션 로그 확인
kubectl logs -n api -l hook=db-migrate
```

</details>

### 연습문제 3: Kustomize 다중 환경 설정

웹 애플리케이션을 위한 Kustomize base와 `dev`, `staging`, `production`용 오버레이를 생성하세요. 각 환경은 다른 레플리카 수, 리소스 제한, 이미지 태그를 가져야 합니다. 프로덕션에는 추가로 HPA를 포함하세요.

<details><summary>정답 보기</summary>

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml

# base/deployment.yaml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
        - name: web
          image: web-app:latest
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 200m
              memory: 256Mi

# base/service.yaml
---
apiVersion: v1
kind: Service
metadata:
  name: web-app
spec:
  selector:
    app: web-app
  ports:
    - port: 80
      targetPort: 8080
```

```yaml
# overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namespace: dev
commonLabels:
  env: dev
images:
  - name: web-app
    newName: registry.example.com/web-app
    newTag: dev-latest

# overlays/staging/kustomization.yaml
---
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namespace: staging
commonLabels:
  env: staging
images:
  - name: web-app
    newName: registry.example.com/web-app
    newTag: "2.0.0-rc1"
patches:
  - target:
      kind: Deployment
      name: web-app
    patch: |-
      - op: replace
        path: /spec/replicas
        value: 2

# overlays/production/kustomization.yaml
---
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
  - hpa.yaml
namespace: production
commonLabels:
  env: production
images:
  - name: web-app
    newName: registry.example.com/web-app
    newTag: "2.0.0"
patches:
  - path: resource-patch.yaml
  - target:
      kind: Deployment
      name: web-app
    patch: |-
      - op: replace
        path: /spec/replicas
        value: 5
```

```yaml
# overlays/production/resource-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  template:
    spec:
      containers:
        - name: web
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: "1"
              memory: 1Gi

# overlays/production/hpa.yaml
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 5
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

```bash
# 각 환경 미리보기
kubectl kustomize overlays/dev
kubectl kustomize overlays/staging
kubectl kustomize overlays/production

# 적용
kubectl apply -k overlays/production
```

</details>

### 연습문제 4: Helmfile 다중 서비스 배포

완전한 애플리케이션 스택을 배포하는 Helmfile을 작성하세요: NGINX Ingress 컨트롤러, PostgreSQL, Redis, 사용자 정의 애플리케이션 차트. `dev`와 `production` 환경 모두에 대해 적절한 값으로 구성하세요.

<details><summary>정답 보기</summary>

```yaml
# helmfile.yaml
repositories:
  - name: bitnami
    url: https://charts.bitnami.com/bitnami
  - name: ingress-nginx
    url: https://kubernetes.github.io/ingress-nginx

environments:
  dev:
    values:
      - env: dev
      - ingressReplicas: 1
      - dbStorage: 1Gi
      - appReplicas: 1
  production:
    values:
      - env: production
      - ingressReplicas: 3
      - dbStorage: 50Gi
      - appReplicas: 5

releases:
  - name: ingress
    namespace: ingress-nginx
    chart: ingress-nginx/ingress-nginx
    version: 4.10.0
    values:
      - controller:
          replicaCount: {{ .Values.ingressReplicas }}
          metrics:
            enabled: true

  - name: postgresql
    namespace: database
    chart: bitnami/postgresql
    version: 15.2.0
    values:
      - auth:
          database: myapp
          username: myapp
          existingSecret: pg-credentials
        primary:
          persistence:
            size: {{ .Values.dbStorage }}

  - name: redis
    namespace: cache
    chart: bitnami/redis
    version: 18.6.0
    values:
      - architecture: standalone
        auth:
          enabled: true
          existingSecret: redis-credentials

  - name: my-app
    namespace: app-{{ .Values.env }}
    chart: ./charts/my-app
    values:
      - replicaCount: {{ .Values.appReplicas }}
        image:
          tag: {{ requiredEnv "APP_VERSION" | quote }}
        config:
          REDIS_HOST: redis-master.cache.svc.cluster.local
          DATABASE_HOST: postgresql.database.svc.cluster.local
    needs:
      - database/postgresql
      - cache/redis
      - ingress-nginx/ingress
```

```bash
# dev 환경 배포
APP_VERSION=1.0.0 helmfile -e dev sync

# 프로덕션 배포
APP_VERSION=1.0.0 helmfile -e production sync

# 적용 전 비교
APP_VERSION=1.1.0 helmfile -e production diff

# 업데이트 적용
APP_VERSION=1.1.0 helmfile -e production apply
```

</details>

### 연습문제 5: Helm 차트를 Kustomize로 마이그레이션

간단한 웹 앱을 위한 기존 Helm 차트(Deployment + Service + Ingress)를 일반 YAML로 렌더링하고 `base`, `dev`, `production` 오버레이가 있는 Kustomize 구조를 생성하세요. 전체 마이그레이션 프로세스를 보여주세요.

<details><summary>정답 보기</summary>

```bash
# 1단계: Helm 차트를 일반 YAML로 렌더링
helm template my-app ./web-api \
  --namespace default \
  --set ingress.enabled=true > rendered.yaml

# 2단계: 개별 파일로 분리
# (yq와 같은 도구를 사용하거나 수동으로)
mkdir -p kustomize/base

# Deployment 추출
helm template my-app ./web-api --show-only templates/deployment.yaml > kustomize/base/deployment.yaml

# Service 추출
helm template my-app ./web-api --show-only templates/service.yaml > kustomize/base/service.yaml

# Ingress 추출
helm template my-app ./web-api --show-only templates/ingress.yaml \
  --set ingress.enabled=true > kustomize/base/ingress.yaml

# 3단계: 렌더링된 YAML 정리
# Helm 관련 레이블 제거 (helm.sh/chart, app.kubernetes.io/managed-by: Helm)
# 릴리스 이름 접두사 제거
# 결과는 깨끗하고 유효한 YAML이어야 함
```

```yaml
# kustomize/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml

commonLabels:
  app.kubernetes.io/name: web-api
```

```yaml
# kustomize/overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namespace: dev
images:
  - name: my-org/web-api
    newTag: dev-latest
patches:
  - target:
      kind: Deployment
      name: web-api
    patch: |-
      - op: replace
        path: /spec/replicas
        value: 1
configMapGenerator:
  - name: web-api-config
    literals:
      - LOG_LEVEL=debug
      - PORT=8080
```

```yaml
# kustomize/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
  - ../../base/ingress.yaml
  - hpa.yaml
namespace: production
images:
  - name: my-org/web-api
    newTag: "1.0.0"
patches:
  - target:
      kind: Deployment
      name: web-api
    patch: |-
      - op: replace
        path: /spec/replicas
        value: 5
  - path: resource-patch.yaml
configMapGenerator:
  - name: web-api-config
    literals:
      - LOG_LEVEL=warn
      - PORT=8080
```

```bash
# 두 환경이 올바르게 렌더링되는지 확인
kubectl kustomize kustomize/overlays/dev
kubectl kustomize kustomize/overlays/production

# 적용
kubectl apply -k kustomize/overlays/dev
kubectl apply -k kustomize/overlays/production

# 원래 Helm 렌더와 비교
diff <(helm template my-app ./web-api --namespace production) \
     <(kubectl kustomize kustomize/overlays/production)
```

</details>

---

**이전**: [CNI와 고급 네트워킹](./08_CNI_and_Advanced_Networking.md) | **다음**: [커스텀 리소스 정의](./10_Custom_Resource_Definitions.md)
