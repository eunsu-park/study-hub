# 09. Helm and Kustomize

**Previous**: [CNI and Advanced Networking](./08_CNI_and_Advanced_Networking.md) | **Next**: [Custom Resource Definitions](./10_Custom_Resource_Definitions.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create, template, and deploy Helm charts with values files and dependency management
2. Use Helm hooks and tests to manage release lifecycle events
3. Build Kustomize overlays with strategic merge patches and JSON patches
4. Compare Helm and Kustomize to choose the right tool for each scenario
5. Manage multi-chart deployments with Helmfile

---

As Kubernetes clusters grow in complexity, managing raw YAML manifests becomes untenable. A single microservice might require a Deployment, Service, ConfigMap, HPA, NetworkPolicy, and ServiceAccount -- multiplied across dozens of services and multiple environments. Helm and Kustomize are the two dominant tools that solve this problem in fundamentally different ways: Helm uses client-side templating with Go templates, while Kustomize uses template-free, overlay-based patching. This lesson covers both tools in depth, from basic usage through advanced patterns, and helps you choose the right approach for your use case.

> **Templating vs Patching:** Helm generates YAML from templates and values -- flexible but can produce hard-to-read templates. Kustomize patches valid YAML with overlays -- simpler but less flexible for parameterization. Many teams use both: Helm for third-party charts, Kustomize for application-specific overlays.

## Table of Contents

- [Theory & Principles](#theory--principles)
- [1. Helm Concepts](#1-helm-concepts)
  - [1.1 Charts, Releases, and Repositories](#11-charts-releases-and-repositories)
  - [1.2 Helm Architecture](#12-helm-architecture)
- [2. Helm Chart Structure](#2-helm-chart-structure)
  - [2.1 Directory Layout](#21-directory-layout)
  - [2.2 Chart.yaml](#22-chartyaml)
  - [2.3 Values and Templates](#23-values-and-templates)
- [3. Helm Templates Deep Dive](#3-helm-templates-deep-dive)
  - [3.1 Built-in Objects](#31-built-in-objects)
  - [3.2 Template Functions and Pipelines](#32-template-functions-and-pipelines)
  - [3.3 Flow Control](#33-flow-control)
  - [3.4 Named Templates and Helpers](#34-named-templates-and-helpers)
  - [3.5 Subcharts and Dependencies](#35-subcharts-and-dependencies)
- [4. Helm Hooks and Tests](#4-helm-hooks-and-tests)
  - [4.1 Hook Types](#41-hook-types)
  - [4.2 Database Migration Hook](#42-database-migration-hook)
  - [4.3 Helm Tests](#43-helm-tests)
- [5. Chart Development Best Practices](#5-chart-development-best-practices)
- [6. Kustomize Basics](#6-kustomize-basics)
  - [6.1 Base and Overlays](#61-base-and-overlays)
  - [6.2 Kustomization File](#62-kustomization-file)
- [7. Kustomize Patches](#7-kustomize-patches)
  - [7.1 Strategic Merge Patches](#71-strategic-merge-patches)
  - [7.2 JSON Patches](#72-json-patches)
- [8. Kustomize Generators and Transformers](#8-kustomize-generators-and-transformers)
  - [8.1 ConfigMap and Secret Generators](#81-configmap-and-secret-generators)
  - [8.2 Transformers](#82-transformers)
- [9. Helm vs Kustomize](#9-helm-vs-kustomize)
- [10. Helmfile](#10-helmfile)
- [Exercises](#exercises)

---

## 1. Helm Concepts

### Theory: Templating vs Patching: Two Philosophies

The same problem — "I need to deploy 50 manifests with environment-specific variations" — has two opposite answers:

**Templating (Helm).** Treat YAML as text. Embed variables and control flow (`{{ .Values.image.tag }}`, `{{- if .Values.ingress.enabled }}`) directly in the source. The renderer (`helm template`) substitutes values and emits final YAML. Pros: maximum expressiveness — any string can be parameterized, you can have conditional sections, loops over lists, computed values via Sprig functions. Cons: the templates are not valid YAML themselves (you cannot lint them as YAML); whitespace is fragile; complex charts become unreadable; you cannot just open the file in an IDE and see what gets deployed.

**Patching (Kustomize).** Treat YAML as structured data. The base is a valid manifest. Overlays are valid YAML patches that merge into the base. The renderer (`kustomize build`) applies the patches and emits final YAML. Pros: all files are valid YAML — IDE highlighting, schema validation, kubectl apply on the base alone all work; no template syntax to learn; predictable composition. Cons: less expressive — you cannot easily say "add this label to every container that has X"; complex transformations require JSON Patch which is its own learning curve; conditional inclusion is awkward.

Neither is universally better. The best teams use both: **Helm for third-party charts** (where you want to consume a vendor-maintained package and just override values), **Kustomize for internal manifests** (where you want to see exactly what is deploying without a templating layer).

A subtle property: Helm's output depends on a Go template engine that may behave differently between Helm versions; Kustomize's output is deterministic from the spec. So Kustomize tends to win in GitOps where the rendered YAML is the source of truth.

### Theory: Helm: A Package Manager With a State Machine

Helm is more than just a templating tool — it is a **package manager**. A Helm `Chart` is a directory with a defined structure (`Chart.yaml` metadata, `values.yaml` defaults, `templates/` directory of templates, optional `charts/` for subcharts). When you `helm install`, three things happen:

1. **Render**: combine templates with values to produce final YAML.
2. **Apply**: kubectl-style apply to the cluster.
3. **Record a Release**: store the rendered manifest, version number, and metadata as a Secret in the target namespace.

The third part is what differentiates Helm from a templating tool. A **Release** is a named installation with a version history:

```
$ helm history my-app
REVISION  STATUS      CHART       APP VERSION  DESCRIPTION
1         superseded  my-app-1.0  v1.0         Install complete
2         superseded  my-app-1.1  v1.1         Upgrade complete
3         deployed    my-app-1.2  v1.2         Upgrade complete
```

Each upgrade stores the entire rendered manifest as a new Secret. `helm rollback my-app 1` re-applies the manifest from revision 1. This is why Helm rollbacks are safe and atomic — they use stored prior state, not template-time recomputation that might have changed.

Hooks (`helm.sh/hook: pre-install`, `post-upgrade`, etc.) extend this state machine — a hook annotation tells Helm to run a Job before the next phase, e.g., "run db-migrate before installing." Failed hooks block the release transition.

### 1.1 Charts, Releases, and Repositories

| Concept | Description | Analogy |
|---------|-------------|---------|
| **Chart** | A package of Kubernetes resource templates | A recipe |
| **Release** | An installed instance of a chart | A meal prepared from the recipe |
| **Repository** | A collection of charts | A cookbook |
| **Values** | Configuration parameters for a chart | Ingredient substitutions |

```bash
# Core Helm workflow
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Search for charts
helm search repo nginx
helm search hub wordpress    # Search Artifact Hub

# Install a chart (creates a release)
helm install my-nginx bitnami/nginx --namespace web --create-namespace

# List releases
helm list -A

# Upgrade a release with new values
helm upgrade my-nginx bitnami/nginx --set replicaCount=3

# Rollback to a previous revision
helm rollback my-nginx 1

# Uninstall a release
helm uninstall my-nginx --namespace web

# View release history
helm history my-nginx
```

### 1.2 Helm Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Helm 3 Architecture (Tiller-less)                       │
│                                                          │
│  ┌──────────┐   helm install   ┌──────────────────┐     │
│  │ helm CLI │─────────────────▶│ Kubernetes API    │     │
│  │          │                  │ Server            │     │
│  │ 1. Read  │                  │                   │     │
│  │    chart │                  │ 2. Store release  │     │
│  │ 2. Render│                  │    as Secret in   │     │
│  │    templates                │    the namespace  │     │
│  │ 3. Send  │                  │                   │     │
│  │    manifests                │ 3. Create         │     │
│  │          │                  │    resources      │     │
│  └──────────┘                  └──────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

```bash
# Helm stores release metadata as Secrets
kubectl get secrets -n web -l "owner=helm"
# NAME                         TYPE                 DATA
# sh.helm.release.v1.my-nginx.v1   helm.sh/release.v1   1
# sh.helm.release.v1.my-nginx.v2   helm.sh/release.v1   1
```

---

## 2. Helm Chart Structure

### 2.1 Directory Layout

```
my-app/
├── Chart.yaml           # Chart metadata (name, version, dependencies)
├── Chart.lock           # Locked dependency versions
├── values.yaml          # Default configuration values
├── values.schema.json   # JSON Schema for values validation (optional)
├── .helmignore          # Files to exclude from packaging
├── templates/           # Kubernetes manifest templates
│   ├── _helpers.tpl     # Named template definitions
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   ├── serviceaccount.yaml
│   ├── configmap.yaml
│   ├── NOTES.txt        # Post-install usage notes
│   └── tests/
│       └── test-connection.yaml
├── charts/              # Dependency charts (vendored)
└── crds/                # Custom Resource Definitions (applied before templates)
```

### 2.2 Chart.yaml

```yaml
apiVersion: v2
name: my-app
description: A Helm chart for My Application
type: application          # "application" or "library"
version: 1.2.0             # Chart version (SemVer)
appVersion: "3.5.1"        # Application version

keywords:
  - web
  - api

maintainers:
  - name: Jane Doe
    email: jane@example.com

dependencies:
  - name: postgresql
    version: "15.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: postgresql.enabled
  - name: redis
    version: "18.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: redis.enabled
    alias: cache
```

### 2.3 Values and Templates

```yaml
# values.yaml
replicaCount: 2

image:
  repository: my-app
  tag: "3.5.1"
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

## 3. Helm Templates Deep Dive

### Theory: Helm Templates: Sprig, Pipelines, Named Templates

Helm uses Go templates plus the [Sprig](http://masterminds.github.io/sprig/) function library. The result is a small functional language living inside YAML:

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

Three patterns to internalize:

- **Pipelines (`|`)** chain transformations: `{{ .Values.foo | upper | quote }}` produces `"BAR"`. Sprig provides hundreds of functions (string manipulation, math, lists, dicts, dates, crypto).
- **Named templates (`define` / `include`)** are reusable snippets stored in `_helpers.tpl`. The convention is `{{ include "myapp.labels" . }}` to emit a standard label block in many places. `include` (vs `template`) is preferred because it can be piped further.
- **`required`, `default`, `tpl`** are the safety net: `default` provides fallback, `required` errors loudly when a value is missing, `tpl` evaluates a string as a template (useful when values themselves contain templates).

Whitespace is famously tricky. `{{- ... -}}` trims surrounding whitespace; `nindent N` indents a multi-line block by N spaces. Most Helm chart bugs are whitespace bugs producing invalid YAML.

Subcharts in `charts/` allow composition (e.g., your app chart depends on `redis` and `postgres` charts). Values from the parent flow down via the chart name as a key (`redis.enabled: false` disables the redis subchart). This is the Helm answer to dependency management.

### 3.1 Built-in Objects

| Object | Description | Example |
|--------|-------------|---------|
| `.Values` | Values from `values.yaml` and `--set` flags | `.Values.image.tag` |
| `.Chart` | Contents of `Chart.yaml` | `.Chart.Name`, `.Chart.Version` |
| `.Release` | Release metadata | `.Release.Name`, `.Release.Namespace` |
| `.Template` | Current template info | `.Template.Name`, `.Template.BasePath` |
| `.Capabilities` | Cluster capabilities | `.Capabilities.APIVersions.Has "batch/v1"` |
| `.Files` | Non-template files in the chart | `.Files.Get "config.ini"` |

### 3.2 Template Functions and Pipelines

```yaml
# templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "my-app.fullname" . }}
data:
  # String functions
  app-name: {{ .Chart.Name | upper | quote }}
  description: {{ .Chart.Description | trunc 63 | trimSuffix "-" }}

  # Default values
  log-level: {{ .Values.logLevel | default "info" | quote }}

  # Type conversion
  max-connections: {{ .Values.maxConnections | int | quote }}

  # Ternary
  debug-mode: {{ ternary "true" "false" .Values.debug | quote }}

  # Include file contents
  nginx.conf: |-
    {{ .Files.Get "files/nginx.conf" | indent 4 | trim }}

  # Range over a map
  {{- range $key, $value := .Values.env }}
  {{ $key }}: {{ $value | quote }}
  {{- end }}
```

### 3.3 Flow Control

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

### 3.4 Named Templates and Helpers

```yaml
# templates/_helpers.tpl

{{/*
Expand the name of the chart.
*/}}
{{- define "my-app.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a fully qualified app name.
Truncate at 63 chars because some Kubernetes fields are limited.
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
Common labels
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
Selector labels
*/}}
{{- define "my-app.selectorLabels" -}}
app.kubernetes.io/name: {{ include "my-app.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account name
*/}}
{{- define "my-app.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "my-app.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
```

### 3.5 Subcharts and Dependencies

```bash
# Download chart dependencies (defined in Chart.yaml)
helm dependency update ./my-app

# This downloads charts to my-app/charts/
ls my-app/charts/
# postgresql-15.2.0.tgz  redis-18.1.0.tgz

# Override subchart values through parent values.yaml
```

```yaml
# values.yaml -- subchart values are nested under the dependency name
postgresql:
  enabled: true
  auth:
    database: myapp
    username: myapp
    password: secret    # Use --set in production, not plaintext
  primary:
    persistence:
      size: 10Gi

# Alias "cache" maps to the redis subchart
cache:
  enabled: true
  architecture: standalone
  auth:
    enabled: true
```

---

## 4. Helm Hooks and Tests

### 4.1 Hook Types

Helm hooks execute at specific points in a release lifecycle.

| Hook | When | Use Case |
|------|------|----------|
| `pre-install` | Before any chart resources are installed | Create secrets, check prerequisites |
| `post-install` | After all chart resources are installed | Notifications, initial data load |
| `pre-upgrade` | Before upgrade begins | Database backup |
| `post-upgrade` | After upgrade completes | Run migrations |
| `pre-delete` | Before deletion begins | Backup data |
| `post-delete` | After deletion completes | Clean up external resources |
| `pre-rollback` | Before rollback begins | Snapshot current state |
| `post-rollback` | After rollback completes | Verify rollback succeeded |
| `test` | When `helm test` is run | Connectivity and health checks |

### 4.2 Database Migration Hook

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
    "helm.sh/hook-weight": "-5"          # Lower weight runs first
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

### 4.3 Helm Tests

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
# Run Helm tests
helm test my-release --namespace production
# NAME: my-release
# STATUS: deployed
# TEST SUITE:     my-release-my-app-test
# Last Started:   Mon Jan 15 10:00:00 2024
# Last Completed: Mon Jan 15 10:00:05 2024
# Phase:          Succeeded
```

---

## 5. Chart Development Best Practices

```bash
# Create a new chart from the default scaffold
helm create my-new-chart

# Lint the chart
helm lint ./my-new-chart

# Render templates locally (without installing)
helm template my-release ./my-new-chart --values custom-values.yaml

# Render with debug output
helm template my-release ./my-new-chart --debug

# Dry-run against the cluster (validates against API server)
helm install my-release ./my-new-chart --dry-run --debug

# Package the chart
helm package ./my-new-chart
# my-new-chart-0.1.0.tgz

# Push to an OCI registry
helm push my-new-chart-0.1.0.tgz oci://registry.example.com/charts
```

**Best practices summary:**

| Practice | Rationale |
|----------|-----------|
| Use `_helpers.tpl` for all label/name computation | Single source of truth, DRY |
| Add `values.schema.json` | Validate values before rendering |
| Include `NOTES.txt` | Show users how to access the app |
| Set sensible defaults in `values.yaml` | Chart should work with zero configuration |
| Use `checksum/config` annotation | Auto-restart pods on ConfigMap changes |
| Never hardcode namespaces | Use `.Release.Namespace` |
| Pin image tags, not `latest` | Reproducible deployments |
| Support `nameOverride` and `fullnameOverride` | Standard Helm convention |

---

## 6. Kustomize Basics

### Theory: Kustomize: Structured Merge and Patch Algebra

Kustomize starts from valid YAML — your `base/` directory contains real, deployable manifests. Overlays apply structured transformations:

```
base/
  deployment.yaml      # replicas: 1, image: nginx:1.25
  service.yaml
  kustomization.yaml   # lists the resources

overlays/prod/
  kustomization.yaml   # patches: scale to 5, change image tag, add prod labels
  deployment-patch.yaml
```

Three transformation mechanisms:

**1. Strategic Merge Patch.** Looks like a partial Kubernetes manifest; Kustomize knows the schema and merges intelligently — e.g., if you patch `containers: [{ name: app, image: x }]`, it modifies the existing container named `app` rather than appending a new one. Schema-aware, intuitive for the common case.

**2. JSON Patch (RFC 6902).** Explicit operations on JSON paths: `{op: replace, path: /spec/replicas, value: 5}`. Verbose but precise; required when strategic merge cannot express the change (e.g., editing an array element by index).

**3. Generators (configMapGenerator, secretGenerator).** Build ConfigMaps and Secrets from files, literals, or env files. Produces immutable objects with content-hash suffixes (`my-config-h7f4d8`); when content changes, the hash changes, forcing a Pod restart that picks up the new config. This solves the "ConfigMap update doesn't restart pods" problem elegantly.

Kustomize composes well — you can have `base → overlays/staging → overlays/prod-east-1` with each layer adding more specifics. Components (newer feature) let you mix-in cross-cutting concerns like "add monitoring" or "add network policies" across multiple overlays.

The mental model: **Helm renders templates with parameters; Kustomize composes layers of patches.** Same end result, very different ergonomics.

Kustomize is built into `kubectl` and takes a fundamentally different approach from Helm: instead of templating, it patches valid YAML with overlays.

### 6.1 Base and Overlays

```
my-app/
├── base/                     # Shared base configuration
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
└── overlays/
    ├── dev/                  # Development overrides
    │   ├── kustomization.yaml
    │   ├── replica-patch.yaml
    │   └── env-configmap.yaml
    ├── staging/              # Staging overrides
    │   ├── kustomization.yaml
    │   └── replica-patch.yaml
    └── production/           # Production overrides
        ├── kustomization.yaml
        ├── replica-patch.yaml
        ├── hpa.yaml
        └── resource-patch.yaml
```

### 6.2 Kustomization File

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
# base/deployment.yaml (valid, deployable YAML -- not a template)
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
  - hpa.yaml                # Additional resources for production

namespace: production        # Set namespace for all resources

namePrefix: prod-            # Prefix all resource names

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
# Preview the rendered output
kubectl kustomize overlays/production

# Apply directly
kubectl apply -k overlays/production

# Or using kustomize CLI
kustomize build overlays/production | kubectl apply -f -

# Diff against running cluster
kubectl diff -k overlays/production
```

---

## 7. Kustomize Patches

### 7.1 Strategic Merge Patches

Strategic merge patches merge the patch into the base using Kubernetes-aware merge strategies.

```yaml
# overlays/production/replica-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app            # Must match the base resource name
spec:
  replicas: 5             # Override replicas

# overlays/production/resource-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    spec:
      containers:
        - name: app       # Must match container name
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
# Inline patch in kustomization.yaml (no separate file needed)
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

### 7.2 JSON Patches

JSON patches (RFC 6902) provide precise, operation-based modifications.

```yaml
# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

patches:
  # JSON patch: add a sidecar container
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

  # JSON patch: replace the image pull policy
  - target:
      kind: Deployment
      name: my-app
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/imagePullPolicy
        value: Always

  # JSON patch: remove a field
  - target:
      kind: Service
      name: my-app
    patch: |-
      - op: remove
        path: /spec/type
```

---

## 8. Kustomize Generators and Transformers

### 8.1 ConfigMap and Secret Generators

Generators create ConfigMaps and Secrets with content-based hash suffixes, ensuring pods are restarted when config changes.

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
      disableNameSuffixHash: true    # Don't add hash suffix

secretGenerator:
  - name: db-credentials
    literals:
      - username=admin
      - password=secret123     # Use SOPS or sealed-secrets in production
    type: Opaque
  - name: tls-cert
    files:
      - tls.crt=certs/server.crt
      - tls.key=certs/server.key
    type: kubernetes.io/tls
```

```bash
# Generated ConfigMap name includes a hash suffix
kubectl kustomize overlays/production | grep "name: app-config"
# name: app-config-7h8g9k    <-- hash suffix changes when content changes

# Deployment references are automatically updated
# containers:
#   env:
#     - name: DATABASE_HOST
#       valueFrom:
#         configMapKeyRef:
#           name: app-config-7h8g9k    <-- updated automatically
```

### 8.2 Transformers

```yaml
# kustomization.yaml with transformers
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

# Add common labels to all resources
commonLabels:
  team: platform
  env: production

# Add common annotations to all resources
commonAnnotations:
  note: "Managed by Kustomize"

# Set namespace for all resources
namespace: production

# Add name prefix/suffix to all resources
namePrefix: prod-
nameSuffix: -v2

# Transform image references
images:
  - name: my-app
    newName: gcr.io/my-project/my-app
    newTag: v3.5.1
  - name: nginx
    newName: nginx
    newTag: "1.27"
    digest: sha256:abc123...    # Pin by digest for maximum reproducibility

# Add resource constraints via patches
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

| Aspect | Helm | Kustomize |
|--------|------|-----------|
| **Approach** | Templating (Go templates) | Patching (overlays) |
| **Base files** | Templates (not valid YAML) | Valid YAML (deployable as-is) |
| **Parameterization** | `values.yaml`, `--set` flags | Patches, generators, transformers |
| **Package management** | Charts, repositories, OCI | No packaging concept |
| **Release tracking** | Built-in (Secrets in cluster) | None (use GitOps tools) |
| **Rollback** | Built-in (`helm rollback`) | None (use Git) |
| **Lifecycle hooks** | Pre/post install/upgrade/delete | None |
| **Testing** | `helm test` | None |
| **Learning curve** | Higher (Go templates) | Lower (YAML patching) |
| **Ecosystem** | Huge (ArtifactHub, Bitnami) | Smaller |
| **Third-party software** | Best choice (pre-made charts) | Harder (must write base YAML) |
| **GitOps** | Supported (ArgoCD, Flux) | Native fit (ArgoCD, Flux) |
| **kubectl integration** | Separate binary | Built-in (`kubectl -k`) |

**When to use Helm:**
- Installing third-party software (databases, monitoring stacks)
- Complex parameterization across many environments
- Release management with rollback capability
- Shared charts for multiple teams

**When to use Kustomize:**
- Application-specific Kubernetes manifests
- Simple environment variations (dev/staging/prod)
- GitOps workflows where Git is the source of truth
- Teams that prefer working with valid YAML

**Use both together:**

```bash
# Render Helm chart, then customize with Kustomize
helm template my-release bitnami/postgresql \
  --values values.yaml \
  --namespace production > base/postgresql.yaml

# Then use Kustomize overlays for environment-specific patches
kubectl apply -k overlays/production
```

---

## 10. Helmfile

Helmfile manages multiple Helm releases declaratively.

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
      - postgresql/secrets.yaml    # Encrypted with SOPS

  - name: my-app
    namespace: production
    chart: ./charts/my-app
    values:
      - my-app/values.yaml
      - my-app/{{ .Environment.Name }}.yaml
    needs:
      - database/postgresql         # Install postgresql first
      - ingress-nginx/ingress-nginx
```

```bash
# Install Helmfile
brew install helmfile

# Sync all releases for an environment
helmfile -e production sync

# Diff before applying
helmfile -e production diff

# Apply changes
helmfile -e production apply

# Destroy all releases
helmfile -e production destroy

# Lint all charts
helmfile -e production lint

# List releases
helmfile -e production list
```

---

## Exercises

### Exercise 1: Create a Helm Chart

Create a Helm chart called `web-api` that includes a Deployment, Service, ConfigMap, and optional Ingress. The chart should accept values for replica count, image tag, environment variables, and Ingress configuration. Test it with `helm template` and `helm lint`.

<details><summary>Show Answer</summary>

```bash
# Create the chart scaffold
helm create web-api

# Remove unnecessary defaults
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
# Lint the chart
helm lint ./web-api
# ==> Linting ./web-api
# [INFO] Chart.yaml: icon is recommended
# 1 chart(s) linted, 0 chart(s) failed

# Render templates
helm template my-api ./web-api --set replicaCount=3 --set ingress.enabled=true

# Dry-run install
helm install my-api ./web-api --dry-run --debug --namespace api --create-namespace

# Install for real
helm install my-api ./web-api --namespace api --create-namespace

# Verify
helm list -n api
kubectl get all -n api
```

</details>

### Exercise 2: Helm Hooks for Database Migration

Add a pre-upgrade hook to the `web-api` chart that runs database migrations. The hook should use the same image as the main application but execute a different command. Include proper hook-delete-policy.

<details><summary>Show Answer</summary>

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
# Upgrade triggers the hook
helm upgrade my-api ./web-api --set image.tag=1.1.0

# Check migration job status
kubectl get jobs -n api -l hook=db-migrate
# NAME                        COMPLETIONS   DURATION   AGE
# my-api-web-api-db-migrate   1/1           5s         10s

# View migration logs
kubectl logs -n api -l hook=db-migrate
```

</details>

### Exercise 3: Kustomize Multi-Environment Setup

Create a Kustomize base for a web application and overlays for `dev`, `staging`, and `production`. Each environment should have different replica counts, resource limits, and image tags. Production should additionally include an HPA.

<details><summary>Show Answer</summary>

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
# Preview each environment
kubectl kustomize overlays/dev
kubectl kustomize overlays/staging
kubectl kustomize overlays/production

# Apply
kubectl apply -k overlays/production
```

</details>

### Exercise 4: Helmfile Multi-Service Deployment

Write a Helmfile that deploys a complete application stack: NGINX Ingress controller, PostgreSQL, Redis, and a custom application chart. Configure it for both `dev` and `production` environments with appropriate values for each.

<details><summary>Show Answer</summary>

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
# Deploy dev environment
APP_VERSION=1.0.0 helmfile -e dev sync

# Deploy production
APP_VERSION=1.0.0 helmfile -e production sync

# Diff before applying
APP_VERSION=1.1.0 helmfile -e production diff

# Apply update
APP_VERSION=1.1.0 helmfile -e production apply
```

</details>

### Exercise 5: Migrate Helm Chart to Kustomize

Given an existing Helm chart for a simple web app (Deployment + Service + Ingress), render it to plain YAML and create a Kustomize structure with `base`, `dev`, and `production` overlays. Show the complete migration process.

<details><summary>Show Answer</summary>

```bash
# Step 1: Render the Helm chart to plain YAML
helm template my-app ./web-api \
  --namespace default \
  --set ingress.enabled=true > rendered.yaml

# Step 2: Split into individual files
# (You can use a tool like yq or do it manually)
mkdir -p kustomize/base

# Extract Deployment
helm template my-app ./web-api --show-only templates/deployment.yaml > kustomize/base/deployment.yaml

# Extract Service
helm template my-app ./web-api --show-only templates/service.yaml > kustomize/base/service.yaml

# Extract Ingress
helm template my-app ./web-api --show-only templates/ingress.yaml \
  --set ingress.enabled=true > kustomize/base/ingress.yaml

# Step 3: Clean up rendered YAML
# Remove Helm-specific labels (helm.sh/chart, app.kubernetes.io/managed-by: Helm)
# Remove release-name prefixes from resource names
# Result should be clean, valid YAML
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
# Verify both environments render correctly
kubectl kustomize kustomize/overlays/dev
kubectl kustomize kustomize/overlays/production

# Apply
kubectl apply -k kustomize/overlays/dev
kubectl apply -k kustomize/overlays/production

# Compare output with the original Helm render
diff <(helm template my-app ./web-api --namespace production) \
     <(kubectl kustomize kustomize/overlays/production)
```

</details>

---

**Previous**: [CNI and Advanced Networking](./08_CNI_and_Advanced_Networking.md) | **Next**: [Custom Resource Definitions](./10_Custom_Resource_Definitions.md)
