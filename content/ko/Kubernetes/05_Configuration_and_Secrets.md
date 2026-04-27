# 05. 구성 관리와 시크릿(Configuration and Secrets)

**이전**: [스토리지와 영속성](./04_Storage_and_Persistence.md) | **다음**: [RBAC와 보안](./06_RBAC_and_Security.md)

## 학습 목표

- 여러 접근 방식(리터럴, 파일, 디렉토리)을 사용하여 ConfigMap을 생성하고 관리할 수 있다
- Secret 유형, 인코딩, 안전한 마운트 패턴을 이해할 수 있다
- 성능과 안전성을 위해 불변(immutable) ConfigMap과 Secret을 구현할 수 있다
- 외부 시크릿 관리 시스템(External Secrets Operator, Sealed Secrets, Vault)을 통합할 수 있다
- 다중 환경 배포를 위한 구성 모범 사례를 적용할 수 있다

---

구성 관리(Configuration Management)는 Kubernetes에서 핵심적인 운영 관심사입니다. 애플리케이션은 데이터베이스 URL, 기능 플래그(feature flag), API 키, TLS 인증서 등이 필요하며, 이 모든 것은 컨테이너 이미지와 별도로 관리해야 합니다. Kubernetes는 민감하지 않은 데이터를 위한 ConfigMap과 민감한 데이터를 위한 Secret을 제공하지만, 프로덕션 환경에서는 적절한 보안을 위해 외부 시크릿 관리 시스템이 필요한 경우가 많습니다.

## 목차

1. [ConfigMap](#1-configmap)
2. [시크릿(Secrets)](#2-시크릿secrets)
3. [불변 ConfigMap과 Secret](#3-불변-configmap과-secret)
4. [External Secrets Operator](#4-external-secrets-operator)
5. [Sealed Secrets](#5-sealed-secrets)
6. [HashiCorp Vault 통합](#6-hashicorp-vault-통합)
7. [시크릿 교체 패턴](#7-시크릿-교체-패턴)
8. [구성 모범 사례](#8-구성-모범-사례)
9. [환경별 구성](#9-환경별-구성)
10. [저장 시 시크릿 암호화(EncryptionConfiguration)](#10-저장-시-시크릿-암호화encryptionconfiguration)
11. [연습문제](#연습문제)

---

## 1. ConfigMap

ConfigMap은 비밀이 아닌 구성 데이터를 키-값 쌍으로 저장합니다. ConfigMap은 구성을 컨테이너 이미지에서 분리하여 애플리케이션을 이식 가능하게 만듭니다.

### 1.1 ConfigMap 생성

**리터럴 값에서 생성:**

```bash
kubectl create configmap app-config \
  --from-literal=DATABASE_HOST=postgres.default.svc \
  --from-literal=DATABASE_PORT=5432 \
  --from-literal=LOG_LEVEL=info
```

```bash
# ConfigMap 조회
kubectl get configmap app-config -o yaml
```

```
data:
  DATABASE_HOST: postgres.default.svc
  DATABASE_PORT: "5432"
  LOG_LEVEL: info
```

**파일에서 생성:**

```bash
# 설정 파일 생성
cat > /tmp/app.properties <<EOF
database.host=postgres.default.svc
database.port=5432
log.level=info
max.connections=100
EOF

kubectl create configmap app-config --from-file=/tmp/app.properties
# Key = 파일명 (app.properties), Value = 파일 내용
```

**디렉토리에서 생성:**

```bash
# 디렉토리의 모든 파일이 키-값 쌍이 됨
mkdir -p /tmp/config
echo "postgres.default.svc" > /tmp/config/database_host
echo "5432" > /tmp/config/database_port

kubectl create configmap app-config --from-file=/tmp/config/
# Keys: database_host, database_port
```

**YAML 매니페스트에서 생성:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: default
data:
  # 단순 키-값 쌍
  DATABASE_HOST: "postgres.default.svc"
  DATABASE_PORT: "5432"
  LOG_LEVEL: "info"
  MAX_CONNECTIONS: "100"

  # 여러 줄 구성 파일
  nginx.conf: |
    server {
        listen 80;
        server_name localhost;
        location / {
            proxy_pass http://backend:8080;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        location /health {
            return 200 'OK';
            add_header Content-Type text/plain;
        }
    }

  # YAML 형식의 애플리케이션 구성
  config.yaml: |
    server:
      port: 8080
      host: 0.0.0.0
    database:
      host: postgres.default.svc
      port: 5432
      pool_size: 20
    logging:
      level: info
      format: json
```

### 1.2 ConfigMap을 환경 변수로 사용

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-config
spec:
  containers:
    - name: app
      image: my-app:v1.0
      env:
        # 개별 키 참조
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: DATABASE_HOST
        - name: DB_PORT
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: DATABASE_PORT
      # 모든 키를 한 번에 로드
      envFrom:
        - configMapRef:
            name: app-config
            optional: false
```

### 1.3 ConfigMap을 볼륨으로 마운트

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-config-volume
spec:
  containers:
    - name: app
      image: my-app:v1.0
      volumeMounts:
        - name: config-volume
          mountPath: /etc/app/config
          readOnly: true
        # 특정 키만 특정 파일로 마운트
        - name: nginx-config
          mountPath: /etc/nginx/nginx.conf
          subPath: nginx.conf
          readOnly: true
  volumes:
    - name: config-volume
      configMap:
        name: app-config
    - name: nginx-config
      configMap:
        name: app-config
        items:
          - key: nginx.conf
            path: nginx.conf
            mode: 0644
```

### 1.4 ConfigMap 업데이트

```bash
# ConfigMap 업데이트
kubectl edit configmap app-config
# 또는:
kubectl create configmap app-config --from-literal=LOG_LEVEL=debug --dry-run=client -o yaml | kubectl apply -f -

# 마운트된 볼륨에 약 1분 이내에 변경사항이 반영됨
# (kubelet의 --sync-frequency 플래그로 설정 가능, 기본값: 1분)

# 참고: 환경 변수는 업데이트되지 않음. 파드를 재시작해야 함.
# 참고: subPath 마운트는 업데이트되지 않음. 전체 볼륨 마운트만 자동 업데이트됨.
```

### 1.5 ConfigMap 크기 제한

- 최대 크기: ConfigMap당 **1 MiB** (1,048,576 바이트)
- 더 큰 구성이 필요한 경우, PersistentVolume에서 마운트하거나 init 컨테이너(init container)를 사용하여 구성을 가져오는 것을 고려하세요

---

## 2. 시크릿(Secrets)

### 이론: 동일한 객체 형태, 다른 저장 시맨틱

두 객체 모두 다음과 같습니다:

```yaml
apiVersion: v1
kind: ConfigMap   # 또는 Secret
metadata:
  name: my-config
data:             # Secret: stringData도 가능 (자동 base64)
  key1: value1
  key2: value2
```

차이는:

- **인코딩.** Secret 값은 API 표현에서 base64 인코딩됩니다. **이는 암호화가 아닙니다** — `kubectl get secret -o yaml` 권한이 있는 누구나 base64를 되돌릴 수 있습니다. 인코딩은 바이너리 키(TLS 인증서, JKS 파일)를 YAML로 깔끔히 운반하기 위함입니다.
- **etcd 암호화.** 프로덕션 클러스터는 `EncryptionConfiguration`(6강 §10)을 구성하여 Secret(과 선택적으로 다른 리소스)이 etcd 디스크에 닿기 전에 암호화합니다. 없으면 etcd 백업을 가진 누구든 모든 시크릿을 평문으로 가집니다.
- **RBAC 기본값.** 권장 클러스터 RBAC는 ConfigMap 읽기 권한보다 Secret 읽기 권한을 더 엄격히 제한합니다 — 많은 내장 role이 ConfigMap은 list할 수 있지만 Secret은 안 됩니다.
- **로깅 위생.** kubectl, audit log, 대부분의 컨트롤러는 Secret 내용을 redact(`****`)하지만 ConfigMap 내용은 그렇지 않습니다. 이는 강제보다는 관습이지만, 일관됩니다.
- **크기 제한.** 둘 다 객체당 약 1MB로 제한됩니다(etcd 값 크기 제한). 큰 구성은 ConfigMap이 아니라 볼륨에 둡니다.

따라서 값이 새는 것이 보안 사고일 때 Secret을, 새는 것이 그저 부끄러운 일일 때 ConfigMap을 선택하세요.

### 이론: 세 가지 주입 메커니즘

데이터가 API에 들어간 뒤, 프로세스로 전달하는 방법은 세 가지입니다. 각각 다른 업데이트 시맨틱을 가집니다:

**1. 환경 변수(`envFrom` / `env.valueFrom`).** 컨테이너 시작 시점에 읽어 프로세스의 `environ`에 구워집니다. **절대 갱신되지 않습니다** — ConfigMap이 바뀌어도 프로세스는 재시작 전까지 옛 값을 봅니다. 정적 구성(데이터베이스 호스트명)에는 적합하지만, 재시작 없이 회전하고 싶은 것에는 치명적입니다.

**2. 명령 인자(command-line arguments).** 동일 — 시작 시 치환되고 절대 갱신되지 않습니다. 모든 구성을 CLI 플래그로 읽는 도구에 사용됩니다.

**3. Projected 볼륨(`volumeMounts`로 `configMap` / `secret` / `projected`).** kubelet이 API 서버의 객체 뷰로부터 채우는 tmpfs 마운트. **kubelet은 기저 ConfigMap/Secret이 변경될 때 이 볼륨을 주기적으로(기본 약 60초) 갱신합니다.** 변경 사항을 반영하려면 애플리케이션이 파일을 다시 읽어야 합니다 — nginx(`-s reload`) 같은 것에는 적합하지만, 부팅 시에만 읽는 것에는 부적합.

주입 메커니즘은 "재시작 없이 이 시크릿을 회전하라"는 요구사항과 3개월 전에 작성한 YAML 사이의 숨은 결합입니다.

미묘한 점 — 여러 ConfigMap/Secret을 `projected`로 마운트하면, kubelet은 이들을 단일 원자적 심볼릭 링크 swap으로 씁니다. Reader는 전체 옛 버전이나 전체 새 버전 중 하나만 봅니다 — 절반 갱신된 상태는 절대 보지 않습니다. 이것이 projected 모드에서 hot reload를 안전하게 만듭니다.

시크릿(Secret)은 비밀번호, 토큰, TLS 인증서와 같은 민감한 데이터를 저장합니다. ConfigMap과 유사하지만 추가적인 보안 고려사항이 있습니다.

### 2.1 시크릿 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| `Opaque` | 임의의 사용자 정의 데이터 (기본값) | API 키, 비밀번호 |
| `kubernetes.io/tls` | TLS 인증서와 키 | HTTPS 인증서 |
| `kubernetes.io/dockerconfigjson` | Docker 레지스트리 자격 증명 | 이미지 풀 시크릿(image pull secret) |
| `kubernetes.io/basic-auth` | 기본 인증(Basic Authentication) | 사용자명/비밀번호 |
| `kubernetes.io/ssh-auth` | SSH 개인 키 | Git SSH 키 |
| `kubernetes.io/service-account-token` | ServiceAccount 토큰 | 자동 생성 |

### 2.2 시크릿 생성

**Opaque 시크릿:**

```bash
kubectl create secret generic db-credentials \
  --from-literal=username=admin \
  --from-literal=password='S3cur3P@ss!'
```

**TLS 시크릿:**

```bash
kubectl create secret tls my-tls \
  --cert=./tls.crt \
  --key=./tls.key
```

**Docker 레지스트리 시크릿:**

```bash
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=myuser \
  --docker-password=ghp_xxxxxxxxxxxx \
  --docker-email=user@example.com
```

**YAML 매니페스트에서 생성:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
data:
  # 값은 반드시 base64로 인코딩해야 함
  username: YWRtaW4=              # echo -n "admin" | base64
  password: UzNjdXIzUEBzcyE=    # echo -n "S3cur3P@ss!" | base64
---
# 대안: 평문을 위한 stringData 사용 (자동 인코딩)
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials-plain
type: Opaque
stringData:
  username: admin
  password: "S3cur3P@ss!"
  connection-string: "postgresql://admin:S3cur3P@ss!@postgres:5432/mydb"
```

> **중요**: base64는 인코딩이지 암호화가 아닙니다. Secret 객체에 접근할 수 있는 누구나 값을 디코딩할 수 있습니다. 기본적으로 Secret은 ConfigMap보다 약간 더 안전할 뿐입니다.

### 2.3 시크릿 사용

**환경 변수로 사용:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-secrets
spec:
  containers:
    - name: app
      image: my-app:v1.0
      env:
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
      # 모든 키 로드
      envFrom:
        - secretRef:
            name: db-credentials
            optional: false
```

**볼륨 마운트로 사용:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-tls
spec:
  containers:
    - name: nginx
      image: nginx:1.25
      volumeMounts:
        - name: tls-certs
          mountPath: /etc/nginx/ssl
          readOnly: true
        - name: db-creds
          mountPath: /etc/secrets
          readOnly: true
  volumes:
    - name: tls-certs
      secret:
        secretName: my-tls
        defaultMode: 0400          # 제한적 권한
    - name: db-creds
      secret:
        secretName: db-credentials
        items:
          - key: username
            path: db-user          # /etc/secrets/db-user
          - key: password
            path: db-pass          # /etc/secrets/db-pass
            mode: 0400
```

### 2.4 이미지 풀 시크릿(Image Pull Secrets)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: private-registry-pod
spec:
  imagePullSecrets:
    - name: regcred              # Docker 레지스트리 시크릿
  containers:
    - name: app
      image: ghcr.io/myorg/my-app:v1.0
```

이미지 풀 시크릿을 ServiceAccount에 연결하여 모든 파드가 자동으로 사용하도록 할 수 있습니다:

```bash
kubectl patch serviceaccount default \
  -p '{"imagePullSecrets": [{"name": "regcred"}]}'
```

### 2.5 시크릿 보안 고려사항

```
기본 Kubernetes Secret 보안:
├── etcd에 저장 (base64 인코딩, 기본적으로 암호화되지 않음)
├── TLS를 통해 전송 (API 서버 ↔ kubelet)
├── Secret에 대한 RBAC 접근 권한이 있는 누구나 접근 가능
├── 파드 스펙에서 볼 수 있음 (kubectl get pod -o yaml에 secretKeyRef 이름 표시)
└── tmpfs로 마운트 (노드 디스크에 기록되지 않음)
```

**etcd 저장 시 암호화(encryption at rest) 활성화:**

```yaml
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
      - identity: {}    # 암호화되지 않은 시크릿 읽기를 위한 폴백
```

```bash
# API 서버에 암호화 설정 적용
# kube-apiserver 플래그에 추가:
# --encryption-provider-config=/etc/kubernetes/encryption-config.yaml

# 기존 시크릿 재암호화
kubectl get secrets --all-namespaces -o json | kubectl replace -f -
```

---

## 3. 불변 ConfigMap과 Secret

불변(Immutable) ConfigMap과 Secret은 생성 후 업데이트할 수 없습니다. 이는 다음을 제공합니다:
- **성능**: kubelet이 불변 객체에 대한 지속적인 watch 폴링을 건너뜀
- **안전성**: 프로덕션에서 실수로 구성이 변경되는 것을 방지
- **확장성**: API 서버 부하 감소 (이 객체들에 대한 watch 없음)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config-v2
immutable: true                  # 생성 후 수정 불가
data:
  DATABASE_HOST: "postgres.default.svc"
  LOG_LEVEL: "info"
---
apiVersion: v1
kind: Secret
metadata:
  name: api-key-v3
immutable: true
type: Opaque
stringData:
  api-key: "sk-xxxxxxxxxxxxxxxxxxxx"
```

```bash
# 불변 ConfigMap을 업데이트하려고 하면 실패
kubectl edit configmap app-config-v2
# Error: "app-config-v2" is immutable

# "업데이트"하려면 새 버전을 만들고 파드 참조를 업데이트
kubectl create configmap app-config-v3 \
  --from-literal=DATABASE_HOST=postgres.default.svc \
  --from-literal=LOG_LEVEL=debug

# 디플로이먼트가 새 ConfigMap을 사용하도록 업데이트
kubectl set env deployment/my-app --from=configmap/app-config-v3

# 이전 ConfigMap 삭제
kubectl delete configmap app-config-v2
```

### 3.1 버전 관리 패턴

```yaml
# 불변 구성에 버전 접미사 사용
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config-20240115-001
  labels:
    app: my-app
    config-version: "20240115-001"
immutable: true
data:
  config.yaml: |
    server:
      port: 8080
    database:
      host: postgres.production.svc
      pool_size: 50
```

---

## 4. External Secrets Operator

### 이론: 외부 시크릿 저장소 — 클러스터 외부에서 조정

네이티브 Secret은 "개발자가 한 번 시크릿을 만들고 절대 회전하지 않는" 케이스에 적합합니다. 프로덕션은 더 많이 요구합니다 — 회전, 감사, 다중 클러스터 중앙 관리, HSM 통합. 세 가지 패턴:

**External Secrets Operator (ESO).** 클러스터 내 컨트롤러가 `ExternalSecret` CRD를 watch합니다:

```yaml
kind: ExternalSecret
spec:
  refreshInterval: 1h
  secretStoreRef: { name: aws-store, kind: ClusterSecretStore }
  target: { name: db-credentials }
  data:
    - secretKey: password
      remoteRef: { key: prod/db, property: password }
```

컨트롤러는 매 `refreshInterval`마다 AWS Secrets Manager(또는 Vault, GCP Secret Manager, Azure Key Vault)에서 읽고, 값을 `db-credentials`라는 네이티브 Kubernetes `Secret`으로 구체화하고, Pod는 평소처럼 소비합니다. 조정 루프는 "클러스터 Secret을 상위 값과 일치시켜라"입니다. 상위에서의 회전이 자동으로 전파됩니다.

**Sealed Secrets.** 컨트롤러가 `SealedSecret` CRD(공개 키로 암호화되며, 비공개 짝은 클러스터에만 존재)를 네이티브 Secret으로 복호화합니다. `SealedSecret` YAML을 git에 안전하게 커밋할 수 있게 해줍니다 — 클러스터만 복호화 가능. 외부 저장소 불필요; 자동 회전도 없음.

**Vault Agent / CSI driver.** Vault Agent는 init 컨테이너와 공유 볼륨을 통해 시크릿을 Pod에 주입합니다 — Secrets Store CSI 드라이버는 Vault의 시크릿을 볼륨으로 직접 마운트합니다. 둘 다 etcd를 완전히 우회합니다 — Kubernetes Secret 객체가 만들어지지 않습니다. etcd 암호화로 충분치 않을 때 가장 강력한 모델입니다.

선택은 컴플라이언스에 중요합니다 — "시크릿이 절대 etcd에 있어서는 안 된다"가 요구사항이면 CSI/Vault. "시크릿이 24시간마다 회전되어야 한다"가 요구사항이면 ESO.

External Secrets Operator(ESO)는 외부 시크릿 관리 시스템(AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, HashiCorp Vault)의 시크릿을 Kubernetes Secret으로 동기화합니다.

### 4.1 설치

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets \
  external-secrets/external-secrets \
  -n external-secrets \
  --create-namespace
```

### 4.2 SecretStore 구성

```yaml
# ClusterSecretStore: 모든 네임스페이스에서 사용 가능
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
            namespace: external-secrets
```

### 4.3 ExternalSecret 정의

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
  namespace: production
spec:
  refreshInterval: 5m              # 5분마다 동기화
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore

  target:
    name: db-credentials           # 생성할 K8s Secret 이름
    creationPolicy: Owner          # ESO가 Secret 수명주기를 관리
    deletionPolicy: Retain         # ExternalSecret 삭제 시 Secret 유지
    template:
      type: Opaque
      data:
        # 시크릿 데이터 템플릿
        connection-string: "postgresql://{{ .username }}:{{ .password }}@{{ .host }}:5432/{{ .database }}"

  data:
    - secretKey: username
      remoteRef:
        key: production/database    # AWS Secrets Manager 시크릿 이름
        property: username          # 시크릿 내 JSON 키

    - secretKey: password
      remoteRef:
        key: production/database
        property: password

    - secretKey: host
      remoteRef:
        key: production/database
        property: host

    - secretKey: database
      remoteRef:
        key: production/database
        property: dbname
```

```bash
# 동기화 상태 확인
kubectl get externalsecret database-credentials -n production

# 출력:
# NAME                    STORE                  REFRESH INTERVAL   STATUS
# database-credentials    aws-secrets-manager    5m                 SecretSynced

# 생성된 Kubernetes Secret
kubectl get secret db-credentials -n production -o yaml
```

### 4.4 Push Secrets (양방향)

```yaml
# K8s Secret을 외부 저장소로 푸시
apiVersion: external-secrets.io/v1alpha1
kind: PushSecret
metadata:
  name: push-to-aws
spec:
  secretStoreRefs:
    - name: aws-secrets-manager
      kind: ClusterSecretStore
  selector:
    secret:
      name: generated-credentials
  data:
    - match:
        secretKey: api-key
        remoteRef:
          remoteKey: production/api-credentials
          property: key
```

---

## 5. Sealed Secrets

Bitnami의 Sealed Secrets는 Git 리포지토리에 안전하게 저장할 수 있도록 시크릿을 암호화합니다. 클러스터의 컨트롤러만이 이를 복호화할 수 있습니다.

### 5.1 설치

```bash
# 컨트롤러 설치
helm repo add sealed-secrets https://bitnami-labs.github.io/sealed-secrets
helm install sealed-secrets sealed-secrets/sealed-secrets \
  -n kube-system

# 클라이언트 도구 설치
brew install kubeseal   # macOS
# 또는 GitHub 릴리스에서 다운로드
```

### 5.2 Sealed Secret 생성

```bash
# 일반 시크릿 생성 (dry-run)
kubectl create secret generic db-credentials \
  --from-literal=username=admin \
  --from-literal=password='S3cur3P@ss!' \
  --dry-run=client -o yaml > /tmp/secret.yaml

# kubeseal로 암호화
kubeseal \
  --controller-name=sealed-secrets \
  --controller-namespace=kube-system \
  --format=yaml \
  < /tmp/secret.yaml \
  > sealed-secret.yaml

# Sealed Secret은 Git에 안전하게 커밋 가능
cat sealed-secret.yaml
```

Sealed Secret의 형태:

```yaml
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: db-credentials
  namespace: default
spec:
  encryptedData:
    username: AgBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...
    password: AgCyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy...
  template:
    metadata:
      name: db-credentials
    type: Opaque
```

```bash
# Sealed Secret 적용 (컨트롤러가 일반 Secret으로 복호화)
kubectl apply -f sealed-secret.yaml

# 일반 Secret이 생성되었는지 확인
kubectl get secret db-credentials
kubectl get secret db-credentials -o jsonpath='{.data.username}' | base64 -d
# admin
```

### 5.3 스코프 모드

| 스코프 | 설명 | 바인딩 대상 |
|--------|------|------------|
| strict (기본값) | Secret 이름 + 네임스페이스 | 정확한 이름과 네임스페이스 |
| namespace-wide | 네임스페이스 내 모든 이름 | 네임스페이스만 |
| cluster-wide | 모든 이름, 모든 네임스페이스 | 없음 |

```bash
# 네임스페이스 범위 스코프
kubeseal --scope namespace-wide < secret.yaml > sealed.yaml

# 클러스터 범위 스코프 (가장 덜 제한적)
kubeseal --scope cluster-wide < secret.yaml > sealed.yaml
```

### 5.4 키 교체(Key Rotation)

```bash
# 컨트롤러가 30일마다 자동으로 시링 키를 교체
# 이전 키는 기존 SealedSecret을 복호화할 수 있도록 유지

# 새 키로 모든 SealedSecret 재암호화
kubeseal --re-encrypt < sealed-secret.yaml > sealed-secret-new.yaml
```

---

## 6. HashiCorp Vault 통합

HashiCorp Vault는 접근 제어, 감사 로깅, 동적 시크릿을 갖춘 중앙 집중식 시크릿 관리를 제공합니다.

### 6.1 Vault Agent Injector

Vault Agent Injector는 파드에 Vault Agent 사이드카를 주입하여 시크릿을 가져오는 변경 웹훅(mutating webhook)입니다.

```bash
# Injector와 함께 Vault 설치
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault \
  --set "injector.enabled=true" \
  --set "server.dev.enabled=true"    # 테스트용 Dev 모드
```

### 6.2 Vault 구성

```bash
# Kubernetes 인증 방법 활성화
kubectl exec -it vault-0 -- vault auth enable kubernetes

# Kubernetes 인증 구성
kubectl exec -it vault-0 -- vault write auth/kubernetes/config \
  kubernetes_host="https://kubernetes.default.svc:443"

# 정책 생성
kubectl exec -it vault-0 -- vault policy write app-policy - <<EOF
path "secret/data/production/*" {
  capabilities = ["read"]
}
path "database/creds/readonly" {
  capabilities = ["read"]
}
EOF

# 애플리케이션 역할 생성
kubectl exec -it vault-0 -- vault write auth/kubernetes/role/app-role \
  bound_service_account_names=app-sa \
  bound_service_account_namespaces=production \
  policies=app-policy \
  ttl=1h

# 시크릿 저장
kubectl exec -it vault-0 -- vault kv put secret/production/database \
  username=admin \
  password="S3cur3P@ss!" \
  host="postgres.production.svc"
```

### 6.3 파드에 시크릿 주입

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-sa
  namespace: production
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
      annotations:
        # Vault Agent Injector 어노테이션
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "app-role"

        # 데이터베이스 자격 증명 주입
        vault.hashicorp.com/agent-inject-secret-db-creds: "secret/data/production/database"
        vault.hashicorp.com/agent-inject-template-db-creds: |
          {{- with secret "secret/data/production/database" -}}
          export DB_HOST="{{ .Data.data.host }}"
          export DB_USER="{{ .Data.data.username }}"
          export DB_PASS="{{ .Data.data.password }}"
          {{- end }}

        # 자동 교체: 주기적으로 시크릿 재조회
        vault.hashicorp.com/agent-inject-command-db-creds: "sh -c 'kill -HUP $(pidof my-app) || true'"
    spec:
      serviceAccountName: app-sa
      containers:
        - name: app
          image: my-app:v1.0
          command:
            - sh
            - -c
            - |
              source /vault/secrets/db-creds
              exec ./my-app
```

### 6.4 Vault CSI Provider

인젝터의 대안으로, CSI를 통해 시크릿을 파일로 마운트합니다:

```yaml
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: vault-db-creds
spec:
  provider: vault
  parameters:
    vaultAddress: "http://vault.default:8200"
    roleName: "app-role"
    objects: |
      - objectName: "db-username"
        secretPath: "secret/data/production/database"
        secretKey: "username"
      - objectName: "db-password"
        secretPath: "secret/data/production/database"
        secretKey: "password"
  # 선택적으로 Kubernetes Secret과 동기화
  secretObjects:
    - secretName: vault-db-secret
      type: Opaque
      data:
        - objectName: db-username
          key: username
        - objectName: db-password
          key: password
---
apiVersion: v1
kind: Pod
metadata:
  name: app-csi-vault
spec:
  serviceAccountName: app-sa
  containers:
    - name: app
      image: my-app:v1.0
      volumeMounts:
        - name: secrets
          mountPath: /mnt/secrets
          readOnly: true
      env:
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: vault-db-secret
              key: username
  volumes:
    - name: secrets
      csi:
        driver: secrets-store.csi.k8s.io
        readOnly: true
        volumeAttributes:
          secretProviderClass: vault-db-creds
```

---

## 7. 시크릿 교체 패턴

### 7.1 이중 시크릿 교체(Dual-Secret Rotation)

교체 중에 이전 자격 증명과 새 자격 증명을 모두 지원합니다:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-credentials
  labels:
    rotation-phase: dual       # 교체 상태 추적
type: Opaque
stringData:
  current-key: "new-api-key-v2"
  previous-key: "old-api-key-v1"
```

```go
package main

import (
	"fmt"
	"os"
)

// 이중 키 교체를 지원하는 애플리케이션 코드
func getAPIKey() string {
	// 현재 키를 먼저 시도
	currentKey := os.Getenv("CURRENT_API_KEY")
	if currentKey != "" {
		return currentKey
	}
	// 교체 중 이전 키로 폴백
	previousKey := os.Getenv("PREVIOUS_API_KEY")
	if previousKey != "" {
		fmt.Println("WARNING: Using previous API key during rotation")
		return previousKey
	}
	panic("No API key configured")
}
```

### 7.2 시크릿 변경 시 롤링 재시작

볼륨으로 마운트된 ConfigMap과 Secret은 자동 업데이트되지만, 환경 변수는 파드 재시작이 필요합니다. 어노테이션을 사용하여 롤링 재시작을 트리거합니다:

```bash
# 어노테이션을 업데이트하여 롤링 재시작 트리거
kubectl patch deployment my-app -p \
  "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"secret-version\":\"$(date +%s)\"}}}}}"
```

또는 해시 기반 접근 방식을 사용합니다:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    metadata:
      annotations:
        # 시크릿의 해시를 계산하여 어노테이션에 저장
        # 시크릿이 변경되면 해시가 변경되어 롤아웃 트리거
        checksum/secret: "sha256:abc123..."
    spec:
      containers:
        - name: app
          image: my-app:v1.0
          envFrom:
            - secretRef:
                name: db-credentials
```

### 7.3 Reloader

Stakater Reloader를 사용하여 ConfigMap이나 Secret이 변경될 때 자동으로 디플로이먼트를 재시작합니다:

```bash
# Reloader 설치
helm repo add stakater https://stakater.github.io/stakater-charts
helm install reloader stakater/reloader -n kube-system
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  annotations:
    # 특정 리소스 감시
    reloader.stakater.com/auto: "true"
    # 또는 특정 ConfigMap/Secret 감시:
    # configmap.reloader.stakater.com/reload: "app-config"
    # secret.reloader.stakater.com/reload: "db-credentials"
spec:
  template:
    spec:
      containers:
        - name: app
          image: my-app:v1.0
          envFrom:
            - configMapRef:
                name: app-config
            - secretRef:
                name: db-credentials
```

---

## 8. 구성 모범 사례

### 8.1 관심사 분리(Separation of Concerns)

```
구성 계층 구조:
├── 컨테이너 이미지: 코드 + 기본 구성
├── ConfigMap: 환경별 오버라이드
├── Secret: 민감한 값 (자격 증명, 키)
├── 환경 변수: 단순 키-값 오버라이드
└── 커맨드라인 인자: 런타임 플래그
```

### 8.2 명명 규칙

```yaml
# 설명적이고 버전이 지정된 이름 사용
apiVersion: v1
kind: ConfigMap
metadata:
  name: frontend-config           # app-component + "config"
  labels:
    app.kubernetes.io/name: frontend
    app.kubernetes.io/component: web
    app.kubernetes.io/part-of: my-platform
    config.kubernetes.io/version: "3"
```

### 8.3 ConfigMap에 시크릿을 저장하지 않기

```yaml
# 나쁜 예: ConfigMap에 민감한 데이터
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATABASE_URL: "postgresql://admin:password123@postgres:5432/mydb"
  # 비밀번호가 평문으로 보임!

# 좋은 예: 구성과 자격 증명 분리
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATABASE_HOST: "postgres.default.svc"
  DATABASE_PORT: "5432"
  DATABASE_NAME: "mydb"
---
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
stringData:
  username: admin
  password: "password123"
```

### 8.4 환경 변수보다 볼륨 마운트 사용

```yaml
# 구성 파일에는 볼륨 마운트를 선호
# - ConfigMap 변경 시 자동 업데이트 (재시작 불필요)
# - 여러 줄 구성에 적합
# - 자식 프로세스나 크래시 덤프에 노출되지 않음

spec:
  containers:
    - name: app
      volumeMounts:
        - name: config
          mountPath: /etc/app/config.yaml
          subPath: config.yaml
  volumes:
    - name: config
      configMap:
        name: app-config
```

### 8.5 시크릿에 대한 RBAC

```yaml
# 필요한 서비스 어카운트만 시크릿에 접근하도록 제한
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-reader
  namespace: production
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["db-credentials", "api-keys"]   # 특정 시크릿만
    verbs: ["get"]
    # 모든 시크릿에 대한 "list"나 "watch"는 절대 부여하지 않기
```

### 8.6 하드코딩 피하기

```yaml
# 나쁜 예: 하드코딩된 값
spec:
  containers:
    - name: app
      env:
        - name: API_URL
          value: "https://api.production.example.com"

# 좋은 예: ConfigMap 참조
spec:
  containers:
    - name: app
      env:
        - name: API_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: API_URL
```

---

## 9. 환경별 구성

### 9.1 Kustomize 오버레이(Overlays)

```
config/
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   ├── configmap.yaml
│   └── secret.yaml
├── overlays/
│   ├── dev/
│   │   ├── kustomization.yaml
│   │   └── configmap-patch.yaml
│   ├── staging/
│   │   ├── kustomization.yaml
│   │   └── configmap-patch.yaml
│   └── production/
│       ├── kustomization.yaml
│       └── configmap-patch.yaml
```

**Base ConfigMap:**

```yaml
# config/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: "info"
  CACHE_TTL: "300"
  FEATURE_NEW_UI: "false"
```

**프로덕션 오버레이:**

```yaml
# config/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
patches:
  - path: configmap-patch.yaml
configMapGenerator:
  - name: app-config
    behavior: merge
    literals:
      - LOG_LEVEL=warn
      - CACHE_TTL=3600
      - FEATURE_NEW_UI=true
      - REPLICAS=10
```

```bash
# 프로덕션 구성 미리보기
kubectl kustomize config/overlays/production/

# 적용
kubectl apply -k config/overlays/production/
```

### 9.2 Helm Values

```yaml
# values.yaml (기본값)
config:
  logLevel: info
  cacheTTL: 300
  featureFlags:
    newUI: false

# values-production.yaml
config:
  logLevel: warn
  cacheTTL: 3600
  featureFlags:
    newUI: true
```

```yaml
# templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Release.Name }}-config
data:
  LOG_LEVEL: {{ .Values.config.logLevel | quote }}
  CACHE_TTL: {{ .Values.config.cacheTTL | quote }}
  FEATURE_NEW_UI: {{ .Values.config.featureFlags.newUI | quote }}
```

```bash
# 환경별 values로 배포
helm install my-app ./chart -f values-production.yaml
```

### 9.3 환경별 ConfigMap

```yaml
# dev-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: dev
data:
  config.yaml: |
    server:
      port: 8080
      debug: true
    database:
      host: postgres.dev.svc
      pool_size: 5
    logging:
      level: debug
      format: text
---
# production-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: production
data:
  config.yaml: |
    server:
      port: 8080
      debug: false
    database:
      host: postgres.production.svc
      pool_size: 50
    logging:
      level: warn
      format: json
```

---

## 연습문제

### 연습문제 1: ConfigMap 관리

여러 줄의 nginx 구성 파일이 포함된 ConfigMap을 생성하세요. 이를 nginx 파드에 마운트하고 사용자 정의 구성이 활성화되어 있는지 확인하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/configmap-exercise.yaml로 저장
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-custom-config
data:
  default.conf: |
    server {
        listen 80;
        server_name localhost;

        location / {
            root /usr/share/nginx/html;
            index index.html;
        }

        location /health {
            return 200 '{"status":"healthy"}';
            add_header Content-Type application/json;
        }

        location /info {
            return 200 'Server: nginx\nConfig: custom\n';
            add_header Content-Type text/plain;
        }
    }
---
apiVersion: v1
kind: Pod
metadata:
  name: nginx-custom
spec:
  containers:
    - name: nginx
      image: nginx:1.25
      ports:
        - containerPort: 80
      volumeMounts:
        - name: nginx-config
          mountPath: /etc/nginx/conf.d
          readOnly: true
      resources:
        requests:
          cpu: "50m"
          memory: "64Mi"
  volumes:
    - name: nginx-config
      configMap:
        name: nginx-custom-config
```

```bash
kubectl apply -f /tmp/configmap-exercise.yaml
kubectl wait --for=condition=Ready pod/nginx-custom --timeout=60s

# 구성이 마운트되었는지 확인
kubectl exec nginx-custom -- cat /etc/nginx/conf.d/default.conf

# 사용자 정의 엔드포인트 테스트
kubectl exec nginx-custom -- curl -s localhost/health
# {"status":"healthy"}

kubectl exec nginx-custom -- curl -s localhost/info
# Server: nginx
# Config: custom

# ConfigMap 확인
kubectl get configmap nginx-custom-config -o yaml

# 정리
kubectl delete pod nginx-custom
kubectl delete configmap nginx-custom-config
```

</details>

### 연습문제 2: TLS를 사용한 시크릿

자체 서명 TLS 인증서를 생성하고 Kubernetes Secret으로 저장한 후, 해당 인증서를 사용하여 HTTPS를 제공하도록 nginx 파드를 구성하세요.

<details>
<summary>정답 보기</summary>

```bash
# 자체 서명 인증서 생성
openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout /tmp/tls.key \
  -out /tmp/tls.crt \
  -subj "/CN=nginx-tls.default.svc/O=exercise"

# TLS 시크릿 생성
kubectl create secret tls nginx-tls-secret \
  --cert=/tmp/tls.crt \
  --key=/tmp/tls.key
```

```yaml
# /tmp/tls-exercise.yaml로 저장
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-tls-config
data:
  default.conf: |
    server {
        listen 443 ssl;
        server_name localhost;

        ssl_certificate /etc/nginx/ssl/tls.crt;
        ssl_certificate_key /etc/nginx/ssl/tls.key;

        location / {
            return 200 'Hello from HTTPS!\n';
            add_header Content-Type text/plain;
        }
    }
---
apiVersion: v1
kind: Pod
metadata:
  name: nginx-tls
spec:
  containers:
    - name: nginx
      image: nginx:1.25
      ports:
        - containerPort: 443
      volumeMounts:
        - name: tls-certs
          mountPath: /etc/nginx/ssl
          readOnly: true
        - name: nginx-config
          mountPath: /etc/nginx/conf.d
          readOnly: true
      resources:
        requests:
          cpu: "50m"
          memory: "64Mi"
  volumes:
    - name: tls-certs
      secret:
        secretName: nginx-tls-secret
        defaultMode: 0400
    - name: nginx-config
      configMap:
        name: nginx-tls-config
```

```bash
kubectl apply -f /tmp/tls-exercise.yaml
kubectl wait --for=condition=Ready pod/nginx-tls --timeout=60s

# HTTPS가 작동하는지 확인
kubectl exec nginx-tls -- curl -sk https://localhost/
# Hello from HTTPS!

# 인증서 확인
kubectl exec nginx-tls -- openssl s_client -connect localhost:443 -servername localhost </dev/null 2>/dev/null | openssl x509 -noout -subject
# subject=CN = nginx-tls.default.svc, O = exercise

# 시크릿 유형 확인
kubectl get secret nginx-tls-secret
# TYPE: kubernetes.io/tls

# 정리
kubectl delete pod nginx-tls
kubectl delete configmap nginx-tls-config
kubectl delete secret nginx-tls-secret
rm /tmp/tls.key /tmp/tls.crt
```

</details>

### 연습문제 3: 불변 ConfigMap 버전 관리

불변 ConfigMap을 생성하고 이를 사용하는 애플리케이션을 배포한 후, 새 버전을 만들고 롤링 업데이트를 수행하여 구성을 "업데이트"하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/immutable-v1.yaml로 저장
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config-v1
  labels:
    app: config-demo
    version: v1
immutable: true
data:
  LOG_LEVEL: "info"
  FEATURE_FLAG: "false"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: config-demo
spec:
  replicas: 3
  selector:
    matchLabels:
      app: config-demo
  template:
    metadata:
      labels:
        app: config-demo
    spec:
      containers:
        - name: app
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              echo "Config: LOG_LEVEL=$LOG_LEVEL, FEATURE_FLAG=$FEATURE_FLAG"
              sleep 3600
          envFrom:
            - configMapRef:
                name: app-config-v1
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
```

```bash
# v1 배포
kubectl apply -f /tmp/immutable-v1.yaml
kubectl rollout status deployment/config-demo

# v1 구성 확인
kubectl exec deploy/config-demo -- sh -c 'echo "LOG_LEVEL=$LOG_LEVEL FEATURE_FLAG=$FEATURE_FLAG"'
# LOG_LEVEL=info FEATURE_FLAG=false

# 불변 ConfigMap 편집 시도 (실패해야 함)
kubectl patch configmap app-config-v1 -p '{"data":{"LOG_LEVEL":"debug"}}'
# Error: ConfigMap is immutable

# v2 생성
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config-v2
  labels:
    app: config-demo
    version: v2
immutable: true
data:
  LOG_LEVEL: "debug"
  FEATURE_FLAG: "true"
EOF

# 디플로이먼트를 v2 사용하도록 업데이트
kubectl set env deployment/config-demo --from=configmap/app-config-v2
kubectl rollout status deployment/config-demo

# v2 구성 확인
kubectl exec deploy/config-demo -- sh -c 'echo "LOG_LEVEL=$LOG_LEVEL FEATURE_FLAG=$FEATURE_FLAG"'
# LOG_LEVEL=debug FEATURE_FLAG=true

# 이전 ConfigMap 삭제
kubectl delete configmap app-config-v1

# 정리
kubectl delete deployment config-demo
kubectl delete configmap app-config-v2
```

</details>

### 연습문제 4: 다중 소스 구성

ConfigMap, Secret, Downward API의 구성을 단일 프로젝티드 볼륨(projected volume)으로 결합하는 파드를 생성하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/projected-exercise.yaml로 저장
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-settings
data:
  app.conf: |
    log_level=info
    max_retries=3
    timeout=30s
---
apiVersion: v1
kind: Secret
metadata:
  name: app-credentials
type: Opaque
stringData:
  api-key: "sk-abc123def456"
  db-password: "SuperSecret123!"
---
apiVersion: v1
kind: Pod
metadata:
  name: projected-demo
  labels:
    app: projected-demo
    version: v1.0.0
  annotations:
    environment: staging
spec:
  containers:
    - name: app
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "=== Configuration (ConfigMap) ==="
          cat /config/app.conf
          echo ""
          echo "=== Credentials (Secret) ==="
          echo "API Key: $(cat /config/credentials/api-key)"
          echo "DB Password: $(cat /config/credentials/db-password)"
          echo ""
          echo "=== Pod Info (Downward API) ==="
          echo "Pod Name: $(cat /config/podinfo/name)"
          echo "Namespace: $(cat /config/podinfo/namespace)"
          echo "Labels:"
          cat /config/podinfo/labels
          echo ""
          echo "=== Service Account Token ==="
          echo "Token (first 50 chars): $(head -c 50 /config/token)"
          echo "..."
          sleep 3600
      volumeMounts:
        - name: all-config
          mountPath: /config
          readOnly: true
      resources:
        requests:
          cpu: "50m"
          memory: "32Mi"
  volumes:
    - name: all-config
      projected:
        sources:
          - configMap:
              name: app-settings
              items:
                - key: app.conf
                  path: app.conf
          - secret:
              name: app-credentials
              items:
                - key: api-key
                  path: credentials/api-key
                  mode: 0400
                - key: db-password
                  path: credentials/db-password
                  mode: 0400
          - downwardAPI:
              items:
                - path: podinfo/name
                  fieldRef:
                    fieldPath: metadata.name
                - path: podinfo/namespace
                  fieldRef:
                    fieldPath: metadata.namespace
                - path: podinfo/labels
                  fieldRef:
                    fieldPath: metadata.labels
          - serviceAccountToken:
              path: token
              expirationSeconds: 3600
              audience: "kubernetes.default.svc"
```

```bash
kubectl apply -f /tmp/projected-exercise.yaml
kubectl wait --for=condition=Ready pod/projected-demo --timeout=60s

# 출력 보기
kubectl logs projected-demo

# 디렉토리 구조 확인
kubectl exec projected-demo -- find /config -type f
# /config/app.conf
# /config/credentials/api-key
# /config/credentials/db-password
# /config/podinfo/name
# /config/podinfo/namespace
# /config/podinfo/labels
# /config/token

# 시크릿 파일 권한 확인
kubectl exec projected-demo -- ls -la /config/credentials/
# 0400 권한이 표시되어야 함

# 정리
kubectl delete pod projected-demo
kubectl delete configmap app-settings
kubectl delete secret app-credentials
```

</details>

### 연습문제 5: Kustomize를 사용한 환경별 구성

ConfigMap이 포함된 기본 디플로이먼트와 서로 다른 구성 값을 가진 두 개의 오버레이(dev, production)를 생성하세요. 두 환경을 모두 배포하고 차이점을 확인하세요.

<details>
<summary>정답 보기</summary>

```bash
# 디렉토리 구조 생성
mkdir -p /tmp/kustomize-exercise/{base,overlays/dev,overlays/production}
```

```yaml
# /tmp/kustomize-exercise/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - configmap.yaml
```

```yaml
# /tmp/kustomize-exercise/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: env-config
data:
  LOG_LEVEL: "info"
  CACHE_ENABLED: "true"
  CACHE_TTL: "300"
```

```yaml
# /tmp/kustomize-exercise/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: env-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: env-app
  template:
    metadata:
      labels:
        app: env-app
    spec:
      containers:
        - name: app
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              echo "Environment: $ENVIRONMENT"
              echo "Log Level: $LOG_LEVEL"
              echo "Cache: $CACHE_ENABLED (TTL: $CACHE_TTL)"
              sleep 3600
          envFrom:
            - configMapRef:
                name: env-config
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
```

```yaml
# /tmp/kustomize-exercise/overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namespace: dev-env
patches:
  - target:
      kind: ConfigMap
      name: env-config
    patch: |
      - op: replace
        path: /data/LOG_LEVEL
        value: "debug"
      - op: add
        path: /data/ENVIRONMENT
        value: "development"
```

```yaml
# /tmp/kustomize-exercise/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namespace: prod-env
patches:
  - target:
      kind: Deployment
      name: env-app
    patch: |
      - op: replace
        path: /spec/replicas
        value: 3
  - target:
      kind: ConfigMap
      name: env-config
    patch: |
      - op: replace
        path: /data/LOG_LEVEL
        value: "warn"
      - op: replace
        path: /data/CACHE_TTL
        value: "3600"
      - op: add
        path: /data/ENVIRONMENT
        value: "production"
```

```bash
# 네임스페이스 생성
kubectl create namespace dev-env
kubectl create namespace prod-env

# dev 구성 미리보기
kubectl kustomize /tmp/kustomize-exercise/overlays/dev/

# production 구성 미리보기
kubectl kustomize /tmp/kustomize-exercise/overlays/production/

# 두 환경 모두 배포
kubectl apply -k /tmp/kustomize-exercise/overlays/dev/
kubectl apply -k /tmp/kustomize-exercise/overlays/production/

# 디플로이먼트 대기
kubectl -n dev-env rollout status deployment/env-app
kubectl -n prod-env rollout status deployment/env-app

# 구성 비교
echo "=== Dev Environment ==="
kubectl -n dev-env exec deploy/env-app -- env | grep -E "LOG_LEVEL|CACHE|ENVIRONMENT" | sort

echo "=== Production Environment ==="
kubectl -n prod-env exec deploy/env-app -- env | grep -E "LOG_LEVEL|CACHE|ENVIRONMENT" | sort

# Dev:     LOG_LEVEL=debug, CACHE_TTL=300, ENVIRONMENT=development
# Prod:    LOG_LEVEL=warn,  CACHE_TTL=3600, ENVIRONMENT=production

# 레플리카 수 비교
echo "Dev replicas: $(kubectl -n dev-env get deploy env-app -o jsonpath='{.spec.replicas}')"
echo "Prod replicas: $(kubectl -n prod-env get deploy env-app -o jsonpath='{.spec.replicas}')"
# Dev: 1, Prod: 3

# 정리
kubectl delete namespace dev-env prod-env
rm -rf /tmp/kustomize-exercise
```

</details>

---

## 10. 저장 시 시크릿 암호화(EncryptionConfiguration)

### 이론: 저장 시·전송 중·사용 중 암호화

세 계층, 각각 다른 공격자에 대한 방어:

- **전송 중(in transit).** API 서버, etcd, kubelet, CNI 컴포넌트 간 TLS. 클러스터 부트스트랩(kubeadm)에서 대부분 자동, EKS/GKE/AKS에서 관리형. 네트워크상의 도청에 대한 방어.
- **etcd에 저장 시(at rest).** `EncryptionConfiguration`이 KMS 프로바이더(AWS KMS, GCP KMS, Vault transit)와 함께 envelope 암호화 스킴을 활성화합니다. 이것이 없으면 어떤 etcd 백업이든 자격 증명 덤프입니다. 백업 도난과 물리적 디스크 도난에 대한 방어.
- **사용 중(in use), Pod 내부.** 시크릿이 tmpfs 파일이나 환경 변수로 마운트되면, 실행 중인 컨테이너는 평문으로 가집니다. 여기서의 방어는 워크로드 측입니다 — 파일 권한 강화, `readOnlyRootFilesystem` 사용, 절대 값 로깅 금지, 사용 후 메모리 스크럽. Kubernetes 자체는 Pod 경계에서 멈춥니다.

흔한 아키텍처 통찰: **시크릿이 API 서버를 떠나 Pod에 들어가는 순간, Kubernetes는 더 이상 보호할 수 없습니다.** 외부 시크릿 저장소(Vault, AWS Secrets Manager)의 핵심은 마법이 아닙니다 — 상위 시크릿을 중앙에서 회전할 수 있고, 클러스터 내 반영이 분 단위로 갱신된다는 것입니다.

기본적으로 Kubernetes는 etcd에 시크릿(Secret)을 base64로 인코딩된 평문으로 저장합니다.
`EncryptionConfiguration`은 API 서버가 etcd에 쓰기 전에 시크릿 데이터를 암호화하도록 지시하여,
무단 etcd 접근으로부터 보호합니다.

### 10.1 EncryptionConfiguration 매니페스트

이 파일을 컨트롤 플레인 노드에 배치하고(예: `/etc/kubernetes/encryption-config.yaml`)
kube-apiserver 매니페스트의 `--encryption-provider-config`로 참조합니다.

```yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      # aesgcm: 인증 암호화, 새 클러스터에 권장
      - aesgcm:
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-key>   # openssl rand -base64 32

      # aescbc: 구형 AES-CBC 모드 (HMAC-SHA256 인증)
      - aescbc:
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-key>

      # kms: 외부 KMS 공급자 사용 (AWS KMS, GCP KMS, HashiCorp Vault)
      # - kms:
      #     name: myKmsPlugin
      #     endpoint: unix:///tmp/socketfile.sock
      #     cachesize: 100
      #     timeout: 3s

      # identity: 암호화 없음 (기존 데이터 읽기 폴백으로 항상 마지막에 배치)
      - identity: {}
```

공급자 우선순위: **첫 번째** 공급자가 쓰기에 사용되며, 모든 공급자가 순서대로 읽기에 시도됩니다. 암호화 활성화 후 기존의 암호화되지 않은 시크릿을 읽을 수 있도록 `identity`를 마지막에 유지하세요.

### 10.2 공급자 비교

| 공급자 | 알고리즘 | 비고 |
|--------|----------|------|
| `aesgcm` | AES-GCM | 인증 암호화; 권장 |
| `aescbc` | AES-CBC + HMAC-SHA256 | 폭넓게 지원됨; 구형 클러스터 |
| `kms` (v1) | 엔벨로프 암호화 | 키 관리를 외부 KMS에 위임 |
| `kms` (v2) | 엔벨로프 암호화 | KMS v2 API (1.29에서 GA); 향상된 성능 |
| `identity` | 없음 | 암호화 없음; 읽기 폴백으로 사용 |
| `secretbox` | XSalsa20+Poly1305 | 빠름; 덜 일반적 |

### 10.3 기존 시크릿 마이그레이션

`EncryptionConfiguration`을 활성화한 후, etcd에 있는 기존 시크릿은 암호화되지 않은 상태로 유지됩니다. 모든 시크릿을 다시 쓰기하여 강제로 재암호화합니다:

```bash
# 기존 모든 시크릿 재암호화 (identity로 읽고, 새 공급자로 씀)
kubectl get secrets --all-namespaces -o json | kubectl replace -f -

# etcd에서 시크릿이 암호화되었는지 확인 (평문 base64가 아닌 알아볼 수 없는 데이터여야 함)
# 컨트롤 플레인 노드에서 실행:
ETCDCTL_API=3 etcdctl \
  --cacert /etc/kubernetes/pki/etcd/ca.crt \
  --cert   /etc/kubernetes/pki/etcd/server.crt \
  --key    /etc/kubernetes/pki/etcd/server.key \
  get /registry/secrets/default/my-secret | hexdump -C | head
```

마이그레이션 후, 설정에서 `identity` 공급자를 제거하고 API 서버를 재시작할 수 있습니다. 이후 새 시크릿은 키 없이는 읽을 수 없게 됩니다.

### 10.4 키 교체

1. 새 키를 **첫 번째** 항목으로 추가합니다 (새 쓰기가 이 키를 사용).
2. API 서버를 재시작합니다.
3. 모든 시크릿을 재암호화합니다 (위 단계).
4. 이전 키 항목을 제거하고 API 서버를 다시 재시작합니다.

---

**이전**: [스토리지와 영속성](./04_Storage_and_Persistence.md) | **다음**: [RBAC와 보안](./06_RBAC_and_Security.md)
