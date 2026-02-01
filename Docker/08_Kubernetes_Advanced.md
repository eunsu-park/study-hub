# 08. Kubernetes 심화

## 학습 목표
- Ingress를 통한 외부 트래픽 라우팅
- StatefulSet으로 상태 있는 애플리케이션 관리
- PersistentVolume/PersistentVolumeClaim 활용
- ConfigMap과 Secret 고급 사용법
- DaemonSet과 Job 활용

## 목차
1. [Ingress](#1-ingress)
2. [StatefulSet](#2-statefulset)
3. [영구 스토리지](#3-영구-스토리지)
4. [ConfigMap 고급](#4-configmap-고급)
5. [DaemonSet과 Job](#5-daemonset과-job)
6. [고급 스케줄링](#6-고급-스케줄링)
7. [연습 문제](#7-연습-문제)

---

## 1. Ingress

### 1.1 Ingress 개념

```
┌─────────────────────────────────────────────────────────────┐
│                    Ingress 아키텍처                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   인터넷                                                    │
│      │                                                      │
│      ▼                                                      │
│  ┌───────────────────────────────────────────┐             │
│  │         Ingress Controller                 │             │
│  │    (nginx, traefik, haproxy 등)           │             │
│  └───────────────────┬───────────────────────┘             │
│                      │                                      │
│        ┌─────────────┼─────────────┐                       │
│        │             │             │                        │
│        ▼             ▼             ▼                        │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│   │Ingress  │  │Ingress  │  │Ingress  │                   │
│   │Resource │  │Resource │  │Resource │                   │
│   └────┬────┘  └────┬────┘  └────┬────┘                   │
│        │             │             │                        │
│        ▼             ▼             ▼                        │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│   │Service A│  │Service B│  │Service C│                   │
│   └─────────┘  └─────────┘  └─────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Ingress Controller 설치

```bash
# NGINX Ingress Controller 설치
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# 설치 확인
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx

# IngressClass 확인
kubectl get ingressclass
```

### 1.3 기본 Ingress

```yaml
# simple-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: simple-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80

---
# 호스트 기반 라우팅
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: host-based-ingress
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
  - host: admin.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: admin-service
            port:
              number: 3000
```

### 1.4 경로 기반 라우팅

```yaml
# path-based-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: path-based-ingress
  annotations:
    nginx.ingress.kubernetes.io/use-regex: "true"
spec:
  ingressClassName: nginx
  rules:
  - host: app.example.com
    http:
      paths:
      # /api/* → api-service
      - path: /api(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: api-service
            port:
              number: 8080
      # /static/* → static-service
      - path: /static
        pathType: Prefix
        backend:
          service:
            name: static-service
            port:
              number: 80
      # 기본 → frontend
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
```

### 1.5 TLS 설정

```yaml
# tls-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tls-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - secure.example.com
    secretName: tls-secret  # TLS Secret
  rules:
  - host: secure.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: secure-service
            port:
              number: 443

---
# TLS Secret 생성
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
```

### 1.6 고급 Ingress 설정

```yaml
# advanced-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: advanced-ingress
  annotations:
    # 기본 설정
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"

    # Rate Limiting
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-connections: "50"

    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://frontend.example.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"

    # 인증
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: basic-auth
    nginx.ingress.kubernetes.io/auth-realm: "Authentication Required"

    # 커스텀 헤더
    nginx.ingress.kubernetes.io/configuration-snippet: |
      add_header X-Frame-Options "SAMEORIGIN";
      add_header X-Content-Type-Options "nosniff";

spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls
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

---

## 2. StatefulSet

### 2.1 StatefulSet 개념

```
┌─────────────────────────────────────────────────────────────┐
│            StatefulSet vs Deployment                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Deployment (Stateless)                                     │
│  ┌───────┐ ┌───────┐ ┌───────┐                             │
│  │pod-xyz│ │pod-abc│ │pod-123│  랜덤 이름, 교체 가능        │
│  └───────┘ └───────┘ └───────┘                             │
│                                                             │
│  StatefulSet (Stateful)                                     │
│  ┌───────┐ ┌───────┐ ┌───────┐                             │
│  │web-0  │ │web-1  │ │web-2  │  순서 보장, 고유 ID         │
│  │  ↓    │ │  ↓    │ │  ↓    │                             │
│  │pvc-0  │ │pvc-1  │ │pvc-2  │  각자 전용 스토리지         │
│  └───────┘ └───────┘ └───────┘                             │
│                                                             │
│  특징:                                                      │
│  • 순서대로 생성/삭제 (0 → 1 → 2)                          │
│  • 고정된 네트워크 ID (pod-name.service-name)              │
│  • 영구 스토리지 연결                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 StatefulSet 정의

```yaml
# statefulset-example.yaml
apiVersion: v1
kind: Service
metadata:
  name: web-headless
  labels:
    app: web
spec:
  ports:
  - port: 80
    name: web
  clusterIP: None  # Headless Service
  selector:
    app: web

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  serviceName: "web-headless"  # Headless Service 연결
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80
          name: web
        volumeMounts:
        - name: www
          mountPath: /usr/share/nginx/html

  # 볼륨 클레임 템플릿
  volumeClaimTemplates:
  - metadata:
      name: www
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: standard
      resources:
        requests:
          storage: 1Gi

  # 업데이트 전략
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 0  # 이 번호 이상의 Pod만 업데이트

  # Pod 관리 정책
  podManagementPolicy: OrderedReady  # 또는 Parallel
```

### 2.3 데이터베이스 StatefulSet

```yaml
# mysql-statefulset.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mysql-config
data:
  my.cnf: |
    [mysqld]
    bind-address = 0.0.0.0
    default_authentication_plugin = mysql_native_password

---
apiVersion: v1
kind: Secret
metadata:
  name: mysql-secret
type: Opaque
stringData:
  root-password: "rootpass123"
  user-password: "userpass123"

---
apiVersion: v1
kind: Service
metadata:
  name: mysql-headless
spec:
  clusterIP: None
  selector:
    app: mysql
  ports:
  - port: 3306

---
apiVersion: v1
kind: Service
metadata:
  name: mysql
spec:
  selector:
    app: mysql
    statefulset.kubernetes.io/pod-name: mysql-0  # Primary만
  ports:
  - port: 3306

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql-headless
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      initContainers:
      - name: init-mysql
        image: mysql:8.0
        command:
        - bash
        - "-c"
        - |
          set -ex
          # Pod 인덱스에서 server-id 생성
          [[ `hostname` =~ -([0-9]+)$ ]] || exit 1
          ordinal=${BASH_REMATCH[1]}
          echo [mysqld] > /mnt/conf.d/server-id.cnf
          echo server-id=$((100 + $ordinal)) >> /mnt/conf.d/server-id.cnf
        volumeMounts:
        - name: conf
          mountPath: /mnt/conf.d

      containers:
      - name: mysql
        image: mysql:8.0
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: root-password
        ports:
        - containerPort: 3306
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
        - name: conf
          mountPath: /etc/mysql/conf.d
        - name: config
          mountPath: /etc/mysql/my.cnf
          subPath: my.cnf
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
        livenessProbe:
          exec:
            command: ["mysqladmin", "ping"]
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command: ["mysql", "-h", "127.0.0.1", "-e", "SELECT 1"]
          initialDelaySeconds: 5
          periodSeconds: 2

      volumes:
      - name: conf
        emptyDir: {}
      - name: config
        configMap:
          name: mysql-config

  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast
      resources:
        requests:
          storage: 10Gi
```

### 2.4 StatefulSet 관리

```bash
# StatefulSet 조회
kubectl get statefulset
kubectl describe statefulset web

# Pod 확인 (순서대로 이름 부여)
kubectl get pods -l app=web
# NAME    READY   STATUS
# web-0   1/1     Running
# web-1   1/1     Running
# web-2   1/1     Running

# DNS 확인
# 각 Pod: web-0.web-headless.default.svc.cluster.local
kubectl run -it --rm debug --image=busybox -- nslookup web-0.web-headless

# 스케일링
kubectl scale statefulset web --replicas=5

# 롤링 업데이트
kubectl set image statefulset/web nginx=nginx:1.26

# 특정 Pod만 업데이트 (partition 사용)
kubectl patch statefulset web -p '{"spec":{"updateStrategy":{"rollingUpdate":{"partition":2}}}}'

# 삭제 (PVC는 유지됨)
kubectl delete statefulset web
kubectl delete pvc -l app=web  # PVC 삭제
```

---

## 3. 영구 스토리지

### 3.1 스토리지 계층 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    스토리지 계층 구조                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────┐               │
│  │              Pod                         │               │
│  │  ┌─────────────────────────────────┐   │               │
│  │  │     Volume Mount                 │   │               │
│  │  │     /data                        │   │               │
│  │  └─────────────┬───────────────────┘   │               │
│  └─────────────────┼───────────────────────┘               │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │     PersistentVolumeClaim (PVC)         │               │
│  │     • 스토리지 요청                      │               │
│  │     • 네임스페이스 범위                  │               │
│  └─────────────────┬───────────────────────┘               │
│                    │ 바인딩                                 │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │     PersistentVolume (PV)               │               │
│  │     • 실제 스토리지                      │               │
│  │     • 클러스터 범위                      │               │
│  └─────────────────┬───────────────────────┘               │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │     StorageClass                        │               │
│  │     • 동적 프로비저닝                    │               │
│  │     • 스토리지 유형 정의                 │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 StorageClass

```yaml
# storageclass.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: kubernetes.io/gce-pd  # 클라우드에 따라 다름
parameters:
  type: pd-ssd
reclaimPolicy: Delete  # Delete, Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer  # Immediate

---
# AWS EBS StorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: aws-fast
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
reclaimPolicy: Delete
allowVolumeExpansion: true

---
# 로컬 StorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: local-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
```

### 3.3 PersistentVolume (정적 프로비저닝)

```yaml
# pv-static.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-manual
  labels:
    type: local
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: manual
  hostPath:
    path: /mnt/data

---
# NFS PV
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-nfs
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: nfs
  nfs:
    server: nfs-server.example.com
    path: /exports/data

---
# AWS EBS PV
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-aws-ebs
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: aws-fast
  awsElasticBlockStore:
    volumeID: vol-0123456789abcdef
    fsType: ext4
```

### 3.4 PersistentVolumeClaim

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: app-data-pvc
  namespace: production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast  # 동적 프로비저닝
  # selector:             # 정적 바인딩 시 사용
  #   matchLabels:
  #     type: local

---
# Pod에서 PVC 사용
apiVersion: v1
kind: Pod
metadata:
  name: app-with-storage
spec:
  containers:
  - name: app
    image: myapp:latest
    volumeMounts:
    - name: data
      mountPath: /app/data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: app-data-pvc
```

### 3.5 볼륨 확장 및 스냅샷

```yaml
# PVC 확장 (allowVolumeExpansion: true 필요)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: expandable-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi  # 5Gi에서 확장
  storageClassName: fast

---
# VolumeSnapshot (CSI 드라이버 필요)
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: data-snapshot
spec:
  volumeSnapshotClassName: csi-snapclass
  source:
    persistentVolumeClaimName: app-data-pvc

---
# 스냅샷에서 PVC 복원
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: restored-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast
  dataSource:
    name: data-snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
```

---

## 4. ConfigMap 고급

### 4.1 ConfigMap 생성 방법

```bash
# 리터럴로 생성
kubectl create configmap app-config \
  --from-literal=LOG_LEVEL=info \
  --from-literal=MAX_CONNECTIONS=100

# 파일로 생성
kubectl create configmap nginx-config \
  --from-file=nginx.conf

# 디렉토리로 생성
kubectl create configmap app-configs \
  --from-file=config/
```

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  # 단순 키-값
  LOG_LEVEL: "info"
  DATABASE_HOST: "db.example.com"

  # 파일 형태
  app.properties: |
    server.port=8080
    server.host=0.0.0.0
    logging.level=INFO

  nginx.conf: |
    server {
        listen 80;
        server_name localhost;

        location / {
            root /usr/share/nginx/html;
            index index.html;
        }

        location /api {
            proxy_pass http://backend:8080;
        }
    }
```

### 4.2 ConfigMap 사용 방법

```yaml
# configmap-usage.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp:latest

    # 환경 변수로 주입
    env:
    - name: LOG_LEVEL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: LOG_LEVEL

    # 전체 ConfigMap을 환경 변수로
    envFrom:
    - configMapRef:
        name: app-config
      prefix: APP_  # 선택적 접두사

    volumeMounts:
    # 파일로 마운트
    - name: config-volume
      mountPath: /etc/app
    # 특정 키만 마운트
    - name: nginx-volume
      mountPath: /etc/nginx/conf.d

  volumes:
  - name: config-volume
    configMap:
      name: app-config
      # 전체 항목
  - name: nginx-volume
    configMap:
      name: app-config
      items:
      - key: nginx.conf
        path: default.conf
        mode: 0644
```

### 4.3 ConfigMap 변경 감지

```yaml
# 자동 리로드 (Reloader 사용)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
  annotations:
    # stakater/Reloader 어노테이션
    reloader.stakater.com/auto: "true"
    # 또는 특정 ConfigMap만
    configmap.reloader.stakater.com/reload: "app-config"
spec:
  template:
    spec:
      containers:
      - name: app
        image: myapp:latest
        volumeMounts:
        - name: config
          mountPath: /etc/app
      volumes:
      - name: config
        configMap:
          name: app-config

---
# Sidecar로 변경 감지
apiVersion: v1
kind: Pod
metadata:
  name: app-with-reloader
spec:
  containers:
  - name: app
    image: myapp:latest
    volumeMounts:
    - name: config
      mountPath: /etc/app

  - name: config-reloader
    image: jimmidyson/configmap-reload:v0.5.0
    args:
    - --volume-dir=/etc/app
    - --webhook-url=http://localhost:8080/-/reload
    volumeMounts:
    - name: config
      mountPath: /etc/app
      readOnly: true

  volumes:
  - name: config
    configMap:
      name: app-config
```

---

## 5. DaemonSet과 Job

### 5.1 DaemonSet

```yaml
# daemonset.yaml
# 모든 노드에 Pod 배포
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  labels:
    app: node-exporter
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      # 특정 노드에만 배포
      nodeSelector:
        monitoring: "true"

      tolerations:
      # 마스터 노드에도 배포
      - key: node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule

      containers:
      - name: node-exporter
        image: prom/node-exporter:v1.6.1
        ports:
        - containerPort: 9100
          hostPort: 9100
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        resources:
          limits:
            cpu: 200m
            memory: 100Mi
          requests:
            cpu: 100m
            memory: 50Mi

      hostNetwork: true
      hostPID: true

      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys

  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
```

### 5.2 Job

```yaml
# job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-migration
spec:
  # 완료 조건
  completions: 1        # 성공해야 할 Pod 수
  parallelism: 1        # 동시 실행 Pod 수
  backoffLimit: 3       # 실패 시 재시도 횟수
  activeDeadlineSeconds: 600  # 최대 실행 시간

  # 완료 후 삭제
  ttlSecondsAfterFinished: 3600

  template:
    spec:
      restartPolicy: Never  # OnFailure 또는 Never
      containers:
      - name: migrator
        image: myapp/migrator:latest
        command: ["python", "migrate.py"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url

---
# 병렬 Job
apiVersion: batch/v1
kind: Job
metadata:
  name: parallel-processing
spec:
  completions: 10     # 총 10개 완료
  parallelism: 3      # 동시에 3개씩
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: worker
        image: myapp/worker:latest
```

### 5.3 CronJob

```yaml
# cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: db-backup
spec:
  schedule: "0 2 * * *"  # 매일 새벽 2시
  timeZone: "Asia/Seoul"

  # 동시 실행 정책
  concurrencyPolicy: Forbid  # Allow, Forbid, Replace

  # 시작 데드라인
  startingDeadlineSeconds: 300

  # 성공/실패 히스토리
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1

  # 일시 중지
  suspend: false

  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: postgres:15
            command:
            - /bin/sh
            - -c
            - |
              pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > /backup/db_$(date +%Y%m%d).sql
              aws s3 cp /backup/db_$(date +%Y%m%d).sql s3://backups/
            env:
            - name: DB_HOST
              value: "postgres"
            - name: DB_USER
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: username
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: password
            volumeMounts:
            - name: backup
              mountPath: /backup
          volumes:
          - name: backup
            emptyDir: {}
```

---

## 6. 고급 스케줄링

### 6.1 Node Affinity

```yaml
# node-affinity.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  affinity:
    nodeAffinity:
      # 필수 조건
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: gpu
            operator: In
            values:
            - "true"
          - key: kubernetes.io/arch
            operator: In
            values:
            - amd64

      # 선호 조건
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
          - key: gpu-type
            operator: In
            values:
            - nvidia-a100
      - weight: 50
        preference:
          matchExpressions:
          - key: gpu-type
            operator: In
            values:
            - nvidia-v100

  containers:
  - name: gpu-app
    image: nvidia/cuda:12.0-base
    resources:
      limits:
        nvidia.com/gpu: 1
```

### 6.2 Pod Affinity/Anti-Affinity

```yaml
# pod-affinity.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      affinity:
        # 캐시 Pod와 같은 노드 선호
        podAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - cache
              topologyKey: kubernetes.io/hostname

        # 같은 앱의 다른 Pod와 다른 노드에 배치
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - web
            topologyKey: kubernetes.io/hostname

      containers:
      - name: web
        image: nginx:latest
```

### 6.3 Taints와 Tolerations

```bash
# 노드에 Taint 추가
kubectl taint nodes node1 dedicated=gpu:NoSchedule
kubectl taint nodes node2 special=true:PreferNoSchedule
kubectl taint nodes node3 critical=true:NoExecute

# Taint 제거
kubectl taint nodes node1 dedicated=gpu:NoSchedule-
```

```yaml
# tolerations.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  tolerations:
  # 정확히 일치
  - key: "dedicated"
    operator: "Equal"
    value: "gpu"
    effect: "NoSchedule"

  # 키만 존재하면 됨
  - key: "special"
    operator: "Exists"
    effect: "PreferNoSchedule"

  # NoExecute + tolerationSeconds
  - key: "critical"
    operator: "Equal"
    value: "true"
    effect: "NoExecute"
    tolerationSeconds: 3600  # 1시간 후 축출

  containers:
  - name: app
    image: myapp:latest
```

### 6.4 Topology Spread Constraints

```yaml
# topology-spread.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distributed-app
spec:
  replicas: 6
  selector:
    matchLabels:
      app: distributed
  template:
    metadata:
      labels:
        app: distributed
    spec:
      topologySpreadConstraints:
      # Zone 간 균등 분배
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: distributed

      # 노드 간 균등 분배
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: distributed

      containers:
      - name: app
        image: myapp:latest
```

---

## 7. 연습 문제

### 연습 1: 마이크로서비스 Ingress
```yaml
# 요구사항:
# - api.example.com/v1/* → api-v1-service
# - api.example.com/v2/* → api-v2-service
# - TLS 적용, HTTP→HTTPS 리다이렉트
# - Rate limiting 적용

# Ingress 작성
```

### 연습 2: Redis Cluster StatefulSet
```yaml
# 요구사항:
# - 3개 노드 Redis Cluster
# - 각 노드에 1Gi PVC
# - Headless Service로 노드 간 통신
# - 적절한 리소스 제한

# StatefulSet 작성
```

### 연습 3: 로그 수집 DaemonSet
```yaml
# 요구사항:
# - 모든 노드에서 /var/log 수집
# - Elasticsearch로 전송
# - ConfigMap으로 설정 관리
# - 마스터 노드에도 배포

# DaemonSet 작성
```

### 연습 4: 배치 데이터 처리 Job
```yaml
# 요구사항:
# - 100개 데이터 처리 (completions: 100)
# - 10개씩 병렬 처리 (parallelism: 10)
# - 실패 시 3회 재시도
# - 2시간 내 완료 필수
# - 완료 후 24시간 뒤 삭제

# Job 작성
```

---

## 다음 단계

- [09_Helm_패키지관리](09_Helm_패키지관리.md) - Helm 차트
- [10_CI_CD_파이프라인](10_CI_CD_파이프라인.md) - 자동화 배포
- [07_Kubernetes_보안](07_Kubernetes_보안.md) - 보안 복습

## 참고 자료

- [Kubernetes Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- [StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [DaemonSet](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/)

---

[← 이전: Kubernetes 보안](07_Kubernetes_보안.md) | [다음: Helm 패키지관리 →](09_Helm_패키지관리.md) | [목차](00_Overview.md)
