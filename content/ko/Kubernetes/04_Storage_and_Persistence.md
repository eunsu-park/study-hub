# 04. 스토리지와 영속성

**이전**: [네트워킹 기초](./03_Networking_Fundamentals.md) | **다음**: [구성과 시크릿](./05_Configuration_and_Secrets.md)

## 학습 목표
- 볼륨 유형과 적절한 사용 사례를 이해한다
- PersistentVolume, PersistentVolumeClaim, StorageClass를 구성한다
- 동적 프로비저닝(Dynamic Provisioning)을 구현하고 접근 모드와 회수 정책을 관리한다
- CSI 드라이버와 볼륨 스냅샷을 활용한다
- 스테이트풀셋(StatefulSets)과 임시 워크로드를 위한 스토리지 패턴을 설계한다

---

컨테이너는 설계상 일시적(Ephemeral)입니다 — 컨테이너가 재시작되면 파일시스템이
이미지 상태로 초기화됩니다. 쿠버네티스 볼륨은 개별 컨테이너와 파드(Pod)보다
오래 지속되는 스토리지를 제공하여 이 문제를 해결합니다. 이 레슨에서는 기본 볼륨부터
CSI 드라이버를 사용한 프로덕션급 동적 프로비저닝까지 전체 스토리지 스택을 다룹니다.

볼륨 투어에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 쿠버네티스가 스토리지를 PV(공급 측)와 PVC(수요 측)로 분리한 이유, 바인딩 알고리즘이 둘을 매칭하는 방식, StorageClass 기반 동적 프로비저닝이 내부적으로 무엇을 트리거하는지, 그리고 CSI 인터페이스가 스토리지 벤더들이 쿠버네티스 코어를 수정하지 않게 만든 방법을 다룹니다.

## 목차
0. [이론과 원리](#이론과-원리)
1. [볼륨과 볼륨 유형](#1-볼륨과-볼륨-유형)
2. [PersistentVolume (PV)](#2-persistentvolume-pv)
3. [PersistentVolumeClaim (PVC)](#3-persistentvolumeclaim-pvc)
4. [StorageClass와 동적 프로비저닝](#4-storageclass와-동적-프로비저닝)
5. [접근 모드](#5-접근-모드)
6. [회수 정책](#6-회수-정책)
7. [CSI (컨테이너 스토리지 인터페이스)](#7-csi-컨테이너-스토리지-인터페이스)
8. [볼륨 스냅샷](#8-볼륨-스냅샷)
9. [임시 볼륨](#9-임시-볼륨)
10. [스테이트풀셋 스토리지 패턴](#10-스테이트풀셋-스토리지-패턴)
11. [연습문제](#연습문제)

---

## 이론과 원리

스토리지는 쿠버네티스에서 두 가지 상반된 관심사가 만나는 지점입니다 — 컨테이너는 일시적이고 교체 가능하기를 원하지만, 데이터베이스·큐·캐시는 데이터가 자신을 쓴 프로세스보다 오래 살아남아야 한다고 요구합니다. 쿠버네티스는 일시성을 패치해 없애는 대신 **스토리지를 워크로드와 분리**함으로써 이를 해결합니다. 워크로드는 "이러한 속성을 가진 이만큼의 스토리지가 필요하다"고 선언하고, 별도의 서브시스템이 이를 제공·attach·mount·회수합니다. 이 섹션은 공급-수요 모델, 바인딩 알고리즘, 동적 프로비저닝, 그리고 전체를 확장 가능하게 만든 CSI 플러그인 계약을 설명합니다.

### A. PV / PVC — 공급-수요의 분리

쿠버네티스는 스토리지를 시장(marketplace)으로 모델링합니다:

- **PersistentVolume (PV)**는 *공급 측*입니다 — 클러스터에 존재하는 스토리지 조각으로, 속성(용량, 접근 모드, 회수 정책, storage class, 백킹 드라이버)을 가집니다. PV는 클러스터 범위(네임스페이스가 없음)이며, 인프라 어딘가에서 사용 가능한 스토리지를 기술합니다.
- **PersistentVolumeClaim (PVC)**는 *수요 측*입니다 — 속성(요청 용량, 필요 접근 모드, 선택적 storage class)을 가진 네임스페이스 범위의 스토리지 요청. PVC는 워크로드 소유자가 작성합니다.
- **바인딩 컨트롤러**가 PVC를 PV에 매칭합니다.

이 분리가 중요한 이유는 스토리지 관리자와 앱 개발자가 다른 시간 척도로 사고하기 때문입니다. 클러스터 운영자가 PV 풀을 사전 프로비저닝하거나(또는 동적 프로비저닝을 설정하거나, §C), 개발자는 기저 디스크가 EBS인지 Ceph RBD인지 NFS인지 로컬 SSD인지 모르고 PVC를 사용합니다. 동일한 PVC YAML이 모든 클라우드와 온프레미스에서 동작합니다.

Pod는 **PVC를 이름으로** 참조하지, PV를 직접 참조하지 않습니다. 이 간접 참조가 동일한 워크로드 매니페스트를 환경 간에 배포할 수 있게 합니다.

### B. 바인딩 알고리즘

PVC가 생성되면, 컨트롤러는 다음 모두를 만족하는 PV를 찾습니다:

1. **용량 ≥ 요청.** 5Gi PVC는 10Gi PV에 바인딩됩니다(차이는 낭비됨; PV는 분할되지 않음).
2. **AccessMode가 PV의 지원 집합에 포함됨.** RWO(ReadWriteOnce: 한 노드), ROX(ReadOnlyMany), RWX(ReadWriteMany), RWOP(ReadWriteOncePod, 단일 파드). 블록 스토리지는 RWO; 네트워크 파일시스템은 RWX 가능.
3. **StorageClass 매칭** (정적 프로비저닝의 경우 "" / nil 케이스 포함).
4. PVC가 지정한 경우 **Selector / volumeName 매칭**.

여러 PV가 매치되면, 컨트롤러는 낭비를 최소화하기 위해 적합한 가장 작은 것을 고릅니다. 매치되는 것이 없고 StorageClass에 provisioner가 설정되어 있으면 동적 프로비저닝이 작동합니다(§C). 그렇지 않으면 PVC는 Pending에 머뭅니다.

바인딩되면 PV와 PVC는 배타적입니다 — 1:1 바인딩이며 양 객체의 `spec.claimRef` / `spec.volumeName`에 저장됩니다. PVC가 삭제되고 동일 이름으로 재생성되더라도 새 PV를 받습니다(또는 회수 정책에 따라 옛 PV가 여전히 바인딩되어 있으면 Pending에 머뭅니다).

Pod 스케줄러도 관여합니다 — `volumeBindingMode: WaitForFirstConsumer`이면 바인딩이 Pod가 실제로 PVC를 사용할 때까지 지연되어, PV가 선택된 노드와 동일 영역에서 생성될 수 있습니다. 이 옵션이 없으면 영역 A에 PV가 만들어지고, 그러면 스케줄러가 Pod를 영역 A에 배치해야 하는 — 과제약 — 상황이 발생할 수 있습니다.

### C. StorageClass를 통한 동적 프로비저닝

워크로드 믹스를 미리 알지 못하는 클러스터에서 PV를 사전 프로비저닝하는 것은 운영적으로 고통스럽습니다. **StorageClass** 추상은 클러스터가 *PV를 온디맨드로 생성*하게 해줍니다:

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

`storageClassName: gp3`인 PVC가 도착하면, 컨트롤러는 명명된 provisioner("EBS CSI 드라이버")에게 실제 EBS 볼륨과 그에 대응하는 PV를 만들어 달라고 요청합니다. 그러면 PVC는 자동 생성된 PV에 바인딩됩니다. 이로써 스토리지가 정적 인벤토리 문제에서 온디맨드 유틸리티로 전환됩니다.

여러 StorageClass로 티어를 제공할 수 있습니다(범용 워크로드용 `gp3`, 데이터베이스용 `io2`, 아카이브용 `cold`). 하나를 기본값으로 지정할 수 있습니다(`storageclass.kubernetes.io/is-default-class: "true"`) — 명시 클래스가 없는 PVC도 동작하도록.

회수 정책은 PVC가 삭제될 때 무엇이 일어날지 결정합니다 — `Delete`(기저 볼륨도 삭제 — 파괴적!), `Retain`(PV를 `Released` 상태로 남겨 수동 정리; 프로덕션 데이터에 사용), `Recycle`(deprecated; 기본 scrub-and-reuse).

### D. CSI — 스토리지를 확장 가능하게 만든 플러그인 계약

CSI 이전에는 모든 스토리지 드라이버(NFS, RBD, EBS, GCE PD, ...)가 Kubernetes 코어에 컴파일되어 있었습니다. 벤더 추가는 Kubernetes 릴리스를 필요로 했습니다. **CSI(Container Storage Interface)**는 어떤 벤더든 out-of-tree로 구현할 수 있는 표준 gRPC 인터페이스를 정의해 이를 깨뜨렸습니다.

CSI 드라이버는 두 부분입니다:

- **Controller plugin** (클러스터 전역): `CreateVolume` / `DeleteVolume`, 스냅샷 작업, (클라우드 프로바이더의 경우) 노드에 대한 attach/detach를 처리. kube-system 네임스페이스에서 Deployment로 실행.
- **Node plugin** (노드별 DaemonSet): `NodeStageVolume` / `NodePublishVolume`을 처리 — 디바이스를 포맷하고 Pod의 파일시스템 네임스페이스에 마운트.

kubelet은 EBS나 RBD를 알지 못합니다 — CSI gRPC 메서드만 호출합니다. 이 격리가 현대 스토리지 벤더가 단일 Helm 차트만 제공해도 아무 것도 재컴파일하지 않고 완전한 Kubernetes 통합을 얻는 이유입니다.

동적 프로비저닝된 PVC를 사용하는 Pod의 라이프사이클:

1. PVC 생성 → external-provisioner 사이드카가 CSI `CreateVolume` 호출 → 클라우드가 디스크 생성, provisioner가 PV 생성, 바인딩 컨트롤러가 PVC↔PV 바인딩.
2. Pod가 노드에 스케줄됨 → external-attacher가 CSI `ControllerPublishVolume` 호출 → 클라우드가 디스크를 그 노드에 attach.
3. 노드의 kubelet이 CSI `NodeStageVolume`(필요 시 포맷, staging dir에 마운트)과 `NodePublishVolume`(Pod의 파일시스템에 bind-mount) 호출.
4. Pod 실행.
5. Pod 삭제 → 역순: `NodeUnpublish`, `NodeUnstage`, `ControllerUnpublish`. PVC 삭제(`Delete` 회수 정책)는 `DeleteVolume`을 트리거.

볼륨 스냅샷과 클론은 CSI 선택적 기능(`VolumeSnapshot`, `VolumeSnapshotClass`)으로 동일 패턴을 따릅니다 — 요청 객체, CSI 호출, 둘을 잇는 컨트롤러.

### 이론에서 아래의 YAML으로

이제 레슨은 이 추상을 안내합니다:

- **섹션 1 (볼륨과 볼륨 유형)**은 PV가 아닌 더 낮은 수준의 Pod 범위 볼륨(emptyDir, configMap, projected 등)을 다룹니다 — 영속성 모델 도입 전에 유용합니다.
- **섹션 2–3 (PV, PVC)**는 §A입니다 — 공급과 수요 객체. 두 YAML을 나란히 두고 `accessModes`와 `storage`가 둘을 어떻게 연결하는지 보세요.
- **섹션 4 (StorageClass, 동적 프로비저닝)**은 §C입니다 — `provisioner`와 `parameters`가 실제 클라우드 드라이버에 어떻게 매핑되는지 봅니다.
- **섹션 5 (접근 모드)**는 §B의 바인딩 알고리즘 제약을 구체적 RWO/ROX/RWX 예제로 풀어냅니다.
- **섹션 6 (회수 정책)**은 §C의 파괴적 vs 보존 선택입니다 — 잘못 고르면 PVC 삭제로 데이터를 잃을 수 있습니다.
- **섹션 7 (CSI)**는 §D입니다 — controller + node 플러그인 아키텍처 다이어그램을 봅니다.
- **섹션 8 (볼륨 스냅샷)**은 백업/복원 패턴을 위해 CSI 스냅샷 기능을 사용합니다.
- **섹션 9 (임시 볼륨)**은 관련된 설계 — 일반(generic) ephemeral 볼륨은 단기 스토리지에 PVC 머신너리를 사용합니다.
- **섹션 10 (StatefulSet 스토리지 패턴)**은 스토리지를 레플리카별 식별이 필요한 워크로드(2강 §C)에 다시 묶습니다.

PV/PVC를 공급/수요로, StorageClass를 온디맨드 프로비저닝으로 보고 나면, 모든 스토리지 YAML은 §B 바인딩 알고리즘의 네 부분을 특수화하는 것에 불과합니다.

---

## 1. 볼륨과 볼륨 유형

쿠버네티스 볼륨은 파드 내 컨테이너가 접근할 수 있는 디렉토리입니다. 볼륨의
라이프사이클은 유형에 따라 다릅니다 — 일부는 파드에 묶이고, 다른 것은 독립적으로
영속됩니다.

### 1.1 emptyDir

파드가 노드에 할당될 때 생성되고, 파드가 제거될 때 삭제됩니다.
파드 내 모든 컨테이너가 읽고 쓸 수 있습니다.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: emptydir-demo
spec:
  containers:
    - name: writer
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          while true; do
            echo "$(date): Log entry" >> /data/app.log
            sleep 5
          done
      volumeMounts:
        - name: shared-data
          mountPath: /data

    - name: reader
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          tail -f /data/app.log
      volumeMounts:
        - name: shared-data
          mountPath: /data
          readOnly: true

  volumes:
    - name: shared-data
      emptyDir: {}
      # emptyDir:
      #   medium: Memory      # tmpfs 사용 (RAM 기반)
      #   sizeLimit: 256Mi    # 크기 제한 적용
```

사용 사례:
- 계산을 위한 임시 공간
- 파드 내 컨테이너 간 데이터 공유
- 캐시 디렉토리

### 1.2 hostPath

호스트 노드의 파일시스템에서 파일이나 디렉토리를 파드에 마운트합니다.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hostpath-demo
spec:
  containers:
    - name: app
      image: busybox:1.36
      command: ["sleep", "3600"]
      volumeMounts:
        - name: host-logs
          mountPath: /host-logs
          readOnly: true
  volumes:
    - name: host-logs
      hostPath:
        path: /var/log
        type: Directory    # 존재해야 함; 디렉토리가 아니면 실패
```

hostPath 유형:

| 유형 | 동작 |
|------|----------|
| `""` (빈 문자열) | 검사 없음; 필요하면 생성 |
| `DirectoryOrCreate` | 없으면 디렉토리 생성 |
| `Directory` | 디렉토리로 존재해야 함 |
| `FileOrCreate` | 없으면 파일 생성 |
| `File` | 파일로 존재해야 함 |
| `Socket` | Unix 소켓으로 존재해야 함 |

> **경고**: hostPath 볼륨은 보안 위험입니다. 컨테이너 격리를 우회하고
> 노드의 모든 파일에 접근할 수 있습니다. 프로덕션에서는 피하고, 대신
> PersistentVolume을 사용하세요.

### 1.3 configMap과 secret

ConfigMap 또는 Secret 데이터를 파일로 마운트합니다:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: config-volume-demo
spec:
  containers:
    - name: app
      image: nginx:1.25
      volumeMounts:
        - name: config-vol
          mountPath: /etc/nginx/conf.d
        - name: tls-vol
          mountPath: /etc/nginx/ssl
          readOnly: true
  volumes:
    - name: config-vol
      configMap:
        name: nginx-config
        items:
          - key: default.conf
            path: default.conf     # 특정 키만 마운트
    - name: tls-vol
      secret:
        secretName: nginx-tls
        defaultMode: 0400          # 파일 권한
```

### 1.4 projected

여러 볼륨 소스를 단일 마운트로 결합합니다:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: projected-demo
spec:
  containers:
    - name: app
      image: busybox:1.36
      command: ["sleep", "3600"]
      volumeMounts:
        - name: all-in-one
          mountPath: /projected
  volumes:
    - name: all-in-one
      projected:
        sources:
          - configMap:
              name: app-config
              items:
                - key: config.yaml
                  path: config.yaml
          - secret:
              name: app-secret
              items:
                - key: api-key
                  path: credentials/api-key
          - downwardAPI:
              items:
                - path: labels
                  fieldRef:
                    fieldPath: metadata.labels
                - path: cpu-request
                  resourceFieldRef:
                    containerName: app
                    resource: requests.cpu
          - serviceAccountToken:
              path: token
              expirationSeconds: 3600
              audience: vault
```

### 1.5 downwardAPI

파드 메타데이터를 파일로 노출합니다:

```yaml
volumes:
  - name: podinfo
    downwardAPI:
      items:
        - path: "name"
          fieldRef:
            fieldPath: metadata.name
        - path: "namespace"
          fieldRef:
            fieldPath: metadata.namespace
        - path: "labels"
          fieldRef:
            fieldPath: metadata.labels
        - path: "annotations"
          fieldRef:
            fieldPath: metadata.annotations
        - path: "cpu-request"
          resourceFieldRef:
            containerName: app
            resource: requests.cpu
            divisor: "1m"    # 밀리코어로 표현
```

---

## 2. PersistentVolume (PV)

PersistentVolume은 관리자가 프로비저닝하거나 StorageClass에 의해 동적으로
프로비저닝되는 클러스터 전역 스토리지 리소스입니다. PV는 어떤 파드와도 독립적인
라이프사이클을 가집니다.

### 2.1 정적 프로비저닝(Static Provisioning)

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-nfs-data
  labels:
    type: nfs
    environment: production
spec:
  capacity:
    storage: 100Gi
  volumeMode: Filesystem          # 또는 Block
  accessModes:
    - ReadWriteMany               # 여러 노드가 읽기-쓰기로 마운트 가능
  persistentVolumeReclaimPolicy: Retain
  storageClassName: nfs-slow      # StorageClass에 연결 (또는 클래스 없이 "")
  mountOptions:
    - hard
    - nfsvers=4.1
  nfs:
    server: 192.168.1.100
    path: /exports/data
```

### 2.2 로컬 스토리지를 가진 PV

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-local-ssd
spec:
  capacity:
    storage: 500Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Delete
  storageClassName: local-ssd
  local:
    path: /mnt/ssd/data
  nodeAffinity:                   # 로컬 볼륨에 필수
    required:
      nodeSelectorTerms:
        - matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values:
                - worker-node-1
```

### 2.3 PV 라이프사이클 단계

```
Available → Bound → Released → (Reclaimed/Deleted)
```

| 단계 | 설명 |
|-------|-------------|
| Available | PV가 자유 상태이며 아직 PVC에 바인딩되지 않음 |
| Bound | PV가 PVC에 바인딩됨 |
| Released | PVC가 삭제됨; PV가 아직 회수되지 않음 |
| Failed | 자동 회수 실패 |

```bash
# PersistentVolume 나열
kubectl get pv

# 출력:
# NAME           CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS      STORAGECLASS
# pv-nfs-data    100Gi      RWX            Retain           Available   nfs-slow
# pv-local-ssd   500Gi      RWO            Delete           Bound       local-ssd

# PV 상세 정보
kubectl describe pv pv-nfs-data
```

---

## 3. PersistentVolumeClaim (PVC)

PVC는 사용자의 스토리지 요청입니다. 용량, 접근 모드, 스토리지 클래스를 기반으로
요청을 충족하는 PV에 바인딩됩니다.

### 3.1 기본 PVC

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-claim
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard       # StorageClass와 매칭
  # selector:                      # 선택 사항: 특정 PV 선택
  #   matchLabels:
  #     type: nfs
```

### 3.2 파드에서 PVC 사용

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-storage
spec:
  containers:
    - name: app
      image: postgres:16
      volumeMounts:
        - name: pg-data
          mountPath: /var/lib/postgresql/data
      env:
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
  volumes:
    - name: pg-data
      persistentVolumeClaim:
        claimName: data-claim
        # readOnly: false          # 기본값
```

### 3.3 PV-PVC 바인딩

**동적 프로비저닝 흐름** (클라우드 환경에서 가장 일반적):

```
PVC 생성
    │
    ▼
StorageClass 선택 (storageClassName 필드로)
    │
    ▼
동적 프로비저너 알림 (예: ebs.csi.aws.com)
    │
    ▼
PV 자동 생성 (클라우드 API 호출: CreateVolume)
    │
    ▼
PVC ←── Bound ──→ PV
    │
    ▼
파드가 볼륨을 마운트함
```

**정적 프로비저닝 흐름** (미리 생성된 PV):

```
관리자가 PV 생성  ──→  PVC 생성  ──→  Kubernetes가 바인딩 (StorageClass +
                                       용량 + 접근 모드 매칭)
```

바인딩 알고리즘은 다음을 기반으로 PVC를 PV에 매칭합니다:

1. **StorageClass**: 정확히 일치해야 함
2. **접근 모드**: PV가 요청된 모든 모드를 지원해야 함
3. **용량**: PV 용량 >= PVC 요청
4. **셀렉터**: 지정된 경우, PV 레이블이 일치해야 함
5. **볼륨 모드**: 일치해야 함 (Filesystem 또는 Block)

```bash
# 바인딩 상태 확인
kubectl get pvc data-claim

# 출력:
# NAME         STATUS   VOLUME         CAPACITY   ACCESS MODES   STORAGECLASS
# data-claim   Bound    pv-nfs-data    100Gi      RWX            nfs-slow

# Pending이면 이벤트에서 이유 확인
kubectl describe pvc data-claim | grep -A 10 Events
```

### 3.4 PVC 확장

바인딩된 PVC 확장 (`allowVolumeExpansion: true`인 StorageClass 필요):

```bash
# PVC를 편집하여 크기 증가
kubectl patch pvc data-claim -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'

# 확장 상태 확인
kubectl get pvc data-claim
kubectl describe pvc data-claim | grep -A 5 Conditions

# 파일시스템 확장의 경우 파드 재시작이 필요할 수 있음
# (CSI 드라이버의 온라인 확장 지원에 따라 다름)
```

---

## 4. StorageClass와 동적 프로비저닝

StorageClass는 **동적 프로비저닝(Dynamic Provisioning)**을 가능하게 합니다 — PVC가
클래스에서 스토리지를 요청하면 PV가 자동으로 생성됩니다.

### 4.1 StorageClass 정의

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
  annotations:
    storageclass.kubernetes.io/is-default-class: "false"
provisioner: ebs.csi.aws.com     # CSI 드라이버 이름
parameters:
  type: gp3
  iops: "5000"
  throughput: "250"
  encrypted: "true"
  fsType: ext4
reclaimPolicy: Delete            # PVC 삭제 시 PV 삭제
allowVolumeExpansion: true       # PVC 크기 조정 허용
volumeBindingMode: WaitForFirstConsumer  # 파드가 스케줄될 때까지 바인딩 지연
mountOptions:
  - discard
  - noatime
```

### 4.2 볼륨 바인딩 모드

| 모드 | 동작 | 사용 사례 |
|------|----------|----------|
| Immediate | PVC가 생성되자마자 PV 프로비저닝 | 네트워크 연결 스토리지 |
| WaitForFirstConsumer | PVC를 사용하는 파드가 스케줄될 때 PV 프로비저닝 | 로컬 또는 존 특화 스토리지 |

`WaitForFirstConsumer`는 토폴로지 인식 스토리지에 중요합니다:

```yaml
# WaitForFirstConsumer 없이, PV가 존 A에 프로비저닝될 수 있지만
# 파드가 존 B에 스케줄됨 → Pending 상태에 빠짐

apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: zone-aware
provisioner: ebs.csi.aws.com
volumeBindingMode: WaitForFirstConsumer   # 파드와 같은 존에 PV 생성
allowedTopologies:
  - matchLabelExpressions:
      - key: topology.kubernetes.io/zone
        values:
          - us-east-1a
          - us-east-1b
```

### 4.3 기본 StorageClass

하나의 StorageClass를 기본으로 표시할 수 있습니다. `storageClassName`이 없는 PVC가 이를 사용합니다.

```bash
# 기본 StorageClass 확인
kubectl get storageclass

# 출력:
# NAME                 PROVISIONER                RECLAIMPOLICY   VOLUMEBINDINGMODE
# standard (default)   k8s.io/minikube-hostpath   Delete          Immediate
# fast-ssd             ebs.csi.aws.com            Delete          WaitForFirstConsumer

# StorageClass를 기본으로 설정
kubectl patch storageclass fast-ssd -p \
  '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

# 이전 클래스에서 기본 제거
kubectl patch storageclass standard -p \
  '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}'
```

### 4.4 클라우드 공급자 StorageClass

**AWS EBS:**
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: aws-gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
  kmsKeyId: "arn:aws:kms:us-east-1:123456:key/abcd-1234"
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
```

**GCP 영구 디스크(Persistent Disk):**
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gcp-ssd
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-ssd
  replication-type: regional-pd    # 리전 복제
reclaimPolicy: Retain
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
```

**Azure 디스크(Disk):**
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: azure-premium
provisioner: disk.csi.azure.com
parameters:
  skuName: Premium_LRS
  cachingMode: ReadOnly
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
```

---

## 5. 접근 모드

접근 모드는 볼륨이 노드에 마운트되는 방식을 정의합니다.

| 모드 | 약어 | 설명 |
|------|-------------|-------------|
| ReadWriteOnce | RWO | 단일 노드에서 읽기-쓰기로 마운트 |
| ReadOnlyMany | ROX | 여러 노드에서 읽기 전용으로 마운트 |
| ReadWriteMany | RWX | 여러 노드에서 읽기-쓰기로 마운트 |
| ReadWriteOncePod | RWOP | 단일 파드에서 읽기-쓰기로 마운트 (v1.29+) |

### 5.1 스토리지 유형별 접근 모드 지원

| 스토리지 유형 | RWO | ROX | RWX | RWOP |
|-------------|-----|-----|-----|------|
| AWS EBS | 예 | 아니오 | 아니오 | 예 |
| GCP PD | 예 | 예 | 아니오 | 예 |
| Azure Disk | 예 | 아니오 | 아니오 | 예 |
| NFS | 예 | 예 | 예 | 아니오 |
| CephFS | 예 | 예 | 예 | 아니오 |
| 로컬 볼륨 | 예 | 아니오 | 아니오 | 예 |

### 5.2 ReadWriteOncePod (RWOP)

파드 수준에서 배타적 접근을 보장합니다 (RWO보다 엄격):

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: exclusive-claim
spec:
  accessModes:
    - ReadWriteOncePod       # 단 하나의 파드만 이것을 마운트할 수 있음
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

RWO vs RWOP:
- **RWO**: *같은 노드*의 여러 파드가 볼륨을 마운트할 수 있음
- **RWOP**: 전체 클러스터에서 *하나의 파드*만 볼륨을 마운트할 수 있음

---

## 6. 회수 정책

PVC가 삭제되면 회수 정책(Reclaim Policy)이 PV에 일어나는 일을 결정합니다.

| 정책 | 동작 | 사용 사례 |
|--------|----------|----------|
| Retain | PV가 유지됨; 데이터 보존; 수동 정리 필요 | 프로덕션 데이터 |
| Delete | PV와 기반 스토리지가 삭제됨 | 임시/재현 가능한 데이터 |
| Recycle | 더 이상 사용하지 않음; `rm -rf /vol/*` 수행 | 레거시 전용 |

### 6.1 회수 정책 변경

```bash
# 기존 PV의 회수 정책 변경
kubectl patch pv pv-nfs-data -p '{"spec":{"persistentVolumeReclaimPolicy":"Retain"}}'
```

### 6.2 Released PV 복구

PVC가 삭제되고 PV에 `Retain` 정책이 있으면, PV는 `Released` 상태가 됩니다.
다시 바인딩하려면:

```bash
# 1. Released PV 확인
kubectl get pv pv-nfs-data
# STATUS: Released

# 2. claimRef를 제거하여 Available로 만들기
kubectl patch pv pv-nfs-data -p '{"spec":{"claimRef":null}}'

# 3. PV가 이제 Available이며 새 PVC에 바인딩될 수 있음
kubectl get pv pv-nfs-data
# STATUS: Available
```

> **경고**: 다시 바인딩할 때 데이터에 주의하세요. 기존 데이터가 볼륨에
> 남아 있습니다. 새 소비자가 기존 데이터를 예상하거나 처리할 수 있는지 확인하세요.

---

## 7. CSI (컨테이너 스토리지 인터페이스)

CSI는 쿠버네티스와 외부 스토리지 시스템 간의 표준 인터페이스입니다.
인트리(In-tree) 볼륨 플러그인을 아웃오브트리(Out-of-tree) 드라이버로 대체했습니다.

CSI는 스토리지 책임을 두 가지 플러그인 유형으로 분리합니다:

```
컨테이너 오케스트레이터 (CO)            노드
┌──────────────────────────┐          ┌────────────────────────────┐
│  API Server / Controller  │          │  kubelet                   │
│                           │          │    │                        │
│  external-provisioner ────┼──gRPC───▶│  Node Plugin (DaemonSet)  │
│  (PVC 감시)               │          │    │                        │
│          │                │          │    ▼                        │
│          ▼                │          │  NodeStageVolume()          │
│  Controller Plugin        │          │  NodePublishVolume()        │
│  (Deployment)             │          └────────────────────────────┘
│    CreateVolume()         │
│    DeleteVolume()         │
│    ControllerPublish()    │
└──────────────────────────┘
```

### 7.1 CSI 아키텍처

```
┌─────────────────────────────────────────────────┐
│                Kubernetes                        │
│  ┌──────────────┐         ┌──────────────────┐  │
│  │  API Server   │         │  kubelet         │  │
│  └──────┬───────┘         └────────┬─────────┘  │
│         │                          │             │
│  ┌──────┴───────┐         ┌────────┴─────────┐  │
│  │  External     │         │  CSI Node        │  │
│  │  Provisioner  │         │  Driver Plugin   │  │
│  │  Sidecar      │         │  (per node)      │  │
│  └──────┬───────┘         └────────┬─────────┘  │
│         │                          │             │
└─────────┼──────────────────────────┼─────────────┘
          │     CSI gRPC API         │
    ┌─────┴──────────────────────────┴─────┐
    │         CSI Driver Controller         │
    │  (CreateVolume, DeleteVolume,          │
    │   ControllerPublish, Snapshot)         │
    └──────────────────┬───────────────────┘
                       │
              ┌────────┴────────┐
              │  Storage Backend │
              │  (AWS EBS, etc.) │
              └─────────────────┘
```

### 7.2 CSI 드라이버 컴포넌트

| 컴포넌트 | 실행 위치 | 목적 |
|-----------|---------|---------|
| 컨트롤러 플러그인 | 디플로이먼트 (1-3 레플리카) | 볼륨 생성/삭제, 스냅샷 |
| 노드 플러그인 | 데몬셋 (모든 노드) | 볼륨 마운트/언마운트, 포맷 |
| External Provisioner | 컨트롤러와 함께 사이드카 | PVC 감시, CreateVolume 트리거 |
| External Attacher | 컨트롤러와 함께 사이드카 | 볼륨을 노드에 연결 |
| External Snapshotter | 컨트롤러와 함께 사이드카 | 스냅샷 생성/삭제 |
| Node Driver Registrar | 노드 플러그인과 함께 사이드카 | kubelet에 드라이버 등록 |

### 7.3 CSI 드라이버 설치 (AWS EBS 예시)

```bash
# Helm을 사용하여 AWS EBS CSI 드라이버 설치
helm repo add aws-ebs-csi-driver https://kubernetes-sigs.github.io/aws-ebs-csi-driver
helm repo update

helm install aws-ebs-csi-driver aws-ebs-csi-driver/aws-ebs-csi-driver \
  --namespace kube-system \
  --set controller.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="arn:aws:iam::123456:role/ebs-csi-role"

# 드라이버가 실행 중인지 확인
kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-ebs-csi-driver

# CSIDriver 오브젝트 확인
kubectl get csidriver ebs.csi.aws.com -o yaml
```

### 7.4 CSIDriver 오브젝트

```yaml
apiVersion: storage.k8s.io/v1
kind: CSIDriver
metadata:
  name: ebs.csi.aws.com
spec:
  attachRequired: true           # ControllerPublishVolume 필요
  podInfoOnMount: false          # NodePublishVolume에 파드 정보 전달하지 않음
  fsGroupPolicy: File            # fsGroup 소유권 적용
  volumeLifecycleModes:
    - Persistent                 # PV/PVC 워크플로우 지원
  storageCapacity: true          # 스토리지 용량 보고
  tokenRequests: []
  requiresRepublish: false
  seLinuxMount: false
```

### 7.5 스토리지 용량 추적

CSI 드라이버는 토폴로지별 가용 스토리지 용량을 보고할 수 있습니다:

```bash
# 스토리지 용량 확인 (드라이버가 지원하는 경우)
kubectl get csistoragecapacities -A

# 출력:
# NAMESPACE     NAME             STORAGE CLASS   CAPACITY     NODE TOPOLOGY
# kube-system   csi-cap-abc123   fast-ssd        450Gi        node=worker-1
# kube-system   csi-cap-def456   fast-ssd        320Gi        node=worker-2
```

스케줄러는 이 정보를 사용하여 스토리지가 충분하지 않은 노드에 파드를
스케줄링하는 것을 방지합니다.

---

## 8. 볼륨 스냅샷

볼륨 스냅샷은 볼륨의 특정 시점 복사본을 생성합니다. 스냅샷을 지원하는 CSI
드라이버와 스냅샷 컨트롤러가 필요합니다.

### 8.1 VolumeSnapshotClass

```yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshotClass
metadata:
  name: ebs-snapshot-class
driver: ebs.csi.aws.com
deletionPolicy: Delete           # VolumeSnapshot 삭제 시 스냅샷 삭제
# deletionPolicy: Retain         # VolumeSnapshot 삭제 후에도 스냅샷 유지
parameters:
  # 드라이버 특화 파라미터
  tagSpecification_1: "backup=true"
```

### 8.2 스냅샷 생성

```yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: data-snapshot-20240115
spec:
  volumeSnapshotClassName: ebs-snapshot-class
  source:
    persistentVolumeClaimName: data-claim    # 소스 PVC
```

```bash
# 스냅샷 생성
kubectl apply -f snapshot.yaml

# 스냅샷 상태 확인
kubectl get volumesnapshot data-snapshot-20240115

# 출력:
# NAME                      READYTOUSE   RESTORESIZE   SNAPSHOTCLASS
# data-snapshot-20240115     true         50Gi          ebs-snapshot-class

# 상세 상태
kubectl describe volumesnapshot data-snapshot-20240115
```

### 8.3 스냅샷에서 복원

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-restored
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
  dataSource:
    name: data-snapshot-20240115
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
```

### 8.4 PVC 복제

스냅샷을 거치지 않고 기존 PVC에서 새 PVC를 생성합니다:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-clone
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
  dataSource:
    name: data-claim             # 소스 PVC
    kind: PersistentVolumeClaim
    # PVC 복제에는 apiGroup 불필요
```

### 8.5 예약된 스냅샷

쿠버네티스에는 내장 스냅샷 스케줄링이 없습니다. 크론잡(CronJob)을 사용합니다:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: snapshot-scheduler
spec:
  schedule: "0 */6 * * *"        # 6시간마다
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: snapshot-creator
          restartPolicy: OnFailure
          containers:
            - name: snapshot
              image: bitnami/kubectl:1.29
              command:
                - sh
                - -c
                - |
                  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
                  cat <<SNAP | kubectl apply -f -
                  apiVersion: snapshot.storage.k8s.io/v1
                  kind: VolumeSnapshot
                  metadata:
                    name: data-snap-${TIMESTAMP}
                    labels:
                      app: scheduled-snapshot
                  spec:
                    volumeSnapshotClassName: ebs-snapshot-class
                    source:
                      persistentVolumeClaimName: data-claim
                  SNAP
                  echo "Snapshot data-snap-${TIMESTAMP} created"
```

---

## 9. 임시 볼륨

임시 볼륨(Ephemeral Volumes)은 파드와 함께 생성되고 삭제됩니다. 영속이 필요 없는
임시 데이터에 유용합니다.

### 9.1 일반 임시 볼륨(Generic Ephemeral Volumes)

파드 범위 스토리지에 모든 StorageClass를 사용합니다:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ephemeral-demo
spec:
  containers:
    - name: app
      image: my-app:v1.0
      volumeMounts:
        - name: scratch
          mountPath: /scratch
  volumes:
    - name: scratch
      ephemeral:
        volumeClaimTemplate:
          spec:
            accessModes: ["ReadWriteOnce"]
            storageClassName: fast-ssd
            resources:
              requests:
                storage: 10Gi
```

시스템이 `<pod-name>-scratch` (파드 이름 + 볼륨 이름)이라는 PVC를 생성합니다.
파드가 삭제되면 PVC도 자동으로 삭제됩니다.

### 9.2 CSI 임시 볼륨

일부 CSI 드라이버는 인라인 임시 볼륨을 지원합니다:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: csi-ephemeral-demo
spec:
  containers:
    - name: app
      image: my-app:v1.0
      volumeMounts:
        - name: secret-store
          mountPath: /mnt/secrets
          readOnly: true
  volumes:
    - name: secret-store
      csi:
        driver: secrets-store.csi.k8s.io
        readOnly: true
        volumeAttributes:
          secretProviderClass: aws-secrets
```

### 9.3 emptyDir vs 일반 임시 볼륨

| 기능 | emptyDir | 일반 임시 볼륨 |
|---------|----------|-------------------|
| 스토리지 백엔드 | 노드 디스크 또는 tmpfs | 모든 StorageClass |
| 크기 적용 | sizeLimit (소프트) | PVC 쿼터 (하드) |
| 성능 | 노드 로컬 | 백엔드에 따라 다름 |
| 스냅샷 지원 | 아니오 | 예 (드라이버가 지원하는 경우) |
| 메트릭 | 제한적 | 전체 CSI 메트릭 |

---

## 10. 스테이트풀셋 스토리지 패턴

### 10.1 VolumeClaimTemplates

스테이트풀셋(StatefulSets)은 `volumeClaimTemplates`를 사용하여 각 파드에 고유한 PVC를 생성합니다:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: elasticsearch
spec:
  serviceName: es-headless
  replicas: 3
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      initContainers:
        - name: fix-permissions
          image: busybox:1.36
          command: ["sh", "-c", "chown -R 1000:1000 /usr/share/elasticsearch/data"]
          volumeMounts:
            - name: data
              mountPath: /usr/share/elasticsearch/data
      containers:
        - name: elasticsearch
          image: elasticsearch:8.12.0
          ports:
            - containerPort: 9200
              name: http
            - containerPort: 9300
              name: transport
          env:
            - name: cluster.name
              value: "k8s-cluster"
            - name: node.name
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: discovery.seed_hosts
              value: "es-headless"
            - name: cluster.initial_master_nodes
              value: "elasticsearch-0,elasticsearch-1,elasticsearch-2"
          volumeMounts:
            - name: data
              mountPath: /usr/share/elasticsearch/data
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"

  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 100Gi
```

이것은 다음을 생성합니다:
```
PVC: data-elasticsearch-0 → PV (100Gi, fast-ssd)
PVC: data-elasticsearch-1 → PV (100Gi, fast-ssd)
PVC: data-elasticsearch-2 → PV (100Gi, fast-ssd)
```

### 10.2 PVC 보존 정책 (v1.27+)

스테이트풀셋이 스케일다운되거나 삭제될 때 PVC에 일어나는 일을 제어합니다:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: elasticsearch
spec:
  persistentVolumeClaimRetentionPolicy:
    whenDeleted: Retain      # 스테이트풀셋 삭제 시 PVC 유지
    whenScaled: Delete       # 스케일다운 시 PVC 삭제
    # 옵션: Retain (기본) 또는 Delete
```

### 10.3 파드당 다중 볼륨

```yaml
volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
  - metadata:
      name: wal               # WAL(Write-Ahead Log)용 별도 볼륨
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: ultra-ssd   # WAL용 더 빠른 스토리지
      resources:
        requests:
          storage: 20Gi
```

### 10.4 데이터 마이그레이션 패턴

스토리지를 업그레이드할 때 `volumeClaimTemplates`를 수정할 수 없습니다. 이 패턴을 사용하세요:

```bash
# 1. 스테이트풀셋 스케일다운
kubectl scale statefulset elasticsearch --replicas=0

# 2. 모든 PVC의 스냅샷 생성
for i in 0 1 2; do
  cat <<EOF | kubectl apply -f -
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: data-migration-$i
spec:
  source:
    persistentVolumeClaimName: data-elasticsearch-$i
EOF
done

# 3. 업데이트된 스토리지 클래스로 스냅샷에서 새 PVC 생성
for i in 0 1 2; do
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-elasticsearch-new-$i
spec:
  accessModes: ["ReadWriteOnce"]
  storageClassName: new-fast-ssd
  resources:
    requests:
      storage: 200Gi
  dataSource:
    name: data-migration-$i
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
EOF
done

# 4. 이전 PVC 삭제 및 새 것으로 교체
for i in 0 1 2; do
  kubectl delete pvc data-elasticsearch-$i
  # 참고: kubectl은 PVC 이름을 바꿀 수 없음; 스테이트풀셋을 재생성해야 할 수 있음
done

# 5. 스테이트풀셋 스케일업
kubectl scale statefulset elasticsearch --replicas=3
```

---

## 연습문제

### 연습문제 1: PV와 PVC 바인딩

5Gi 용량의 PersistentVolume과 3Gi를 요청하는 PersistentVolumeClaim을 생성합니다.
올바르게 바인딩되는지 확인한 다음, PVC를 파드에 마운트하고 데이터를 씁니다.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/pv-pvc-exercise.yaml로 저장
apiVersion: v1
kind: PersistentVolume
metadata:
  name: exercise-pv
spec:
  capacity:
    storage: 5Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Delete
  storageClassName: manual
  hostPath:
    path: /tmp/exercise-pv-data
    type: DirectoryOrCreate
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: exercise-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 3Gi
  storageClassName: manual
---
apiVersion: v1
kind: Pod
metadata:
  name: storage-writer
spec:
  containers:
    - name: writer
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "Hello from persistent storage!" > /data/greeting.txt
          echo "Written at $(date)" >> /data/greeting.txt
          cat /data/greeting.txt
          sleep 3600
      volumeMounts:
        - name: persistent-data
          mountPath: /data
  volumes:
    - name: persistent-data
      persistentVolumeClaim:
        claimName: exercise-pvc
```

```bash
kubectl apply -f /tmp/pv-pvc-exercise.yaml

# PV와 PVC가 바인딩되었는지 확인
kubectl get pv exercise-pv
# STATUS: Bound

kubectl get pvc exercise-pvc
# STATUS: Bound, VOLUME: exercise-pv

# 데이터가 작성되었는지 확인
kubectl wait --for=condition=Ready pod/storage-writer --timeout=60s
kubectl exec storage-writer -- cat /data/greeting.txt

# 파드 삭제, 같은 PVC로 새 파드 생성 — 데이터가 영속됨
kubectl delete pod storage-writer
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: storage-reader
spec:
  containers:
    - name: reader
      image: busybox:1.36
      command: ["sh", "-c", "cat /data/greeting.txt && sleep 3600"]
      volumeMounts:
        - name: persistent-data
          mountPath: /data
  volumes:
    - name: persistent-data
      persistentVolumeClaim:
        claimName: exercise-pvc
EOF
kubectl wait --for=condition=Ready pod/storage-reader --timeout=60s
kubectl exec storage-reader -- cat /data/greeting.txt
# 첫 번째 파드가 작성한 데이터가 표시되어야 함

# 정리
kubectl delete pod storage-reader
kubectl delete pvc exercise-pvc
kubectl delete pv exercise-pv
```

</details>

### 연습문제 2: 동적 프로비저닝

minikube의 hostpath 프로비저너용 StorageClass를 생성합니다. 그런 다음 동적
프로비저닝을 사용하는 PVC를 생성합니다. PV가 자동으로 생성되는지 확인하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/dynamic-provision.yaml로 저장
# minikube에서는 기본 StorageClass가 이미 동적 프로비저닝을 지원합니다
# 커스텀 StorageClass를 생성해봅시다
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: custom-hostpath
provisioner: k8s.io/minikube-hostpath
reclaimPolicy: Delete
volumeBindingMode: Immediate
allowVolumeExpansion: true
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dynamic-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
  storageClassName: custom-hostpath
```

```bash
kubectl apply -f /tmp/dynamic-provision.yaml

# 동적 프로비저닝 확인
kubectl get pvc dynamic-pvc
# STATUS: Bound (거의 즉시)

# 자동 생성된 PV 확인
kubectl get pv
# "pvc-<uuid>" 이름의 새 PV가 나타나야 함, dynamic-pvc에 바인딩됨

PV_NAME=$(kubectl get pvc dynamic-pvc -o jsonpath='{.spec.volumeName}')
kubectl describe pv $PV_NAME
# Source.Type은 HostPath여야 함
# StorageClass는 custom-hostpath여야 함

# 파드로 테스트
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: dynamic-test
spec:
  containers:
    - name: test
      image: busybox:1.36
      command: ["sh", "-c", "echo 'Dynamic provisioning works!' > /data/test.txt && cat /data/test.txt && sleep 3600"]
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: dynamic-pvc
EOF
kubectl wait --for=condition=Ready pod/dynamic-test --timeout=60s
kubectl exec dynamic-test -- cat /data/test.txt

# 정리
kubectl delete pod dynamic-test
kubectl delete pvc dynamic-pvc
kubectl delete storageclass custom-hostpath
```

</details>

### 연습문제 3: 볼륨 스냅샷

PVC를 생성하고 데이터를 쓴 다음, 볼륨 스냅샷을 찍고, 스냅샷을 새 PVC로
복원하여 데이터를 확인합니다.

<details>
<summary>정답 보기</summary>

```yaml
# 참고: 이 연습문제는 스냅샷을 지원하는 CSI 드라이버가 필요합니다.
# minikube에서는 volumesnapshots 애드온을 활성화하세요:
# minikube addons enable volumesnapshots
# minikube addons enable csi-hostpath-driver

# /tmp/snapshot-exercise.yaml로 저장
# 1단계: 소스 PVC 생성 및 데이터 쓰기
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: source-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  storageClassName: csi-hostpath-sc    # minikube CSI hostpath
---
apiVersion: v1
kind: Pod
metadata:
  name: data-writer
spec:
  containers:
    - name: writer
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "Original data written at $(date)" > /data/important.txt
          echo "This data will survive a snapshot restore" >> /data/important.txt
          cat /data/important.txt
          sleep 3600
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: source-pvc
```

```bash
# minikube에서 필요한 애드온 활성화
minikube addons enable volumesnapshots
minikube addons enable csi-hostpath-driver

# 소스 PVC 생성 및 데이터 쓰기
kubectl apply -f /tmp/snapshot-exercise.yaml
kubectl wait --for=condition=Ready pod/data-writer --timeout=120s
kubectl exec data-writer -- cat /data/important.txt

# 스냅샷 생성
cat <<EOF | kubectl apply -f -
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: data-snapshot
spec:
  volumeSnapshotClassName: csi-hostpath-snapclass
  source:
    persistentVolumeClaimName: source-pvc
EOF

# 스냅샷이 준비될 때까지 대기
kubectl get volumesnapshot data-snapshot -w
# READYTOUSE가 true가 되어야 함

# 스냅샷에서 복원
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: restored-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  storageClassName: csi-hostpath-sc
  dataSource:
    name: data-snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
---
apiVersion: v1
kind: Pod
metadata:
  name: data-reader
spec:
  containers:
    - name: reader
      image: busybox:1.36
      command: ["sh", "-c", "cat /data/important.txt && sleep 3600"]
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: restored-pvc
EOF

kubectl wait --for=condition=Ready pod/data-reader --timeout=120s
kubectl exec data-reader -- cat /data/important.txt
# 원본 데이터가 표시되어야 함

# 정리
kubectl delete pod data-writer data-reader
kubectl delete pvc source-pvc restored-pvc
kubectl delete volumesnapshot data-snapshot
```

</details>

### 연습문제 4: 스토리지를 가진 스테이트풀셋

3개의 레플리카를 가진 스테이트풀셋을 생성합니다. 각각 자체 1Gi PVC를 가집니다.
각 파드의 볼륨에 고유한 데이터를 씁니다. 1개 레플리카로 스케일다운한 다음 3개로
복구하고, 데이터가 영속되는지 확인하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/sts-storage.yaml로 저장
apiVersion: v1
kind: Service
metadata:
  name: sts-headless
spec:
  clusterIP: None
  selector:
    app: sts-storage-demo
  ports:
    - port: 80
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: sts-storage-demo
spec:
  serviceName: sts-headless
  replicas: 3
  selector:
    matchLabels:
      app: sts-storage-demo
  template:
    metadata:
      labels:
        app: sts-storage-demo
    spec:
      containers:
        - name: app
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              # 파일이 없으면 파드 정체성을 스토리지에 기록
              if [ ! -f /data/identity.txt ]; then
                echo "Pod $(hostname) created at $(date)" > /data/identity.txt
              fi
              echo "=== Stored Identity ==="
              cat /data/identity.txt
              sleep 3600
          volumeMounts:
            - name: data
              mountPath: /data
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 1Gi
```

```bash
kubectl apply -f /tmp/sts-storage.yaml
kubectl rollout status statefulset/sts-storage-demo

# 각 파드가 자체 PVC를 가지는지 확인
kubectl get pvc -l app=sts-storage-demo
# data-sts-storage-demo-0   Bound   1Gi
# data-sts-storage-demo-1   Bound   1Gi
# data-sts-storage-demo-2   Bound   1Gi

# 각 파드의 고유 데이터 읽기
for i in 0 1 2; do
  echo "=== Pod $i ==="
  kubectl exec sts-storage-demo-$i -- cat /data/identity.txt
done

# 1개로 스케일다운
kubectl scale statefulset sts-storage-demo --replicas=1

# PVC는 삭제되지 않음 (기본적으로 보존됨)
kubectl get pvc -l app=sts-storage-demo
# 3개의 PVC가 모두 여전히 존재

# 3개로 다시 스케일업
kubectl scale statefulset sts-storage-demo --replicas=3
kubectl rollout status statefulset/sts-storage-demo

# 데이터 영속성 확인
for i in 0 1 2; do
  echo "=== Pod $i ==="
  kubectl exec sts-storage-demo-$i -- cat /data/identity.txt
done
# 각 파드가 원래 생성 시간을 표시해야 함

# 정리
kubectl delete statefulset sts-storage-demo
kubectl delete svc sts-headless
kubectl delete pvc -l app=sts-storage-demo
```

</details>

### 연습문제 5: 임시 볼륨

임시 공간을 위해 일반 임시 볼륨을 사용하는 파드를 생성합니다. 데이터를 쓰고,
파드를 삭제하고, 볼륨이 정리되는지 확인하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/ephemeral-exercise.yaml로 저장
apiVersion: v1
kind: Pod
metadata:
  name: ephemeral-worker
spec:
  containers:
    - name: processor
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "Processing data..."
          # 임시 데이터 쓰기
          for i in $(seq 1 100); do
            echo "Record $i: $(date)" >> /scratch/output.csv
          done
          echo "Wrote $(wc -l < /scratch/output.csv) records to scratch volume"
          ls -la /scratch/
          sleep 3600
      volumeMounts:
        - name: scratch-space
          mountPath: /scratch
      resources:
        requests:
          cpu: "50m"
          memory: "64Mi"
  volumes:
    - name: scratch-space
      ephemeral:
        volumeClaimTemplate:
          spec:
            accessModes: ["ReadWriteOnce"]
            resources:
              requests:
                storage: 1Gi
```

```bash
kubectl apply -f /tmp/ephemeral-exercise.yaml
kubectl wait --for=condition=Ready pod/ephemeral-worker --timeout=120s

# 자동 생성된 PVC 확인
kubectl get pvc
# NAME                              STATUS   VOLUME          CAPACITY
# ephemeral-worker-scratch-space    Bound    pvc-xxxxxxxx    1Gi

# 데이터가 작성되었는지 확인
kubectl exec ephemeral-worker -- cat /scratch/output.csv | head -5

# PVC 이름 확인
PVC_NAME=$(kubectl get pvc -o jsonpath='{.items[?(@.metadata.name=="ephemeral-worker-scratch-space")].metadata.name}')
echo "Ephemeral PVC: $PVC_NAME"

# 파드 삭제
kubectl delete pod ephemeral-worker

# PVC가 자동으로 삭제되는지 확인
kubectl get pvc
# ephemeral-worker-scratch-space가 사라져야 함 (파드가 소유)

# 아직 잠시 존재한다면 잠깐 대기
sleep 5
kubectl get pvc
# 사라져야 함

echo "Ephemeral volume was automatically cleaned up"
```

</details>

---

**이전**: [네트워킹 기초](./03_Networking_Fundamentals.md) | **다음**: [구성과 시크릿](./05_Configuration_and_Secrets.md)
