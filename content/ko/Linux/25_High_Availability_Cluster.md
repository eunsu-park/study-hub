# 고가용성 클러스터

**이전**: [클라우드 통합](./24_Cloud_Integration.md) | **다음**: [트러블슈팅 가이드](./26_Troubleshooting_Guide.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. Corosync, Pacemaker, DRBD를 포함한 HA 클러스터 아키텍처의 구성 요소를 설명한다
2. 적절한 쿼럼(Quorum) 및 속성 설정을 통해 pcs로 2노드 클러스터를 구성한다
3. 가상 IP(Virtual IP), 파일시스템, 서비스를 포함한 Pacemaker 리소스를 생성하고 관리한다
4. 올바른 페일오버(Failover)를 위한 리소스 그룹, 순서(Ordering) 제약, 배치(Colocation) 제약을 정의한다
5. 클러스터 노드 간 실시간 블록 수준 복제를 위해 DRBD를 설정한다
6. 프로덕션 환경에서 스플릿 브레인(Split-Brain) 시나리오를 방지하기 위해 STONITH 펜싱(Fencing)을 구현한다
7. 페일오버 절차를 테스트하고 crm_mon, pcs로 클러스터 상태를 모니터링한다

**난이도**: ⭐⭐⭐⭐⭐ (최고급)

---

## 목차

1. [고가용성 개요](#1-고가용성-개요)
2. [Corosync 설정](#2-corosync-설정)
3. [Pacemaker 설정](#3-pacemaker-설정)
4. [리소스 관리](#4-리소스-관리)
5. [DRBD 설정](#5-drbd-설정)
6. [Fencing (STONITH)](#6-fencing-stonith)
7. [실전 클러스터 구성](#7-실전-클러스터-구성)

---

데이터베이스, 웹 애플리케이션, 파일 서버와 같은 핵심 서비스에서는 단 몇 분의 다운타임도 매출 손실과 SLA 위반으로 이어질 수 있습니다. 고가용성(High Availability) 클러스터링은 하나의 노드가 장애를 겪을 때 자동으로 인계받는 여러 노드에서 서비스를 실행함으로써 단일 장애점(Single Point of Failure)을 제거합니다. 이 레슨은 클러스터 통신을 담당하는 Corosync, 리소스 관리를 담당하는 Pacemaker, 데이터 복제를 담당하는 DRBD, 스플릿 브레인(Split-Brain) 방지를 위한 펜싱(Fencing)으로 구성된 완전한 HA 스택을 다루며, 수동 개입 없이 하드웨어 장애를 견딜 수 있는 시스템을 구축하는 역량을 제공합니다.

## 1. 고가용성 개요

### HA 클러스터 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    HA Cluster                               │
│                                                             │
│  ┌───────────────────┐       ┌───────────────────┐         │
│  │      Node 1       │       │      Node 2       │         │
│  │   (Active)        │       │   (Standby)       │         │
│  │                   │       │                   │         │
│  │  ┌─────────────┐  │       │  ┌─────────────┐  │         │
│  │  │ Pacemaker   │◄─┼───────┼─►│ Pacemaker   │  │         │
│  │  └─────────────┘  │       │  └─────────────┘  │         │
│  │         │         │       │         │         │         │
│  │  ┌─────────────┐  │       │  ┌─────────────┐  │         │
│  │  │  Corosync   │◄─┼───────┼─►│  Corosync   │  │         │
│  │  └─────────────┘  │       │  └─────────────┘  │         │
│  │         │         │       │         │         │         │
│  │  ┌─────────────┐  │       │  ┌─────────────┐  │         │
│  │  │   DRBD      │◄─┼───────┼─►│   DRBD      │  │         │
│  │  │  (Primary)  │  │       │  │ (Secondary) │  │         │
│  │  └─────────────┘  │       │  └─────────────┘  │         │
│  │         │         │       │                   │         │
│  │  ┌─────────────┐  │       │                   │         │
│  │  │ Application │  │       │                   │         │
│  │  │  (Running)  │  │       │                   │         │
│  │  └─────────────┘  │       │                   │         │
│  └───────────────────┘       └───────────────────┘         │
│             │                                               │
│             ▼                                               │
│      ┌─────────────┐                                       │
│      │  Virtual IP │  ← 클라이언트 접속점                   │
│      └─────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### HA 구성 요소

| 구성 요소 | 역할 |
|-----------|------|
| **Corosync** | 클러스터 통신 및 멤버십 관리 |
| **Pacemaker** | 리소스 관리 및 페일오버 |
| **DRBD** | 블록 레벨 스토리지 복제 |
| **STONITH/Fencing** | Split-brain 방지 |

### 패키지 설치

```bash
# Ubuntu/Debian
sudo apt install pacemaker corosync pcs resource-agents fence-agents

# RHEL/CentOS
sudo yum install pacemaker corosync pcs resource-agents fence-agents-all
```

---

## 2. Corosync 설정

### 기본 설정

```bash
# /etc/corosync/corosync.conf
totem {
    version: 2
    cluster_name: mycluster
    transport: knet

    crypto_cipher: aes256
    crypto_hash: sha256

    interface {
        ringnumber: 0
        bindnetaddr: 192.168.1.0
        mcastport: 5405
    }
}

logging {
    to_logfile: yes
    logfile: /var/log/corosync/corosync.log
    to_syslog: yes
    timestamp: on
}

quorum {
    provider: corosync_votequorum
    two_node: 1
    wait_for_all: 1
}

nodelist {
    node {
        ring0_addr: node1.example.com
        nodeid: 1
    }
    node {
        ring0_addr: node2.example.com
        nodeid: 2
    }
}
```

### pcs를 이용한 클러스터 설정

```bash
# pcsd 서비스 시작
sudo systemctl enable pcsd
sudo systemctl start pcsd

# hacluster 사용자 비밀번호 설정 (모든 노드)
sudo passwd hacluster

# 노드 인증 (한 노드에서 실행)
sudo pcs host auth node1 node2

# 클러스터 생성
sudo pcs cluster setup mycluster node1 node2

# 클러스터 시작
sudo pcs cluster start --all

# 클러스터 활성화 (부팅 시 자동 시작)
sudo pcs cluster enable --all

# 상태 확인
sudo pcs cluster status
sudo pcs status
```

### Corosync 상태 확인

```bash
# 멤버십 확인
sudo corosync-cmapctl | grep members

# 쿼럼 상태
sudo corosync-quorumtool

# 링 상태
sudo corosync-cfgtool -s
```

---

## 3. Pacemaker 설정

### 클러스터 속성 설정

```bash
# STONITH 비활성화 (테스트용, 운영에서는 필수)
sudo pcs property set stonith-enabled=false

# 쿼럼 정책
sudo pcs property set no-quorum-policy=ignore  # 2노드 클러스터

# 기본 스티키니스 (리소스 이동 방지)
sudo pcs resource defaults update resource-stickiness=100

# 속성 확인
sudo pcs property list
```

### 클러스터 상태 확인

```bash
# 전체 상태
sudo pcs status

# 리소스 상태
sudo pcs resource status

# 노드 상태
sudo pcs node status

# 제약 조건
sudo pcs constraint list --full

# 클러스터 설정 확인
sudo pcs config
```

---

## 4. 리소스 관리

### 기본 리소스 생성

```bash
# Virtual IP 리소스
sudo pcs resource create VirtualIP ocf:heartbeat:IPaddr2 \
    ip=192.168.1.100 \
    cidr_netmask=24 \
    op monitor interval=30s

# 웹 서버 리소스
sudo pcs resource create WebServer ocf:heartbeat:nginx \
    configfile=/etc/nginx/nginx.conf \
    op start timeout=40s \
    op stop timeout=60s \
    op monitor interval=10s

# 파일시스템 리소스
sudo pcs resource create WebFS ocf:heartbeat:Filesystem \
    device=/dev/drbd0 \
    directory=/var/www \
    fstype=ext4 \
    op start timeout=60s \
    op stop timeout=60s
```

### 리소스 그룹

```bash
# 그룹 생성 (순서대로 시작, 역순으로 중지)
sudo pcs resource group add WebGroup \
    WebFS \
    VirtualIP \
    WebServer

# 그룹 상태 확인
sudo pcs resource show WebGroup
```

### 리소스 제약 조건

```bash
# 위치 제약 (특정 노드 선호)
sudo pcs constraint location WebServer prefers node1=100
sudo pcs constraint location WebServer avoids node2

# 순서 제약 (시작 순서)
sudo pcs constraint order WebFS then VirtualIP then WebServer

# 콜로케이션 제약 (같은 노드에서 실행)
sudo pcs constraint colocation add WebServer with VirtualIP INFINITY
sudo pcs constraint colocation add VirtualIP with WebFS INFINITY

# 제약 조건 확인
sudo pcs constraint list --full
```

### 리소스 관리 명령

```bash
# 리소스 시작/중지
sudo pcs resource enable WebServer
sudo pcs resource disable WebServer

# 리소스 이동 (수동 페일오버)
sudo pcs resource move WebServer node2

# 이동 제약 제거 (원래 위치로 돌아갈 수 있게)
sudo pcs resource clear WebServer

# 리소스 재시작
sudo pcs resource restart WebServer

# 리소스 삭제
sudo pcs resource delete WebServer
```

---

## 5. DRBD 설정

### DRBD 설치

```bash
# Ubuntu/Debian
sudo apt install drbd-utils

# RHEL/CentOS (ELRepo 사용)
sudo rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
sudo yum install https://www.elrepo.org/elrepo-release-9.el9.elrepo.noarch.rpm
sudo yum install drbd90-utils kmod-drbd90
```

### DRBD 리소스 설정

```bash
# /etc/drbd.d/r0.res (양쪽 노드 동일)
resource r0 {
    protocol C;

    startup {
        wfc-timeout  15;
        degr-wfc-timeout 60;
    }

    net {
        cram-hmac-alg sha1;
        shared-secret "mysecret123";
    }

    disk {
        on-io-error detach;
    }

    on node1 {
        device    /dev/drbd0;
        disk      /dev/sdb1;
        address   192.168.1.11:7788;
        meta-disk internal;
    }

    on node2 {
        device    /dev/drbd0;
        disk      /dev/sdb1;
        address   192.168.1.12:7788;
        meta-disk internal;
    }
}
```

### DRBD 초기화

```bash
# 메타데이터 생성 (양쪽 노드)
sudo drbdadm create-md r0

# DRBD 시작 (양쪽 노드)
sudo drbdadm up r0

# Primary 설정 (한 노드에서만)
sudo drbdadm primary --force r0

# 상태 확인
cat /proc/drbd
sudo drbdadm status

# 파일시스템 생성 (Primary에서)
sudo mkfs.ext4 /dev/drbd0
```

### Pacemaker와 DRBD 통합

```bash
# DRBD 리소스 생성
sudo pcs resource create DRBD ocf:linbit:drbd \
    drbd_resource=r0 \
    op monitor interval=60s

# 마스터/슬레이브 설정
sudo pcs resource promotable DRBD \
    promoted-max=1 \
    clone-max=2 \
    notify=true

# 파일시스템 리소스
sudo pcs resource create WebFS ocf:heartbeat:Filesystem \
    device=/dev/drbd0 \
    directory=/var/www \
    fstype=ext4

# 제약 조건: WebFS는 DRBD Primary에서만
sudo pcs constraint colocation add WebFS with DRBD-clone INFINITY with-rsc-role=Master
sudo pcs constraint order promote DRBD-clone then start WebFS
```

### DRBD 상태 확인

```bash
# 상태 확인
sudo drbdadm status
sudo drbdadm dstate r0
sudo drbdadm cstate r0
sudo drbdadm role r0

# 동기화 상태
cat /proc/drbd

# 예시 출력:
# version: 8.4.11
#  0: cs:Connected ro:Primary/Secondary ds:UpToDate/UpToDate C r-----
```

---

## 6. Fencing (STONITH)

### Fencing 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    Split-Brain 상황                         │
│                                                             │
│  ┌───────────┐     네트워크 단절     ┌───────────┐         │
│  │   Node1   │ ────────X──────────  │   Node2   │         │
│  │ (Primary) │                       │ (Primary) │ ← 위험! │
│  └───────────┘                       └───────────┘         │
│       │                                     │               │
│       └─────────────┬───────────────────────┘               │
│                     │                                       │
│              ┌──────┴──────┐                               │
│              │ Shared Data │ ← 데이터 손상 가능!            │
│              └─────────────┘                               │
│                                                             │
│  해결책: STONITH (Shoot The Other Node In The Head)        │
│  - 문제가 있는 노드를 강제로 리셋/전원 차단                 │
└─────────────────────────────────────────────────────────────┘
```

### 가상환경 Fencing (fence_virsh)

```bash
# fence_virsh 에이전트 테스트
sudo fence_virsh -a qemu+ssh://hypervisor -l root \
    -p password -n vm-node1 -o status

# Pacemaker 리소스 설정
sudo pcs stonith create fence_node1 fence_virsh \
    pcmk_host_list="node1" \
    ipaddr="qemu+ssh://hypervisor" \
    login="root" \
    passwd="password" \
    port="vm-node1" \
    ssl_insecure=1

sudo pcs stonith create fence_node2 fence_virsh \
    pcmk_host_list="node2" \
    ipaddr="qemu+ssh://hypervisor" \
    login="root" \
    passwd="password" \
    port="vm-node2" \
    ssl_insecure=1

# 위치 제약 (자신의 노드에서 실행되지 않도록)
sudo pcs constraint location fence_node1 avoids node1
sudo pcs constraint location fence_node2 avoids node2
```

### IPMI Fencing

```bash
# fence_ipmilan 에이전트
sudo pcs stonith create fence_node1_ipmi fence_ipmilan \
    pcmk_host_list="node1" \
    ipaddr="192.168.1.101" \
    login="admin" \
    passwd="password" \
    lanplus=1 \
    power_timeout=20

sudo pcs stonith create fence_node2_ipmi fence_ipmilan \
    pcmk_host_list="node2" \
    ipaddr="192.168.1.102" \
    login="admin" \
    passwd="password" \
    lanplus=1 \
    power_timeout=20
```

### 클라우드 Fencing

```bash
# AWS (fence_aws)
sudo pcs stonith create fence_aws fence_aws \
    pcmk_host_map="node1:i-0123456789abcdef0;node2:i-0fedcba9876543210" \
    region="ap-northeast-2" \
    power_timeout=60

# GCP (fence_gce)
sudo pcs stonith create fence_gcp fence_gce \
    pcmk_host_map="node1:instance-1;node2:instance-2" \
    project="my-project" \
    zone="asia-northeast3-a"
```

### Fencing 테스트

```bash
# STONITH 활성화
sudo pcs property set stonith-enabled=true

# Fencing 테스트
sudo stonith_admin --reboot node2 --verbose

# 수동으로 노드 fence
sudo pcs stonith fence node2

# Fencing 히스토리
sudo stonith_admin --history node2
```

---

## 7. 실전 클러스터 구성

### 2노드 웹 서버 클러스터

```bash
#!/bin/bash
# setup-ha-cluster.sh

# 클러스터 설정
pcs cluster setup mycluster node1 node2

# 클러스터 시작
pcs cluster start --all
pcs cluster enable --all

# 속성 설정
pcs property set stonith-enabled=true
pcs property set no-quorum-policy=ignore

# STONITH 리소스 (예: IPMI)
pcs stonith create fence_node1 fence_ipmilan \
    pcmk_host_list="node1" \
    ipaddr="10.0.0.101" login="admin" passwd="password" lanplus=1

pcs stonith create fence_node2 fence_ipmilan \
    pcmk_host_list="node2" \
    ipaddr="10.0.0.102" login="admin" passwd="password" lanplus=1

pcs constraint location fence_node1 avoids node1
pcs constraint location fence_node2 avoids node2

# DRBD 리소스
pcs resource create DRBD ocf:linbit:drbd drbd_resource=r0 \
    op monitor interval=60s
pcs resource promotable DRBD promoted-max=1 clone-max=2 notify=true

# 파일시스템 리소스
pcs resource create WebFS ocf:heartbeat:Filesystem \
    device=/dev/drbd0 directory=/var/www fstype=ext4

# VIP 리소스
pcs resource create VIP ocf:heartbeat:IPaddr2 \
    ip=192.168.1.100 cidr_netmask=24

# 웹 서버 리소스
pcs resource create WebServer ocf:heartbeat:nginx \
    configfile=/etc/nginx/nginx.conf

# 리소스 그룹
pcs resource group add WebGroup WebFS VIP WebServer

# 제약 조건
pcs constraint colocation add WebGroup with DRBD-clone INFINITY with-rsc-role=Master
pcs constraint order promote DRBD-clone then start WebGroup

# 상태 확인
pcs status
```

### 페일오버 테스트

```bash
# 현재 상태 확인
pcs status

# 수동 페일오버 테스트
pcs resource move WebGroup node2

# 노드 standby 모드
pcs node standby node1

# standby 해제
pcs node unstandby node1

# 시뮬레이션 (리소스 강제 중지)
pcs resource debug-stop WebServer

# 페일백 (원래 노드로 복귀)
pcs resource clear WebGroup
```

### 모니터링

```bash
# 실시간 상태 모니터링
watch -n 1 'pcs status'

# 클러스터 이벤트 로그
sudo journalctl -u pacemaker -f

# crm_mon (상세 모니터링)
sudo crm_mon -1
sudo crm_mon -Afr  # 전체 정보, 페일카운트, 리소스

# 리소스 히스토리
pcs resource history show WebServer
```

---

## 연습 문제

### 문제 1: 클러스터 설정

pcs를 사용하여 2노드 클러스터를 설정하는 명령 순서를 작성하세요.

### 문제 2: 리소스 그룹

VIP (192.168.1.200), 파일시스템 (/dev/sdb1 → /data), PostgreSQL 서비스를 포함하는 리소스 그룹을 생성하세요.

### 문제 3: DRBD 복제

DRBD 리소스 r0의 현재 상태와 동기화 상태를 확인하는 명령을 작성하세요.

---

## 정답

### 문제 1 정답

```bash
# 1. pcsd 시작 및 인증
sudo systemctl enable pcsd
sudo systemctl start pcsd
sudo passwd hacluster
sudo pcs host auth node1 node2

# 2. 클러스터 생성 및 시작
sudo pcs cluster setup mycluster node1 node2
sudo pcs cluster start --all
sudo pcs cluster enable --all

# 3. 기본 속성 설정
sudo pcs property set stonith-enabled=false  # 테스트용
sudo pcs property set no-quorum-policy=ignore
```

### 문제 2 정답

```bash
# VIP 리소스
sudo pcs resource create VIP ocf:heartbeat:IPaddr2 \
    ip=192.168.1.200 cidr_netmask=24

# 파일시스템 리소스
sudo pcs resource create DataFS ocf:heartbeat:Filesystem \
    device=/dev/sdb1 directory=/data fstype=ext4

# PostgreSQL 리소스
sudo pcs resource create PostgreSQL ocf:heartbeat:pgsql \
    pgctl=/usr/lib/postgresql/14/bin/pg_ctl \
    pgdata=/var/lib/postgresql/14/main \
    op start timeout=60s \
    op stop timeout=60s \
    op monitor interval=10s

# 그룹 생성
sudo pcs resource group add DBGroup DataFS VIP PostgreSQL

# 순서 제약 (명시적)
sudo pcs constraint order DataFS then VIP then PostgreSQL
```

### 문제 3 정답

```bash
# DRBD 상태 확인
sudo drbdadm status r0

# /proc/drbd 확인
cat /proc/drbd

# 연결 상태
sudo drbdadm cstate r0

# 디스크 상태
sudo drbdadm dstate r0

# 역할 확인
sudo drbdadm role r0

# 상세 상태 (모든 리소스)
sudo drbdadm status all
```

---

## 다음 단계

- [트러블슈팅 가이드](./26_Troubleshooting_Guide.md) - 시스템 문제 진단 및 해결

---

## 참고 자료

- [Pacemaker Documentation](https://clusterlabs.org/pacemaker/doc/)
- [DRBD User's Guide](https://linbit.com/drbd-user-guide/)
- [Red Hat HA Cluster](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html/configuring_and_managing_high_availability_clusters/index)
- `man pcs`, `man corosync`, `man drbdadm`
