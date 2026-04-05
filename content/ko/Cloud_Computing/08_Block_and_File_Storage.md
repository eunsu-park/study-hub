# 블록 및 파일 스토리지 (EBS/EFS vs Persistent Disk/Filestore)

**이전**: [객체 스토리지](./07_Object_Storage.md) | **다음**: [VPC](./09_Virtual_Private_Cloud.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 블록, 파일, 객체 스토리지를 구분하고 각각의 적절한 사용 사례를 설명할 수 있습니다
2. AWS EBS와 GCP Persistent Disk의 볼륨 유형 및 성능 특성을 비교할 수 있습니다
3. IOPS, 처리량, 비용 요구 사항에 따라 적절한 볼륨 유형을 선택할 수 있습니다
4. AWS EFS와 GCP Filestore를 사용하여 공유 파일 스토리지를 구성할 수 있습니다
5. 블록 스토리지 볼륨에 대한 스냅샷 및 백업 전략을 구현할 수 있습니다
6. 인스턴스 연결 스토리지(Instance-attached Storage)와 네트워크 연결 스토리지(Network-attached Storage)의 차이를 설명할 수 있습니다

---

객체 스토리지가 대규모 비정형 데이터를 처리하는 반면, 많은 워크로드는 블록 및 파일 스토리지만이 제공할 수 있는 낮은 지연 시간과 높은 처리량을 필요로 합니다. 데이터베이스는 랜덤 I/O를 위해 블록 볼륨이 필요하고, 공유 애플리케이션 데이터는 종종 POSIX 호환 파일 시스템을 요구합니다. 각 워크로드에 적합한 스토리지 유형을 선택하는 것은 성능과 비용 최적화 모두에 있어 매우 중요합니다.

## 1. 스토리지 유형 비교

### 1.1 블록 vs 파일 vs 객체 스토리지

| 유형 | 특징 | 사용 사례 | AWS | GCP |
|------|------|----------|-----|-----|
| **블록** | 저수준 디스크 접근 | DB, OS 부팅 디스크 | EBS | Persistent Disk |
| **파일** | 공유 파일시스템 | 공유 스토리지, CMS | EFS | Filestore |
| **객체** | HTTP 기반, 무제한 | 백업, 미디어, 로그 | S3 | Cloud Storage |

### 1.2 서비스 매핑

| 기능 | AWS | GCP |
|------|-----|-----|
| 블록 스토리지 | EBS (Elastic Block Store) | Persistent Disk (PD) |
| 공유 파일 스토리지 | EFS (Elastic File System) | Filestore |
| 로컬 SSD | Instance Store | Local SSD |

---

## 2. 블록 스토리지

### 2.1 AWS EBS (Elastic Block Store)

**EBS 볼륨 유형:**

| 유형 | 용도 | IOPS | 처리량 | 비용 |
|------|------|------|--------|------|
| **gp3** | 범용 SSD | 최대 16,000 | 최대 1,000 MB/s | 낮음 |
| **gp2** | 범용 SSD (이전) | 최대 16,000 | 최대 250 MB/s | 중간 |
| **io2** | 프로비저닝 IOPS | 최대 64,000 | 최대 1,000 MB/s | 높음 |
| **st1** | 처리량 최적화 HDD | 최대 500 | 최대 500 MB/s | 낮음 |
| **sc1** | 콜드 HDD | 최대 250 | 최대 250 MB/s | 매우 낮음 |

```bash
# EBS 볼륨 생성
aws ec2 create-volume \
    --availability-zone ap-northeast-2a \
    --size 100 \
    --volume-type gp3 \
    --iops 3000 \
    --throughput 125 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=MyVolume}]'

# EC2에 볼륨 연결
aws ec2 attach-volume \
    --volume-id vol-1234567890abcdef0 \
    --instance-id i-1234567890abcdef0 \
    --device /dev/sdf

# 인스턴스 내에서 마운트
sudo mkfs -t xfs /dev/xvdf
sudo mkdir /data
sudo mount /dev/xvdf /data

# fstab에 추가 (영구 마운트)
echo '/dev/xvdf /data xfs defaults,nofail 0 2' | sudo tee -a /etc/fstab
```

### 2.2 GCP Persistent Disk

**Persistent Disk 유형:**

| 유형 | 용도 | IOPS (읽기) | 처리량 (읽기) | 비용 |
|------|------|------------|--------------|------|
| **pd-standard** | HDD | 최대 7,500 | 최대 180 MB/s | 낮음 |
| **pd-balanced** | SSD (균형) | 최대 80,000 | 최대 1,200 MB/s | 중간 |
| **pd-ssd** | SSD (고성능) | 최대 100,000 | 최대 1,200 MB/s | 높음 |
| **pd-extreme** | 고IOPS SSD | 최대 120,000 | 최대 2,400 MB/s | 매우 높음 |

```bash
# Persistent Disk 생성
gcloud compute disks create my-disk \
    --zone=asia-northeast3-a \
    --size=100GB \
    --type=pd-ssd

# VM에 디스크 연결
gcloud compute instances attach-disk my-instance \
    --disk=my-disk \
    --zone=asia-northeast3-a

# 인스턴스 내에서 마운트
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir /data
sudo mount -o discard,defaults /dev/sdb /data

# fstab에 추가
echo UUID=$(sudo blkid -s UUID -o value /dev/sdb) /data ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
```

---

## 3. 스냅샷

### 3.1 AWS EBS 스냅샷

```bash
# 스냅샷 생성
aws ec2 create-snapshot \
    --volume-id vol-1234567890abcdef0 \
    --description "My snapshot" \
    --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=MySnapshot}]'

# 스냅샷 목록 조회
aws ec2 describe-snapshots \
    --owner-ids self \
    --query 'Snapshots[*].[SnapshotId,VolumeId,StartTime,State]'

# 스냅샷에서 볼륨 복원
aws ec2 create-volume \
    --availability-zone ap-northeast-2a \
    --snapshot-id snap-1234567890abcdef0 \
    --volume-type gp3

# 스냅샷 복사 (다른 리전)
aws ec2 copy-snapshot \
    --source-region ap-northeast-2 \
    --source-snapshot-id snap-1234567890abcdef0 \
    --destination-region us-east-1

# 스냅샷 삭제
aws ec2 delete-snapshot --snapshot-id snap-1234567890abcdef0
```

**자동 스냅샷 (Data Lifecycle Manager):**
```bash
# DLM 정책 생성 (매일 스냅샷, 7일 보관)
aws dlm create-lifecycle-policy \
    --description "Daily snapshots" \
    --state ENABLED \
    --execution-role-arn arn:aws:iam::123456789012:role/AWSDataLifecycleManagerDefaultRole \
    --policy-details '{
        "ResourceTypes": ["VOLUME"],
        "TargetTags": [{"Key": "Backup", "Value": "true"}],
        "Schedules": [{
            "Name": "DailySnapshots",
            "CreateRule": {"Interval": 24, "IntervalUnit": "HOURS", "Times": ["03:00"]},
            "RetainRule": {"Count": 7}
        }]
    }'
```

### 3.2 GCP 스냅샷

```bash
# 스냅샷 생성
gcloud compute snapshots create my-snapshot \
    --source-disk=my-disk \
    --source-disk-zone=asia-northeast3-a

# 스냅샷 목록 조회
gcloud compute snapshots list

# 스냅샷에서 디스크 복원
gcloud compute disks create restored-disk \
    --source-snapshot=my-snapshot \
    --zone=asia-northeast3-a

# 스냅샷 삭제
gcloud compute snapshots delete my-snapshot
```

**스냅샷 스케줄:**
```bash
# 스케줄 정책 생성 (매일, 7일 보관)
gcloud compute resource-policies create snapshot-schedule daily-snapshot \
    --region=asia-northeast3 \
    --max-retention-days=7 \
    --start-time=04:00 \
    --daily-schedule

# 디스크에 스케줄 연결
gcloud compute disks add-resource-policies my-disk \
    --resource-policies=daily-snapshot \
    --zone=asia-northeast3-a
```

---

## 4. 볼륨 확장

### 4.1 AWS EBS 볼륨 확장

```bash
# 1. 볼륨 크기 수정 (온라인 가능)
aws ec2 modify-volume \
    --volume-id vol-1234567890abcdef0 \
    --size 200

# 2. 수정 상태 확인
aws ec2 describe-volumes-modifications \
    --volume-id vol-1234567890abcdef0

# 3. 인스턴스 내에서 파일시스템 확장
# XFS
sudo xfs_growfs -d /data

# ext4
sudo resize2fs /dev/xvdf
```

### 4.2 GCP Persistent Disk 확장

```bash
# 1. 디스크 크기 확장 (온라인 가능)
gcloud compute disks resize my-disk \
    --size=200GB \
    --zone=asia-northeast3-a

# 2. 인스턴스 내에서 파일시스템 확장
# ext4
sudo resize2fs /dev/sdb

# XFS
sudo xfs_growfs /data
```

---

## 5. 파일 스토리지

### 5.1 AWS EFS (Elastic File System)

**특징:**
- NFS v4.1 프로토콜
- 자동 확장/축소
- 멀티 AZ 지원
- 최대 수천 개 EC2 동시 연결

```bash
# 1. EFS 파일 시스템 생성
aws efs create-file-system \
    --performance-mode generalPurpose \
    --throughput-mode bursting \
    --encrypted \
    --tags Key=Name,Value=my-efs

# 2. 마운트 타겟 생성 (각 서브넷)
aws efs create-mount-target \
    --file-system-id fs-12345678 \
    --subnet-id subnet-12345678 \
    --security-groups sg-12345678

# 3. EC2에서 마운트
sudo yum install -y amazon-efs-utils
sudo mkdir /efs
sudo mount -t efs fs-12345678:/ /efs

# 또는 NFS로 마운트
sudo mount -t nfs4 -o nfsvers=4.1 \
    fs-12345678.efs.ap-northeast-2.amazonaws.com:/ /efs

# fstab에 추가
echo 'fs-12345678:/ /efs efs defaults,_netdev 0 0' | sudo tee -a /etc/fstab
```

**EFS 스토리지 클래스:**
| 클래스 | 용도 | 비용 |
|--------|------|------|
| Standard | 자주 액세스 | 높음 |
| Infrequent Access (IA) | 드문 액세스 | 낮음 |
| Archive | 장기 보관 | 매우 낮음 |

```bash
# 수명 주기 정책 설정 (30일 후 IA로 이동)
aws efs put-lifecycle-configuration \
    --file-system-id fs-12345678 \
    --lifecycle-policies '[{"TransitionToIA":"AFTER_30_DAYS"}]'
```

### 5.2 GCP Filestore

**특징:**
- NFS v3 프로토콜
- 사전 프로비저닝된 용량
- 고성능 옵션 제공

**Filestore 티어:**
| 티어 | 용량 | 성능 | 용도 |
|------|------|------|------|
| Basic HDD | 1TB-63.9TB | 100 MB/s | 파일 공유 |
| Basic SSD | 2.5TB-63.9TB | 1,200 MB/s | 고성능 |
| Zonal | 1TB-100TB | 최대 2,560 MB/s | 고성능 워크로드 |
| Enterprise | 1TB-10TB | 최대 1,200 MB/s | 미션 크리티컬 |

```bash
# 1. Filestore 인스턴스 생성
gcloud filestore instances create my-filestore \
    --zone=asia-northeast3-a \
    --tier=BASIC_SSD \
    --file-share=name=vol1,capacity=2.5TB \
    --network=name=default

# 2. Filestore 정보 조회
gcloud filestore instances describe my-filestore \
    --zone=asia-northeast3-a

# 3. VM에서 마운트
sudo apt-get install -y nfs-common
sudo mkdir /filestore
sudo mount 10.0.0.2:/vol1 /filestore

# fstab에 추가
echo '10.0.0.2:/vol1 /filestore nfs defaults,_netdev 0 0' | sudo tee -a /etc/fstab
```

---

## 6. 로컬 SSD

### 6.1 AWS Instance Store

Instance Store는 EC2 인스턴스에 물리적으로 연결된 임시 스토리지입니다.

**특징:**
- 인스턴스 중지/종료 시 데이터 손실
- 매우 높은 IOPS
- 추가 비용 없음 (인스턴스 가격에 포함)

```bash
# Instance Store가 있는 인스턴스 유형 확인
aws ec2 describe-instance-types \
    --filters "Name=instance-storage-supported,Values=true" \
    --query 'InstanceTypes[*].[InstanceType,InstanceStorageInfo.TotalSizeInGB]'

# 예: i3.large, d2.xlarge 등

# 인스턴스 내에서 마운트
sudo mkfs.xfs /dev/nvme1n1
sudo mkdir /local-ssd
sudo mount /dev/nvme1n1 /local-ssd
```

### 6.2 GCP Local SSD

```bash
# Local SSD가 있는 인스턴스 생성
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --machine-type=n2-standard-4 \
    --local-ssd=interface=NVME \
    --local-ssd=interface=NVME

# 인스턴스 내에서 마운트
sudo mkfs.ext4 /dev/nvme0n1
sudo mkdir /local-ssd
sudo mount /dev/nvme0n1 /local-ssd
```

**Local SSD 특징:**
- 375GB 단위로 추가
- 최대 24개 (9TB)
- 인스턴스 중지 시 데이터 손실
- Live Migration 불가 (일부)

---

## 7. 성능 최적화

### 7.1 IOPS vs 처리량

```
IOPS (Input/Output Per Second):
- 초당 읽기/쓰기 작업 수
- 작은 랜덤 I/O에 중요
- 데이터베이스, 트랜잭션 처리

처리량 (Throughput):
- 초당 전송 데이터량 (MB/s)
- 큰 순차 I/O에 중요
- 비디오 스트리밍, 빅데이터
```

### 7.2 최적화 팁

**AWS EBS:**
```bash
# gp3 IOPS/처리량 조정
aws ec2 modify-volume \
    --volume-id vol-xxx \
    --iops 10000 \
    --throughput 500

# EBS 최적화 인스턴스 사용
aws ec2 run-instances \
    --instance-type m5.large \
    --ebs-optimized \
    ...
```

**GCP Persistent Disk:**
```bash
# 더 큰 디스크 = 더 높은 성능
# pd-ssd 100GB: 최대 3,000 IOPS
# pd-ssd 500GB: 최대 15,000 IOPS
# pd-ssd 1TB: 최대 30,000 IOPS

# 성능을 위해 디스크 크기 증가
gcloud compute disks resize my-disk --size=500GB
```

---

## 8. 비용 비교

### 8.1 블록 스토리지 비용 (서울 리전)

| 유형 | AWS EBS | GCP PD |
|------|---------|--------|
| 범용 SSD | $0.114/GB (gp3) | $0.102/GB (pd-balanced) |
| 고성능 SSD | $0.138/GB (io1) | $0.180/GB (pd-ssd) |
| HDD | $0.054/GB (st1) | $0.044/GB (pd-standard) |

### 8.2 파일 스토리지 비용

| 서비스 | 비용 |
|--------|------|
| AWS EFS Standard | ~$0.33/GB/월 |
| AWS EFS IA | ~$0.025/GB/월 |
| GCP Filestore Basic SSD | ~$0.24/GB/월 |
| GCP Filestore Basic HDD | ~$0.12/GB/월 |

---

## 9. 실습: 공유 스토리지 설정

### 9.1 AWS EFS 멀티 인스턴스 마운트

```bash
# 1. 두 개의 서브넷에 마운트 타겟 생성
aws efs create-mount-target --file-system-id fs-xxx --subnet-id subnet-1 --security-groups sg-xxx
aws efs create-mount-target --file-system-id fs-xxx --subnet-id subnet-2 --security-groups sg-xxx

# 2. 보안 그룹에 NFS 규칙 추가
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxx \
    --protocol tcp \
    --port 2049 \
    --source-group sg-instance

# 3. 각 인스턴스에서 마운트
# Instance 1
sudo mkdir /shared && sudo mount -t efs fs-xxx:/ /shared
echo "Hello from Instance 1" | sudo tee /shared/test.txt

# Instance 2
sudo mkdir /shared && sudo mount -t efs fs-xxx:/ /shared
cat /shared/test.txt  # "Hello from Instance 1" 출력
```

### 9.2 GCP Filestore 멀티 인스턴스 마운트

```bash
# 1. 방화벽 규칙 추가
gcloud compute firewall-rules create allow-nfs \
    --allow tcp:2049,tcp:111,udp:2049,udp:111 \
    --source-ranges 10.0.0.0/8

# 2. 각 인스턴스에서 마운트
# Instance 1
sudo mkdir /shared && sudo mount 10.0.0.2:/vol1 /shared
echo "Hello from Instance 1" | sudo tee /shared/test.txt

# Instance 2
sudo mkdir /shared && sudo mount 10.0.0.2:/vol1 /shared
cat /shared/test.txt  # "Hello from Instance 1" 출력
```

---

## 10. 다음 단계

- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC 네트워킹
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - 데이터베이스 스토리지

---

## 연습 문제

### 연습 문제 1: 스토리지 유형 선택

각 워크로드에 가장 적합한 스토리지 유형(블록/EBS, 파일/EFS, 객체/S3)을 선택하고 이유를 설명하세요:

1. EC2에서 실행되는 PostgreSQL 데이터베이스는 낮은 지연 시간의 무작위 읽기/쓰기 접근이 필요합니다.
2. 콘텐츠 관리 시스템(CMS)이 10개의 EC2 웹 서버에 걸쳐 실행되며, 모든 서버가 동일한 업로드된 미디어 파일을 동시에 읽고 써야 합니다.
3. 로그 집계 시스템이 하루에 500 GB의 압축 로그 파일을 아카이브하며, 거의 읽히지 않습니다.
4. 애플리케이션이 EC2 인스턴스가 실행되는 동안에만 필요한 중간 연산 파일을 저장하기 위한 임시 스크래치(scratch) 공간이 필요합니다.

<details>
<summary>정답 보기</summary>

1. **블록 스토리지(EBS)** — 데이터베이스는 무작위 I/O 작업(개별 행 읽기/쓰기)을 위해 블록 수준 접근이 필요합니다. EBS `gp3` 또는 `io2`가 PostgreSQL에 필요한 낮은 지연 시간과 높은 IOPS를 제공합니다. 객체 스토리지는 작업당 높은 지연 시간이, 파일 스토리지는 NFS 오버헤드가 있습니다.

2. **파일 스토리지(EFS)** — 여러 EC2 인스턴스가 동일한 파일 시스템을 동시에 마운트하는 것이 NFS 기반 파일 스토리지의 정의적 사용 사례입니다. EFS는 10개의 웹 서버 모두가 동일한 파일 시스템을 마운트하고 동일한 미디어 파일을 읽고 쓸 수 있게 합니다. 블록 스토리지(EBS)는 읽기-쓰기 모드에서 한 번에 하나의 인스턴스에만 연결할 수 있습니다.

3. **객체 스토리지(S3)** — 로그 아카이브는 한 번 쓰고 거의 읽지 않으며 전체 볼륨이 방대할 수 있습니다. S3의 GB당 과금, 무제한 용량, 라이프사이클 규칙(Glacier로 자동 전환)이 가장 비용 효율적인 선택입니다. 블록 및 파일 스토리지는 콜드 데이터에 대해 GB당 훨씬 더 비쌉니다.

4. **인스턴스 스토어(Instance Store, 로컬 SSD)** — 인스턴스 수명 동안만 필요한 임시 스크래치 공간에는, 호스트에 물리적으로 연결된 인스턴스 스토어(NVMe SSD)가 추가 비용 없이 최고의 IOPS와 처리량을 제공합니다. 인스턴스 중지 시 데이터가 손실되지만, 임시 스크래치 데이터에는 허용됩니다. 임시 데이터를 보관할 EBS 볼륨 비용을 절약할 수 있습니다.

</details>

### 연습 문제 2: EBS 볼륨 유형 선택

고빈도 트레이딩(HFT, High-Frequency Trading) 애플리케이션이 다음 I/O 특성을 가진 데이터베이스를 필요로 합니다:
- 지속적으로 50,000 IOPS 필요
- 500 MB/s 처리량
- 2 TB 용량
- 서브 밀리초(sub-millisecond) 지연 시간이 중요

어떤 EBS 볼륨 유형을 사용해야 합니까? 생성 AWS CLI 명령어를 작성하세요.

<details>
<summary>정답 보기</summary>

**볼륨 유형: `io2`** (프로비저닝된 IOPS SSD)

이유:
- `gp3`는 최대 16,000 IOPS를 지원 — 50,000 IOPS에 불충분합니다.
- `io2`는 볼륨당 최대 64,000 IOPS를 지원합니다(지원 인스턴스에서 io2 Block Express로 최대 256,000).
- `io2`는 HFT 워크로드에 필요한 일관되고 서브 밀리초 지연 시간을 제공합니다.
- `st1`/`sc1`은 HDD 기반으로 지연 시간 요건이 있는 무작위 I/O에 부적합합니다.

```bash
aws ec2 create-volume \
    --availability-zone ap-northeast-2a \
    --size 2000 \
    --volume-type io2 \
    --iops 50000 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=HFT-DB-Volume}]'
```

**참고**: 50,000 IOPS를 달성하려면 EC2 인스턴스도 이를 지원해야 합니다. 스토리지 최적화 인스턴스(i3/i4i) 또는 높은 EBS 대역폭을 지원하는 대형 컴퓨팅 인스턴스(예: `m5.8xlarge`)를 사용하세요.

</details>

### 연습 문제 3: EBS 스냅샷(Snapshot)과 복원

운영(production) EC2 인스턴스에 주요 OS 업데이트 전에 백업이 필요한 EBS 볼륨(`vol-0abc123`)이 있습니다. 다음을 위한 AWS CLI 명령어를 작성하세요:
1. 설명적인 이름으로 볼륨의 스냅샷을 생성합니다.
2. 해당 볼륨의 모든 스냅샷을 나열합니다.
3. 스냅샷에서 AZ `ap-northeast-2b`에 새 EBS 볼륨을 생성합니다(다른 AZ로 복원이 필요한 경우를 위해).

<details>
<summary>정답 보기</summary>

```bash
# 1단계: 설명이 있는 스냅샷 생성
aws ec2 create-snapshot \
    --volume-id vol-0abc123 \
    --description "Pre-OS-update backup - $(date +%Y-%m-%d)" \
    --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=pre-os-update-backup},{Key=Purpose,Value=manual-backup}]'

# 2단계: 이 볼륨의 스냅샷 나열 (본인 소유)
aws ec2 describe-snapshots \
    --owner-ids self \
    --filters "Name=volume-id,Values=vol-0abc123" \
    --query 'Snapshots[*].[SnapshotId,StartTime,State,Description]' \
    --output table

# 3단계: ap-northeast-2b에 스냅샷에서 새 볼륨 생성
# (snap-0xyz456을 2단계의 실제 스냅샷 ID로 교체)
aws ec2 create-volume \
    --snapshot-id snap-0xyz456 \
    --availability-zone ap-northeast-2b \
    --volume-type gp3 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=restored-from-pre-os-update}]'
```

**핵심 인사이트**: 스냅샷은 리전 리소스입니다(AZ 특정이 아님). 동일 리전 내 어느 AZ에도 스냅샷을 복원할 수 있으며, 이를 통해 교차 AZ 재해 복구와 볼륨 마이그레이션이 가능합니다.

</details>

### 연습 문제 4: EFS vs EBS 아키텍처 결정

웹 애플리케이션 팀이 두 가지 스토리지 아키텍처를 논의 중입니다:

**옵션 A**: 각 EC2 웹 서버가 사용자 업로드 파일에 대한 자체 EBS 볼륨을 가집니다.
**옵션 B**: 모든 EC2 웹 서버가 사용자 업로드 파일에 대한 공유 EFS 파일 시스템을 마운트합니다.

다음 측면에서 각 옵션의 의미를 분석하세요: (1) 데이터 일관성(consistency), (2) 스케일링, (3) 비용

<details>
<summary>정답 보기</summary>

| 측면 | 옵션 A: 인스턴스별 EBS | 옵션 B: 공유 EFS |
|------|------------------------|-----------------|
| **데이터 일관성** | 한 서버에 업로드된 파일이 다른 서버에는 보이지 않습니다. 사용자는 어떤 서버가 요청을 처리하느냐에 따라 다른 응답을 받게 됩니다 — 파일 스토리지에서 치명적인 버그입니다. | 모든 서버가 동일한 파일 시스템을 봅니다. 서버 1을 통해 업로드된 파일이 서버 2에서 즉시 읽을 수 있습니다. |
| **스케일링** | 새 EC2 인스턴스 추가 시 빈 볼륨으로 시작됩니다. 인스턴스 간 파일을 수동으로 동기화하거나 CDN/S3 레이어를 추가해야 합니다. | 새 인스턴스가 EFS를 마운트하면 기존 모든 파일에 자동으로 접근 가능합니다. 수평적 스케일링이 원활합니다. |
| **비용** | EBS: ~$0.10/GB/월(gp3). 100 GB × 10대 서버 = 1 TB EBS = **월 $100** | EFS Standard: ~$0.30/GB/월이지만 실제 데이터 100 GB(공유)에 대해서만 = **월 $30**. EFS는 GB당 더 비싸지만 하나의 사본에만 비용이 발생합니다. |

**결론**: 옵션 B(EFS)가 공유 웹 서버 파일 스토리지의 올바른 아키텍처입니다. 옵션 A는 올바르게 작동하기 위해 동기화 메커니즘(예: S3 + 동기화 스크립트)을 구축해야 하므로 비용 이점이 사라집니다.

**모범 사례**: 진정한 대규모 웹 애플리케이션의 경우, 사전 서명된 POST(pre-signed POST)를 사용하여 사용자 업로드를 S3에 직접 저장하고 CloudFront를 통해 제공하세요. 이렇게 하면 EFS가 완전히 필요 없어지고 글로벌 CDN 전달이 가능합니다.

</details>

### 연습 문제 5: 스토리지 비용 비교

다음 스토리지 요건에 대한 월별 비용을 계산하고 요건을 충족하는 가장 저렴한 옵션을 결정하세요:

요건: 월 2회 접근하는 10 TB 데이터(매번 약 100 GB 순차 읽기), 15분 이내 접근 가능해야 함

선택지:
- EBS `st1` (처리량 최적화 HDD): $0.045/GB/월
- S3 Standard-IA: $0.0138/GB/월 + GB당 $0.01 검색 비용
- S3 Glacier Flexible Retrieval: $0.005/GB/월 + GB당 $0.01 검색 비용(긴급(expedited) 티어: ~15분)

<details>
<summary>정답 보기</summary>

**옵션 1: EBS st1**
- 스토리지: 10,000 GB × $0.045 = **월 $450**
- 검색: $0 (연결된 스토리지, 접근당 수수료 없음)
- 합계: **월 $450**

**옵션 2: S3 Standard-IA**
- 스토리지: 10,000 GB × $0.0138 = **월 $138**
- 검색: 2회 × 100 GB × $0.01/GB = **월 $2**
- 합계: **월 $140**

**옵션 3: S3 Glacier Flexible Retrieval (긴급 검색)**
- 스토리지: 10,000 GB × $0.005 = **월 $50**
- 검색: 긴급 검색(1~5분, 15분 SLA 충족)은 ~$0.03/GB
  - 2회 × 100 GB × $0.03 = **월 $6**
- 긴급 요청: 요청당 $0.01 (무시할 만한 수준)
- 합계: **~월 $56**

**최우선 선택: S3 Glacier Flexible Retrieval** ~월 $56 — EBS보다 87% 저렴하고 긴급 검색 티어를 사용하면 15분 가용성 요건을 충족합니다.

**중요 주의사항**: Glacier 긴급 검색은 최선 노력(best-effort) 방식으로 피크 수요 중에는 사용 불가할 수 있습니다. 보장된 15분 가용성이 필요하다면 밀리초 접근을 제공하는 S3 Glacier Instant Retrieval($0.004/GB/월 + $0.03/GB 검색)을 고려하세요.

</details>

---

## 참고 자료

- [AWS EBS Documentation](https://docs.aws.amazon.com/ebs/)
- [AWS EFS Documentation](https://docs.aws.amazon.com/efs/)
- [GCP Persistent Disk](https://cloud.google.com/compute/docs/disks)
- [GCP Filestore](https://cloud.google.com/filestore/docs)
