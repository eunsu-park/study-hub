# 관리형 관계형 데이터베이스 (RDS / Cloud SQL)

**이전**: [로드밸런싱 & CDN](./10_Load_Balancing_CDN.md) | **다음**: [NoSQL 데이터베이스](./12_NoSQL_Databases.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 자체 호스팅 설치 대비 관리형 데이터베이스 서비스(managed database service)의 장점을 설명할 수 있습니다
2. 지원 엔진과 기능 측면에서 AWS RDS와 GCP Cloud SQL을 비교할 수 있습니다
3. 적절한 크기(sizing)로 관리형 관계형 데이터베이스 인스턴스를 프로비저닝할 수 있습니다
4. 자동 백업, 특정 시점 복원(point-in-time recovery), 읽기 복제본(read replica)을 구성할 수 있습니다
5. 프로덕션 워크로드를 위한 Multi-AZ(고가용성) 배포를 구현할 수 있습니다
6. 고성능 대안으로서 AWS Aurora와 GCP AlloyDB/Spanner를 설명할 수 있습니다
7. 암호화, VPC 배치, 접근 제어를 포함한 보안 모범 사례를 적용할 수 있습니다

---

관계형 데이터베이스(relational database)는 대부분의 비즈니스 애플리케이션의 근간이지만, 안정적으로 운영하려면 백업, 패치, 복제, 장애 조치(failover)에 대한 전문 지식이 필요합니다. 관리형 데이터베이스 서비스(managed database service)는 이러한 운영 부담을 클라우드 제공자에게 위임하여, 팀이 인프라 유지보수 대신 스키마 설계와 쿼리 성능에 집중할 수 있게 합니다.

## 1. 관리형 DB 개요

### 1.1 관리형 vs 자체 관리

| 작업 | 자체 관리 (EC2) | 관리형 (RDS/Cloud SQL) |
|------|----------------|----------------------|
| 하드웨어 프로비저닝 | 사용자 | 제공자 |
| OS 패치 | 사용자 | 제공자 |
| DB 설치/설정 | 사용자 | 제공자 |
| 백업 | 사용자 | 자동 |
| 고가용성 | 사용자 | 옵션 제공 |
| 스케일링 | 수동 | 버튼 클릭 |
| 모니터링 | 설정 필요 | 기본 제공 |

### 1.2 서비스 비교

| 항목 | AWS | GCP |
|------|-----|-----|
| 관리형 RDB | RDS | Cloud SQL |
| 고성능 DB | Aurora | Cloud Spanner, AlloyDB |
| 지원 엔진 | MySQL, PostgreSQL, MariaDB, Oracle, SQL Server | MySQL, PostgreSQL, SQL Server |

---

## 2. AWS RDS

### 2.1 RDS 인스턴스 생성

```bash
# DB 서브넷 그룹 생성
aws rds create-db-subnet-group \
    --db-subnet-group-name my-subnet-group \
    --db-subnet-group-description "My DB subnets" \
    --subnet-ids subnet-1 subnet-2

# 파라미터 그룹 생성 (선택)
aws rds create-db-parameter-group \
    --db-parameter-group-name my-params \
    --db-parameter-group-family mysql8.0 \
    --description "Custom parameters"

# RDS 인스턴스 생성
aws rds create-db-instance \
    --db-instance-identifier my-database \
    --db-instance-class db.t3.micro \
    --engine mysql \
    --engine-version 8.0 \
    --master-username admin \
    --master-user-password MyPassword123! \
    --allocated-storage 20 \
    --storage-type gp3 \
    --db-subnet-group-name my-subnet-group \
    --vpc-security-group-ids sg-12345678 \
    --backup-retention-period 7 \
    --multi-az \
    --publicly-accessible false

# 생성 상태 확인
aws rds describe-db-instances \
    --db-instance-identifier my-database \
    --query 'DBInstances[0].DBInstanceStatus'
```

### 2.2 Multi-AZ 배포

```
┌─────────────────────────────────────────────────────────────┐
│  VPC                                                        │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │     AZ-a            │  │     AZ-b            │          │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │          │
│  │  │  Primary DB   │──┼──┼──│  Standby DB   │  │          │
│  │  │  (읽기/쓰기)  │  │  │  │  (동기 복제)  │  │          │
│  │  └───────────────┘  │  │  └───────────────┘  │          │
│  └─────────────────────┘  └─────────────────────┘          │
│              ↑ 자동 장애 조치                               │
└─────────────────────────────────────────────────────────────┘
```

```bash
# 기존 인스턴스를 Multi-AZ로 변경
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --multi-az \
    --apply-immediately
```

### 2.3 읽기 복제본

```bash
# 읽기 복제본 생성 (같은 리전)
aws rds create-db-instance-read-replica \
    --db-instance-identifier my-read-replica \
    --source-db-instance-identifier my-database

# 다른 리전에 읽기 복제본 (크로스 리전)
aws rds create-db-instance-read-replica \
    --db-instance-identifier my-replica-us \
    --source-db-instance-identifier arn:aws:rds:ap-northeast-2:123456789012:db:my-database \
    --region us-east-1

# 복제본 승격 (마스터로 변환)
aws rds promote-read-replica \
    --db-instance-identifier my-read-replica
```

### 2.4 백업 및 복원

```bash
# 수동 스냅샷 생성
aws rds create-db-snapshot \
    --db-instance-identifier my-database \
    --db-snapshot-identifier my-snapshot-2024

# 스냅샷에서 복원
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier my-restored-db \
    --db-snapshot-identifier my-snapshot-2024

# 특정 시점 복원 (Point-in-Time Recovery)
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier my-database \
    --target-db-instance-identifier my-pitr-db \
    --restore-time 2024-01-15T10:00:00Z

# 자동 백업 설정 확인/변경
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --backup-retention-period 14 \
    --preferred-backup-window "03:00-04:00"
```

---

## 3. GCP Cloud SQL

### 3.1 Cloud SQL 인스턴스 생성

```bash
# Cloud SQL API 활성화
gcloud services enable sqladmin.googleapis.com

# MySQL 인스턴스 생성
gcloud sql instances create my-database \
    --database-version=MYSQL_8_0 \
    --tier=db-f1-micro \
    --region=asia-northeast3 \
    --root-password=MyPassword123! \
    --storage-size=10GB \
    --storage-type=SSD \
    --backup-start-time=03:00 \
    --availability-type=REGIONAL

# PostgreSQL 인스턴스 생성
gcloud sql instances create my-postgres \
    --database-version=POSTGRES_15 \
    --tier=db-g1-small \
    --region=asia-northeast3

# 인스턴스 정보 확인
gcloud sql instances describe my-database
```

### 3.2 고가용성 (HA)

```bash
# 고가용성 인스턴스 생성
gcloud sql instances create my-ha-db \
    --database-version=MYSQL_8_0 \
    --tier=db-n1-standard-2 \
    --region=asia-northeast3 \
    --availability-type=REGIONAL \
    --root-password=MyPassword123!

# 기존 인스턴스를 HA로 변경
gcloud sql instances patch my-database \
    --availability-type=REGIONAL
```

### 3.3 읽기 복제본

```bash
# 읽기 복제본 생성
gcloud sql instances create my-read-replica \
    --master-instance-name=my-database \
    --region=asia-northeast3

# 복제본 승격
gcloud sql instances promote-replica my-read-replica

# 복제본 목록 확인
gcloud sql instances list --filter="masterInstanceName:my-database"
```

### 3.4 백업 및 복원

```bash
# 온디맨드 백업 생성
gcloud sql backups create \
    --instance=my-database \
    --description="Manual backup"

# 백업 목록 확인
gcloud sql backups list --instance=my-database

# 백업에서 복원 (새 인스턴스)
gcloud sql instances restore-backup my-restored-db \
    --backup-instance=my-database \
    --backup-id=1234567890

# Point-in-Time Recovery
gcloud sql instances clone my-database my-pitr-db \
    --point-in-time="2024-01-15T10:00:00Z"
```

---

## 4. 연결 설정

### 4.1 AWS RDS 연결

**보안 그룹 설정:**
```bash
# RDS 보안 그룹에 애플리케이션 접근 허용
aws ec2 authorize-security-group-ingress \
    --group-id sg-rds \
    --protocol tcp \
    --port 3306 \
    --source-group sg-app

# 엔드포인트 확인
aws rds describe-db-instances \
    --db-instance-identifier my-database \
    --query 'DBInstances[0].Endpoint'
```

**애플리케이션에서 연결:**
```python
import pymysql

connection = pymysql.connect(
    host='my-database.xxxx.ap-northeast-2.rds.amazonaws.com',
    user='admin',
    password='MyPassword123!',
    database='mydb',
    port=3306
)
```

### 4.2 GCP Cloud SQL 연결

**연결 방법:**

1. **퍼블릭 IP (권장하지 않음)**
```bash
# 퍼블릭 IP 허용
gcloud sql instances patch my-database \
    --authorized-networks=203.0.113.0/24

# 연결
mysql -h <PUBLIC_IP> -u root -p
```

2. **Cloud SQL Proxy (권장)**
```bash
# Cloud SQL Proxy 다운로드
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.linux.amd64
chmod +x cloud-sql-proxy

# Proxy 실행
./cloud-sql-proxy PROJECT_ID:asia-northeast3:my-database

# 다른 터미널에서 연결
mysql -h 127.0.0.1 -u root -p
```

3. **Private IP (VPC 내부)**
```bash
# Private IP 활성화
gcloud sql instances patch my-database \
    --network=projects/PROJECT_ID/global/networks/my-vpc

# VPC 내 인스턴스에서 연결
mysql -h <PRIVATE_IP> -u root -p
```

**Python 연결 (Cloud SQL Connector):**
```python
from google.cloud.sql.connector import Connector
import pymysql

connector = Connector()

def get_conn():
    return connector.connect(
        "project:region:instance",
        "pymysql",
        user="root",
        password="password",
        db="mydb"
    )

connection = get_conn()
```

---

## 5. 성능 최적화

### 5.1 인스턴스 크기 조정

**AWS RDS:**
```bash
# 인스턴스 클래스 변경
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --db-instance-class db.m5.large \
    --apply-immediately

# 스토리지 확장 (축소 불가)
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --allocated-storage 100
```

**GCP Cloud SQL:**
```bash
# 머신 타입 변경
gcloud sql instances patch my-database \
    --tier=db-n1-standard-4

# 스토리지 확장
gcloud sql instances patch my-database \
    --storage-size=100GB
```

### 5.2 파라미터 튜닝

**AWS RDS 파라미터 그룹:**
```bash
# 파라미터 변경
aws rds modify-db-parameter-group \
    --db-parameter-group-name my-params \
    --parameters "ParameterName=max_connections,ParameterValue=500,ApplyMethod=pending-reboot"

aws rds modify-db-parameter-group \
    --db-parameter-group-name my-params \
    --parameters "ParameterName=innodb_buffer_pool_size,ParameterValue={DBInstanceClassMemory*3/4},ApplyMethod=pending-reboot"
```

**GCP Cloud SQL 플래그:**
```bash
# 플래그 설정
gcloud sql instances patch my-database \
    --database-flags=max_connections=500,innodb_buffer_pool_size=1073741824
```

---

## 6. Aurora / AlloyDB / Spanner

### 6.1 AWS Aurora

Aurora는 클라우드 네이티브 관계형 데이터베이스입니다.

**특징:**
- MySQL/PostgreSQL 호환
- 최대 128TB 자동 확장
- 6개 복제본 (3개 AZ)
- 읽기 복제본 최대 15개
- 서버리스 옵션 (Aurora Serverless)

```bash
# Aurora 클러스터 생성
aws rds create-db-cluster \
    --db-cluster-identifier my-aurora \
    --engine aurora-mysql \
    --engine-version 8.0.mysql_aurora.3.04.0 \
    --master-username admin \
    --master-user-password MyPassword123! \
    --db-subnet-group-name my-subnet-group \
    --vpc-security-group-ids sg-12345678

# Aurora 인스턴스 추가
aws rds create-db-instance \
    --db-instance-identifier my-aurora-instance-1 \
    --db-cluster-identifier my-aurora \
    --db-instance-class db.r5.large \
    --engine aurora-mysql
```

### 6.2 GCP Cloud Spanner

Spanner는 글로벌 분산 관계형 데이터베이스입니다.

**특징:**
- 글로벌 트랜잭션
- 무제한 확장
- 99.999% SLA
- PostgreSQL 호환 인터페이스

```bash
# Spanner 인스턴스 생성
gcloud spanner instances create my-spanner \
    --config=regional-asia-northeast3 \
    --nodes=1 \
    --description="My Spanner instance"

# 데이터베이스 생성
gcloud spanner databases create mydb \
    --instance=my-spanner
```

### 6.3 GCP AlloyDB

AlloyDB는 PostgreSQL 호환 고성능 데이터베이스입니다.

```bash
# AlloyDB 클러스터 생성
gcloud alloydb clusters create my-cluster \
    --region=asia-northeast3 \
    --password=MyPassword123!

# 기본 인스턴스 생성
gcloud alloydb instances create primary \
    --cluster=my-cluster \
    --region=asia-northeast3 \
    --instance-type=PRIMARY \
    --cpu-count=2
```

---

## 7. 비용 비교

### 7.1 AWS RDS 비용 (서울)

| 인스턴스 | vCPU | 메모리 | 시간당 비용 |
|----------|------|--------|------------|
| db.t3.micro | 2 | 1 GB | ~$0.02 |
| db.t3.small | 2 | 2 GB | ~$0.04 |
| db.m5.large | 2 | 8 GB | ~$0.18 |
| db.r5.large | 2 | 16 GB | ~$0.26 |

**추가 비용:**
- 스토리지: gp3 $0.114/GB/월
- 백업: 보관량 × $0.095/GB/월
- Multi-AZ: 인스턴스 비용 2배

### 7.2 GCP Cloud SQL 비용 (서울)

| 티어 | vCPU | 메모리 | 시간당 비용 |
|------|------|--------|------------|
| db-f1-micro | 공유 | 0.6 GB | ~$0.01 |
| db-g1-small | 공유 | 1.7 GB | ~$0.03 |
| db-n1-standard-2 | 2 | 7.5 GB | ~$0.13 |
| db-n1-highmem-2 | 2 | 13 GB | ~$0.16 |

**추가 비용:**
- 스토리지: SSD $0.180/GB/월
- 고가용성: 인스턴스 비용 2배
- 백업: $0.08/GB/월

---

## 8. 보안

### 8.1 암호화

**AWS RDS:**
```bash
# 저장 시 암호화 (생성 시)
aws rds create-db-instance \
    --storage-encrypted \
    --kms-key-id arn:aws:kms:...:key/xxx \
    ...

# SSL 강제
aws rds modify-db-parameter-group \
    --db-parameter-group-name my-params \
    --parameters "ParameterName=require_secure_transport,ParameterValue=1"
```

**GCP Cloud SQL:**
```bash
# SSL 인증서 생성
gcloud sql ssl client-certs create my-client \
    --instance=my-database \
    --common-name=my-client

# SSL 필수 설정
gcloud sql instances patch my-database \
    --require-ssl
```

### 8.2 IAM 인증

**AWS RDS IAM 인증:**
```bash
# IAM 인증 활성화
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --enable-iam-database-authentication

# 임시 토큰 생성
aws rds generate-db-auth-token \
    --hostname my-database.xxxx.rds.amazonaws.com \
    --port 3306 \
    --username iam_user
```

**GCP Cloud SQL IAM:**
```bash
# IAM 인증 활성화
gcloud sql instances patch my-database \
    --enable-database-flags \
    --database-flags=cloudsql_iam_authentication=on

# IAM 사용자 추가
gcloud sql users create user@example.com \
    --instance=my-database \
    --type=CLOUD_IAM_USER
```

---

## 9. 다음 단계

- [12_NoSQL_Databases.md](./12_NoSQL_Databases.md) - NoSQL 데이터베이스
- [PostgreSQL/](../PostgreSQL/) - PostgreSQL 상세

---

## 연습 문제

### 연습 문제 1: 관리형 DB vs 자체 호스팅 결정

팀이 EC2에서 PostgreSQL을 실행(자체 관리)할지 AWS RDS for PostgreSQL(관리형)을 사용할지 고민 중입니다. 우선순위는 최소한의 운영 오버헤드, 자동화된 백업, 고가용성입니다.

RDS가 자동으로 처리하지만 자체 관리 EC2에서는 상당한 수동 작업이 필요한 세 가지 구체적인 운영 작업을 나열하고, 자체 관리 EC2가 더 나은 선택이 될 시나리오 하나를 설명하세요.

<details>
<summary>정답 보기</summary>

**RDS가 자동화하는 세 가지 작업**:

1. **자동 백업 및 포인트-인-타임 복구(PITR, Point-In-Time Recovery)** — RDS는 자동으로 일별 스냅샷을 찍고 트랜잭션 로그를 S3에 스트리밍하여, 보관 기간(최대 35일) 내 어느 초에나 PITR이 가능합니다. 자체 관리 EC2에서는 백업 스크립트를 작성·유지하고, 백업 스토리지를 관리하고, 복원 절차를 테스트하고, 장애를 모니터링해야 합니다.

2. **Multi-AZ 장애 조치(failover)** — Multi-AZ 활성화 시 RDS는 자동으로 다른 AZ에 동기식 스탠바이 복제본을 유지합니다. 기본(primary)이 실패하면 AWS가 수동 개입 없이 ~1~2분 내에 스탠바이를 기본으로 승격합니다. EC2에서는 복제를 수동으로 구성(예: repmgr을 사용한 PostgreSQL 스트리밍 복제)하고 장애 조치 스크립트나 Pacemaker/Corosync를 설정해야 합니다.

3. **마이너 버전 패치** — RDS는 유지 보수 윈도우(maintenance window) 동안 마이너 엔진 패치를 자동으로 적용하도록 설정할 수 있습니다. EC2에서는 새 PostgreSQL 릴리스를 모니터링하고, 호환성을 테스트하고, 다운타임을 예약하고, 업데이트를 수동으로 적용해야 합니다.

**자체 관리 EC2가 더 나은 시나리오**:
- **OS 수준 접근**이 필요한 경우(커스텀 커널 파라미터, 휴지 페이지(huge pages), I/O 스케줄러 설정), RDS에서 사용할 수 없는 **PostgreSQL 확장**(특정 contrib 확장 또는 커스텀 컴파일 플러그인)이 필요하거나, RDS 파라미터 그룹의 지원 범위를 넘는 PostgreSQL 버전과 설정에 대한 완전한 제어가 필요할 때입니다.

</details>

### 연습 문제 2: RDS Multi-AZ vs 읽기 복제본(Read Replica)

RDS Multi-AZ와 RDS 읽기 복제본의 차이를 설명하세요. 각 시나리오에서 어떤 기능이 해결책인지 설명하세요:

1. 마케팅 팀의 무거운 보고서 쿼리가 프로덕션 애플리케이션을 느리게 만들고 있습니다.
2. 프로덕션 데이터베이스의 AZ가 하드웨어 장애를 경험하여 데이터베이스에 접근할 수 없습니다.

<details>
<summary>정답 보기</summary>

**Multi-AZ**:
- 목적: **고가용성(high availability)과 자동 장애 조치**
- 스탠바이는 다른 AZ의 동기식 복제본입니다.
- 스탠바이는 읽기 트래픽을 처리하지 않으며, 장애 조치를 위해서만 존재합니다.
- 장애 조치는 자동이며(~1~2분), DNS 엔드포인트가 업데이트됩니다.
- 스탠바이는 애플리케이션에 투명하게 처리됩니다.

**읽기 복제본(Read Replica)**:
- 목적: **읽기 스케일링(read scaling)과 읽기 트래픽 오프로딩**
- 기본에서 하나 이상의 복제본으로 비동기 복제합니다.
- 복제본은 읽기 쿼리를 처리하여 기본의 부하를 줄입니다.
- 복제본은 같은 리전, 다른 리전, 또는 독립 기본으로 승격될 수 있습니다.
- 읽기 복제본으로의 자동 장애 조치는 없습니다(승격은 수동).

**시나리오 답변**:

1. **읽기 복제본** — 보고서 쿼리를 하나 이상의 읽기 복제본으로 보낼 수 있습니다. 이렇게 하면 프로덕션 기본의 CPU/IO 부담을 줄여 애플리케이션의 기본 성능을 회복합니다. 보고 도구는 기본 대신 읽기 복제본 엔드포인트에 연결합니다.

2. **Multi-AZ** — RDS Multi-AZ는 자동으로 AZ 장애를 감지하고 동기식 스탠바이를 기본으로 승격합니다. DNS 엔드포인트는 자동으로 새 기본을 가리키며, 일반적으로 1~2분 내에 완료됩니다. 애플리케이션이 재연결하여 최소 다운타임으로 작업을 재개합니다.

</details>

### 연습 문제 3: 데이터베이스 보안 설정

프로덕션 이커머스 애플리케이션을 위한 RDS PostgreSQL 인스턴스를 프로비저닝하고 있습니다. 적용할 주요 보안 설정을 나열하고 각각이 중요한 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

1. **`--publicly-accessible false`로 프라이빗 서브넷에 인스턴스 배치**
   - RDS 인스턴스에 공개 IP가 없어야 합니다. VPC 내 리소스(애플리케이션 서버)만 연결할 수 있습니다. 직접적인 인터넷 노출을 방지합니다.

2. **애플리케이션 계층에서만 접근을 허용하는 전용 보안 그룹 사용**
   ```bash
   # 애플리케이션 서버 보안 그룹에서만 PostgreSQL 포트 5432 허용
   aws ec2 authorize-security-group-ingress \
       --group-id sg-rds \
       --protocol tcp \
       --port 5432 \
       --source-group sg-app-server
   ```

3. **`--storage-encrypted`로 저장 데이터 암호화 활성화**
   - 저장 데이터(스토리지, 백업, 스냅샷, 로그)가 AES-256으로 암호화됩니다. PCI DSS 및 HIPAA 규정 준수에 필요합니다. 성능 영향은 최소화됩니다.

4. **최소 7일 보관 기간으로 자동 백업 활성화**
   - 포인트-인-타임 복구(PITR)를 활성화합니다. 실수로 인한 데이터 삭제나 손상 복구에 필수적입니다.
   ```bash
   --backup-retention-period 7
   ```

5. **AWS Secrets Manager를 사용하여 마스터 비밀번호 순환(rotation)**
   - 데이터베이스 자격 증명을 애플리케이션 코드에 절대 하드코딩하지 마세요. Secrets Manager에 저장하고 자동 순환을 활성화합니다. IAM 역할을 사용하여 애플리케이션이 시크릿을 검색할 수 있는 접근 권한을 부여합니다.

6. **삭제 보호 활성화**
   - 데이터베이스 인스턴스의 실수로 인한 삭제를 방지합니다:
   ```bash
   aws rds modify-db-instance \
       --db-instance-identifier my-db \
       --deletion-protection \
       --apply-immediately
   ```

</details>

### 연습 문제 4: Aurora vs RDS PostgreSQL

SaaS 회사가 빠르게 성장하고 있으며 읽기 스케일링(현재 읽기 복제본 2개)과 재해 복구를 위한 교차 리전 복제가 필요할 것으로 예상합니다. 현재 RDS PostgreSQL 인스턴스는 `db.r5.2xlarge`입니다.

Aurora PostgreSQL로 마이그레이션해야 하는지 평가하세요. Aurora가 명시된 요건에 제공하는 주요 이점은 무엇입니까?

<details>
<summary>정답 보기</summary>

**요건에 대한 Aurora의 이점**:

1. **읽기 스케일링**: Aurora는 최대 **15개의 저지연 읽기 복제본**을 지원합니다(RDS PostgreSQL의 5개 대비). Aurora 복제본은 동일한 공유 분산 스토리지 볼륨에서 읽으며 복제 지연(replica lag)이 일반적으로 100ms 미만입니다. 복제본 추가 시 추가 스토리지 복사본이 필요 없어 비용이 절감됩니다.

2. **교차 리전 재해 복구**: Aurora Global Database는 보조 리전으로 **1초 미만의 복제 지연**을 가능하게 합니다(RDS 교차 리전 읽기 복제본의 수 분 대비). 보조 리전으로의 장애 조치는 관리되며 1분 미만에 완료됩니다.

3. **스토리지 확장성**: Aurora의 공유 스토리지 레이어는 사용자 개입 없이 10 GB씩 최대 128 TB까지 자동으로 증가합니다. 스토리지 용량을 미리 추정할 필요가 없습니다.

4. **성능**: Aurora는 표준 PostgreSQL 대비 최대 3배의 처리량을 주장합니다. 공유 스토리지 아키텍처는 전통적인 RDS Multi-AZ 설정에 있는 복제 오버헤드를 제거합니다.

**마이그레이션 전 고려사항**:
- Aurora PostgreSQL은 동등한 RDS 인스턴스보다 vCPU-시간당 20~40% 더 비쌉니다.
- 모든 PostgreSQL 확장(extension)이 Aurora에서 사용 가능하지는 않습니다.
- 마이그레이션 시 호환성 검증을 위한 테스트 기간이 필요합니다.

**권장**: 여러 읽기 복제본과 교차 리전 DR이 모두 필요하다면, Aurora PostgreSQL이 더 적합합니다. Aurora Global Database의 운영 단순성과 우수한 복제본 성능은 성장하는 SaaS 워크로드에 있어 더 높은 인스턴스 비용을 정당화합니다.

</details>

### 연습 문제 5: 포인트-인-타임 복구(PITR) 시나리오

RDS MySQL 데이터베이스가 오늘 14:32 UTC에 잘못된 `DELETE` 문으로 인한 심각한 데이터 손상 이벤트가 발생했습니다. DBA가 오류 2분 전인 14:30 UTC의 데이터를 복구해야 합니다.

오늘 14:30 UTC에서 새 인스턴스로 데이터베이스를 복원하는 AWS CLI 명령어를 작성하고, 복원 후 어떤 일이 발생하는지 설명하세요.

<details>
<summary>정답 보기</summary>

```bash
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier my-production-db \
    --target-db-instance-identifier my-production-db-restored \
    --restore-time 2026-02-24T14:30:00Z \
    --db-instance-class db.r5.2xlarge \
    --multi-az \
    --db-subnet-group-name my-subnet-group \
    --vpc-security-group-ids sg-rds
```

**복원 후 발생하는 일**:

1. AWS가 **새** RDS 인스턴스(`my-production-db-restored`)를 생성합니다 — 원본 데이터베이스는 수정하지 않습니다. 원본 인스턴스는 계속 실행됩니다.

2. 복원은 가장 최근의 자동 백업에서 `14:30:00Z`까지의 모든 트랜잭션 로그를 적용하여 해당 정확한 순간의 데이터베이스 상태를 재구성합니다.

3. 새 인스턴스는 새 엔드포인트(DNS 호스트명)를 갖습니다. 새 인스턴스를 가리키도록 애플리케이션 연결 문자열을 업데이트하거나, 새 인스턴스를 사용하여 누락된 데이터만 추출하고 원본 인스턴스에 재생해야 합니다.

4. **일반적인 접근 방식**: 복원된 인스턴스를 사용하여 영향받은 테이블/행을 내보낸 다음, 프로덕션 인스턴스에 가져옵니다. 이렇게 하면 전체 애플리케이션을 새 엔드포인트로 전환하는 것을 피하면서 다운타임을 최소화할 수 있습니다.

5. 복구가 완료되고 검증된 후, 두 데이터베이스 인스턴스 비용을 피하기 위해 복원된 인스턴스를 삭제합니다.

**전제 조건**: `--backup-retention-period`가 최소 1일이어야 하고 자동 백업이 활성화되어 있어야 합니다. PITR은 보관 기간 내 어느 시점에서나 사용 가능합니다.

</details>

---

## 참고 자료

- [AWS RDS Documentation](https://docs.aws.amazon.com/rds/)
- [AWS Aurora Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/)
- [GCP Cloud SQL Documentation](https://cloud.google.com/sql/docs)
- [GCP Cloud Spanner](https://cloud.google.com/spanner/docs)
