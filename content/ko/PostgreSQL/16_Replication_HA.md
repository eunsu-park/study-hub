# 16. 복제와 고가용성 (Replication & High Availability)

**이전**: [쿼리 최적화](./15_Query_Optimization.md) | **다음**: [윈도우 함수](./17_Window_Functions.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 물리적 복제(스트리밍 복제)와 논리적 복제(Logical Replication)를 비교하고 각각의 적절한 사용 사례를 식별할 수 있다
2. WAL 설정과 복제 슬롯(Replication Slot)을 포함한 Primary-Standby 스트리밍 복제 환경을 구성할 수 있다
3. 선택적 테이블 복제를 위해 Publication과 Subscription을 사용한 논리적 복제를 설정할 수 있다
4. 시스템 뷰를 사용하여 복제 지연(Replication Lag), 슬롯 상태, WAL 누적을 모니터링할 수 있다
5. 수동 페일오버(Failover)와 스위치오버(Switchover) 작업을 수행하고, pg_rewind로 이전 Primary를 복구할 수 있다
6. Patroni, etcd, HAProxy를 활용한 고가용성(High Availability) 아키텍처를 설계할 수 있다
7. 복제 클러스터와 커넥션 풀링(PgBouncer)을 통합할 수 있다

---

다운타임은 비용을 초래하고 사용자의 신뢰를 떨어뜨립니다. 가용성이 중요한 모든 애플리케이션 -- 사실상 거의 모든 프로덕션 시스템 -- 에서 단일 PostgreSQL 서버는 단일 장애 지점(Single Point of Failure)이 됩니다. 복제(Replication)는 스탠바이 서버에 데이터 복사본을 생성하여 읽기 트래픽을 처리하고, 재해 복구(Disaster Recovery)를 제공하며, Primary 장애 발생 시 수 초 내에 역할을 전환할 수 있게 합니다. 이 레슨에서는 기본적인 스트리밍 복제부터 자동 페일오버를 갖춘 프로덕션 수준의 고가용성 구성까지 전체 스펙트럼을 다룹니다.

## 목차
1. [복제 개요](#1-복제-개요)
2. [물리적 복제 (스트리밍 복제)](#2-물리적-복제-스트리밍-복제)
3. [논리적 복제](#3-논리적-복제)
4. [복제 모니터링](#4-복제-모니터링)
5. [페일오버와 스위치오버](#5-페일오버와-스위치오버)
6. [고가용성 솔루션](#6-고가용성-솔루션)
7. [연습 문제](#7-연습-문제)

---

## 1. 복제 개요

### 1.1 복제의 목적

```
┌─────────────────────────────────────────────────────────────────┐
│                    복제 목적                                      │
├─────────────────┬───────────────────────────────────────────────┤
│ 고가용성 (HA)   │ 장애 시 자동/수동 페일오버로 다운타임 최소화      │
│ 읽기 확장       │ 읽기 쿼리를 스탠바이로 분산                      │
│ 재해 복구 (DR)  │ 지리적으로 분산된 복제본으로 재해 대비           │
│ 백업            │ 스탠바이에서 백업 수행, 운영 부하 감소           │
│ 데이터 분석     │ 복제본에서 무거운 분석 쿼리 실행                 │
└─────────────────┴───────────────────────────────────────────────┘
```

### 1.2 복제 종류 비교

```
┌────────────────┬─────────────────────┬─────────────────────┐
│                │   물리적 복제        │   논리적 복제        │
├────────────────┼─────────────────────┼─────────────────────┤
│ 복제 단위      │ 바이트 레벨 (WAL)   │ 행 단위 변경사항     │
│ 복제 대상      │ 전체 클러스터       │ 선택적 (테이블)      │
│ 버전 호환      │ 동일 메이저 버전    │ 다른 버전 가능       │
│ 스탠바이 쿼리  │ 읽기 전용           │ 읽기/쓰기 가능       │
│ 설정 복잡도   │ 간단                │ 중간                 │
│ 용도          │ HA, 읽기 확장       │ 마이그레이션, 통합   │
└────────────────┴─────────────────────┴─────────────────────┘
```

### 1.3 WAL (Write-Ahead Logging) 기초

```sql
-- WAL(Write-Ahead Log)은 이미 장애 복구를 위해 기록됨 — 복제는 동일한 WAL 레코드를
-- 대기 서버로 전송할 뿐이므로 Primary에 최소한의 오버헤드만 추가.
-- 이 이중 목적 설계가 PostgreSQL 복제가 기본적으로 효율적인 이유
SHOW wal_level;           -- replica 또는 logical
SHOW max_wal_senders;     -- WAL 송신 프로세스 수
SHOW max_replication_slots;
SHOW wal_keep_size;       -- WAL 보관 크기

-- WAL 위치 확인
SELECT pg_current_wal_lsn();           -- 현재 WAL 위치
SELECT pg_walfile_name(pg_current_wal_lsn());  -- WAL 파일명
```

---

## 2. 물리적 복제 (스트리밍 복제)

### 2.1 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                   스트리밍 복제 아키텍처                          │
│                                                                 │
│   Primary                           Standby                    │
│   ┌─────────────┐                   ┌─────────────┐           │
│   │             │    WAL Stream     │             │           │
│   │  PostgreSQL │ ────────────────► │  PostgreSQL │           │
│   │   (R/W)     │                   │   (R/O)     │           │
│   │             │                   │             │           │
│   │ ┌─────────┐ │                   │ ┌─────────┐ │           │
│   │ │wal_sender│─┼───────────────────┼─│wal_recv │ │           │
│   │ └─────────┘ │                   │ └─────────┘ │           │
│   └─────────────┘                   └─────────────┘           │
│                                                                 │
│   [동기/비동기 선택 가능]                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Primary 서버 설정

```bash
# postgresql.conf (Primary)
listen_addresses = '*'
wal_level = replica          # 스트리밍 복제에 필요한 최소 레벨
max_wal_senders = 5          # Standby당 1개 + pg_basebackup 여유분
wal_keep_size = 1GB          # 느린 Standby가 슬롯 없이도 따라잡을 수 있도록 WAL 보관
max_replication_slots = 5    # 슬롯은 WAL 재활용 방지 — Standby마다 하나씩 생성 권장

# 동기 복제 설정 (선택적)
synchronous_commit = on
synchronous_standby_names = 'standby1'

# pg_hba.conf (복제 접속 허용)
# TYPE  DATABASE        USER            ADDRESS                 METHOD
host    replication     replicator      192.168.1.0/24          scram-sha-256
```

```sql
-- 복제 전용 사용자 생성
CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'secure_password';

-- 복제 슬롯 생성 (권장)
SELECT pg_create_physical_replication_slot('standby1_slot');

-- 복제 슬롯 확인
SELECT * FROM pg_replication_slots;
```

### 2.3 Standby 서버 설정

```bash
# 1. Primary에서 기본 백업 생성
pg_basebackup -h primary_host -U replicator -D /var/lib/postgresql/data \
    -Fp -Xs -P -R

# -R 옵션: standby.signal 파일과 primary_conninfo 자동 생성
```

```bash
# postgresql.conf (Standby)
hot_standby = on                  # WAL 재생 중에도 읽기 쿼리 허용
hot_standby_feedback = on         # Primary에 Standby 쿼리 상태 전달 — Primary가
                                  # Standby가 아직 필요한 행을 vacuum하는 것을 방지
max_standby_streaming_delay = 30s # Standby 쿼리가 WAL 재생을 차단할 수 있는 시간
                                  # — 데이터 최신성과 쿼리 안정성 사이의 균형
```

```bash
# postgresql.auto.conf (pg_basebackup -R로 자동 생성)
primary_conninfo = 'host=primary_host port=5432 user=replicator password=secure_password'
primary_slot_name = 'standby1_slot'
```

### 2.4 동기 vs 비동기 복제

```sql
-- 비동기 복제 (기본값)
-- Primary 커밋 후 즉시 반환, Standby 지연 가능
synchronous_commit = on  -- local만 보장

-- 동기 복제
synchronous_commit = on
synchronous_standby_names = 'FIRST 1 (standby1, standby2)'

-- 동기 복제 옵션
-- remote_write: 원격 OS 버퍼까지
-- remote_apply: 원격 적용까지 (가장 안전, 가장 느림)
synchronous_commit = remote_apply
```

```
동기 복제 구성 예시:
┌─────────────────────────────────────────────────────────────────┐
│ synchronous_standby_names = 'FIRST 2 (s1, s2, s3)'             │
│                                                                 │
│   - FIRST 2: 첫 2개 스탠바이의 확인 필요                        │
│   - ANY 2: 아무 2개 스탠바이의 확인 필요                        │
│   - s1, s2, s3: application_name 기반 우선순위                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5 캐스케이딩 복제

```
┌──────────────────────────────────────────────────────────────┐
│              캐스케이딩 복제 토폴로지                          │
│                                                              │
│   Primary ──► Standby1 ──► Standby2 ──► Standby3           │
│              (중계)       (중계)       (최종)               │
│                                                              │
│   장점:                                                      │
│   - Primary 부하 감소                                        │
│   - 네트워크 대역폭 효율화                                    │
│   - 지리적 분산에 유리                                        │
└──────────────────────────────────────────────────────────────┘
```

```bash
# Standby1 (중계 서버)
# postgresql.conf
hot_standby = on

# Standby2 (Standby1에서 복제 받음)
# primary_conninfo에 Standby1 주소 설정
primary_conninfo = 'host=standby1_host ...'
```

---

## 3. 논리적 복제

### 3.1 논리적 복제 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                   논리적 복제 아키텍처                           │
│                                                                 │
│   Publisher                         Subscriber                 │
│   ┌─────────────┐                   ┌─────────────┐           │
│   │ PostgreSQL  │   Publication     │ PostgreSQL  │           │
│   │             │ ────────────────► │             │           │
│   │  테이블 A   │   Subscription    │  테이블 A   │           │
│   │  테이블 B   │                   │  테이블 B   │           │
│   └─────────────┘                   └─────────────┘           │
│                                                                 │
│   특징:                                                         │
│   - 테이블 단위 선택적 복제                                      │
│   - 다른 PostgreSQL 버전 간 복제 가능                           │
│   - Subscriber도 쓰기 가능                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Publisher 설정

```sql
-- postgresql.conf
-- wal_level = logical  (필수)

-- Publication 생성
CREATE PUBLICATION my_pub FOR TABLE users, orders;

-- 모든 테이블 발행
CREATE PUBLICATION all_tables_pub FOR ALL TABLES;

-- 특정 작업만 발행
CREATE PUBLICATION insert_only_pub
FOR TABLE products
WITH (publish = 'insert');

-- 행 필터 (PostgreSQL 15+)
CREATE PUBLICATION active_users_pub
FOR TABLE users WHERE (status = 'active');

-- 열 필터 (PostgreSQL 15+)
CREATE PUBLICATION partial_pub
FOR TABLE users (id, name, email);

-- Publication 확인
SELECT * FROM pg_publication;
SELECT * FROM pg_publication_tables;
```

### 3.3 Subscriber 설정

```sql
-- 대상 테이블 생성 (동일한 스키마 필요)
CREATE TABLE users (LIKE source_db.users INCLUDING ALL);
CREATE TABLE orders (LIKE source_db.orders INCLUDING ALL);

-- Subscription 생성
CREATE SUBSCRIPTION my_sub
CONNECTION 'host=publisher_host dbname=source_db user=replicator password=xxx'
PUBLICATION my_pub;

-- 초기 데이터 복사 없이 (이미 동기화된 경우)
CREATE SUBSCRIPTION my_sub
CONNECTION '...'
PUBLICATION my_pub
WITH (copy_data = false);

-- Subscription 관리
ALTER SUBSCRIPTION my_sub DISABLE;
ALTER SUBSCRIPTION my_sub ENABLE;
ALTER SUBSCRIPTION my_sub REFRESH PUBLICATION;

-- Subscription 상태 확인
SELECT * FROM pg_subscription;
SELECT * FROM pg_stat_subscription;
```

### 3.4 논리적 복제 사용 사례

```sql
-- 1. 버전 업그레이드 (최소 다운타임)
-- 구버전 → 신버전으로 논리 복제 설정 후 스위치오버

-- 2. 선택적 데이터 복제 (데이터 웨어하우스)
CREATE PUBLICATION analytics_pub
FOR TABLE sales, customers, products
WHERE (region = 'APAC');

-- 3. 데이터 통합 (여러 소스 → 하나의 타겟)
-- Source DB 1
CREATE PUBLICATION region1_pub FOR TABLE orders;

-- Source DB 2
CREATE PUBLICATION region2_pub FOR TABLE orders;

-- Target DB
CREATE SUBSCRIPTION sub1 ... PUBLICATION region1_pub;
CREATE SUBSCRIPTION sub2 ... PUBLICATION region2_pub;

-- 4. 실시간 리포팅 데이터베이스
CREATE PUBLICATION reporting_pub
FOR TABLE transactions, accounts, audit_logs;
```

### 3.5 충돌 처리

```sql
-- 논리 복제 시 충돌 발생 가능
-- (Subscriber에서도 쓰기 허용되므로)

-- 충돌 확인
SELECT * FROM pg_stat_subscription;
-- srsubstate: 'e' = error

-- 충돌 시 옵션:
-- 1. 충돌 행 수동 해결
-- 2. 해당 트랜잭션 건너뛰기
SELECT pg_replication_origin_advance(
    'pg_' || subid::text,  -- origin name
    '0/XXXXXXX'::pg_lsn    -- 건너뛸 LSN
);

-- 3. 복제 재시작
ALTER SUBSCRIPTION my_sub DISABLE;
-- 문제 해결 후
ALTER SUBSCRIPTION my_sub ENABLE;
```

---

## 4. 복제 모니터링

### 4.1 복제 상태 확인

```sql
-- Primary: WAL 송신 상태
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    sync_state,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) AS replay_lag_bytes
FROM pg_stat_replication;

-- 복제 지연 시간 (Primary)
SELECT
    client_addr,
    state,
    write_lag,
    flush_lag,
    replay_lag
FROM pg_stat_replication;

-- Standby: 현재 복제 상태
SELECT
    pg_is_in_recovery() AS is_standby,
    pg_last_wal_receive_lsn() AS received_lsn,
    pg_last_wal_replay_lsn() AS replayed_lsn,
    pg_last_xact_replay_timestamp() AS last_replay_time,
    EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag_seconds;
```

### 4.2 복제 슬롯 모니터링

```sql
-- 복제 슬롯 상태
SELECT
    slot_name,
    slot_type,
    active,
    restart_lsn,
    pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) AS retained_bytes
FROM pg_replication_slots;

-- 비활성 슬롯으로 인한 WAL 누적 확인
SELECT
    slot_name,
    pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) AS retained
FROM pg_replication_slots
WHERE NOT active;

-- 비활성 슬롯 정리 (주의!)
SELECT pg_drop_replication_slot('unused_slot');
```

### 4.3 모니터링 뷰 생성

```sql
-- 종합 복제 모니터링 뷰
CREATE VIEW v_replication_status AS
SELECT
    'physical' AS repl_type,
    client_addr::text,
    application_name,
    state,
    sync_state,
    pg_size_pretty(pg_wal_lsn_diff(sent_lsn, replay_lsn)) AS lag_size,
    COALESCE(replay_lag::text, 'N/A') AS lag_time
FROM pg_stat_replication

UNION ALL

SELECT
    'logical' AS repl_type,
    subconninfo,
    subname,
    CASE WHEN subenabled THEN 'active' ELSE 'disabled' END,
    'async',
    'N/A',
    'N/A'
FROM pg_subscription;
```

---

## 5. 페일오버와 스위치오버

### 5.1 개념 정리

```
┌─────────────────────────────────────────────────────────────────┐
│ 스위치오버 (Switchover)                                         │
│ - 계획된 역할 전환                                              │
│ - 유지보수, 업그레이드 시 사용                                   │
│ - 데이터 손실 없음                                              │
│                                                                 │
│ 페일오버 (Failover)                                             │
│ - 장애 시 비계획적 역할 전환                                     │
│ - Primary 장애 시 Standby가 승격                                │
│ - 비동기 복제 시 데이터 손실 가능                                │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 수동 페일오버

```bash
# Standby에서 승격 (pg_ctl 사용)
pg_ctl promote -D /var/lib/postgresql/data

# 또는 SQL 사용
SELECT pg_promote();

# 또는 트리거 파일 사용 (레거시)
touch /var/lib/postgresql/data/promote
```

```sql
-- 승격 후 확인
SELECT pg_is_in_recovery();  -- false면 Primary
```

### 5.3 pg_rewind를 사용한 이전 Primary 복구

```bash
# 장애 후 이전 Primary를 새 Standby로 변환
# (타임라인 분기 해결)

# 1. 이전 Primary 정지
pg_ctl stop -D /var/lib/postgresql/data

# 2. pg_rewind 실행
pg_rewind --target-pgdata=/var/lib/postgresql/data \
          --source-server="host=new_primary port=5432 user=replicator"

# 3. standby.signal 생성 및 설정
touch /var/lib/postgresql/data/standby.signal

# 4. 시작
pg_ctl start -D /var/lib/postgresql/data
```

### 5.4 자동 페일오버 스크립트 예시

```bash
#!/bin/bash
# simple_failover.sh

PRIMARY_HOST="primary"
STANDBY_HOST="standby"
VIP="192.168.1.100"

check_primary() {
    pg_isready -h $PRIMARY_HOST -p 5432 -q
    return $?
}

promote_standby() {
    ssh $STANDBY_HOST "pg_ctl promote -D /var/lib/postgresql/data"
}

move_vip() {
    # 기존 Primary에서 VIP 제거
    ssh $PRIMARY_HOST "ip addr del $VIP/24 dev eth0" 2>/dev/null
    # 새 Primary에 VIP 할당
    ssh $STANDBY_HOST "ip addr add $VIP/24 dev eth0"
}

# 메인 로직
if ! check_primary; then
    echo "Primary 장애 감지, 페일오버 시작..."
    promote_standby
    sleep 5
    move_vip
    echo "페일오버 완료"
fi
```

---

## 6. 고가용성 솔루션

### 6.1 Patroni

```yaml
# patroni.yml
scope: postgres-cluster
name: node1

restapi:
  listen: 0.0.0.0:8008
  connect_address: node1:8008

etcd:
  hosts: etcd1:2379,etcd2:2379,etcd3:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 10
    maximum_lag_on_failover: 1048576
    postgresql:
      use_pg_rewind: true
      parameters:
        wal_level: replica
        hot_standby: on
        max_wal_senders: 5
        max_replication_slots: 5
        wal_keep_size: 1GB

  initdb:
    - encoding: UTF8
    - data-checksums

postgresql:
  listen: 0.0.0.0:5432
  connect_address: node1:5432
  data_dir: /var/lib/postgresql/data
  authentication:
    replication:
      username: replicator
      password: rep_password
    superuser:
      username: postgres
      password: postgres_password
```

```bash
# Patroni 클러스터 상태 확인
patronictl -c /etc/patroni/patroni.yml list

# 수동 스위치오버
patronictl -c /etc/patroni/patroni.yml switchover

# 수동 페일오버 (Primary 강제 제거)
patronictl -c /etc/patroni/patroni.yml failover
```

### 6.2 고가용성 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                 Patroni + HAProxy 아키텍처                      │
│                                                                 │
│   ┌───────────────┐                                            │
│   │   HAProxy     │ ◄── VIP                                    │
│   │  (Load Bal)   │                                            │
│   └───────┬───────┘                                            │
│           │                                                     │
│     ┌─────┴─────┐                                              │
│     │           │                                              │
│   ┌─┴─┐       ┌─┴─┐       ┌───┐                               │
│   │N1 │       │N2 │       │N3 │    PostgreSQL + Patroni       │
│   └─┬─┘       └─┬─┘       └─┬─┘                               │
│     │           │           │                                   │
│   ┌─┴───────────┴───────────┴─┐                               │
│   │      etcd Cluster          │   분산 합의 저장소            │
│   └───────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 HAProxy 설정

```
# haproxy.cfg
global
    maxconn 1000

defaults
    mode tcp
    timeout connect 10s
    timeout client 30s
    timeout server 30s

listen postgres_write
    bind *:5432
    option httpchk GET /master
    http-check expect status 200
    default-server inter 3s fall 3 rise 2 on-marked-down shutdown-sessions
    server node1 node1:5432 check port 8008
    server node2 node2:5432 check port 8008
    server node3 node3:5432 check port 8008

listen postgres_read
    bind *:5433
    balance roundrobin
    option httpchk GET /replica
    http-check expect status 200
    default-server inter 3s fall 3 rise 2
    server node1 node1:5432 check port 8008
    server node2 node2:5432 check port 8008
    server node3 node3:5432 check port 8008
```

### 6.4 PgBouncer와 연동

```ini
# pgbouncer.ini
[databases]
mydb = host=haproxy_vip port=5432 dbname=mydb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
```

### 6.5 클라우드 환경 고가용성

```sql
-- AWS RDS: Multi-AZ 자동 페일오버
-- 설정 시 자동 구성됨

-- Azure Database for PostgreSQL: HA 옵션
-- Zone-redundant HA 선택

-- GCP Cloud SQL: Regional HA
-- failover replica 자동 구성

-- 애플리케이션 연결 문자열
-- 읽기/쓰기 분리 예시
-- Primary: postgresql://primary.example.com:5432/mydb
-- Read: postgresql://read.example.com:5432/mydb
```

---

## 7. 연습 문제

### 연습 1: 스트리밍 복제 구성
Docker를 사용하여 Primary-Standby 구성을 설정하세요.

```bash
# docker-compose.yml
version: '3.8'
services:
  primary:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_INITDB_ARGS: "--data-checksums"
    command: |
      postgres
      -c wal_level=replica
      -c max_wal_senders=3
      -c max_replication_slots=3
      -c hot_standby=on
    ports:
      - "5432:5432"
    volumes:
      - primary_data:/var/lib/postgresql/data

  standby:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
      PGDATA: /var/lib/postgresql/data
    depends_on:
      - primary
    # standby 초기화 스크립트 필요
    ports:
      - "5433:5432"
    volumes:
      - standby_data:/var/lib/postgresql/data

volumes:
  primary_data:
  standby_data:
```

### 연습 2: 논리 복제 설정
특정 테이블만 복제하는 논리 복제를 구성하세요.

```sql
-- Publisher (source_db)
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price NUMERIC(10,2),
    category VARCHAR(50)
);

INSERT INTO products (name, price, category) VALUES
    ('Laptop', 999.99, 'Electronics'),
    ('Book', 29.99, 'Books');

CREATE PUBLICATION products_pub FOR TABLE products;

-- Subscriber (target_db)
CREATE TABLE products (LIKE source_db.products);
CREATE SUBSCRIPTION products_sub
CONNECTION 'host=source_host dbname=source_db user=replicator'
PUBLICATION products_pub;
```

### 연습 3: 복제 모니터링 대시보드
복제 상태를 종합적으로 보여주는 쿼리를 작성하세요.

```sql
-- 예시 답안
SELECT
    'Replication Lag' AS metric,
    COALESCE(
        (SELECT pg_size_pretty(pg_wal_lsn_diff(sent_lsn, replay_lsn))
         FROM pg_stat_replication
         LIMIT 1),
        'No standby'
    ) AS value
UNION ALL
SELECT
    'Standby Count',
    (SELECT COUNT(*)::text FROM pg_stat_replication)
UNION ALL
SELECT
    'Replication Slots',
    (SELECT COUNT(*)::text FROM pg_replication_slots);
```

---

## 참고 자료
- [PostgreSQL Replication](https://www.postgresql.org/docs/current/high-availability.html)
- [Logical Replication](https://www.postgresql.org/docs/current/logical-replication.html)
- [Patroni Documentation](https://patroni.readthedocs.io/)
- [pg_basebackup](https://www.postgresql.org/docs/current/app-pgbasebackup.html)

---

**이전**: [쿼리 최적화](./15_Query_Optimization.md) | **다음**: [윈도우 함수](./17_Window_Functions.md)
