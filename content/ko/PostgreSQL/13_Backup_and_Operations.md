# 백업과 운영

**이전**: [트리거](./12_Triggers.md) | **다음**: [JSON과 JSONB](./14_JSON_JSONB.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. 백업 전략(backup strategy)의 중요성을 설명하고, 논리적 백업(logical backup)과 물리적 백업(physical backup)을 구별한다
2. pg_dump와 pg_dumpall을 사용하여 선택적 백업 및 전체 클러스터 백업을 수행한다
3. psql과 pg_restore를 다양한 포맷 옵션으로 사용하여 데이터베이스를 복원한다
4. WAL 아카이빙(WAL archiving)을 설정하고 pg_basebackup으로 물리적 백업을 수행한다
5. 보존 정책(retention policy)과 cron 스케줄링을 적용한 자동 백업 스크립트를 작성한다
6. pg_stat_activity, 잠금 쿼리(lock query), 캐시 히트율(cache hit ratio)을 활용하여 데이터베이스 상태를 모니터링한다
7. VACUUM, ANALYZE, REINDEX를 포함한 정기 유지보수 작업을 수행한다
8. 느린 쿼리 감지 및 연결 감사(connection auditing)를 위한 PostgreSQL 로깅을 설정한다

---

테스트되지 않은 백업 전략을 가진 데이터베이스는 재앙을 기다리는 것과 다름없습니다. 하드웨어 장애, 인적 오류, 소프트웨어 버그는 언제든지 발생할 수 있으며, 그럴 때 백업은 사소한 불편과 치명적인 데이터 손실을 가르는 차이가 됩니다. 백업 외에도 성능 모니터링, 연결 관리, 유지보수 작업 등의 일상적인 운영(day-to-day operations)은 PostgreSQL 설치 환경을 건강하고 응답성 있게 유지합니다. 이 레슨에서는 모든 PostgreSQL 실무자에게 필요한 핵심 DBA 툴킷을 다룹니다.

---

## 1. 백업의 중요성

데이터베이스 백업은 데이터 손실을 방지하는 가장 중요한 작업입니다.

```
┌──────────────────────────────────────────────────────────┐
│                    백업 전략                              │
├──────────────────────────────────────────────────────────┤
│  • 정기 백업: 매일/매주 전체 백업                         │
│  • 증분 백업: 변경분만 백업 (WAL 아카이빙)               │
│  • 복제: 실시간 복제 서버 구성                           │
└──────────────────────────────────────────────────────────┘
```

---

## 2. pg_dump - 논리적 백업

### 이론: Logical 백업 — `pg_dump`

`pg_dump`는 일반 client처럼 서버에 연결해서, 빈 데이터베이스에 대해 실행하면 스키마와 데이터를 재생성하는 SQL statement 시퀀스를 생성.

#### B.1 무엇을 캡처

- 모든 `CREATE TABLE`, `CREATE INDEX`, `CREATE FUNCTION` 등 — 전체 스키마.
- 모든 행 데이터, `INSERT` statement로(또는 기본 `--format=plain`의 경우 `COPY` block).
- 시퀀스와 현재 값.
- 권한(GRANT statement).

출력 포맷:

| 포맷 | 플래그 | 복원 도구 | 병렬화 |
|------|--------|-----------|--------|
| Plain SQL | `--format=plain`(기본) | `psql` | 없음 |
| Custom(binary) | `-Fc` | `pg_restore` | 가능(`-j`) |
| Directory | `-Fd` | `pg_restore` | 가능 |
| Tar | `-Ft` | `pg_restore` | 없음 |

#### B.2 무엇을 캡처하지 *않는가*

- **role, tablespace, 기타 클러스터 전역 객체**. `pg_dump`는 데이터베이스 단위. 글로벌 카탈로그를 받으려면 `pg_dumpall` 사용.
- **설정 파일**(`postgresql.conf`, `pg_hba.conf`).
- **WAL**과 특정 시점으로 roll forward할 수 있는 능력.

#### B.3 일관성

`pg_dump`는 단일 REPEATABLE READ 트랜잭션(또는 `--serializable-deferrable`로 SERIALIZABLE)에서 실행되므로, dump는 단일 snapshot을 표현 — dump 동안 데이터베이스가 수정되어도 내부적으로 일관됨. dump는 "dump 시작 시점"입니다. dump 중 commit된 것은 출력에 *없음*.

이것이 근본적 tradeoff — logical 백업은 느림(테이블당 큰 SELECT 1번 + 모든 애플리케이션 수준 행 포맷팅)이고 시간 차원에서 손실(PITR 없음), 그러나 버전 portable, 포맷 portable, on-disk 손상에서 살아남음.

### 기본 백업

```bash
# 단일 데이터베이스 백업
pg_dump dbname > backup.sql

# 사용자/호스트 지정
pg_dump -U username -h localhost dbname > backup.sql

# 압축 백업
pg_dump dbname | gzip > backup.sql.gz
```

### 포맷 옵션

```bash
# 평문 SQL (-Fp, 기본값)
pg_dump -Fp dbname > backup.sql

# 커스텀 포맷 (-Fc, 압축됨, 선택적 복원 가능)
pg_dump -Fc dbname > backup.dump

# 디렉토리 포맷 (-Fd, 병렬 백업/복원 지원)
pg_dump -Fd dbname -f backup_dir

# tar 포맷 (-Ft)
pg_dump -Ft dbname > backup.tar
```

### 선택적 백업

```bash
# 특정 테이블만
pg_dump -t users -t orders dbname > tables.sql

# 특정 테이블 제외
pg_dump -T logs -T temp_* dbname > backup.sql

# 스키마만 (데이터 제외)
pg_dump -s dbname > schema.sql

# 데이터만 (스키마 제외)
pg_dump -a dbname > data.sql

# 특정 스키마만
pg_dump -n public dbname > public_schema.sql
```

### Docker에서 백업

```bash
# Docker 컨테이너에서 pg_dump 실행
docker exec -t postgres-container pg_dump -U postgres dbname > backup.sql

# 압축 백업
docker exec -t postgres-container pg_dump -U postgres dbname | gzip > backup.sql.gz
```

---

## 3. pg_dumpall - 전체 클러스터 백업

모든 데이터베이스와 전역 객체(사용자, 권한 등)를 백업합니다.

```bash
# 전체 클러스터 백업
pg_dumpall -U postgres > full_backup.sql

# 전역 객체만 (사용자, Role 등)
pg_dumpall -U postgres --globals-only > globals.sql

# 역할만
pg_dumpall -U postgres --roles-only > roles.sql
```

---

## 4. pg_restore - 복원

### 이론: Point-In-Time Recovery (PITR)

PITR는 physical base backup과 WAL 파일의 연속 archive를 결합해서 base 백업 *이후의 어떤* 시점으로든 복구.

#### D.1 설정

1. **WAL archiving 활성화**. `archive_mode = on`과 `archive_command = 'cp %p /archive/%f'`(또는 S3 push 등) 설정. 가득 차는 모든 WAL segment가 archive에 복사됨.
2. **`pg_basebackup`으로 base 백업**. 시작 시간 기록.
3. **WAL이 생성되는 대로 연속 archive**. archive는 시간이 지나면서 커지지만 각 segment는 작음(기본 16 MB).

#### D.2 특정 시점으로 복구

"어제 14:32:00"으로 복구하려면:

1. base 백업을 fresh `PGDATA`에 복원.
2. `restore_command = 'cp /archive/%f %p'`와 `recovery_target_time = '2026-04-25 14:32:00'` 설정.
3. 서버 시작. PostgreSQL이 base 백업의 LSN부터 archive의 WAL을 replay하고, target time에서 멈추고, 데이터베이스를 엶.

"어떤 시점"의 granularity는 WAL 레코드 단위 — 사실상 COMMIT 단위.

#### D.3 RPO와 RTO

- **PITR의 RPO(Recovery Point Objective)** — WAL archive 간격만큼 낮음. `archive_timeout = 60s`이면 최대 60초의 작업을 잃을 수 있음.
- **PITR의 RTO(Recovery Time Objective)** — base 백업 복사 시간 + 그 이후 WAL replay 시간. 1 TB base + 24시간 WAL이면 시간 단위가 될 수 있음.

더 짧은 RTO를 위해 **streaming replication**(16번 레슨) 사용 — hot standby가 이미 떠 있고 WAL을 연속 replay 중이므로, failover가 초 단위.

### SQL 파일 복원

```bash
# 평문 SQL 복원
psql dbname < backup.sql

# 새 데이터베이스 생성 후 복원
createdb newdb
psql newdb < backup.sql
```

### 커스텀/디렉토리 포맷 복원

```bash
# 커스텀 포맷 복원
pg_restore -d dbname backup.dump

# 새 데이터베이스로 복원
createdb newdb
pg_restore -d newdb backup.dump

# 특정 테이블만 복원
pg_restore -d dbname -t users backup.dump

# 병렬 복원 (4 작업자)
pg_restore -d dbname -j 4 backup_dir
```

### 복원 옵션

```bash
# 기존 객체 삭제 후 복원
pg_restore -d dbname --clean backup.dump

# 오류 무시하고 계속
pg_restore -d dbname --if-exists backup.dump

# 데이터만 복원
pg_restore -d dbname --data-only backup.dump

# 스키마만 복원
pg_restore -d dbname --schema-only backup.dump
```

---

## 5. 물리적 백업 (pg_basebackup)

전체 데이터 디렉토리를 백업합니다.

```bash
# 기본 백업
pg_basebackup -D /backup/path -U postgres -Fp -Xs -P

# 압축 백업
pg_basebackup -D /backup/path -U postgres -Ft -z -P

# 옵션 설명:
# -D: 백업 디렉토리
# -Fp: 평문 포맷
# -Ft: tar 포맷
# -Xs: WAL 스트리밍
# -z: gzip 압축
# -P: 진행률 표시
```

### 이론: Physical 백업 — `pg_basebackup`

`pg_basebackup`은 서버가 실행 중일 때 `PGDATA` 디렉터리 전체를 복사. copy 동안 WAL도 stream해서 결과 백업이 일관되게 함.

#### C.1 메커니즘

1. **Replication 프로토콜로 연결**(replication 가능 role 필요).
2. **Primary에 `pg_start_backup('label');` 알림**(또는 내부 등가의 `--checkpoint=fast` 모드 실행).
3. **`PGDATA/` 아래 모든 파일을 destination에 복사**.
4. **복사 중 생성된 WAL 레코드를 병렬 stream**.
5. **Primary에 `pg_stop_backup();` 알리고**, stop 시점의 WAL position 캡처.
6. 결과는 stop position까지의 WAL과 결합되어 self-consistent 상태로 replay 가능한 `PGDATA` snapshot.

복원된 데이터베이스는 `pg_stop_backup()`의 LSN — 그 시점까지 commit된 모든 트랜잭션 포함.

#### C.2 무엇을 캡처

- `PGDATA`의 모든 것 — heap, 인덱스, 시스템 카탈로그, stop까지의 WAL, 설정 파일.
- 테이블스페이스(destination에서 재배치하려면 `--tablespace-mapping` 사용).

이는 복구 시점 source의 데이터 디렉터리와 byte-for-byte 동일. 복원은 "fresh `PGDATA`에 파일을 추출하고 서버 시작" — logical 복원보다 훨씬 빠름.

#### C.3 한계

- **메이저 버전 동일만**. on-disk 포맷이 PG 14와 PG 15 사이에 바뀜, 한 메이저 버전의 `pg_basebackup`을 다른 메이저 버전으로 복원 불가.
- **아키텍처와 OS endianness 동일**(대부분).
- **부분 선택 불가**. 클러스터 전체, 모든 데이터베이스, 모든 테이블.

### WAL 아카이빙 설정

`postgresql.conf`:
```
wal_level = replica
archive_mode = on
archive_command = 'cp %p /archive/%f'
```

---

## 6. 자동 백업 스크립트

### 일일 백업 스크립트

```bash
#!/bin/bash
# daily_backup.sh

# 설정
DB_NAME="mydb"
DB_USER="postgres"
BACKUP_DIR="/backup/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

# 백업 디렉토리 생성
mkdir -p $BACKUP_DIR

# 백업 실행
pg_dump -U $DB_USER -Fc $DB_NAME > $BACKUP_DIR/${DB_NAME}_${DATE}.dump

# 압축
gzip $BACKUP_DIR/${DB_NAME}_${DATE}.dump

# 오래된 백업 삭제
find $BACKUP_DIR -name "*.dump.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: ${DB_NAME}_${DATE}.dump.gz"
```

### Cron 설정

```bash
# crontab -e
# 매일 새벽 2시 백업
0 2 * * * /scripts/daily_backup.sh >> /var/log/backup.log 2>&1
```

---

## 7. 모니터링

### 데이터베이스 크기

```sql
-- 데이터베이스별 크기
SELECT
    datname,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;

-- 테이블별 크기
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC
LIMIT 10;
```

### 연결 상태

```sql
-- 현재 연결 수
SELECT COUNT(*) FROM pg_stat_activity;

-- 상태별 연결
SELECT state, COUNT(*)
FROM pg_stat_activity
GROUP BY state;

-- 활성 쿼리
SELECT
    pid,
    now() - query_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE state != 'idle'
  AND query NOT LIKE '%pg_stat_activity%'
ORDER BY duration DESC;
```

### 느린 쿼리

```sql
-- 오래 실행 중인 쿼리 (5초 이상)
SELECT
    pid,
    now() - query_start AS duration,
    query
FROM pg_stat_activity
WHERE state = 'active'
  AND now() - query_start > interval '5 seconds';
```

### 잠금 상태

```sql
-- 잠금 대기 중인 쿼리
SELECT
    blocked.pid AS blocked_pid,
    blocked.query AS blocked_query,
    blocking.pid AS blocking_pid,
    blocking.query AS blocking_query
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking
    ON blocking.pid = ANY(pg_blocking_pids(blocked.pid));
```

---

## 8. 성능 통계

### 테이블 통계

```sql
-- 테이블 접근 통계
SELECT
    schemaname,
    relname,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    n_tup_ins,
    n_tup_upd,
    n_tup_del
FROM pg_stat_user_tables
ORDER BY seq_scan DESC
LIMIT 10;
```

### 인덱스 사용률

```sql
-- 사용되지 않는 인덱스
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

### 캐시 히트율

```sql
-- 캐시 히트율 (99% 이상이 좋음)
SELECT
    sum(blks_hit) * 100.0 / sum(blks_hit + blks_read) AS cache_hit_ratio
FROM pg_stat_database;
```

---

## 9. 유지보수

### 이론: WAL Redo 알고리즘

02번 레슨 §D는 WAL을 "로그가 먼저 쓰이고, 데이터 파일은 lazy하게 갱신됨"으로 소개. 복구는 그 역과정입니다.

#### A.1 무엇이 replay되는가

PostgreSQL이 충돌 후 시작될 때(또는 primary로부터 WAL을 읽는 standby로서), **redo 루프**를 실행:

```
position = 마지막으로 완료된 checkpoint의 LSN
while position에 더 많은 WAL 레코드가 있는 동안:
    record = read_wal(position)
    apply(record)         # idempotent — 레코드 재적용은 안전
    position = record.next_lsn
```

각 WAL 레코드는 *물리적* 페이지 변경을 기술. 적용은 기록된 바이트를 기록된 페이지 offset에 쓰는 것. 적용은 **idempotent** — 변경이 이미 페이지에 있으면(충돌 전 페이지가 쓰였기 때문) 재적용은 같은 바이트 패턴을 만듭니다. 그래서 WAL replay가 디스크에 정확히 무엇이 닿았는지 알 필요 없이 "마지막 checkpoint"부터 안전하게 시작할 수 있습니다.

#### A.2 Full-page image

checkpoint 이후, 각 페이지를 건드리는 *첫* WAL 레코드는 diff뿐 아니라 **8 KB 페이지 전체 image**를 담음. 이는 torn write(전원이 나갈 때 부분적으로 쓰인 페이지)로부터 보호 — full-page image는 디스크에 무엇이 있든 페이지를 처음부터 재구성할 수 있음. 비용은 checkpoint 직후의 상당한 WAL 양 burst — `wal_compression`이 이를 줄임.

#### A.3 복구 종료

복구는 WAL의 끝에서 멈춤. 충돌 복구는 `pg_wal/`의 현재 WAL 끝. PITR는 target(`recovery_target_time`, `recovery_target_xid` 등)을 지정하고 그 지점에 도달하면 복구가 멈춤. 멈춘 뒤 데이터베이스가 연결을 위해 열림.

### VACUUM

불필요한 공간을 정리합니다.

```sql
-- 일반 VACUUM
VACUUM;
VACUUM users;

-- VACUUM FULL (테이블 재구성, 잠금 발생)
VACUUM FULL users;

-- VACUUM ANALYZE (통계 갱신 포함)
VACUUM ANALYZE users;
```

### ANALYZE

쿼리 최적화를 위한 통계를 수집합니다.

```sql
ANALYZE;
ANALYZE users;
```

### REINDEX

인덱스를 재구성합니다.

```sql
REINDEX TABLE users;
REINDEX DATABASE mydb;
```

### 자동 VACUUM 설정

`postgresql.conf`:
```
autovacuum = on
autovacuum_naptime = 1min
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
```

---

## 10. 로그 설정

`postgresql.conf`:

```
# 로그 대상
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d.log'

# 로그 레벨
log_min_messages = warning
log_min_error_statement = error

# 쿼리 로깅
log_statement = 'ddl'           # none, ddl, mod, all
log_duration = off
log_min_duration_statement = 1000  # 1초 이상 걸리는 쿼리만

# 연결 로깅
log_connections = on
log_disconnections = on
```

---

## 11. 보안 설정

### pg_hba.conf

```
# TYPE  DATABASE    USER        ADDRESS         METHOD

# 로컬 연결
local   all         all                         peer

# IPv4 로컬 연결
host    all         all         127.0.0.1/32    scram-sha-256

# 특정 네트워크 허용
host    mydb        appuser     192.168.1.0/24  scram-sha-256

# 특정 IP 거부
host    all         all         192.168.1.100   reject
```

### SSL 설정

```
# postgresql.conf
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
```

---

## 12. 실습 예제

### 실습 1: 백업 및 복원

```bash
# 1. 백업
pg_dump -U postgres -Fc mydb > mydb_backup.dump

# 2. 새 데이터베이스 생성
createdb -U postgres mydb_restored

# 3. 복원
pg_restore -U postgres -d mydb_restored mydb_backup.dump

# 4. 확인
psql -U postgres -d mydb_restored -c "SELECT COUNT(*) FROM users;"
```

### 실습 2: 모니터링 쿼리 저장

```sql
-- 모니터링 뷰 생성
CREATE VIEW v_db_stats AS
SELECT
    datname,
    pg_size_pretty(pg_database_size(datname)) AS size,
    numbackends AS connections
FROM pg_database
WHERE datistemplate = false;

CREATE VIEW v_slow_queries AS
SELECT
    pid,
    now() - query_start AS duration,
    state,
    query
FROM pg_stat_activity
WHERE state = 'active'
  AND now() - query_start > interval '5 seconds';

-- 사용
SELECT * FROM v_db_stats;
SELECT * FROM v_slow_queries;
```

### 실습 3: 유지보수 스크립트

```sql
-- 정기 유지보수 프로시저
CREATE PROCEDURE run_maintenance()
AS $$
BEGIN
    -- 통계 갱신
    ANALYZE;

    -- 불필요한 공간 정리
    VACUUM;

    RAISE NOTICE '유지보수 완료: %', NOW();
END;
$$ LANGUAGE plpgsql;

-- 실행
CALL run_maintenance();
```

---

## 13. 체크리스트

### 일일 체크

- [ ] 백업 성공 확인
- [ ] 디스크 사용량 확인
- [ ] 연결 수 확인
- [ ] 오류 로그 확인

### 주간 체크

- [ ] 인덱스 사용률 확인
- [ ] 느린 쿼리 분석
- [ ] 테이블 크기 추이

### 월간 체크

- [ ] 백업 복원 테스트
- [ ] 불필요한 데이터 정리
- [ ] 성능 추이 분석

---

**이전**: [트리거](./12_Triggers.md) | **다음**: [JSON과 JSONB](./14_JSON_JSONB.md)
