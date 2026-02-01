# 백업과 운영

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

## 마무리

이것으로 PostgreSQL 학습 자료를 마칩니다.

**학습 순서 복습**:
1. 기초 → DB 관리 → 테이블 → CRUD → 조건/정렬
2. JOIN → 집계 → 서브쿼리 → 뷰/인덱스
3. 함수 → 트랜잭션 → 트리거 → 백업/운영

더 깊이 있는 학습을 위해:
- [PostgreSQL 공식 문서](https://www.postgresql.org/docs/)
- [PostgreSQL Tutorial](https://www.postgresqltutorial.com/)
