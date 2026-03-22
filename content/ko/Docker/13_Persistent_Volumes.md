# 13. 영구 볼륨(Persistent Volumes)

**이전**: [보안 모범 사례](./12_Security_Best_Practices.md) | **다음**: [멀티 스테이지 빌드 패턴](./14_Multi_Stage_Build_Patterns.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Docker 볼륨(Volume), 바인드 마운트(Bind Mount), tmpfs 마운트를 구분하고 적절한 옵션을 선택한다
2. 이름이 있는 볼륨(Named Volume)과 익명 볼륨(Anonymous Volume)을 생성하고 관리한다
3. 볼륨 드라이버와 플러그인을 사용하여 원격 및 클라우드 기반 스토리지를 구성한다
4. 컨테이너화된 애플리케이션의 데이터 백업 및 복원 전략을 구현한다
5. 여러 컨테이너 간에 볼륨을 안전하게 공유한다
6. PostgreSQL, MySQL, MongoDB 등 데이터베이스에 대한 스토리지 모범 사례를 적용한다
7. 볼륨 검사, 정리 및 라이프사이클 관리 명령어를 사용한다

## 목차
1. [Docker 스토리지 개요](#1-docker-스토리지-개요)
2. [볼륨 vs 바인드 마운트 vs tmpfs](#2-볼륨-vs-바인드-마운트-vs-tmpfs)
3. [이름 있는 볼륨과 익명 볼륨](#3-이름-있는-볼륨과-익명-볼륨)
4. [볼륨 드라이버와 플러그인](#4-볼륨-드라이버와-플러그인)
5. [볼륨 라이프사이클 관리](#5-볼륨-라이프사이클-관리)
6. [데이터 백업 및 복원](#6-데이터-백업-및-복원)
7. [컨테이너 간 볼륨 공유](#7-컨테이너-간-볼륨-공유)
8. [데이터베이스 스토리지 모범 사례](#8-데이터베이스-스토리지-모범-사례)
9. [볼륨 명령어 참조](#9-볼륨-명령어-참조)
10. [연습 문제](#10-연습-문제)

**난이도**: ⭐⭐⭐

---

컨테이너는 설계상 일시적(ephemeral)입니다. 컨테이너를 제거하면 쓰기 가능 레이어에 기록된 모든 데이터가 사라집니다. 영구 볼륨은 데이터를 컨테이너 라이프사이클에서 분리하여 이 근본적인 문제를 해결합니다. Docker의 스토리지 서브시스템을 이해하는 것은 데이터베이스, 메시지 큐, 파일 기반 애플리케이션과 같은 상태가 있는(stateful) 워크로드를 프로덕션에서 실행하는 데 필수적입니다.

---

## 1. Docker 스토리지 개요

### 컨테이너 파일시스템(Container Filesystem)

모든 Docker 컨테이너는 이미지의 읽기 전용 레이어와 그 위에 있는 얇은 쓰기 가능 레이어로 구성된 계층형 파일시스템을 갖습니다.

```
┌──────────────────────────────────────────────┐
│           Container Writable Layer           │  ← 컨테이너 제거 시 손실
├──────────────────────────────────────────────┤
│           Image Layer N (read-only)          │
├──────────────────────────────────────────────┤
│           Image Layer N-1 (read-only)        │
├──────────────────────────────────────────────┤
│           ...                                │
├──────────────────────────────────────────────┤
│           Base Image Layer (read-only)       │
└──────────────────────────────────────────────┘
```

쓰기 가능 레이어는 **COW(Copy-on-Write)** 전략을 사용합니다. 컨테이너가 하위 레이어의 파일을 수정하면, 해당 파일이 먼저 쓰기 가능 레이어로 복사됩니다. 읽기에는 효율적이지만 쓰기가 많은 워크로드에는 오버헤드가 추가됩니다.

### 영구 스토리지가 필요한 이유

```
┌─────────────────────────────────────────────────────────────────┐
│                   영구 스토리지 없이 (Without)                    │
│                                                                  │
│  Container A (실행 중)              Container A (제거됨)          │
│  ┌────────────────────┐         ┌────────────────────┐          │
│  │  /var/lib/mysql     │   ──►  │   데이터 손실!      │          │
│  │  (writable layer)   │         │                    │          │
│  └────────────────────┘         └────────────────────┘          │
│                                                                  │
│                   영구 스토리지와 함께 (With)                      │
│                                                                  │
│  Container A          Container B (대체)                         │
│  ┌──────────┐         ┌──────────┐                              │
│  │  mount ──┼────┐    │  mount ──┼────┐                         │
│  └──────────┘    │    └──────────┘    │                         │
│                  ▼                    ▼                          │
│           ┌──────────────────────────────┐                      │
│           │    Volume: db_data           │  ← 데이터 유지       │
│           └──────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 볼륨 vs 바인드 마운트 vs tmpfs

Docker는 데이터를 유지하기 위한 세 가지 메커니즘을 제공합니다:

### 비교 표

| 기능 | 볼륨(Volumes) | 바인드 마운트(Bind Mounts) | tmpfs |
|---|---|---|---|
| Docker에 의해 관리 | 예 | 아니오 | 아니오 |
| 호스트 위치 | `/var/lib/docker/volumes/` | 호스트 어디든 | 메모리만 |
| 컨테이너 제거 후 유지 | 예 | 예 (호스트 파일 유지) | 아니오 |
| 볼륨 드라이버 지원 | 예 | 아니오 | 아니오 |
| 이미지 데이터로 사전 채움 | 예 | 아니오 | 아니오 |
| 성능 | 네이티브 | 네이티브 | 가장 빠름 (RAM) |
| 사용 사례 | 프로덕션 데이터 | 개발, 설정 | 민감한 임시 데이터 |

### 볼륨(Volumes)

볼륨은 데이터를 유지하기 위한 권장 메커니즘입니다. Docker가 호스트 파일시스템의 스토리지 위치를 관리합니다.

```bash
# 이름 있는 볼륨 생성 및 사용
docker volume create mydata
docker run -d --name app -v mydata:/app/data nginx

# --mount 구문 사용 (더 명시적, 권장)
docker run -d --name app \
  --mount type=volume,source=mydata,target=/app/data \
  nginx
```

### 바인드 마운트(Bind Mounts)

바인드 마운트는 특정 호스트 디렉토리를 컨테이너에 매핑합니다. 라이브 코드 리로딩이 필요한 개발 워크플로우에 이상적입니다.

```bash
# 현재 디렉토리를 바인드 마운트
docker run -d --name dev \
  -v $(pwd)/src:/app/src \
  node:20

# --mount 구문 사용
docker run -d --name dev \
  --mount type=bind,source=$(pwd)/src,target=/app/src \
  node:20

# 읽기 전용 바인드 마운트
docker run -d --name app \
  --mount type=bind,source=$(pwd)/config,target=/app/config,readonly \
  myapp
```

> **주의**: 바인드 마운트는 컨테이너의 파일을 덮어쓸 수 있습니다. 빈 호스트 디렉토리를 파일이 있는 컨테이너 경로에 마운트하면 해당 파일이 보이지 않게 됩니다.

### tmpfs 마운트

tmpfs 마운트는 호스트 메모리에만 데이터를 저장합니다. 데이터는 디스크에 기록되지 않으며 컨테이너가 중지되면 손실됩니다.

```bash
# 민감한 임시 데이터를 위한 tmpfs 마운트
docker run -d --name secure \
  --mount type=tmpfs,target=/app/secrets,tmpfs-size=100m \
  myapp

# 짧은 구문
docker run -d --name secure \
  --tmpfs /app/secrets:size=100m \
  myapp
```

tmpfs 사용 사례:
- 임시 세션 데이터
- 디스크에 절대 기록되어서는 안 되는 시크릿
- 계산을 위한 스크래치 공간

---

## 3. 이름 있는 볼륨과 익명 볼륨

### 이름 있는 볼륨(Named Volumes)

이름 있는 볼륨은 명시적인 이름을 가지며 참조 및 관리가 쉽습니다.

```bash
# 이름 있는 볼륨 생성
docker volume create app_data

# 볼륨 목록
docker volume ls

# docker run에서 사용
docker run -d -v app_data:/data myapp

# docker-compose.yml에서 사용
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    image: myapp
    volumes:
      - app_data:/data

volumes:
  app_data:
    driver: local
```

### 익명 볼륨(Anonymous Volumes)

익명 볼륨은 이름 없이 마운트 포인트를 지정할 때 생성됩니다. Docker가 임의의 해시를 이름으로 할당합니다.

```bash
# 익명 볼륨 -- Docker가 임의의 이름 생성
docker run -d -v /data myapp

# Dockerfile의 VOLUME 지시어도 익명 볼륨을 생성
```

```dockerfile
# Dockerfile
FROM postgres:16
VOLUME /var/lib/postgresql/data
```

```bash
# 볼륨 목록 -- 익명 볼륨은 해시 이름을 가짐
docker volume ls
# DRIVER    VOLUME NAME
# local     app_data
# local     a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6

# 익명 볼륨은 추적 및 관리가 어려움
# 이름 있는 볼륨이 항상 권장됨
```

### 볼륨 레이블(Volume Labels)

레이블을 사용하여 볼륨에 메타데이터를 첨부할 수 있습니다:

```bash
# 레이블이 있는 볼륨 생성
docker volume create \
  --label project=myapp \
  --label environment=production \
  myapp_data

# 레이블로 볼륨 필터링
docker volume ls --filter label=project=myapp
```

---

## 4. 볼륨 드라이버와 플러그인

### 로컬 드라이버 옵션(Local Driver Options)

기본 `local` 드라이버는 특정 파일시스템 유형으로 볼륨을 생성하는 옵션을 지원합니다:

```bash
# 특정 마운트 옵션으로 볼륨 생성
docker volume create --driver local \
  --opt type=nfs \
  --opt o=addr=192.168.1.100,rw \
  --opt device=:/exports/data \
  nfs_data

# tmpfs 기반 볼륨 생성
docker volume create --driver local \
  --opt type=tmpfs \
  --opt device=tmpfs \
  --opt o=size=500m \
  tmpfs_vol

# 특정 블록 장치에 ext4 볼륨 생성
docker volume create --driver local \
  --opt type=ext4 \
  --opt device=/dev/sdb1 \
  fast_storage
```

### 서드파티 볼륨 드라이버

```
┌────────────────────────────────────────────────────────────┐
│                볼륨 드라이버 생태계                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  클라우드 스토리지          │  네트워크 스토리지              │
│  ┌───────────────────┐   │  ┌───────────────────┐         │
│  │ REX-Ray (AWS EBS,  │   │  │ NFS               │         │
│  │   Azure Disk, GCE) │   │  │ CIFS/Samba        │         │
│  │ NetApp Trident     │   │  │ GlusterFS         │         │
│  │ DigitalOcean Block │   │  │ CephFS            │         │
│  └───────────────────┘   │  └───────────────────┘         │
│                          │                                  │
│  특수 목적               │  분산 스토리지                    │
│  ┌───────────────────┐   │  ┌───────────────────┐         │
│  │ Convoy (스냅샷)     │   │  │ Portworx          │         │
│  │ Flocker (마이그레이션)│   │  │ StorageOS         │         │
│  │ Local-persist       │   │  │ Longhorn          │         │
│  └───────────────────┘   │  └───────────────────┘         │
└────────────────────────────────────────────────────────────┘
```

```bash
# 볼륨 플러그인 설치
docker plugin install rexray/ebs

# 플러그인으로 볼륨 생성
docker volume create -d rexray/ebs \
  --opt size=100 \
  --opt volumetype=gp3 \
  ebs_data

# docker-compose.yml에서 사용
```

```yaml
# 외부 볼륨 드라이버를 사용하는 docker-compose.yml
version: "3.9"
services:
  db:
    image: postgres:16
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
    driver: rexray/ebs
    driver_opts:
      size: "100"
      volumetype: "gp3"
```

---

## 5. 볼륨 라이프사이클 관리

### 볼륨 생성 및 검사

```bash
# 볼륨 생성
docker volume create mydata

# 볼륨 세부 정보 검사
docker volume inspect mydata
```

```json
[
    {
        "CreatedAt": "2025-01-15T10:30:00Z",
        "Driver": "local",
        "Labels": {},
        "Mountpoint": "/var/lib/docker/volumes/mydata/_data",
        "Name": "mydata",
        "Options": {},
        "Scope": "local"
    }
]
```

### 사용하지 않는 볼륨 찾기

```bash
# 모든 볼륨 나열
docker volume ls

# 미사용(dangling) 볼륨 나열
docker volume ls -f dangling=true

# 볼륨 디스크 사용량 표시
docker system df -v | grep "VOLUME NAME" -A 100
```

### 볼륨 정리(Pruning)

```bash
# 모든 미사용 볼륨 제거 (대화형 확인)
docker volume prune

# 확인 없이 모든 미사용 볼륨 제거
docker volume prune -f

# 레이블이 있는 볼륨을 포함하여 모든 미사용 볼륨 제거
docker volume prune --all

# 경고: 정리는 되돌릴 수 없습니다! 실행 전에 항상 확인하세요.
```

### 볼륨 라이프사이클 다이어그램

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  create   │────►│  attach   │────►│  detach   │────►│  remove   │
│           │     │ (docker   │     │ (docker   │     │ (docker   │
│ docker    │     │  run -v)  │     │  stop/rm) │     │  volume   │
│ volume    │     │           │     │           │     │  rm)      │
│ create    │     │           │     │           │     │           │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                       │                │
                       │     ┌──────────┘
                       ▼     ▼
                  ┌──────────────┐
                  │   re-attach   │
                  │  (새 컨테이너가 │
                  │   같은 볼륨을   │
                  │   마운트)      │
                  └──────────────┘
```

---

## 6. 데이터 백업 및 복원

### 헬퍼 컨테이너를 사용한 백업 전략

```bash
# 볼륨을 tar 아카이브로 백업
docker run --rm \
  -v mydata:/source:ro \
  -v $(pwd)/backups:/backup \
  alpine \
  tar czf /backup/mydata-$(date +%Y%m%d_%H%M%S).tar.gz -C /source .

# 볼륨 내용의 압축 아카이브가 생성됩니다
```

### 백업에서 복원

```bash
# 새 볼륨 생성
docker volume create mydata_restored

# 백업에서 복원
docker run --rm \
  -v mydata_restored:/target \
  -v $(pwd)/backups:/backup:ro \
  alpine \
  sh -c "cd /target && tar xzf /backup/mydata-20250115_103000.tar.gz"
```

### 자동 백업 스크립트

```bash
#!/bin/bash
# backup-volumes.sh -- Docker 볼륨 자동 백업

BACKUP_DIR="/opt/backups/docker-volumes"
RETENTION_DAYS=30
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# 모든 이름 있는 볼륨 가져오기
volumes=$(docker volume ls -q --filter dangling=false)

for vol in $volumes; do
    echo "볼륨 백업 중: $vol"
    docker run --rm \
        -v "$vol":/source:ro \
        -v "$BACKUP_DIR":/backup \
        alpine \
        tar czf "/backup/${vol}_${DATE}.tar.gz" -C /source .

    if [ $? -eq 0 ]; then
        echo "  ✓ 백업 성공: ${vol}_${DATE}.tar.gz"
    else
        echo "  ✗ 백업 실패: $vol"
    fi
done

# 오래된 백업 정리
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
echo "${RETENTION_DAYS}일 이상 된 백업을 정리했습니다"
```

### 데이터베이스별 백업

데이터베이스의 경우, 파일시스템 수준 백업보다 논리적 백업(SQL 덤프)을 권장합니다:

```bash
# PostgreSQL 백업
docker exec my_postgres \
  pg_dump -U myuser -d mydb > backup.sql

# MySQL 백업
docker exec my_mysql \
  mysqldump -u root -p"$MYSQL_ROOT_PASSWORD" mydb > backup.sql

# MongoDB 백업
docker exec my_mongo \
  mongodump --archive=/tmp/backup.archive --gzip
docker cp my_mongo:/tmp/backup.archive ./backup.archive
```

---

## 7. 컨테이너 간 볼륨 공유

### 공유 볼륨 패턴(Shared Volume Pattern)

여러 컨테이너가 데이터 교환을 위해 동일한 볼륨을 마운트할 수 있습니다:

```yaml
# docker-compose.yml
version: "3.9"
services:
  # 쓰기 컨테이너 - 로그 파일 생성
  writer:
    image: alpine
    command: sh -c "while true; do echo $$(date) >> /shared/log.txt; sleep 5; done"
    volumes:
      - shared_data:/shared

  # 읽기 컨테이너 - 로그 파일 처리
  reader:
    image: alpine
    command: tail -f /shared/log.txt
    volumes:
      - shared_data:/shared:ro
    depends_on:
      - writer

volumes:
  shared_data:
```

### 공유 정적 자산을 가진 웹 애플리케이션

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    build: .
    volumes:
      - static_files:/app/static

  nginx:
    image: nginx:alpine
    volumes:
      - static_files:/usr/share/nginx/html/static:ro
    ports:
      - "80:80"
    depends_on:
      - app

volumes:
  static_files:
```

### 동시성 고려사항(Concurrency Considerations)

```
┌──────────────────────────────────────────────────────────────┐
│              볼륨 공유: 동시성 위험                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  안전한 패턴:                                                  │
│  ┌───────────────────────────────────────────────────┐       │
│  │ • 하나의 쓰기, 여러 읽기 (읽기 전용 마운트)         │       │
│  │ • 각 컨테이너가 다른 파일에 쓰기                    │       │
│  │ • 애플리케이션 수준 잠금 (예: flock)                │       │
│  └───────────────────────────────────────────────────┘       │
│                                                               │
│  위험한 패턴:                                                  │
│  ┌───────────────────────────────────────────────────┐       │
│  │ • 같은 파일에 여러 쓰기                            │       │
│  │ • 조율 없이 데이터베이스 파일 공유                  │       │
│  │ • 잠금 메커니즘 없음                               │       │
│  └───────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. 데이터베이스 스토리지 모범 사례

### PostgreSQL

```yaml
# docker-compose.yml
version: "3.9"
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: appuser
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    volumes:
      # 데이터 디렉토리용 이름 있는 볼륨
      - pgdata:/var/lib/postgresql/data
      # 커스텀 설정용 바인드 마운트
      - ./postgresql.conf:/etc/postgresql/postgresql.conf:ro
      # 초기화 스크립트용 바인드 마운트
      - ./init-scripts:/docker-entrypoint-initdb.d:ro
    secrets:
      - db_password
    deploy:
      resources:
        limits:
          memory: 2G

volumes:
  pgdata:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/postgres

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

### MySQL

```yaml
# docker-compose.yml
version: "3.9"
services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD_FILE: /run/secrets/mysql_root_pw
      MYSQL_DATABASE: myapp
    volumes:
      - mysqldata:/var/lib/mysql
      - ./my.cnf:/etc/mysql/conf.d/custom.cnf:ro
    secrets:
      - mysql_root_pw
    # 데이터 일관성 보장
    command: >
      --innodb-flush-log-at-trx-commit=1
      --sync-binlog=1

volumes:
  mysqldata:
    driver: local

secrets:
  mysql_root_pw:
    file: ./secrets/mysql_root_pw.txt
```

### MongoDB

```yaml
# docker-compose.yml
version: "3.9"
services:
  mongo:
    image: mongo:7
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD_FILE: /run/secrets/mongo_pw
    volumes:
      - mongodata:/data/db
      - mongoconfigdb:/data/configdb
    secrets:
      - mongo_pw

volumes:
  mongodata:
    driver: local
  mongoconfigdb:
    driver: local

secrets:
  mongo_pw:
    file: ./secrets/mongo_pw.txt
```

### 일반 데이터베이스 스토리지 가이드라인

```
┌──────────────────────────────────────────────────────────────┐
│              데이터베이스 볼륨 모범 사례                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. 항상 이름 있는 볼륨을 사용 (익명 볼륨 사용 금지)            │
│  2. 프로덕션에서 데이터베이스 데이터에 바인드 마운트 사용 금지   │
│  3. 적절한 파일시스템 권한 설정                                 │
│  4. 조직화를 위한 볼륨 레이블 사용                              │
│  5. 정기적인 백업 일정 구현                                     │
│  6. 복원 절차를 주기적으로 테스트                                │
│  7. 볼륨 디스크 사용량 모니터링                                 │
│  8. 쓰기가 많은 워크로드에 전용 스토리지 사용                   │
│  9. 데이터베이스 볼륨에 대한 I/O 스케줄러 튜닝 고려             │
│  10. 데이터베이스 인스턴스 간 볼륨 공유 금지                    │
│      (데이터베이스 네이티브 복제를 사용하는 경우 제외)           │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. 볼륨 명령어 참조

### 필수 명령어

```bash
# 볼륨 생성
docker volume create [OPTIONS] [VOLUME]

# 볼륨 나열
docker volume ls [OPTIONS]

# 볼륨 검사
docker volume inspect [VOLUME]

# 볼륨 제거
docker volume rm [VOLUME...]

# 미사용 볼륨 제거
docker volume prune [OPTIONS]
```

### 실용적인 예시

```bash
# 특정 레이블로 볼륨 생성
docker volume create --label env=prod --label app=web webdata

# 필터로 볼륨 나열
docker volume ls --filter driver=local
docker volume ls --filter label=env=prod
docker volume ls --filter dangling=true

# 볼륨 목록 포맷팅
docker volume ls --format "{{.Name}}\t{{.Driver}}\t{{.Mountpoint}}"

# 볼륨 마운트 포인트 가져오기
docker volume inspect --format '{{.Mountpoint}}' mydata

# 볼륨을 사용하는 컨테이너 확인
docker ps -a --filter volume=mydata \
  --format "{{.ID}}\t{{.Names}}\t{{.Status}}"

# 볼륨 간 데이터 복사
docker run --rm \
  -v source_vol:/source:ro \
  -v target_vol:/target \
  alpine sh -c "cp -a /source/. /target/"

# 전체 볼륨 디스크 사용량 확인
docker system df -v
```

---

## 10. 연습 문제

### 연습 1: 볼륨 기초 (초급)

이름 있는 볼륨을 생성하고, 데이터를 쓰는 컨테이너를 실행한 후, 컨테이너를 제거하고 새 컨테이너에서 데이터가 유지되는지 확인하세요.

```bash
# 1. "exercise_data"라는 이름 있는 볼륨 생성
# 2. alpine 컨테이너를 실행하여 /data/hello.txt에 "Hello Volumes!" 쓰기
# 3. 컨테이너 제거
# 4. 같은 볼륨을 마운트하는 새 alpine 컨테이너 실행
# 5. 파일 내용 확인
```

<details>
<summary>풀이</summary>

```bash
docker volume create exercise_data
docker run --rm -v exercise_data:/data alpine sh -c "echo 'Hello Volumes!' > /data/hello.txt"
docker run --rm -v exercise_data:/data alpine cat /data/hello.txt
# 출력: Hello Volumes!
docker volume rm exercise_data
```

</details>

### 연습 2: 백업 및 복원 (중급)

이름 있는 볼륨으로 PostgreSQL 컨테이너를 설정하고, 데이터를 삽입하고, 볼륨을 백업하고, 새 볼륨으로 복원한 후 데이터를 확인하세요.

```bash
# 1. "pg_exercise" 이름 있는 볼륨으로 PostgreSQL 컨테이너 시작
# 2. 테이블 생성 및 샘플 데이터 삽입
# 3. tar 방식으로 볼륨 백업
# 4. "pg_exercise_restored" 새 볼륨 생성
# 5. 새 볼륨으로 백업 복원
# 6. 복원된 볼륨으로 새 PostgreSQL 컨테이너 시작
# 7. 데이터 확인
```

<details>
<summary>풀이</summary>

```bash
# PostgreSQL 시작
docker run -d --name pg_test \
  -e POSTGRES_PASSWORD=testpass \
  -v pg_exercise:/var/lib/postgresql/data \
  postgres:16-alpine

# 초기화 대기
sleep 5

# 데이터 삽입
docker exec pg_test psql -U postgres -c "
  CREATE TABLE users (id SERIAL, name TEXT);
  INSERT INTO users (name) VALUES ('Alice'), ('Bob');
"

# 일관된 백업을 위해 컨테이너 중지
docker stop pg_test

# 백업
docker run --rm \
  -v pg_exercise:/source:ro \
  -v $(pwd):/backup \
  alpine tar czf /backup/pg_backup.tar.gz -C /source .

# 복원 볼륨 생성 및 복원
docker volume create pg_exercise_restored
docker run --rm \
  -v pg_exercise_restored:/target \
  -v $(pwd):/backup:ro \
  alpine sh -c "cd /target && tar xzf /backup/pg_backup.tar.gz"

# 새 컨테이너로 확인
docker run -d --name pg_restored \
  -e POSTGRES_PASSWORD=testpass \
  -v pg_exercise_restored:/var/lib/postgresql/data \
  postgres:16-alpine

sleep 5
docker exec pg_restored psql -U postgres -c "SELECT * FROM users;"

# 정리
docker rm -f pg_test pg_restored
docker volume rm pg_exercise pg_exercise_restored
rm pg_backup.tar.gz
```

</details>

### 연습 3: 멀티 컨테이너 볼륨 공유 (중급)

"generator" 컨테이너가 공유 볼륨에 타임스탬프 항목을 쓰고 "web" 컨테이너가 nginx를 통해 해당 항목을 제공하는 docker-compose 설정을 만드세요.

<details>
<summary>풀이</summary>

```yaml
# docker-compose.yml
version: "3.9"
services:
  generator:
    image: alpine
    command: >
      sh -c "mkdir -p /shared/html &&
             while true; do
               echo \"<p>Generated at: $$(date)</p>\" >> /shared/html/index.html;
               sleep 10;
             done"
    volumes:
      - shared:/shared

  web:
    image: nginx:alpine
    volumes:
      - shared:/usr/share/nginx:ro
    ports:
      - "8080:80"
    depends_on:
      - generator

volumes:
  shared:
```

```bash
docker compose up -d
# http://localhost:8080을 방문하여 타임스탬프 항목 확인
# 기다린 후 새로고침하여 새 항목 확인
docker compose down -v
```

</details>

### 연습 4: 볼륨 드라이버 탐구 (고급)

NFS 기반 볼륨을 생성하고(또는 특정 마운트 옵션으로 로컬 드라이버로 시뮬레이션) 여러 컨테이너에서 동시에 사용할 수 있음을 보여주세요.

<details>
<summary>풀이</summary>

```bash
# 특정 로컬 드라이버 옵션으로 볼륨 생성 (NFS 시뮬레이션)
docker volume create \
  --driver local \
  --opt type=tmpfs \
  --opt device=tmpfs \
  --opt o=size=50m \
  shared_tmpfs

# 같은 볼륨을 사용하는 두 컨테이너 실행
docker run -d --name writer \
  -v shared_tmpfs:/data \
  alpine sh -c "while true; do date >> /data/log.txt; sleep 2; done"

docker run -d --name reader \
  -v shared_tmpfs:/data:ro \
  alpine sh -c "while true; do echo '--- Latest ---'; tail -3 /data/log.txt 2>/dev/null; sleep 5; done"

# 읽기 컨테이너 로그 확인
sleep 10
docker logs reader

# 볼륨 검사
docker volume inspect shared_tmpfs

# 정리
docker rm -f writer reader
docker volume rm shared_tmpfs
```

</details>

---

**이전**: [보안 모범 사례](./12_Security_Best_Practices.md) | **다음**: [멀티 스테이지 빌드 패턴](./14_Multi_Stage_Build_Patterns.md)
