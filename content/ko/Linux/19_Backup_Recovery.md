# 백업 및 복구

**이전**: [로그 관리](./18_Log_Management.md) | **다음**: [커널 관리](./20_Kernel_Management.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. RTO 및 RPO 목표에 따라 3-2-1 규칙(3-2-1 rule)을 사용한 백업 전략을 설계한다
2. 공간 효율을 위해 하드 링크(hard link)를 활용한 rsync 증분 백업을 수행한다
3. 암호화, 압축, 자동 정리(automated pruning)를 포함한 Borg Backup 저장소를 구성한다
4. dd와 LVM 스냅샷을 사용하여 시스템 수준의 디스크 이미지를 생성한다
5. 백업에서 부팅 가능한 시스템까지의 베어메탈(bare-metal) 재해복구 절차를 실행한다
6. systemd 타이머로 백업 워크플로우를 자동화하고 모니터링 알림을 구현한다

---

## 목차

1. [백업 전략 개요](#1-백업-전략-개요)
2. [rsync 고급 사용법](#2-rsync-고급-사용법)
3. [Borg Backup](#3-borg-backup)
4. [tar/cpio 백업](#4-tarcpio-백업)
5. [시스템 이미지 백업](#5-시스템-이미지-백업)
6. [재해복구 전략](#6-재해복구-전략)
7. [자동화 및 모니터링](#7-자동화-및-모니터링)

---

모든 시스템은 결국 장애에 직면하게 됩니다 -- 하드웨어 결함, 인적 오류, 랜섬웨어, 자연재해 등 그 원인은 다양합니다. 문제는 데이터를 복구해야 하는 *일이 일어날지 여부*가 아니라 *언제* 일어나는가입니다. 잘 설계된 백업 및 복구 전략은 비즈니스를 계속 운영하게 해주는 보험입니다. 이 레슨에서는 백업을 단순한 체크리스트 항목에서 실전에서 검증된 복구 역량으로 전환할 수 있는 도구, 기술, 계획 원칙을 다룹니다.

## 1. 백업 전략 개요

### 3-2-1 백업 규칙

```
┌─────────────────────────────────────────────────────────────┐
│                    3-2-1 백업 규칙                          │
├─────────────────────────────────────────────────────────────┤
│  3: 데이터 사본 3개 유지                                    │
│     └── 원본 + 백업 2개                                     │
│                                                             │
│  2: 서로 다른 2종류의 저장 매체                             │
│     └── 로컬 디스크 + 외장 디스크 또는 NAS                  │
│                                                             │
│  1: 1개는 오프사이트 (원격지)                               │
│     └── 클라우드 또는 물리적 원격지                         │
└─────────────────────────────────────────────────────────────┘
```

### 백업 유형

| 유형 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **전체 백업** | 모든 데이터 복사 | 복구 간단 | 시간/용량 많이 소요 |
| **증분 백업** | 마지막 백업 이후 변경분만 | 빠르고 용량 적음 | 복구 시 체인 필요 |
| **차등 백업** | 마지막 전체 백업 이후 변경분 | 증분보다 복구 간단 | 증분보다 용량 큼 |
| **스냅샷** | 특정 시점의 파일시스템 상태 | 즉시 생성 | 저장소 의존적 |

### RTO와 RPO

```
┌─────────────────────────────────────────────────────────────┐
│ RPO (Recovery Point Objective)                              │
│ = 허용 가능한 데이터 손실 시간                              │
│ = 마지막 백업 이후 얼마나 데이터를 잃어도 되는가?          │
│                                                             │
│ RTO (Recovery Time Objective)                               │
│ = 서비스 복구까지 허용 가능한 시간                          │
│ = 장애 발생 후 얼마 내에 복구해야 하는가?                   │
└─────────────────────────────────────────────────────────────┘

타임라인:
──────────────────────────────────────────────────────────────
     마지막 백업           장애 발생           복구 완료
         │                    │                   │
         │◄─────── RPO ──────►│◄────── RTO ──────►│
         │    (데이터 손실)    │   (다운타임)      │
```

---

## 2. rsync 고급 사용법

### 기본 문법

```bash
rsync [옵션] 원본 대상

# 로컬 복사
rsync -av /source/ /backup/

# 원격 복사 (SSH)
rsync -av /source/ user@server:/backup/
rsync -av user@server:/source/ /backup/
```

### 주요 옵션

```bash
# 기본 옵션 조합
rsync -avz --progress /source/ /backup/

# 상세 옵션
-a, --archive       # 아카이브 모드 (-rlptgoD와 동일)
-v, --verbose       # 상세 출력
-z, --compress      # 전송 중 압축
-P                  # --progress --partial 조합
--progress          # 진행 상황 표시
--partial           # 부분 전송 파일 유지

# 삭제 옵션
--delete            # 대상에만 있는 파일 삭제
--delete-before     # 전송 전 삭제
--delete-after      # 전송 후 삭제
--delete-excluded   # 제외된 파일도 삭제

# 동기화 정밀 옵션
-c, --checksum      # 체크섬으로 비교 (느림)
-u, --update        # 대상이 더 새로우면 건너뛰기
--ignore-existing   # 존재하는 파일 건너뛰기
```

### 제외 패턴

```bash
# 특정 패턴 제외
rsync -av --exclude='*.log' --exclude='cache/' /source/ /backup/

# 제외 파일 사용
rsync -av --exclude-from='exclude.txt' /source/ /backup/
```

```bash
# exclude.txt 예시
*.log
*.tmp
*.cache
.git/
node_modules/
__pycache__/
.DS_Store
Thumbs.db
```

### 증분 백업 스크립트

```bash
#!/bin/bash
# incremental-backup.sh

# 설정
SOURCE="/data"
BACKUP_BASE="/backup"
LATEST_LINK="$BACKUP_BASE/latest"
DATE=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_PATH="$BACKUP_BASE/$DATE"

# 이전 백업이 있으면 하드링크 사용
if [ -d "$LATEST_LINK" ]; then
    LINK_DEST="--link-dest=$LATEST_LINK"
else
    LINK_DEST=""
fi

# rsync 실행
rsync -av --delete \
    $LINK_DEST \
    --exclude='*.tmp' \
    --exclude='cache/' \
    "$SOURCE/" \
    "$BACKUP_PATH/"

# 최신 링크 업데이트
rm -f "$LATEST_LINK"
ln -s "$BACKUP_PATH" "$LATEST_LINK"

# 30일 이상 된 백업 삭제
find "$BACKUP_BASE" -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_PATH"
```

### SSH 키 설정 (원격 백업용)

```bash
# 백업 전용 키 생성
ssh-keygen -t ed25519 -f ~/.ssh/backup_key -N ""

# 원격 서버에 키 복사
ssh-copy-id -i ~/.ssh/backup_key.pub user@backup-server

# 자동화를 위한 config 설정
cat >> ~/.ssh/config << EOF
Host backup-server
    HostName 192.168.1.100
    User backupuser
    IdentityFile ~/.ssh/backup_key
    StrictHostKeyChecking no
EOF

# 원격 백업 실행
rsync -avz -e "ssh -i ~/.ssh/backup_key" /data/ backup-server:/backup/
```

### 대역폭 제한

```bash
# 10MB/s로 제한
rsync -av --bwlimit=10000 /source/ /backup/

# 업무 시간 외에만 빠른 속도
if [ $(date +%H) -ge 18 ] || [ $(date +%H) -lt 8 ]; then
    BWLIMIT=""
else
    BWLIMIT="--bwlimit=5000"
fi
rsync -av $BWLIMIT /source/ /backup/
```

---

## 3. Borg Backup

### Borg 소개

Borg Backup은 중복 제거, 압축, 암호화를 지원하는 백업 프로그램입니다.

```bash
# 설치
# Ubuntu/Debian
sudo apt install borgbackup

# RHEL/CentOS
sudo yum install epel-release
sudo yum install borgbackup

# pip로 설치
pip install borgbackup
```

### 저장소 초기화

```bash
# 로컬 저장소 생성
borg init --encryption=repokey /backup/borg-repo

# 원격 저장소 생성
borg init --encryption=repokey user@server:/backup/borg-repo

# 암호화 옵션
# none       - 암호화 없음
# repokey    - 저장소에 키 저장 (권장)
# keyfile    - 로컬 파일에 키 저장
# repokey-blake2 - 더 빠른 해시
```

### 백업 생성

```bash
# 기본 백업
borg create /backup/borg-repo::backup-{now} /data

# 옵션 사용
borg create \
    --verbose \
    --progress \
    --stats \
    --compression lz4 \
    --exclude '*.tmp' \
    --exclude 'cache/' \
    /backup/borg-repo::backup-{now:%Y-%m-%d_%H-%M} \
    /home \
    /etc \
    /var/www
```

### 압축 옵션

| 옵션 | 설명 | 속도 | 압축률 |
|------|------|------|--------|
| `none` | 압축 없음 | 가장 빠름 | 없음 |
| `lz4` | 빠른 압축 | 빠름 | 낮음 |
| `zstd` | 균형잡힌 압축 | 보통 | 중간 |
| `zlib` | gzip 호환 | 느림 | 높음 |
| `lzma` | 최대 압축 | 매우 느림 | 최고 |

### 백업 관리

```bash
# 백업 목록
borg list /backup/borg-repo

# 백업 상세 정보
borg info /backup/borg-repo::backup-2024-01-15

# 백업 내용 확인
borg list /backup/borg-repo::backup-2024-01-15

# 특정 경로만 확인
borg list /backup/borg-repo::backup-2024-01-15 /home/user/

# 백업 비교
borg diff /backup/borg-repo::backup-2024-01-14 backup-2024-01-15
```

### 복구

```bash
# 전체 복구
cd /restore
borg extract /backup/borg-repo::backup-2024-01-15

# 특정 파일/디렉토리 복구
borg extract /backup/borg-repo::backup-2024-01-15 home/user/documents

# 원래 경로로 복구
cd /
borg extract /backup/borg-repo::backup-2024-01-15 etc/nginx/

# 특정 시점으로 마운트 (FUSE)
mkdir /mnt/borg
borg mount /backup/borg-repo::backup-2024-01-15 /mnt/borg
# 파일 탐색 후
borg umount /mnt/borg
```

### 보존 정책 (Pruning)

```bash
# 자동 정리
borg prune \
    --keep-hourly=24 \
    --keep-daily=7 \
    --keep-weekly=4 \
    --keep-monthly=12 \
    --keep-yearly=2 \
    /backup/borg-repo

# 드라이런 (실제 삭제 안 함)
borg prune --dry-run --list \
    --keep-daily=7 \
    /backup/borg-repo
```

### Borg 백업 스크립트

```bash
#!/bin/bash
# borg-backup.sh

# 환경 변수 설정
export BORG_REPO="user@backup-server:/backup/borg-repo"
export BORG_PASSPHRASE="your-secure-passphrase"

# 로그 파일
LOG_FILE="/var/log/borg-backup.log"

# 백업 함수
backup() {
    echo "Starting backup: $(date)" >> "$LOG_FILE"

    borg create \
        --verbose \
        --filter AME \
        --list \
        --stats \
        --compression lz4 \
        --exclude-caches \
        --exclude '/home/*/.cache' \
        --exclude '/var/tmp/*' \
        --exclude '/var/cache/*' \
        ::'{hostname}-{now:%Y-%m-%d_%H:%M}' \
        /etc \
        /home \
        /var/www \
        /var/lib/mysql \
        2>> "$LOG_FILE"

    backup_exit=$?

    echo "Backup finished with exit code: $backup_exit" >> "$LOG_FILE"
}

# 정리 함수
prune() {
    echo "Starting prune: $(date)" >> "$LOG_FILE"

    borg prune \
        --list \
        --keep-hourly=24 \
        --keep-daily=7 \
        --keep-weekly=4 \
        --keep-monthly=6 \
        2>> "$LOG_FILE"

    echo "Prune finished" >> "$LOG_FILE"
}

# 저장소 정합성 검사 (주간)
check() {
    if [ $(date +%u) -eq 7 ]; then
        echo "Starting check: $(date)" >> "$LOG_FILE"
        borg check 2>> "$LOG_FILE"
        echo "Check finished" >> "$LOG_FILE"
    fi
}

# 실행
backup
prune
check

# 알림 (실패 시)
if [ $backup_exit -ne 0 ]; then
    echo "Backup failed!" | mail -s "Borg Backup Alert" admin@example.com
fi
```

---

## 4. tar/cpio 백업

### tar 백업

```bash
# 기본 압축 백업
tar -czvf backup.tar.gz /data

# 증분 백업 (snapshot 사용)
tar --create \
    --gzip \
    --listed-incremental=/backup/snapshot.snar \
    --file=/backup/backup-$(date +%Y%m%d).tar.gz \
    /data

# 복원
tar --extract \
    --gzip \
    --listed-incremental=/dev/null \
    --file=/backup/backup-20240115.tar.gz \
    -C /restore

# 제외 패턴
tar -czvf backup.tar.gz \
    --exclude='*.log' \
    --exclude='cache' \
    /data
```

### cpio 백업

```bash
# 백업 생성
find /data -print | cpio -ov > backup.cpio

# 압축 백업
find /data -print | cpio -ov | gzip > backup.cpio.gz

# 복원
cpio -iv < backup.cpio

# 압축 파일 복원
gunzip -c backup.cpio.gz | cpio -iv

# 특정 파일만 복원
cpio -iv "*.conf" < backup.cpio
```

---

## 5. 시스템 이미지 백업

### dd를 이용한 디스크 이미지

```bash
# 전체 디스크 백업
sudo dd if=/dev/sda of=/backup/disk.img bs=4M status=progress

# 압축과 함께
sudo dd if=/dev/sda bs=4M status=progress | gzip > /backup/disk.img.gz

# 복원
sudo dd if=/backup/disk.img of=/dev/sda bs=4M status=progress

# 압축된 이미지 복원
gunzip -c /backup/disk.img.gz | sudo dd of=/dev/sda bs=4M status=progress

# 파티션만 백업
sudo dd if=/dev/sda1 of=/backup/partition.img bs=4M status=progress
```

### Clonezilla

```bash
# Clonezilla Live USB 생성 후

# 디스크 이미지 생성 (명령줄)
/usr/sbin/ocs-sr -q2 -c -j2 -z1 -i 4096 -sfsck -senc -p true \
    savedisk img_name sda

# 복원
/usr/sbin/ocs-sr -g auto -e1 auto -e2 -r -j2 -c -scr -p true \
    restoredisk img_name sda
```

### LVM 스냅샷 백업

```bash
# 스냅샷 생성
sudo lvcreate -L 10G -s -n data-snap /dev/vg0/data

# 스냅샷 마운트
sudo mkdir /mnt/snapshot
sudo mount -o ro /dev/vg0/data-snap /mnt/snapshot

# 백업 수행
rsync -av /mnt/snapshot/ /backup/data/

# 정리
sudo umount /mnt/snapshot
sudo lvremove /dev/vg0/data-snap
```

---

## 6. 재해복구 전략

### DR 계획 구성요소

```
┌─────────────────────────────────────────────────────────────┐
│                    재해복구 계획                            │
├─────────────────────────────────────────────────────────────┤
│  1. 리스크 평가                                             │
│     - 잠재적 위협 식별                                      │
│     - 비즈니스 영향 분석                                    │
│                                                             │
│  2. 복구 목표 설정                                          │
│     - RTO/RPO 정의                                          │
│     - 우선순위 결정                                         │
│                                                             │
│  3. 백업 전략                                               │
│     - 백업 유형 및 주기                                     │
│     - 저장 위치 (온사이트/오프사이트)                      │
│                                                             │
│  4. 복구 절차                                               │
│     - 단계별 복구 가이드                                    │
│     - 연락처 및 역할                                        │
│                                                             │
│  5. 테스트 및 유지보수                                      │
│     - 정기적 복구 테스트                                    │
│     - 문서 업데이트                                         │
└─────────────────────────────────────────────────────────────┘
```

### 복구 체크리스트

```bash
#!/bin/bash
# disaster-recovery-checklist.sh

echo "=== 재해복구 체크리스트 ==="

# 1. 하드웨어 상태 확인
echo "[1] 하드웨어 상태 확인"
lsblk
free -h
cat /proc/cpuinfo | grep "model name" | head -1

# 2. 네트워크 연결 확인
echo "[2] 네트워크 연결"
ip addr show
ping -c 3 8.8.8.8

# 3. 백업 저장소 접근 확인
echo "[3] 백업 저장소 접근"
# 로컬 백업
ls -la /backup/
# 원격 백업
ssh backup-server "ls -la /backup/"

# 4. 백업 무결성 검증
echo "[4] 백업 무결성"
# Borg 검증
borg check /backup/borg-repo

# 5. 복구 테스트 (샘플)
echo "[5] 샘플 파일 복구 테스트"
mkdir -p /tmp/recovery-test
borg extract /backup/borg-repo::latest etc/hostname -C /tmp/recovery-test
diff /etc/hostname /tmp/recovery-test/etc/hostname

echo "=== 체크리스트 완료 ==="
```

### 베어메탈 복구 절차

```bash
# 1. 부팅 미디어로 부팅 (Ubuntu Live USB 등)

# 2. 네트워크 설정
ip addr add 192.168.1.100/24 dev eth0
ip route add default via 192.168.1.1

# 3. 디스크 파티셔닝
parted /dev/sda mklabel gpt
parted /dev/sda mkpart primary ext4 1MiB 512MiB    # /boot
parted /dev/sda mkpart primary ext4 512MiB 100%    # /
mkfs.ext4 /dev/sda1
mkfs.ext4 /dev/sda2

# 4. 마운트
mount /dev/sda2 /mnt
mkdir /mnt/boot
mount /dev/sda1 /mnt/boot

# 5. 백업에서 복원
# rsync 사용
rsync -av backup-server:/backup/latest/ /mnt/

# 또는 Borg 사용
borg extract backup-server:/backup/borg-repo::latest -C /mnt

# 6. chroot로 부트로더 설치
mount --bind /dev /mnt/dev
mount --bind /proc /mnt/proc
mount --bind /sys /mnt/sys
chroot /mnt

grub-install /dev/sda
update-grub

exit

# 7. 정리 및 재부팅
umount -R /mnt
reboot
```

---

## 7. 자동화 및 모니터링

### systemd 타이머를 이용한 자동화

```bash
# /etc/systemd/system/backup.service
[Unit]
Description=Daily Backup Service
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh
User=root
Nice=19
IOSchedulingClass=idle

[Install]
WantedBy=multi-user.target
```

```bash
# /etc/systemd/system/backup.timer
[Unit]
Description=Run backup daily

[Timer]
OnCalendar=*-*-* 02:00:00
RandomizedDelaySec=1800
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
# 타이머 활성화
sudo systemctl enable --now backup.timer

# 상태 확인
systemctl list-timers --all | grep backup
```

### 백업 모니터링 스크립트

```bash
#!/bin/bash
# backup-monitor.sh

BACKUP_DIR="/backup"
MAX_AGE_HOURS=26
ALERT_EMAIL="admin@example.com"
LOGFILE="/var/log/backup-monitor.log"

check_backup_age() {
    local newest=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "20*" | sort -r | head -1)

    if [ -z "$newest" ]; then
        echo "ERROR: No backup found" | tee -a "$LOGFILE"
        return 1
    fi

    local age_seconds=$(($(date +%s) - $(stat -c %Y "$newest")))
    local age_hours=$((age_seconds / 3600))

    if [ $age_hours -gt $MAX_AGE_HOURS ]; then
        echo "WARNING: Latest backup is $age_hours hours old" | tee -a "$LOGFILE"
        return 1
    fi

    echo "OK: Latest backup is $age_hours hours old ($newest)" | tee -a "$LOGFILE"
    return 0
}

check_backup_size() {
    local today=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "$(date +%Y-%m-%d)*" | head -1)
    local yesterday=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "$(date -d yesterday +%Y-%m-%d)*" | head -1)

    if [ -n "$today" ] && [ -n "$yesterday" ]; then
        local size_today=$(du -s "$today" | awk '{print $1}')
        local size_yesterday=$(du -s "$yesterday" | awk '{print $1}')

        # 50% 이상 차이나면 경고
        local diff=$((size_today - size_yesterday))
        local threshold=$((size_yesterday / 2))

        if [ ${diff#-} -gt $threshold ]; then
            echo "WARNING: Significant size change: $size_yesterday -> $size_today" | tee -a "$LOGFILE"
            return 1
        fi
    fi

    return 0
}

check_disk_space() {
    local usage=$(df "$BACKUP_DIR" | awk 'NR==2 {print $5}' | tr -d '%')

    if [ $usage -gt 90 ]; then
        echo "CRITICAL: Backup disk usage at ${usage}%" | tee -a "$LOGFILE"
        return 1
    elif [ $usage -gt 80 ]; then
        echo "WARNING: Backup disk usage at ${usage}%" | tee -a "$LOGFILE"
        return 1
    fi

    echo "OK: Backup disk usage at ${usage}%" | tee -a "$LOGFILE"
    return 0
}

# 메인 실행
echo "=== Backup Monitor: $(date) ===" >> "$LOGFILE"
ERRORS=0

check_backup_age || ((ERRORS++))
check_backup_size || ((ERRORS++))
check_disk_space || ((ERRORS++))

if [ $ERRORS -gt 0 ]; then
    tail -20 "$LOGFILE" | mail -s "Backup Monitor Alert" "$ALERT_EMAIL"
fi

exit $ERRORS
```

### Prometheus 메트릭 수집

```bash
#!/bin/bash
# backup-metrics.sh (node_exporter textfile collector용)

METRICS_FILE="/var/lib/node_exporter/textfile_collector/backup.prom"
BACKUP_DIR="/backup"

# 최신 백업 시간
newest=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "20*" | sort -r | head -1)
if [ -n "$newest" ]; then
    backup_timestamp=$(stat -c %Y "$newest")
    echo "backup_last_success_timestamp $backup_timestamp" > "$METRICS_FILE"
fi

# 백업 크기
backup_size=$(du -sb "$newest" 2>/dev/null | awk '{print $1}')
echo "backup_size_bytes $backup_size" >> "$METRICS_FILE"

# 디스크 사용량
disk_usage=$(df "$BACKUP_DIR" | awk 'NR==2 {print $3}')
disk_total=$(df "$BACKUP_DIR" | awk 'NR==2 {print $2}')
echo "backup_disk_used_bytes $((disk_usage * 1024))" >> "$METRICS_FILE"
echo "backup_disk_total_bytes $((disk_total * 1024))" >> "$METRICS_FILE"
```

---

## 연습 문제

### 문제 1: rsync 증분 백업

하드링크를 이용한 rsync 증분 백업 스크립트를 작성하세요:
- 원본: `/home/user`
- 백업 위치: `/backup/home`
- 일일 백업, 최신 백업을 `latest` 심볼릭 링크로 연결
- 30일 이상 된 백업 자동 삭제

### 문제 2: Borg 복구

Borg 저장소에서 특정 날짜의 백업 중 `/etc/nginx/` 디렉토리만 복구하는 명령을 작성하세요.

### 문제 3: DR 테스트

분기별 재해복구 테스트 절차를 작성하세요. 포함해야 할 항목:
- 백업 무결성 검증
- 샘플 데이터 복구 테스트
- 전체 시스템 복구 테스트 (가능하면)

---

## 정답

### 문제 1 정답

```bash
#!/bin/bash

SOURCE="/home/user"
BACKUP_BASE="/backup/home"
DATE=$(date +%Y-%m-%d)
BACKUP_PATH="$BACKUP_BASE/$DATE"
LATEST="$BACKUP_BASE/latest"

# 하드링크 옵션
if [ -d "$LATEST" ]; then
    LINK="--link-dest=$LATEST"
else
    LINK=""
fi

# 백업 실행
rsync -av --delete $LINK "$SOURCE/" "$BACKUP_PATH/"

# 최신 링크 업데이트
rm -f "$LATEST"
ln -s "$BACKUP_PATH" "$LATEST"

# 30일 이상 된 백업 삭제
find "$BACKUP_BASE" -maxdepth 1 -type d -name "20*" -mtime +30 -exec rm -rf {} \;
```

### 문제 2 정답

```bash
# 백업 목록 확인
borg list /backup/borg-repo

# 특정 날짜 백업에서 nginx 설정 복구
borg extract /backup/borg-repo::backup-2024-01-15 etc/nginx

# 다른 경로로 복구
mkdir /tmp/restore
cd /tmp/restore
borg extract /backup/borg-repo::backup-2024-01-15 etc/nginx
```

### 문제 3 정답

```markdown
# 분기별 DR 테스트 절차

## 1. 백업 무결성 검증 (매 분기)
- [ ] Borg check 실행: `borg check /backup/borg-repo`
- [ ] 백업 목록 검토: `borg list /backup/borg-repo`
- [ ] 최근 백업 상세 확인: `borg info /backup/borg-repo::latest`

## 2. 샘플 데이터 복구 테스트 (매 분기)
- [ ] 테스트 디렉토리 생성
- [ ] 설정 파일 복구 테스트 (/etc/)
- [ ] 데이터 파일 복구 테스트 (/var/www/)
- [ ] 복구된 파일 무결성 확인 (diff 또는 checksum)

## 3. 전체 시스템 복구 테스트 (반기)
- [ ] 테스트 VM 또는 물리 서버 준비
- [ ] 베어메탈 복구 절차 실행
- [ ] 부팅 확인
- [ ] 서비스 정상 동작 확인
- [ ] 데이터 무결성 검증

## 4. 문서화 및 개선
- [ ] 테스트 결과 문서화
- [ ] 발견된 문제점 기록
- [ ] 절차 개선 사항 반영
- [ ] RTO/RPO 달성 여부 확인
```

---

## 다음 단계

- [커널 관리](./20_Kernel_Management.md) - 커널 컴파일, 모듈, GRUB 설정

---

## 참고 자료

- [rsync Manual](https://rsync.samba.org/documentation.html)
- [Borg Backup Documentation](https://borgbackup.readthedocs.io/)
- [Clonezilla](https://clonezilla.org/)
- `man rsync`, `man tar`, `man dd`, `man borg`
