# 16. 저장소 관리

**이전**: [컨테이너 내부 구조](./15_Container_Internals.md) | **다음**: [SELinux/AppArmor](./17_SELinux_AppArmor.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. LVM 물리 볼륨(Physical Volume), 볼륨 그룹(Volume Group), 논리 볼륨(Logical Volume)을 생성하고 관리한다
2. LVM 볼륨을 확장·축소하고, LVM 스냅샷(snapshot)을 사용하여 일관된 백업을 수행한다
3. RAID 레벨(0, 1, 5, 6, 10)을 비교하고 mdadm으로 소프트웨어 RAID를 구성한다
4. 워크로드 요구사항에 따라 적절한 파일시스템(ext4, XFS, Btrfs)을 선택한다
5. LUKS로 디스크를 암호화하고 부팅 시 자동 잠금 해제를 구성한다
6. LUKS 암호화와 LVM을 결합하여 안전하고 유연한 스토리지를 구성한다
7. SMART로 디스크 건강 상태를 모니터링하고 정기적인 파일시스템 유지보수를 수행한다

## 목차
1. [스토리지 기초](#1-스토리지-기초)
2. [LVM](#2-lvm)
3. [RAID](#3-raid)
4. [파일시스템](#4-파일시스템)
5. [디스크 암호화](#5-디스크-암호화)
6. [모니터링과 유지보수](#6-모니터링과-유지보수)
7. [연습 문제](#7-연습-문제)

---

데이터는 모든 서버에서 가장 소중한 자산이며, 스토리지 계층은 그것을 보호하는 핵심입니다. 잘못 구성된 RAID 어레이, 암호화되지 않은 디스크, 가득 찬 파티션은 데이터 손실, 보안 침해, 또는 장기 장애로 이어질 수 있습니다. 이 레슨에서는 유연성을 위한 LVM(Logical Volume Manager), 이중화를 위한 RAID, 안정성을 위한 최신 파일시스템, 그리고 암호화를 위한 LUKS 등 프로덕션 수준의 스토리지 인프라를 구축하기 위해 모든 Linux 관리자가 반드시 익혀야 할 핵심 스토리지 기술을 다룹니다.

## 1. 스토리지 기초

### 1.1 스토리지 계층

```
┌─────────────────────────────────────────────────────────────┐
│                    스토리지 계층 구조                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  애플리케이션                                               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────┐               │
│  │           VFS (Virtual File System)     │               │
│  └─────────────────────────────────────────┘               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────┐               │
│  │    파일시스템 (ext4, XFS, Btrfs)        │               │
│  └─────────────────────────────────────────┘               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────┐               │
│  │    블록 장치 (LVM, RAID, LUKS)          │               │
│  └─────────────────────────────────────────┘               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────┐               │
│  │    디스크 드라이버                       │               │
│  └─────────────────────────────────────────┘               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────┐               │
│  │    물리 디스크 (HDD, SSD, NVMe)         │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 디스크 정보 확인

```bash
# 블록 장치 목록
lsblk
lsblk -f  # 파일시스템 포함

# 디스크 상세 정보
fdisk -l
parted -l

# SMART 정보 (디스크 건강)
smartctl -a /dev/sda

# 디스크 UUID
blkid

# 디스크 사용량
df -h
df -i  # inode 사용량
```

### 1.3 파티션 관리

```bash
# fdisk (MBR)
fdisk /dev/sdb
# n - 새 파티션
# d - 파티션 삭제
# p - 파티션 테이블 출력
# w - 저장 후 종료

# parted (GPT 권장)
parted /dev/sdb
(parted) mklabel gpt
(parted) mkpart primary ext4 0% 50%
(parted) mkpart primary ext4 50% 100%
(parted) print
(parted) quit

# gdisk (GPT)
gdisk /dev/sdb
```

---

## 2. LVM

### 2.1 LVM 개념

```
┌─────────────────────────────────────────────────────────────┐
│                       LVM 구조                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Logical Volume (LV)                                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │   lv1   │ │   lv2   │ │   lv3   │  ← 파일시스템 생성    │
│  │  (root) │ │  (home) │ │  (data) │                       │
│  └─────────┴─────────┴─────────┘                           │
│           │                                                 │
│           ▼                                                 │
│  Volume Group (VG)                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                       vg0                            │   │
│  │   ← 여러 PV를 하나로 묶음                            │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                 │
│           ▼                                                 │
│  Physical Volume (PV)                                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │  /dev/  │ │  /dev/  │ │  /dev/  │  ← 실제 디스크/파티션 │
│  │  sda1   │ │  sdb1   │ │  sdc1   │                       │
│  └─────────┘ └─────────┘ └─────────┘                       │
│                                                             │
│  장점:                                                      │
│  • 동적 볼륨 크기 조절                                      │
│  • 여러 디스크를 하나로 통합                                │
│  • 스냅샷 지원                                              │
│  • 온라인 확장 가능                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 LVM 생성

```bash
# 1. Physical Volume (PV) 생성
pvcreate /dev/sdb1
pvcreate /dev/sdc1

# PV 확인
pvs
pvdisplay /dev/sdb1

# 2. Volume Group (VG) 생성
vgcreate vg_data /dev/sdb1 /dev/sdc1

# VG 확인
vgs
vgdisplay vg_data

# 3. Logical Volume (LV) 생성
lvcreate -L 10G -n lv_home vg_data
lvcreate -l 100%FREE -n lv_data vg_data  # 남은 공간 전부

# LV 확인
lvs
lvdisplay /dev/vg_data/lv_home

# 4. 파일시스템 생성 및 마운트
mkfs.ext4 /dev/vg_data/lv_home
mkdir /mnt/home
mount /dev/vg_data/lv_home /mnt/home

# /etc/fstab 추가
echo '/dev/vg_data/lv_home /home ext4 defaults 0 2' >> /etc/fstab
```

### 2.3 LVM 확장

```bash
# VG에 새 디스크 추가
pvcreate /dev/sdd1
vgextend vg_data /dev/sdd1

# LV 확장
lvextend -L +5G /dev/vg_data/lv_home
# 또는 전체 여유 공간 사용
lvextend -l +100%FREE /dev/vg_data/lv_home

# 파일시스템 확장
# ext4
resize2fs /dev/vg_data/lv_home

# XFS (확장만 가능)
xfs_growfs /mnt/home

# 한 번에 LV + 파일시스템 확장
lvextend -r -L +5G /dev/vg_data/lv_home
```

### 2.4 LVM 축소

```bash
# ⚠️ 주의: 데이터 백업 필수!

# ext4 축소 (언마운트 필요)
umount /mnt/home
e2fsck -f /dev/vg_data/lv_home
resize2fs /dev/vg_data/lv_home 8G
lvreduce -L 8G /dev/vg_data/lv_home
mount /dev/vg_data/lv_home /mnt/home

# XFS는 축소 불가!
```

### 2.5 LVM 스냅샷

```bash
# 스냅샷 생성
lvcreate -L 1G -s -n snap_home /dev/vg_data/lv_home

# 스냅샷 확인
lvs
lvdisplay /dev/vg_data/snap_home

# 스냅샷 마운트 (복구 전 확인)
mount -o ro /dev/vg_data/snap_home /mnt/snapshot

# 스냅샷에서 복원
lvconvert --merge /dev/vg_data/snap_home
# 재부팅 필요할 수 있음

# 스냅샷 삭제
lvremove /dev/vg_data/snap_home
```

---

## 3. RAID

### 3.1 RAID 레벨

```
┌─────────────────────────────────────────────────────────────┐
│                      RAID 레벨 비교                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RAID 0 (Striping)                                          │
│  ┌─────┬─────┬─────┬─────┐                                 │
│  │ A1  │ A2  │ A3  │ A4  │  장점: 최고 성능               │
│  │ B1  │ B2  │ B3  │ B4  │  단점: 장애 허용 없음          │
│  └─────┴─────┴─────┴─────┘  용량: 100%                    │
│  Disk1 Disk2 Disk3 Disk4                                   │
│                                                             │
│  RAID 1 (Mirroring)                                         │
│  ┌─────┬─────┐                                              │
│  │ A1  │ A1  │  장점: 완벽한 복제                          │
│  │ B1  │ B1  │  단점: 용량 50%                             │
│  └─────┴─────┘  용량: 50%                                  │
│  Disk1 Disk2                                                │
│                                                             │
│  RAID 5 (Striping + Parity)                                 │
│  ┌─────┬─────┬─────┐                                        │
│  │ A1  │ A2  │ Ap  │  장점: 성능 + 장애 허용(1개)         │
│  │ B1  │ Bp  │ B2  │  단점: 쓰기 시 패리티 계산           │
│  │ Cp  │ C1  │ C2  │  용량: (n-1)/n                        │
│  └─────┴─────┴─────┘  최소 3개 디스크                      │
│  Disk1 Disk2 Disk3                                          │
│                                                             │
│  RAID 6 (Double Parity)                                     │
│  ┌─────┬─────┬─────┬─────┐                                 │
│  │ A1  │ A2  │ Ap  │ Aq  │  장점: 2개 디스크 장애 허용    │
│  │ B1  │ Bp  │ Bq  │ B2  │  단점: 더 느린 쓰기            │
│  └─────┴─────┴─────┴─────┘  용량: (n-2)/n                  │
│                                                             │
│  RAID 10 (1+0, Striped Mirrors)                            │
│  ┌─────┬─────┐ ┌─────┬─────┐                               │
│  │ A1  │ A1  │ │ A2  │ A2  │  장점: 성능 + 신뢰성         │
│  │ B1  │ B1  │ │ B2  │ B2  │  단점: 용량 50%              │
│  └─────┴─────┘ └─────┴─────┘  최소 4개 디스크             │
│  Mirror 1      Mirror 2                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 mdadm으로 RAID 구성

```bash
# mdadm 설치
apt install mdadm

# RAID 1 생성
mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sdb1 /dev/sdc1

# RAID 5 생성
mdadm --create /dev/md1 --level=5 --raid-devices=3 /dev/sdd1 /dev/sde1 /dev/sdf1

# RAID 10 생성
mdadm --create /dev/md2 --level=10 --raid-devices=4 /dev/sdg1 /dev/sdh1 /dev/sdi1 /dev/sdj1

# RAID 상태 확인
cat /proc/mdstat
mdadm --detail /dev/md0

# 설정 저장
mdadm --detail --scan >> /etc/mdadm/mdadm.conf
update-initramfs -u

# 파일시스템 생성
mkfs.ext4 /dev/md0
mount /dev/md0 /mnt/raid
```

### 3.3 RAID 관리

```bash
# 디스크 장애 시뮬레이션
mdadm --manage /dev/md0 --fail /dev/sdb1

# 장애 디스크 제거
mdadm --manage /dev/md0 --remove /dev/sdb1

# 새 디스크 추가
mdadm --manage /dev/md0 --add /dev/sdk1

# 스페어 디스크 추가
mdadm --manage /dev/md0 --add-spare /dev/sdl1

# 재구성 상태 확인
cat /proc/mdstat
watch -n 1 cat /proc/mdstat

# RAID 중지/시작
mdadm --stop /dev/md0
mdadm --assemble /dev/md0 /dev/sdb1 /dev/sdc1
```

### 3.4 RAID 확장

```bash
# RAID 5/6에 디스크 추가
mdadm --grow /dev/md1 --raid-devices=4 --add /dev/sdm1

# 크기 확장 후 파일시스템 확장
mdadm --grow /dev/md1 --size=max
resize2fs /dev/md1
```

---

## 4. 파일시스템

### 4.1 파일시스템 비교

```
┌────────────────────────────────────────────────────────────────┐
│                    파일시스템 비교                              │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│              │     ext4     │     XFS      │     Btrfs        │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│ 최대 볼륨    │    1 EB      │    8 EB      │    16 EB         │
│ 최대 파일    │   16 TB      │    8 EB      │    16 EB         │
│ 저널링       │     예       │     예       │     CoW          │
│ 온라인 확장  │     예       │     예       │     예           │
│ 온라인 축소  │     아니오   │    아니오    │     예           │
│ 스냅샷       │   (LVM)      │   (LVM)      │     예 (내장)    │
│ 압축         │     아니오   │    아니오    │     예           │
│ 체크섬       │     메타     │   아니오     │     예 (전체)    │
│ 적합한 용도  │  범용/부팅   │  대용량/DB   │  NAS/스냅샷      │
└──────────────┴──────────────┴──────────────┴──────────────────┘
```

### 4.2 ext4 관리

```bash
# 생성
mkfs.ext4 /dev/sdb1
mkfs.ext4 -L "DATA" /dev/sdb1  # 레이블 지정

# 옵션으로 생성
mkfs.ext4 -b 4096 -i 16384 -O ^has_journal /dev/sdb1

# 정보 확인
tune2fs -l /dev/sdb1

# 설정 변경
tune2fs -L "NEW_LABEL" /dev/sdb1  # 레이블
tune2fs -c 30 /dev/sdb1           # 마운트 횟수 후 fsck
tune2fs -i 1m /dev/sdb1           # 1달 후 fsck
tune2fs -O ^has_journal /dev/sdb1 # 저널 비활성화

# 파일시스템 검사
e2fsck -f /dev/sdb1
e2fsck -p /dev/sdb1  # 자동 수정

# 조각 모음 (온라인)
e4defrag /dev/sdb1
```

### 4.3 XFS 관리

```bash
# 생성
mkfs.xfs /dev/sdb1
mkfs.xfs -L "DATA" -f /dev/sdb1

# 정보 확인
xfs_info /dev/sdb1
xfs_info /mnt/data  # 마운트된 경우

# 레이블 변경
xfs_admin -L "NEW_LABEL" /dev/sdb1

# 파일시스템 확장 (온라인)
xfs_growfs /mnt/data

# 파일시스템 검사 (마운트 해제)
xfs_repair /dev/sdb1
xfs_repair -n /dev/sdb1  # 검사만

# 조각 모음 (온라인)
xfs_fsr /mnt/data
```

### 4.4 Btrfs 관리

```bash
# 생성
mkfs.btrfs /dev/sdb1
mkfs.btrfs -L "DATA" -m raid1 -d raid1 /dev/sdb1 /dev/sdc1

# 정보 확인
btrfs filesystem show
btrfs filesystem df /mnt/btrfs

# 서브볼륨 생성
btrfs subvolume create /mnt/btrfs/subvol1

# 스냅샷
btrfs subvolume snapshot /mnt/btrfs/subvol1 /mnt/btrfs/snap1
btrfs subvolume snapshot -r /mnt/btrfs/subvol1 /mnt/btrfs/snap_ro  # 읽기 전용

# 서브볼륨 목록
btrfs subvolume list /mnt/btrfs

# 스냅샷 삭제
btrfs subvolume delete /mnt/btrfs/snap1

# 압축 활성화
mount -o compress=zstd /dev/sdb1 /mnt/btrfs

# 디스크 추가
btrfs device add /dev/sdd1 /mnt/btrfs
btrfs balance start /mnt/btrfs

# 스크럽 (데이터 무결성 검사)
btrfs scrub start /mnt/btrfs
btrfs scrub status /mnt/btrfs
```

---

## 5. 디스크 암호화

### 5.1 LUKS 암호화

```bash
# LUKS 볼륨 생성
cryptsetup luksFormat /dev/sdb1
# 암호 입력

# LUKS 열기
cryptsetup open /dev/sdb1 encrypted_disk
# /dev/mapper/encrypted_disk 생성됨

# 파일시스템 생성
mkfs.ext4 /dev/mapper/encrypted_disk

# 마운트
mount /dev/mapper/encrypted_disk /mnt/encrypted

# 언마운트 및 닫기
umount /mnt/encrypted
cryptsetup close encrypted_disk
```

### 5.2 LUKS 관리

```bash
# LUKS 정보
cryptsetup luksDump /dev/sdb1

# 키 추가 (최대 8개)
cryptsetup luksAddKey /dev/sdb1

# 키 제거
cryptsetup luksRemoveKey /dev/sdb1

# 키 파일 사용
dd if=/dev/urandom of=/root/keyfile bs=1024 count=4
chmod 400 /root/keyfile
cryptsetup luksAddKey /dev/sdb1 /root/keyfile

# 키 파일로 열기
cryptsetup open /dev/sdb1 encrypted_disk --key-file /root/keyfile
```

### 5.3 부팅 시 자동 마운트

```bash
# /etc/crypttab
# <name> <device> <key file> <options>
encrypted_disk /dev/sdb1 /root/keyfile luks

# /etc/fstab
/dev/mapper/encrypted_disk /mnt/encrypted ext4 defaults 0 2
```

### 5.4 LUKS + LVM

```bash
# 암호화된 PV 생성
cryptsetup luksFormat /dev/sdb1
cryptsetup open /dev/sdb1 crypt_pv

# LVM 구성
pvcreate /dev/mapper/crypt_pv
vgcreate vg_encrypted /dev/mapper/crypt_pv
lvcreate -l 100%FREE -n lv_data vg_encrypted

# 파일시스템
mkfs.ext4 /dev/vg_encrypted/lv_data
```

---

## 6. 모니터링과 유지보수

### 6.1 디스크 상태 모니터링

```bash
# SMART 모니터링
smartctl -H /dev/sda              # 건강 상태
smartctl -a /dev/sda              # 전체 정보
smartctl -t short /dev/sda        # 짧은 테스트
smartctl -l selftest /dev/sda     # 테스트 결과

# smartd 데몬 설정
# /etc/smartd.conf
/dev/sda -a -o on -S on -s (S/../.././02|L/../../6/03)
#                             매일 2시 단축테스트 | 토요일 3시 긴 테스트

# I/O 통계
iostat -x 1

# 디스크 대기열
cat /sys/block/sda/queue/nr_requests
```

### 6.2 파일시스템 유지보수

```bash
# 정기 점검 (ext4)
tune2fs -c 30 /dev/sda1     # 30번 마운트 후 fsck
tune2fs -i 1m /dev/sda1     # 1개월마다 fsck

# 예약 블록 설정 (ext4)
tune2fs -m 1 /dev/sda1      # 1%로 줄임 (기본 5%)

# TRIM (SSD)
fstrim -v /                  # 수동 TRIM
systemctl enable fstrim.timer  # 주기적 TRIM

# 디스크 사용량 분석
du -sh /*
ncdu /
```

### 6.3 백업 전략

```bash
# dd로 전체 디스크 백업
dd if=/dev/sda of=/backup/sda.img bs=64M status=progress

# 압축 백업
dd if=/dev/sda bs=64M | gzip > /backup/sda.img.gz

# 파티션 복제
dd if=/dev/sda1 of=/dev/sdb1 bs=64M status=progress

# rsync로 파일 백업
rsync -avz --delete /data/ /backup/data/

# LVM 스냅샷으로 일관된 백업
lvcreate -L 1G -s -n snap_backup /dev/vg/lv_data
mount -o ro /dev/vg/snap_backup /mnt/snap
rsync -avz /mnt/snap/ /backup/
umount /mnt/snap
lvremove -f /dev/vg/snap_backup
```

---

## 7. 연습 문제

### 연습 1: LVM 구성
```bash
# 요구사항:
# 1. 2개 디스크로 VG 생성
# 2. 3개 LV 생성 (root 20G, home 50G, data 나머지)
# 3. 각각 ext4, xfs, ext4로 포맷
# 4. fstab에 영구 마운트 설정

# 명령어 작성:
```

### 연습 2: RAID 5 구성
```bash
# 요구사항:
# 1. 4개 디스크로 RAID 5 구성
# 2. 1개는 스페어로 설정
# 3. 장애 시뮬레이션 및 복구

# 명령어 작성:
```

### 연습 3: 암호화된 LVM
```bash
# 요구사항:
# 1. LUKS로 디스크 암호화
# 2. 암호화된 볼륨 위에 LVM 구성
# 3. 부팅 시 자동 마운트 (키 파일 사용)

# 설정 파일 및 명령어:
```

### 연습 4: Btrfs 스냅샷 관리
```bash
# 요구사항:
# 1. Btrfs 볼륨 생성
# 2. 서브볼륨 구조 설계
# 3. 스냅샷 정책 수립 (매일, 매주)
# 4. 스냅샷에서 복원 테스트

# 스크립트 작성:
```

---

## 다음 단계

- [13_systemd_심화](13_systemd_심화.md) - systemd 복습
- [14_성능_튜닝](14_성능_튜닝.md) - I/O 튜닝
- [15_컨테이너_내부_구조](15_컨테이너_내부_구조.md) - 컨테이너 볼륨

## 참고 자료

- [LVM Administrator's Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/configuring_and_managing_logical_volumes/)
- [Linux RAID Wiki](https://raid.wiki.kernel.org/)
- [Btrfs Wiki](https://btrfs.wiki.kernel.org/)
- [LUKS Documentation](https://gitlab.com/cryptsetup/cryptsetup)

---

[← 이전: 컨테이너 내부 구조](15_컨테이너_내부_구조.md) | [목차](00_Overview.md)
