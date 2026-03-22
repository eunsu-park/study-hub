# 파일시스템 탐색

**이전**: [리눅스 기초](./01_Linux_Basics.md) | **다음**: [파일/디렉토리 관리](./03_File_Directory_Management.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 리눅스 파일시스템 계층 표준(Filesystem Hierarchy Standard, FHS)과 주요 디렉토리의 목적을 설명한다
2. 절대 경로(absolute path)와 상대 경로(relative path)를 구분한다
3. cd, pwd, 특수 디렉토리 기호(~, .., -, .)를 사용해 파일시스템을 탐색한다
4. ls로 디렉토리 내용을 나열하고 긴 형식(long-format) 출력을 해석한다
5. find, locate, which, whereis를 사용해 파일을 검색한다
6. file과 stat 명령어로 파일 메타데이터를 확인한다
7. 와일드카드 패턴(globbing)을 사용해 파일을 효율적으로 매칭한다

---

리눅스에서 모든 파일과 디렉토리는 `/`를 루트로 하는 단일 통합 트리 안에 존재합니다. 별도의 드라이브 문자(C:, D:)를 사용하는 Windows와 달리, 리눅스는 장치, 설정, 사용자 데이터를 모두 하나의 계층 구조 아래 구성합니다. 어디에 무엇이 있는지 알고 어떻게 이동하는지 아는 것이 모든 리눅스 시스템에서 생산적으로 작업하는 첫걸음입니다.

## 1. 리눅스 디렉토리 구조

리눅스는 FHS(Filesystem Hierarchy Standard)를 따르는 트리 구조입니다.

```
/                          ← 루트 (최상위)
├── bin/                   ← 필수 실행 파일 (ls, cp, mv 등)
├── boot/                  ← 부팅 관련 파일 (커널, 부트로더)
├── dev/                   ← 장치 파일 (디스크, USB 등)
├── etc/                   ← 시스템 설정 파일
├── home/                  ← 사용자 홈 디렉토리
│   ├── user1/
│   └── user2/
├── lib/                   ← 공유 라이브러리
├── media/                 ← 이동식 미디어 마운트
├── mnt/                   ← 임시 마운트 포인트
├── opt/                   ← 추가 소프트웨어 패키지
├── proc/                  ← 프로세스 정보 (가상 파일시스템)
├── root/                  ← root 사용자 홈
├── run/                   ← 런타임 데이터
├── sbin/                  ← 시스템 관리 명령어
├── srv/                   ← 서비스 데이터
├── sys/                   ← 커널/장치 정보 (가상)
├── tmp/                   ← 임시 파일 (재부팅 시 삭제)
├── usr/                   ← 사용자 프로그램
│   ├── bin/              ← 사용자 명령어
│   ├── lib/              ← 라이브러리
│   ├── local/            ← 로컬 설치 프로그램
│   └── share/            ← 공유 데이터
└── var/                   ← 가변 데이터
    ├── log/              ← 로그 파일
    ├── cache/            ← 캐시
    └── www/              ← 웹 서버 파일
```

---

## 2. 주요 디렉토리 설명

| 디렉토리 | 설명 | 예시 |
|----------|------|------|
| `/` | 루트, 모든 디렉토리의 시작점 | - |
| `/home` | 일반 사용자 홈 디렉토리 | `/home/ubuntu` |
| `/root` | root 사용자 홈 | - |
| `/etc` | 시스템 설정 파일 | `/etc/passwd`, `/etc/hosts` |
| `/var` | 로그, 캐시 등 가변 데이터 | `/var/log/syslog` |
| `/tmp` | 임시 파일 (모든 사용자 쓰기 가능) | - |
| `/usr` | 사용자 프로그램, 라이브러리 | `/usr/bin/python3` |
| `/opt` | 서드파티 소프트웨어 | `/opt/google/chrome` |
| `/bin`, `/sbin` | 시스템 필수 명령어 | `/bin/ls`, `/sbin/reboot` |
| `/dev` | 장치 파일 | `/dev/sda`, `/dev/null` |
| `/proc` | 프로세스/커널 정보 (가상) | `/proc/cpuinfo` |

---

## 3. 경로의 이해

### 절대경로 (Absolute Path)

루트(`/`)부터 시작하는 전체 경로입니다.

```bash
# 절대경로 예시
/home/ubuntu/documents/file.txt
/etc/nginx/nginx.conf
/var/log/syslog
```

### 상대경로 (Relative Path)

현재 위치를 기준으로 한 경로입니다.

```bash
# 현재 위치가 /home/ubuntu 일 때
documents/file.txt      # → /home/ubuntu/documents/file.txt
./documents/file.txt    # → 같은 의미 (현재 디렉토리)
../shared/data.txt      # → /home/shared/data.txt
```

### 특수 디렉토리

| 기호 | 의미 | 예시 |
|------|------|------|
| `.` | 현재 디렉토리 | `./script.sh` |
| `..` | 상위 디렉토리 | `cd ..` |
| `~` | 홈 디렉토리 | `cd ~` = `cd /home/사용자` |
| `-` | 이전 디렉토리 | `cd -` |
| `/` | 루트 디렉토리 | `cd /` |

```bash
# 특수 디렉토리 활용
cd ~              # 홈 디렉토리로
cd ~/documents    # 홈/documents로
cd ..             # 상위 디렉토리로
cd ../..          # 2단계 상위로
cd -              # 직전 디렉토리로
```

---

## 4. pwd - 현재 위치 확인

```bash
# 현재 작업 디렉토리 출력
pwd
```

출력:
```
/home/ubuntu/projects
```

---

## 5. cd - 디렉토리 이동

### 기본 사용법

```bash
# 절대경로로 이동
cd /var/log

# 상대경로로 이동
cd documents

# 홈 디렉토리로
cd
cd ~

# 상위 디렉토리로
cd ..

# 이전 디렉토리로
cd -
```

### 활용 예시

```bash
# 현재 위치 확인
pwd                    # /home/ubuntu

# documents로 이동
cd documents
pwd                    # /home/ubuntu/documents

# 상위로 이동
cd ..
pwd                    # /home/ubuntu

# 이전 디렉토리로
cd -
pwd                    # /home/ubuntu/documents
```

---

## 6. ls - 디렉토리 내용 보기

### 기본 사용법

```bash
# 현재 디렉토리
ls

# 특정 디렉토리
ls /var/log

# 여러 디렉토리
ls /home /tmp
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-l` | 상세 정보 (long format) |
| `-a` | 숨김 파일 포함 (all) |
| `-h` | 용량을 읽기 쉽게 (human-readable) |
| `-R` | 하위 디렉토리까지 재귀적 |
| `-t` | 수정 시간순 정렬 |
| `-S` | 파일 크기순 정렬 |
| `-r` | 역순 정렬 |
| `-d` | 디렉토리 자체 정보 |

### 옵션 조합

```bash
# 상세 + 숨김 파일
ls -la

# 상세 + 용량 읽기 쉽게
ls -lh

# 최신 파일 먼저
ls -lt

# 큰 파일 먼저
ls -lS

# 자주 쓰는 조합
ls -lah
```

### ls -l 출력 해석

```
-rw-r--r-- 1 ubuntu ubuntu 4096 Jan 23 14:30 file.txt
│├──┬───┤ │ │      │      │    │            │
││  │   │ │ │      │      │    │            └── 파일명
││  │   │ │ │      │      │    └── 수정 시간
││  │   │ │ │      │      └── 파일 크기 (바이트)
││  │   │ │ │      └── 그룹
││  │   │ │ └── 소유자
││  │   │ └── 하드링크 수
││  │   └── 기타 권한 (r--)
││  └── 그룹 권한 (r--)
│└── 소유자 권한 (rw-)
└── 파일 타입 (- 파일, d 디렉토리)
```

### 파일 타입 표시

| 문자 | 타입 |
|------|------|
| `-` | 일반 파일 |
| `d` | 디렉토리 |
| `l` | 심볼릭 링크 |
| `c` | 문자 장치 |
| `b` | 블록 장치 |
| `s` | 소켓 |
| `p` | 파이프 |

---

## 7. 파일 찾기

### find - 파일 검색

```bash
# 기본 문법
find [경로] [조건] [동작]

# 이름으로 찾기
find /home -name "*.txt"

# 대소문자 무시
find /home -iname "readme*"

# 타입 지정 (f: 파일, d: 디렉토리)
find /var -type f -name "*.log"
find /home -type d -name "config"

# 크기로 찾기
find / -size +100M          # 100MB 초과
find / -size -1k            # 1KB 미만

# 수정 시간으로 찾기
find /var/log -mtime -7     # 7일 이내 수정
find /tmp -mtime +30        # 30일 이전

# 권한으로 찾기
find / -perm 777

# 소유자로 찾기
find /home -user ubuntu
```

### find와 동작 결합

```bash
# 찾은 파일 삭제
find /tmp -name "*.tmp" -delete

# 찾은 파일에 명령 실행
find /home -name "*.sh" -exec chmod +x {} \;

# 찾은 파일 목록 출력
find /var/log -name "*.log" -print
```

### locate - 빠른 검색

데이터베이스를 이용한 빠른 검색입니다.

```bash
# 파일 검색
locate nginx.conf

# 대소문자 무시
locate -i readme

# 데이터베이스 업데이트 (관리자)
sudo updatedb
```

### which - 명령어 위치

```bash
# 명령어 실행 파일 위치
which python3
```

출력:
```
/usr/bin/python3
```

### whereis - 명령어 관련 파일

```bash
# 실행 파일, 소스, 매뉴얼 위치
whereis nginx
```

출력:
```
nginx: /usr/sbin/nginx /usr/lib/nginx /etc/nginx /usr/share/nginx /usr/share/man/man8/nginx.8.gz
```

---

## 8. 파일 내용 미리보기

### file - 파일 타입 확인

```bash
file document.pdf
file script.sh
file image.jpg
```

출력:
```
document.pdf: PDF document, version 1.4
script.sh: Bourne-Again shell script, ASCII text executable
image.jpg: JPEG image data, JFIF standard 1.01
```

### stat - 파일 상세 정보

```bash
stat file.txt
```

출력:
```
  File: file.txt
  Size: 1234            Blocks: 8          IO Block: 4096   regular file
Device: 801h/2049d      Inode: 123456      Links: 1
Access: (0644/-rw-r--r--)  Uid: ( 1000/  ubuntu)   Gid: ( 1000/  ubuntu)
Access: 2024-01-23 10:00:00.000000000 +0900
Modify: 2024-01-23 09:30:00.000000000 +0900
Change: 2024-01-23 09:30:00.000000000 +0900
 Birth: 2024-01-20 15:00:00.000000000 +0900
```

---

## 9. 와일드카드 (Globbing)

| 패턴 | 설명 | 예시 |
|------|------|------|
| `*` | 0개 이상의 문자 | `*.txt`, `log*` |
| `?` | 정확히 1개 문자 | `file?.txt` |
| `[abc]` | a, b, c 중 하나 | `file[123].txt` |
| `[a-z]` | a~z 범위 중 하나 | `file[a-z].txt` |
| `[!abc]` | a, b, c 제외 | `file[!0-9].txt` |

```bash
# 모든 txt 파일
ls *.txt

# log로 시작하는 파일
ls log*

# 한 글자 숫자 파일
ls file?.txt

# 숫자로 끝나는 파일
ls file[0-9].txt

# a-c로 시작하는 파일
ls [a-c]*.txt
```

---

## 10. 실습 예제

### 실습 1: 디렉토리 탐색

```bash
# 1. 현재 위치 확인
pwd

# 2. 루트로 이동
cd /

# 3. 디렉토리 구조 확인
ls -l

# 4. /var/log로 이동
cd /var/log

# 5. 로그 파일 확인
ls -lh

# 6. 홈으로 돌아가기
cd ~
```

### 실습 2: 상세 정보 확인

```bash
# 숨김 파일 포함 전체 목록
ls -la ~

# 최근 수정된 파일 확인
ls -lt /var/log | head -10

# 큰 파일 찾기
ls -lhS /var/log | head -5
```

### 실습 3: 파일 찾기

```bash
# 홈에서 .conf 파일 찾기
find ~ -name "*.conf" 2>/dev/null

# /etc에서 nginx 관련 파일 찾기
find /etc -name "*nginx*" 2>/dev/null

# 100MB 이상 파일 찾기
find / -size +100M 2>/dev/null | head -10

# 7일 이내 수정된 로그 찾기
find /var/log -mtime -7 -name "*.log"
```

### 실습 4: 시스템 디렉토리 탐색

```bash
# CPU 정보 확인
cat /proc/cpuinfo | head -20

# 메모리 정보
cat /proc/meminfo | head -10

# 시스템 호스트명
cat /etc/hostname

# 현재 로그인 사용자 확인
cat /etc/passwd | head -5
```

---

## 다음 단계

[파일과 디렉토리 관리](./03_File_Directory_Management.md)에서 파일과 디렉토리를 생성, 복사, 이동, 삭제하는 방법을 배워봅시다!
