# 쉘 스크립팅

## 1. 쉘 스크립트 기초

쉘 스크립트는 명령어들을 파일에 모아 자동화하는 프로그램입니다.

### 첫 번째 스크립트

```bash
#!/bin/bash
# 이것은 주석입니다
echo "Hello, World!"
```

### shebang (#!)

스크립트의 첫 줄은 인터프리터를 지정합니다.

```bash
#!/bin/bash        # bash 쉘
#!/bin/sh          # 표준 쉘
#!/usr/bin/env bash  # 환경에서 bash 찾기 (이식성 좋음)
```

### 스크립트 실행

```bash
# 1. 실행 권한 부여 후 실행
chmod +x script.sh
./script.sh

# 2. bash로 직접 실행
bash script.sh

# 3. source로 실행 (현재 쉘에서)
source script.sh
. script.sh
```

---

## 2. 변수

### 변수 선언과 사용

```bash
#!/bin/bash

# 변수 선언 (= 양쪽에 공백 없음!)
name="John"
age=25
readonly PI=3.14159    # 상수

# 변수 사용
echo $name
echo ${name}           # 권장 (명확함)
echo "Hello, ${name}!"
echo "Age: $age"

# 변수 삭제
unset name
```

### 특수 변수

| 변수 | 설명 |
|------|------|
| `$0` | 스크립트 이름 |
| `$1` ~ `$9` | 위치 매개변수 |
| `$#` | 매개변수 개수 |
| `$@` | 모든 매개변수 (개별) |
| `$*` | 모든 매개변수 (문자열) |
| `$?` | 직전 명령 종료 상태 |
| `$$` | 현재 프로세스 PID |
| `$!` | 마지막 백그라운드 PID |

```bash
#!/bin/bash
echo "스크립트: $0"
echo "첫 번째 인자: $1"
echo "두 번째 인자: $2"
echo "인자 개수: $#"
echo "모든 인자: $@"
echo "PID: $$"
```

### 환경 변수

```bash
# 환경 변수 보기
env
printenv

# 주요 환경 변수
echo $HOME       # 홈 디렉토리
echo $USER       # 사용자명
echo $PATH       # 실행 경로
echo $PWD        # 현재 디렉토리
echo $SHELL      # 현재 쉘

# 환경 변수 설정
export MY_VAR="value"

# 스크립트 내에서 설정
#!/bin/bash
export PATH="$PATH:/opt/myapp/bin"
```

### 변수 기본값

```bash
# 변수가 없으면 기본값 사용
name=${name:-"default"}

# 변수가 없으면 기본값 할당
name=${name:="default"}

# 변수가 없으면 에러
name=${name:?"변수가 설정되지 않음"}
```

---

## 3. 입력 받기

### read 명령어

```bash
#!/bin/bash

# 기본 입력
echo -n "이름을 입력하세요: "
read name
echo "안녕하세요, $name님!"

# 프롬프트와 함께
read -p "나이를 입력하세요: " age
echo "당신은 ${age}살입니다."

# 비밀번호 입력 (표시 안 함)
read -sp "비밀번호: " password
echo
echo "비밀번호가 설정되었습니다."

# 시간 제한
read -t 5 -p "5초 내 입력하세요: " input

# 여러 변수에 입력
read -p "이름과 나이: " name age
echo "$name, $age"
```

### 명령줄 인자

```bash
#!/bin/bash
# 사용: ./script.sh arg1 arg2

if [ $# -lt 2 ]; then
    echo "사용법: $0 <이름> <나이>"
    exit 1
fi

name=$1
age=$2
echo "$name님은 ${age}살입니다."
```

---

## 4. 조건문

### if 문

```bash
#!/bin/bash

# 기본 형식
if [ 조건 ]; then
    명령어
fi

# if-else
if [ 조건 ]; then
    명령어1
else
    명령어2
fi

# if-elif-else
if [ 조건1 ]; then
    명령어1
elif [ 조건2 ]; then
    명령어2
else
    명령어3
fi
```

### 비교 연산자

#### 숫자 비교

| 연산자 | 설명 |
|--------|------|
| `-eq` | 같다 (equal) |
| `-ne` | 다르다 (not equal) |
| `-gt` | 크다 (greater than) |
| `-ge` | 크거나 같다 |
| `-lt` | 작다 (less than) |
| `-le` | 작거나 같다 |

```bash
#!/bin/bash
num=10

if [ $num -gt 5 ]; then
    echo "$num은 5보다 큽니다."
fi

if [ $num -eq 10 ]; then
    echo "$num은 10입니다."
fi
```

#### 문자열 비교

| 연산자 | 설명 |
|--------|------|
| `=` | 같다 |
| `!=` | 다르다 |
| `-z` | 빈 문자열 |
| `-n` | 비어있지 않음 |

```bash
#!/bin/bash
str="hello"

if [ "$str" = "hello" ]; then
    echo "문자열이 hello입니다."
fi

if [ -z "$empty" ]; then
    echo "변수가 비어있습니다."
fi

if [ -n "$str" ]; then
    echo "변수에 값이 있습니다."
fi
```

#### 파일 테스트

| 연산자 | 설명 |
|--------|------|
| `-e` | 파일 존재 |
| `-f` | 일반 파일 |
| `-d` | 디렉토리 |
| `-r` | 읽기 가능 |
| `-w` | 쓰기 가능 |
| `-x` | 실행 가능 |
| `-s` | 파일 크기 > 0 |

```bash
#!/bin/bash
file="/etc/passwd"

if [ -e "$file" ]; then
    echo "파일이 존재합니다."
fi

if [ -f "$file" ]; then
    echo "일반 파일입니다."
fi

if [ -d "/home" ]; then
    echo "디렉토리입니다."
fi

if [ -r "$file" ]; then
    echo "읽을 수 있습니다."
fi
```

### 논리 연산자

```bash
#!/bin/bash

# AND
if [ $a -gt 0 ] && [ $b -gt 0 ]; then
    echo "둘 다 양수"
fi

# OR
if [ $a -gt 0 ] || [ $b -gt 0 ]; then
    echo "하나 이상 양수"
fi

# NOT
if [ ! -f "file.txt" ]; then
    echo "파일이 없습니다."
fi

# [[ ]]에서 사용 (권장)
if [[ $a -gt 0 && $b -gt 0 ]]; then
    echo "둘 다 양수"
fi
```

### case 문

```bash
#!/bin/bash
read -p "과일을 선택하세요 (apple/banana/orange): " fruit

case $fruit in
    apple)
        echo "사과를 선택했습니다."
        ;;
    banana)
        echo "바나나를 선택했습니다."
        ;;
    orange)
        echo "오렌지를 선택했습니다."
        ;;
    *)
        echo "알 수 없는 과일입니다."
        ;;
esac
```

---

## 5. 반복문

### for 문

```bash
#!/bin/bash

# 리스트 반복
for name in Alice Bob Charlie; do
    echo "Hello, $name!"
done

# 범위 반복
for i in {1..5}; do
    echo "Number: $i"
done

# 증가값
for i in {0..10..2}; do
    echo "Even: $i"
done

# C 스타일
for ((i=0; i<5; i++)); do
    echo "Index: $i"
done

# 파일 목록
for file in *.txt; do
    echo "Processing: $file"
done

# 명령어 출력
for user in $(cat /etc/passwd | cut -d: -f1); do
    echo "User: $user"
done
```

### while 문

```bash
#!/bin/bash

# 기본 while
count=1
while [ $count -le 5 ]; do
    echo "Count: $count"
    ((count++))
done

# 파일 읽기
while read line; do
    echo "Line: $line"
done < file.txt

# 무한 루프
while true; do
    echo "실행 중... (Ctrl+C로 종료)"
    sleep 1
done
```

### until 문

```bash
#!/bin/bash

# 조건이 참이 될 때까지 반복
count=1
until [ $count -gt 5 ]; do
    echo "Count: $count"
    ((count++))
done
```

### break와 continue

```bash
#!/bin/bash

# break - 루프 종료
for i in {1..10}; do
    if [ $i -eq 5 ]; then
        break
    fi
    echo $i
done

# continue - 다음 반복으로
for i in {1..5}; do
    if [ $i -eq 3 ]; then
        continue
    fi
    echo $i
done
```

---

## 6. 함수

### 함수 정의

```bash
#!/bin/bash

# 방법 1
function greet() {
    echo "Hello, $1!"
}

# 방법 2
say_bye() {
    echo "Goodbye, $1!"
}

# 호출
greet "World"
say_bye "World"
```

### 함수 매개변수

```bash
#!/bin/bash

print_info() {
    echo "이름: $1"
    echo "나이: $2"
    echo "인자 수: $#"
}

print_info "John" 25
```

### 반환값

```bash
#!/bin/bash

# return으로 종료 상태 반환 (0-255)
is_even() {
    if [ $(($1 % 2)) -eq 0 ]; then
        return 0    # true
    else
        return 1    # false
    fi
}

if is_even 4; then
    echo "짝수입니다."
fi

# echo로 값 반환
add() {
    echo $(($1 + $2))
}

result=$(add 5 3)
echo "5 + 3 = $result"
```

### 지역 변수

```bash
#!/bin/bash

my_func() {
    local local_var="나는 지역 변수"
    global_var="나는 전역 변수"
    echo "$local_var"
}

my_func
echo "$global_var"     # 출력됨
echo "$local_var"      # 출력 안 됨
```

---

## 7. 배열

```bash
#!/bin/bash

# 배열 선언
fruits=("apple" "banana" "orange")

# 인덱스로 접근
echo ${fruits[0]}     # apple
echo ${fruits[1]}     # banana

# 모든 요소
echo ${fruits[@]}

# 요소 개수
echo ${#fruits[@]}

# 배열 추가
fruits+=("grape")

# 배열 반복
for fruit in "${fruits[@]}"; do
    echo $fruit
done

# 인덱스와 함께
for i in "${!fruits[@]}"; do
    echo "$i: ${fruits[$i]}"
done
```

---

## 8. 실무 스크립트 예제

### 백업 스크립트

```bash
#!/bin/bash
# backup.sh - 디렉토리 백업 스크립트

SOURCE_DIR="${1:-/var/www}"
BACKUP_DIR="${2:-/backup}"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="backup_${DATE}.tar.gz"

# 백업 디렉토리 확인
if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
fi

# 소스 디렉토리 확인
if [ ! -d "$SOURCE_DIR" ]; then
    echo "오류: $SOURCE_DIR 디렉토리가 없습니다."
    exit 1
fi

# 백업 실행
echo "백업 시작: $SOURCE_DIR -> $BACKUP_DIR/$BACKUP_FILE"
tar -czvf "$BACKUP_DIR/$BACKUP_FILE" -C "$(dirname $SOURCE_DIR)" "$(basename $SOURCE_DIR)"

if [ $? -eq 0 ]; then
    echo "백업 완료: $BACKUP_DIR/$BACKUP_FILE"

    # 30일 이상 된 백업 삭제
    find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +30 -delete
    echo "오래된 백업 정리 완료"
else
    echo "백업 실패!"
    exit 1
fi
```

### 서버 상태 체크 스크립트

```bash
#!/bin/bash
# health_check.sh - 서버 상태 체크

LOG_FILE="/var/log/health_check.log"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_disk() {
    local usage=$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')
    if [ "$usage" -gt 80 ]; then
        log_message "경고: 디스크 사용량 ${usage}%"
        return 1
    fi
    log_message "디스크: ${usage}% 사용 중"
    return 0
}

check_memory() {
    local usage=$(free | awk '/Mem:/ {printf("%.0f", $3/$2 * 100)}')
    if [ "$usage" -gt 80 ]; then
        log_message "경고: 메모리 사용량 ${usage}%"
        return 1
    fi
    log_message "메모리: ${usage}% 사용 중"
    return 0
}

check_service() {
    local service=$1
    if systemctl is-active --quiet "$service"; then
        log_message "서비스 $service: 실행 중"
        return 0
    else
        log_message "경고: 서비스 $service 중지됨"
        return 1
    fi
}

# 메인
log_message "===== 상태 체크 시작 ====="
check_disk
check_memory
check_service "sshd"
check_service "nginx" 2>/dev/null || true
log_message "===== 상태 체크 완료 ====="
```

### 로그 분석 스크립트

```bash
#!/bin/bash
# log_analyzer.sh - 로그 분석

LOG_FILE="${1:-/var/log/syslog}"

if [ ! -f "$LOG_FILE" ]; then
    echo "파일을 찾을 수 없습니다: $LOG_FILE"
    exit 1
fi

echo "=== 로그 분석: $LOG_FILE ==="
echo

echo "=== 에러 수 ==="
grep -ci "error" "$LOG_FILE"

echo
echo "=== 최근 에러 10개 ==="
grep -i "error" "$LOG_FILE" | tail -10

echo
echo "=== 시간대별 로그 수 ==="
awk '{print $3}' "$LOG_FILE" | cut -d: -f1 | sort | uniq -c | sort -k2n
```

### 사용자 생성 스크립트

```bash
#!/bin/bash
# create_users.sh - 파일에서 사용자 일괄 생성
# 사용: sudo ./create_users.sh users.txt
# users.txt 형식: username:password:groupname

USER_FILE="$1"

if [ -z "$USER_FILE" ] || [ ! -f "$USER_FILE" ]; then
    echo "사용법: $0 <사용자파일>"
    exit 1
fi

while IFS=: read -r username password groupname; do
    # 빈 줄 또는 주석 건너뛰기
    [[ -z "$username" || "$username" =~ ^# ]] && continue

    # 사용자 존재 확인
    if id "$username" &>/dev/null; then
        echo "사용자 존재: $username (건너뜀)"
        continue
    fi

    # 그룹 생성 (없으면)
    if ! getent group "$groupname" &>/dev/null; then
        groupadd "$groupname"
        echo "그룹 생성: $groupname"
    fi

    # 사용자 생성
    useradd -m -s /bin/bash -g "$groupname" "$username"
    echo "$username:$password" | chpasswd
    echo "사용자 생성: $username (그룹: $groupname)"

done < "$USER_FILE"

echo "완료!"
```

---

## 9. 디버깅

### 디버그 모드

```bash
# 스크립트 실행 추적
bash -x script.sh

# 스크립트 내에서 활성화
#!/bin/bash
set -x    # 디버그 시작
# 명령어들...
set +x    # 디버그 종료

# 에러 발생 시 종료
set -e

# 미정의 변수 사용 시 에러
set -u

# 파이프 에러 감지
set -o pipefail

# 모범 사례 (스크립트 시작)
#!/bin/bash
set -euo pipefail
```

---

## 10. 실습 예제

### 실습 1: 간단한 계산기

```bash
#!/bin/bash
# calculator.sh

read -p "첫 번째 숫자: " num1
read -p "연산자 (+, -, *, /): " op
read -p "두 번째 숫자: " num2

case $op in
    +) result=$((num1 + num2)) ;;
    -) result=$((num1 - num2)) ;;
    \*) result=$((num1 * num2)) ;;
    /) result=$((num1 / num2)) ;;
    *) echo "잘못된 연산자"; exit 1 ;;
esac

echo "결과: $num1 $op $num2 = $result"
```

### 실습 2: 파일 정리 스크립트

```bash
#!/bin/bash
# organize.sh - 파일 확장자별 정리

DIR="${1:-.}"

for file in "$DIR"/*; do
    [ -f "$file" ] || continue

    ext="${file##*.}"
    mkdir -p "$DIR/$ext"
    mv "$file" "$DIR/$ext/"
    echo "이동: $file -> $DIR/$ext/"
done
```

---

## 다음 단계

[10_Network_Basics.md](./10_Network_Basics.md)에서 네트워크 관리를 배워봅시다!
