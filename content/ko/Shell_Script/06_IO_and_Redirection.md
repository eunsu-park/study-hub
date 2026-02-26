# Lesson 06: I/O와 리다이렉션(Redirection)

**난이도**: ⭐⭐⭐

**이전**: [함수와 라이브러리](./05_Functions_and_Libraries.md) | **다음**: [문자열 처리와 텍스트 조작](./07_String_Processing.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 파일 디스크립터(file descriptor)의 작동 방식을 설명하고 다중 스트림 I/O를 위한 사용자 정의 FD를 생성할 수 있습니다
2. 고급 리다이렉션(redirection) 기법을 적용하여 stdout과 stderr를 분리, 병합, 교환, 저장/복원할 수 있습니다
3. 인라인 데이터와 템플릿 생성을 위해 히어 도큐먼트(here document)와 히어 스트링(here string)을 작성할 수 있습니다
4. 파이프라인에서 서브셸(subshell) 변수 스코프 문제를 피하기 위해 프로세스 치환(process substitution)을 활용할 수 있습니다
5. 이름 있는 파이프(named pipe, FIFO)를 사용하여 생산자-소비자(producer-consumer) 통신을 구현할 수 있습니다
6. 서브셸 스코프 손실과 PIPESTATUS 확인을 포함한 일반적인 파이프 함정을 식별할 수 있습니다
7. 원자적(atomic) 파일 업데이트, 파일 잠금(file-locking) 루틴, 다중 대상 로깅 패턴을 작성할 수 있습니다

---

I/O 리다이렉션은 데이터 파이프라인, 로깅, 프로세스 조정을 위한 셸 스크립팅을 강력하게 만드는 핵심입니다. 단순한 `>` 와 `|` 를 넘어서, bash는 중간 파일 없이 정교한 데이터 흐름을 구축할 수 있는 파일 디스크립터(file descriptor), 프로세스 치환(process substitution), 히어 도큐먼트(here document), 이름 있는 파이프(named pipe)를 제공합니다. 로그 핸들러를 작성하거나, 다중 스트림 명령 출력을 파싱하거나, 동시 생산자-소비자 워크플로우를 구축할 때 이 기술들이 필요합니다.

## 1. 파일 디스크립터(File Descriptors)

파일 디스크립터(FD)는 열린 파일이나 I/O 스트림을 참조하는 정수입니다. bash에서 I/O 리다이렉션을 마스터하려면 이들을 이해하는 것이 필수적입니다.

### 1.1 표준 파일 디스크립터

모든 프로세스는 세 개의 표준 파일 디스크립터를 가집니다:

| FD | 이름 | 목적 | 기본값 |
|----|------|---------|---------|
| 0 | stdin | 표준 입력(Standard input) | 키보드 |
| 1 | stdout | 표준 출력(Standard output) | 터미널 |
| 2 | stderr | 표준 에러(Standard error) | 터미널 |

```bash
#!/bin/bash

# stdin(FD 0)에서 읽기
read -p "Enter your name: " name
echo "Hello, $name"

# stdout(FD 1)에 쓰기
echo "This goes to stdout" >&1  # 명시적 (일반 echo와 동일)

# stderr(FD 2)에 쓰기
echo "This is an error message" >&2
```

### 1.2 사용자 정의 파일 디스크립터

3-9번 파일 디스크립터를 사용자 정의로 생성할 수 있습니다:

```bash
#!/bin/bash

# FD 3에 읽기용으로 파일 열기
exec 3< input.txt

# FD 3에서 읽기
while read -u 3 line; do
    echo "Line: $line"
done

# FD 3 닫기
exec 3<&-

# FD 4에 쓰기용으로 파일 열기
exec 4> output.txt

# FD 4에 쓰기
echo "First line" >&4
echo "Second line" >&4

# FD 4 닫기
exec 4>&-
```

### 1.3 읽기/쓰기용 파일 디스크립터 열기

```bash
#!/bin/bash

# FD 5에 읽기와 쓰기 모두 가능하게 파일 열기
exec 5<> datafile.txt

# 현재 내용 읽기
while read -u 5 line; do
    echo "Read: $line"
done

# 새 내용 쓰기 (추가됨)
echo "New data" >&5

# FD 5 닫기
exec 5>&-
```

### 1.4 파일 디스크립터 복제

```bash
#!/bin/bash

# stdout(FD 1)을 FD 3으로 복제
exec 3>&1

# 이제 stdout을 파일로 리다이렉트
exec 1> output.log

# 이것은 output.log로 감
echo "Logging to file"

# 이것은 여전히 터미널로 감 (FD 3 사용)
echo "Direct to terminal" >&3

# FD 3에서 stdout 복원
exec 1>&3

# FD 3 닫기
exec 3>&-

# 이제 다시 터미널로
echo "Back to normal stdout"
```

### 1.5 파일 디스크립터 검사

```bash
#!/bin/bash

# 현재 셸의 파일 디스크립터 보기
ls -l /dev/fd/
# 또는
ls -l /proc/self/fd/

# FD가 열려있는지 확인
if [[ -e /dev/fd/3 ]]; then
    echo "FD 3 is open"
else
    echo "FD 3 is closed"
fi

# FD에 대한 정보 얻기
exec 5> myfile.txt
readlink /proc/self/fd/5  # 파일 경로 표시
exec 5>&-
```

## 2. 고급 리다이렉션

기본 `>`와 `<` 이외에도, bash는 강력한 리다이렉션 연산자를 제공합니다.

### 2.1 Stderr 별도로 리다이렉트

```bash
#!/bin/bash

# stdout을 file1로, stderr를 file2로 리다이렉트
command > stdout.log 2> stderr.log

# 예제: C 프로그램 컴파일
gcc program.c -o program > compile_output.txt 2> compile_errors.txt

# 컴파일 에러가 있는지 확인
if [[ -s compile_errors.txt ]]; then
    echo "Compilation failed:"
    cat compile_errors.txt
else
    echo "Compilation successful!"
fi
```

### 2.2 Stdout과 Stderr 병합

```bash
#!/bin/bash

# 방법 1: stderr를 stdout으로 리다이렉트
command > output.log 2>&1

# 방법 2: 단축 표기법 (Bash 4+)
command &> output.log

# 방법 3: 둘 다 추가
command >> output.log 2>&1

# 예제: 테스트 스위트 실행
./run_tests.sh &> test_results.log

# 이것은 잘못됨 (순서가 중요):
command 2>&1 > output.log  # stderr는 여전히 터미널로!
# 올바른 방법:
command > output.log 2>&1  # stderr가 stdout을 따라 파일로
```

### 2.3 출력 버리기

```bash
#!/bin/bash

# stdout 버리기
command > /dev/null

# stderr 버리기
command 2> /dev/null

# 둘 다 버리기
command &> /dev/null

# 예제: 무음 작업
if some_command &> /dev/null; then
    echo "Command succeeded (silently)"
fi

# stderr 유지, stdout 버리기
command > /dev/null

# 예제: 명령어가 존재하는지 확인
if command -v python3 > /dev/null 2>&1; then
    echo "python3 is installed"
fi
```

### 2.4 Stdout과 Stderr 바꾸기

```bash
#!/bin/bash

# stdout과 stderr 바꾸기
command 3>&1 1>&2 2>&3 3>&-

# 설명:
# 3>&1  - stdout을 FD 3에 저장
# 1>&2  - stdout을 stderr로 리다이렉트
# 2>&3  - stderr를 FD 3으로 리다이렉트 (원래 stdout)
# 3>&-  - FD 3 닫기

# 실전 예제: 에러 메시지를 stdout으로, 일반 출력을 stderr로
swap_outputs() {
    "$@" 3>&1 1>&2 2>&3 3>&-
}

# 이제 에러가 stdout에 나타남 (캡처 가능)
errors=$(swap_outputs some_command)
```

### 2.5 파일 디스크립터 저장 및 복원

```bash
#!/bin/bash

# 원래의 stdout과 stderr 저장
exec 3>&1 4>&2

# stdout과 stderr를 파일로 리다이렉트
exec 1> output.log 2> error.log

# 여기 있는 명령들은 로그 파일에 씀
echo "This goes to output.log"
echo "This is an error" >&2

# 원래의 stdout과 stderr 복원
exec 1>&3 2>&4

# 백업 FD 닫기
exec 3>&- 4>&-

# 이제 다시 터미널로
echo "Back to terminal"
```

### 2.6 추가(Append) vs 덮어쓰기(Truncate)

```bash
#!/bin/bash

# 파일 덮어쓰기
echo "New content" > file.txt

# 파일에 추가
echo "Additional content" >> file.txt

# stderr 추가
command 2>> error.log

# stdout과 stderr 모두 추가
command &>> output.log
```

## 3. Here Documents와 Here Strings

Here documents는 임시 파일을 만들지 않고 명령에 여러 줄의 입력을 제공합니다.

### 3.1 기본 Here Document

```bash
#!/bin/bash

# 기본 here document
cat <<EOF
This is a multi-line
here document.
It can contain variables: $HOME
And command substitution: $(date)
EOF

# 들여쓰기 포함 (<<-는 선행 탭 제거, 공백은 아님)
cat <<-EOF
	This is indented with tabs
	The tabs will be removed
	But the text stays aligned
EOF
```

### 3.2 변수 확장 없는 Here Document

```bash
#!/bin/bash

# 구분자를 따옴표로 묶어 확장 방지
cat <<'EOF'
Variables are literal: $HOME
Command substitution is literal: $(date)
This is useful for generating scripts or code.
EOF

# 예제: bash 스크립트 생성
cat <<'SCRIPT' > myscript.sh
#!/bin/bash
echo "Hello from generated script"
echo "Current directory: $PWD"
SCRIPT

chmod +x myscript.sh
```

### 3.3 변수에 Here Document 저장

```bash
#!/bin/bash

# here document를 변수에 할당
read -r -d '' sql_query <<EOF
SELECT u.name, u.email, o.order_id
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.status = 'pending'
ORDER BY o.created_at DESC
LIMIT 10;
EOF

echo "Executing query:"
echo "$sql_query"

# 대안 방법 (명령 치환 사용)
json_data=$(cat <<EOF
{
    "name": "John Doe",
    "email": "john@example.com",
    "age": 30,
    "roles": ["admin", "user"]
}
EOF
)

echo "$json_data"
```

### 3.4 명령 입력과 Here Document

```bash
#!/bin/bash

# 명령에 여러 줄 입력 보내기
mysql -u root -p <<SQL
USE mydb;
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO users (username, email) VALUES ('alice', 'alice@example.com');
SQL

# Python 스크립트 실행
python3 <<PYTHON
import sys
import json

data = {
    'message': 'Hello from Python',
    'version': sys.version
}

print(json.dumps(data, indent=2))
PYTHON
```

### 3.5 Here Strings

```bash
#!/bin/bash

# Here string: 한 줄 입력
grep "pattern" <<< "This is a test pattern string"

# 변수 파이핑에 유용
while read -r word; do
    echo "Word: $word"
done <<< "one two three four five"

# 예제: CSV 줄 파싱
IFS=',' read -r name age city <<< "John,30,NYC"
echo "Name: $name, Age: $age, City: $city"

# 문자열을 Base64 인코딩
encoded=$(base64 <<< "Secret message")
echo "Encoded: $encoded"

# 다시 디코딩
decoded=$(base64 -d <<< "$encoded")
echo "Decoded: $decoded"
```

### 3.6 실전 템플릿 생성

```bash
#!/bin/bash

generate_html() {
    local title=$1
    local content=$2

    cat <<HTML
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>$title</title>
</head>
<body>
    <h1>$title</h1>
    <p>$content</p>
    <footer>Generated on $(date)</footer>
</body>
</html>
HTML
}

# HTML 페이지 생성
generate_html "My Page" "Welcome to my website!" > index.html

# 설정 파일 생성
generate_config() {
    local host=$1
    local port=$2

    cat <<CONFIG > app.conf
# Application Configuration
# Generated: $(date)

[server]
host = $host
port = $port
workers = 4

[database]
host = localhost
port = 5432
name = myapp

[logging]
level = INFO
file = /var/log/myapp.log
CONFIG
}

generate_config "0.0.0.0" "8080"
```

## 4. 프로세스 치환(Process Substitution)

프로세스 치환은 명령 출력에 대한 임시 명명된 파이프를 생성하여, 파일이 필요한 곳에 명령을 사용할 수 있게 합니다.

### 4.1 입력 프로세스 치환

```bash
#!/bin/bash

# 두 명령의 출력 비교
diff <(ls dir1) <(ls dir2)

# 더 복잡한 예제: 정렬된 리스트 비교
diff <(sort file1.txt) <(sort file2.txt)

# 두 시스템의 실행 중인 프로세스 비교
diff <(ssh server1 ps aux | sort) <(ssh server2 ps aux | sort)

# 예제: 두 명령 출력의 공통 줄 찾기
comm -12 <(sort list1.txt) <(sort list2.txt)
```

### 4.2 출력 프로세스 치환

```bash
#!/bin/bash

# 여러 파일에 동시에 쓰기
tee >(grep "ERROR" > errors.log) \
    >(grep "WARN" > warnings.log) \
    >(grep "INFO" > info.log) \
    < application.log > /dev/null

# 예제: 로그를 심각도로 분할
process_logs() {
    local logfile=$1

    cat "$logfile" | tee \
        >(grep "ERROR" > errors.log) \
        >(grep "WARN" > warnings.log) \
        > all.log
}
```

### 4.3 서브셸 변수 스코프 문제 회피

```bash
#!/bin/bash

# 문제: 파이프라인의 변수는 서브셸에 있음
count=0
cat file.txt | while read line; do
    ((count++))
done
echo "Lines: $count"  # 출력: 0 (변수가 수정되지 않음!)

# 해결책 1: 프로세스 치환
count=0
while read line; do
    ((count++))
done < <(cat file.txt)
echo "Lines: $count"  # 올바른 카운트

# 해결책 2: 명령 치환과 here string 사용 (작은 파일용)
count=0
while read line; do
    ((count++))
done <<< "$(cat file.txt)"
echo "Lines: $count"  # 올바른 카운트
```

### 4.4 다중 입력 스트림

```bash
#!/bin/bash

# 여러 파일에서 병렬로 읽기
paste <(cut -d',' -f1 file1.csv) \
      <(cut -d',' -f2 file2.csv) \
      <(cut -d',' -f3 file3.csv)

# 예제: 여러 소스의 데이터 병합
while read -u 3 name && read -u 4 age && read -u 5 city; do
    echo "$name is $age years old and lives in $city"
done 3< <(cut -d',' -f1 data.csv) \
     4< <(cut -d',' -f2 data.csv) \
     5< <(cut -d',' -f3 data.csv)
```

### 4.5 실전 예제

```bash
#!/bin/bash

# 예제 1: 최근 24시간 내 수정된 파일 중 패턴이 포함된 파일 찾기
grep "TODO" <(find . -type f -mtime -1 -exec cat {} \;)

# 예제 2: 로그 파일 모니터링 및 알림 전송
while read line; do
    if [[ $line == *"CRITICAL"* ]]; then
        echo "Alert: $line" | mail -s "Critical Error" admin@example.com
    fi
done < <(tail -f /var/log/app.log)

# 예제 3: 압축 파일을 추출하지 않고 처리
while read line; do
    echo "Processing: $line"
done < <(gunzip -c data.txt.gz)

# 예제 4: 처리를 위한 임시 파일 리스트 생성
tar czf backup.tar.gz -T <(find /data -type f -mtime -7)
```

## 5. 명명된 파이프(Named Pipes - FIFOs)

명명된 파이프는 파일시스템을 통한 프로세스 간 통신을 가능하게 합니다.

### 5.1 FIFO 생성 및 사용

```bash
#!/bin/bash

# 명명된 파이프 생성
mkfifo mypipe

# 생산자 (백그라운드 프로세스)
{
    for i in {1..10}; do
        echo "Message $i"
        sleep 1
    done > mypipe
} &

# 소비자
while read line; do
    echo "Received: $line"
done < mypipe

# 정리
rm mypipe
```

### 5.2 생산자-소비자 패턴

```bash
#!/bin/bash

PIPE="/tmp/data_pipe_$$"

# 파이프 생성 및 정리용 trap 설정
mkfifo "$PIPE"
trap "rm -f '$PIPE'" EXIT

# 생산자: 데이터 생성
producer() {
    local pipe=$1
    echo "Producer starting..."

    for i in {1..100}; do
        echo "Data item $i: $(date +%s)"
        sleep 0.1
    done > "$pipe"

    echo "Producer finished"
}

# 소비자: 데이터 처리
consumer() {
    local pipe=$1
    echo "Consumer starting..."

    local count=0
    while read line; do
        ((count++))
        # 데이터 처리 (작업 시뮬레이션)
        [[ $((count % 10)) -eq 0 ]] && echo "Processed $count items"
    done < "$pipe"

    echo "Consumer finished: $count items processed"
}

# 백그라운드에서 생산자 실행
producer "$PIPE" &
producer_pid=$!

# 포그라운드에서 소비자 실행
consumer "$PIPE"

# 생산자가 끝날 때까지 대기
wait $producer_pid
```

### 5.3 양방향 통신

```bash
#!/bin/bash

REQUEST_PIPE="/tmp/request_$$"
RESPONSE_PIPE="/tmp/response_$$"

# 파이프 생성
mkfifo "$REQUEST_PIPE" "$RESPONSE_PIPE"
trap "rm -f '$REQUEST_PIPE' '$RESPONSE_PIPE'" EXIT

# 서버 프로세스
server() {
    echo "Server started"

    while true; do
        # 요청 읽기
        read request < "$REQUEST_PIPE"

        # 요청 처리
        case $request in
            "PING")
                echo "PONG" > "$RESPONSE_PIPE"
                ;;
            "TIME")
                date > "$RESPONSE_PIPE"
                ;;
            "QUIT")
                echo "BYE" > "$RESPONSE_PIPE"
                break
                ;;
            *)
                echo "ERROR: Unknown command" > "$RESPONSE_PIPE"
                ;;
        esac
    done

    echo "Server stopped"
}

# 클라이언트 함수
client() {
    local command=$1

    # 요청 전송
    echo "$command" > "$REQUEST_PIPE"

    # 응답 읽기
    read response < "$RESPONSE_PIPE"
    echo "Response: $response"
}

# 백그라운드에서 서버 시작
server &
server_pid=$!

sleep 1  # 서버가 시작할 시간 줌

# 요청 전송
client "PING"
client "TIME"
client "QUIT"

# 서버 대기
wait $server_pid
```

### 5.4 FIFO vs 프로세스 치환 사용 시기

| 특징 | FIFO | 프로세스 치환(Process Substitution) |
|---------|------|---------------------|
| 지속성 | 예 (삭제될 때까지) | 아니오 (자동 정리) |
| 다중 reader/writer | 예 | 아니오 |
| 명시적 동기화 | 예 | 아니오 |
| 백그라운드 사용 | 쉬움 | 복잡함 |
| 정리 필요 | 수동 | 자동 |
| 최적 용도 | 장기 실행 IPC | 일회성 작업 |

```bash
#!/bin/bash

# 일회성 비교에는 프로세스 치환 사용
diff <(command1) <(command2)

# 지속적 통신에는 FIFO 사용
mkfifo /tmp/logpipe
tail -f /var/log/app.log > /tmp/logpipe &
while read line; do
    process_log_line "$line"
done < /tmp/logpipe
```

## 6. 파이프 함정과 해결책

### 6.1 서브셸 변수 스코프 손실

```bash
#!/bin/bash

# 문제: 파이프라인의 마지막 명령이 서브셸에서 실행됨
total=0
cat numbers.txt | while read num; do
    ((total += num))
done
echo "Total: $total"  # 출력: 0 (수정되지 않음!)

# 해결책 1: 프로세스 치환
total=0
while read num; do
    ((total += num))
done < <(cat numbers.txt)
echo "Total: $total"  # 올바름

# 해결책 2: lastpipe 사용 (Bash 4.2+, 스크립트에서만)
shopt -s lastpipe
total=0
cat numbers.txt | while read num; do
    ((total += num))
done
echo "Total: $total"  # 올바름

# 해결책 3: 임시 파일
tmpfile=$(mktemp)
cat numbers.txt > "$tmpfile"
total=0
while read num; do
    ((total += num))
done < "$tmpfile"
rm "$tmpfile"
echo "Total: $total"  # 올바름
```

### 6.2 PIPESTATUS 배열

```bash
#!/bin/bash

# 파이프라인의 모든 명령 종료 상태 확인
command1 | command2 | command3

# PIPESTATUS는 모든 명령의 종료 코드를 포함
echo "Exit codes: ${PIPESTATUS[@]}"

# 예제: 파이프라인 어디서든 실패 감지
false | true | true
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "First command failed"
fi

# 실전 예제: 데이터베이스 파이프라인
{
    mysql -u root -p mydb -e "SELECT * FROM users" | \
    grep "active" | \
    sort -k2
} 2>/dev/null

pipeline_status=("${PIPESTATUS[@]}")
if [[ ${pipeline_status[0]} -ne 0 ]]; then
    echo "Database query failed"
elif [[ ${pipeline_status[1]} -ne 0 ]]; then
    echo "Grep failed"
elif [[ ${pipeline_status[2]} -ne 0 ]]; then
    echo "Sort failed"
else
    echo "Pipeline succeeded"
fi
```

### 6.3 파이프라인 에러 처리

```bash
#!/bin/bash

# pipefail 활성화: 어떤 명령이든 실패하면 파이프라인 실패
set -o pipefail

# 이제 파이프라인은 어떤 명령이든 실패하면 0이 아닌 값을 반환
if command1 | command2 | command3; then
    echo "Pipeline succeeded"
else
    echo "Pipeline failed"
fi

# 실전 예제: 안전한 데이터 처리
set -euo pipefail  # 에러, 미정의 변수, 파이프라인 실패 시 종료

process_data() {
    local input=$1
    local output=$2

    cat "$input" | \
        grep -v "^#" | \
        sort -u | \
        sed 's/foo/bar/g' \
        > "$output"

    # 어떤 명령이든 실패하면 스크립트 종료
}

# pipefail과 함께 에러 처리
set -o pipefail
if ! tar czf backup.tar.gz --exclude="*.tmp" -T <(find /data -type f); then
    echo "Backup failed" >&2
    exit 1
fi
```

### 6.4 명명된 파이프 교착 상태 방지

```bash
#!/bin/bash

# 문제: reader/writer가 조율되지 않으면 교착 상태
mkfifo mypipe
echo "data" > mypipe  # 영원히 블록! (reader 없음)

# 해결책 1: 읽기와 쓰기 모두로 파이프 열기
mkfifo mypipe
exec 3<> mypipe  # 읽기/쓰기로 열기

echo "data" >&3  # 쓰기
read line <&3    # 읽기
exec 3>&-        # 닫기

rm mypipe

# 해결책 2: 적절한 동기화와 함께 백그라운드 프로세스
mkfifo mypipe
trap "rm -f mypipe" EXIT

# 백그라운드의 reader
cat < mypipe &
reader_pid=$!

# Writer
echo "data" > mypipe

# reader 대기
wait $reader_pid
```

## 7. 실전 I/O 패턴

### 7.1 여러 목적지로 Tee

```bash
#!/bin/bash

# 기본 tee: 파일과 stdout에 쓰기
echo "Important message" | tee log.txt

# 여러 파일
echo "Message" | tee file1.txt file2.txt file3.txt

# 추가 모드
echo "New entry" | tee -a logfile.txt

# 복잡한 예제: 분할 처리
cat data.txt | tee \
    >(grep "ERROR" > errors.log) \
    >(grep "WARN" > warnings.log) \
    >(wc -l > linecount.txt) \
    | grep "INFO" > info.log
```

### 7.2 콘솔과 파일에 로깅

```bash
#!/bin/bash

# 로깅 설정
LOGFILE="application.log"

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # 콘솔과 파일 모두에 로그
    echo "[$timestamp] [$level] $message" | tee -a "$LOGFILE"
}

# 사용
log "INFO" "Application started"
log "WARN" "Configuration file not found, using defaults"
log "ERROR" "Failed to connect to database"

# 대안: 모든 출력 리다이렉트
exec > >(tee -a "$LOGFILE")
exec 2>&1

# 이제 모든 출력이 콘솔과 파일 모두로 감
echo "This appears in both places"
ls /nonexistent  # 에러도 로깅됨
```

### 7.3 동일 파일 안전하게 읽고 쓰기

```bash
#!/bin/bash

# 잘못된 방법: 읽기 전에 파일을 덮어씀!
sort file.txt > file.txt  # file.txt가 비워짐!

# 해결책 1: sponge 사용 (moreutils에서)
sort file.txt | sponge file.txt

# 해결책 2: 임시 파일 사용
sort file.txt > file.txt.tmp && mv file.txt.tmp file.txt

# 해결책 3: -i 플래그로 제자리 편집 (지원되는 경우)
sed -i 's/foo/bar/g' file.txt

# 원자적 파일 교체
update_config() {
    local config_file=$1
    local tmpfile=$(mktemp)

    # 파일 처리
    process_config < "$config_file" > "$tmpfile"

    # 원자적 교체
    mv "$tmpfile" "$config_file"
}
```

### 7.4 원자적 파일 쓰기

```bash
#!/bin/bash

# 원자적 쓰기 패턴: 임시에 쓰고 그 다음 이동
atomic_write() {
    local target_file=$1
    local content=$2

    local tmpfile=$(mktemp "${target_file}.XXXXXX")

    # 임시 파일에 쓰기
    echo "$content" > "$tmpfile"

    # 쓰기가 성공했는지 확인
    if [[ $? -eq 0 ]]; then
        # 원자적 이동 (같은 파일시스템에서)
        mv "$tmpfile" "$target_file"
    else
        rm -f "$tmpfile"
        return 1
    fi
}

# 사용
atomic_write "config.json" '{"setting": "value"}'

# 복잡한 예제: 중요한 파일 업데이트
update_critical_file() {
    local file=$1
    local tmpfile=$(mktemp)

    # 정리용 trap 설정
    trap "rm -f '$tmpfile'" RETURN

    # 새 콘텐츠 생성
    if ! generate_content > "$tmpfile"; then
        echo "Error: Failed to generate content" >&2
        return 1
    fi

    # 새 콘텐츠 검증
    if ! validate_content "$tmpfile"; then
        echo "Error: Content validation failed" >&2
        return 1
    fi

    # 원본과 동일한 권한 설정
    chmod --reference="$file" "$tmpfile" 2>/dev/null

    # 원자적 교체
    mv "$tmpfile" "$file"
}
```

### 7.5 안전한 동시 접근을 위한 파일 잠금

```bash
#!/bin/bash

# 파일 잠금에 flock 사용
update_counter() {
    local counter_file="counter.txt"
    local lockfile="counter.lock"

    # 배타적 잠금 획득 (FD 200)
    {
        flock -x 200

        # 현재 값 읽기
        local count=0
        [[ -f $counter_file ]] && count=$(cat "$counter_file")

        # 증가
        ((count++))

        # 다시 쓰기
        echo "$count" > "$counter_file"

        echo "Counter updated to: $count"

    } 200>"$lockfile"
}

# 여러 프로세스가 안전하게 이것을 호출 가능
for i in {1..10}; do
    update_counter &
done
wait

# 최종 값
echo "Final count: $(cat counter.txt)"

# 대안: 인라인 잠금
{
    flock -x 200

    # 임계 영역
    echo "Exclusive access to resource"
    sleep 2

} 200>/tmp/mylock
```

### 7.6 FIFO를 사용한 진행 상황 표시

```bash
#!/bin/bash

# 진행 상황 파이프 생성
mkfifo /tmp/progress_$$
trap "rm -f /tmp/progress_$$" EXIT

# 진행 상황 모니터 (백그라운드)
{
    while read percent message; do
        printf "\r[%-50s] %d%% %s" \
            "$(printf '#%.0s' $(seq 1 $((percent / 2))))" \
            "$percent" \
            "$message"
    done < /tmp/progress_$$
    echo
} &
monitor_pid=$!

# 작업 프로세스
{
    total=100
    for i in $(seq 1 $total); do
        # 작업 시뮬레이션
        sleep 0.05

        # 진행 상황 보고
        percent=$((i * 100 / total))
        echo "$percent Processing item $i" > /tmp/progress_$$
    done
} &
worker_pid=$!

# 완료 대기
wait $worker_pid
wait $monitor_pid
```

## 연습 문제

### 문제 1: 다중 대상 로거

다음 기능을 가진 로깅 시스템 생성:
- 로그 레벨 (DEBUG, INFO, WARN, ERROR)과 메시지 허용
- 모든 로그를 `all.log`에 쓰기
- ERROR 로그를 `error.log`에 쓰기
- WARN과 ERROR를 `important.log`에 쓰기
- ERROR와 WARN을 stderr에, 나머지는 stdout에 표시
- 각 로그 항목에 타임스탬프와 호스트명 추가
- 파일이 10MB를 초과하면 로그 로테이션 구현

### 문제 2: 파이프라인 모니터

다음 기능을 가진 스크립트 작성:
- 다단계 파이프라인 실행 (예: download | decompress | process | upload)
- PIPESTATUS를 사용하여 각 단계의 종료 상태 모니터링
- 프로세스 치환을 사용하여 파일에 진행 상황 로깅
- 실패한 단계에 대한 재시도 로직 구현
- 어느 단계가 실패했는지와 그 이유 보고
- 총 시간과 처리량 계산

### 문제 3: FIFO 기반 큐 시스템

명명된 파이프를 사용한 간단한 작업 큐 구현:
- 큐에 작업을 보내는 `job_submit` 명령 생성
- 큐에서 작업을 처리하는 `job_worker` 생성
- 여러 동시 워커 지원
- 작업 상태 추적 구현 (pending, running, completed, failed)
- 워커 충돌을 우아하게 처리
- 큐 상태를 확인하는 `job_status` 명령 제공

### 문제 4: 설정 검증기

다음 기능을 가진 도구 구축:
- stdin 또는 파일 인자에서 설정 파일 읽기
- 검증 명령을 사용하여 구문 검증
- 유효하면, 원자적으로 이전 config 교체
- 유효하지 않으면, stderr에 에러 표시하고 이전 config 유지
- 교체 전 백업 생성 (마지막 5개 백업 유지)
- 타임스탬프와 함께 모든 변경 사항 로깅
- dry-run 모드 지원 (교체하지 않고 검증만)

### 문제 5: FD를 사용한 스트림 프로세서

스트림 처리 프레임워크 생성:
- FD 3, 4, 5에 3개 입력 스트림 열기
- 타임스탬프와 함께 스트림 병합
- 정규식 패턴에 기반한 필터링
- 내용에 따라 다른 파일로 출력 분할
- 통계 유지 (스트림당 처리된 줄, 일치, 에러)
- 실시간 모니터링을 위해 프로세스 치환 사용
- 스트림 종료를 우아하게 처리

**이전**: [05_Functions_and_Libraries.md](./05_Functions_and_Libraries.md) | **다음**: [07_String_Processing.md](./07_String_Processing.md)
