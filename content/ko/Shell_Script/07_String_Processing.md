# Lesson 07: 문자열 처리와 텍스트 조작

**난이도**: ⭐⭐⭐

**이전**: [I/O와 리다이렉션](./06_IO_and_Redirection.md) | **다음**: [Bash에서의 정규 표현식](./08_Regex_in_Bash.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 내장 매개변수 확장(parameter expansion)을 사용하여 부분 문자열 추출, 치환, 대소문자 변환을 수행한다
2. `printf`를 사용하여 패딩, 정렬, 타입 안전 포맷을 포함한 형식화된 출력을 생성한다
3. `tr`을 사용하여 문자 변환(transliteration)과 삭제를 수행한다
4. 구분자 모드와 문자 범위 모드로 `cut`을 사용하여 구조화된 텍스트에서 필드를 추출한다
5. `paste`와 관계형(relational) `join`을 사용하여 파일을 열(column) 단위로 병합한다
6. `column` 명령어로 표 형식 출력을 정렬한다
7. `jq` 필터, 조건문, 파이프라인을 사용하여 JSON 데이터를 조회하고 변환한다
8. 내장 도구와 외부 도구를 결합하여 다단계 텍스트 처리 파이프라인을 구축한다

---

셸 스크립트는 로그 파일 파싱, CSV 데이터에서 필드 추출, 설정 형식 변환, 보고서 생성 등 텍스트 처리를 끊임없이 수행합니다. Bash는 간단한 작업을 위한 내장 문자열 연산을 제공하지만, 실무 파이프라인에서는 `tr`, `cut`, `jq`, `printf` 같은 도구와 함께 조합해야 합니다. 내장 확장(expansion)을 쓸지 외부 도구를 쓸지 판단하는 능력이 빠르고 가독성 좋은 스크립트를 작성하는 핵심입니다.

## 1. 내장 문자열 연산 복습

외부 도구를 탐색하기 전에, bash의 강력한 내장 문자열 연산을 복습해보겠습니다.

### 1.1 매개변수 확장(Parameter Expansion) 빠른 참조

```bash
#!/bin/bash

text="Hello World"

# 길이
echo "${#text}"  # 출력: 11

# 부분 문자열 추출
echo "${text:0:5}"   # 출력: Hello
echo "${text:6}"     # 출력: World
echo "${text: -5}"   # 출력: World (- 앞의 공백 주의)

# 시작부터 제거 (최단 일치)
filename="path/to/file.txt"
echo "${filename#*/}"    # 출력: to/file.txt

# 시작부터 제거 (최장 일치)
echo "${filename##*/}"   # 출력: file.txt

# 끝에서 제거 (최단 일치)
echo "${filename%.*}"    # 출력: path/to/file

# 끝에서 제거 (최장 일치)
echo "${filename%%/*}"   # 출력: path
```

### 1.2 문자열 치환

```bash
#!/bin/bash

text="foo bar foo baz foo"

# 첫 번째 일치 치환
echo "${text/foo/FOO}"    # 출력: FOO bar foo baz foo

# 모든 일치 치환
echo "${text//foo/FOO}"   # 출력: FOO bar FOO baz FOO

# 첫 번째 일치 제거
echo "${text/foo}"        # 출력:  bar foo baz foo

# 모든 일치 제거
echo "${text//foo}"       # 출력:  bar  baz

# 시작 부분 치환
echo "${text/#foo/START}" # 출력: START bar foo baz foo

# 끝 부분 치환
text2="foo bar foo"
echo "${text2/%foo/END}"  # 출력: foo bar END
```

### 1.3 대소문자 변환

```bash
#!/bin/bash

text="Hello World"

# 소문자로 변환
echo "${text,,}"          # 출력: hello world
echo "${text,}"           # 출력: hello World (첫 문자만)

# 대문자로 변환
echo "${text^^}"          # 출력: HELLO WORLD
echo "${text^}"           # 출력: Hello World (첫 문자만)

# 대소문자 토글 (첫 문자)
echo "${text~}"

# 대소문자 토글 (모든 문자)
echo "${text~~}"
```

### 1.4 문자열 연결과 반복

```bash
#!/bin/bash

# 연결
first="Hello"
last="World"
full="$first $last"
echo "$full"  # 출력: Hello World

# 변수에 추가
message="Hello"
message+=" World"
echo "$message"  # 출력: Hello World

# 문자열 반복 (printf 사용)
repeat_string() {
    local string=$1
    local count=$2
    printf "%${count}s" | tr ' ' "$string"
}

echo "$(repeat_string '=' 40)"  # 출력: ========================================

# 대안: bash 루프 사용
repeat_string2() {
    local string=$1
    local count=$2
    local result=""
    for ((i=0; i<count; i++)); do
        result+="$string"
    done
    echo "$result"
}

echo "$(repeat_string2 '-' 20)"  # 출력: --------------------
```

### 1.5 문자열 비교

```bash
#!/bin/bash

str1="hello"
str2="world"

# 동등
[[ $str1 == $str2 ]] && echo "Equal" || echo "Not equal"

# 부등
[[ $str1 != $str2 ]] && echo "Different"

# 사전순 비교
[[ $str1 < $str2 ]] && echo "$str1 comes before $str2"
[[ $str1 > $str2 ]] && echo "$str1 comes after $str2"

# 비어있는지 확인
[[ -z $str1 ]] && echo "Empty" || echo "Not empty"

# 비어있지 않은지 확인
[[ -n $str1 ]] && echo "Not empty"

# 패턴 매칭
[[ $str1 == h* ]] && echo "Starts with h"
[[ $str1 == *o ]] && echo "Ends with o"
```

## 2. printf 포매팅

`printf` 명령은 강력한 문자열 포매팅 기능을 제공합니다.

### 2.1 기본 형식 지정자

```bash
#!/bin/bash

# 문자열
printf "%s\n" "Hello World"

# 정수
printf "%d\n" 42

# 부동소수점
printf "%f\n" 3.14159
printf "%.2f\n" 3.14159  # 출력: 3.14

# 16진수
printf "%x\n" 255   # 출력: ff
printf "%X\n" 255   # 출력: FF

# 8진수
printf "%o\n" 64    # 출력: 100

# 문자 (ASCII)
printf "%c\n" 65    # 출력: A
```

### 2.2 너비와 정밀도

```bash
#!/bin/bash

# 최소 너비 (오른쪽 정렬)
printf "%10s\n" "Hello"     # 출력:      Hello

# 왼쪽 정렬
printf "%-10s\n" "Hello"    # 출력: Hello

# 0으로 패딩된 숫자
printf "%05d\n" 42          # 출력: 00042

# 부동소수점 정밀도
printf "%.3f\n" 3.14159     # 출력: 3.142

# 너비와 정밀도 함께
printf "%10.2f\n" 3.14159   # 출력:       3.14
```

### 2.3 포매팅된 테이블 만들기

```bash
#!/bin/bash

# 테이블 헤더 출력
printf "%-15s %-10s %10s\n" "Name" "Status" "Count"
printf "%-15s %-10s %10s\n" "===============" "==========" "=========="

# 데이터 행 출력
printf "%-15s %-10s %10d\n" "Alice" "Active" 42
printf "%-15s %-10s %10d\n" "Bob" "Inactive" 17
printf "%-15s %-10s %10d\n" "Charlie" "Active" 93

# 출력:
# Name            Status          Count
# =============== ==========     ==========
# Alice           Active             42
# Bob             Inactive           17
# Charlie         Active             93
```

### 2.4 변수로 printf

```bash
#!/bin/bash

# 포매팅된 문자열을 변수에 저장
printf -v timestamp "%(%Y-%m-%d %H:%M:%S)T" -1
echo "Current time: $timestamp"

# 복잡한 문자열 포맷
printf -v sql_query "SELECT * FROM %s WHERE id = %d" "users" 42
echo "$sql_query"

# CSV 줄 만들기
printf -v csv_line "%s,%d,%.2f" "Product A" 100 19.99
echo "$csv_line"
```

### 2.5 printf로 패턴 반복

```bash
#!/bin/bash

# 수평선 출력
printf '=%.0s' {1..50}
echo

# 포매팅된 구분자 출력
printf '%*s\n' 50 | tr ' ' '-'

# 진행 상황 막대 생성
create_progress_bar() {
    local percent=$1
    local width=50
    local filled=$((percent * width / 100))

    printf "["
    printf "%${filled}s" | tr ' ' '#'
    printf "%$((width - filled))s" | tr ' ' '-'
    printf "] %3d%%\n" "$percent"
}

create_progress_bar 75
# 출력: [#####################################---------------] 75%
```

### 2.6 실전 포매팅 예제

```bash
#!/bin/bash

# 통화 포맷
format_currency() {
    local amount=$1
    printf "$%'.2f\n" "$amount"
}

format_currency 1234567.89  # 출력: $1,234,567.89

# 파일 크기 포맷
format_size() {
    local size=$1
    local units=("B" "KB" "MB" "GB" "TB")
    local unit=0

    while ((size > 1024 && unit < 4)); do
        size=$((size / 1024))
        ((unit++))
    done

    printf "%.2f %s\n" "$size" "${units[$unit]}"
}

format_size 1048576  # 출력: 1.00 MB

# 기간 포맷 (초를 HH:MM:SS로)
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))

    printf "%02d:%02d:%02d\n" "$hours" "$minutes" "$secs"
}

format_duration 3665  # 출력: 01:01:05
```

## 3. tr 명령

`tr` (translate) 명령은 문자 단위 변환을 수행합니다.

### 3.1 문자 변환

```bash
#!/bin/bash

# 문자 변환
echo "hello" | tr 'a-z' 'A-Z'  # 출력: HELLO
echo "WORLD" | tr 'A-Z' 'a-z'  # 출력: world

# 특정 문자 치환
echo "hello" | tr 'l' 'L'      # 출력: heLLo

# 다중 치환
echo "hello world" | tr 'elo' 'ELO'  # 출력: hELLO wOrLd

# 문자 회전 (ROT13)
echo "Hello World" | tr 'A-Za-z' 'N-ZA-Mn-za-m'  # 출력: Uryyb Jbeyq
```

### 3.2 문자 삭제

```bash
#!/bin/bash

# 특정 문자 삭제
echo "hello123world456" | tr -d '0-9'  # 출력: helloworld

# 공백 삭제
echo "  hello   world  " | tr -d ' '   # 출력: helloworld

# 줄바꿈 삭제 (줄 합치기)
cat multiline.txt | tr -d '\n'

# 모든 모음 제거
echo "Hello World" | tr -d 'aeiouAEIOU'  # 출력: Hll Wrld

# 구두점 제거
echo "Hello, World!" | tr -d '[:punct:]'  # 출력: Hello World
```

### 3.3 문자 압축

```bash
#!/bin/bash

# 반복된 문자 압축
echo "hello    world" | tr -s ' '     # 출력: hello world

# 여러 공백을 단일 공백으로 압축
echo "too    many     spaces" | tr -s '[:space:]' ' '

# 중복 빈 줄 제거
cat file.txt | tr -s '\n'

# 특정 문자 압축
echo "booook" | tr -s 'o'              # 출력: bok
```

### 3.4 보수 집합(Complement Set)

```bash
#!/bin/bash

# 영숫자만 유지 (나머지 모두 삭제)
echo "Hello, World! 123" | tr -cd '[:alnum:]'  # 출력: HelloWorld123

# 숫자만 유지
echo "Price: $19.99" | tr -cd '0-9'            # 출력: 1999

# 모든 비출력 문자 제거
cat file.txt | tr -cd '[:print:]\n'
```

### 3.5 문자 클래스

```bash
#!/bin/bash

# 사용 가능한 문자 클래스
# [:alnum:]  - 영숫자 문자
# [:alpha:]  - 알파벳 문자
# [:digit:]  - 숫자
# [:lower:]  - 소문자
# [:upper:]  - 대문자
# [:space:]  - 공백 문자
# [:punct:]  - 구두점 문자
# [:print:]  - 출력 가능 문자

# 예제
echo "Hello123" | tr '[:lower:]' '[:upper:]'   # 출력: HELLO123
echo "ABC def" | tr '[:upper:]' '[:lower:]'    # 출력: abc def
echo "Hello World" | tr -d '[:space:]'         # 출력: HelloWorld
echo "test@email.com" | tr -cd '[:alnum:]@.'   # 유효한 이메일 문자만 유지
```

### 3.6 실전 tr 예제

```bash
#!/bin/bash

# DOS/Windows 줄 끝을 Unix로 변환
tr -d '\r' < dos_file.txt > unix_file.txt

# 제목에서 URL slug 생성
echo "My Blog Post Title!" | tr '[:upper:] ' '[:lower:]-' | tr -cd '[:alnum:]-'
# 출력: my-blog-post-title

# 전화번호 숫자 추출
echo "Phone: (555) 123-4567" | tr -cd '0-9'    # 출력: 5551234567

# 텍스트에서 제어 문자 제거
cat file.txt | tr -d '[:cntrl:]'

# 파일명의 공백을 밑줄로 변환
filename="My Document.txt"
new_filename=$(echo "$filename" | tr ' ' '_')
echo "$new_filename"  # 출력: My_Document.txt
```

## 4. cut 명령

`cut` 명령은 각 줄에서 필드나 문자를 추출합니다.

### 4.1 문자 추출

```bash
#!/bin/bash

# 특정 문자 추출
echo "Hello World" | cut -c 1-5      # 출력: Hello
echo "Hello World" | cut -c 7-       # 출력: World
echo "Hello World" | cut -c -5       # 출력: Hello
echo "Hello World" | cut -c 1,7      # 출력: HW

# 여러 범위 추출
echo "abcdefghij" | cut -c 1-3,5-7   # 출력: abcefg
```

### 4.2 구분자로 필드 추출

```bash
#!/bin/bash

# CSV 파싱
echo "Alice,30,Engineer" | cut -d',' -f1     # 출력: Alice
echo "Alice,30,Engineer" | cut -d',' -f2     # 출력: 30
echo "Alice,30,Engineer" | cut -d',' -f1,3   # 출력: Alice,Engineer

# 탭 구분 (기본값)
echo -e "A\tB\tC\tD" | cut -f2               # 출력: B

# 파일에서 추출
cut -d':' -f1,3 /etc/passwd  # 사용자명과 UID 추출

# 여러 필드
echo "one:two:three:four:five" | cut -d':' -f2-4  # 출력: two:three:four
```

### 4.3 바이트 추출

```bash
#!/bin/bash

# 바이트 추출 (ASCII의 경우 문자와 유사)
echo "Hello" | cut -b 1-3   # 출력: Hel

# 바이너리 데이터나 멀티바이트 문자에 유용
# 참고: -b는 멀티바이트 UTF-8을 별도의 바이트로 처리
```

### 4.4 보수(Complement) (출력 억제)

```bash
#!/bin/bash

# 지정된 필드를 제외한 모든 필드 출력
echo "A,B,C,D,E" | cut -d',' -f1-3 --complement  # 출력: D,E
```

### 4.5 실전 cut 예제

```bash
#!/bin/bash

# 로그에서 IP 주소 추출
cut -d' ' -f1 access.log | sort -u

# /etc/passwd에서 사용자 목록 얻기
cut -d':' -f1 /etc/passwd

# 파일명에서 확장자 추출
echo "document.pdf" | rev | cut -d'.' -f1 | rev  # 출력: pdf

# 명령 출력 파싱
ps aux | tail -n +2 | cut -c 66-  # 명령 열 추출

# 타임스탬프에서 날짜 추출
echo "2024-02-13 15:30:45" | cut -d' ' -f1  # 출력: 2024-02-13

# 특정 열로 CSV 파싱
cut -d',' -f2,4,6 data.csv > extracted.csv
```

## 5. paste와 join

이 명령들은 여러 파일의 데이터를 병합합니다.

### 5.1 paste 명령

```bash
#!/bin/bash

# 파일을 나란히 병합
# file1.txt: A B C
# file2.txt: 1 2 3
paste file1.txt file2.txt
# 출력:
# A    1
# B    2
# C    3

# 사용자 정의 구분자
paste -d',' file1.txt file2.txt
# 출력:
# A,1
# B,2
# C,3

# 직렬 모드 (첫 번째 파일의 모든 줄, 그 다음 두 번째)
paste -s file1.txt file2.txt
# 출력:
# A    B    C
# 1    2    3

# 여러 파일 병합
paste file1.txt file2.txt file3.txt

# 여러 파일에서 CSV 생성
paste -d',' names.txt ages.txt cities.txt > output.csv
```

### 5.2 join 명령

```bash
#!/bin/bash

# 공통 필드로 파일 조인
# users.txt:    passwords.txt:
# 1 alice       1 pass123
# 2 bob         2 pass456
# 3 charlie     3 pass789

join users.txt passwords.txt
# 출력:
# 1 alice pass123
# 2 bob pass456
# 3 charlie pass789

# 사용자 정의 구분자
join -t',' users.csv passwords.csv

# 다른 필드로 조인
join -1 2 -2 1 file1.txt file2.txt  # file1 필드 2, file2 필드 1

# 외부 조인 (일치하지 않는 줄 포함)
join -a1 file1.txt file2.txt  # file1에서 일치하지 않는 것 포함
join -a2 file1.txt file2.txt  # file2에서 일치하지 않는 것 포함
join -a1 -a2 file1.txt file2.txt  # 완전 외부 조인

# 출력 형식 지정
join -o 1.1,1.2,2.2 file1.txt file2.txt
```

### 5.3 실전 예제

```bash
#!/bin/bash

# 이름과 성 결합
paste -d' ' first_names.txt last_names.txt > full_names.txt

# 열 데이터에서 테이블 생성
paste -d'|' col1.txt col2.txt col3.txt | column -t -s'|'

# 사용자 정보와 로그인 기록 조인
sort -k1 users.txt > users_sorted.txt
sort -k1 logins.txt > logins_sorted.txt
join -t',' users_sorted.txt logins_sorted.txt > user_logins.txt

# 데이터 전치 (행을 열로)
paste -s -d',' data.txt

# 번호가 매겨진 목록 생성
paste -d' ' <(seq 1 10) items.txt
# 출력:
# 1 item1
# 2 item2
# ...
```

## 6. column 명령

`column` 명령은 출력을 테이블로 포맷합니다.

### 6.1 기본 열 포매팅

```bash
#!/bin/bash

# 테이블로 자동 포맷
cat <<EOF | column -t
Name Age City
Alice 30 NYC
Bob 25 LA
Charlie 35 Chicago
EOF
# 출력:
# Name     Age  City
# Alice    30   NYC
# Bob      25   LA
# Charlie  35   Chicago

# 사용자 정의 구분자
echo -e "A,B,C\n1,2,3\n4,5,6" | column -t -s','
# 출력:
# A  B  C
# 1  2  3
# 4  5  6
```

### 6.2 행보다 열 먼저 채우기

```bash
#!/bin/bash

# 열 생성 (신문 스타일)
seq 1 20 | column -c 40
# 출력 (근사값):
# 1  5   9   13  17
# 2  6   10  14  18
# 3  7   11  15  19
# 4  8   12  16  20
```

### 6.3 JSON 포매팅

```bash
#!/bin/bash

# JSON을 테이블로 포맷 (-J 플래그가 있는 column 필요)
# 참고: GNU column은 -J 플래그가 있지만, BSD column은 없음

# 대안: jq로 데이터 준비 후 column 사용
jq -r '.[] | [.name, .age, .city] | @tsv' data.json | column -t
```

### 6.4 실전 예제

```bash
#!/bin/bash

# 명령 출력 포맷
ps aux | head -n 10 | column -t

# 정렬된 설정 파일 생성
cat > config.conf <<EOF
port=8080
host=localhost
debug=true
workers=4
EOF

cat config.conf | column -t -s'='
# 출력:
# port     8080
# host     localhost
# debug    true
# workers  4

# CSV 데이터를 깔끔하게 포맷
column -t -s',' data.csv

# 정렬된 메뉴 생성
cat <<MENU | column -t
1|Start|Launch the application
2|Stop|Terminate the application
3|Restart|Restart the application
4|Status|Check application status
MENU
```

## 7. jq로 JSON 처리

`jq`는 강력한 커맨드라인 JSON 프로세서입니다.

### 7.1 기본 필터

```bash
#!/bin/bash

# JSON 예쁘게 출력
echo '{"name":"Alice","age":30}' | jq '.'

# 필드 추출
echo '{"name":"Alice","age":30}' | jq '.name'  # 출력: "Alice"

# 중첩 필드 추출
echo '{"user":{"name":"Alice","age":30}}' | jq '.user.name'  # 출력: "Alice"

# 배열 요소
echo '["a","b","c"]' | jq '.[1]'  # 출력: "b"

# 배열 슬라이스
echo '[1,2,3,4,5]' | jq '.[2:4]'  # 출력: [3,4]
```

### 7.2 배열 연산

```bash
#!/bin/bash

# 배열 반복
echo '[1,2,3]' | jq '.[]'
# 출력:
# 1
# 2
# 3

# 배열에 맵
echo '[1,2,3]' | jq 'map(. * 2)'  # 출력: [2,4,6]

# 배열 필터
echo '[1,2,3,4,5]' | jq 'map(select(. > 2))'  # 출력: [3,4,5]

# 배열 길이
echo '[1,2,3,4,5]' | jq 'length'  # 출력: 5

# 배열 합계
echo '[1,2,3,4,5]' | jq 'add'  # 출력: 15

# 고유 값 얻기
echo '[1,2,2,3,3,3]' | jq 'unique'  # 출력: [1,2,3]

# 배열 정렬
echo '[3,1,2]' | jq 'sort'  # 출력: [1,2,3]
```

### 7.3 객체 연산

```bash
#!/bin/bash

# 키 얻기
echo '{"a":1,"b":2,"c":3}' | jq 'keys'  # 출력: ["a","b","c"]

# 값 얻기
echo '{"a":1,"b":2,"c":3}' | jq '.[]'
# 출력:
# 1
# 2
# 3

# 키 존재 확인
echo '{"a":1,"b":2}' | jq 'has("a")'  # 출력: true

# 필드 추가
echo '{"a":1}' | jq '. + {b: 2}'  # 출력: {"a":1,"b":2}

# 필드 삭제
echo '{"a":1,"b":2}' | jq 'del(.b)'  # 출력: {"a":1}
```

### 7.4 조건 논리

```bash
#!/bin/bash

# If-then-else
echo '{"age":25}' | jq 'if .age >= 18 then "adult" else "minor" end'

# 조건과 함께 select
echo '[{"name":"Alice","age":30},{"name":"Bob","age":17}]' | \
    jq '.[] | select(.age >= 18)'
# 출력: {"name":"Alice","age":30}

# 다중 조건
echo '[1,2,3,4,5]' | jq '.[] | select(. > 2 and . < 5)'
# 출력:
# 3
# 4
```

### 7.5 문자열 보간

```bash
#!/bin/bash

# 문자열 보간
echo '{"first":"Alice","last":"Smith"}' | \
    jq '"\(.first) \(.last)"'
# 출력: "Alice Smith"

# 객체 만들기
echo '{"name":"Alice"}' | \
    jq '{greeting: "Hello, \(.name)!"}'
# 출력: {"greeting":"Hello, Alice!"}
```

### 7.6 실전 jq 예제

```bash
#!/bin/bash

# API 응답 파싱
curl -s https://api.github.com/users/torvalds | jq '.name, .location, .public_repos'

# 객체 배열에서 특정 필드 추출
jq '.users[] | {name, email}' users.json

# JSON에서 CSV 생성
jq -r '.[] | [.name, .age, .city] | @csv' data.json

# 필터 및 변환
jq '.users | map(select(.active == true) | {name, email})' data.json

# 필드별 그룹화
jq 'group_by(.category)' items.json

# 중첩 구조 평탄화
jq '[.users[].orders[]] | length' data.json

# 필드 업데이트
jq '.users[] |= if .name == "Alice" then .status = "admin" else . end' data.json

# JSON 파일 병합
jq -s '.[0] * .[1]' file1.json file2.json

# 사용자 정의 들여쓰기로 예쁘게 출력
jq --indent 4 '.' data.json

# 원시 출력 (따옴표 없음)
jq -r '.name' data.json

# 압축 출력
jq -c '.' data.json
```

## 8. yq로 YAML 처리

`yq`는 jq와 유사한 YAML 프로세서입니다 (참고: yq라는 이름의 도구가 여러 개 있으며, 예제는 mikefarah/yq 사용).

### 8.1 기본 연산

```bash
#!/bin/bash

# 값 읽기
yq '.database.host' config.yaml

# 중첩 배열 읽기
yq '.servers[0].name' config.yaml

# 모든 배열 요소 읽기
yq '.servers[].name' config.yaml

# 키 얻기
yq 'keys' config.yaml
```

### 8.2 YAML 수정

```bash
#!/bin/bash

# 값 업데이트
yq '.database.port = 5432' config.yaml

# 새 필드 추가
yq '.newfield = "value"' config.yaml

# 필드 삭제
yq 'del(.oldfield)' config.yaml

# 제자리 업데이트
yq -i '.database.host = "localhost"' config.yaml

# YAML 파일 병합
yq eval-all 'select(fileIndex == 0) * select(fileIndex == 1)' file1.yaml file2.yaml
```

### 8.3 형식 변환

```bash
#!/bin/bash

# YAML을 JSON으로
yq -o=json '.' config.yaml

# JSON을 YAML로
yq -P '.' data.json

# YAML 예쁘게 출력
yq '.' config.yaml
```

### 8.4 실전 예제

```bash
#!/bin/bash

# 데이터베이스 자격 증명 추출
DB_HOST=$(yq '.database.host' config.yaml)
DB_PORT=$(yq '.database.port' config.yaml)
DB_NAME=$(yq '.database.name' config.yaml)

# 설정 업데이트
yq -i ".app.version = \"$NEW_VERSION\"" config.yaml
yq -i ".app.updated_at = \"$(date -Iseconds)\"" config.yaml

# YAML 구문 검증
if yq '.' config.yaml > /dev/null 2>&1; then
    echo "Valid YAML"
else
    echo "Invalid YAML"
fi

# 모든 서비스 포트 추출
yq '.services[].port' docker-compose.yml

# YAML에서 환경 파일 만들기
yq -o=props '.env' config.yaml > .env
```

## 9. 실전 텍스트 처리 파이프라인

### 9.1 로그 분석

```bash
#!/bin/bash

# 타임스탬프와 함께 에러 메시지 추출
grep ERROR app.log | cut -d' ' -f1-2,4- | sort | uniq -c

# 유형별 에러 카운트
grep ERROR app.log | cut -d':' -f3 | sort | uniq -c | sort -rn

# 접근 로그의 상위 10개 IP 주소
cut -d' ' -f1 access.log | sort | uniq -c | sort -rn | head -10

# 로그 항목 파싱 및 재포맷
awk -F'[\\[\\]]' '{print $1, $2, $3}' access.log | \
    column -t > formatted.log
```

### 9.2 데이터 변환

```bash
#!/bin/bash

# CSV를 JSON으로 변환
csv_to_json() {
    local csv_file=$1

    # 헤더 읽기
    IFS=',' read -ra headers < "$csv_file"

    # 데이터 행 처리
    tail -n +2 "$csv_file" | while IFS=',' read -ra values; do
        echo "{"
        for i in "${!headers[@]}"; do
            printf '  "%s": "%s"' "${headers[$i]}" "${values[$i]}"
            [[ $i -lt $((${#headers[@]} - 1)) ]] && echo ","
        done
        echo "\n}"
    done | jq -s '.'
}

# 필드명 변환
jq 'map({username: .user, email_address: .email, full_name: "\(.first) \(.last)"})' \
    input.json > output.json

# 데이터 피벗 (행을 열로)
paste -s -d',' data.txt
```

### 9.3 보고서 생성

```bash
#!/bin/bash

# 시스템 보고서 생성
generate_report() {
    cat <<EOF | column -t -s'|'
Metric|Value|Unit
---|---|---
CPU Usage|$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')%|
Memory Used|$(free -m | awk 'NR==2{print $3}')|MB
Disk Usage|$(df -h / | awk 'NR==2{print $5}')|
Uptime|$(uptime -p)|
Load Average|$(uptime | awk -F'load average:' '{print $2}')|
EOF
}

# CSV에서 마크다운 테이블 생성
csv_to_markdown() {
    local csv=$1

    # 헤더
    head -1 "$csv" | tr ',' '|' | sed 's/^/|/' | sed 's/$/|/'

    # 구분자
    head -1 "$csv" | tr ',' '|' | sed 's/[^|]/-/g' | sed 's/^/|/' | sed 's/$/|/'

    # 데이터
    tail -n +2 "$csv" | tr ',' '|' | sed 's/^/|/' | sed 's/$/|/'
}
```

## 연습 문제

### 문제 1: 고급 CSV 프로세서

다음 기능을 가진 스크립트 생성:
- 헤더 행이 있는 CSV 파일 읽기
- 각 행 검증 (올바른 필드 수, 데이터 타입)
- 열 값에 기반한 행 필터링 지원 (예: age > 30)
- 다중 열로 정렬 지원
- 특정 열 선택 지원
- CSV, JSON 또는 포맷된 테이블로 출력
- 쉼표가 있는 따옴표로 묶인 필드를 올바르게 처리

### 문제 2: 로그 파서 및 분석기

다음 기능을 가진 로그 분석 도구 구축:
- 일반 로그 형식 파싱 (Apache, Nginx, syslog)
- 타임스탬프, 레벨, 메시지, 소스 추출
- 통계 생성 (에러율, 상위 에러, 시간 분포)
- ASCII 문자를 사용한 타임라인 시각화 생성
- 시간 범위, 레벨, 패턴으로 필터링 지원
- 마크다운 또는 HTML 형식으로 보고서 출력

### 문제 3: 설정 파일 변환기

다음 기능을 가진 도구 작성:
- JSON, YAML, TOML, INI, ENV 형식 간 변환
- 각 형식의 구문 검증
- 가능한 경우 주석 보존
- 중첩 구조 지원
- 배열 및 복잡한 타입 처리
- 커맨드라인을 통해 특정 값 추출/업데이트 가능
- 여러 설정 파일 병합 지원

### 문제 4: 텍스트 템플릿 엔진

다음 기능을 가진 템플릿 프로세서 구현:
- 플레이스홀더가 있는 템플릿 파일 읽기 (예: {{variable}})
- 조건문 지원: {{#if condition}}...{{/if}}
- 루프 지원: {{#each items}}...{{/each}}
- 인클루드 지원: {{> include file.txt}}
- JSON/YAML 파일 또는 환경에서 변수 읽기
- 필터 지원: {{variable|upper}}, {{variable|date}}
- 중첩 데이터 구조 처리

### 문제 5: 데이터 검증 프레임워크

다음 기능을 가진 검증 도구 생성:
- YAML 형식으로 검증 규칙 정의
- 규칙에 대해 CSV/JSON/YAML 데이터 검증
- 규칙 지원: required, type, range, pattern, length, custom
- 줄 번호와 필드명과 함께 검증 에러 보고
- 교차 필드 검증 지원 (예: end_date > start_date)
- 여러 형식으로 검증 보고서 생성
- 일반 문제 자동 수정 가능 (공백 제거, 대소문자 변환)

**이전**: [06_IO_and_Redirection.md](./06_IO_and_Redirection.md) | **다음**: [08_Regex_in_Bash.md](./08_Regex_in_Bash.md)
