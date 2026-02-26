# 텍스트 처리

**이전**: [파일/디렉토리 관리](./03_File_Directory_Management.md) | **다음**: [권한과 소유권](./05_Permissions_Ownership.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 정규 표현식(regular expression)을 사용해 grep으로 파일에서 패턴을 검색한다
2. 텍스트 매칭을 위한 기본 및 확장 정규 표현식(extended regular expression)을 작성한다
3. cut으로 구조화된 데이터에서 특정 필드를 추출한다
4. sort와 uniq로 데이터를 정렬하고 중복을 제거한다
5. wc로 줄 수, 단어 수, 바이트 수를 센다
6. sed로 텍스트 치환과 행 조작을 수행한다
7. awk로 구조화된 텍스트 데이터를 처리하고 계산을 수행한다
8. 파이프(pipe)와 리다이렉션(redirection)을 사용해 다단계 데이터 처리 파이프라인을 구성한다

---

텍스트 처리는 리눅스가 진정으로 빛을 발하는 영역입니다. "한 가지 일을 잘하는 작은 도구들"이라는 유닉스 철학 덕분에 grep, sed, awk, sort 등의 유틸리티를 강력한 데이터 파이프라인으로 연결할 수 있습니다. 시스템 관리자들은 로그 분석, 지표 추출, 설정 파일 변환을 위해 이 도구들을 매일 사용합니다. 텍스트 처리를 마스터하면 리눅스 초보 사용자에서 고급 사용자로 도약할 수 있습니다.

## 1. grep - 텍스트 검색

grep은 파일에서 패턴을 검색하는 강력한 도구입니다.

### 기본 사용법

```bash
# 기본 검색
grep "pattern" file.txt

# 여러 파일 검색
grep "error" *.log

# 디렉토리 재귀 검색
grep -r "TODO" ./src/
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-i` | 대소문자 무시 (ignore case) |
| `-r`, `-R` | 재귀 검색 (recursive) |
| `-n` | 줄 번호 표시 (line number) |
| `-v` | 패턴 제외 (invert) |
| `-c` | 매칭 줄 수 (count) |
| `-l` | 파일명만 출력 (files with matches) |
| `-L` | 매칭 안 된 파일명 |
| `-w` | 단어 단위 매칭 (word) |
| `-A n` | 매칭 후 n줄 표시 (after) |
| `-B n` | 매칭 전 n줄 표시 (before) |
| `-C n` | 매칭 전후 n줄 표시 (context) |
| `-E` | 확장 정규표현식 |
| `-o` | 매칭 부분만 출력 |

### 옵션 사용 예제

```bash
# 대소문자 무시
grep -i "error" log.txt

# 줄 번호 표시
grep -n "function" script.js

# 매칭되지 않는 줄
grep -v "comment" code.py

# 매칭 줄 수
grep -c "import" *.py

# 파일명만 출력
grep -l "password" *.conf

# 단어 단위 매칭
grep -w "log" file.txt    # "log"만 매칭, "logging" 제외

# 전후 문맥 표시
grep -A 3 "ERROR" app.log    # 매칭 후 3줄
grep -B 2 "ERROR" app.log    # 매칭 전 2줄
grep -C 2 "ERROR" app.log    # 전후 2줄

# 재귀 검색 + 줄 번호
grep -rn "TODO" ./
```

---

## 2. 기본 정규표현식

| 패턴 | 설명 | 예시 |
|------|------|------|
| `.` | 임의의 한 문자 | `a.c` → abc, adc |
| `*` | 앞 문자 0회 이상 | `ab*c` → ac, abc, abbc |
| `^` | 줄 시작 | `^Error` → 줄 시작의 Error |
| `$` | 줄 끝 | `end$` → 줄 끝의 end |
| `[ ]` | 문자 클래스 | `[aeiou]` → 모음 |
| `[^ ]` | 부정 문자 클래스 | `[^0-9]` → 숫자 아닌 것 |
| `\` | 이스케이프 | `\.` → 실제 점 |

```bash
# 줄 시작이 Error
grep "^Error" log.txt

# 줄 끝이 ;
grep ";$" code.c

# a로 시작하고 t로 끝나는 3글자
grep "a.t" file.txt    # ant, art, act

# 숫자로 시작하는 줄
grep "^[0-9]" data.txt

# 빈 줄 찾기
grep "^$" file.txt

# 주석 줄 (#으로 시작)
grep "^#" config.conf
```

### 확장 정규표현식 (-E)

| 패턴 | 설명 | 예시 |
|------|------|------|
| `+` | 앞 문자 1회 이상 | `ab+c` → abc, abbc (ac 제외) |
| `?` | 앞 문자 0 또는 1회 | `colou?r` → color, colour |
| `|` | OR | `cat|dog` |
| `( )` | 그룹 | `(ab)+` → ab, abab |
| `{n}` | 정확히 n회 | `a{3}` → aaa |
| `{n,m}` | n~m회 | `a{2,4}` → aa, aaa, aaaa |

```bash
# 확장 정규표현식 사용
grep -E "error|warning|critical" log.txt

# IP 주소 패턴
grep -E "[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}" access.log

# 이메일 패턴 (간단)
grep -E "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" file.txt

# 전화번호 패턴
grep -E "[0-9]{3}-[0-9]{3,4}-[0-9]{4}" contacts.txt
```

---

## 3. cut - 필드 추출

### 기본 사용법

```bash
# 구분자로 필드 추출
cut -d'구분자' -f필드번호 file

# 문자 위치로 추출
cut -c시작-끝 file
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-d` | 구분자 지정 (delimiter) |
| `-f` | 필드 번호 (field) |
| `-c` | 문자 위치 (character) |

```bash
# 콜론 구분, 1번 필드 (사용자명)
cut -d':' -f1 /etc/passwd

# 여러 필드
cut -d':' -f1,3,6 /etc/passwd

# 필드 범위
cut -d',' -f2-4 data.csv

# 문자 위치
cut -c1-10 file.txt

# 탭 구분 (기본값)
cut -f2 file.tsv
```

예시 (/etc/passwd):
```bash
cat /etc/passwd | head -3
```
```
root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash
```

```bash
cut -d':' -f1,6 /etc/passwd | head -3
```
```
root:/root
daemon:/usr/sbin
ubuntu:/home/ubuntu
```

---

## 4. sort - 정렬

### 기본 사용법

```bash
# 기본 정렬 (알파벳순)
sort file.txt

# 역순 정렬
sort -r file.txt

# 숫자 정렬
sort -n numbers.txt

# 중복 제거
sort -u file.txt
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-r` | 역순 (reverse) |
| `-n` | 숫자 정렬 (numeric) |
| `-k` | 특정 필드로 정렬 (key) |
| `-t` | 구분자 지정 |
| `-u` | 중복 제거 (unique) |
| `-h` | 용량 단위 정렬 (human) |

```bash
# 숫자 정렬
sort -n scores.txt

# 역순 숫자 정렬
sort -rn scores.txt

# 2번째 필드로 정렬 (쉼표 구분)
sort -t',' -k2 data.csv

# 3번째 필드 숫자 정렬
sort -t':' -k3 -n /etc/passwd

# 용량 단위 정렬
du -h | sort -h

# 파일 크기 역순 정렬
ls -lh | sort -k5 -hr
```

---

## 5. uniq - 중복 제거

uniq는 **연속된** 중복만 처리하므로 보통 sort와 함께 사용합니다.

```bash
# 연속 중복 제거
uniq file.txt

# 중복 횟수 표시
uniq -c file.txt

# 중복된 줄만 표시
uniq -d file.txt

# 중복 안 된 줄만 표시
uniq -u file.txt
```

```bash
# sort와 함께 사용
sort file.txt | uniq

# 중복 횟수 계산 후 정렬
sort file.txt | uniq -c | sort -rn

# 자주 등장하는 IP 상위 10개
cat access.log | cut -d' ' -f1 | sort | uniq -c | sort -rn | head -10
```

---

## 6. wc - 카운트

```bash
# 줄, 단어, 바이트 수
wc file.txt
```

출력:
```
  100   500  3000 file.txt
   │     │     │
   │     │     └── 바이트 수
   │     └── 단어 수
   └── 줄 수
```

```bash
# 줄 수만
wc -l file.txt

# 단어 수만
wc -w file.txt

# 문자 수만
wc -c file.txt

# 여러 파일
wc -l *.txt

# 파이프와 함께
cat /etc/passwd | wc -l
```

---

## 7. sed - 스트림 편집

sed는 텍스트 변환 도구입니다.

### 기본 치환

```bash
# 문법: s/패턴/대체문자열/플래그
sed 's/old/new/' file.txt        # 각 줄의 첫 번째만
sed 's/old/new/g' file.txt       # 모든 occurrences (global)
sed 's/old/new/gi' file.txt      # 대소문자 무시
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-i` | 파일 직접 수정 (in-place) |
| `-e` | 여러 명령 |
| `-n` | 출력 억제 |

```bash
# 파일 직접 수정
sed -i 's/old/new/g' file.txt

# 백업하며 수정
sed -i.bak 's/old/new/g' file.txt

# 여러 치환
sed -e 's/a/A/g' -e 's/b/B/g' file.txt

# 특정 줄만 치환
sed '5s/old/new/' file.txt       # 5번째 줄만
sed '1,10s/old/new/g' file.txt   # 1-10번째 줄
```

### 줄 삭제

```bash
# 특정 줄 삭제
sed '5d' file.txt               # 5번째 줄 삭제
sed '1,5d' file.txt             # 1-5번째 줄 삭제
sed '/pattern/d' file.txt       # 패턴 포함 줄 삭제

# 빈 줄 삭제
sed '/^$/d' file.txt

# 주석 줄 삭제
sed '/^#/d' config.conf
```

### 줄 출력

```bash
# 특정 줄 출력
sed -n '5p' file.txt            # 5번째 줄만
sed -n '1,10p' file.txt         # 1-10번째 줄
sed -n '/pattern/p' file.txt    # 패턴 포함 줄

# 줄 번호와 함께
sed -n '=;p' file.txt
```

---

## 8. awk - 패턴 처리

awk는 텍스트 처리를 위한 프로그래밍 언어입니다.

### 기본 구조

```bash
awk 'pattern { action }' file
```

### 필드 변수

| 변수 | 설명 |
|------|------|
| `$0` | 전체 줄 |
| `$1` | 첫 번째 필드 |
| `$2` | 두 번째 필드 |
| `$NF` | 마지막 필드 |
| `NR` | 현재 줄 번호 |
| `NF` | 필드 개수 |

```bash
# 첫 번째 필드 출력
awk '{print $1}' file.txt

# 여러 필드
awk '{print $1, $3}' file.txt

# 구분자 지정
awk -F':' '{print $1, $6}' /etc/passwd

# 마지막 필드
awk '{print $NF}' file.txt

# 줄 번호와 함께
awk '{print NR, $0}' file.txt
```

### 조건부 출력

```bash
# 조건 필터링
awk '$3 > 100 {print $0}' data.txt

# 패턴 매칭
awk '/error/ {print $0}' log.txt

# 특정 필드 패턴
awk '$1 ~ /^192/ {print $0}' access.log

# 여러 조건
awk '$2 > 50 && $3 < 100 {print $1}' data.txt
```

### 계산

```bash
# 합계
awk '{sum += $1} END {print sum}' numbers.txt

# 평균
awk '{sum += $1; count++} END {print sum/count}' numbers.txt

# 최대값
awk 'BEGIN {max=0} $1 > max {max=$1} END {print max}' numbers.txt
```

### 포맷팅

```bash
# 포맷 출력
awk '{printf "%-10s %5d\n", $1, $2}' data.txt

# 헤더 추가
awk 'BEGIN {print "Name\tScore"} {print $1"\t"$2}' data.txt
```

---

## 9. 파이프와 리다이렉션

### 파이프 (|)

명령어 출력을 다른 명령어 입력으로 전달합니다.

```bash
# 명령어 연결
ls -l | grep ".txt"
cat file.txt | sort | uniq
ps aux | grep nginx | grep -v grep
```

### 출력 리다이렉션

| 기호 | 설명 |
|------|------|
| `>` | 파일로 출력 (덮어쓰기) |
| `>>` | 파일에 추가 |
| `2>` | 에러 출력 리다이렉션 |
| `2>&1` | 에러를 표준 출력으로 |
| `&>` | 표준+에러 모두 (bash) |

```bash
# 결과를 파일로
ls -l > filelist.txt

# 파일에 추가
echo "new line" >> file.txt

# 에러만 파일로
command 2> error.log

# 출력과 에러 모두
command > output.txt 2>&1
command &> all.log

# 에러 무시
command 2>/dev/null

# 모든 출력 무시
command > /dev/null 2>&1
```

### 입력 리다이렉션

```bash
# 파일에서 입력
sort < unsorted.txt

# Here Document
cat << EOF
여러 줄
텍스트
입력
EOF
```

---

## 10. 실습 예제

### 실습 1: 로그 분석

```bash
# 샘플 로그 생성
cat << 'EOF' > access.log
192.168.1.10 - - [23/Jan/2024:10:15:32] "GET /index.html" 200
192.168.1.20 - - [23/Jan/2024:10:15:33] "GET /api/users" 200
192.168.1.10 - - [23/Jan/2024:10:15:34] "POST /api/login" 401
192.168.1.30 - - [23/Jan/2024:10:15:35] "GET /style.css" 200
192.168.1.10 - - [23/Jan/2024:10:15:36] "GET /api/data" 500
192.168.1.20 - - [23/Jan/2024:10:15:37] "GET /index.html" 200
EOF

# 에러(4xx, 5xx) 찾기
grep -E " [45][0-9]{2}$" access.log

# IP별 요청 수
cut -d' ' -f1 access.log | sort | uniq -c | sort -rn

# 상태 코드별 통계
awk '{print $NF}' access.log | sort | uniq -c | sort -rn
```

### 실습 2: 사용자 정보 추출

```bash
# 일반 사용자만 (UID >= 1000)
awk -F':' '$3 >= 1000 {print $1, $6}' /etc/passwd

# 쉘별 사용자 수
cut -d':' -f7 /etc/passwd | sort | uniq -c | sort -rn

# /home 사용자 목록
grep "/home/" /etc/passwd | cut -d':' -f1
```

### 실습 3: 데이터 변환

```bash
# CSV 생성
cat << 'EOF' > data.csv
name,score,grade
Alice,95,A
Bob,82,B
Charlie,78,C
David,91,A
EOF

# 점수 합계
awk -F',' 'NR>1 {sum+=$2} END {print "Total:", sum}' data.csv

# 평균
awk -F',' 'NR>1 {sum+=$2; c++} END {print "Average:", sum/c}' data.csv

# A등급 학생
awk -F',' '$3=="A" {print $1}' data.csv

# 점수 내림차순 정렬
sort -t',' -k2 -rn data.csv | head -5
```

### 실습 4: 텍스트 변환

```bash
# 모든 소문자를 대문자로
cat file.txt | tr 'a-z' 'A-Z'

# 특정 단어 치환
sed 's/error/ERROR/g' log.txt

# 여러 치환
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt

# 빈 줄 제거
sed '/^$/d' file.txt

# 공백을 탭으로
sed 's/  */\t/g' file.txt
```

### 실습 5: 복합 파이프라인

```bash
# 가장 큰 파일 10개
find /var/log -type f -exec ls -l {} \; 2>/dev/null | \
  sort -k5 -rn | head -10

# 특정 프로세스의 메모리 사용량
ps aux | grep nginx | grep -v grep | \
  awk '{sum += $6} END {print sum/1024 " MB"}'

# 실시간 로그에서 에러만 필터링
tail -f /var/log/syslog | grep --line-buffered -i error
```

---

## 다음 단계

[권한과 소유권](./05_Permissions_Ownership.md)에서 파일 권한과 소유권 관리를 배워봅시다!
