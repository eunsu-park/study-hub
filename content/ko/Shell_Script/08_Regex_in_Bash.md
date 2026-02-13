# 레슨 08: Bash에서의 정규 표현식(Regular Expressions)

**난이도**: ⭐⭐⭐

**이전**: [07_String_Processing.md](./07_String_Processing.md) | **다음**: [09_Process_Management.md](./09_Process_Management.md)

---

## 1. Glob vs Regex

Bash 스크립팅에서 glob과 regex의 차이점을 이해하는 것은 매우 중요합니다.

### 1.1 근본적인 차이점

| 특징 | Glob | Regex |
|---------|------|-------|
| **목적** | 파일명 매칭 | 문자열 패턴 매칭 |
| **컨텍스트** | 파일 작업, case 문 | `[[ =~ ]]`, grep, sed, awk |
| **`*` 의미** | 0개 이상의 문자 | 이전 문자의 0개 이상 반복 |
| **`.` 의미** | 리터럴 점 | 임의의 단일 문자 |
| **`?` 의미** | 정확히 하나의 문자 | 이전 문자의 0개 또는 1개 |
| **문자 클래스** | `[abc]` | `[abc]` (동일) |
| **부정** | `[!abc]` | `[^abc]` |
| **앵커** | 없음 (암묵적) | `^` (시작), `$` (끝) |
| **그룹** | `{a,b}` (중괄호 확장) | `(a\|b)` (선택) |

### 1.2 Glob 예제

```bash
#!/bin/bash

# Glob은 파일명 매칭에 사용됩니다
ls *.txt              # .txt로 끝나는 모든 파일
ls test?.log          # test1.log, test2.log 등
ls [abc]*.txt         # a, b, c로 시작하는 파일
ls [!0-9]*            # 숫자로 시작하지 않는 파일
ls file{1,2,3}.txt    # file1.txt, file2.txt, file3.txt

# case 문에서
case $filename in
    *.txt)    echo "Text file" ;;
    *.jpg|*.png) echo "Image file" ;;
    test*)    echo "Test file" ;;
esac

# 조건문에서
if [[ $filename == *.txt ]]; then
    echo "Text file"
fi
```

### 1.3 Regex 예제

```bash
#!/bin/bash

# Regex는 문자열 매칭에 사용됩니다
[[ $string =~ ^[0-9]+$ ]] && echo "All digits"
[[ $email =~ ^[a-z]+@[a-z]+\.[a-z]+$ ]] && echo "Valid email pattern"

# grep과 함께
grep '^ERROR' logfile.txt        # ERROR로 시작하는 라인
grep 'test.*done' logfile.txt    # 'test' 다음에 'done'이 오는 라인

# sed와 함께
sed 's/[0-9]\+/NUM/g' file.txt   # 숫자를 NUM으로 변경
```

### 1.4 일반적인 혼동 지점

```bash
#!/bin/bash

# 틀림: =~에 glob 패턴 사용
[[ $str =~ *.txt ]]  # 이것은 리터럴 "*" 다음에 ".txt"를 매칭합니다!

# 올바름: ==에 glob 사용
[[ $str == *.txt ]]  # .txt로 끝나는 문자열 매칭

# 올바름: =~에 regex 사용
[[ $str =~ .*\.txt$ ]]  # Regex: .txt로 끝나는 모든 것

# Glob: *는 0개 이상의 문자를 의미
echo test* matches: test, test1, test123

# Regex: *는 이전 문자의 0개 이상 반복을 의미
[[ $str =~ test* ]]  # 매칭: tes, test, testt, testtt

# Glob: ?는 정확히 하나의 문자를 의미
ls file?.txt  # 매칭: file1.txt, fileA.txt

# Regex: ?는 이전 문자의 0개 또는 1개를 의미
[[ $str =~ tests? ]]  # 매칭: test, tests
```

## 2. =~ 연산자

`=~` 연산자는 Bash의 `[[ ]]` 구조에서 regex 매칭을 수행합니다.

### 2.1 기본 사용법

```bash
#!/bin/bash

# 간단한 패턴 매칭
string="hello123"

if [[ $string =~ [0-9] ]]; then
    echo "Contains a digit"
fi

# 앵커가 있는 패턴
if [[ $string =~ ^hello ]]; then
    echo "Starts with 'hello'"
fi

if [[ $string =~ [0-9]$ ]]; then
    echo "Ends with a digit"
fi

# 전체 문자열 매칭
if [[ $string =~ ^[a-z]+[0-9]+$ ]]; then
    echo "Letters followed by numbers"
fi
```

### 2.2 인용 동작

```bash
#!/bin/bash

string="test123"

# 틀림: 인용된 패턴은 리터럴이 됨
if [[ $string =~ "[0-9]+" ]]; then
    echo "Never matches - looks for literal '[0-9]+'"
fi

# 올바름: 인용되지 않은 패턴
if [[ $string =~ [0-9]+ ]]; then
    echo "Matches one or more digits"
fi

# 올바름: 패턴을 포함하는 변수 (인용됨)
pattern='[0-9]+'
if [[ $string =~ $pattern ]]; then
    echo "Matches - variable expansion is NOT quoted"
fi

# 중요: 이식성과 특수 문자를 위해 변수를 사용하세요
```

### 2.3 반환 값

```bash
#!/bin/bash

string="test123"

# 반환 값
if [[ $string =~ [0-9]+ ]]; then
    echo "Match found (returns 0)"
else
    echo "No match (returns 1)"
fi

# 반환 값을 직접 사용
validate_email() {
    [[ $1 =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]
    # 매칭되면 0, 매칭되지 않으면 1 반환
}

if validate_email "user@example.com"; then
    echo "Valid email"
fi
```

### 2.4 변수에 패턴 저장

```bash
#!/bin/bash

# 명확성과 재사용성을 위해 변수에 패턴 정의
readonly IP_PATTERN='^([0-9]{1,3}\.){3}[0-9]{1,3}$'
readonly EMAIL_PATTERN='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
readonly URL_PATTERN='^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

validate_ip() {
    [[ $1 =~ $IP_PATTERN ]]
}

validate_email() {
    [[ $1 =~ $EMAIL_PATTERN ]]
}

validate_url() {
    [[ $1 =~ $URL_PATTERN ]]
}

# 사용법
if validate_ip "192.168.1.1"; then
    echo "Valid IP"
fi
```

## 3. BASH_REMATCH

`BASH_REMATCH` 배열은 regex 매칭에서 캡처된 그룹을 저장합니다.

### 3.1 기본 캡처 그룹

```bash
#!/bin/bash

string="John Doe, age 30"
pattern='([A-Z][a-z]+) ([A-Z][a-z]+), age ([0-9]+)'

if [[ $string =~ $pattern ]]; then
    echo "Full match: ${BASH_REMATCH[0]}"
    echo "First name: ${BASH_REMATCH[1]}"
    echo "Last name: ${BASH_REMATCH[2]}"
    echo "Age: ${BASH_REMATCH[3]}"
fi

# 출력:
# Full match: John Doe, age 30
# First name: John
# Last name: Doe
# Age: 30
```

### 3.2 구조화된 데이터 추출

```bash
#!/bin/bash

# 로그 항목 파싱
log_entry="2024-02-13 14:30:45 [ERROR] Database connection failed"
pattern='^([0-9-]+) ([0-9:]+) \[([A-Z]+)\] (.+)$'

if [[ $log_entry =~ $pattern ]]; then
    date="${BASH_REMATCH[1]}"
    time="${BASH_REMATCH[2]}"
    level="${BASH_REMATCH[3]}"
    message="${BASH_REMATCH[4]}"

    echo "Date: $date"
    echo "Time: $time"
    echo "Level: $level"
    echo "Message: $message"
fi
```

### 3.3 중첩 그룹

```bash
#!/bin/bash

# URL 파싱
url="https://user:pass@example.com:8080/path/to/resource?key=value"
pattern='^(https?)://([^:]+):([^@]+)@([^:]+):([0-9]+)(/[^?]*)(\?.*)?$'

if [[ $url =~ $pattern ]]; then
    protocol="${BASH_REMATCH[1]}"
    username="${BASH_REMATCH[2]}"
    password="${BASH_REMATCH[3]}"
    host="${BASH_REMATCH[4]}"
    port="${BASH_REMATCH[5]}"
    path="${BASH_REMATCH[6]}"
    query="${BASH_REMATCH[7]}"

    echo "Protocol: $protocol"
    echo "Username: $username"
    echo "Password: $password"
    echo "Host: $host"
    echo "Port: $port"
    echo "Path: $path"
    echo "Query: $query"
fi
```

### 3.4 루프에서 여러 매칭

```bash
#!/bin/bash

# 텍스트에서 모든 이메일 주소 추출
text="Contact us at support@example.com or sales@example.com for more info."
pattern='[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

while [[ $text =~ $pattern ]]; do
    email="${BASH_REMATCH[0]}"
    echo "Found: $email"

    # 다음 매칭을 찾기 위해 매칭된 부분 제거
    text="${text#*"$email"}"
done

# 출력:
# Found: support@example.com
# Found: sales@example.com
```

### 3.5 실용적인 추출 함수

```bash
#!/bin/bash

# 문자열에서 키-값 쌍 추출
parse_key_value() {
    local text=$1
    declare -gA parsed_data

    local pattern='([a-zA-Z_][a-zA-Z0-9_]*)=([^,]+)'

    while [[ $text =~ $pattern ]]; do
        local key="${BASH_REMATCH[1]}"
        local value="${BASH_REMATCH[2]}"

        parsed_data[$key]="$value"

        # 매칭된 부분 제거
        text="${text#*"${BASH_REMATCH[0]}"}"
    done
}

# 사용법
data="name=Alice,age=30,city=NYC,role=admin"
parse_key_value "$data"

for key in "${!parsed_data[@]}"; do
    echo "$key: ${parsed_data[$key]}"
done
```

## 4. 확장 정규 표현식(Extended Regular Expressions)

ERE(확장 정규 표현식)는 더 강력한 패턴 매칭을 제공합니다.

### 4.1 문자 클래스

```bash
#!/bin/bash

# POSIX 문자 클래스
[[ $char =~ [[:alpha:]] ]]   # 알파벳 문자
[[ $char =~ [[:digit:]] ]]   # 숫자
[[ $char =~ [[:alnum:]] ]]   # 영숫자
[[ $char =~ [[:space:]] ]]   # 공백 문자
[[ $char =~ [[:punct:]] ]]   # 구두점
[[ $char =~ [[:upper:]] ]]   # 대문자
[[ $char =~ [[:lower:]] ]]   # 소문자
[[ $char =~ [[:xdigit:]] ]]  # 16진수 숫자

# 사용자 정의 문자 클래스
[[ $char =~ [aeiouAEIOU] ]]  # 모음
[[ $char =~ [^aeiouAEIOU] ]] # 자음 (부정)
[[ $char =~ [0-9a-fA-F] ]]   # 16진수 문자
```

### 4.2 수량자(Quantifiers)

```bash
#!/bin/bash

# 기본 수량자
[[ $str =~ a+ ]]      # 하나 이상의 'a'
[[ $str =~ a* ]]      # 0개 이상의 'a'
[[ $str =~ a? ]]      # 0개 또는 1개의 'a'

# 경계 수량자
[[ $str =~ a{3} ]]    # 정확히 3개의 'a'
[[ $str =~ a{3,} ]]   # 3개 이상의 'a'
[[ $str =~ a{3,5} ]]  # 3개에서 5개 사이의 'a'

# 실용적인 예제
[[ $str =~ ^[0-9]{3}-[0-9]{2}-[0-9]{4}$ ]]  # SSN 형식: 123-45-6789
[[ $str =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]  # IP 주소
[[ $str =~ ^[a-zA-Z]{2,20}$ ]]  # 이름: 2-20개의 문자
```

### 4.3 선택(Alternation)

```bash
#!/bin/bash

# 선택 (OR)
[[ $str =~ ^(yes|no|maybe)$ ]]
[[ $str =~ \.(jpg|png|gif)$ ]]
[[ $str =~ ^(http|https|ftp):// ]]

# 그룹과 함께
[[ $str =~ ^(Mr|Mrs|Ms|Dr)\. [A-Z][a-z]+ [A-Z][a-z]+$ ]]
# 매칭: Mr. John Smith, Dr. Jane Doe 등

# 복잡한 선택
file_pattern='.*\.(txt|log|conf|cfg|ini|yaml|yml|json|xml)$'
[[ $filename =~ $file_pattern ]]
```

### 4.4 그룹화(Grouping)

```bash
#!/bin/bash

# 캡처를 위한 그룹
pattern='(https?)://([^/]+)(/.*)?'
url="https://example.com/path/to/page"

if [[ $url =~ $pattern ]]; then
    protocol="${BASH_REMATCH[1]}"
    domain="${BASH_REMATCH[2]}"
    path="${BASH_REMATCH[3]}"
fi

# 수량자를 위한 그룹
[[ $str =~ ^(ab)+ ]]        # 매칭: ab, abab, ababab
[[ $str =~ ^([0-9]{3}-){2}[0-9]{4}$ ]]  # 전화번호: 555-123-4567

# 선택을 위한 그룹
[[ $str =~ ^(red|green|blue) (car|bike|boat)$ ]]
# 매칭: "red car", "green bike", "blue boat" 등
```

### 4.5 앵커(Anchors)

```bash
#!/bin/bash

# 시작과 끝 앵커
[[ $str =~ ^hello ]]   # 'hello'로 시작
[[ $str =~ world$ ]]   # 'world'로 끝남
[[ $str =~ ^test$ ]]   # 정확히 'test'

# 단어 경계 (grep과 함께, =~에서 직접 사용 안 됨)
echo "hello world" | grep -E '\bhello\b'  # 완전한 단어로 'hello' 매칭

# 실용적인 예제
[[ $str =~ ^# ]]       # 주석 라인 (#으로 시작)
[[ $str =~ ;$ ]]       # 세미콜론으로 끝남
[[ $str =~ ^$ ]]       # 빈 라인
[[ $str =~ ^[[:space:]]*$ ]]  # 공백 라인 (공백만 있음)
```

### 4.6 ERE vs BRE 비교

| 특징 | BRE (기본) | ERE (확장) |
|---------|-------------|----------------|
| 그룹화 | `\(\)` | `()` |
| 선택 | `\|` | `|` |
| 수량자 `+`, `?` | 지원 안 됨 | `+`, `?` |
| 경계 수량자 | `\{n,m\}` | `{n,m}` |
| 사용처 | grep, sed (기본) | grep -E, egrep, awk |
| Bash `=~` | ERE 사용 | 네이티브 |

```bash
#!/bin/bash

# BRE (grep 기본)
echo "test123" | grep '\([a-z]\+\)[0-9]\+'

# ERE (grep -E 또는 egrep)
echo "test123" | grep -E '([a-z]+)[0-9]+'

# Bash =~는 ERE 사용
[[ "test123" =~ ^([a-z]+)([0-9]+)$ ]]
```

## 5. 실용적인 검증 함수

### 5.1 이메일 검증

```bash
#!/bin/bash

#
# 이메일 주소 검증
#
# 반환값: 유효하면 0, 유효하지 않으면 1
#
validate_email() {
    local email=$1
    local pattern='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    [[ $email =~ $pattern ]]
}

# 테스트 케이스
test_email() {
    local email=$1
    if validate_email "$email"; then
        echo "✓ Valid: $email"
    else
        echo "✗ Invalid: $email"
    fi
}

test_email "user@example.com"      # ✓ Valid
test_email "user.name@example.co.uk"  # ✓ Valid
test_email "user+tag@example.com"  # ✓ Valid
test_email "invalid@"              # ✗ Invalid
test_email "@example.com"          # ✗ Invalid
test_email "user@example"          # ✗ Invalid
```

### 5.2 IPv4 주소 검증

```bash
#!/bin/bash

#
# IPv4 주소 검증 (범위 확인 포함)
#
validate_ipv4() {
    local ip=$1
    local pattern='^([0-9]{1,3}\.){3}[0-9]{1,3}$'

    # 형식 확인
    if ! [[ $ip =~ $pattern ]]; then
        return 1
    fi

    # 각 옥텟이 0-255인지 확인
    local IFS='.'
    read -ra octets <<< "$ip"

    for octet in "${octets[@]}"; do
        if ((octet < 0 || octet > 255)); then
            return 1
        fi
    done

    return 0
}

# 테스트 케이스
test_ip() {
    local ip=$1
    if validate_ipv4 "$ip"; then
        echo "✓ Valid: $ip"
    else
        echo "✗ Invalid: $ip"
    fi
}

test_ip "192.168.1.1"      # ✓ Valid
test_ip "10.0.0.1"         # ✓ Valid
test_ip "255.255.255.255"  # ✓ Valid
test_ip "256.1.1.1"        # ✗ Invalid (256 > 255)
test_ip "192.168.1"        # ✗ Invalid (불완전)
test_ip "192.168.1.1.1"    # ✗ Invalid (옥텟 수 초과)
```

### 5.3 날짜 형식 검증

```bash
#!/bin/bash

#
# YYYY-MM-DD 형식의 날짜 검증
#
validate_date() {
    local date=$1
    local pattern='^([0-9]{4})-([0-9]{2})-([0-9]{2})$'

    if ! [[ $date =~ $pattern ]]; then
        return 1
    fi

    local year="${BASH_REMATCH[1]}"
    local month="${BASH_REMATCH[2]}"
    local day="${BASH_REMATCH[3]}"

    # 월 검증
    if ((month < 1 || month > 12)); then
        return 1
    fi

    # 일 검증
    if ((day < 1 || day > 31)); then
        return 1
    fi

    # date 명령으로 추가 검증
    if ! date -d "$date" > /dev/null 2>&1; then
        return 1
    fi

    return 0
}

# 테스트 케이스
test_date() {
    local date=$1
    if validate_date "$date"; then
        echo "✓ Valid: $date"
    else
        echo "✗ Invalid: $date"
    fi
}

test_date "2024-02-13"  # ✓ Valid
test_date "2024-12-31"  # ✓ Valid
test_date "2024-02-30"  # ✗ Invalid (2월 30일은 존재하지 않음)
test_date "2024-13-01"  # ✗ Invalid (월 13)
test_date "24-02-13"    # ✗ Invalid (잘못된 형식)
```

### 5.4 URL 검증

```bash
#!/bin/bash

#
# URL 검증
#
validate_url() {
    local url=$1
    local pattern='^(https?|ftp)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^[:space:]]*)?$'

    [[ $url =~ $pattern ]]
}

# 더 포괄적인 URL 검증
validate_url_detailed() {
    local url=$1
    local pattern='^(https?|ftp)://(([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|localhost|([0-9]{1,3}\.){3}[0-9]{1,3})(:[0-9]{1,5})?(/[^[:space:]]*)?$'

    [[ $url =~ $pattern ]]
}

# 테스트 케이스
test_url() {
    local url=$1
    if validate_url_detailed "$url"; then
        echo "✓ Valid: $url"
    else
        echo "✗ Invalid: $url"
    fi
}

test_url "https://example.com"              # ✓ Valid
test_url "http://example.com/path/to/page"  # ✓ Valid
test_url "https://sub.example.com:8080/api" # ✓ Valid
test_url "ftp://files.example.com"          # ✓ Valid
test_url "https://localhost:3000"           # ✓ Valid
test_url "https://192.168.1.1:8080"         # ✓ Valid
test_url "htp://example.com"                # ✗ Invalid (프로토콜 오타)
test_url "https://example"                  # ✗ Invalid (TLD 없음)
```

### 5.5 시맨틱 버전 검증

```bash
#!/bin/bash

#
# 시맨틱 버전(semver) 검증
# 형식: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
#
validate_semver() {
    local version=$1
    local pattern='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$'

    [[ $version =~ $pattern ]]
}

# semver 구성 요소 추출
parse_semver() {
    local version=$1
    local pattern='^([0-9]+)\.([0-9]+)\.([0-9]+)(-([a-zA-Z0-9.-]+))?(\+([a-zA-Z0-9.-]+))?$'

    if [[ $version =~ $pattern ]]; then
        echo "Major: ${BASH_REMATCH[1]}"
        echo "Minor: ${BASH_REMATCH[2]}"
        echo "Patch: ${BASH_REMATCH[3]}"
        echo "Prerelease: ${BASH_REMATCH[5]}"
        echo "Build: ${BASH_REMATCH[7]}"
        return 0
    fi

    return 1
}

# 테스트 케이스
test_semver() {
    local version=$1
    if validate_semver "$version"; then
        echo "✓ Valid: $version"
        parse_semver "$version"
    else
        echo "✗ Invalid: $version"
    fi
    echo
}

test_semver "1.0.0"                    # ✓ Valid
test_semver "1.0.0-alpha"              # ✓ Valid
test_semver "1.0.0-alpha.1"            # ✓ Valid
test_semver "1.0.0+20240213"           # ✓ Valid
test_semver "1.0.0-beta+exp.sha.5114f85"  # ✓ Valid
test_semver "1.0"                      # ✗ Invalid (불완전)
test_semver "v1.0.0"                   # ✗ Invalid ('v' 접두사 있음)
```

### 5.6 포괄적인 검증 프레임워크

```bash
#!/bin/bash

# 검증 규칙
declare -A VALIDATORS=(
    [email]='validate_email'
    [ipv4]='validate_ipv4'
    [date]='validate_date'
    [url]='validate_url'
    [semver]='validate_semver'
)

# 범용 검증 함수
validate() {
    local type=$1
    local value=$2
    local validator="${VALIDATORS[$type]}"

    if [[ -z $validator ]]; then
        echo "Error: Unknown validator type: $type" >&2
        return 2
    fi

    if ! type "$validator" > /dev/null 2>&1; then
        echo "Error: Validator function not found: $validator" >&2
        return 2
    fi

    "$validator" "$value"
}

# 사용 예제
validate_input() {
    local field=$1
    local value=$2
    local type=$3

    if validate "$type" "$value"; then
        echo "✓ $field is valid"
        return 0
    else
        echo "✗ $field is invalid: $value" >&2
        return 1
    fi
}

# 예제: 사용자 입력 검증
validate_input "Email" "user@example.com" "email"
validate_input "IP Address" "192.168.1.1" "ipv4"
validate_input "Version" "2.1.0-beta" "semver"
```

## 6. grep 및 sed와 함께 Regex 사용

### 6.1 확장 Regex를 사용한 grep

```bash
#!/bin/bash

# grep -E로 확장 regex 사용
grep -E '^[0-9]+$' file.txt          # 숫자만 포함하는 라인
grep -E '(error|warning|critical)' log.txt  # 여러 패턴
grep -E '\b[A-Z]{3,}\b' file.txt     # 3개 이상의 대문자 단어

# 대소문자 구분 안 함
grep -iE 'error' log.txt

# 매칭 반전
grep -vE '^#' config.txt             # 주석이 아닌 라인

# 매칭 카운트
grep -cE 'pattern' file.txt

# 컨텍스트 표시
grep -E -A 3 -B 3 'ERROR' log.txt    # 전후 3줄
```

### 6.2 sed 패턴 매칭

```bash
#!/bin/bash

# regex를 사용한 기본 치환
sed 's/[0-9]\+/NUM/g' file.txt       # 숫자 치환

# 앵커된 패턴
sed 's/^#.*//' file.txt              # 주석 라인 제거
sed 's/[[:space:]]\+$//' file.txt    # 후행 공백 제거

# 그룹과 역참조
sed 's/\([0-9]\{3\}\)-\([0-9]\{2\}\)-\([0-9]\{4\}\)/(\1) \2-\3/' # SSN 형식화
# 123-45-6789 → (123) 45-6789

# 조건부 처리
sed '/pattern/s/old/new/' file.txt   # 매칭되는 라인에서만 치환
sed '/^#/d' file.txt                 # 주석 라인 삭제
```

### 6.3 다중 라인 매칭

```bash
#!/bin/bash

# grep 다중 라인 (GNU grep의 -Pzo와 함께)
grep -Pzo '(?s)function.*?\{.*?\}' code.js

# sed 다중 라인
sed -n '/start/,/end/p' file.txt     # 범위 출력

# awk 다중 라인
awk '/start/,/end/' file.txt
```

## 7. 성능 고려 사항

### 7.1 Regex 컴파일

```bash
#!/bin/bash

# 비효율적: 매 반복마다 Regex 컴파일
for item in "${items[@]}"; do
    if [[ $item =~ ^[0-9]+$ ]]; then
        process "$item"
    fi
done

# 효율적: 한 번 컴파일, 여러 번 사용
pattern='^[0-9]+$'
for item in "${items[@]}"; do
    if [[ $item =~ $pattern ]]; then
        process "$item"
    fi
done
```

### 7.2 재앙적 역추적(Catastrophic Backtracking) 회피

```bash
#!/bin/bash

# 위험: 재앙적 역추적을 일으킬 수 있음
# 패턴: (a+)+b
# 문자열: "aaaaaaaaaaaaaaaaaaaaaaaac" (끝에 'b' 없음)
# 이것은 지수 시간이 걸립니다!

bad_pattern='(a+)+b'
# 같은 내용에 중첩된 수량자 피하기

# 안전: 원자 그룹이나 소유 수량자 사용 (지원되는 경우)
# 또는 역추적을 피하도록 재구성
good_pattern='a+b'

# 일반 규칙: (.*)*,  (.+)+ 같은 패턴 피하기
```

### 7.3 간단한 연산 vs Regex

```bash
#!/bin/bash

# 간단한 문자열 연산이 regex보다 빠를 때

# 접두사 확인
# 느림:
[[ $str =~ ^prefix ]]

# 빠름:
[[ $str == prefix* ]]

# 접미사 확인
# 느림:
[[ $str =~ suffix$ ]]

# 빠름:
[[ $str == *suffix ]]

# 포함 확인
# 느림:
[[ $str =~ substring ]]

# 빠름:
[[ $str == *substring* ]]

# 패턴 매칭이 정말 필요할 때 regex 사용
# 리터럴 부분 문자열 확인에는 간단한 연산 사용
```

### 7.4 패턴 벤치마킹

```bash
#!/bin/bash

benchmark_regex() {
    local pattern=$1
    local test_string=$2
    local iterations=10000

    local start=$(date +%s%N)

    for ((i=0; i<iterations; i++)); do
        [[ $test_string =~ $pattern ]] > /dev/null
    done

    local end=$(date +%s%N)
    local elapsed=$(( (end - start) / 1000000 ))

    echo "Pattern: $pattern"
    echo "Time: ${elapsed}ms for $iterations iterations"
    echo "Avg: $((elapsed * 1000 / iterations))μs per match"
}

# 패턴 비교
benchmark_regex '^[0-9]+$' "12345"
benchmark_regex '[0-9]' "12345"
```

## 8. 일반 패턴 참조

### 8.1 패턴 라이브러리

| 패턴 | 설명 | 예제 |
|---------|-------------|---------|
| `^[0-9]+$` | 정수 | 123, 456 |
| `^[0-9]*\.[0-9]+$` | 소수 | 3.14, 0.5 |
| `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$` | 이메일 | user@example.com |
| `^https?://[^\s]+$` | URL | https://example.com |
| `^([0-9]{1,3}\.){3}[0-9]{1,3}$` | IPv4 | 192.168.1.1 |
| `^[0-9]{4}-[0-9]{2}-[0-9]{2}$` | 날짜 YYYY-MM-DD | 2024-02-13 |
| `^([01][0-9]|2[0-3]):[0-5][0-9]$` | 시간 HH:MM | 14:30 |
| `^[0-9]{3}-[0-9]{2}-[0-9]{4}$` | SSN | 123-45-6789 |
| `^\(\d{3}\) \d{3}-\d{4}$` | 전화번호 (미국) | (555) 123-4567 |
| `^/.*$` | Unix 경로 | /path/to/file |
| `^[a-fA-F0-9]{32}$` | MD5 해시 | 5d41402abc4b2a76b9719d911017c592 |
| `^[0-9a-f]{40}$` | SHA-1 해시 | aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d |
| `^[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}$` | IBAN | GB82WEST12345698765432 |

### 8.2 실용적인 패턴 예제

```bash
#!/bin/bash

# 신용카드 (간소화 - 체크섬 검증 안 함)
CC_PATTERN='^[0-9]{4}[[:space:]-]?[0-9]{4}[[:space:]-]?[0-9]{4}[[:space:]-]?[0-9]{4}$'

# 16진수 색상 코드
COLOR_PATTERN='^#[0-9a-fA-F]{6}$'

# MAC 주소
MAC_PATTERN='^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$'

# 사용자명 (영숫자, 밑줄, 하이픈, 3-16자)
USERNAME_PATTERN='^[a-zA-Z0-9_-]{3,16}$'

# 강력한 비밀번호 (최소 8자, 대문자, 소문자, 숫자, 특수문자)
PASSWORD_PATTERN='^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%^&*]).{8,}$'

# 도메인명
DOMAIN_PATTERN='^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$'

# 파일 확장자
EXT_PATTERN='\.(txt|log|conf|json|yaml|xml)$'
```

## 연습 문제

### 문제 1: 고급 입력 검증기
포괄적인 입력 검증 라이브러리를 만드세요:
- 필드당 여러 검증 규칙 지원 (예: 필수, 유형, 길이, 패턴)
- 사용자 정의 검증기 구현 (예: 비밀번호 강도, 신용카드, IBAN)
- 상세한 에러 메시지 반환 (무엇이 실패했고 왜 실패했는지)
- 조건부 검증 지원 (필드 A에 값이 있으면 필드 B 필수)
- 중첩된 데이터 구조 검증 (JSON과 같은 객체)
- 모든 에러가 포함된 검증 리포트 생성

### 문제 2: Regex를 사용한 로그 파서
다음 기능을 가진 로그 파서를 만드세요:
- 로그 형식 자동 감지 (Apache, Nginx, syslog, 사용자 정의)
- regex를 사용하여 타임스탬프, 레벨, 소스, 메시지 추출
- 다중 라인 로그 항목 처리 (스택 트레이스 등)
- 로그 형식 검증 및 잘못된 형식의 항목 보고
- 타임스탬프를 다른 형식으로 변환
- 복잡한 regex 패턴으로 로그 필터링
- 통계 생성 (상위 에러, 시간 분포)

### 문제 3: 데이터 정제기(Sanitizer)
데이터 정제 도구를 구현하세요:
- 다양한 컨텍스트(SQL, HTML, shell)에 대한 특수 문자 제거/이스케이프
- 전화번호 검증 및 정규화 (여러 형식)
- 이메일 주소 검증 및 정규화
- regex를 사용하여 민감한 데이터 제거 (SSN, 신용카드, API 키)
- 날짜/시간 검증 및 형식화
- URL 처리 (파싱, 검증, 정규화)
- 정제 리포트 생성

### 문제 4: 설정 파일 파서
범용 설정 파서를 만드세요:
- INI, YAML과 유사한 형식, 사용자 정의 key=value 형식 파싱
- 섹션, 중첩 키, 배열 지원
- regex를 사용하여 구문 검증
- 타입 변환으로 값 추출 (string, int, bool, array)
- 변수 치환 지원 (예: ${HOME}/path)
- 패턴에 대한 값 검증 (예: 포트는 1-65535여야 함)
- 포함 및 상속 지원

### 문제 5: 패턴 기반 라우터
URL/경로 라우터를 만드세요:
- 패턴으로 경로 정의 (예: /user/:id, /posts/:year/:month/:slug)
- regex와 BASH_REMATCH를 사용하여 경로 매개변수 추출
- 선택적 매개변수 지원 (/path/:id?, /path/:id*)
- regex 제약 조건 지원 (/user/:id([0-9]+))
- 우선순위로 경로 매칭
- 패턴과 매개변수로부터 URL 생성
- 쿼리 문자열 파싱 지원

**이전**: [07_String_Processing.md](./07_String_Processing.md) | **다음**: [09_Process_Management.md](./09_Process_Management.md)
