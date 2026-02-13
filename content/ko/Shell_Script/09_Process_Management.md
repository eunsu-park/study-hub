# 레슨 09: 프로세스 관리 및 작업 제어(Process Management and Job Control)

**난이도**: ⭐⭐⭐

**이전**: [08_Regex_in_Bash.md](./08_Regex_in_Bash.md) | **다음**: [10_Error_Handling.md](./10_Error_Handling.md)

## 1. 백그라운드 프로세스(Background Processes)

셸 스크립트는 백그라운드에서 프로세스를 실행할 수 있어 병렬 실행과 성능 향상을 가능하게 합니다.

### 기본 백그라운드 실행

```bash
#!/bin/bash

# &로 명령을 백그라운드에서 실행
sleep 10 &
echo "Sleep started in background"

# 마지막 백그라운드 프로세스의 PID
SLEEP_PID=$!
echo "Sleep PID: $SLEEP_PID"

# 다른 작업 계속 수행
echo "Doing other work..."

# 백그라운드 프로세스가 완료될 때까지 대기
wait $SLEEP_PID
echo "Sleep completed"
```

### wait 명령

```bash
#!/bin/bash

# 여러 백그라운드 프로세스 시작
sleep 2 &
PID1=$!

sleep 3 &
PID2=$!

sleep 1 &
PID3=$!

echo "Started 3 background processes: $PID1, $PID2, $PID3"

# 모든 백그라운드 작업 대기
wait
echo "All processes completed"

# 특정 PID 대기
sleep 5 &
SPECIFIC_PID=$!
wait $SPECIFIC_PID
echo "Specific process $SPECIFIC_PID completed with exit code: $?"

# Bash 4.3+: 임의의 작업이 완료될 때까지 대기
if [ "${BASH_VERSINFO[0]}" -ge 4 ] && [ "${BASH_VERSINFO[1]}" -ge 3 ]; then
    sleep 2 &
    sleep 4 &
    sleep 1 &

    wait -n  # 다음 작업이 완료될 때까지 대기
    echo "First job completed"

    wait -n
    echo "Second job completed"

    wait
    echo "All remaining jobs completed"
fi
```

### 작업 제어 명령

```bash
#!/bin/bash

# 장시간 실행되는 프로세스 시작
sleep 100 &

# 모든 작업 나열
jobs
# 출력: [1]+ Running    sleep 100 &

# PID와 함께 나열
jobs -l
# 출력: [1]+ 12345 Running    sleep 100 &

# 실행 중인 작업만 나열
jobs -r

# 중지된 작업만 나열
jobs -s

# 작업을 포그라운드로 가져오기
# fg %1    # (대화형 셸에서 주석 해제)

# 작업을 백그라운드로 보내기 (중지된 경우)
# bg %1    # (대화형 셸에서 주석 해제)

# 작업 참조:
# %1       - 작업 번호 1
# %?sleep  - 명령에 "sleep"이 포함된 작업
# %%       - 현재 작업
# %+       - 현재 작업 (%%와 동일)
# %-       - 이전 작업

# 백그라운드 작업 종료
kill %1
```

### 작업 제어 예제

```bash
#!/bin/bash

# 작업 제어를 보여주는 함수
job_control_demo() {
    echo "Starting 3 jobs..."

    (sleep 5; echo "Job 1 done") &
    (sleep 3; echo "Job 2 done") &
    (sleep 7; echo "Job 3 done") &

    # 모든 작업 표시
    jobs

    # 작업 2를 특정하여 대기
    wait %2
    echo "Job 2 has completed"

    # 나머지 모두 대기
    wait
    echo "All jobs completed"
}

job_control_demo
```

## 2. 병렬 실행(Parallel Execution)

독립적인 작업을 병렬로 실행하면 스크립트를 크게 가속화할 수 있습니다.

### 기본 병렬 패턴

```bash
#!/bin/bash

# 순차 실행 (느림)
sequential() {
    for i in {1..5}; do
        sleep 1
        echo "Task $i completed"
    done
}

# 병렬 실행 (빠름)
parallel_basic() {
    for i in {1..5}; do
        (
            sleep 1
            echo "Task $i completed"
        ) &
    done
    wait
}

echo "Sequential (5 seconds):"
time sequential

echo -e "\nParallel (1 second):"
time parallel_basic
```

### 백그라운드 PID 추적

```bash
#!/bin/bash

# 배열에 PID 저장
pids=()

for i in {1..5}; do
    sleep $((RANDOM % 3 + 1)) &
    pids+=($!)
    echo "Started job $i with PID ${pids[-1]}"
done

# 각 PID를 대기하고 종료 상태 확인
for pid in "${pids[@]}"; do
    wait "$pid"
    status=$?
    echo "PID $pid exited with status $status"
done
```

### 동시성 제한

```bash
#!/bin/bash

# 최대 N개의 작업을 병렬로 실행
MAX_JOBS=3

run_with_limit() {
    local max_jobs=$1
    shift
    local jobs=("$@")

    local pids=()
    local count=0

    for job in "${jobs[@]}"; do
        # 한계에 도달하면 하나가 완료될 때까지 대기
        if [ "${#pids[@]}" -ge "$max_jobs" ]; then
            wait -n  # Bash 4.3+
            # 완료된 PID 제거
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    unset 'pids[$i]'
                fi
            done
            pids=("${pids[@]}")  # 배열 재색인
        fi

        # 새 작업 시작
        eval "$job" &
        pids+=($!)
        echo "Started job: $job (PID: $!)"
    done

    # 남은 작업 대기
    wait
    echo "All jobs completed"
}

# 사용 예제
jobs=(
    "sleep 2; echo 'Job 1 done'"
    "sleep 1; echo 'Job 2 done'"
    "sleep 3; echo 'Job 3 done'"
    "sleep 1; echo 'Job 4 done'"
    "sleep 2; echo 'Job 5 done'"
)

run_with_limit $MAX_JOBS "${jobs[@]}"
```

### xargs를 사용한 병렬 실행

```bash
#!/bin/bash

# xargs로 파일을 병렬 처리
process_file() {
    local file=$1
    echo "Processing $file..."
    sleep 1
    echo "$file processed"
}

export -f process_file

# 4개의 작업을 병렬로 실행
echo -e "file1.txt\nfile2.txt\nfile3.txt\nfile4.txt\nfile5.txt" | \
    xargs -P 4 -I {} bash -c 'process_file "{}"'

# 대안: parallel (GNU parallel 도구, 설치된 경우)
# seq 1 10 | parallel -j 4 'echo Processing {}; sleep 1'
```

### 병렬 처리 템플릿

```bash
#!/bin/bash

# 범용 병렬 프로세서
parallel_process() {
    local max_parallel=$1
    local processor_func=$2
    shift 2
    local items=("$@")

    local count=0
    local pids=()

    for item in "${items[@]}"; do
        # 동시성 제한
        while [ "${#pids[@]}" -ge "$max_parallel" ]; do
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}"
                    echo "Job for '$item' completed (PID: ${pids[$i]}, status: $?)"
                    unset 'pids[$i]'
                fi
            done
            pids=("${pids[@]}")
            sleep 0.1
        done

        # 새 작업 시작
        $processor_func "$item" &
        pids+=($!)
        ((count++))
    done

    # 남은 모든 작업 대기
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "Processed $count items"
}

# 예제 프로세서 함수
my_processor() {
    local item=$1
    sleep $((RANDOM % 3 + 1))
    echo "Processed: $item"
}

export -f my_processor

# 최대 5개 병렬로 20개 항목 처리
items=($(seq 1 20))
parallel_process 5 my_processor "${items[@]}"
```

## 3. 서브셸(Subshells)

서브셸은 자체 변수 스코프를 가진 격리된 실행 환경을 생성합니다.

### 서브셸 구문

```bash
#!/bin/bash

# 괄호는 서브셸을 생성합니다
VAR="outer"

(
    VAR="inner"
    echo "Inside subshell: $VAR"
)

echo "Outside subshell: $VAR"  # 여전히 "outer"

# 명령 치환도 서브셸을 생성합니다
result=$(
    VAR="command substitution"
    echo "$VAR"
)
echo "Result: $result"
echo "VAR is still: $VAR"  # 여전히 "outer"
```

### 서브셸 vs 명령 그룹화

```bash
#!/bin/bash

# 서브셸 ( ) - 별도 프로세스, 격리된 스코프
VAR="original"
( VAR="subshell"; cd /tmp; pwd )
echo "VAR: $VAR"       # "original"
echo "PWD: $PWD"       # 변경되지 않음

# 명령 그룹화 { } - 동일한 프로세스, 공유 스코프
VAR="original"
{ VAR="grouped"; echo "In group: $VAR"; }
echo "VAR: $VAR"       # "grouped"

# 참고: { }는 공백과 세미콜론/줄바꿈이 } 앞에 필요합니다
```

### 실용적인 서브셸 사용

```bash
#!/bin/bash

# 1. 임시 디렉토리 변경
(cd /tmp && ls -la)  # 이후 원래 디렉토리로 돌아옴

# 2. 임시 환경 변경
(
    export PATH="/custom/path:$PATH"
    export CUSTOM_VAR="value"
    ./my_program  # 수정된 환경 사용
)
# 여기서 환경 복원됨

# 3. 리다이렉션을 위한 그룹화
(
    echo "Log entry 1"
    echo "Log entry 2"
    echo "Log entry 3"
) >> logfile.txt

# 4. 백그라운드 작업 격리
for i in {1..3}; do
    (
        # 각 반복은 격리된 환경을 가짐
        ITERATION=$i
        sleep 1
        echo "Iteration $ITERATION complete"
    ) &
done
wait

# 5. 그룹화된 명령이 있는 파이프라인
(echo "line 1"; echo "line 2"; echo "line 3") | grep "line 2"
```

### 변수 스코프 함의

```bash
#!/bin/bash

# 서브셸에서 수정된 변수는 유지되지 않음
counter=0

while read line; do
    ((counter++))  # while이 서브셸에 있으면 작동하지 않습니다!
done < <(seq 1 10)

echo "Counter: $counter"  # 10 (프로세스 치환은 while에 대한 서브셸 회피)

# 이것은 서브셸을 생성합니다 (while에 파이프)
counter=0
seq 1 10 | while read line; do
    ((counter++))
done
echo "Counter: $counter"  # 0 (서브셸이 변경 사항을 격리)

# 해결책 1: 프로세스 치환 사용 (위에 표시됨)
# 해결책 2: here 문자열 또는 리다이렉션 사용
# 해결책 3: 파일 디스크립터 사용
```

## 4. 시그널(Signals)

시그널은 프로세스에 이벤트를 알리는 소프트웨어 인터럽트입니다.

### 일반적인 시그널

| 시그널 | 번호 | 설명 | 기본 동작 |
|--------|--------|-------------|----------------|
| SIGHUP | 1 | 행업 (터미널 닫힘) | 종료 |
| SIGINT | 2 | 인터럽트 (Ctrl+C) | 종료 |
| SIGQUIT | 3 | 종료 (Ctrl+\) | 종료 + 코어 덤프 |
| SIGKILL | 9 | 킬 (캐치 불가) | 종료 |
| SIGTERM | 15 | 종료 요청 | 종료 |
| SIGSTOP | 19 | 중지 (캐치 불가) | 중지 |
| SIGCONT | 18 | 중지된 경우 계속 | 계속 |
| SIGUSR1 | 10 | 사용자 정의 시그널 1 | 종료 |
| SIGUSR2 | 12 | 사용자 정의 시그널 2 | 종료 |
| SIGPIPE | 13 | 깨진 파이프 | 종료 |
| SIGCHLD | 17 | 자식 프로세스 변경됨 | 무시 |
| SIGALRM | 14 | 타이머 만료 | 종료 |

### 시그널 전송

```bash
#!/bin/bash

# 백그라운드 프로세스 시작
sleep 100 &
PID=$!

# 프로세스에 시그널 전송
kill -SIGTERM $PID  # 정중한 종료 요청
# kill -15 $PID     # 동일, 번호 사용

# kill -SIGKILL $PID  # 강제 종료 (캐치 불가)
# kill -9 $PID        # 동일, 번호 사용

# 프로세스가 실행 중인지 확인 (시그널 0)
if kill -0 $PID 2>/dev/null; then
    echo "Process $PID is running"
else
    echo "Process $PID is not running"
fi

# 프로세스 그룹에 시그널 전송
# kill -TERM -$$  # 전체 프로세스 그룹 종료
```

### 시그널 나열

```bash
#!/bin/bash

# 모든 시그널 나열
kill -l

# 번호로부터 시그널 이름 가져오기
kill -l 9   # KILL

# 이름으로부터 시그널 번호 가져오기
kill -l TERM  # 15
```

## 5. trap 명령

`trap` 명령은 스크립트가 시그널을 캐치하고 처리할 수 있게 합니다.

### 기본 trap 구문

```bash
#!/bin/bash

# trap 구문: trap 'commands' SIGNAL [SIGNAL...]

# Ctrl+C (SIGINT) 캐치
trap 'echo "Caught SIGINT (Ctrl+C)! Exiting..."; exit 1' INT

echo "Press Ctrl+C to trigger the trap..."
sleep 30
echo "Completed normally"
```

### 여러 시그널 트랩

```bash
#!/bin/bash

cleanup() {
    echo "Cleanup function called by signal: $1"
    # 여기서 정리 수행
    exit 0
}

# 여러 시그널 트랩
trap 'cleanup SIGINT' INT
trap 'cleanup SIGTERM' TERM
trap 'cleanup SIGHUP' HUP

echo "Script running (PID: $$)..."
echo "Try: kill -TERM $$"
while true; do
    sleep 1
done
```

### 시그널 무시

```bash
#!/bin/bash

# SIGINT 무시 (빈 문자열)
trap '' INT

echo "Try Ctrl+C - it won't work!"
sleep 5

# 기본 동작으로 재설정
trap - INT

echo "Now Ctrl+C will work again"
sleep 5
```

### EXIT에 대한 트랩

```bash
#!/bin/bash

# EXIT 의사 시그널: 스크립트 종료 시 트리거됨 (모든 이유)
trap 'echo "Script exiting..."' EXIT

echo "Starting script"
sleep 2
echo "Ending script"

# EXIT 트랩은 스크립트가 어떻게 종료되든 여기서 실행됩니다
```

### trap을 사용한 디버깅

```bash
#!/bin/bash

# DEBUG 의사 시그널: 각 명령 전에 실행됨
trap 'echo "Executing: $BASH_COMMAND"' DEBUG

echo "First command"
x=10
echo "x is $x"
((x++))
echo "x is now $x"

trap - DEBUG  # DEBUG 트랩 비활성화
```

## 6. 정리 패턴(Cleanup Patterns)

적절한 정리는 스크립트가 임시 파일, 락 파일 또는 고아 프로세스를 남기지 않도록 보장합니다.

### 임시 파일 정리

```bash
#!/bin/bash

# 임시 파일 생성
TMPFILE=$(mktemp) || exit 1

# 종료 시 정리 보장
trap 'rm -f "$TMPFILE"' EXIT

echo "Using temp file: $TMPFILE"

# 임시 파일 사용
echo "data" > "$TMPFILE"
cat "$TMPFILE"

# 종료 시 자동으로 정리됩니다
```

### 포괄적인 정리

```bash
#!/bin/bash

# 정리 함수
cleanup() {
    local exit_code=$?

    echo "Performing cleanup..."

    # 임시 파일 제거
    [ -n "$TMPFILE" ] && [ -f "$TMPFILE" ] && rm -f "$TMPFILE"
    [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ] && rm -rf "$TMPDIR"

    # 락 파일 해제
    [ -n "$LOCKFILE" ] && [ -f "$LOCKFILE" ] && rm -f "$LOCKFILE"

    # 자식 프로세스 종료
    [ -n "$WORKER_PID" ] && kill "$WORKER_PID" 2>/dev/null

    # 상태 복원
    [ -n "$ORIGINAL_DIR" ] && cd "$ORIGINAL_DIR"

    echo "Cleanup complete"
    exit "$exit_code"
}

# 트랩 설정
trap cleanup EXIT INT TERM

# 원래 디렉토리 기억
ORIGINAL_DIR=$PWD

# 임시 리소스 생성
TMPFILE=$(mktemp)
TMPDIR=$(mktemp -d)
LOCKFILE="/tmp/myscript.lock"

echo "Resources created:"
echo "  TMPFILE: $TMPFILE"
echo "  TMPDIR: $TMPDIR"
echo "  LOCKFILE: $LOCKFILE"

# 작업 시뮬레이션
echo "Working..."
sleep 2

# 자동으로 정리됩니다
```

### 락 파일 패턴

```bash
#!/bin/bash

LOCKFILE="/var/lock/myscript.lock"

# 락 획득
acquire_lock() {
    if [ -e "$LOCKFILE" ]; then
        echo "Another instance is running (lock file exists)"
        exit 1
    fi

    # 우리 PID로 락 파일 생성
    echo $$ > "$LOCKFILE"

    # 정리 보장
    trap 'rm -f "$LOCKFILE"; exit' EXIT INT TERM
}

# 대안: mkdir로 원자적 락
acquire_lock_atomic() {
    local lockdir="/var/lock/myscript.lock.d"

    if mkdir "$lockdir" 2>/dev/null; then
        trap 'rmdir "$lockdir"; exit' EXIT INT TERM
        return 0
    else
        echo "Another instance is running"
        return 1
    fi
}

acquire_lock

echo "Lock acquired, doing work..."
sleep 5
echo "Done"
```

### 멱등적 정리(Idempotent Cleanup)

```bash
#!/bin/bash

# 안전하게 여러 번 호출할 수 있는 정리
cleanup() {
    # 플래그를 사용하여 정리된 것을 추적
    [ -n "$CLEANUP_DONE" ] && return
    CLEANUP_DONE=1

    echo "Running cleanup..."

    # 제거하기 전에 확인
    if [ -n "$TMPFILE" ] && [ -f "$TMPFILE" ]; then
        rm -f "$TMPFILE"
        echo "Removed $TMPFILE"
    fi

    # 실행 중인 경우에만 프로세스 종료
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill "$CHILD_PID"
        wait "$CHILD_PID" 2>/dev/null
        echo "Killed process $CHILD_PID"
    fi
}

trap cleanup EXIT INT TERM

TMPFILE=$(mktemp)
sleep 100 &
CHILD_PID=$!

echo "Press Ctrl+C to test cleanup..."
sleep 30
```

## 7. 코프로세스(Coprocesses) (coproc)

Bash 4.0+는 양방향 통신을 위한 코프로세스를 지원합니다.

### 기본 코프로세스

```bash
#!/bin/bash

# 코프로세스 시작
coproc BC { bc; }

# 코프로세스에 쓰기 (stdin)
echo "5 + 3" >&"${BC[1]}"

# 코프로세스에서 읽기 (stdout)
read -u "${BC[0]}" result
echo "Result: $result"

# 코프로세스 닫기
eval "exec ${BC[1]}>&-"
wait $BC_PID
```

### 명명된 코프로세스

```bash
#!/bin/bash

# 명명된 코프로세스
coproc CALC { bc -l; }

# 계산 함수
calculate() {
    echo "$1" >&"${CALC[1]}"
    read -u "${CALC[0]}" result
    echo "$result"
}

# 계산기 사용
echo "sqrt(16) = $(calculate 'sqrt(16)')"
echo "10 / 3 = $(calculate '10 / 3')"
echo "e(1) = $(calculate 'e(1)')"

# 정리
eval "exec ${CALC[1]}>&-"
wait $CALC_PID
```

### 대화형 코프로세스 예제

```bash
#!/bin/bash

# 셸 코프로세스 시작
coproc SHELL { bash; }

# 코프로세스에서 명령 실행 함수
exec_in_coproc() {
    local cmd=$1
    echo "$cmd" >&"${SHELL[1]}"
    echo "echo '<<<END>>>'" >&"${SHELL[1]}"

    while read -u "${SHELL[0]}" line; do
        [ "$line" = "<<<END>>>" ] && break
        echo "$line"
    done
}

# 명령 실행
echo "Current directory:"
exec_in_coproc "pwd"

echo -e "\nFiles:"
exec_in_coproc "ls -1"

echo -e "\nEnvironment variable:"
exec_in_coproc "echo \$HOME"

# 정리
echo "exit" >&"${SHELL[1]}"
wait $SHELL_PID
```

### 에러 처리가 있는 코프로세스

```bash
#!/bin/bash

# 에러 처리가 있는 코프로세스 시작
start_coproc() {
    if ! coproc WORKER { python3 -u -c '
import sys
while True:
    try:
        line = input()
        if line == "QUIT":
            break
        # 라인 처리
        print(f"Processed: {line}")
        sys.stdout.flush()
    except EOFError:
        break
'; }; then
        echo "Failed to start coprocess"
        return 1
    fi

    # 정리 설정
    trap 'echo "QUIT" >&"${WORKER[1]}" 2>/dev/null; wait $WORKER_PID 2>/dev/null' EXIT
}

# 코프로세스 사용
send_to_worker() {
    echo "$1" >&"${WORKER[1]}"
    read -u "${WORKER[0]}" -t 5 response || {
        echo "Timeout or error reading from worker"
        return 1
    }
    echo "$response"
}

start_coproc

send_to_worker "task1"
send_to_worker "task2"
send_to_worker "task3"
```

## 8. 프로세스 우선순위(Process Priority)

프로세스의 CPU 및 I/O 우선순위를 제어합니다.

### nice 명령

```bash
#!/bin/bash

# Nice 값은 -20 (최고 우선순위)에서 19 (최저)까지
# 기본값은 0

# 낮은 우선순위로 실행 (nice 값 10)
nice -n 10 ./cpu-intensive-script.sh

# 높은 우선순위로 실행 (음수 값은 root 필요)
# sudo nice -n -10 ./important-script.sh

# 현재 nice 값 확인
echo "Current nice value: $(nice)"
```

### renice 명령

```bash
#!/bin/bash

# 프로세스 시작
./my-script.sh &
PID=$!

# 실행 중인 프로세스의 우선순위 변경
renice -n 15 -p $PID

# 사용자의 모든 프로세스 renice
# sudo renice -n 10 -u username

# 그룹의 모든 프로세스 renice
# sudo renice -n 5 -g groupname
```

### ionice 명령

```bash
#!/bin/bash

# I/O 스케줄링 클래스:
# 0 - None (기본)
# 1 - Real-time (최고 우선순위, root 필요)
# 2 - Best-effort (기본)
# 3 - Idle (다른 I/O가 없을 때만)

# idle I/O 우선순위로 실행
ionice -c 3 ./disk-intensive-script.sh

# best-effort, 우선순위 4 (0-7, 낮을수록 높은 우선순위)로 실행
ionice -c 2 -n 4 ./my-script.sh

# 실행 중인 프로세스의 I/O 우선순위 변경
ionice -c 3 -p $PID
```

### 결합된 우선순위 예제

```bash
#!/bin/bash

# CPU와 I/O 집약적 작업을 낮은 우선순위로 실행
run_low_priority() {
    local cmd=$1

    # nice로 시작
    nice -n 19 bash -c "$cmd" &
    local pid=$!

    # I/O 우선순위를 idle로 설정
    ionice -c 3 -p $pid

    echo "Started low priority process: $pid"
    echo "Nice: $(ps -o nice= -p $pid)"
    echo "I/O class: $(ionice -p $pid)"

    wait $pid
}

# 사용 예제
run_low_priority "find / -name '*.log' 2>/dev/null | xargs gzip"
```

### 우선순위 관리 스크립트

```bash
#!/bin/bash

# 프로세스 우선순위 관리
manage_priority() {
    local pid=$1
    local cpu_priority=$2  # -20 ~ 19
    local io_class=$3      # 0-3
    local io_priority=$4   # 0-7

    echo "Managing priority for PID $pid"

    # CPU 우선순위 설정
    if [ -n "$cpu_priority" ]; then
        if renice -n "$cpu_priority" -p "$pid" >/dev/null 2>&1; then
            echo "  CPU priority set to $cpu_priority"
        else
            echo "  Failed to set CPU priority (may need sudo)"
        fi
    fi

    # I/O 우선순위 설정
    if [ -n "$io_class" ]; then
        local ionice_cmd="ionice -c $io_class"
        [ -n "$io_priority" ] && ionice_cmd="$ionice_cmd -n $io_priority"

        if $ionice_cmd -p "$pid" >/dev/null 2>&1; then
            echo "  I/O priority set to class $io_class"
        else
            echo "  Failed to set I/O priority (may need sudo)"
        fi
    fi

    # 현재 우선순위 표시
    echo "  Current nice value: $(ps -o nice= -p $pid)"
    echo "  Current I/O: $(ionice -p $pid | head -1)"
}

# 예제: 낮은 우선순위로 백업 실행
echo "Starting backup..."
./backup.sh &
BACKUP_PID=$!

manage_priority $BACKUP_PID 19 3
wait $BACKUP_PID
echo "Backup complete"
```

## 연습 문제

### 문제 1: 병렬 파일 프로세서

설정 가능한 최대 동시 작업 수로 여러 파일을 병렬로 처리하는 스크립트를 작성하세요. 각 파일은 작업을 시뮬레이션하는 함수(sleep)로 처리되어야 하며, 스크립트는 각 파일이 처리를 시작하고 완료할 때 보고해야 합니다.

요구 사항:
- 디렉토리 경로와 최대 동시 작업 수를 인수로 받음
- 디렉토리의 모든 `.txt` 파일 처리
- 총 처리 시간 추적 및 보고
- 에러를 우아하게 처리

### 문제 2: 시그널 안전 다운로드 매니저

다음 기능을 가진 다운로드 매니저 스크립트를 만드세요:
- 여러 URL을 병렬로 다운로드
- 진행 상황을 상태 파일에 저장
- SIGINT (Ctrl+C)로 중단하고 나중에 재개 가능
- SIGTERM에서 부분 다운로드 정리
- 모든 정리를 적절히 처리하기 위해 trap 사용

### 문제 3: 백그라운드 작업 모니터

다음 기능을 가진 작업 모니터링 시스템을 구현하세요:
- 여러 백그라운드 작업 시작
- 매초마다 상태 모니터링
- 각 작업이 완료될 때 보고
- 진행 표시기 표시
- 동시 작업을 최대값으로 제한 (예: 3)
- 작업 실패를 처리하고 실패한 작업을 한 번 재시도

### 문제 4: 코프로세스 계산기 서비스

코프로세스를 사용하여 계산기 서비스를 만드세요:
- `bc` 코프로세스 시작
- 표현식을 입력할 명령줄 인터페이스 제공
- 히스토리 지원 (최근 10개 계산 표시)
- 에러 처리 (잘못된 표현식)
- 적절한 종료 시퀀스 구현

### 문제 5: 우선순위 기반 작업 스케줄러

다음 기능을 가진 작업 스케줄러를 만드세요:
- 우선순위 수준(high, medium, low)을 가진 작업 수락
- 높은 우선순위 작업은 nice 값 0으로 실행
- 중간 우선순위 작업은 nice 값 10으로 실행
- 낮은 우선순위 작업은 nice 값 19와 I/O 클래스 idle로 실행
- 모든 실행 중인 작업의 상태 보고
- 총 동시 작업을 5개로 제한
- 종료 시 적절한 정리 구현

---

**이전**: [08_Regex_in_Bash.md](./08_Regex_in_Bash.md) | **다음**: [10_Error_Handling.md](./10_Error_Handling.md)
