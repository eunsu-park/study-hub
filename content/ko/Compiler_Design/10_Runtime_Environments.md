# 레슨 10: 런타임 환경(Runtime Environments)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 실행 중인 프로그램의 표준 메모리 레이아웃(코드, 정적, 스택, 힙)을 **설명**할 수 있다
2. 활성화 레코드(Activation Records)(스택 프레임)와 그 내용을 **설명**할 수 있다
3. 호출 규약(Calling Conventions)(cdecl, stdcall, System V AMD64 ABI)과 그 함의를 **비교**할 수 있다
4. 값에 의한 전달, 참조에 의한 전달, 이름에 의한 전달 등 매개변수 전달 메커니즘을 **구현**할 수 있다
5. 액세스 링크(Access Links)와 디스플레이(Displays)를 사용하여 중첩 함수를 **처리**할 수 있다
6. 정적 스코핑(Static Scoping)과 동적 스코핑(Dynamic Scoping)의 차이 및 런타임 구현을 **구별**할 수 있다
7. 힙 관리 전략(자유 리스트(Free Lists), 버디 시스템(Buddy System), 가비지 컬렉션(Garbage Collection))을 **설명**할 수 있다
8. Python으로 런타임 호출 스택을 **시뮬레이션**할 수 있다

---

## 1. 저장 구성(Storage Organization)

### 1.1 메모리 모델

컴파일된 프로그램이 실행될 때, 운영 체제는 서로 다른 영역으로 구성된 메모리를 할당합니다. 각 영역은 특정 목적을 가지고 있습니다:

```
높은 주소
┌─────────────────────┐
│       Stack         │  ← 아래 방향으로 성장
│         │           │
│         ▼           │
│                     │
│         ▲           │
│         │           │
│        Heap         │  ← 위 방향으로 성장
├─────────────────────┤
│    Static/Global    │  ← 로드 시 고정 크기
│       Data          │
├─────────────────────┤
│    Read-Only Data   │  ← 문자열 리터럴, 상수
├─────────────────────┤
│       Code          │  ← 텍스트 세그먼트 (명령어)
│      (Text)         │
└─────────────────────┘
낮은 주소
```

### 1.2 코드(텍스트) 세그먼트

**코드 세그먼트**(또는 텍스트 세그먼트)는 컴파일된 프로그램의 기계 명령어를 보관합니다.

**특성**:
- **읽기 전용**: 명령어의 우발적이거나 악의적인 수정을 방지
- **공유 가능**: 동일한 프로그램을 실행하는 여러 프로세스가 단일 복사본을 공유 가능
- **고정 크기**: 컴파일/링크 시점에 결정
- **한 번 로드**: 프로세스 시작 시 메모리에 매핑

### 1.3 정적/전역 데이터(Static/Global Data)

**정적 데이터** 영역은 전역 변수와 함수 내에서 `static`으로 선언된 변수를 저장합니다. 이 영역의 크기는 컴파일 시점에 결정되며 고정됩니다.

일반적으로 다음과 같이 세분화됩니다:

| 하위 영역 | 내용 | 예시 |
|------------|----------|---------|
| `.data` | 초기화된 전역/정적 변수 | `int count = 42;` |
| `.bss` | 초기화되지 않은 전역/정적 변수 (0으로 채워짐) | `static int buffer[1024];` |
| `.rodata` | 읽기 전용 데이터: 문자열 리터럴, 상수 | `"Hello, World!"` |

**주소 지정**: 정적 영역의 변수는 링크 시점에 알 수 있는 **절대 주소**를 가집니다. 컴파일러는 직접 주소 참조를 생성합니다.

### 1.4 스택(The Stack)

**스택**은 함수 호출 관리에 사용됩니다. 스택에는 다음이 저장됩니다:

- 각 활성 함수 호출에 대한 **활성화 레코드(Activation Records)**(스택 프레임)
- 함수의 **지역 변수**
- 함수에 전달된 **매개변수**
- 호출자를 재개하기 위한 **반환 주소**
- 피호출자(Callee)가 보존해야 하는 **저장된 레지스터**

스택은 대부분의 아키텍처에서 **아래 방향**으로 성장합니다(낮은 주소 방향). **스택 포인터**(SP)는 스택의 현재 상단을 표시하고, **프레임 포인터**(FP, 베이스 포인터 BP라고도 함)는 현재 프레임 내의 고정 기준점을 표시합니다.

### 1.5 힙(The Heap)

**힙**은 동적으로 할당된 메모리를 위해 사용됩니다 -- 컴파일 시점에 크기나 수명을 결정할 수 없는 데이터입니다.

**예시**:
- C에서의 `malloc()` / `free()`
- C++에서의 `new` / `delete`
- Java, Python에서의 객체 생성
- 동적 크기 배열, 연결 리스트, 트리

힙은 **위 방향**으로 성장합니다(더 높은 주소 방향). 스택과 힙 사이에는 미사용 주소 공간이 있어 두 영역 모두 성장할 수 있는 공간을 제공합니다.

### 1.6 주소 공간 레이아웃 무작위화(ASLR)

현대 운영 체제는 프로그램 실행 시마다 스택, 힙, 공유 라이브러리의 위치를 무작위로 배치합니다. 이를 통해 공격자가 메모리 주소를 예측하기 어렵게 만들어 버퍼 오버플로우 악용과 반환 지향 프로그래밍(ROP) 공격을 완화합니다.

---

## 2. 활성화 레코드(Activation Records)(스택 프레임)

### 2.1 활성화 레코드란?

함수가 호출될 때마다 새로운 **활성화 레코드**(또는 **스택 프레임**)가 런타임 스택에 푸시됩니다. 이 레코드에는 함수를 실행하고 호출자에게 반환하는 데 필요한 모든 정보가 포함되어 있습니다.

### 2.2 활성화 레코드의 구조

일반적인 활성화 레코드는 다음과 같습니다(높은 주소에서 낮은 주소 방향으로 성장):

```
┌───────────────────────┐  높은 주소
│    호출자가 전달한     │  ← 호출자가 푸시 (스택에 있는 경우)
│    인수               │
├───────────────────────┤
│    반환 주소          │  ← CALL 명령어가 푸시
├───────────────────────┤  ◀── 프레임 포인터 (FP / BP)
│    저장된 이전 FP     │  ← 동적 링크 (이전 프레임 포인터)
├───────────────────────┤
│    저장된 레지스터    │  ← 피호출자가 저장한 레지스터
├───────────────────────┤
│    지역 변수          │  ← 함수의 지역 저장소
├───────────────────────┤
│    임시 변수          │  ← 컴파일러가 생성한 임시 변수
├───────────────────────┤
│    나가는 인수        │  ← 이 함수가 호출하는 함수의
│    (필요한 경우)      │     인수
└───────────────────────┘  ◀── 스택 포인터 (SP)
                           낮은 주소
```

### 2.3 구성 요소 상세 설명

#### 반환 주소(Return Address)

피호출자가 끝난 후 제어가 반환되어야 하는 호출자의 명령어 주소입니다. x86에서 `CALL` 명령어는 자동으로 반환 주소를 스택에 푸시합니다.

#### 동적 링크(Dynamic Link)(저장된 프레임 포인터)

호출자의 활성화 레코드(특히 호출자의 프레임 포인터)를 가리키는 포인터입니다. 이는 디버깅(스택 언와인딩)을 위해 순회할 수 있는 프레임의 **체인**을 형성합니다.

```
       호출자 프레임
       ┌──────────┐
FP ──▶ │ saved FP │ ───▶ 이전 프레임 ...
       ├──────────┤
       │  locals  │
       └──────────┘
```

#### 정적 링크(Static Link)(액세스 링크)

중첩 함수에 사용됩니다(섹션 5에서 설명). 렉시컬 방식으로 둘러싸는(Lexically Enclosing) 함수의 활성화 레코드를 가리킵니다.

#### 저장된 레지스터(Saved Registers)

피호출자가 보존해야 하는 레지스터(피호출자 저장 레지스터). 함수는 진입 시 이들을 저장하고 반환 전에 복원합니다.

#### 지역 변수(Local Variables)

프레임 포인터로부터 알려진 오프셋에 할당된 함수의 지역 변수 저장소입니다.

#### 임시 변수(Temporaries)

레지스터에 맞지 않는 컴파일러가 생성한 임시 값입니다.

### 2.4 지역 변수 접근

지역 변수는 프레임 포인터로부터 고정 **오프셋**으로 접근됩니다:

```
FP로부터 오프셋 -8에 선언된 변수 x:
    x = FP - 8

FP로부터 오프셋 +16에 선언된 매개변수 p (반환 주소 위):
    p = FP + 16
```

프레임 포인터는 함수 실행 중 스택 포인터가 이동하더라도(예: 중첩 호출을 위한 인수 푸시 시) 안정적인 기준점을 제공합니다.

### 2.5 예시: 함수 호출 시퀀스

다음 C 코드를 고려해 보겠습니다:

```c
int add(int a, int b) {
    int result = a + b;
    return result;
}

int main() {
    int x = 3;
    int y = 4;
    int z = add(x, y);
    return z;
}
```

`add(x, y)` 호출 중 발생하는 이벤트 순서:

**1. 호출자(main) -- 호출 전**:
```
push y        ; 두 번째 인수 푸시 (또는 레지스터 사용)
push x        ; 첫 번째 인수 푸시 (또는 레지스터 사용)
call add      ; 반환 주소 푸시, add로 점프
```

**2. 피호출자(add) -- 함수 프롤로그**:
```
push rbp      ; 호출자의 프레임 포인터 저장
mov rbp, rsp  ; 새 프레임 포인터 설정
sub rsp, 16   ; 지역 변수 공간 할당
```

**3. 피호출자(add) -- 함수 본문**:
```
mov eax, [rbp+16]   ; 매개변수 a 로드
add eax, [rbp+24]   ; 매개변수 b 더하기
mov [rbp-8], eax     ; 결과 저장
```

**4. 피호출자(add) -- 함수 에필로그**:
```
mov eax, [rbp-8]     ; 반환값을 레지스터에 로드
mov rsp, rbp         ; 지역 변수 해제
pop rbp              ; 호출자의 프레임 포인터 복원
ret                  ; 반환 주소를 팝하고 돌아가기
```

**5. 호출자(main) -- 호출 후**:
```
add rsp, 16          ; 인수 정리 (cdecl에서)
mov [rbp-24], eax    ; 반환값을 z에 저장
```

---

## 3. 호출 규약(Calling Conventions)

### 3.1 호출 규약이란?

**호출 규약**은 다음을 정의하는 프로토콜입니다:

1. **인수 전달 방법** (레지스터? 스택? 어떤 순서?)
2. **스택 정리 담당** (호출자 또는 피호출자?)
3. **보존할 레지스터** (호출자 저장 vs 피호출자 저장)
4. **반환값 전달 방법**
5. **스택 프레임 구성 방법**

호출 규약은 별도로 컴파일된 함수들이 올바르게 상호 작용할 수 있도록 보장합니다.

### 3.2 cdecl (C 선언)

32비트 x86에서 C의 기본 호출 규약입니다.

| 측면 | cdecl |
|--------|-------|
| 인수 | 오른쪽에서 왼쪽으로 스택에 푸시 |
| 스택 정리 | 호출자가 정리 |
| 반환값 | `EAX`에 (정수), `ST(0)`에 (부동소수점) |
| 피호출자 저장 레지스터 | `EBX`, `ESI`, `EDI`, `EBP` |
| 가변인수 지원 | 예 (호출자가 인수 개수를 앎) |

**오른쪽에서 왼쪽으로 푸시**하면 첫 번째 인수가 가장 낮은 스택 주소에, 스택 상단에 가장 가깝게 위치합니다. 이를 통해 첫 번째 인수가 항상 알려진 오프셋에 있으므로 가변인수 함수(`printf` 등)가 가능합니다.

```
// 호출: add(3, 4)
push 4         ; 두 번째 인수
push 3         ; 첫 번째 인수
call add
add esp, 8    ; 호출자가 정리 (인수 2개 × 4바이트)
```

### 3.3 stdcall

Windows API(Win32 API)에서 사용합니다.

| 측면 | stdcall |
|--------|---------|
| 인수 | 오른쪽에서 왼쪽으로 스택에 푸시 |
| 스택 정리 | **피호출자**가 정리 |
| 반환값 | `EAX`에 |
| 가변인수 지원 | 불가 (피호출자가 정확한 인수 개수를 알아야 함) |

```
// 피호출자 에필로그에 포함:
ret 8          ; 반환하고 8바이트 팝 (인수 2개 × 4바이트)
```

**장점**: 정리 명령어가 모든 호출 위치가 아닌 피호출자에 한 번만 나타나므로 코드 크기가 약간 줄어듭니다.

**단점**: 가변인수 함수를 지원할 수 없습니다.

### 3.4 System V AMD64 ABI (Linux/macOS x86-64)

Linux, macOS, FreeBSD 및 기타 유닉스 계열 시스템에서 사용되는 현대적인 64비트 호출 규약입니다.

| 측면 | System V AMD64 |
|--------|----------------|
| 정수 인수 (처음 6개) | `RDI`, `RSI`, `RDX`, `RCX`, `R8`, `R9` |
| 부동소수점 인수 (처음 8개) | `XMM0`--`XMM7` |
| 추가 인수 | 오른쪽에서 왼쪽으로 스택에 푸시 |
| 스택 정리 | 호출자 |
| 반환값 | `RAX` (정수), `XMM0` (부동소수점) |
| 피호출자 저장 레지스터 | `RBX`, `RBP`, `R12`--`R15` |
| 스택 정렬 | `CALL` 전 16바이트 정렬 |
| 레드 존(Red zone) | RSP 아래 128바이트 (리프 함수는 RSP 조정 없이 사용 가능) |

**예시**: `f(1, 2, 3, 4, 5, 6, 7, 8)` 호출:

```asm
; 인수 1-6은 레지스터에
mov rdi, 1
mov rsi, 2
mov rdx, 3
mov rcx, 4
mov r8, 5
mov r9, 6
; 인수 7-8은 스택에 (오른쪽에서 왼쪽으로)
push 8
push 7
call f
add rsp, 16   ; 호출자가 스택 인수 정리
```

### 3.5 비교 표

| 기능 | cdecl (x86) | stdcall (x86) | System V AMD64 |
|---------|-------------|---------------|----------------|
| 인수 전달 | 스택만 | 스택만 | 레지스터 6개 + 스택 |
| 인수 순서 | 오른쪽에서 왼쪽 | 오른쪽에서 왼쪽 | 왼쪽에서 오른쪽 (레지스터) |
| 정리 | 호출자 | 피호출자 | 호출자 |
| 가변인수 | 예 | 아니오 | 예 |
| 성능 | 보통 | 보통 | 더 좋음 (레지스터 전달) |
| 플랫폼 | Unix/Windows 32비트 | Windows 32비트 | Linux/macOS 64비트 |

### 3.6 Windows x64 호출 규약

참고로, Windows 64비트는 다른 규약을 사용합니다:

| 측면 | Windows x64 |
|--------|-------------|
| 정수 인수 (처음 4개) | `RCX`, `RDX`, `R8`, `R9` |
| 부동소수점 인수 (처음 4개) | `XMM0`--`XMM3` |
| 섀도우 공간(Shadow space) | 피호출자 사용을 위해 호출자가 32바이트 예약 |
| 반환값 | `RAX` (정수), `XMM0` (부동소수점) |
| 스택 정렬 | 16바이트 정렬 |

---

## 4. 매개변수 전달 메커니즘(Parameter Passing Mechanisms)

### 4.1 값에 의한 전달(Call by Value)

**호출자**가 인수 표현식을 평가하고 값의 **복사본**을 피호출자에게 전달합니다. 피호출자 내에서 매개변수를 수정해도 원래 변수에는 영향을 미치지 않습니다.

```c
void increment(int x) {
    x = x + 1;   // 지역 복사본만 수정
}

int main() {
    int a = 5;
    increment(a);
    // a는 여전히 5
}
```

**구현**: 값이 피호출자의 매개변수 슬롯(레지스터 또는 스택 위치)에 복사됩니다.

**언어**: C, Java (기본 타입), Go (비포인터 타입)

### 4.2 참조에 의한 전달(Call by Reference)

호출자가 인수의 **주소**(참조)를 전달합니다. 피호출자는 이 주소를 통해 원래 변수를 읽고 수정할 수 있습니다.

```cpp
void increment(int &x) {
    x = x + 1;   // 원래 변수 수정
}

int main() {
    int a = 5;
    increment(a);
    // a는 이제 6
}
```

**구현**: 변수의 주소가 전달됩니다. 피호출자 내에서 매개변수에 대한 모든 접근은 포인터를 통한 간접 메모리 접근입니다.

```
; 호출자:
lea rdi, [rbp-8]    ; a의 주소 전달
call increment

; 피호출자:
mov eax, [rdi]      ; a를 읽기 위해 역참조
add eax, 1
mov [rdi], eax      ; a를 쓰기 위해 역참조
```

**언어**: C++ (참조), Fortran (기본값), C# (`ref` 매개변수)

### 4.3 값-결과에 의한 전달(Call by Value-Result)(복사-입력, 복사-출력)

함수가 호출될 때 값이 **복사 입력**되고 함수가 반환될 때 **복사 출력**됩니다. 이는 에일리어싱(Aliasing)이 발생할 때 참조에 의한 전달과 다릅니다.

```
procedure swap(x, y):
    // 실제 인수의 복사본 생성 (복사 입력)
    temp = x
    x = y
    y = temp
    // 복사본을 실제 인수에 다시 씀 (복사 출력)
```

참조에 의한 전달로 `swap(a, a)`를 호출하면 두 매개변수가 동일한 변수를 참조하므로 결과가 정의되지 않습니다. 값-결과에 의한 전달에서는 최종 값이 어느 복사 출력이 마지막에 발생하는지에 따라 달라집니다.

**언어**: Ada (`in out` 매개변수)

### 4.4 이름에 의한 전달(Call by Name)

인수는 호출 위치에서 평가되지 않습니다. 대신, 인수 표현식의 **텍스트**(또는 클로저와 유사한 썽크(Thunk))가 전달됩니다. 피호출자가 매개변수를 참조할 때마다 표현식이 호출자의 환경에서 재평가됩니다.

이는 역사적으로 Algol 60과 연관됩니다. 인수는 본질적으로 **썽크** -- 호출될 때 표현식을 평가하는 매개변수 없는 함수입니다.

**고전 예시 -- Jensen의 장치**:

```
// Algol 60 의사 코드
real procedure sum(i, lo, hi, expr);
    name i, expr;       // 이름에 의한 전달
    value lo, hi;       // 값에 의한 전달
    integer i, lo, hi;
    real expr;
begin
    real s;
    s := 0;
    for i := lo step 1 until hi do
        s := s + expr;  // 매 반복마다 expr 재평가
    sum := s;
end;

// 사용: i가 1부터 10까지 i*i의 합 계산
result = sum(i, 1, 10, i*i);
```

`expr`이 참조될 때마다 썽크 `i*i`가 현재 `i` 값으로 평가됩니다.

**구현**: 썽크는 표현식 코드와 이를 평가할 환경을 포함하는 작은 클로저입니다.

```python
# Python으로 썽크를 사용한 이름에 의한 전달 시뮬레이션

def call_by_name_demo():
    """썽크를 사용한 이름에 의한 전달 시연."""
    a = [1, 2, 3, 4, 5]
    i = 0

    def i_thunk():
        """i의 현재 값을 반환하는 썽크."""
        return i

    def a_i_thunk():
        """호출될 때마다 a[i]를 평가하는 썽크."""
        return a[i]

    def set_a_i(val):
        """a[i]를 설정하는 썽크."""
        a[i] = val

    # 이름에 의한 전달로 swap(i, a[i]) 시뮬레이션
    # 매개변수에 대한 각 접근은 썽크를 재평가
    print(f"Before: i={i}, a={a}")

    # swap 본문: temp = x; x = y; y = temp
    temp = i_thunk()            # temp = i (0으로 평가)
    i = a_i_thunk()             # i = a[i] = a[0] = 1
    set_a_i(temp)               # a[i] = temp, 하지만 이제 i=1이므로 a[1] = 0

    print(f"After:  i={i}, a={a}")
    # 결과: i=1, a=[1, 0, 3, 4, 5]
    # 참고: 쓰는 시점에 i가 1로 변경되었으므로 a[0]은 변경되지 않음

call_by_name_demo()
```

### 4.5 매개변수 전달 메커니즘 비교

| 메커니즘 | 평가 시점 | 에일리어싱 효과 | 성능 |
|-----------|----------------|------------------|-------------|
| 값에 의한 전달 | 호출 위치 | 없음 | 복사 비용 |
| 참조에 의한 전달 | 호출 위치 | 있음 | 간접 참조 비용 |
| 값-결과에 의한 전달 | 호출 + 반환 시 | 복사 순서로 정의 | 두 번의 복사 |
| 이름에 의한 전달 | 사용할 때마다 | 복잡함 | 사용마다 썽크 오버헤드 |

---

## 5. 중첩 함수와 정적 스코핑(Nested Functions and Static Scoping)

### 5.1 중첩 스코프의 문제

Pascal, Ada, Python, ML과 같은 언어는 함수를 다른 함수 안에 중첩할 수 있습니다. 중첩 함수는 둘러싸는 스코프의 변수에 접근할 수 있습니다:

```python
def outer():
    x = 10

    def middle():
        y = 20

        def inner():
            # inner는 x, y 및 자신의 지역 변수에 접근 가능
            return x + y + 30

        return inner()

    return middle()
```

`inner`가 실행될 때 `outer` 프레임의 `x`와 `middle` 프레임의 `y`에 접근해야 합니다. 하지만 이들 프레임은 스택에 있고 `inner`의 프레임이 가장 위에 있습니다. `inner`는 어떻게 렉시컬 방식으로 둘러싸는 함수들의 프레임을 찾을 수 있을까요?

### 5.2 액세스 링크(Access Links)(정적 링크)

**액세스 링크**(또는 **정적 링크**)는 각 활성화 레코드에 저장된 포인터로, **렉시컬 방식으로 둘러싸는 함수**의 활성화 레코드를 가리킵니다.

```
스택:
┌────────────────────┐
│  inner의 프레임   │
│  access link ──────┼───┐
│  local: (없음)     │   │
├────────────────────┤   │
│  middle의 프레임  │ ◀─┘
│  access link ──────┼───┐
│  local: y = 20     │   │
├────────────────────┤   │
│  outer의 프레임   │ ◀─┘
│  access link = nil │
│  local: x = 10     │
└────────────────────┘
```

깊이 $d_{\text{var}}$의 변수에 깊이 $d_{\text{func}}$의 함수에서 접근하려면, 런타임은 $d_{\text{func}} - d_{\text{var}}$개의 액세스 링크를 따라갑니다.

**변수 접근의 시간 복잡도**: $O(d_{\text{func}} - d_{\text{var}})$, 중첩 깊이 차이에 비례합니다.

### 5.3 액세스 링크 유지 방법

깊이 $d_f$의 함수 $f$가 깊이 $d_g$의 함수 $g$를 호출할 때:

1. $d_g = d_f + 1$인 경우 (직접 중첩 함수 호출):
   - $g$의 액세스 링크는 $f$의 프레임을 가리킵니다.

2. $d_g \leq d_f$인 경우 (같은 수준 또는 외부 수준의 함수 호출):
   - $f$의 프레임에서 $d_f - d_g + 1$개의 액세스 링크를 따라 $g$의 렉시컬 방식으로 둘러싸는 함수의 프레임을 찾습니다.
   - $g$의 액세스 링크는 그 프레임을 가리킵니다.

### 5.4 디스플레이(Displays)

**디스플레이**는 액세스 링크를 위한 배열 기반 최적화입니다. 링크 체인을 따르는 대신, 디스플레이는 전역 배열 $D$를 유지하며 $D[i]$는 중첩 깊이 $i$의 가장 최근 활성화 레코드에 대한 포인터를 가집니다.

```
디스플레이 D:
D[0] ──▶ outer의 프레임
D[1] ──▶ middle의 프레임
D[2] ──▶ inner의 프레임
```

**깊이 $k$의 변수 접근**: 단순히 $D[k]$를 조회하고 오프셋을 더합니다. 이는 중첩 깊이와 무관하게 $O(1)$입니다.

**유지 관리**: 깊이 $k$의 함수에 진입 시:
1. $D[k]$의 이전 값을 저장
2. $D[k]$를 현재 프레임 포인터로 설정

나갈 때:
1. 저장된 값에서 $D[k]$ 복원

### 5.5 비교: 액세스 링크 vs 디스플레이

| 측면 | 액세스 링크 | 디스플레이 |
|--------|-------------|----------|
| 저장소 | 프레임당 포인터 하나 | 전역 배열 (크기 = 최대 깊이) |
| 변수 접근 | $O(\text{깊이 차이})$ | $O(1)$ |
| 유지 관리 | 단순 포인터 대입 | 배열 항목 저장/복원 |
| 클로저 | 자연스러움 (링크가 클로저의 일부) | 더 복잡함 (배열 상태 캡처 필요) |
| 사용처 | 현대 컴파일러 | 구형 컴파일러 (Burroughs B5000) |

### 5.6 클로저(Closures)

**클로저**는 함수를 렉시컬 환경(액세스 링크 또는 동등한 것)과 함께 캡처합니다. 함수가 값으로 반환되거나 데이터 구조에 저장될 때, 클로저는 함수가 둘러싸는 스코프의 변수에 여전히 접근할 수 있도록 보장합니다.

```python
def make_adder(x):
    def add(y):
        return x + y    # x는 make_adder의 스코프에서 캡처됨
    return add           # 클로저 반환

add5 = make_adder(5)
print(add5(3))           # 출력: 8
```

**구현 과제**:
- 둘러싸는 함수의 프레임이 스택에 있으면, 함수가 반환될 때 해제됩니다.
- 클로저는 캡처된 변수들을 살려두어야 하며, 일반적으로 스택 대신 **힙**에 할당합니다 (이를 "변수 탈출(Variable Escape)" 또는 "클로저 변환(Closure Conversion)"이라 합니다).

---

## 6. 동적 스코핑(Dynamic Scoping) vs 정적 스코핑(Static Scoping)

### 6.1 정적(렉시컬) 스코핑

**정적 스코핑**에서는 변수의 바인딩이 프로그램의 텍스트(렉시컬 구조)에 의해 결정됩니다. 변수 참조는 컴파일 시점에 둘러싸는 스코프를 바깥쪽으로 검색하여 해석됩니다.

```python
x = 10

def foo():
    return x     # 항상 전역 x (=10)를 참조

def bar():
    x = 20
    return foo() # foo는 여전히 x=10을 봄 (정적 스코핑)

print(bar())     # 출력: 10
```

**구현**: 런타임에서 액세스 링크 또는 디스플레이 사용. 스코프 체인은 함수가 **정의**된 곳에 의해 결정되며, **호출**되는 곳이 아닙니다.

### 6.2 동적 스코핑

**동적 스코핑**에서는 변수의 바인딩이 런타임 호출 체인에 의해 결정됩니다. 변수 참조는 가장 최근의 바인딩을 찾기 위해 **호출 스택**을 통해 검색됩니다.

```
x = 10

function foo():
    return x     // 동적 스코핑에서는 foo를 호출한 사람에 따라 다름

function bar():
    x = 20
    return foo() // foo는 bar의 바인딩이 스택에 있으므로 x=20을 봄

bar()            // 20 반환 (동적 스코핑)
```

**언어**: 초기 Lisp, Bash/셸 스크립트, Emacs Lisp, Perl (`local` 사용).

### 6.3 런타임 구현

#### 정적 스코핑 런타임
- 렉시컬 중첩을 따르는 **액세스 링크** 사용
- 변수 위치는 **컴파일 시점**에 결정 (특정 프레임에서의 오프셋)
- 액세스 링크 체인은 함수가 **정의**된 곳을 기반으로 함

#### 동적 스코핑 런타임

두 가지 일반적인 구현이 있습니다:

**1. 깊은 접근(Deep access)**: 변수 바인딩을 찾기 위해 **호출 스택**(동적 체인)을 위로 올라가며 검색합니다. 각 프레임은 변수 이름과 값을 저장합니다.

```
function lookup(var_name):
    frame = current_frame
    while frame is not null:
        if var_name in frame.locals:
            return frame.locals[var_name]
        frame = frame.dynamic_link    // 호출 체인 따라가기
    error("unbound variable")
```

**시간 복잡도**: $O(d)$ (d는 호출 스택의 깊이).

**2. 얕은 접근(Shallow access)**: **중앙 테이블**(변수 이름당 하나의 항목)을 유지하여 항상 현재 바인딩을 보관합니다. 함수가 진입될 때 이전 바인딩을 저장하고 새 것을 설치합니다. 함수가 종료될 때 이전 바인딩을 복원합니다.

```
central_table = {}
save_stack = {}

function enter_scope(var_name, value):
    save old central_table[var_name]
    central_table[var_name] = value

function exit_scope(var_name):
    central_table[var_name] = saved value
```

**시간 복잡도**: 변수 접근은 $O(1)$; $k$개의 지역 변수를 가진 스코프 진입/종료는 $O(k)$.

### 6.4 Python 시뮬레이션: 정적 스코핑 vs 동적 스코핑

```python
"""정적 스코핑과 동적 스코핑의 차이를 시연합니다."""


class Environment:
    """변수 조회를 위한 간단한 환경."""

    def __init__(self, bindings=None, parent=None):
        self.bindings = bindings or {}
        self.parent = parent

    def lookup(self, name):
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        raise NameError(f"Unbound variable: {name}")

    def set(self, name, value):
        self.bindings[name] = value


# ---------- 정적 스코핑 ----------

class StaticScopingInterpreter:
    """
    정적(렉시컬) 스코핑을 사용하는 인터프리터.
    각 함수는 정의 환경을 캡처합니다.
    """

    def __init__(self):
        self.global_env = Environment()

    def define_var(self, name, value):
        self.global_env.set(name, value)

    def define_function(self, name, params, body, def_env=None):
        """
        함수를 클로저로 저장: (params, body, defining_env).
        """
        env = def_env or self.global_env
        closure = (params, body, env)  # 정의 환경 캡처
        self.global_env.set(name, closure)
        return closure

    def call_function(self, name, args):
        """
        함수 호출. 변수 조회는 정의 환경을 사용합니다.
        """
        closure = self.global_env.lookup(name)
        params, body, def_env = closure

        # 정의 환경을 부모로 하는 새 스코프 생성 (정적)
        call_env = Environment(
            bindings=dict(zip(params, args)),
            parent=def_env  # <-- 정적: 정의 환경 사용
        )
        return body(call_env)


class DynamicScopingInterpreter:
    """
    동적 스코핑을 사용하는 인터프리터.
    변수 조회는 호출 체인을 따릅니다.
    """

    def __init__(self):
        self.global_env = Environment()

    def define_var(self, name, value):
        self.global_env.set(name, value)

    def define_function(self, name, params, body):
        func = (params, body)
        self.global_env.set(name, func)
        return func

    def call_function(self, name, args, caller_env=None):
        """
        함수 호출. 변수 조회는 호출자의 환경을 사용합니다.
        """
        func = self.global_env.lookup(name)
        params, body = func

        parent = caller_env or self.global_env

        # 호출자 환경을 부모로 하는 새 스코프 생성 (동적)
        call_env = Environment(
            bindings=dict(zip(params, args)),
            parent=parent  # <-- 동적: 호출자의 환경 사용
        )
        return body(call_env)


def demo_scoping():
    """차이를 시연합니다."""

    # --- 정적 스코핑 ---
    print("=== Static Scoping ===")
    static = StaticScopingInterpreter()
    static.define_var("x", 10)

    # foo는 x를 반환 (x=10인 전역 스코프에서 정의)
    static.define_function("foo", [],
        lambda env: env.lookup("x"))

    # bar는 x=20을 지역적으로 설정하고 foo를 호출
    def bar_body_static(env):
        return static.call_function("foo", [])

    static.define_function("bar", [],
        lambda env: (
            env.set("x", 20),
            static.call_function("foo", [])
        )[-1])

    result = static.call_function("bar", [])
    print(f"  bar() calls foo(), foo sees x = {result}")
    # 정적: foo는 x=10을 봄 (정의 환경에서)

    # --- 동적 스코핑 ---
    print("\n=== Dynamic Scoping ===")
    dynamic = DynamicScopingInterpreter()
    dynamic.define_var("x", 10)

    dynamic.define_function("foo", [],
        lambda env: env.lookup("x"))

    def bar_body_dynamic(env):
        env.set("x", 20)
        return dynamic.call_function("foo", [], caller_env=env)

    dynamic.define_function("bar", [], bar_body_dynamic)

    result = dynamic.call_function("bar", [])
    print(f"  bar() calls foo(), foo sees x = {result}")
    # 동적: foo는 x=20을 봄 (호출 체인의 bar 환경에서)


if __name__ == "__main__":
    demo_scoping()
```

**예상 출력**:
```
=== Static Scoping ===
  bar() calls foo(), foo sees x = 10

=== Dynamic Scoping ===
  bar() calls foo(), foo sees x = 20
```

---

## 7. 힙 관리(Heap Management)

### 7.1 왜 힙 할당이 필요한가?

스택은 **후입선출(LIFO)** 방식의 수명을 가진 데이터에 대해 효율적인 메모리 관리를 제공합니다. 하지만 모든 데이터가 이 패턴을 따르지는 않습니다:

- 생성한 함수보다 수명이 긴 객체
- 동적으로 증가하거나 축소하는 데이터 구조 (리스트, 트리, 해시 테이블)
- 둘러싸는 스코프의 변수를 캡처하는 클로저

이런 데이터는 **힙**에 할당되어야 합니다.

### 7.2 명시적 할당과 해제

C와 C++같은 언어에서는 프로그래머가 힙 메모리를 명시적으로 관리합니다:

```c
int *p = malloc(sizeof(int) * 100);  // 할당
// ... p 사용 ...
free(p);                              // 해제
```

**명시적 관리의 문제점**:
- **메모리 누수**: 메모리 해제를 잊어버림
- **댕글링 포인터(Dangling Pointers)**: 해제된 메모리 사용
- **이중 해제(Double Free)**: 동일한 메모리를 두 번 해제
- **단편화(Fragmentation)**: 힙 전반에 걸쳐 흩어진 빈 블록

### 7.3 자유 리스트 관리(Free List Management)

**자유 리스트**는 빈 메모리 블록의 연결 리스트입니다. 할당자는 각 할당 요청에 맞는 블록을 찾기 위해 이 리스트를 검색합니다.

```
힙:
┌─────┬──────────┬─────┬──────┬─────┬──────────┐
│USED │  FREE    │USED │ FREE │USED │  FREE    │
│100B │  200B    │150B │  50B │80B  │  300B    │
└─────┴──────────┴─────┴──────┴─────┴──────────┘

자유 리스트:
head ──▶ [200B] ──▶ [50B] ──▶ [300B] ──▶ null
```

#### 할당 전략

**최초 적합(First Fit)**: 자유 리스트의 처음부터 스캔하여 충분히 큰 첫 번째 블록을 반환합니다.
- 빠른 할당
- 힙의 시작 부분에 단편화를 유발하는 경향

**최적 적합(Best Fit)**: 전체 자유 리스트를 스캔하여 충분히 큰 가장 작은 블록을 반환합니다.
- 할당당 낭비되는 공간 최소화
- 느림 (전체 리스트 스캔 필요); 많은 작은 사용 불가 단편 생성

**최악 적합(Worst Fit)**: 가장 큰 빈 블록을 반환합니다.
- 가장 큰 나머지 단편을 남김 (나중에 유용할 수 있음)
- 역시 느림; 실제로는 종종 성능이 낮음

**다음 적합(Next Fit)**: 최초 적합과 비슷하지만 이전 검색이 끝난 곳에서 스캔을 시작합니다.
- 힙 전반에 걸쳐 할당을 더 균등하게 분배
- 항상 시작 부분이 단편화되는 것을 방지

#### 병합(Coalescing)

블록이 해제될 때, 할당자는 인접한 블록도 빈지 확인하고 **병합**합니다:

```
free(B) 전:
┌─────┬──────────┬─────┬──────────┐
│  A  │  B(used) │  C  │  D(free) │
│free │          │free │          │
└─────┴──────────┴─────┴──────────┘

병합하여 free(B) 후:
┌─────────────────────────┬──────────┐
│   A + B + C (병합)      │  D(free) │
│        free             │          │
└─────────────────────────┴──────────┘
```

병합은 외부 단편화를 줄입니다. 병합을 효율적으로 하려면, 각 블록은 일반적으로 다음을 저장합니다:
- 블록 크기와 할당 상태를 포함하는 **헤더**
- 역방향 병합을 가능하게 하는 블록 크기를 포함하는 **푸터**(경계 태그)

### 7.4 버디 시스템(Buddy System)

**버디 시스템**은 크기가 2의 거듭제곱인 블록으로 힙 메모리를 구성합니다. 이는 분할과 병합을 단순화합니다.

**알고리즘**:

1. 메모리는 $2^0, 2^1, 2^2, \ldots, 2^k$ 크기의 블록으로 나뉩니다
2. 각 블록 크기에 대한 별도의 자유 리스트를 유지합니다

$n$ 바이트 **할당**:
1. $n$을 다음 2의 거듭제곱인 $2^j$로 올림
2. 크기 $\geq 2^j$의 가장 작은 빈 블록 찾기
3. 찾은 블록이 필요한 것보다 크면 (크기 $2^{j+k}$인 경우):
   - 크기 $2^j$의 블록을 얻을 때까지 **버디**(두 개의 같은 절반)로 반복 분할
   - 사용되지 않은 버디는 해당 자유 리스트에 추가

**해제**:
1. 블록 해제
2. **버디**(분할의 다른 절반)도 빈지 확인
3. 그렇다면, 더 큰 블록으로 병합
4. 재귀적으로 병합 반복

**버디 찾기**: 주소 $A$에서 크기 $2^j$인 블록의 버디는 다음 위치에 있습니다:

$$\text{buddy}(A, j) = A \oplus 2^j$$

여기서 $\oplus$는 비트 단위 XOR 연산입니다.

**예시**:

```
초기 상태: 1024바이트의 블록 하나

100바이트 요청 (128로 올림):
1024 → 분할 → 512 + 512
           → 분할 → 256 + 256
               → 분할 → 128 + 128
                           ↑ 할당

할당 후 자유 리스트:
512: [512에 있는 블록]
256: [256에 있는 블록]
128: [128에 있는 블록]    (다른 버디)
```

```python
"""버디 시스템 할당자 시뮬레이션."""

import math


class BuddyAllocator:
    """
    단순화된 버디 시스템 메모리 할당자.
    모든 크기는 2의 거듭제곱.
    """

    def __init__(self, total_size: int):
        """total_size로 초기화 (2의 거듭제곱이어야 함)."""
        self.total_size = total_size
        self.min_block = 16  # 최소 블록 크기
        self.max_order = int(math.log2(total_size))
        self.min_order = int(math.log2(self.min_block))

        # 순서별로 인덱싱된 자유 리스트 (2^order = 블록 크기)
        # 각 항목은 블록 시작 주소의 집합
        self.free_lists: dict[int, set] = {
            order: set() for order in range(self.min_order, self.max_order + 1)
        }

        # 초기에는 하나의 큰 빈 블록
        self.free_lists[self.max_order].add(0)

        # 할당된 블록 추적: 주소 -> 순서
        self.allocated: dict[int, int] = {}

    def _order_for_size(self, size: int) -> int:
        """요청한 크기 >= 블록 크기인 가장 작은 순서를 찾습니다."""
        order = max(self.min_order, math.ceil(math.log2(max(size, 1))))
        return order

    def _buddy_address(self, address: int, order: int) -> int:
        """XOR을 사용하여 버디의 주소를 계산합니다."""
        return address ^ (1 << order)

    def allocate(self, size: int) -> int:
        """
        최소 'size' 바이트의 블록을 할당합니다.
        시작 주소를 반환하거나, 할당 실패 시 -1을 반환합니다.
        """
        needed_order = self._order_for_size(size)

        # 사용 가능한 가장 작은 블록 찾기
        found_order = -1
        for order in range(needed_order, self.max_order + 1):
            if self.free_lists[order]:
                found_order = order
                break

        if found_order == -1:
            print(f"  FAILED: Cannot allocate {size} bytes")
            return -1

        # 자유 리스트에서 블록 제거
        address = min(self.free_lists[found_order])  # 가장 낮은 주소 선택
        self.free_lists[found_order].remove(address)

        # 필요한 순서로 분할
        current_order = found_order
        while current_order > needed_order:
            current_order -= 1
            # 상위 절반에 버디 생성
            buddy_addr = address + (1 << current_order)
            self.free_lists[current_order].add(buddy_addr)

        # 할당 기록
        self.allocated[address] = needed_order
        block_size = 1 << needed_order

        print(f"  Allocated {size}B at address {address} "
              f"(block size {block_size}B, order {needed_order})")
        return address

    def free(self, address: int):
        """이전에 할당된 블록을 해제하고 버디와 병합합니다."""
        if address not in self.allocated:
            print(f"  ERROR: Address {address} not allocated")
            return

        order = self.allocated.pop(address)
        block_size = 1 << order
        print(f"  Freeing address {address} (block size {block_size}B, order {order})")

        # 버디와 병합
        current_addr = address
        current_order = order

        while current_order < self.max_order:
            buddy_addr = self._buddy_address(current_addr, current_order)

            if buddy_addr in self.free_lists[current_order]:
                # 버디가 빔 -- 병합!
                self.free_lists[current_order].remove(buddy_addr)
                # 병합된 블록은 더 낮은 주소에서 시작
                current_addr = min(current_addr, buddy_addr)
                current_order += 1
                print(f"    Coalesced with buddy at {buddy_addr} "
                      f"-> new block at {current_addr} (order {current_order})")
            else:
                break

        # (가능하면 병합된) 블록을 자유 리스트에 추가
        self.free_lists[current_order].add(current_addr)

    def print_state(self):
        """할당자의 현재 상태를 출력합니다."""
        print("\n  Free lists:")
        for order in range(self.min_order, self.max_order + 1):
            if self.free_lists[order]:
                size = 1 << order
                addrs = sorted(self.free_lists[order])
                print(f"    Order {order} ({size:4d}B): {addrs}")

        if self.allocated:
            print("  Allocated blocks:")
            for addr in sorted(self.allocated.keys()):
                order = self.allocated[addr]
                print(f"    Address {addr}: {1 << order}B (order {order})")
        print()


def demo_buddy():
    """버디 시스템 할당 및 해제를 시연합니다."""
    print("=== Buddy System Allocator (1024 bytes) ===\n")
    allocator = BuddyAllocator(1024)

    print("Initial state:")
    allocator.print_state()

    print("--- Allocations ---")
    a1 = allocator.allocate(100)   # 128B 필요
    a2 = allocator.allocate(200)   # 256B 필요
    a3 = allocator.allocate(50)    # 64B 필요
    a4 = allocator.allocate(60)    # 64B 필요

    print("\nAfter allocations:")
    allocator.print_state()

    print("--- Deallocations ---")
    allocator.free(a3)
    allocator.free(a4)

    print("\nAfter freeing a3 and a4:")
    allocator.print_state()

    allocator.free(a1)

    print("After freeing a1:")
    allocator.print_state()

    allocator.free(a2)

    print("After freeing a2 (everything freed):")
    allocator.print_state()


if __name__ == "__main__":
    demo_buddy()
```

### 7.5 가비지 컬렉션(Garbage Collection)(개요)

자동 메모리 관리를 지원하는 언어(Java, Python, Go, OCaml)는 **가비지 컬렉션(GC)**을 사용하여 도달할 수 없는 힙 객체를 회수합니다.

주요 GC 전략:

| 전략 | 설명 | 장점 | 단점 |
|----------|-------------|------|------|
| **참조 계산(Reference counting)** | 각 객체는 자신을 가리키는 참조 수를 추적 | 즉각적인 회수; 단순 | 순환 참조 처리 불가; 카운터 오버헤드 |
| **표시-청소(Mark-and-sweep)** | 루트에서 도달 가능한 객체 표시, 미표시 청소 | 순환 처리 | 전체 정지(Stop-the-world) 일시 중지; 단편화 |
| **표시-압축(Mark-and-compact)** | 표시-청소와 같지만 살아있는 객체 압축 | 단편화 없음 | 비용이 큰 객체 이동 |
| **복사(Cheney)** | 살아있는 객체를 새 공간에 복사 | 빠른 할당; 단편화 없음 | 사용 가능한 메모리 절반 |
| **세대별(Generational)** | 객체를 나이로 분류; 젊은 세대를 더 자주 수집 | 세대별 가설 활용 | 복잡한 구현 |

**세대별 가설**은 대부분의 객체가 젊은 시절에 사망한다고 말합니다. 세대별 컬렉터는 **보육원(Nursery)**(젊은 세대)을 자주 수집하고 **구세대(Old generation)**는 드물게 수집하여 이를 활용합니다.

---

## 8. 실제 메모리 레이아웃

### 8.1 C 프로그램 레이아웃 살펴보기

```c
#include <stdio.h>
#include <stdlib.h>

// 전역/정적 데이터 (.data와 .bss)
int global_initialized = 42;     // .data
int global_uninitialized;        // .bss
static int static_var = 100;     // .data
const char *string_lit = "hello"; // .data에 포인터, .rodata에 문자열

void function() {
    // 스택
    int local_var = 10;
    int array[100];

    // 힙
    int *heap_data = malloc(sizeof(int) * 50);

    printf("Code:    %p (function address)\n", (void*)function);
    printf("Global:  %p (global_initialized)\n", (void*)&global_initialized);
    printf("BSS:     %p (global_uninitialized)\n", (void*)&global_uninitialized);
    printf("Static:  %p (static_var)\n", (void*)&static_var);
    printf("Literal: %p (string literal)\n", (void*)string_lit);
    printf("Stack:   %p (local_var)\n", (void*)&local_var);
    printf("Heap:    %p (malloc'd data)\n", (void*)heap_data);

    free(heap_data);
}

int main() {
    function();
    return 0;
}
```

**x86-64 Linux의 일반적인 출력** (ASLR로 인해 주소가 다를 수 있음):
```
Code:    0x55a3b7c00169
Global:  0x55a3b7e03010
BSS:     0x55a3b7e03018
Static:  0x55a3b7e03014
Literal: 0x55a3b7c00200
Stack:   0x7ffd9a3b4c0c
Heap:    0x55a3b8a046b0
```

스택 주소(`0x7ffd...`)와 힙 주소(`0x55a3...`) 사이의 큰 간격에 주목하십시오.

### 8.2 x86-64의 스택 프레임 레이아웃 (System V ABI)

```
높은 주소
┌─────────────────────────┐
│ Argument 8 (if any)     │  [RBP + 24]
├─────────────────────────┤
│ Argument 7              │  [RBP + 16]
├─────────────────────────┤
│ Return address          │  [RBP + 8]   (CALL에 의해 푸시)
├─────────────────────────┤
│ Saved RBP               │  [RBP + 0]   ◀── RBP가 여기를 가리킴
├─────────────────────────┤
│ Local variable 1        │  [RBP - 8]
├─────────────────────────┤
│ Local variable 2        │  [RBP - 16]
├─────────────────────────┤
│ Saved callee-saved regs │  [RBP - 24]
├─────────────────────────┤
│ Alignment padding       │
├─────────────────────────┤
│ Outgoing arg 7+         │  ◀── RSP가 여기를 가리킴
└─────────────────────────┘
낮은 주소
```

**레드 존(Red zone)**: System V AMD64에서 리프 함수(다른 함수를 호출하지 않는 함수)는 RSP를 조정하지 않고 RSP 아래 최대 128바이트까지 사용할 수 있습니다. 이는 작은 리프 함수에서 `sub rsp` / `add rsp`의 오버헤드를 방지합니다.

---

## 9. 예외를 위한 스택 언와인딩(Stack Unwinding for Exceptions)

### 9.1 문제

예외가 던져지면, 제어는 호출 스택 위의 여러 프레임을 건너 적절한 예외 처리기로 전달되어야 합니다. 그 사이의 모든 프레임은 적절하게 정리되어야 합니다(소멸자 호출, 리소스 해제).

### 9.2 접근 방법

#### 테이블 기반 언와인딩

현대 컴파일러(GCC, Clang)는 각 프레임의 레지스터를 복원하고 언와인드하는 방법을 설명하는 **언와인드 테이블**을 생성합니다. 이 테이블은 코드와 함께 저장되며 예외가 던져질 때만 참조됩니다.

```
.eh_frame 섹션:
  함수: foo
    오프셋 0에서:  CFA = RSP + 8
    오프셋 4에서:  CFA = RSP + 16, RBP = [CFA - 16]
    오프셋 8에서:  CFA = RBP + 16
```

**장점**: 예외가 던져지지 않을 때 비용 없음(정상 경로에 추가 명령어 없음).

**단점**: 테이블이 바이너리 크기를 증가시킵니다.

#### Setjmp/Longjmp 기반

구형 접근 방식: 각 try 블록에서 `setjmp`를 사용하여 현재 상태를 저장하고, 처리기로 점프하기 위해 `longjmp`를 사용합니다.

**장점**: 단순한 구현.

**단점**: `setjmp`는 예외가 던져지지 않아도 비용이 발생합니다.

### 9.3 스택 언와인딩 과정

```
baz()에서 예외가 던져질 때의 호출 스택:

main() → foo() → bar() → baz()
                           ↑ 여기서 예외 던져짐

언와인딩:
1. baz()에서 처리기 검색 → 없음
2. baz()의 프레임 언와인드 (정리 실행)
3. bar()에서 처리기 검색 → 없음
4. bar()의 프레임 언와인드 (정리 실행)
5. foo()에서 처리기 검색 → 발견
6. foo()의 catch 블록으로 제어 전달
```

---

## 10. Python 시뮬레이션: 런타임 호출 스택

```python
"""
활성화 레코드를 포함한 런타임 호출 스택 시뮬레이션.
함수 호출, 반환, 중첩 스코프를 시연합니다.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ActivationRecord:
    """
    함수 호출을 위한 활성화 레코드(스택 프레임).
    """
    function_name: str
    return_address: int                          # 반환할 명령어 인덱스
    parameters: dict = field(default_factory=dict)
    local_variables: dict = field(default_factory=dict)
    temporaries: dict = field(default_factory=dict)
    saved_registers: dict = field(default_factory=dict)
    static_link: Optional['ActivationRecord'] = None   # 중첩 함수용
    dynamic_link: Optional['ActivationRecord'] = None   # 호출자 프레임
    return_value: Any = None

    def get_variable(self, name: str) -> Any:
        """이 프레임에서 변수를 조회합니다."""
        if name in self.local_variables:
            return self.local_variables[name]
        if name in self.parameters:
            return self.parameters[name]
        return None

    def set_variable(self, name: str, value: Any):
        """지역 변수를 설정합니다."""
        self.local_variables[name] = value

    def __str__(self):
        parts = [f"  Frame: {self.function_name}()"]
        parts.append(f"    Return addr: {self.return_address}")
        if self.parameters:
            parts.append(f"    Parameters: {self.parameters}")
        if self.local_variables:
            parts.append(f"    Locals: {self.local_variables}")
        if self.temporaries:
            parts.append(f"    Temps: {self.temporaries}")
        if self.return_value is not None:
            parts.append(f"    Return value: {self.return_value}")
        if self.static_link:
            parts.append(f"    Static link -> {self.static_link.function_name}()")
        if self.dynamic_link:
            parts.append(f"    Dynamic link -> {self.dynamic_link.function_name}()")
        return "\n".join(parts)


class RuntimeStack:
    """
    런타임 호출 스택 시뮬레이션.
    """

    def __init__(self):
        self.frames: list[ActivationRecord] = []
        self.pc: int = 0  # 프로그램 카운터

    @property
    def current_frame(self) -> Optional[ActivationRecord]:
        return self.frames[-1] if self.frames else None

    @property
    def depth(self) -> int:
        return len(self.frames)

    def push_frame(self, function_name: str, parameters: dict,
                   return_address: int,
                   static_link: Optional[ActivationRecord] = None):
        """새 활성화 레코드 푸시 (함수 호출)."""
        frame = ActivationRecord(
            function_name=function_name,
            return_address=return_address,
            parameters=parameters,
            dynamic_link=self.current_frame,
            static_link=static_link,
        )
        self.frames.append(frame)
        print(f"\n>>> CALL {function_name}({parameters})")
        print(f"    Stack depth: {self.depth}")

    def pop_frame(self) -> ActivationRecord:
        """현재 활성화 레코드 팝 (함수 반환)."""
        if not self.frames:
            raise RuntimeError("Stack underflow!")

        frame = self.frames.pop()
        print(f"\n<<< RETURN from {frame.function_name}() "
              f"= {frame.return_value}")
        print(f"    Stack depth: {self.depth}")

        # 프로그램 카운터 복원
        self.pc = frame.return_address

        return frame

    def lookup_variable(self, name: str, use_static_scope: bool = True) -> Any:
        """
        정적 또는 동적 스코프 체인을 사용하여 변수를 조회합니다.
        """
        if use_static_scope:
            # 정적 링크 따라가기 (렉시컬 스코핑)
            frame = self.current_frame
            while frame is not None:
                value = frame.get_variable(name)
                if value is not None:
                    return value
                frame = frame.static_link
        else:
            # 동적 링크 따라가기 (동적 스코핑)
            frame = self.current_frame
            while frame is not None:
                value = frame.get_variable(name)
                if value is not None:
                    return value
                frame = frame.dynamic_link

        raise NameError(f"Variable '{name}' not found")

    def print_stack(self):
        """전체 호출 스택을 출력합니다."""
        print("\n=== Runtime Stack ===")
        if not self.frames:
            print("  (empty)")
            return
        for i in range(len(self.frames) - 1, -1, -1):
            marker = " ◀── TOP" if i == len(self.frames) - 1 else ""
            print(f"\n  [{i}]{marker}")
            print(self.frames[i])
        print("=" * 40)


def demo_factorial():
    """
    재귀 팩토리얼 함수 실행을 시뮬레이션합니다.

    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    result = factorial(4)
    """
    print("=" * 60)
    print("Simulating: result = factorial(4)")
    print("=" * 60)

    stack = RuntimeStack()

    # main 프레임 푸시
    stack.push_frame("main", {}, return_address=0)
    stack.current_frame.set_variable("result", None)

    # 재귀 호출: factorial(4) -> factorial(3) -> ... -> factorial(1)
    def simulate_factorial(n, return_addr):
        stack.push_frame("factorial", {"n": n}, return_address=return_addr)

        if n <= 1:
            stack.current_frame.return_value = 1
            stack.print_stack()
            returned = stack.pop_frame()
            return returned.return_value
        else:
            # 재귀 호출
            sub_result = simulate_factorial(n - 1, return_addr + 1)
            result = n * sub_result
            stack.current_frame.return_value = result
            returned = stack.pop_frame()
            return returned.return_value

    result = simulate_factorial(4, 10)

    stack.current_frame.set_variable("result", result)
    print(f"\nmain: result = {result}")

    stack.print_stack()

    # main 팝
    stack.current_frame.return_value = result
    stack.pop_frame()


def demo_nested_functions():
    """
    액세스 링크를 사용한 중첩 함수 호출을 시뮬레이션합니다.

    def outer():
        x = 10
        def inner():
            return x + 20   # 정적 링크를 통해 outer의 x에 접근
        return inner()

    result = outer()
    """
    print("\n" + "=" * 60)
    print("Simulating: nested functions with access links")
    print("=" * 60)

    stack = RuntimeStack()

    # main 프레임
    stack.push_frame("main", {}, return_address=0)

    # outer 프레임
    stack.push_frame("outer", {}, return_address=1)
    stack.current_frame.set_variable("x", 10)
    outer_frame = stack.current_frame

    # outer에 대한 정적 링크를 가진 inner 프레임
    stack.push_frame("inner", {}, return_address=2,
                     static_link=outer_frame)

    stack.print_stack()

    # inner는 정적 링크를 통해 x에 접근
    x = stack.lookup_variable("x", use_static_scope=True)
    result = x + 20
    print(f"\ninner: x (via static link) = {x}")
    print(f"inner: result = {result}")

    # inner에서 반환
    stack.current_frame.return_value = result
    stack.pop_frame()

    # outer에서 반환
    stack.current_frame.return_value = result
    stack.pop_frame()

    # main에 저장
    stack.current_frame.set_variable("result", result)
    print(f"\nmain: result = {result}")
    stack.pop_frame()


if __name__ == "__main__":
    demo_factorial()
    demo_nested_functions()
```

---

## 11. 런타임 환경에 대한 컴파일러 지원

### 11.1 컴파일러가 생성하는 것

컴파일러는 런타임 환경을 올바르게 관리하는 코드를 생성할 책임이 있습니다:

1. **함수 프롤로그(Function prologue)**: 각 함수 시작 시의 코드:
   - 이전 프레임 포인터 저장
   - 새 프레임 포인터 설정
   - 지역 변수 공간 할당
   - 피호출자 저장 레지스터 저장

2. **함수 에필로그(Function epilogue)**: 각 함수 끝의 코드:
   - 피호출자 저장 레지스터 복원
   - 지역 변수 해제
   - 이전 프레임 포인터 복원
   - 호출자에게 반환

3. **변수 접근 코드**: 스코프에 따라 변수의 주소를 계산하는 명령어:
   - 지역 변수: FP로부터의 오프셋
   - 매개변수: FP로부터의 양의 오프셋 (또는 레지스터)
   - 비지역 변수: 액세스 링크 따라가기 + 오프셋

4. **호출 위치 코드**: 각 함수 호출 시의 명령어:
   - 인수 평가 및 전달
   - 호출자 저장 레지스터 저장
   - 호출 수행
   - 반환값 처리

### 11.2 심볼 테이블 정보

컴파일러의 심볼 테이블은 다음을 기록해야 합니다:

| 정보 | 목적 |
|-------------|---------|
| 변수 타입 및 크기 | 스택 오프셋 및 접근 너비 결정 |
| 스코프 레벨 | 따라갈 액세스 링크 수 결정 |
| 프레임 내 오프셋 | 주소 계산 |
| 매개변수 인덱스 | 인수를 위한 레지스터 또는 스택 슬롯 결정 |
| 클로저에 캡처되었는지? | 그렇다면, 스택 대신 힙에 할당 |

### 11.3 프레임 레이아웃 최적화

현대 컴파일러는 프레임 레이아웃을 최적화합니다:

- **지역 변수 재배열**: 패딩(정렬)을 최소화
- **가능하면 프레임 포인터 생략**: SP 상대 주소 지정 사용, 레지스터 절약
- **레지스터 할당**: 자주 사용되는 변수를 스택 대신 레지스터에 유지
- **스택 슬롯 공유**: 수명이 겹치지 않는 변수가 동일한 스택 슬롯을 공유 가능

---

## 12. 요약

이 레슨에서는 프로그램이 런타임에 메모리에서 어떻게 구성되는지 살펴보았습니다:

1. **메모리 레이아웃**: 코드, 정적 데이터, 스택(아래 방향으로 성장), 힙(위 방향으로 성장)이 네 가지 주요 영역입니다.

2. **활성화 레코드**는 함수 호출에 필요한 모든 것을 저장합니다: 매개변수, 지역 변수, 반환 주소, 저장된 레지스터, 스코프 링크.

3. **호출 규약**(cdecl, stdcall, System V AMD64)은 인수 전달, 스택 정리, 레지스터 사용 프로토콜을 정의합니다. System V AMD64 ABI는 효율성을 위해 처음 6개의 정수 인수를 레지스터로 전달합니다.

4. **매개변수 전달 메커니즘**에는 값에 의한 전달(복사), 참조에 의한 전달(주소), 값-결과에 의한 전달(복사 입/출력), 이름에 의한 전달(썽크)이 있습니다.

5. **중첩 함수**는 둘러싸는 스코프의 변수에 도달하기 위해 액세스 링크 또는 디스플레이가 필요합니다. 클로저는 캡처된 변수를 둘러싸는 함수의 수명 이후에도 살려두어야 합니다.

6. **정적 스코핑**은 렉시컬 중첩으로 변수를 해석(컴파일 시점에 결정 가능)하고, **동적 스코핑**은 런타임 호출 체인으로 해석합니다.

7. **힙 관리**는 자유 리스트(최초 적합, 최적 적합) 또는 버디 시스템을 사용합니다. 가비지 컬렉션은 관리되는 언어에서 회수를 자동화합니다.

8. **예외를 위한 스택 언와인딩**은 정상 경로에서 비용 없는 예외 처리를 위해 테이블 기반 메커니즘을 사용합니다.

---

## 연습 문제

### 연습 1: 스택 프레임 다이어그램

System V AMD64에서 다음 C 함수 호출에 대한 완전한 스택 레이아웃(FP로부터의 실제 바이트 오프셋 포함)을 그리시오:

```c
int compute(int a, int b, int c, int d, int e, int f, int g, int h) {
    int x = a + b;
    int y = c + d;
    int z = e + f + g + h;
    return x + y + z;
}
```

참고: `a`--`f`는 레지스터로 전달; `g`와 `h`는 스택에 있습니다.

### 연습 2: 액세스 링크

다음 중첩 함수 구조에서 `innermost()`가 실행될 때 액세스 링크가 있는 스택 프레임을 그리시오:

```
function level0():
    var a = 1
    function level1():
        var b = 2
        function level2():
            var c = 3
            function level3():
                return a + b + c   // 어떻게 a, b, c를 찾을까?
            return level3()
        return level2()
    return level1()
```

`level3()`에서 `a`에 도달하려면 몇 개의 액세스 링크를 따라야 합니까?

### 연습 3: 호출 규약 비교

다음 함수 호출을 cdecl (32비트)와 System V AMD64 (64비트) 두 가지 방식으로 x86 어셈블리로 번역하시오:

```c
int result = multiply_add(2, 3, 4, 5, 6, 7, 8);
```

각 규약에 대한 인수 전달, 호출 명령어, 스택 정리를 보이시오.

### 연습 4: 동적 vs 정적 스코핑

다음 프로그램의 정적 스코핑과 동적 스코핑 하에서의 실행을 추적하시오. 각 경우에 `baz()`는 어떤 값을 반환합니까?

```
x = 1

function foo():
    return x

function bar():
    x = 2
    return foo()

function baz():
    x = 3
    return bar()
```

### 연습 5: 버디 시스템

512바이트의 총 메모리를 가진 버디 시스템 할당자가 주어졌을 때:

1. 크기 50, 120, 30, 60의 블록을 할당한 후의 상태를 보이시오
2. 50바이트와 120바이트 블록을 해제한 후의 상태를 보이시오
3. 병합이 발생합니까? 그렇다면 어떤 블록이 병합되는지 설명하시오.

### 연습 6: 구현 과제

`RuntimeStack` 시뮬레이션을 다음을 지원하도록 확장하시오:
1. **예외 처리**: 스택 언와인딩이 있는 try/catch 블록 구현
2. **클로저**: 함수가 중첩 함수를 반환할 때, 캡처된 변수가 여전히 접근 가능하도록 보장 (캡처된 변수의 힙 할당 시뮬레이션)

여러 프레임을 통해 예외를 던지는 프로그램으로 테스트하시오.

---

[이전: 중간 표현](./09_Intermediate_Representations.md) | [다음: 코드 생성](./11_Code_Generation.md) | [개요](./00_Overview.md)
