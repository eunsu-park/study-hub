# 컴파일러 설계(Compiler Design)

## 토픽 개요

컴파일러 설계는 컴퓨터 과학의 핵심 분야 중 하나로, 사람이 읽을 수 있는 프로그래밍 언어와 기계가 실행할 수 있는 코드 사이의 간극을 잇습니다. 컴파일러는 고수준 언어로 작성된 소스 코드를 — 전형적으로 기계 코드, 바이트코드, 또는 다른 프로그래밍 언어로 — 일련의 명확히 정의된 단계를 거쳐 하위 수준의 표현으로 변환합니다.

이 토픽은 컴파일의 전체 파이프라인을 다룹니다: 원시 문자(raw character)를 토큰(token)으로 스캔하는 것부터 시작해, 토큰을 구조화된 표현으로 파싱하고, 의미를 분석하고, 중간 코드(intermediate code)를 생성하고, 이를 최적화하고, 최종적으로 목적 코드를 생성하는 것까지 다룹니다. 이 과정에서 형식 언어 이론(formal language theory), 알고리즘 설계, 그래프 이론, 컴퓨터 아키텍처와의 깊은 연관성을 발견하게 될 것입니다.

컴파일러 설계를 이해하는 것은 컴파일러를 직접 만들기 위해서뿐만 아니라 다음과 같은 목적으로도 유용합니다:

- **언어 설계(Language design)**: 도메인 특화 언어(DSL, Domain-Specific Language) 및 설정 언어 개발
- **도구 제작(Tool building)**: 린터(linter), 포매터(formatter), 트랜스파일러(transpiler), 정적 분석기(static analyzer) 작성
- **성능 이해(Performance understanding)**: 컴파일러가 수행하는 최적화(및 수행할 수 없는 최적화) 파악
- **소프트웨어 공학(Software engineering)**: 방문자(Visitor), 인터프리터(Interpreter), 중간 표현(Intermediate Representation) 같은 패턴을 일반 소프트웨어 설계에 적용
- **보안(Security)**: 코드 인젝션(code injection), 샌드박싱(sandboxing), 코드 분석 도구의 동작 원리 이해

## 학습 목표

이 토픽을 완료하면 다음을 수행할 수 있습니다:

1. 컴파일의 각 단계와 그 상호작용 설명
2. 유한 오토마타(finite automata) 이론을 사용하여 어휘 분석기(lexical analyzer) 구현
3. 하향식(top-down) 및 상향식(bottom-up) 기법을 사용하여 파서(parser) 구성
4. 추상 구문 트리(AST, Abstract Syntax Tree) 설계 및 의미 분석(semantic analysis) 수행
5. 중간 표현(intermediate representation) 생성 및 최적화 수행
6. 런타임 환경(runtime environment), 가비지 컬렉션(garbage collection), 가상 머신(virtual machine) 이해
7. 실용적인 컴파일 작업을 위해 현대적인 컴파일러 인프라(LLVM) 활용
8. 소규모 언어를 위한 간단한 종단 간(end-to-end) 컴파일러 또는 인터프리터 구축

## 선수 지식

컴파일러 설계를 학습하기 전에 다음 내용에 익숙해야 합니다:

- **Algorithm** (이 프로젝트): 특히 그래프 알고리즘, 트리, 동적 프로그래밍
- **Programming** (이 프로젝트): 자료구조, 재귀, OOP에 대한 확실한 이해
- **Python** (이 프로젝트): 대부분의 예제는 명확성을 위해 Python 사용
- **C_Basics** (이 프로젝트): 저수준 코드 생성 이해에 도움
- **이산 수학(Discrete mathematics)**: 집합, 관계, 함수, 증명 기법(귀납법, 모순법)
- **기본 컴퓨터 아키텍처(Basic computer architecture)**: 레지스터, 메모리, 명령어 세트 (Computer_Architecture에서 다룸)

## 레슨 목록

| # | 제목 | 핵심 주제 | 예상 시간 |
|---|-------|------------|----------------|
| [01](./01_Introduction_to_Compilers.md) | 컴파일러 입문 | 컴파일러 구조, 단계, 부트스트래핑, T-다이어그램 | 2-3시간 |
| [02](./02_Lexical_Analysis.md) | 어휘 분석 | 토큰, 정규 표현식, DFA/NFA, 렉서 구현 | 3-4시간 |
| [03](./03_Finite_Automata.md) | 유한 오토마타 | NFA-DFA 변환, 최소화, Myhill-Nerode, Lex/Flex | 3-4시간 |
| [04](./04_Context_Free_Grammars.md) | 문맥 자유 문법 | BNF, 유도, 파스 트리, 모호성, CNF, CYK | 3-4시간 |
| [05](./05_Top_Down_Parsing.md) | 하향식 파싱 | 재귀 하강, LL(1), FIRST/FOLLOW 집합 | 3-4시간 |
| [06](./06_Bottom_Up_Parsing.md) | 상향식 파싱 | LR(0), SLR, LALR, 파서 생성기 (Yacc/Bison) | 4-5시간 |
| [07](./07_Abstract_Syntax_Trees.md) | 추상 구문 트리 | AST 설계, 방문자 패턴, 트리 순회 전략 | 2-3시간 |
| [08](./08_Semantic_Analysis.md) | 의미 분석 | 타입 검사, 심볼 테이블, 범위 규칙 | 3-4시간 |
| [09](./09_Intermediate_Representations.md) | 중간 표현 | 3-주소 코드, SSA 형식, 제어 흐름 그래프 | 3-4시간 |
| [10](./10_Runtime_Environments.md) | 런타임 환경 | 활성화 레코드, 호출 규약, 스택 프레임 | 3-4시간 |
| [11](./11_Code_Generation.md) | 코드 생성 | 명령어 선택, 레지스터 할당, 타일링 | 3-4시간 |
| [12](./12_Optimization_Local_and_Global.md) | 최적화 -- 지역 및 전역 | 데이터 흐름 분석, 상수 전파, 죽은 코드 제거 | 4-5시간 |
| [13](./13_Loop_Optimization.md) | 루프 최적화 | 루프 불변 코드 이동, 강도 감소, 벡터화 | 3-4시간 |
| [14](./14_Garbage_Collection.md) | 가비지 컬렉션 | Mark-Sweep, 복사, 세대별, 참조 카운팅 | 3-4시간 |
| [15](./15_Interpreters_and_Virtual_Machines.md) | 인터프리터와 가상 머신 | 바이트코드, 스택 기반 VM, JIT 컴파일 기초 | 3-4시간 |
| [16](./16_Modern_Compiler_Infrastructure.md) | 현대 컴파일러 인프라 | LLVM IR, 패스 구조, DSL 설계 | 3-4시간 |
| [17](./17_Garbage_Collection.md) | 가비지 컬렉션 -- 심화 | 동시 수집기, 삼색 마킹, 쓰기 배리어, G1/ZGC/Shenandoah, 탈출 분석 | 3-4시간 |
| [18](./18_SSA_Form.md) | SSA 형식 | 정적 단일 할당, 파이 함수, 지배 프론티어, SSA 구성 | 4-5시간 |
| [19](./19_Register_Allocation.md) | 레지스터 할당 | 그래프 색칠, 선형 스캔, 생존 범위 분할, 스필 전략 | 4-5시간 |
| [20](./20_LLVM_IR_Introduction.md) | LLVM IR 입문 | LLVM IR 구문, 타입, 명령어, 기본 블록, 함수 | 3-4시간 |
| [21](./21_LLVM_Backend.md) | LLVM 백엔드 | LLVM 패스 파이프라인, 커스텀 패스, SelectionDAG, 명령어 선택 | 4-5시간 |
| [22](./22_JIT_Compilation.md) | JIT 컴파일 | JIT 기초, 추적 JIT, 메서드 JIT, LLVM ORC, libgccjit | 3-4시간 |
| [23](./23_Type_Inference.md) | 타입 추론 | Hindley-Milner, Algorithm W, 단일화, 제약 기반 타이핑 | 4-5시간 |
| [24](./24_Closure_Conversion.md) | 클로저 변환 | 람다 리프팅, 클로저 변환, 탈함수화, CPS 변환 | 3-4시간 |
| [25](./25_Linking_and_Loading.md) | 링킹과 로딩 | 오브젝트 파일, ELF 형식, 정적/동적 링킹, 심볼 해석 | 3-4시간 |
| [26](./26_Debug_Information.md) | 디버그 정보 | DWARF 형식, 소스 맵, 디버그 정보 생성, 디버거 통합 | 3-4시간 |
| [27](./27_Modern_Parser_Generators.md) | 현대 파서 생성기 | tree-sitter, PEG, Packrat 파싱, 증분 파싱, LSP 통합 | 3-4시간 |
| [28](./28_Capstone_Compiler_Project.md) | 캡스톤 컴파일러 프로젝트 | 완전한 컴파일러 구축: 토이 언어에서 LLVM 코드 생성까지 | 8-10시간 |

**총 예상 시간: 98-129시간**

## 학습 경로 추천

### 경로 1: 기초 우선 (권장)

레슨을 순서대로 따릅니다. 이 경로는 각 개념을 이전 개념 위에 쌓아 올립니다:

```
01 Introduction
    |
02 Lexical Analysis  --->  03 Finite Automata
    |
04 Context-Free Grammars
    |
    +---> 05 Top-Down Parsing
    +---> 06 Bottom-Up Parsing
    |
07 Abstract Syntax Trees
    |
08 Semantic Analysis
    |
09 Intermediate Representations
    |
    +---> 10 Runtime Environments
    +---> 11 Code Generation
    +---> 12 Local/Global Optimization
    +---> 13 Loop Optimization
    |
14 Garbage Collection
    |
15 Interpreters and VMs
    |
16 Modern Compiler Infrastructure
    |
17 Garbage Collection (Advanced)
    |
18 SSA Form
    |
    +---> 19 Register Allocation
    +---> 20 LLVM IR Introduction
    |
21 LLVM Backend
    |
22 JIT Compilation
    |
    +---> 23 Type Inference
    +---> 24 Closure Conversion
    |
25 Linking and Loading
    |
26 Debug Information
    |
27 Modern Parser Generators
    |
28 Capstone Compiler Project
```

### 경로 2: 빠른 실용 컴파일러

동작하는 컴파일러 또는 인터프리터를 빠르게 만들고 싶다면:

```
01 Introduction  -->  02 Lexical Analysis  -->  04 CFGs  -->  05 Top-Down Parsing
    -->  07 ASTs  -->  08 Semantic Analysis  -->  15 Interpreters and VMs
```

### 경로 3: 최적화 집중

컴파일러 최적화와 성능에 관심이 있다면:

```
01 Introduction  -->  09 Intermediate Representations  -->  12 Local/Global Optimization
    -->  13 Loop Optimization  -->  11 Code Generation  -->  16 LLVM
```

### 경로 4: 언어 이론

형식 언어 이론과 파싱에 관심이 있다면:

```
02 Lexical Analysis  -->  03 Finite Automata  -->  04 CFGs
    -->  05 Top-Down Parsing  -->  06 Bottom-Up Parsing
```

## 이 프로젝트의 관련 토픽

| 토픽 | 관련성 |
|-------|-----------|
| **Algorithm** | 데이터 흐름 분석 및 최적화에 사용되는 그래프 알고리즘 (DFS, 위상 정렬) |
| **OS_Theory** | 프로세스 메모리 레이아웃, 링킹(linking), 로딩(loading) — 런타임 환경과 직접 연관 |
| **Computer_Architecture** | 명령어 세트, 레지스터, 캐시 — 코드 생성에 필수적 |
| **Programming** | 컴파일러 구성에 광범위하게 사용되는 설계 패턴 (방문자, 인터프리터) |
| **Python** | 이 토픽의 대부분 예제에서 사용하는 구현 언어 |
| **C_Basics** | 목적 언어 이해; 많은 컴파일러가 C로 작성됨 |
| **Math_for_AI** | 선형 대수 및 그래프 이론 개념이 최적화 알고리즘과 중첩 |

## 추천 교재 및 참고 자료

1. **Aho, Lam, Sethi, Ullman** — *Compilers: Principles, Techniques, and Tools* (2판, "드래곤 북(Dragon Book)")
2. **Cooper, Torczon** — *Engineering a Compiler* (2판)
3. **Appel** — *Modern Compiler Implementation in ML/Java/C*
4. **Muchnick** — *Advanced Compiler Design and Implementation*
5. **Grune, Bal, Jacobs, Langendoen** — *Modern Compiler Design* (2판)

## 예제 코드

이 토픽의 예제 코드는 [`examples/Compiler_Design/`](../../../examples/Compiler_Design/)에서 찾을 수 있습니다.

예제에는 다음이 포함됩니다:
- 렉서(Lexer) 구현 (Python)
- 파서(Parser) 구현 (재귀 하강, 연산자 우선순위)
- AST 구성 및 순회
- 간단한 인터프리터 및 바이트코드 VM
- 최적화 패스(Optimization pass)

---

*이 토픽은 [학습 자료](../../README.md) 컬렉션의 일부입니다.*
