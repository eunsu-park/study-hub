# 형식 언어와 오토마타 이론(Formal Languages and Automata Theory)

## 개요

이 토픽은 계산의 수학적 기초를 다룹니다: 형식 언어(formal languages), 오토마타(automata), 계산 가능성(computability), 복잡도 이론(complexity theory). Compiler_Design 토픽이 이 개념들을 컴파일러 구축에 응용하는 반면, 이 토픽은 이론적 프레임워크 — 엄밀한 정의, 증명, 폐포 성질(closure properties), 그리고 계산의 근본적 한계 — 에 집중합니다.

## 선수 지식

- 기초 이산 수학(sets, functions, relations, proof techniques)
- 프로그래밍 기초(예제 코드 실행을 위해)
- 권장: Compiler_Design(실용적 맥락을 위해)

## 학습 경로

### 파트 I: 유한 오토마타와 정규 언어(Finite Automata and Regular Languages) (L01-L05)

| 레슨 | 제목 | 핵심 개념 |
|------|------|-----------|
| 01 | 형식 언어 입문 | 알파벳, 문자열, 언어, 언어 연산 |
| 02 | 결정적 유한 오토마타 | DFA 정의, 상태 다이어그램, 언어 인식 |
| 03 | 비결정적 유한 오토마타 | NFA, ε-전이, 부분집합 구성, DFA 동치 |
| 04 | 정규 표현식 | 구문, 의미론, 유한 오토마타와의 동치 |
| 05 | 정규 언어의 성질 | 펌핑 보조정리, 폐포 성질, Myhill-Nerode 정리, 최소화 |

### 파트 II: 문맥 자유 언어(Context-Free Languages) (L06-L08)

| 레슨 | 제목 | 핵심 개념 |
|------|------|-----------|
| 06 | 문맥 자유 문법 | 생성 규칙, 유도, 파스 트리, 모호성, 정규형 |
| 07 | 푸시다운 오토마타 | PDA 정의, 수리 방식, CFG와의 동치 |
| 08 | 문맥 자유 언어의 성질 | CFL 펌핑 보조정리, 폐포 성질, 결정 가능성 |

### 파트 III: 튜링 기계와 계산 가능성(Turing Machines and Computability) (L09-L11)

| 레슨 | 제목 | 핵심 개념 |
|------|------|-----------|
| 09 | 튜링 기계 | TM 정의, 변형, 처치-튜링 논제 |
| 10 | 결정 가능성 | 결정 가능 및 인식 가능 언어, 정지 문제 |
| 11 | 환원 가능성 | 매핑 환원, Rice의 정리, Post 대응 문제 |

### 파트 IV: 복잡도 이론과 응용(Complexity Theory and Applications) (L12-L14)

| 레슨 | 제목 | 핵심 개념 |
|------|------|-----------|
| 12 | 계산 복잡도 | 시간 복잡도, P, NP, NP-완전성, Cook-Levin 정리 |
| 13 | 촘스키 계층(The Chomsky Hierarchy) | 타입 0-3 언어, 폐포 요약, 결정 가능성 요약 |
| 14 | 심화 주제와 응용 | 람다 계산법, 문맥 민감 문법, 응용 |

## 다른 토픽과의 관계

- **Compiler_Design**: 어휘 분석에 DFA/NFA, 구문 분석에 CFG 적용 — 실용 지향
- **Algorithm**: 복잡도 클래스(P, NP)는 알고리즘 설계 및 분석과 연결
- **Math_for_AI**: 형식적 증명 기법, 수학적 추론
- **Quantum_Computing**: 양자 복잡도 클래스는 고전 복잡도 이론을 확장

## 예제 코드

실행 가능한 Python 예제가 [`examples/Formal_Languages/`](../../../examples/Formal_Languages/)에 있습니다. 핵심 알고리즘과 시뮬레이터를 구현하여 직접 실험해볼 수 있습니다.

## 참고문헌

- *Introduction to the Theory of Computation* by Michael Sipser (주요 참고서)
- *Introduction to Automata Theory, Languages, and Computation* by Hopcroft, Motwani, Ullman
- *Computational Complexity: A Modern Approach* by Arora and Barak
- *Automata and Computability* by Dexter Kozen
