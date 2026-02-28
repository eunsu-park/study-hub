# 알고리즘 학습 가이드

## 소개

이 폴더는 알고리즘과 자료구조를 체계적으로 학습하기 위한 자료를 담고 있습니다. 복잡도 분석부터 고급 알고리즘까지 단계별로 학습할 수 있으며, 코딩 인터뷰와 알고리즘 대회 준비에 도움이 됩니다.

**대상 독자**: 프로그래밍 기초를 아는 개발자, 코딩 테스트 준비자

---

## 학습 로드맵

```
[기초]                    [중급]                    [고급]
  │                         │                         │
  ▼                         ▼                         ▼
복잡도 분석 ─────▶ 분할 정복 ───────▶ 동적 프로그래밍
  │                    │                      │
  ▼                    ▼                      ▼
배열/문자열 ────▶ 트리/BST ────────▶ 세그먼트 트리
  │                    │                      │
  ▼                    ▼                      ▼
스택/큐 ────────▶ 그래프 기초 ─────▶ 네트워크 플로우
  │                    │                      │
  ▼                    ▼                      ▼
해시테이블 ────▶ 최단경로/MST ────▶ 고급 DP 최적화
```

---

## 선수 지식

- 프로그래밍 기초 (변수, 제어문, 함수)
- 기본 자료구조 (배열, 리스트)
- C, C++, 또는 Python 중 하나 이상의 언어

---

## 파일 목록

### 기초 자료구조 (01-05)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_Complexity_Analysis.md](./01_Complexity_Analysis.md) | ⭐ | Big O, 시간/공간 복잡도, 분석 기법 |
| [02_Arrays_and_Strings.md](./02_Arrays_and_Strings.md) | ⭐ | 2포인터, 슬라이딩 윈도우, 프리픽스 합 |
| [03_Stacks_and_Queues.md](./03_Stacks_and_Queues.md) | ⭐ | 괄호 검사, 후위표기법, 모노토닉 스택 |
| [04_Hash_Tables.md](./04_Hash_Tables.md) | ⭐⭐ | 해시 함수, 충돌 해결, 해시맵/셋 구현 |
| [05_Sorting_Algorithms.md](./05_Sorting_Algorithms.md) | ⭐⭐ | 버블/선택/삽입/병합/퀵/힙 정렬 |

### 탐색과 분할정복 (06-08)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [06_Searching_Algorithms.md](./06_Searching_Algorithms.md) | ⭐⭐ | 이진탐색, 파라메트릭 서치, 해시 탐색 |
| [07_Divide_and_Conquer.md](./07_Divide_and_Conquer.md) | ⭐⭐ | 병합정렬, 퀵정렬, 빠른 거듭제곱 |
| [08_Backtracking.md](./08_Backtracking.md) | ⭐⭐⭐ | N-Queens, 순열/조합, 가지치기 |

### 트리 자료구조 (09-11)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [09_Trees_and_BST.md](./09_Trees_and_BST.md) | ⭐⭐ | 트리 순회, BST 연산, 균형 트리 |
| [10_Heaps_and_Priority_Queues.md](./10_Heaps_and_Priority_Queues.md) | ⭐⭐ | 힙 구조, 힙 정렬, K번째 원소 |
| [11_Trie.md](./11_Trie.md) | ⭐⭐⭐ | 접두사 트리, 자동완성, XOR 트라이 |

### 그래프 (12-17)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [12_Graph_Basics.md](./12_Graph_Basics.md) | ⭐⭐ | 그래프 표현, DFS, BFS |
| [13_Topological_Sort.md](./13_Topological_Sort.md) | ⭐⭐⭐ | Kahn, DFS 기반, 사이클 탐지 |
| [14_Shortest_Path.md](./14_Shortest_Path.md) | ⭐⭐⭐ | Dijkstra, Bellman-Ford, Floyd-Warshall |
| [15_Minimum_Spanning_Tree.md](./15_Minimum_Spanning_Tree.md) | ⭐⭐⭐ | Kruskal, Prim, Union-Find |
| [16_LCA_and_Tree_Queries.md](./16_LCA_and_Tree_Queries.md) | ⭐⭐⭐ | Binary Lifting, Sparse Table |
| [17_Strongly_Connected_Components.md](./17_Strongly_Connected_Components.md) | ⭐⭐⭐⭐ | Tarjan, Kosaraju, 2-SAT |

### DP와 수학 (18-22)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [18_Dynamic_Programming.md](./18_Dynamic_Programming.md) | ⭐⭐⭐ | 메모이제이션, 냅색, LCS, LIS |
| [19_Greedy_Algorithms.md](./19_Greedy_Algorithms.md) | ⭐⭐ | 활동 선택, 허프만 코딩, 탐욕 vs DP |
| [20_Bitmask_DP.md](./20_Bitmask_DP.md) | ⭐⭐⭐ | 비트 연산, TSP, 집합 DP |
| [21_Math_and_Number_Theory.md](./21_Math_and_Number_Theory.md) | ⭐⭐⭐ | 모듈러, GCD/LCM, 소수, 조합론 |
| [22_String_Algorithms.md](./22_String_Algorithms.md) | ⭐⭐⭐ | KMP, Rabin-Karp, Z-알고리즘 |

### 고급 자료구조 (23-24)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [23_Segment_Tree.md](./23_Segment_Tree.md) | ⭐⭐⭐⭐ | 구간 쿼리, Lazy Propagation |
| [24_Fenwick_Tree.md](./24_Fenwick_Tree.md) | ⭐⭐⭐ | BIT, 구간 합, 2D 펜윅 |

### 고급 그래프 (25)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [25_Network_Flow.md](./25_Network_Flow.md) | ⭐⭐⭐⭐ | Ford-Fulkerson, 이분 매칭 |

### 특수 주제 (26-28)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [26_Computational_Geometry.md](./26_Computational_Geometry.md) | ⭐⭐⭐ | CCW, 볼록껍질, 선분 교차 |
| [27_Game_Theory.md](./27_Game_Theory.md) | ⭐⭐⭐ | 님 게임, 스프라그-그런디, 미니맥스 |
| [28_Advanced_DP_Optimization.md](./28_Advanced_DP_Optimization.md) | ⭐⭐⭐⭐ | CHT, D&C 최적화, Knuth 최적화 |

### 고급 자료구조 II (30-32)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [Heavy-Light Decomposition](./30_Heavy_Light_Decomposition.md) | ⭐⭐⭐⭐⭐ | HLD, 트리 경로 쿼리 |
| [링크-컷 트리](./31_Link_Cut_Tree.md) | ⭐⭐⭐⭐⭐ | 동적 트리 연결성, 접근 보조정리 |
| [영속적 세그먼트 트리](./32_Persistent_Segment_Tree.md) | ⭐⭐⭐⭐⭐ | 영속적 자료구조, 세그먼트 트리 버전 관리 |

### 마무리

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [29_Problem_Solving.md](./29_Problem_Solving.md) | ⭐⭐⭐⭐ | 유형별 접근법, LeetCode/백준 추천 |

---

## 추천 학습 순서

### 1단계: 기초 자료구조 (1주)
```
01 → 02 → 03 → 04
```

### 2단계: 정렬과 탐색 (1주)
```
05 → 06 → 07 → 08
```

### 3단계: 트리 (1주)
```
09 → 10 → 11
```

### 4단계: 그래프 기초 (1~2주)
```
12 → 13 → 14 → 15
```

### 5단계: 고급 그래프 (1주)
```
16 → 17
```

### 6단계: DP와 수학 (2주)
```
18 → 19 → 20 → 21 → 22
```

### 7단계: 고급 자료구조 (1주)
```
23 → 24
```

### 8단계: 고급 그래프 및 특수 주제 (2주+)
```
25 → 26 → 27 → 28
```

### 마무리
```
29 (다른 단계와 병렬로 진행 권장)
```

---

## 실습 환경

### 언어별 준비

```bash
# C/C++ (GCC)
gcc --version
g++ --version

# Python
python3 --version

# 온라인 환경
# - https://www.onlinegdb.com/
# - https://replit.com/
```

### 추천 도구

- **IDE**: VS Code, CLion, PyCharm
- **온라인 저지**: 백준(BOJ), LeetCode, 프로그래머스
- **시각화**: https://visualgo.net/

---

## 복잡도 빠른 참조

| 복잡도 | 이름 | 예시 | n=10^6 기준 |
|--------|------|------|-------------|
| O(1) | 상수 | 배열 접근 | 즉시 |
| O(log n) | 로그 | 이진 탐색 | ~20 연산 |
| O(n) | 선형 | 순차 탐색 | 10^6 연산 |
| O(n log n) | 선형 로그 | 병합 정렬 | ~2×10^7 연산 |
| O(n²) | 제곱 | 버블 정렬 | 10^12 연산 (주의) |
| O(2^n) | 지수 | 부분집합 | 불가능 |

---

## 관련 자료

### 다른 폴더와의 연계

| 폴더 | 관련 내용 |
|------|----------|
| [C_Programming/](../C_Programming/00_Overview.md) | 동적 배열, 연결 리스트, 해시 테이블 구현 |
| [CPP/](../CPP/00_Overview.md) | STL 컨테이너, algorithm 헤더 활용 |
| [Python/](../Python/00_Overview.md) | 리스트 컴프리헨션, itertools 활용 |

### 외부 자료

- [백준 온라인 저지](https://www.acmicpc.net/)
- [LeetCode](https://leetcode.com/)
- [프로그래머스](https://programmers.co.kr/)
- [VisuAlgo - 알고리즘 시각화](https://visualgo.net/)

---

## 학습 팁

1. **직접 구현**: 라이브러리 사용 전에 직접 구현해보기
2. **복잡도 분석**: 모든 코드에 대해 시간/공간 복잡도 계산
3. **다양한 언어**: C, C++, Python으로 같은 알고리즘 구현
4. **문제 풀이**: 각 레슨 후 관련 문제 3~5개 풀기
5. **오답 노트**: 틀린 문제는 왜 틀렸는지 기록

---

## 코딩 테스트 준비 가이드

### 난이도별 목표

| 수준 | 목표 | 필수 레슨 |
|------|------|----------|
| 입문 | 기초 문제 해결 | 01~08 |
| 중급 | 대기업 코딩테스트 | 01~17 |
| 고급 | 알고리즘 대회 | 01~25 |
| 최상급 | ICPC/IOI 수준 | 전체 (26~29 포함) |

### 시간 제한 기준

- **1초**: O(n log n) 이하 (n ≤ 10^6)
- **2초**: O(n²) 가능 (n ≤ 5000)
- **5초**: O(n²) 가능 (n ≤ 10000)
