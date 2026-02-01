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

## 실습 예제

`examples/` 폴더에 모든 레슨에 대한 **Python, C, C++ 세 버전**의 예제가 있습니다.

### 폴더 구조

```
examples/
├── python/           # Python 예제 (29개)
│   ├── 01_complexity.py
│   ├── 02_array_string.py
│   ├── ...
│   └── 29_practice.py
│
├── c/                # C 예제 (29개 + Makefile)
│   ├── 01_complexity.c
│   ├── 02_array_string.c
│   ├── ...
│   ├── 29_practice.c
│   └── Makefile
│
└── cpp/              # C++ 예제 (29개 + Makefile)
    ├── 01_complexity.cpp
    ├── 02_array_string.cpp
    ├── ...
    ├── 29_practice.cpp
    └── Makefile
```

### 예제 파일 목록

| 번호 | 주제 | Python | C | C++ |
|------|------|--------|---|-----|
| 01 | 복잡도 분석 | [01_complexity.py](./examples/python/01_complexity.py) | [01_complexity.c](./examples/c/01_complexity.c) | [01_complexity.cpp](./examples/cpp/01_complexity.cpp) |
| 02 | 배열/문자열 | [02_array_string.py](./examples/python/02_array_string.py) | [02_array_string.c](./examples/c/02_array_string.c) | [02_array_string.cpp](./examples/cpp/02_array_string.cpp) |
| 03 | 스택/큐 | [03_stack_queue.py](./examples/python/03_stack_queue.py) | [03_stack_queue.c](./examples/c/03_stack_queue.c) | [03_stack_queue.cpp](./examples/cpp/03_stack_queue.cpp) |
| 04 | 해시테이블 | [04_hash_table.py](./examples/python/04_hash_table.py) | [04_hash_table.c](./examples/c/04_hash_table.c) | [04_hash_table.cpp](./examples/cpp/04_hash_table.cpp) |
| 05 | 정렬 | [05_sorting.py](./examples/python/05_sorting.py) | [05_sorting.c](./examples/c/05_sorting.c) | [05_sorting.cpp](./examples/cpp/05_sorting.cpp) |
| 06 | 탐색 | [06_searching.py](./examples/python/06_searching.py) | [06_searching.c](./examples/c/06_searching.c) | [06_searching.cpp](./examples/cpp/06_searching.cpp) |
| 07 | 분할정복 | [07_divide_conquer.py](./examples/python/07_divide_conquer.py) | [07_divide_conquer.c](./examples/c/07_divide_conquer.c) | [07_divide_conquer.cpp](./examples/cpp/07_divide_conquer.cpp) |
| 08 | 백트래킹 | [08_backtracking.py](./examples/python/08_backtracking.py) | [08_backtracking.c](./examples/c/08_backtracking.c) | [08_backtracking.cpp](./examples/cpp/08_backtracking.cpp) |
| 09 | 트리/BST | [09_tree_bst.py](./examples/python/09_tree_bst.py) | [09_tree_bst.c](./examples/c/09_tree_bst.c) | [09_tree_bst.cpp](./examples/cpp/09_tree_bst.cpp) |
| 10 | 힙 | [10_heap.py](./examples/python/10_heap.py) | [10_heap.c](./examples/c/10_heap.c) | [10_heap.cpp](./examples/cpp/10_heap.cpp) |
| 11 | 트라이 | [11_trie.py](./examples/python/11_trie.py) | [11_trie.c](./examples/c/11_trie.c) | [11_trie.cpp](./examples/cpp/11_trie.cpp) |
| 12 | 그래프 기초 | [12_graph_basic.py](./examples/python/12_graph_basic.py) | [12_graph_basic.c](./examples/c/12_graph_basic.c) | [12_graph_basic.cpp](./examples/cpp/12_graph_basic.cpp) |
| 13 | 위상정렬 | [13_topological_sort.py](./examples/python/13_topological_sort.py) | [13_topological_sort.c](./examples/c/13_topological_sort.c) | [13_topological_sort.cpp](./examples/cpp/13_topological_sort.cpp) |
| 14 | 최단경로 | [14_shortest_path.py](./examples/python/14_shortest_path.py) | [14_shortest_path.c](./examples/c/14_shortest_path.c) | [14_shortest_path.cpp](./examples/cpp/14_shortest_path.cpp) |
| 15 | MST | [15_mst.py](./examples/python/15_mst.py) | [15_mst.c](./examples/c/15_mst.c) | [15_mst.cpp](./examples/cpp/15_mst.cpp) |
| 16 | LCA | [16_lca.py](./examples/python/16_lca.py) | [16_lca.c](./examples/c/16_lca.c) | [16_lca.cpp](./examples/cpp/16_lca.cpp) |
| 17 | SCC | [17_scc.py](./examples/python/17_scc.py) | [17_scc.c](./examples/c/17_scc.c) | [17_scc.cpp](./examples/cpp/17_scc.cpp) |
| 18 | 동적 프로그래밍 | [18_dp.py](./examples/python/18_dp.py) | [18_dp.c](./examples/c/18_dp.c) | [18_dp.cpp](./examples/cpp/18_dp.cpp) |
| 19 | 탐욕 | [19_greedy.py](./examples/python/19_greedy.py) | [19_greedy.c](./examples/c/19_greedy.c) | [19_greedy.cpp](./examples/cpp/19_greedy.cpp) |
| 20 | 비트마스크 DP | [20_bitmask_dp.py](./examples/python/20_bitmask_dp.py) | [20_bitmask_dp.c](./examples/c/20_bitmask_dp.c) | [20_bitmask_dp.cpp](./examples/cpp/20_bitmask_dp.cpp) |
| 21 | 정수론 | [21_number_theory.py](./examples/python/21_number_theory.py) | [21_number_theory.c](./examples/c/21_number_theory.c) | [21_number_theory.cpp](./examples/cpp/21_number_theory.cpp) |
| 22 | 문자열 | [22_string_algorithm.py](./examples/python/22_string_algorithm.py) | [22_string_algorithm.c](./examples/c/22_string_algorithm.c) | [22_string_algorithm.cpp](./examples/cpp/22_string_algorithm.cpp) |
| 23 | 세그먼트 트리 | [23_segment_tree.py](./examples/python/23_segment_tree.py) | [23_segment_tree.c](./examples/c/23_segment_tree.c) | [23_segment_tree.cpp](./examples/cpp/23_segment_tree.cpp) |
| 24 | 펜윅 트리 | [24_fenwick_tree.py](./examples/python/24_fenwick_tree.py) | [24_fenwick_tree.c](./examples/c/24_fenwick_tree.c) | [24_fenwick_tree.cpp](./examples/cpp/24_fenwick_tree.cpp) |
| 25 | 네트워크 플로우 | [25_network_flow.py](./examples/python/25_network_flow.py) | [25_network_flow.c](./examples/c/25_network_flow.c) | [25_network_flow.cpp](./examples/cpp/25_network_flow.cpp) |
| 26 | 기하 | [26_geometry.py](./examples/python/26_geometry.py) | [26_geometry.c](./examples/c/26_geometry.c) | [26_geometry.cpp](./examples/cpp/26_geometry.cpp) |
| 27 | 게임 이론 | [27_game_theory.py](./examples/python/27_game_theory.py) | [27_game_theory.c](./examples/c/27_game_theory.c) | [27_game_theory.cpp](./examples/cpp/27_game_theory.cpp) |
| 28 | 고급 DP | [28_advanced_dp.py](./examples/python/28_advanced_dp.py) | [28_advanced_dp.c](./examples/c/28_advanced_dp.c) | [28_advanced_dp.cpp](./examples/cpp/28_advanced_dp.cpp) |
| 29 | 실전 | [29_practice.py](./examples/python/29_practice.py) | [29_practice.c](./examples/c/29_practice.c) | [29_practice.cpp](./examples/cpp/29_practice.cpp) |

### 예제 실행 방법

```bash
# Python 예제 실행
python examples/python/01_complexity.py
python examples/python/23_segment_tree.py

# Python 전체 실행
for f in examples/python/*.py; do python "$f"; done

# C 예제 컴파일 및 실행
cd examples/c
make                    # 전체 컴파일
./01_complexity         # 개별 실행
make run-23             # Makefile로 실행
make test               # 전체 테스트

# 개별 C 파일 컴파일
gcc -Wall -std=c11 -O2 01_complexity.c -o 01_complexity -lm

# C++ 예제 컴파일 및 실행
cd examples/cpp
make                    # 전체 컴파일
./01_complexity         # 개별 실행
make run-23             # Makefile로 실행
make test               # 전체 테스트

# 개별 C++ 파일 컴파일
g++ -Wall -std=c++17 -O2 01_complexity.cpp -o 01_complexity
```

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
