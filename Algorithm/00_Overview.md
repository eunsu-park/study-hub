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
| [01_복잡도_분석.md](./01_복잡도_분석.md) | ⭐ | Big O, 시간/공간 복잡도, 분석 기법 |
| [02_배열과_문자열.md](./02_배열과_문자열.md) | ⭐ | 2포인터, 슬라이딩 윈도우, 프리픽스 합 |
| [03_스택과_큐_활용.md](./03_스택과_큐_활용.md) | ⭐ | 괄호 검사, 후위표기법, 모노토닉 스택 |
| [04_해시테이블.md](./04_해시테이블.md) | ⭐⭐ | 해시 함수, 충돌 해결, 해시맵/셋 구현 |
| [05_정렬_알고리즘.md](./05_정렬_알고리즘.md) | ⭐⭐ | 버블/선택/삽입/병합/퀵/힙 정렬 |

### 탐색과 분할정복 (06-08)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [06_탐색_알고리즘.md](./06_탐색_알고리즘.md) | ⭐⭐ | 이진탐색, 파라메트릭 서치, 해시 탐색 |
| [07_분할정복.md](./07_분할정복.md) | ⭐⭐ | 병합정렬, 퀵정렬, 빠른 거듭제곱 |
| [08_백트래킹.md](./08_백트래킹.md) | ⭐⭐⭐ | N-Queens, 순열/조합, 가지치기 |

### 트리 자료구조 (09-11)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [09_트리와_이진탐색트리.md](./09_트리와_이진탐색트리.md) | ⭐⭐ | 트리 순회, BST 연산, 균형 트리 |
| [10_힙과_우선순위큐.md](./10_힙과_우선순위큐.md) | ⭐⭐ | 힙 구조, 힙 정렬, K번째 원소 |
| [11_트라이.md](./11_트라이.md) | ⭐⭐⭐ | 접두사 트리, 자동완성, XOR 트라이 |

### 그래프 (12-17)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [12_그래프_기초.md](./12_그래프_기초.md) | ⭐⭐ | 그래프 표현, DFS, BFS |
| [13_위상정렬.md](./13_위상정렬.md) | ⭐⭐⭐ | Kahn, DFS 기반, 사이클 탐지 |
| [14_최단경로.md](./14_최단경로.md) | ⭐⭐⭐ | Dijkstra, Bellman-Ford, Floyd-Warshall |
| [15_최소신장트리.md](./15_최소신장트리.md) | ⭐⭐⭐ | Kruskal, Prim, Union-Find |
| [16_LCA와_트리쿼리.md](./16_LCA와_트리쿼리.md) | ⭐⭐⭐ | Binary Lifting, Sparse Table |
| [17_강한연결요소.md](./17_강한연결요소.md) | ⭐⭐⭐⭐ | Tarjan, Kosaraju, 2-SAT |

### DP와 수학 (18-22)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [18_동적_프로그래밍.md](./18_동적_프로그래밍.md) | ⭐⭐⭐ | 메모이제이션, 냅색, LCS, LIS |
| [19_탐욕_알고리즘.md](./19_탐욕_알고리즘.md) | ⭐⭐ | 활동 선택, 허프만 코딩, 탐욕 vs DP |
| [20_비트마스크_DP.md](./20_비트마스크_DP.md) | ⭐⭐⭐ | 비트 연산, TSP, 집합 DP |
| [21_수학과_정수론.md](./21_수학과_정수론.md) | ⭐⭐⭐ | 모듈러, GCD/LCM, 소수, 조합론 |
| [22_문자열_알고리즘.md](./22_문자열_알고리즘.md) | ⭐⭐⭐ | KMP, Rabin-Karp, Z-알고리즘 |

### 고급 자료구조 (23-24)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [23_세그먼트_트리.md](./23_세그먼트_트리.md) | ⭐⭐⭐⭐ | 구간 쿼리, Lazy Propagation |
| [24_펜윅_트리.md](./24_펜윅_트리.md) | ⭐⭐⭐ | BIT, 구간 합, 2D 펜윅 |

### 고급 그래프 (25)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [25_네트워크_플로우.md](./25_네트워크_플로우.md) | ⭐⭐⭐⭐ | Ford-Fulkerson, 이분 매칭 |

### 특수 주제 (26-28)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [26_기하_알고리즘.md](./26_기하_알고리즘.md) | ⭐⭐⭐ | CCW, 볼록껍질, 선분 교차 |
| [27_게임_이론.md](./27_게임_이론.md) | ⭐⭐⭐ | 님 게임, 스프라그-그런디, 미니맥스 |
| [28_고급_DP_최적화.md](./28_고급_DP_최적화.md) | ⭐⭐⭐⭐ | CHT, D&C 최적화, Knuth 최적화 |

### 마무리

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [29_실전_문제풀이.md](./29_실전_문제풀이.md) | ⭐⭐⭐⭐ | 유형별 접근법, LeetCode/백준 추천 |

---

## 추천 학습 순서

### 1단계: 기초 자료구조 (1주)
```
01_복잡도_분석 → 02_배열과_문자열 → 03_스택과_큐_활용 → 04_해시테이블
```

### 2단계: 정렬과 탐색 (1주)
```
05_정렬_알고리즘 → 06_탐색_알고리즘 → 07_분할정복 → 08_백트래킹
```

### 3단계: 트리 (1주)
```
09_트리와_이진탐색트리 → 10_힙과_우선순위큐 → 11_트라이
```

### 4단계: 그래프 기초 (1~2주)
```
12_그래프_기초 → 13_위상정렬 → 14_최단경로 → 15_최소신장트리
```

### 5단계: 고급 그래프 (1주)
```
16_LCA와_트리쿼리 → 17_강한연결요소
```

### 6단계: DP와 수학 (2주)
```
18_동적_프로그래밍 → 19_탐욕_알고리즘 → 20_비트마스크_DP → 21_수학과_정수론 → 22_문자열_알고리즘
```

### 7단계: 고급 자료구조 (1주)
```
23_세그먼트_트리 → 24_펜윅_트리
```

### 8단계: 고급 그래프 및 특수 주제 (2주+)
```
25_네트워크_플로우 → 26_기하_알고리즘 → 27_게임_이론 → 28_고급_DP_최적화
```

### 마무리
```
29_실전_문제풀이 (다른 단계와 병렬로 진행 권장)
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

`examples/` 폴더에 모든 레슨에 대한 **Python과 C 두 버전**의 예제가 있습니다.

### 폴더 구조

```
examples/
├── python/           # Python 예제 (29개)
│   ├── 01_complexity.py
│   ├── 02_array_string.py
│   ├── ...
│   └── 29_practice.py
│
└── c/                # C 예제 (29개 + Makefile)
    ├── 01_complexity.c
    ├── 02_array_string.c
    ├── ...
    ├── 29_practice.c
    └── Makefile
```

### 예제 파일 목록

| 번호 | 주제 | Python | C |
|------|------|--------|---|
| 01 | 복잡도 분석 | [01_complexity.py](./examples/python/01_complexity.py) | [01_complexity.c](./examples/c/01_complexity.c) |
| 02 | 배열/문자열 | [02_array_string.py](./examples/python/02_array_string.py) | [02_array_string.c](./examples/c/02_array_string.c) |
| 03 | 스택/큐 | [03_stack_queue.py](./examples/python/03_stack_queue.py) | [03_stack_queue.c](./examples/c/03_stack_queue.c) |
| 04 | 해시테이블 | [04_hash_table.py](./examples/python/04_hash_table.py) | [04_hash_table.c](./examples/c/04_hash_table.c) |
| 05 | 정렬 | [05_sorting.py](./examples/python/05_sorting.py) | [05_sorting.c](./examples/c/05_sorting.c) |
| 06 | 탐색 | [06_searching.py](./examples/python/06_searching.py) | [06_searching.c](./examples/c/06_searching.c) |
| 07 | 분할정복 | [07_divide_conquer.py](./examples/python/07_divide_conquer.py) | [07_divide_conquer.c](./examples/c/07_divide_conquer.c) |
| 08 | 백트래킹 | [08_backtracking.py](./examples/python/08_backtracking.py) | [08_backtracking.c](./examples/c/08_backtracking.c) |
| 09 | 트리/BST | [09_tree_bst.py](./examples/python/09_tree_bst.py) | [09_tree_bst.c](./examples/c/09_tree_bst.c) |
| 10 | 힙 | [10_heap.py](./examples/python/10_heap.py) | [10_heap.c](./examples/c/10_heap.c) |
| 11 | 트라이 | [11_trie.py](./examples/python/11_trie.py) | [11_trie.c](./examples/c/11_trie.c) |
| 12 | 그래프 기초 | [12_graph_basic.py](./examples/python/12_graph_basic.py) | [12_graph_basic.c](./examples/c/12_graph_basic.c) |
| 13 | 위상정렬 | [13_topological_sort.py](./examples/python/13_topological_sort.py) | [13_topological_sort.c](./examples/c/13_topological_sort.c) |
| 14 | 최단경로 | [14_shortest_path.py](./examples/python/14_shortest_path.py) | [14_shortest_path.c](./examples/c/14_shortest_path.c) |
| 15 | MST | [15_mst.py](./examples/python/15_mst.py) | [15_mst.c](./examples/c/15_mst.c) |
| 16 | LCA | [16_lca.py](./examples/python/16_lca.py) | [16_lca.c](./examples/c/16_lca.c) |
| 17 | SCC | [17_scc.py](./examples/python/17_scc.py) | [17_scc.c](./examples/c/17_scc.c) |
| 18 | 동적 프로그래밍 | [18_dp.py](./examples/python/18_dp.py) | [18_dp.c](./examples/c/18_dp.c) |
| 19 | 탐욕 | [19_greedy.py](./examples/python/19_greedy.py) | [19_greedy.c](./examples/c/19_greedy.c) |
| 20 | 비트마스크 DP | [20_bitmask_dp.py](./examples/python/20_bitmask_dp.py) | [20_bitmask_dp.c](./examples/c/20_bitmask_dp.c) |
| 21 | 정수론 | [21_number_theory.py](./examples/python/21_number_theory.py) | [21_number_theory.c](./examples/c/21_number_theory.c) |
| 22 | 문자열 | [22_string_algorithm.py](./examples/python/22_string_algorithm.py) | [22_string_algorithm.c](./examples/c/22_string_algorithm.c) |
| 23 | 세그먼트 트리 | [23_segment_tree.py](./examples/python/23_segment_tree.py) | [23_segment_tree.c](./examples/c/23_segment_tree.c) |
| 24 | 펜윅 트리 | [24_fenwick_tree.py](./examples/python/24_fenwick_tree.py) | [24_fenwick_tree.c](./examples/c/24_fenwick_tree.c) |
| 25 | 네트워크 플로우 | [25_network_flow.py](./examples/python/25_network_flow.py) | [25_network_flow.c](./examples/c/25_network_flow.c) |
| 26 | 기하 | [26_geometry.py](./examples/python/26_geometry.py) | [26_geometry.c](./examples/c/26_geometry.c) |
| 27 | 게임 이론 | [27_game_theory.py](./examples/python/27_game_theory.py) | [27_game_theory.c](./examples/c/27_game_theory.c) |
| 28 | 고급 DP | [28_advanced_dp.py](./examples/python/28_advanced_dp.py) | [28_advanced_dp.c](./examples/c/28_advanced_dp.c) |
| 29 | 실전 | [29_practice.py](./examples/python/29_practice.py) | [29_practice.c](./examples/c/29_practice.c) |

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

