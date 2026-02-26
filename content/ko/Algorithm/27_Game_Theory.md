# 28. 게임 이론 (Game Theory)

## 학습 목표
- 조합 게임 이론의 기본 개념 이해
- 승리 상태와 패배 상태 분석
- 님 게임(Nim Game) 해결
- 스프라그-그런디 정리 활용
- 미니맥스 알고리즘 구현

## 1. 조합 게임 이론 기초

### 조합 게임의 특성

```
┌─────────────────────────────────────────────────┐
│          조합 게임 (Combinatorial Game)          │
├─────────────────────────────────────────────────┤
│  1. 두 명의 플레이어가 번갈아 가며 플레이         │
│  2. 완전 정보 게임 (모든 정보가 공개됨)           │
│  3. 운의 요소가 없음 (결정론적)                  │
│  4. 유한한 수의 상태                            │
│  5. 이동할 수 없는 플레이어가 패배               │
└─────────────────────────────────────────────────┘
```

### 승리 상태 (W) vs 패배 상태 (L)

```
패배 상태 (L): 움직일 수 있는 모든 상태가 W
승리 상태 (W): 움직일 수 있는 상태 중 L이 하나라도 있음

    Start
      |
      W ←── 상대를 L로 보낼 수 있음
     / \
    L   W
    |   |
    W   L ←── 모든 이동이 W로만 감

규칙: L에서 시작하면 패배, W에서 시작하면 승리
```

### 기본 예제: 돌 가져가기 게임

```python
def stone_game(n, moves):
    """
    n개의 돌에서 moves 리스트의 개수만큼 가져갈 수 있음
    마지막 돌을 가져가는 사람이 승리
    """
    # dp[i] = True면 승리 상태
    dp = [False] * (n + 1)

    for i in range(1, n + 1):
        for move in moves:
            if i >= move and not dp[i - move]:
                dp[i] = True
                break

    return dp[n]

# 예: 돌 10개, 한 번에 1, 2, 3개 가져갈 수 있음
print(stone_game(10, [1, 2, 3]))  # True
```

---

## 2. 님 게임 (Nim Game)

### 규칙

- 여러 더미의 돌이 있음
- 각 턴에 한 더미에서 1개 이상의 돌을 가져감
- 마지막 돌을 가져가는 사람이 승리

```
더미: [3, 4, 5]

Player 1: 더미 2에서 2개 가져감 → [3, 2, 5]
Player 2: 더미 3에서 3개 가져감 → [3, 2, 2]
...
```

### 님 정리 (Nim Theorem)

**XOR Sum = 0이면 패배, ≠ 0이면 승리**

```
예: [3, 4, 5]
3 = 011
4 = 100
5 = 101
---------
XOR = 010 = 2 ≠ 0 → 첫 번째 플레이어 승리

예: [1, 2, 3]
1 = 01
2 = 10
3 = 11
--------
XOR = 00 = 0 → 첫 번째 플레이어 패배
```

### 구현

```python
def nim_game(piles):
    """
    님 게임의 승자 판별
    Returns: True면 첫 번째 플레이어 승리
    XOR 정리가 성립하는 이유: XOR = 0은 모든 비트가 더미들 사이에서 "균형"을 이룬다는 뜻 --
    하나의 더미를 바꾸는 어떤 수도 어떤 비트의 균형을 깨뜨리고, 두 번째 플레이어는
    항상 균형을 복원할 수 있어서, 결국 첫 번째 플레이어에게 모두-0인 더미를 남김.
    """
    xor_sum = 0
    for pile in piles:
        xor_sum ^= pile  # 모든 더미에 걸쳐 비트별 XOR 누적

    return xor_sum != 0  # XOR이 0이 아니면 첫 번째 플레이어가 항상 0으로 만들 수 있음

def nim_winning_move(piles):
    """
    승리를 위한 최적 수 찾기
    Returns: (더미 인덱스, 가져갈 돌 수) 또는 None
    전략: 어떤 더미를 줄여서 전체 XOR = 0으로 만들어 상대를 패배 상태로 몰아넣기.
    """
    xor_sum = 0
    for pile in piles:
        xor_sum ^= pile

    if xor_sum == 0:
        return None  # 이미 패배 상태 -- 어떤 수를 두든 상대에게 승리 상태를 만들어 줌

    # xor_sum과 XOR하면 줄어드는 더미를 찾기.
    # target = pile ^ xor_sum은 새 XOR 합이 0이 되도록 해당 더미가 되어야 할 값;
    # target < pile이면 돌을 제거하는 것(추가가 아닌)이 보장됨.
    for i, pile in enumerate(piles):
        target = pile ^ xor_sum
        if target < pile:
            return (i, pile - target)  # 더미 i에서 (pile - target)개 가져가기

    return None

# 사용 예시
piles = [3, 4, 5]
if nim_game(piles):
    print("첫 번째 플레이어 승리!")
    move = nim_winning_move(piles)
    print(f"최적 수: 더미 {move[0]}에서 {move[1]}개 가져가기")
else:
    print("두 번째 플레이어 승리!")
```

### 변형: Misère Nim

마지막 돌을 가져가는 사람이 **패배**하는 버전

```python
def misere_nim(piles):
    """
    Misère Nim: 마지막 돌을 가져가면 패배
    """
    xor_sum = 0
    all_ones = True

    for pile in piles:
        xor_sum ^= pile
        if pile > 1:
            all_ones = False

    if all_ones:
        # 모든 더미가 1개 이하면, 더미 개수가 홀수면 패배
        return len([p for p in piles if p > 0]) % 2 == 0
    else:
        # 일반 님과 동일
        return xor_sum != 0
```

---

## 3. 스프라그-그런디 정리

### Grundy Number (Nimber)

모든 조합 게임 상태에는 **Grundy number**가 있고, 이는 해당 상태와 동등한 님 더미의 크기입니다.

```
Grundy(state) = mex({Grundy(next_state) for all next_state})

mex = minimum excludant (집합에 없는 최소 비음수 정수)
mex({0, 1, 2}) = 3
mex({0, 2, 3}) = 1
mex({1, 2}) = 0
```

### 규칙

```
Grundy = 0: 패배 상태 (L)
Grundy > 0: 승리 상태 (W)

여러 독립 게임의 결합:
Grundy(G1 + G2) = Grundy(G1) XOR Grundy(G2)
```

### 구현

```python
def calculate_grundy(state, get_next_states, memo=None):
    """
    상태의 Grundy number 계산.
    Grundy number는 게임 위치의 전체 전략적 가치를 해당 크기의 님 더미와 동등한
    단일 정수로 인코딩 -- 이것이 독립 게임들을 XOR로 결합할 수 있는 이유.
    """
    if memo is None:
        memo = {}

    if state in memo:
        return memo[state]  # 메모이제이션으로 중복 상태의 지수적 재계산을 방지

    next_states = get_next_states(state)

    if not next_states:
        # 종료 상태: 이동 불가는 현재 플레이어의 패배 (정상 플레이 규칙)
        # 패배 상태는 빈 님 더미와 동등 -- Grundy number 0
        memo[state] = 0
        return 0

    # 도달 가능한 모든 다음 상태의 Grundy number 수집
    grundy_set = set()
    for next_state in next_states:
        grundy_set.add(calculate_grundy(next_state, get_next_states, memo))

    # mex (minimum excludant) = grundy_set에 없는 가장 작은 비음수 정수.
    # "첫 번째 빈 자리"를 나타냄 -- 현재 상태는 도달 가능한 위치의
    # Grundy number와 같은 크기로 줄일 수 없는 님 더미처럼 동작.
    mex = 0
    while mex in grundy_set:
        mex += 1

    memo[state] = mex
    return mex

# 예: 1~3개 가져가기 게임
def get_next_states(n):
    """n개 돌에서 1, 2, 3개를 가져간 후 상태"""
    return [n - k for k in [1, 2, 3] if n >= k]

for n in range(10):
    g = calculate_grundy(n, get_next_states)
    print(f"Grundy({n}) = {g}")
# 출력: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1
# 패턴: n % 4
```

### 복합 게임

```python
def combined_game_winner(games):
    """
    여러 독립 게임의 결합
    games: 각 게임의 Grundy number 리스트
    """
    xor_sum = 0
    for g in games:
        xor_sum ^= g

    return xor_sum != 0

# 예: 세 개의 돌 게임 (5개, 7개, 3개)
grundy_5 = calculate_grundy(5, get_next_states)  # 1
grundy_7 = calculate_grundy(7, get_next_states)  # 3
grundy_3 = calculate_grundy(3, get_next_states)  # 3

# 결합: 1 XOR 3 XOR 3 = 1 ≠ 0 → 첫 번째 플레이어 승리
print(combined_game_winner([grundy_5, grundy_7, grundy_3]))
```

---

## 4. 미니맥스 알고리즘

### 개념

두 플레이어가 최적으로 플레이할 때의 결과를 계산

```
Max 플레이어: 점수를 최대화하려 함
Min 플레이어: 점수를 최소화하려 함

        Max
       / | \
      3  5  2
     /|  |  |\
Min 3 1  5  2 4

Max는 5를 선택 (최대)
```

### 기본 구현

```python
def minimax(state, depth, is_maximizing, get_moves, evaluate, is_terminal):
    """
    미니맥스 알고리즘
    state: 현재 게임 상태
    depth: 탐색 깊이
    is_maximizing: True면 Max 플레이어 차례
    get_moves: 가능한 수 반환
    evaluate: 상태의 점수 반환
    is_terminal: 게임 종료 여부
    """
    if depth == 0 or is_terminal(state):
        return evaluate(state)

    moves = get_moves(state)

    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            next_state = apply_move(state, move)
            eval_score = minimax(next_state, depth - 1, False,
                                 get_moves, evaluate, is_terminal)
            max_eval = max(max_eval, eval_score)
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            next_state = apply_move(state, move)
            eval_score = minimax(next_state, depth - 1, True,
                                 get_moves, evaluate, is_terminal)
            min_eval = min(min_eval, eval_score)
        return min_eval
```

### 알파-베타 가지치기

```python
def alphabeta(state, depth, alpha, beta, is_maximizing,
              get_moves, evaluate, is_terminal):
    """
    알파-베타 가지치기로 최적화된 미니맥스.
    alpha = Max가 지금까지 보장할 수 있는 최선의 점수 (Max의 하한)
    beta  = Min이 지금까지 보장할 수 있는 최선의 점수 (Max의 상한)
    beta <= alpha일 때 가지치기하는 이유: Max가 이미 다른 곳에서 보장할 수 있는 것보다
    Min에게 더 나은 값을 주는 분기를 Max는 절대 선택하지 않음.
    최적의 경우(완벽한 수 순서), O(b^d)를 O(b^(d/2))로 줄임.
    """
    if depth == 0 or is_terminal(state):
        return evaluate(state)  # 리프 노드 -- 정적 평가 반환

    moves = get_moves(state)

    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            next_state = apply_move(state, move)
            eval_score = alphabeta(next_state, depth - 1, alpha, beta, False,
                                   get_moves, evaluate, is_terminal)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)  # Max의 하한 업데이트
            if beta <= alpha:
                break  # Min이 다른 곳에서 이미 더 나은 선택지를 가짐 -- 이 하위 트리는 무관
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            next_state = apply_move(state, move)
            eval_score = alphabeta(next_state, depth - 1, alpha, beta, True,
                                   get_moves, evaluate, is_terminal)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)  # Min의 상한 업데이트
            if beta <= alpha:
                break  # Max가 다른 곳에서 이미 더 나은 선택지를 가짐 -- 이 하위 트리는 무관
        return min_eval

# 호출
score = alphabeta(initial_state, max_depth, float('-inf'), float('inf'), True,
                  get_moves, evaluate, is_terminal)
```

---

## 5. 틱택토 예제

```python
class TicTacToe:
    def __init__(self):
        self.board = [[' '] * 3 for _ in range(3)]
        self.current_player = 'X'

    def get_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves

    def make_move(self, row, col):
        self.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def undo_move(self, row, col):
        self.board[row][col] = ' '
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        # 가로, 세로, 대각선 체크
        lines = []
        for i in range(3):
            lines.append([self.board[i][j] for j in range(3)])  # 가로
            lines.append([self.board[j][i] for j in range(3)])  # 세로
        lines.append([self.board[i][i] for i in range(3)])      # 대각선
        lines.append([self.board[i][2-i] for i in range(3)])    # 반대 대각선

        for line in lines:
            if line[0] == line[1] == line[2] != ' ':
                return line[0]
        return None

    def is_terminal(self):
        if self.check_winner():
            return True
        return len(self.get_moves()) == 0

    def evaluate(self):
        winner = self.check_winner()
        if winner == 'X':
            return 1
        elif winner == 'O':
            return -1
        return 0

    def minimax(self, is_maximizing):
        if self.is_terminal():
            return self.evaluate()

        if is_maximizing:
            best = float('-inf')
            for move in self.get_moves():
                self.make_move(*move)
                best = max(best, self.minimax(False))
                self.undo_move(*move)
            return best
        else:
            best = float('inf')
            for move in self.get_moves():
                self.make_move(*move)
                best = min(best, self.minimax(True))
                self.undo_move(*move)
            return best

    def best_move(self):
        """현재 플레이어의 최적 수 찾기"""
        best_score = float('-inf') if self.current_player == 'X' else float('inf')
        best_move = None
        is_max = self.current_player == 'X'

        for move in self.get_moves():
            self.make_move(*move)
            score = self.minimax(not is_max)
            self.undo_move(*move)

            if is_max and score > best_score:
                best_score = score
                best_move = move
            elif not is_max and score < best_score:
                best_score = score
                best_move = move

        return best_move

# 사용 예시
game = TicTacToe()
while not game.is_terminal():
    move = game.best_move()
    print(f"{game.current_player}가 {move}에 둠")
    game.make_move(*move)

winner = game.check_winner()
print(f"승자: {winner if winner else '무승부'}")
```

---

## 6. 실전 문제 패턴

### 패턴 1: 단순 승패 게임

```python
def simple_game_winner(n):
    """
    n개 돌, 1~3개 가져가기
    마지막 가져가면 승리
    """
    return n % 4 != 0

# Grundy 패턴으로 빠르게 해결
```

### 패턴 2: 여러 더미 게임

```python
def multi_pile_game(piles, max_take):
    """
    각 더미에서 1~max_take개 가져가기
    """
    # 각 더미의 Grundy = pile % (max_take + 1)
    xor_sum = 0
    for pile in piles:
        xor_sum ^= (pile % (max_take + 1))

    return xor_sum != 0
```

### 패턴 3: 그래프 게임

```python
def graph_game(adj, start):
    """
    그래프에서 토큰 이동 게임
    이동할 수 없으면 패배
    """
    n = len(adj)
    grundy = [-1] * n

    def calc_grundy(node):
        if grundy[node] != -1:
            return grundy[node]

        if not adj[node]:  # 이동 불가
            grundy[node] = 0
            return 0

        reachable = set()
        for next_node in adj[node]:
            reachable.add(calc_grundy(next_node))

        mex = 0
        while mex in reachable:
            mex += 1

        grundy[node] = mex
        return mex

    return calc_grundy(start) != 0
```

### 패턴 4: 게임 분할

```python
def split_game(n):
    """
    n개 돌을 두 개의 비어있지 않은 더미로 분할
    분할 불가능하면 패배
    """
    grundy = [0] * (n + 1)

    for i in range(2, n + 1):
        reachable = set()
        for j in range(1, i // 2 + 1):
            # i를 j와 i-j로 분할
            reachable.add(grundy[j] ^ grundy[i - j])

        mex = 0
        while mex in reachable:
            mex += 1
        grundy[i] = mex

    return grundy[n] != 0
```

---

## 7. 고급 기법

### Nimber 곱셈

Nimber는 덧셈(XOR) 외에 곱셈도 정의됩니다.

```python
def nim_multiply(a, b):
    """
    Nimber 곱셈 (작은 수에 대해)
    """
    if a < 2 or b < 2:
        return a * b

    # 2의 거듭제곱 분해
    highest_a = a.bit_length() - 1
    highest_b = b.bit_length() - 1

    if highest_a > highest_b:
        a, b = b, a
        highest_a, highest_b = highest_b, highest_a

    if highest_a == 0:
        return a * b

    # 재귀적 계산 (복잡하므로 일반적으로 테이블 사용)
    # 여기선 간단한 경우만 처리
    return a * b  # 실제로는 더 복잡
```

### 게임 트리 저장

```python
class GameTree:
    def __init__(self):
        self.cache = {}

    def solve(self, state, get_next, hash_state):
        """
        메모이제이션으로 게임 트리 탐색
        """
        h = hash_state(state)
        if h in self.cache:
            return self.cache[h]

        next_states = get_next(state)
        if not next_states:
            self.cache[h] = False  # 이동 불가 = 패배
            return False

        # 다음 상태 중 하나라도 패배 상태면 현재는 승리
        for ns in next_states:
            if not self.solve(ns, get_next, hash_state):
                self.cache[h] = True
                return True

        self.cache[h] = False
        return False
```

---

## 8. 시간 복잡도 정리

| 알고리즘 | 시간 복잡도 | 비고 |
|---------|------------|------|
| 님 게임 | O(n) | n = 더미 수 |
| Grundy 계산 | O(상태 수 × 분기 수) | 메모이제이션 필수 |
| 미니맥스 | O(b^d) | b = 분기, d = 깊이 |
| 알파-베타 | O(b^(d/2)) ~ O(b^d) | 최적 순서 시 |

---

## 9. 자주 하는 실수

### 실수 1: mex 계산 오류

```python
# 잘못됨: 최소값 찾기
mex = min(grundy_set)

# 올바름: 없는 최소 비음수
mex = 0
while mex in grundy_set:
    mex += 1
```

### 실수 2: XOR 연산 우선순위

```python
# 잘못됨
result = a ^ b == 0  # (a ^ b) == 0이 아님!

# 올바름
result = (a ^ b) == 0
```

### 실수 3: 종료 조건

```python
# 이동 가능한 상태가 없으면 Grundy = 0
if not next_states:
    return 0  # 패배 상태
```

---

## 10. 연습 문제

| 난이도 | 문제 유형 | 핵심 개념 |
|--------|----------|-----------|
| ★★☆ | 돌 가져가기 | 기본 W/L 분석 |
| ★★☆ | 님 게임 | XOR 활용 |
| ★★★ | Grundy 계산 | mex 함수 |
| ★★★ | 복합 게임 | XOR 결합 |
| ★★★★ | 미니맥스 | 알파-베타 |

---

## 다음 단계

- [28_Advanced_DP_Optimization.md](./28_Advanced_DP_Optimization.md) - 고급 DP 최적화

---

## 학습 점검

1. 님 게임에서 XOR sum이 0이면 왜 패배인가?
2. Grundy number의 mex 함수란?
3. 여러 독립 게임의 Grundy는 어떻게 계산하는가?
4. 알파-베타 가지치기가 작동하는 원리는?
