# 레슨 18: Raft 구현 Part 1 — 리더 선출, 로그 복제, 안전성

[개요](./00_Overview.md) | [이전: 캡스톤 — 분산 KV 스토어 구축](./16_Capstone_Building_Distributed_KV_Store.md) | [다음: Raft 구현 Part 2](./19_Raft_Implementation_Part2.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 랜덤화된 타임아웃과 pre-vote 최적화를 사용한 Raft 리더 선출(leader election) 구현
2. AppendEntries RPC와 커밋 진행(commit advancement)을 포함한 완전한 로그 복제(log replication) 파이프라인 구축
3. Raft 안전성 속성(safety properties) — 선출 안전성(Election Safety), 로그 매칭(Log Matching), 리더 완전성(Leader Completeness) — 증명 및 적용
4. 커밋된 항목을 순서대로 적용하는 결정론적 상태 머신(deterministic state machine) 구성
5. 분할 투표(split vote), 오래된 리더(stale leader), 로그 발산(log divergence)을 포함한 에지 케이스(edge case) 처리

---

## 목차

1. [Raft 구현 소개](#1-raft-구현-소개)
2. [노드 상태와 전환](#2-노드-상태와-전환)
3. [리더 선출 알고리즘](#3-리더-선출-알고리즘)
4. [로그 복제](#4-로그-복제)
5. [안전성 속성과 증명](#5-안전성-속성과-증명)
6. [상태 머신 적용](#6-상태-머신-적용)
7. [에지 케이스 처리](#7-에지-케이스-처리)
8. [완전한 구현](#8-완전한-구현)
9. [리더 선출 테스트](#9-리더-선출-테스트)
10. [요약과 핵심 정리](#10-요약과-핵심-정리)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. Raft 구현 소개

### 1.1 왜 또 다른 Raft 레슨인가?

레슨 06에서 Raft 개념을 소개했다. 이 레슨과 다음 레슨에서는 *프로덕션 수준*의 구현을 단계별로 살펴본다. 레슨 06이 "Raft가 무엇을 하는가"였다면, 이 레슨은 "Raft를 올바르게 구축하는 방법"이다.

Raft를 이해하는 것과 올바르게 구현하는 것 사이의 격차는 매우 크다. 미묘한 버그가 다음과 같은 곳에 숨어 있다:

- **선출 타이머 관리(election timer management)**: 타임아웃 랜덤화의 off-by-one 오류
- **로그 인덱스 계산(log index math)**: 0 기반과 1 기반 인덱싱(indexing) 혼동
- **커밋 진행(commit advancement)**: 여러 텀(term)에 걸쳐 `commitIndex`를 잘못 진행
- **오래된 메시지 처리(stale message handling)**: 이전 텀의 메시지 처리

### 1.2 구현 범위

```
Part 1 (이 레슨)                  Part 2 (레슨 19)
┌────────────────────────┐       ┌────────────────────────┐
│ 리더 선출               │       │ 멤버십 변경             │
│ 로그 복제               │       │ 로그 압축               │
│ 안전성 증명             │       │ 스냅샷                   │
│ 상태 머신               │       │ 선형화 가능 읽기         │
│ 에지 케이스             │       │ 성능 튜닝               │
└────────────────────────┘       └────────────────────────┘
```

### 1.3 아키텍처 개요

```
클라이언트 요청
      │
      ▼
┌─────────────┐
│   리더       │──── AppendEntries RPC ───▶ 팔로워
│             │◀─── 응답 ─────────────────┘
│ ┌─────────┐ │
│ │  로그    │ │  ← 미커밋 항목 추가
│ └────┬────┘ │
│      │      │
│ ┌────▼────┐ │
│ │ 커밋    │ │  ← 과반수 복제 → 커밋됨
│ └────┬────┘ │
│      │      │
│ ┌────▼────┐ │
│ │  상태   │ │  ← 키-값 상태 머신에 적용
│ │ 머신    │ │
│ └─────────┘ │
└─────────────┘
```

---

## 2. 노드 상태와 전환

### 2.1 세 가지 상태

모든 Raft 노드는 주어진 시점에 정확히 세 가지 상태(state) 중 하나에 있다:

```
                    ┌──────────┐
         타임아웃   │          │  투표 수신
        ┌──────────▶│ 후보자    │──────────────┐
        │           │          │              │
        │           └────┬─────┘              │
        │                │                    │
        │          현재 리더 또는              │
        │          새 텀 발견                  │
        │                │                    │
        │                ▼                    ▼
   ┌────┴────┐     ┌──────────┐       ┌──────────┐
   │         │     │          │       │          │
   │ 팔로워   │◀────│ 팔로워    │       │  리더     │
   │         │     │          │       │          │
   └─────────┘     └──────────┘       └──────────┘
        ▲                                   │
        │          더 높은 텀을 가진         │
        │          서버 발견                 │
        └───────────────────────────────────┘
```

### 2.2 영속 상태(Persistent State)

이 필드들은 충돌(crash)에서 살아남아야 한다 (RPC에 응답하기 전에 안정 저장소(stable storage)에 기록):

```python
@dataclass
class PersistentState:
    """RPC에 응답하기 전에 안정 저장소에 영속화해야 하는 상태."""
    current_term: int = 0       # 서버가 본 최신 텀
    voted_for: Optional[str] = None  # 현재 텀에서 투표를 받은 후보자 ID
    log: list = field(default_factory=lambda: [])  # 로그 항목 (첫 인덱스는 1)
```

### 2.3 휘발성 상태(Volatile State)

```python
@dataclass
class VolatileState:
    """충돌 후 재구성 가능한 상태."""
    commit_index: int = 0  # 커밋된 것으로 알려진 가장 높은 로그 항목 인덱스
    last_applied: int = 0  # 상태 머신에 적용된 가장 높은 로그 항목 인덱스

@dataclass
class LeaderVolatileState:
    """리더에서만 유지되는 추가 상태."""
    next_index: dict = field(default_factory=dict)   # 각 서버별: 보낼 다음 로그 인덱스
    match_index: dict = field(default_factory=dict)   # 각 서버별: 복제된 가장 높은 로그 인덱스
```

---

## 3. 리더 선출 알고리즘

### 3.1 선출 타이머(Election Timer)

선출 타이머는 Raft의 심장 박동이다. 이것을 올바르게 구현하는 것이 매우 중요하다:

```python
import random
import time
import threading
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum


class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class RaftNode:
    """
    완전한 Raft 노드 구현 — Part 1: 선출 + 복제.

    이것은 Ongaro & Ousterhout 논문(2014)에 설명된 핵심 Raft 프로토콜을
    구현하며, etcd와 TiKV 구현에서 가져온 추가적인 프로덕션 강화를 포함한다.
    """

    # 선출 타임아웃 범위 (밀리초)
    ELECTION_TIMEOUT_MIN = 150
    ELECTION_TIMEOUT_MAX = 300
    HEARTBEAT_INTERVAL = 50  # ms — ELECTION_TIMEOUT_MIN보다 훨씬 작아야 함

    def __init__(self, node_id: str, peers: list[str], storage_path: str = ""):
        self.node_id = node_id
        self.peers = peers
        self.storage_path = storage_path or f"/tmp/raft_{node_id}.json"

        # --- 영속 상태 (충돌에서 살아남음) ---
        self.current_term: int = 0
        self.voted_for: Optional[str] = None
        self.log: list[dict] = []  # 각 항목: {"term": int, "command": Any, "index": int}

        # --- 휘발성 상태 ---
        self.commit_index: int = 0
        self.last_applied: int = 0
        self.state: NodeState = NodeState.FOLLOWER

        # --- 리더 전용 휘발성 상태 ---
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}

        # --- 타이밍 ---
        self.last_heartbeat_time: float = time.time()
        self.election_timeout: float = self._random_election_timeout()

        # --- 상태 머신 ---
        self.state_machine: Dict[str, str] = {}

        # --- 메시지 큐 (시뮬레이션된 네트워크) ---
        self.inbox: list[dict] = []
        self.outbox: list[dict] = []

        # --- 통계 ---
        self.elections_started: int = 0
        self.elections_won: int = 0
        self.entries_committed: int = 0

        # 영속 상태 로드
        self._load_persistent_state()

    def _random_election_timeout(self) -> float:
        """
        랜덤 선출 타임아웃을 생성한다.

        랜덤화는 분할 투표를 피하는 데 매우 중요하다.
        각 노드는 [ELECTION_TIMEOUT_MIN, ELECTION_TIMEOUT_MAX]에서
        균일하게 랜덤으로 타임아웃을 선택한다.

        범위는 다음이어야 한다:
        - broadcastTime << electionTimeout << MTBF
        - broadcastTime ≈ 0.5ms ~ 20ms (네트워크 RTT)
        - MTBF는 장애 간 평균 시간 (수개월/수년)
        """
        return random.uniform(
            self.ELECTION_TIMEOUT_MIN / 1000.0,
            self.ELECTION_TIMEOUT_MAX / 1000.0,
        )

    # ─── 영속화 ───────────────────────────────────────────

    def _save_persistent_state(self):
        """
        영속 상태를 안정 저장소에 기록한다.

        중요: 이것은 RPC에 응답하기 전에 반드시 완료되어야 한다.
        프로덕션에서는 내구성을 보장하기 위해 fsync()를 사용한다.
        """
        state = {
            "current_term": self.current_term,
            "voted_for": self.voted_for,
            "log": self.log,
        }
        tmp_path = self.storage_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, self.storage_path)  # 원자적 이름 변경

    def _load_persistent_state(self):
        """시작 시 안정 저장소에서 영속 상태를 로드한다."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    state = json.load(f)
                self.current_term = state.get("current_term", 0)
                self.voted_for = state.get("voted_for", None)
                self.log = state.get("log", [])
            except (json.JSONDecodeError, KeyError):
                pass  # 손상된 파일; 처음부터 시작

    # ─── 텀 관리 ───────────────────────────────────────

    def _step_down(self, new_term: int):
        """
        더 높은 텀이 발견되면 팔로워 상태로 전환한다.

        이것은 오래된 리더가 불일치를 일으키는 것을 방지하는
        근본적인 메커니즘이다. 더 높은 텀을 가진 어떤 메시지든
        즉각적인 전환을 유발한다.
        """
        assert new_term >= self.current_term
        self.current_term = new_term
        self.voted_for = None
        self.state = NodeState.FOLLOWER
        self.last_heartbeat_time = time.time()
        self.election_timeout = self._random_election_timeout()
        self._save_persistent_state()

    # ─── 리더 선출 ───────────────────────────────────────

    def _start_election(self):
        """
        후보자 상태로 전환하여 새 선출을 시작한다.

        단계:
        1. currentTerm 증가
        2. 자신에게 투표
        3. 선출 타이머 리셋
        4. 모든 피어에게 RequestVote RPC 전송
        """
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_heartbeat_time = time.time()
        self.election_timeout = self._random_election_timeout()
        self.elections_started += 1

        self._save_persistent_state()

        # RequestVote 메시지 구성
        last_log_index = len(self.log)
        last_log_term = self.log[-1]["term"] if self.log else 0

        for peer in self.peers:
            self.outbox.append({
                "type": "RequestVote",
                "to": peer,
                "from": self.node_id,
                "term": self.current_term,
                "candidate_id": self.node_id,
                "last_log_index": last_log_index,
                "last_log_term": last_log_term,
            })

        # 자신에 대한 투표는 하나의 투표로 계산
        self._votes_received = {self.node_id}
        self._votes_denied = set()

    def _handle_request_vote(self, msg: dict) -> dict:
        """
        수신된 RequestVote RPC를 처리한다.

        다음 조건에서 투표를 승인한다:
        1. 후보자의 텀 ≥ 우리 텀
        2. 이 텀에서 다른 누구에게도 투표하지 않았음
        3. 후보자의 로그가 우리 것만큼 최신(up-to-date)임

        "최신" 검사는 안전성에 매우 중요하다:
        - 먼저 마지막 로그 항목의 텀을 비교
        - 텀이 같으면 로그 길이를 비교
        """
        candidate_term = msg["term"]
        candidate_id = msg["candidate_id"]
        candidate_last_log_index = msg["last_log_index"]
        candidate_last_log_term = msg["last_log_term"]

        # 후보자가 더 높은 텀을 가지면 전환
        if candidate_term > self.current_term:
            self._step_down(candidate_term)

        # 투표를 승인할지 결정
        vote_granted = False

        if candidate_term < self.current_term:
            # 후보자가 뒤쳐져 있음 — 거부
            vote_granted = False
        elif self.voted_for is None or self.voted_for == candidate_id:
            # 아직 투표하지 않았거나, 이미 이 후보자에게 투표함
            # 로그 최신성 검사
            our_last_log_index = len(self.log)
            our_last_log_term = self.log[-1]["term"] if self.log else 0

            if candidate_last_log_term > our_last_log_term:
                vote_granted = True
            elif (candidate_last_log_term == our_last_log_term
                  and candidate_last_log_index >= our_last_log_index):
                vote_granted = True

        if vote_granted:
            self.voted_for = candidate_id
            self.last_heartbeat_time = time.time()  # 선출 타이머 리셋
            self._save_persistent_state()

        return {
            "type": "RequestVoteResponse",
            "to": candidate_id,
            "from": self.node_id,
            "term": self.current_term,
            "vote_granted": vote_granted,
        }

    def _handle_request_vote_response(self, msg: dict):
        """
        RequestVote RPC에 대한 응답을 처리한다.

        과반수의 투표를 받으면 리더가 된다.
        더 높은 텀을 발견하면 전환한다.
        """
        if msg["term"] > self.current_term:
            self._step_down(msg["term"])
            return

        # 같은 텀에서 여전히 후보자인 경우에만 처리
        if self.state != NodeState.CANDIDATE or msg["term"] != self.current_term:
            return

        if msg["vote_granted"]:
            self._votes_received.add(msg["from"])
        else:
            self._votes_denied.add(msg["from"])

        # 과반수를 가지고 있는지 확인
        total_nodes = len(self.peers) + 1  # +1은 자신
        if len(self._votes_received) > total_nodes // 2:
            self._become_leader()

    def _become_leader(self):
        """
        리더 상태로 전환한다.

        모든 피어에 대해 nextIndex와 matchIndex를 초기화한다:
        - nextIndex: 리더의 마지막 로그 인덱스 + 1로 초기화
        - matchIndex: 0으로 초기화

        권위를 확인하기 위해 즉시 하트비트를 전송한다.
        """
        self.state = NodeState.LEADER
        self.elections_won += 1

        last_log_index = len(self.log)
        for peer in self.peers:
            self.next_index[peer] = last_log_index + 1
            self.match_index[peer] = 0

        # 초기 하트비트 전송 (빈 AppendEntries)
        self._send_heartbeats()

    def _send_heartbeats(self):
        """
        모든 피어에게 AppendEntries RPC를 전송한다.

        각 피어에 대해 nextIndex부터 시작하는 로그 항목을 포함한다.
        새 항목이 없으면 하트비트로 기능한다.
        """
        for peer in self.peers:
            self._send_append_entries(peer)

    # ─── 로그 복제 ───────────────────────────────────────

    def _send_append_entries(self, peer: str):
        """
        특정 피어에게 AppendEntries RPC를 전송한다.

        메시지에는 다음이 포함된다:
        - 리더의 텀
        - 일관성 검사를 위한 prevLogIndex와 prevLogTerm
        - 새 항목 (하트비트의 경우 비어 있을 수 있음)
        - 리더의 commitIndex
        """
        next_idx = self.next_index.get(peer, len(self.log) + 1)
        prev_log_index = next_idx - 1
        prev_log_term = 0

        if prev_log_index > 0 and prev_log_index <= len(self.log):
            prev_log_term = self.log[prev_log_index - 1]["term"]

        # 전송할 항목 (nextIndex 이후)
        entries = self.log[next_idx - 1:] if next_idx - 1 < len(self.log) else []

        self.outbox.append({
            "type": "AppendEntries",
            "to": peer,
            "from": self.node_id,
            "term": self.current_term,
            "leader_id": self.node_id,
            "prev_log_index": prev_log_index,
            "prev_log_term": prev_log_term,
            "entries": entries,
            "leader_commit": self.commit_index,
        })

    def _handle_append_entries(self, msg: dict) -> dict:
        """
        수신된 AppendEntries RPC를 처리한다 (팔로워 측).

        단계:
        1. 텀 < currentTerm이면 false 응답
        2. prevLogIndex에 prevLogTerm이 일치하는 항목이 없으면 false 응답
        3. 기존 항목이 새 항목과 충돌하면, 기존 항목과 그 이후
           모든 항목을 삭제
        4. 로그에 아직 없는 새 항목을 추가
        5. leaderCommit > commitIndex이면 commitIndex 전진
        """
        leader_term = msg["term"]

        # 리더가 더 높은 텀을 가지면 전환
        if leader_term > self.current_term:
            self._step_down(leader_term)

        # 리더의 텀이 오래되었으면 거부
        if leader_term < self.current_term:
            return {
                "type": "AppendEntriesResponse",
                "to": msg["from"],
                "from": self.node_id,
                "term": self.current_term,
                "success": False,
                "match_index": 0,
            }

        # 유효한 리더 — 선출 타이머 리셋
        self.state = NodeState.FOLLOWER
        self.last_heartbeat_time = time.time()

        # 일관성 검사
        prev_log_index = msg["prev_log_index"]
        prev_log_term = msg["prev_log_term"]

        if prev_log_index > 0:
            if prev_log_index > len(self.log):
                # prevLogIndex에 항목이 없음
                return {
                    "type": "AppendEntriesResponse",
                    "to": msg["from"],
                    "from": self.node_id,
                    "term": self.current_term,
                    "success": False,
                    "match_index": len(self.log),
                    "conflict_term": 0,
                    "conflict_index": len(self.log) + 1,
                }

            if self.log[prev_log_index - 1]["term"] != prev_log_term:
                # 항목이 존재하지만 텀이 일치하지 않음
                conflict_term = self.log[prev_log_index - 1]["term"]
                # 빠른 백업을 위해 conflict_term의 첫 인덱스 찾기
                conflict_index = prev_log_index
                for i in range(prev_log_index - 1, -1, -1):
                    if i == 0 or self.log[i - 1]["term"] != conflict_term:
                        conflict_index = i + 1
                        break

                return {
                    "type": "AppendEntriesResponse",
                    "to": msg["from"],
                    "from": self.node_id,
                    "term": self.current_term,
                    "success": False,
                    "match_index": conflict_index - 1,
                    "conflict_term": conflict_term,
                    "conflict_index": conflict_index,
                }

        # 새 항목 추가
        entries = msg.get("entries", [])
        for i, entry in enumerate(entries):
            log_index = prev_log_index + 1 + i
            if log_index <= len(self.log):
                if self.log[log_index - 1]["term"] != entry["term"]:
                    # 충돌: 이 항목과 그 이후 모든 항목 삭제
                    self.log = self.log[:log_index - 1]
                    self.log.append(entry)
                # 그렇지 않으면: 항목이 이미 존재하고 일치하므로 건너뜀
            else:
                self.log.append(entry)

        self._save_persistent_state()

        # 커밋 인덱스 전진
        if msg["leader_commit"] > self.commit_index:
            last_new_index = prev_log_index + len(entries)
            self.commit_index = min(msg["leader_commit"], last_new_index)

        # 커밋된 항목 적용
        self._apply_committed_entries()

        return {
            "type": "AppendEntriesResponse",
            "to": msg["from"],
            "from": self.node_id,
            "term": self.current_term,
            "success": True,
            "match_index": prev_log_index + len(entries),
        }

    def _handle_append_entries_response(self, msg: dict):
        """
        AppendEntries RPC에 대한 응답을 처리한다 (리더 측).

        성공 시: 해당 피어의 nextIndex와 matchIndex 업데이트.
        실패 시: nextIndex를 감소시키고 재시도 (최적화 포함).
        """
        if msg["term"] > self.current_term:
            self._step_down(msg["term"])
            return

        if self.state != NodeState.LEADER:
            return

        peer = msg["from"]

        if msg["success"]:
            # 이 피어에 대한 추적 업데이트
            new_match_index = msg["match_index"]
            self.match_index[peer] = max(self.match_index.get(peer, 0), new_match_index)
            self.next_index[peer] = self.match_index[peer] + 1

            # 커밋 인덱스 전진 시도
            self._advance_commit_index()
        else:
            # 최적화: 충돌 정보를 사용한 빠른 백업
            if "conflict_term" in msg and msg["conflict_term"] > 0:
                # 우리 로그에서 conflict_term 검색
                conflict_term = msg["conflict_term"]
                found = False
                for i in range(len(self.log) - 1, -1, -1):
                    if self.log[i]["term"] == conflict_term:
                        self.next_index[peer] = i + 2  # 해당 텀의 마지막 항목 다음
                        found = True
                        break
                if not found:
                    self.next_index[peer] = msg.get("conflict_index", 1)
            else:
                # 단순 백업: 1씩 감소
                self.next_index[peer] = max(1, self.next_index.get(peer, 2) - 1)

            # 업데이트된 nextIndex로 재시도
            self._send_append_entries(peer)

    def _advance_commit_index(self):
        """
        과반수 복제에 기반하여 commitIndex를 전진시킨다.

        다음을 만족하는 가장 높은 N을 찾는다:
        - 과반수의 matchIndex[i] ≥ N
        - log[N].term == currentTerm (핵심 안전성 속성)

        텀 검사는 이전 텀의 항목을 복제본 수를 세어서 커밋하는 것을
        방지한다. 이전 텀의 항목은 현재 텀의 항목이 커밋될 때
        간접적으로 커밋된다.
        """
        for n in range(len(self.log), self.commit_index, -1):
            if self.log[n - 1]["term"] != self.current_term:
                continue

            # 복제본 수 (자신 포함)
            replicas = 1  # 자신 포함
            for peer in self.peers:
                if self.match_index.get(peer, 0) >= n:
                    replicas += 1

            total_nodes = len(self.peers) + 1
            if replicas > total_nodes // 2:
                self.commit_index = n
                self._apply_committed_entries()
                break

    # ─── 상태 머신 ─────────────────────────────────────────

    def _apply_committed_entries(self):
        """
        커밋되었지만 아직 적용되지 않은 항목을 상태 머신에 적용한다.

        항목은 엄격한 로그 순서대로 적용된다. 이것은 모든 노드가
        같은 명령을 같은 순서로 적용하여 선형화 가능성(linearizability)을
        달성하는 것을 보장한다.
        """
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied - 1]
            command = entry.get("command", {})

            op = command.get("op")
            if op == "put":
                self.state_machine[command["key"]] = command["value"]
            elif op == "delete":
                self.state_machine.pop(command["key"], None)
            elif op == "cas":  # Compare-and-swap
                key = command["key"]
                expected = command.get("expected")
                new_value = command["value"]
                if self.state_machine.get(key) == expected:
                    self.state_machine[key] = new_value

            self.entries_committed += 1

    # ─── 클라이언트 인터페이스 ──────────────────────────────

    def propose(self, command: dict) -> bool:
        """
        클러스터에 새 명령을 제안한다 (리더만 가능).

        명령이 복제를 위해 수락되면 True를 반환한다.
        명령은 아직 커밋되지 않았으며 — 호출자는 커밋 상태를
        폴링해야 한다.
        """
        if self.state != NodeState.LEADER:
            return False

        entry = {
            "term": self.current_term,
            "index": len(self.log) + 1,
            "command": command,
        }
        self.log.append(entry)
        self._save_persistent_state()

        # 모든 피어에 즉시 복제
        for peer in self.peers:
            self._send_append_entries(peer)

        return True

    # ─── 틱 (타이머) ──────────────────────────────────────

    def tick(self):
        """
        주기적으로 호출되어 Raft 노드를 구동한다.

        - 팔로워/후보자: 선출 타임아웃 확인
        - 리더: 하트비트 전송
        """
        now = time.time()
        elapsed = now - self.last_heartbeat_time

        if self.state == NodeState.LEADER:
            if elapsed >= self.HEARTBEAT_INTERVAL / 1000.0:
                self.last_heartbeat_time = now
                self._send_heartbeats()
        else:
            if elapsed >= self.election_timeout:
                self._start_election()

    # ─── 메시지 디스패치 ──────────────────────────────────

    def receive(self, msg: dict):
        """수신 메시지를 적절한 핸들러로 라우팅한다."""
        msg_type = msg.get("type")

        if msg_type == "RequestVote":
            response = self._handle_request_vote(msg)
            self.outbox.append(response)
        elif msg_type == "RequestVoteResponse":
            self._handle_request_vote_response(msg)
        elif msg_type == "AppendEntries":
            response = self._handle_append_entries(msg)
            self.outbox.append(response)
        elif msg_type == "AppendEntriesResponse":
            self._handle_append_entries_response(msg)

    def get_status(self) -> dict:
        """디버깅을 위한 현재 노드 상태를 반환한다."""
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "term": self.current_term,
            "voted_for": self.voted_for,
            "log_length": len(self.log),
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "state_machine_size": len(self.state_machine),
            "elections_started": self.elections_started,
            "elections_won": self.elections_won,
            "entries_committed": self.entries_committed,
        }
```

---

## 4. 로그 복제

### 4.1 로그 매칭 속성(Log Matching Property)

Raft는 **로그 매칭 속성**을 유지한다:

> 서로 다른 로그에 있는 두 항목이 같은 인덱스와 텀을 가지면,
> 같은 명령을 저장하며 모든 이전 항목이 동일하다.

이것은 AppendEntries의 일관성 검사에 의해 적용된다:

```
리더 로그:      [1:a] [1:b] [2:c] [3:d] [3:e]
                  ↑ prevLogIndex=4, prevLogTerm=3

팔로워 A:       [1:a] [1:b] [2:c] [3:d]          ← 일치! [3:e] 추가
팔로워 B:       [1:a] [1:b] [2:c]                 ← 실패! 인덱스 4에 항목 없음
팔로워 C:       [1:a] [1:b] [2:c] [2:x]           ← 실패! 인덱스 4에서 텀 불일치
```

### 4.2 로그 발산 처리

팔로워의 로그가 리더와 발산하면, 리더는 마지막 일치 지점을 찾아 그 이후 모든 것을 덮어써야 한다:

```python
def demonstrate_log_divergence():
    """
    리더와 팔로워 사이의 로그 발산을 Raft가 어떻게 처리하는지 시연한다.

    시나리오: 네트워크 파티션이 두 리더가 서로 다른 항목을 수락하도록
    했다. 파티션이 복구된 후, 살아남은 리더는 오래된 팔로워의 로그를
    일치시켜야 한다.
    """
    # 텀 3에서 선출을 이긴 후의 리더 로그
    leader_log = [
        {"term": 1, "index": 1, "command": {"op": "put", "key": "x", "value": "1"}},
        {"term": 1, "index": 2, "command": {"op": "put", "key": "y", "value": "2"}},
        {"term": 2, "index": 3, "command": {"op": "put", "key": "x", "value": "3"}},
        {"term": 3, "index": 4, "command": {"op": "put", "key": "z", "value": "4"}},
    ]

    # 팔로워 로그 — 파티션 중에 발산
    follower_log = [
        {"term": 1, "index": 1, "command": {"op": "put", "key": "x", "value": "1"}},
        {"term": 1, "index": 2, "command": {"op": "put", "key": "y", "value": "2"}},
        {"term": 2, "index": 3, "command": {"op": "put", "key": "x", "value": "3"}},
        {"term": 2, "index": 4, "command": {"op": "put", "key": "w", "value": "9"}},  # 발산
        {"term": 2, "index": 5, "command": {"op": "put", "key": "v", "value": "8"}},  # 발산
    ]

    print("=== 로그 발산 해결 ===")
    print(f"리더 로그:   {[(e['term'], e['index']) for e in leader_log]}")
    print(f"팔로워 로그: {[(e['term'], e['index']) for e in follower_log]}")

    # 리더가 prevLogIndex=4, prevLogTerm=3으로 AppendEntries 시도
    prev_idx = len(leader_log)  # 4
    prev_term = leader_log[prev_idx - 1]["term"]  # 3

    # 팔로워 확인: 인덱스 4의 항목은 텀 2이지 3이 아님 → 충돌
    if follower_log[prev_idx - 1]["term"] != prev_term:
        conflict_term = follower_log[prev_idx - 1]["term"]
        # conflict_term의 첫 인덱스 찾기
        conflict_start = prev_idx
        for i in range(prev_idx - 1, -1, -1):
            if i == 0 or follower_log[i - 1]["term"] != conflict_term:
                conflict_start = i + 1
                break

        print(f"\n인덱스 {prev_idx}에서 충돌 감지")
        print(f"  리더는 텀 {prev_term}, 팔로워는 텀 {conflict_term}")
        print(f"  충돌 시작 인덱스: {conflict_start}")

        # 리더가 nextIndex를 conflict_start로 백업하고
        # 재시도하여 결국 발산을 해결
        follower_log = follower_log[:conflict_start - 1]  # 자르기
        follower_log.extend(leader_log[conflict_start - 1:])  # 리더 항목 추가

        print(f"\n해결 후:")
        print(f"  리더 로그:   {[(e['term'], e['index']) for e in leader_log]}")
        print(f"  팔로워 로그: {[(e['term'], e['index']) for e in follower_log]}")
        print(f"  로그 일치: {leader_log == follower_log}")


demonstrate_log_divergence()
```

### 4.3 빠른 백업 최적화(Fast Backup Optimization)

순진한 접근법은 각 실패마다 `nextIndex`를 1씩 감소시켜, n개의 발산 항목에 대해 O(n)번의 왕복이 필요하다. 빠른 백업 최적화는 거부에 충돌 정보를 포함한다:

```
팔로워 → 리더:
  success = False
  conflictTerm = 충돌 항목의 텀
  conflictIndex = conflictTerm을 가진 첫 인덱스

리더 응답:
  리더가 conflictTerm을 가진 항목이 있으면:
    nextIndex = 리더의 conflictTerm 마지막 항목 + 1
  아니면:
    nextIndex = conflictIndex
```

이것은 왕복 횟수를 O(서로 다른 충돌 텀의 수)로 줄인다.

---

## 5. 안전성 속성과 증명

### 5.1 선출 안전성(Election Safety)

**주장**: 주어진 텀에서 최대 하나의 리더만 선출될 수 있다.

**증명**:
1. 각 노드는 텀당 최대 하나의 후보자에게만 투표한다 (`voted_for` 영속화로 적용).
2. 후보자가 당선되려면 엄격한 과반수 (> N/2)의 투표가 필요하다.
3. 임의의 두 과반수는 최소 하나의 노드에서 겹친다.
4. 겹치는 노드는 최대 하나의 후보자에게 투표했다.
5. 따라서 최대 하나의 후보자만 과반수를 받을 수 있다. ∎

```python
def verify_election_safety(votes: dict, total_nodes: int) -> bool:
    """
    최대 하나의 후보자만 과반수의 투표를 받았는지 검증한다.

    Args:
        votes: {후보자_id: 투표자 집합}
        total_nodes: 클러스터의 총 노드 수

    Returns:
        선출 안전성이 유지되면 True
    """
    majority = total_nodes // 2 + 1
    winners = [c for c, v in votes.items() if len(v) >= majority]

    if len(winners) > 1:
        print(f"안전성 위반: 여러 당선자: {winners}")
        return False

    # 한 투표자가 여러 후보자에게 투표하지 않았는지 검증
    all_voters = set()
    for candidate, voters in votes.items():
        for voter in voters:
            if voter in all_voters:
                print(f"안전성 위반: {voter}가 여러 후보자에게 투표")
                return False
            all_voters.add(voter)

    return True


# 5개 노드로 테스트
votes_safe = {
    "A": {"A", "B", "C"},  # A가 3/5로 당선
    "D": {"D", "E"},        # D가 2/5로 낙선
}
print(f"안전한 선출: {verify_election_safety(votes_safe, 5)}")  # True

votes_unsafe = {
    "A": {"A", "B", "C"},
    "D": {"D", "B", "E"},  # B가 양쪽에 투표! (올바른 구현에서는 불가능)
}
print(f"안전하지 않은 선출: {verify_election_safety(votes_unsafe, 5)}")  # False
```

### 5.2 리더 완전성(Leader Completeness)

**주장**: 로그 항목이 주어진 텀에서 커밋되면, 해당 항목은 모든 더 높은 번호의 텀에서 모든 리더의 로그에 존재한다.

**증명 개요**:
1. 인덱스 i에서 커밋된 항목 E는 과반수 S1에 복제되었다.
2. 미래의 리더 L은 과반수 S2로부터 투표를 받아야 한다.
3. S1 ∩ S2는 비어 있지 않다 (비둘기집 원리).
4. 교집합에 있는 투표자는 자신의 로그에 E를 가지고 있다.
5. RequestVote의 최신성 검사가 L의 로그가 최소한 같은 수준으로 최신임을 보장한다.
6. 따라서 L의 로그는 E를 포함한다 (또는 같은 인덱스에 같거나 더 높은 텀의 항목을 가지며, 로그 매칭에 의해 일치해야 한다). ∎

### 5.3 커밋 규칙의 미묘함

Raft에서 가장 미묘한 안전성 문제는 이전 텀 항목의 커밋 규칙이다:

```
텀 1: 리더 S1이 항목 A를 S1, S2에 복제
       S1이 커밋 전에 충돌

텀 2: S5가 선출됨 (S3, S4, S5가 투표; S5는 빈 로그)
       S5가 항목 B를 S3에 복제
       S5가 충돌

텀 3: S1이 복구, 선출됨 (S1, S2, S3, S4가 투표)
       S1은 인덱스 1에 항목 A를 가지고 있음

잘못된 접근법: S1이 A의 복제본을 셈 (S1, S2 = 2/5 → 부족)
              S1이 A를 S3에 복제, 이제 3/5 복제본 → A를 커밋?

              아니다! 이것은 안전하지 않다. S1이 지금 충돌하면, S5가
              텀 4에서 선출되어 A를 B로 덮어쓸 수 있다!

올바른 접근법: S1은 A를 직접 커밋하지 않는다.
              S1이 텀 3에서 새 항목 C를 추가한다.
              C가 커밋되면 (과반수에 복제), A는 로그에서
              C 앞에 있으므로 간접적으로 커밋된다.
```

```python
def demonstrate_commit_rule():
    """
    Raft가 이전 텀의 항목을 복제본 수를 직접 세어서
    절대 커밋하지 않는 이유를 시연한다.
    """
    print("=== Raft 커밋 규칙 시연 ===\n")

    # 시나리오 설정
    nodes = ["S1", "S2", "S3", "S4", "S5"]

    # 텀 1 이후: S1이 항목 A (텀 1)를 S1과 S2에 복제
    logs = {
        "S1": [{"term": 1, "cmd": "A"}],
        "S2": [{"term": 1, "cmd": "A"}],
        "S3": [],
        "S4": [],
        "S5": [],
    }
    print("텀 1 이후 (S1이 부분 복제 후 충돌):")
    for n, log in logs.items():
        print(f"  {n}: {log}")

    # 텀 2 이후: S5가 선출, 항목 B를 S3에 복제
    logs["S5"] = [{"term": 2, "cmd": "B"}]
    logs["S3"] = [{"term": 2, "cmd": "B"}]
    print("\n텀 2 이후 (S5가 부분 복제 후 충돌):")
    for n, log in logs.items():
        print(f"  {n}: {log}")

    # 텀 3: S1 선출
    # 잘못됨: S1이 A를 S3에 복제, B를 덮어씀
    print("\n--- 잘못된 접근법: 이전 항목의 복제본 수를 세어 커밋 ---")
    logs_wrong = {k: list(v) for k, v in logs.items()}
    logs_wrong["S3"] = [{"term": 1, "cmd": "A"}]  # B를 A로 덮어쓰기
    print("S1이 A를 S3에 복제:")
    replica_count = sum(1 for n in nodes if logs_wrong[n] and logs_wrong[n][0].get("cmd") == "A")
    print(f"  A의 복제본: {replica_count}/5 → {'커밋됨' if replica_count >= 3 else '커밋 안 됨'}")
    print("  S1이 지금 충돌하면, S5가 텀 4에서 당선되어 A를 B로 덮어쓸 수 있다!")
    print("  이것은 안전성 속성을 위반한다.")

    # 올바름: S1이 텀 3에서 항목 C를 추가, A를 간접적으로 커밋
    print("\n--- 올바른 접근법: 현재 텀에서 새 항목 추가 ---")
    logs_correct = {k: list(v) for k, v in logs.items()}
    logs_correct["S1"].append({"term": 3, "cmd": "C"})
    logs_correct["S2"].append({"term": 3, "cmd": "C"})
    logs_correct["S3"] = [{"term": 1, "cmd": "A"}, {"term": 3, "cmd": "C"}]
    print("S1이 A와 C를 S3에 복제:")
    for n in nodes:
        print(f"  {n}: {logs_correct[n]}")

    # C (텀 3)의 복제본 수 확인
    c_replicas = sum(
        1 for n in nodes
        if len(logs_correct[n]) >= 2 and logs_correct[n][1].get("term") == 3
    )
    print(f"\n  C (텀 3)의 복제본: {c_replicas}/5")
    print("  C가 커밋됨 → A가 간접적으로 커밋됨 (로그에서 C 앞에 있으므로)")
    print("  S5는 로그가 덜 최신이므로 미래 선출에서 당선될 수 없다")


demonstrate_commit_rule()
```

---

## 6. 상태 머신 적용

### 6.1 결정론적 실행(Deterministic Execution)

상태 머신은 반드시 결정론적이어야 한다: 같은 로그 항목이 같은 순서로 주어지면, 모든 노드가 같은 상태를 생성해야 한다. 이것은 다음을 의미한다:

- 명령 실행에서 난수 생성 없음
- 벽시계 시간(wall-clock time)에 대한 의존 없음
- 명령 실행 중 외부 I/O 없음
- 해시 맵 반복 순서에 대한 의존 없음 (정렬된 연산 사용)

```python
class DeterministicKVStateMachine:
    """
    Raft를 위한 결정론적 키-값 상태 머신.

    모든 연산은 현재 상태와 명령의 순수 함수이다.
    부작용, 랜덤성, 외부 의존성이 없다.
    """

    def __init__(self):
        self.data: Dict[str, str] = {}
        self.applied_index: int = 0
        self.applied_commands: list = []

    def apply(self, index: int, command: dict) -> dict:
        """
        상태 머신에 명령을 적용한다.

        Args:
            index: 이 명령의 로그 인덱스
            command: 적용할 명령

        Returns:
            명령 실행 결과
        """
        assert index == self.applied_index + 1, (
            f"명령은 순서대로 적용되어야 한다: 예상 {self.applied_index + 1}, 받음 {index}"
        )

        op = command.get("op")
        result = {"ok": True, "index": index}

        if op == "put":
            self.data[command["key"]] = command["value"]
            result["op"] = "put"

        elif op == "get":
            value = self.data.get(command["key"])
            result["op"] = "get"
            result["value"] = value
            result["found"] = value is not None

        elif op == "delete":
            existed = command["key"] in self.data
            if existed:
                del self.data[command["key"]]
            result["op"] = "delete"
            result["existed"] = existed

        elif op == "cas":
            key = command["key"]
            expected = command.get("expected")
            current = self.data.get(key)
            if current == expected:
                self.data[key] = command["value"]
                result["op"] = "cas"
                result["swapped"] = True
            else:
                result["op"] = "cas"
                result["swapped"] = False
                result["current"] = current
                result["ok"] = False

        elif op == "noop":
            result["op"] = "noop"

        else:
            result["ok"] = False
            result["error"] = f"알 수 없는 연산: {op}"

        self.applied_index = index
        self.applied_commands.append((index, command, result))
        return result

    def snapshot(self) -> dict:
        """현재 상태의 스냅샷을 생성한다."""
        return {
            "data": dict(self.data),
            "applied_index": self.applied_index,
        }

    def restore(self, snapshot: dict):
        """스냅샷에서 상태를 복원한다."""
        self.data = dict(snapshot["data"])
        self.applied_index = snapshot["applied_index"]
```

### 6.2 결정론성 검증

```python
def verify_state_machine_determinism():
    """
    두 상태 머신이 같은 명령을 같은 순서로 받았을 때
    동일한 상태를 생성하는지 검증한다.
    """
    commands = [
        {"op": "put", "key": "x", "value": "1"},
        {"op": "put", "key": "y", "value": "2"},
        {"op": "cas", "key": "x", "expected": "1", "value": "10"},
        {"op": "delete", "key": "y"},
        {"op": "put", "key": "z", "value": "3"},
        {"op": "cas", "key": "x", "expected": "999", "value": "bad"},
        {"op": "get", "key": "x"},
    ]

    sm1 = DeterministicKVStateMachine()
    sm2 = DeterministicKVStateMachine()

    for i, cmd in enumerate(commands, 1):
        r1 = sm1.apply(i, cmd)
        r2 = sm2.apply(i, cmd)
        assert r1 == r2, f"인덱스 {i}에서 발산: {r1} != {r2}"

    assert sm1.data == sm2.data
    print("결정론성 검증됨: 두 상태 머신이 동일한 상태를 생성")
    print(f"  최종 상태: {sm1.data}")
    print(f"  {len(commands)}개 명령 적용됨")


verify_state_machine_determinism()
```

---

## 7. 에지 케이스 처리

### 7.1 분할 투표(Split Vote)

분할 투표는 어떤 후보자도 과반수를 받지 못할 때 발생한다. Raft는 랜덤 타임아웃을 통해 이를 처리한다:

```python
def simulate_split_vote():
    """
    분할 투표 시나리오를 시뮬레이션하고 Raft의 랜덤화된
    타임아웃이 이를 해결하는 방법을 시연한다.
    """
    import random

    num_nodes = 5
    num_trials = 10000
    split_votes = 0
    rounds_to_elect = []

    for _ in range(num_trials):
        rounds = 0
        elected = False

        while not elected:
            rounds += 1
            # 각 노드가 랜덤 타임아웃을 선택
            timeouts = [
                random.uniform(150, 300) for _ in range(num_nodes)
            ]

            # 가장 짧은 타임아웃을 가진 노드가 먼저 선출 시작
            sorted_nodes = sorted(range(num_nodes), key=lambda i: timeouts[i])
            first = sorted_nodes[0]
            second = sorted_nodes[1]

            # 두 노드가 10ms 이내로 시작하면 분할 투표 가능성 높음
            if timeouts[second] - timeouts[first] < 10:
                split_votes += 1
                # 둘 다 선출을 시작하여 투표가 분할됨
                continue
            else:
                # 첫 번째 노드가 선출을 시작하고 당선
                elected = True
                rounds_to_elect.append(rounds)

    avg_rounds = sum(rounds_to_elect) / len(rounds_to_elect) if rounds_to_elect else 0
    print(f"분할 투표 시뮬레이션 ({num_trials}번 시도):")
    print(f"  분할 투표: {split_votes}")
    print(f"  선출까지 평균 라운드: {avg_rounds:.2f}")
    print(f"  최대 라운드: {max(rounds_to_elect) if rounds_to_elect else 0}")


simulate_split_vote()
```

### 7.2 오래된 리더 감지(Stale Leader Detection)

리더가 클러스터에서 격리되어도 모를 수 있다:

```python
def demonstrate_stale_leader():
    """
    오래된 리더가 감지되고 무력화되는 방법을 보여준다.

    네트워크 파티션이 복구되면, 오래된 리더는 더 높은 텀을
    발견하고 즉시 전환한다.
    """
    print("=== 오래된 리더 감지 ===\n")

    # 초기 상태: S1이 텀 1의 리더
    print("단계 1: S1이 텀 1의 리더")
    print("  클러스터: [S1(리더,t=1), S2(팔로워,t=1), S3(팔로워,t=1)]")

    # 네트워크 파티션: S1이 격리됨
    print("\n단계 2: 네트워크 파티션이 S1을 격리")
    print("  파티션 A: [S1(리더,t=1)] ← 자신이 여전히 리더라고 생각")
    print("  파티션 B: [S2, S3] ← 새 리더 선출")

    # S2가 파티션 B에서 선출에 당선
    print("\n단계 3: S2가 텀 2에서 선출에 당선")
    print("  파티션 A: [S1(리더,t=1)] ← 오래된 리더")
    print("  파티션 B: [S2(리더,t=2), S3(팔로워,t=2)]")

    # S1이 복제를 시도 — 파티션 B의 팔로워가 거부
    print("\n단계 4: 파티션 복구")
    print("  S1이 AppendEntries(term=1)를 S2에 전송")
    print("  S2가 term=2 > 1로 응답")
    print("  S1이 더 높은 텀을 발견 → 팔로워로 전환")
    print("  S1 업데이트: term=2, state=follower, votedFor=None")
    print("\n  최종: [S1(팔로워,t=2), S2(리더,t=2), S3(팔로워,t=2)]")

    # 클라이언트 영향
    print("\n--- 클라이언트 영향 ---")
    print("  파티션 중 S1에 대한 쓰기: 커밋되지 않음 (과반수 없음)")
    print("  파티션 중 S1로부터의 읽기: 오래된 데이터를 반환할 수 있음!")
    print("  해결책: ReadIndex 또는 LeaseRead (레슨 19에서 다룸)")


demonstrate_stale_leader()
```

### 7.3 Pre-Vote 확장

Pre-Vote는 반복적으로 텀을 증가시키는 격리된 노드로 인한 중단을 방지한다:

```python
def demonstrate_prevote():
    """
    Raft의 Pre-Vote 확장을 시연한다.

    Pre-Vote 없이: 격리된 노드가 각 선출 타임아웃마다 텀을 증가시킨다.
    파티션이 복구되면, 높은 텀이 안정적인 리더를 전환시켜
    클러스터를 방해한다.

    Pre-Vote와 함께: 노드가 텀을 증가시키기 전에 먼저 선출에
    당선될 수 있는지 확인한다.
    """
    print("=== Pre-Vote 확장 ===\n")

    print("--- Pre-Vote 없이 ---")
    leader_term = 1
    partitioned_term = 1

    # 격리된 동안 10번의 선출 타임아웃 시뮬레이션
    for i in range(10):
        partitioned_term += 1  # 각 타임아웃마다 텀 증가

    print(f"  파티션 후 (10번 타임아웃):")
    print(f"    리더 텀: {leader_term}")
    print(f"    격리된 노드 텀: {partitioned_term}")
    print(f"  파티션 복구 → 격리된 노드가 텀 {partitioned_term}으로 메시지 전송")
    print(f"  리더 전환! (텀 {leader_term} < {partitioned_term})")
    print(f"  결과: 불필요한 리더 방해\n")

    print("--- Pre-Vote와 함께 ---")
    leader_term = 1
    partitioned_term = 1

    # Pre-Vote와 함께, 격리된 노드는 과반수로부터 pre-vote를
    # 받을 수 없으므로 텀을 절대 증가시키지 않음
    for i in range(10):
        # Pre-vote 단계: 피어에게 "나에게 투표하겠습니까?" 질문
        pre_votes = 0  # 피어에 도달 불가 → 0 pre-vote
        if pre_votes >= 1:  # 과반수 - 1 필요 (자신은 계산)
            partitioned_term += 1  # 증가하겠지만 여기에 도달하지 않음

    print(f"  파티션 후 (10번 타임아웃):")
    print(f"    리더 텀: {leader_term}")
    print(f"    격리된 노드 텀: {partitioned_term}")
    print(f"  파티션 복구 → 텀이 호환됨")
    print(f"  리더가 정상적으로 계속")
    print(f"  결과: 방해 없음!")


demonstrate_prevote()
```

---

## 8. 완전한 구현

### 8.1 클러스터 시뮬레이션

```python
class RaftCluster:
    """
    테스트를 위한 시뮬레이션된 Raft 클러스터.

    메시지 라우팅, 네트워크 파티션, 타이밍을 처리한다.
    """

    def __init__(self, node_ids: list[str]):
        self.nodes: Dict[str, RaftNode] = {}
        self.partitions: list[set[str]] = []  # 비어 있으면 = 파티션 없음
        self.message_queue: list[dict] = []
        self.dropped_messages: int = 0

        for nid in node_ids:
            peers = [p for p in node_ids if p != nid]
            self.nodes[nid] = RaftNode(nid, peers, f"/tmp/raft_test_{nid}.json")

    def tick_all(self):
        """모든 노드를 한 틱 전진시킨다."""
        for node in self.nodes.values():
            node.tick()
        self._route_messages()

    def _route_messages(self):
        """파티션을 고려하여 노드 간 메시지를 라우팅한다."""
        for node in self.nodes.values():
            while node.outbox:
                msg = node.outbox.pop(0)
                dest = msg.get("to")

                if self._can_communicate(msg["from"], dest):
                    if dest in self.nodes:
                        self.nodes[dest].receive(msg)
                else:
                    self.dropped_messages += 1

    def _can_communicate(self, src: str, dst: str) -> bool:
        """두 노드가 통신할 수 있는지 확인한다 (사이에 파티션이 없는지)."""
        if not self.partitions:
            return True
        for partition in self.partitions:
            if src in partition and dst in partition:
                return True
        return False

    def partition(self, groups: list[list[str]]):
        """네트워크 파티션을 생성한다."""
        self.partitions = [set(g) for g in groups]

    def heal_partition(self):
        """모든 네트워크 파티션을 제거한다."""
        self.partitions = []

    def get_leader(self) -> Optional[str]:
        """현재 리더를 찾는다 (있는 경우)."""
        leaders = [
            nid for nid, node in self.nodes.items()
            if node.state == NodeState.LEADER
        ]
        return leaders[0] if len(leaders) == 1 else None

    def run_until_leader(self, max_ticks: int = 1000) -> Optional[str]:
        """리더가 선출될 때까지 클러스터를 실행한다."""
        for _ in range(max_ticks):
            self.tick_all()
            leader = self.get_leader()
            if leader:
                return leader
            time.sleep(0.001)
        return None

    def status(self):
        """클러스터 상태를 출력한다."""
        print("\n=== 클러스터 상태 ===")
        for nid, node in sorted(self.nodes.items()):
            s = node.get_status()
            print(f"  {nid}: state={s['state']}, term={s['term']}, "
                  f"log={s['log_length']}, commit={s['commit_index']}")
        if self.partitions:
            print(f"  파티션: {[list(p) for p in self.partitions]}")
        print()


def test_leader_election():
    """5개 노드로 기본 리더 선출을 테스트한다."""
    print("=== 테스트: 리더 선출 ===")

    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    leader = cluster.run_until_leader(max_ticks=500)

    if leader:
        print(f"리더 선출됨: {leader}")
        cluster.status()
    else:
        print("타임아웃 내에 리더가 선출되지 않음")

    return cluster


def test_log_replication():
    """클러스터 전체에 걸친 로그 복제를 테스트한다."""
    print("\n=== 테스트: 로그 복제 ===")

    cluster = RaftCluster(["n1", "n2", "n3"])
    leader_id = cluster.run_until_leader()

    if not leader_id:
        print("리더 선출 실패")
        return

    leader = cluster.nodes[leader_id]

    # 일부 명령 제안
    commands = [
        {"op": "put", "key": "x", "value": "1"},
        {"op": "put", "key": "y", "value": "2"},
        {"op": "put", "key": "z", "value": "3"},
    ]

    for cmd in commands:
        leader.propose(cmd)

    # 커밋될 때까지 실행
    for _ in range(200):
        cluster.tick_all()
        time.sleep(0.001)

    cluster.status()

    # 모든 노드가 같은 커밋된 상태를 가지는지 검증
    for nid, node in cluster.nodes.items():
        print(f"  {nid} 상태 머신: {node.state_machine}")

    return cluster


if __name__ == "__main__":
    test_leader_election()
    test_log_replication()
```

---

## 9. 리더 선출 테스트

### 9.1 속성 기반 테스트(Property-Based Testing)

```python
def test_election_safety_property(num_trials: int = 100):
    """
    여러 시도에 걸쳐 선출 안전성 속성을 검증한다:
    텀당 최대 하나의 리더.

    이것은 많은 랜덤 선출을 실행하고 각 선출 후
    불변성을 확인하는 속성 기반 테스트이다.
    """
    violations = 0

    for trial in range(num_trials):
        cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])

        # 랜덤한 수의 틱 동안 실행
        ticks = random.randint(50, 300)
        for _ in range(ticks):
            cluster.tick_all()

            # 확인: 텀당 최대 하나의 리더
            leaders_by_term: Dict[int, list] = {}
            for nid, node in cluster.nodes.items():
                if node.state == NodeState.LEADER:
                    term = node.current_term
                    if term not in leaders_by_term:
                        leaders_by_term[term] = []
                    leaders_by_term[term].append(nid)

            for term, leaders in leaders_by_term.items():
                if len(leaders) > 1:
                    print(f"시도 {trial}에서 위반: "
                          f"텀 {term}에 리더 {leaders}")
                    violations += 1

    print(f"\n선출 안전성 테스트: {num_trials}번 시도, {violations}번 위반")
    return violations == 0
```

### 9.2 시드를 사용한 결정론적 테스트

```python
def test_with_seed(seed: int):
    """
    고정된 랜덤 시드로 결정론적 테스트를 실행한다.

    이것은 재현 가능한 테스트를 가능하게 한다 — 테스트가 실패하면,
    시드를 사용하여 정확한 시나리오를 재현할 수 있다.
    """
    random.seed(seed)
    print(f"\n=== 결정론적 테스트 (seed={seed}) ===")

    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])

    # 단계 1: 리더 선출
    leader = cluster.run_until_leader(max_ticks=300)
    print(f"단계 1: 리더={leader}")

    if not leader:
        print("  리더 선출 안 됨 — 시드가 병리적 타이밍을 생성할 수 있음")
        return

    # 단계 2: 항목 복제
    cluster.nodes[leader].propose({"op": "put", "key": "k1", "value": "v1"})
    for _ in range(100):
        cluster.tick_all()

    # 단계 3: 리더를 파티션
    all_nodes = list(cluster.nodes.keys())
    minority = [leader]
    majority = [n for n in all_nodes if n != leader]
    cluster.partition([minority, majority])
    print(f"단계 3: {leader}를 {majority}에서 파티션")

    # 단계 4: 과반수 파티션에서 새 리더
    for _ in range(300):
        cluster.tick_all()

    new_leader = None
    for nid in majority:
        if cluster.nodes[nid].state == NodeState.LEADER:
            new_leader = nid
            break
    print(f"단계 4: 새 리더={new_leader}")

    # 단계 5: 파티션 복구
    cluster.heal_partition()
    for _ in range(100):
        cluster.tick_all()

    cluster.status()

    # 검증: 이전 리더가 전환되었는지
    old_leader_state = cluster.nodes[leader].state
    print(f"단계 5: 이전 리더 {leader}는 현재 {old_leader_state.value}")
    assert old_leader_state != NodeState.LEADER or leader == new_leader


# 여러 시드로 실행
for seed in [42, 123, 456, 789, 1000]:
    test_with_seed(seed)
```

---

## 10. 요약과 핵심 정리

### 구현 체크리스트

> **RAFT 구현 PART 1 체크리스트**
>
> ☐ RPC에 응답하기 전에 영속 상태를 디스크에 저장
> ☐ 선출 타이머에 랜덤화된 타임아웃 사용
> ☐ RequestVote에 최신 로그 검사 포함
> ☐ AppendEntries 일관성 검사와 충돌 감지
> ☐ 로그 발산에 대한 빠른 백업 최적화
> ☐ 현재 텀 항목에 대해서만 커밋 전진
> ☐ 상태 머신이 엄격한 순서로 항목 적용
> ☐ Pre-Vote로 격리된 노드의 방해 방지
> ☐ 타임아웃 랜덤화로 분할 투표 처리

### 일반적인 버그

| 버그 | 증상 | 수정 |
|-----|------|------|
| `votedFor` 영속화 안 함 | 텀당 여러 리더 | RPC 응답 전 항상 fsync |
| 이전 텀 항목을 카운팅으로 커밋 | 안전성 위반 | 현재 텀 항목만 직접 커밋 |
| 투표 승인 시 타이머 리셋 안 함 | 불필요한 선출 | `handleRequestVote`에서 승인 시 타이머 리셋 |
| 0 기반 vs 1 기반 로그 인덱싱 | 모든 곳에서 off-by-one | 하나의 규칙을 선택하고 일관되게 사용 |
| RPC 응답에서 텀 검사 안 함 | 오래된 리더가 계속됨 | 항상 더 높은 텀에서 확인 및 전환 |

---

## 11. 연습 문제

### 문제 1: 로그 수렴

일련의 파티션과 선출 후 다음 로그가 있을 때, 완전한 수렴에 필요한 최소 AppendEntries 라운드 수를 결정하라:

```
리더 (텀 4):    [1:a] [1:b] [2:c] [4:d] [4:e]
팔로워 A:       [1:a] [1:b] [3:x] [3:y]
팔로워 B:       [1:a] [1:b] [2:c]
팔로워 C:       [1:a]
```

### 문제 2: 안전성 증명

로그 매칭 속성이 AppendEntries에 의해 유지됨을 증명하라. 구체적으로, 일관성 검사가 인덱스 `prevLogIndex`에서 통과하면, 그 인덱스 이전의 모든 항목도 일치해야 함을 보여라.

### 문제 3: 구현 과제

모든 Raft 안전성 속성이 어느 시점에서나 유지되는지 검증하는 `RaftNode.check_invariants()` 메서드를 구현하라:
- `commitIndex <= len(log)`
- `lastApplied <= commitIndex`
- 리더인 경우: commitIndex 이후의 모든 항목이 `term == currentTerm`
- 팔로워인 경우: `votedFor`가 `currentTerm`과 일관됨

### 문제 4: 타이밍 분석

5개 노드 클러스터의 조건:
- 네트워크 RTT: 1ms (데이터센터 내)
- 선출 타임아웃: [150ms, 300ms]
- 하트비트 간격: 50ms

계산:
1. 콜드 스타트에서 리더를 선출하는 최악의 경우 시간
2. 수렴 전 분할 투표 라운드의 최대 수 (확률적)
3. 격리된 리더가 (미커밋) 쓰기를 수락할 수 있는 시간 윈도우

### 문제 5: Pre-Vote 구현

`RaftNode` 클래스에 완전한 Pre-Vote 확장을 구현하라. Pre-Vote RPC는 RequestVote와 동일하지만:
- 발신자의 텀을 증가시키지 않음
- 수신자가 전환되지 않음
- Pre-Vote의 과반수를 받은 경우에만 노드가 실제 선출을 진행

---

## 12. 참고 문헌

1. Ongaro, D. & Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." *USENIX ATC*. (Raft 논문)
2. Ongaro, D. (2014). "Consensus: Bridging Theory and Practice." PhD Dissertation, Stanford University. (확장 Raft)
3. Howard, H. (2014). "ARC: Analysis of Raft Consensus." Cambridge Technical Report.
4. etcd/raft 구현: https://github.com/etcd-io/raft
5. TiKV/raft-rs 구현: https://github.com/tikv/raft-rs
6. Ongaro, D. (2015). "Bug in Single-Server Membership Changes." (Raft 메일링 리스트)
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 9. O'Reilly Media.

---

[다음: 레슨 19 — Raft 구현 Part 2](./19_Raft_Implementation_Part2.md)
