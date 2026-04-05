# 레슨 19: Raft 구현 Part 2 — 멤버십 변경, 로그 압축, 선형화 가능성

[개요](./00_Overview.md) | [이전: Raft 구현 Part 1](./18_Raft_Implementation_Part1.md) | [다음: 분산 해시 테이블](./20_Distributed_Hash_Tables.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. joint consensus와 단일 서버 변경을 사용한 안전한 클러스터 멤버십 변경(membership change) 구현
2. 스냅샷(snapshot)을 사용한 로그 압축(log compaction)으로 메모리와 디스크 사용량 제한
3. 뒤처진 느린 팔로워를 위한 InstallSnapshot RPC 설계
4. ReadIndex와 LeaseRead 최적화를 사용한 선형화 가능 읽기(linearizable read) 구현
5. 처리량(throughput), 지연시간(latency), 리소스 소비를 위한 Raft 성능 튜닝(performance tuning)

---

## 목차

1. [멤버십 변경 개요](#1-멤버십-변경-개요)
2. [단일 서버 멤버십 변경](#2-단일-서버-멤버십-변경)
3. [Joint Consensus](#3-joint-consensus)
4. [로그 압축과 스냅샷](#4-로그-압축과-스냅샷)
5. [InstallSnapshot RPC](#5-installsnapshot-rpc)
6. [선형화 가능 읽기](#6-선형화-가능-읽기)
7. [성능 튜닝](#7-성능-튜닝)
8. [배칭과 파이프라이닝](#8-배칭과-파이프라이닝)
9. [완전한 구현](#9-완전한-구현)
10. [요약과 핵심 정리](#10-요약과-핵심-정리)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 멤버십 변경 개요

### 1.1 문제

클러스터가 실행 중인 동안 멤버십(노드 추가 또는 제거)을 변경하는 것은 Raft에서 가장 어려운 부분 중 하나이다. 근본적인 위험은 전환 중에 두 개의 분리된 과반수가 형성될 수 있다는 것이다 — 하나는 이전 구성에서, 다른 하나는 새 구성에서 — 두 리더와 안전성 위반으로 이어진다.

```
이전 구성: {A, B, C}        과반수 = 2
새 구성:   {A, B, C, D, E}  과반수 = 3

위험 기간:
  {A, B}가 이전 구성에서 과반수 형성 → 리더 선출
  {C, D, E}가 새 구성에서 과반수 형성 → 리더 선출
  두 리더가 동시에!
```

### 1.2 두 가지 접근법

Raft는 두 가지 해결책을 제공한다:

| 접근법 | 복잡도 | 안전성 | 가용성 |
|--------|--------|--------|--------|
| 단일 서버 변경 | 단순 | 안전 (한 번에 하나) | 짧은 비가용 |
| Joint consensus | 복잡 | 안전 (임의 변경) | 비가용 없음 |

---

## 2. 단일 서버 멤버십 변경

### 2.1 알고리즘

가장 간단한 접근법: 한 번에 하나의 서버만 추가하거나 제거한다. 최대 하나의 서버가 다른 구성의 두 과반수가 반드시 겹치므로 안전하다.

```python
import time
import json
import random
import hashlib
from typing import Optional, Dict, List, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MembershipChangeType(Enum):
    ADD_SERVER = "add_server"
    REMOVE_SERVER = "remove_server"


@dataclass
class ConfigurationEntry:
    """Raft 로그의 구성 변경 항목."""
    change_type: MembershipChangeType
    server_id: str
    old_config: set
    new_config: set


class RaftMembership:
    """
    단일 서버 변경으로 Raft 클러스터 멤버십을 관리한다.

    안전성 불변성: 한 번에 하나의 멤버십 변경만 진행 중
    (미커밋)일 수 있다. 이전 변경이 커밋될 때까지
    새 변경을 시작할 수 없다.
    """

    def __init__(self, initial_members: set[str]):
        self.current_config: set[str] = set(initial_members)
        self.pending_config: Optional[set[str]] = None
        self.change_log: list[ConfigurationEntry] = []
        self.committed_config_index: int = 0

    def propose_add(self, server_id: str) -> Optional[ConfigurationEntry]:
        """
        클러스터에 서버 추가를 제안한다.

        변경이 이미 보류 중이면 None을 반환한다.
        """
        if self.pending_config is not None:
            print(f"{server_id} 추가 불가: 변경이 이미 보류 중")
            return None

        if server_id in self.current_config:
            print(f"{server_id} 추가 불가: 이미 멤버")
            return None

        new_config = self.current_config | {server_id}
        entry = ConfigurationEntry(
            change_type=MembershipChangeType.ADD_SERVER,
            server_id=server_id,
            old_config=set(self.current_config),
            new_config=new_config,
        )

        self.pending_config = new_config
        self.change_log.append(entry)

        print(f"제안: ADD {server_id}")
        print(f"  이전 구성: {sorted(self.current_config)}")
        print(f"  새 구성: {sorted(new_config)}")
        print(f"  이전 과반수: {len(self.current_config) // 2 + 1}")
        print(f"  새 과반수: {len(new_config) // 2 + 1}")

        return entry

    def propose_remove(self, server_id: str) -> Optional[ConfigurationEntry]:
        """클러스터에서 서버 제거를 제안한다."""
        if self.pending_config is not None:
            print(f"{server_id} 제거 불가: 변경이 이미 보류 중")
            return None

        if server_id not in self.current_config:
            print(f"{server_id} 제거 불가: 멤버가 아님")
            return None

        new_config = self.current_config - {server_id}
        entry = ConfigurationEntry(
            change_type=MembershipChangeType.REMOVE_SERVER,
            server_id=server_id,
            old_config=set(self.current_config),
            new_config=new_config,
        )

        self.pending_config = new_config
        self.change_log.append(entry)
        return entry

    def commit_change(self):
        """보류 중인 구성 변경을 커밋한다."""
        if self.pending_config is None:
            return

        old = self.current_config
        self.current_config = self.pending_config
        self.pending_config = None
        self.committed_config_index += 1

        print(f"구성 변경 #{self.committed_config_index} 커밋됨")
        print(f"  {sorted(old)} → {sorted(self.current_config)}")

    def abort_change(self):
        """보류 중인 변경을 중단한다 (예: 리더십 상실)."""
        self.pending_config = None
        if self.change_log:
            self.change_log.pop()

    def majority_size(self, config: Optional[set] = None) -> int:
        """구성의 과반수 크기를 계산한다."""
        c = config or self.current_config
        return len(c) // 2 + 1

    def verify_single_overlap(self) -> bool:
        """
        이전 구성과 새 구성(1만큼 차이)이 항상 겹치는지 검증한다.

        모든 두 과반수 Q_old와 Q_new에 대해, Q_old ∩ Q_new ≠ ∅.
        """
        if self.pending_config is None:
            return True

        old = self.current_config
        new = self.pending_config
        old_majority = len(old) // 2 + 1
        new_majority = len(new) // 2 + 1

        # 겹침은 |old Δ new| = 1일 때 보장됨
        diff = old.symmetric_difference(new)
        if len(diff) > 1:
            print(f"경고: 구성이 {len(diff)}개 서버만큼 다름, 1이 아님!")
            return False

        # 형식적 검증: 가능한 모든 과반수 쌍 열거
        from itertools import combinations
        for q_old in combinations(old, old_majority):
            for q_new in combinations(new, new_majority):
                if not set(q_old) & set(q_new):
                    print(f"위반: {set(q_old)}과 {set(q_new)}이 겹치지 않음!")
                    return False

        return True


def demonstrate_membership_changes():
    """안전한 단일 서버 멤버십 변경을 시연한다."""
    print("=== 단일 서버 멤버십 변경 ===\n")

    membership = RaftMembership({"A", "B", "C"})

    # 서버 D 추가
    membership.propose_add("D")
    assert membership.verify_single_overlap()
    membership.commit_change()

    # 서버 E 추가
    membership.propose_add("E")
    assert membership.verify_single_overlap()
    membership.commit_change()

    # 서버 A 제거
    membership.propose_remove("A")
    assert membership.verify_single_overlap()
    membership.commit_change()

    print(f"\n최종 구성: {sorted(membership.current_config)}")
    print(f"변경 횟수: {membership.committed_config_index}")


demonstrate_membership_changes()
```

### 2.2 캐치업 단계(Catch-Up Phase)

새 서버를 추가하기 전에 로그를 따라잡아야 한다. 그렇지 않으면 새 서버가 모든 기록 항목을 수신하는 데 오랜 시간이 걸릴 수 있으며, 그 동안 커밋이 느려진다 (새 서버가 과반수에 기여할 수 없으므로).

```python
class ServerCatchUp:
    """
    클러스터에 합류하는 새 서버의 캐치업 단계를 관리한다.

    새 서버는 투표 멤버가 되기 전에 모든 기존 로그 항목을
    복제해야 한다. 캐치업 중에는 AppendEntries를 수신하지만
    과반수에 포함되지 않는 비투표 학습자(non-voting learner)이다.
    """

    def __init__(self, leader_log_length: int, rounds_threshold: int = 10):
        self.leader_log_length = leader_log_length
        self.rounds_threshold = rounds_threshold
        self.rounds: list[dict] = []
        self.start_time = time.time()

    def record_round(self, entries_behind: int):
        """복제 라운드 결과를 기록한다."""
        self.rounds.append({
            "entries_behind": entries_behind,
            "timestamp": time.time(),
        })

    def is_caught_up(self) -> bool:
        """
        새 서버가 충분히 따라잡았는지 결정한다.

        서버는 뒤처진 항목 수가 하나의 선출 타임아웃 내에
        복제될 수 있을 만큼 작을 때 준비된 것이다.
        """
        if len(self.rounds) < self.rounds_threshold:
            return False

        # 최근 몇 라운드 확인: entries_behind가 감소해야 함
        recent = self.rounds[-self.rounds_threshold:]
        last_behind = recent[-1]["entries_behind"]

        # 하트비트 간격 분량의 항목 이내이면 준비됨
        return last_behind <= 10  # 휴리스틱: 10개 항목 이내

    def report(self) -> dict:
        """캐치업 진행 보고서를 생성한다."""
        if not self.rounds:
            return {"status": "starting", "entries_behind": self.leader_log_length}

        current_behind = self.rounds[-1]["entries_behind"]
        elapsed = time.time() - self.start_time
        progress = 1.0 - (current_behind / max(self.leader_log_length, 1))

        return {
            "status": "catching_up" if not self.is_caught_up() else "ready",
            "entries_behind": current_behind,
            "rounds_completed": len(self.rounds),
            "progress_pct": round(progress * 100, 1),
            "elapsed_seconds": round(elapsed, 2),
        }


def demonstrate_catch_up():
    """새 서버의 캐치업 프로세스를 시연한다."""
    print("=== 서버 캐치업 단계 ===\n")

    leader_log_length = 10000
    catch_up = ServerCatchUp(leader_log_length)

    # 캐치업 라운드 시뮬레이션
    entries_behind = leader_log_length
    round_num = 0

    while not catch_up.is_caught_up():
        # 각 라운드에서 팔로워가 일부 항목을 따라잡지만
        # 리더가 새 항목을 추가할 수도 있음
        caught_up = min(entries_behind, random.randint(500, 2000))
        new_entries = random.randint(0, 50)  # 리더가 새 쓰기를 받음
        entries_behind = entries_behind - caught_up + new_entries
        entries_behind = max(0, entries_behind)

        catch_up.record_round(entries_behind)
        round_num += 1

        if round_num % 5 == 0:
            report = catch_up.report()
            print(f"  라운드 {round_num}: {report['progress_pct']}% 따라잡음, "
                  f"{report['entries_behind']} 뒤처짐")

    report = catch_up.report()
    print(f"\n  {report['rounds_completed']} 라운드 후 캐치업 완료")
    print(f"  서버가 투표 멤버로 합류 준비됨")


demonstrate_catch_up()
```

---

## 3. Joint Consensus

### 3.1 2단계 접근법

Joint consensus는 이전 구성과 새 구성 모두에서 과반수를 요구하는 과도기 구성을 사용하여 임의의 구성 변경을 허용한다:

```
단계 1: C_old → C_old,new (joint 구성)
  - 로그 항목: [C_old,new]
  - 결정에 C_old의 과반수와 C_new의 과반수 모두 필요

단계 2: C_old,new → C_new
  - 로그 항목: [C_new]
  - 커밋되면, C_new만 유효
```

```python
class JointConsensus:
    """
    임의의 멤버십 변경을 위한 joint consensus를 구현한다.

    joint 단계 동안, 이전 구성과 새 구성 모두가 독립적으로
    모든 결정에 대해 과반수를 형성해야 한다.
    """

    def __init__(self, config: set[str]):
        self.old_config: set[str] = set(config)
        self.new_config: Optional[set[str]] = None
        self.phase: str = "stable"  # stable, joint, transitioning

    def start_change(self, new_config: set[str]):
        """new_config로의 멤버십 변경을 시작한다."""
        assert self.phase == "stable", "전환 중에는 변경을 시작할 수 없음"
        self.new_config = set(new_config)
        self.phase = "joint"

        print(f"Joint consensus 진입:")
        print(f"  C_old: {sorted(self.old_config)}")
        print(f"  C_new: {sorted(self.new_config)}")
        print(f"  모든 결정에 양쪽 과반수 필요")

    def check_majority(self, votes: set[str]) -> bool:
        """
        투표 집합이 충분한 과반수를 형성하는지 확인한다.

        Joint 단계: 이전 구성과 새 구성 모두의 과반수 필요.
        안정 단계: 현재 구성의 과반수 필요.
        """
        if self.phase == "stable":
            needed = len(self.old_config) // 2 + 1
            have = len(votes & self.old_config)
            return have >= needed

        elif self.phase == "joint":
            old_needed = len(self.old_config) // 2 + 1
            old_have = len(votes & self.old_config)

            new_needed = len(self.new_config) // 2 + 1
            new_have = len(votes & self.new_config)

            return old_have >= old_needed and new_have >= new_needed

        return False

    def commit_joint(self):
        """
        Joint 구성 항목이 커밋되었다.
        이제 새 구성으로 전환한다.
        """
        assert self.phase == "joint"
        self.phase = "transitioning"
        print(f"Joint 구성 커밋됨. C_new로 전환 중...")

    def commit_new(self):
        """
        새 구성 항목이 커밋되었다.
        전환이 완료되었다.
        """
        assert self.phase == "transitioning"
        self.old_config = self.new_config
        self.new_config = None
        self.phase = "stable"
        print(f"전환 완료. 구성: {sorted(self.old_config)}")


def demonstrate_joint_consensus():
    """다중 서버 변경을 위한 joint consensus를 시연한다."""
    print("=== Joint Consensus ===\n")

    jc = JointConsensus({"A", "B", "C"})

    # {A,B,C}에서 {B,C,D,E}로 변경
    jc.start_change({"B", "C", "D", "E"})

    # Joint 단계에서 과반수 검사 테스트
    test_cases = [
        ({"A", "B", "C"}, "이전 과반수만"),
        ({"B", "C", "D"}, "이전 과반수 + 새 과반수"),
        ({"C", "D", "E"}, "새 과반수만"),
        ({"A", "B", "D", "E"}, "이전 과반수 + 새 과반수"),
        ({"B", "C", "D", "E"}, "이전 과반수 + 새 과반수"),
    ]

    for votes, desc in test_cases:
        result = jc.check_majority(votes)
        print(f"  투표 {sorted(votes):20s} ({desc}): {'통과' if result else '실패'}")

    # 전환 완료
    jc.commit_joint()
    jc.commit_new()


demonstrate_joint_consensus()
```

---

## 4. 로그 압축과 스냅샷

### 4.1 문제

압축 없이는 Raft 로그가 무한히 증가한다. 이것은 디스크 공간을 낭비하고, 시작 시간을 증가시키며 (전체 로그 재생), 새 팔로워에 대한 복제를 느리게 한다.

```
압축 없이:
  로그: [e1] [e2] [e3] ... [e10000] [e10001] ...
  ← 모든 항목이 영원히 유지됨 →

스냅샷 사용:
  인덱스 8000에서의 스냅샷:      로그 꼬리:
  ┌────────────────────┐       [e8001] [e8002] ... [e10001]
  │ 상태 머신 데이터    │       ← 최근 항목만 유지 →
  │ last_included_index│
  │ last_included_term │
  └────────────────────┘
```

### 4.2 스냅샷 구현

```python
@dataclass
class Snapshot:
    """Raft 상태 머신의 특정 시점 스냅샷."""
    last_included_index: int
    last_included_term: int
    data: dict  # 직렬화된 상태 머신 상태
    config: set  # 스냅샷 시점의 클러스터 구성
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        self.size_bytes = len(json.dumps(self.data))


class LogCompactor:
    """
    스냅샷을 통한 로그 압축을 관리한다.

    스냅샷은 로그가 설정 가능한 크기를 초과하면 트리거된다.
    스냅샷은 커밋된 인덱스의 상태 머신 상태를 캡처하고,
    해당 인덱스까지의 모든 로그 항목을 폐기한다.
    """

    # 로그가 이 항목 수를 초과하면 스냅샷
    SNAPSHOT_THRESHOLD = 1000
    # 스냅샷 후 최소 이 수의 항목을 유지 (느린 팔로워를 위해)
    MIN_LOG_RETENTION = 100

    def __init__(self):
        self.log: list[dict] = []
        self.log_offset: int = 0  # 로그의 첫 항목 인덱스
        self.snapshots: list[Snapshot] = []
        self.current_snapshot: Optional[Snapshot] = None
        self.state_machine: dict = {}
        self.commit_index: int = 0

    def append(self, entry: dict):
        """로그에 항목을 추가한다."""
        self.log.append(entry)

    def apply_up_to(self, index: int):
        """주어진 인덱스까지의 로그 항목을 상태 머신에 적용한다."""
        while self.commit_index < index:
            self.commit_index += 1
            relative_idx = self.commit_index - self.log_offset - 1
            if 0 <= relative_idx < len(self.log):
                entry = self.log[relative_idx]
                cmd = entry.get("command", {})
                if cmd.get("op") == "put":
                    self.state_machine[cmd["key"]] = cmd["value"]
                elif cmd.get("op") == "delete":
                    self.state_machine.pop(cmd.get("key"), None)

    def should_snapshot(self) -> bool:
        """로그가 스냅샷이 필요할 만큼 큰지 확인한다."""
        return len(self.log) > self.SNAPSHOT_THRESHOLD

    def create_snapshot(self) -> Snapshot:
        """
        현재 커밋 인덱스에서 스냅샷을 생성한다.

        단계:
        1. 상태 머신 상태를 직렬화
        2. 마지막 포함 인덱스와 텀 기록
        3. 스냅샷 이전의 로그 항목 폐기
        4. 스냅샷을 안정 저장소에 저장
        """
        assert self.commit_index > 0, "커밋된 항목 없이는 스냅샷할 수 없음"

        # 커밋된 항목의 텀 찾기
        relative_idx = self.commit_index - self.log_offset - 1
        last_term = self.log[relative_idx]["term"]

        snapshot = Snapshot(
            last_included_index=self.commit_index,
            last_included_term=last_term,
            data=dict(self.state_machine),
            config=set(),
        )

        # 스냅샷에 포함된 로그 항목 폐기
        # 느린 팔로워를 위해 MIN_LOG_RETENTION 항목 유지
        entries_to_discard = max(0, relative_idx + 1 - self.MIN_LOG_RETENTION)
        if entries_to_discard > 0:
            self.log = self.log[entries_to_discard:]
            self.log_offset += entries_to_discard

        self.current_snapshot = snapshot
        self.snapshots.append(snapshot)

        return snapshot

    def restore_from_snapshot(self, snapshot: Snapshot):
        """
        수신된 스냅샷에서 상태를 복원한다.

        팔로워가 로그 기반 복제로는 너무 뒤처져서
        InstallSnapshot RPC를 수신할 때 호출된다.
        """
        self.current_snapshot = snapshot
        self.state_machine = dict(snapshot.data)
        self.commit_index = snapshot.last_included_index
        self.log_offset = snapshot.last_included_index

        # 스냅샷 이전의 모든 로그 항목 폐기
        self.log = [
            e for e in self.log
            if e.get("index", 0) > snapshot.last_included_index
        ]

    def stats(self) -> dict:
        """압축 통계를 반환한다."""
        return {
            "log_length": len(self.log),
            "log_offset": self.log_offset,
            "commit_index": self.commit_index,
            "snapshots_taken": len(self.snapshots),
            "state_machine_keys": len(self.state_machine),
            "snapshot_size_bytes": self.current_snapshot.size_bytes if self.current_snapshot else 0,
        }


def demonstrate_log_compaction():
    """스냅샷을 사용한 로그 압축을 시연한다."""
    print("=== 로그 압축 ===\n")

    compactor = LogCompactor()
    compactor.SNAPSHOT_THRESHOLD = 100  # 시연을 위해 낮춤

    # 500개의 로그 항목 생성
    for i in range(1, 501):
        compactor.append({
            "term": 1,
            "index": i,
            "command": {"op": "put", "key": f"key_{i % 50}", "value": f"val_{i}"},
        })

        # 커밋된 항목으로 적용
        compactor.apply_up_to(i)

        # 스냅샷 필요 여부 확인
        if compactor.should_snapshot():
            snapshot = compactor.create_snapshot()
            print(f"  인덱스 {snapshot.last_included_index}에서 스냅샷: "
                  f"로그={len(compactor.log)} 항목, "
                  f"상태={len(compactor.state_machine)} 키, "
                  f"크기={snapshot.size_bytes} 바이트")

    print(f"\n최종 통계: {compactor.stats()}")


demonstrate_log_compaction()
```

---

## 5. InstallSnapshot RPC

### 5.1 스냅샷이 필요한 경우

리더는 팔로워의 `nextIndex`가 이미 압축된(스냅샷 후 폐기된) 로그 항목을 가리킬 때 InstallSnapshot RPC를 전송한다:

```
리더:    스냅샷[...인덱스 5000] | [5001] [5002] ... [6000]
                                   ^
팔로워가 인덱스 3000 필요 ─────────┘ 이미 압축됨!

리더는 로그 항목 대신 스냅샷을 보내야 한다.
```

### 5.2 청크 전송(Chunked Transfer)

큰 스냅샷은 네트워크를 차단하지 않도록 청크 단위로 전송된다:

```python
@dataclass
class SnapshotChunk:
    """전송 중인 스냅샷의 청크."""
    term: int
    leader_id: str
    last_included_index: int
    last_included_term: int
    offset: int
    data: bytes
    done: bool


class SnapshotTransfer:
    """
    리더와 팔로워 사이의 청크 스냅샷 전송을 관리한다.

    큰 스냅샷은 청크(기본 1MB)로 분할되어 순차적으로
    전송된다. 팔로워는 청크를 조립하고 모든 청크가 수신되면
    스냅샷을 적용한다.
    """

    CHUNK_SIZE = 1024 * 1024  # 1MB 청크

    def __init__(self, snapshot: Snapshot, leader_id: str, term: int):
        self.snapshot = snapshot
        self.leader_id = leader_id
        self.term = term
        self.serialized = json.dumps(snapshot.data).encode()
        self.total_size = len(self.serialized)
        self.offset = 0
        self.chunks_sent = 0

    def next_chunk(self) -> Optional[SnapshotChunk]:
        """전송할 다음 청크를 생성한다."""
        if self.offset >= self.total_size:
            return None

        end = min(self.offset + self.CHUNK_SIZE, self.total_size)
        chunk = SnapshotChunk(
            term=self.term,
            leader_id=self.leader_id,
            last_included_index=self.snapshot.last_included_index,
            last_included_term=self.snapshot.last_included_term,
            offset=self.offset,
            data=self.serialized[self.offset:end],
            done=(end >= self.total_size),
        )

        self.offset = end
        self.chunks_sent += 1
        return chunk

    def progress(self) -> float:
        """전송 진행률을 백분율로 반환한다."""
        return (self.offset / self.total_size * 100) if self.total_size > 0 else 100.0


class SnapshotReceiver:
    """팔로워 측에서 스냅샷 청크를 수신하고 조립한다."""

    def __init__(self):
        self.buffer: bytearray = bytearray()
        self.expected_offset: int = 0
        self.last_included_index: int = 0
        self.last_included_term: int = 0
        self.chunks_received: int = 0

    def receive_chunk(self, chunk: SnapshotChunk) -> Optional[Snapshot]:
        """
        수신된 스냅샷 청크를 처리한다.

        모든 청크가 수신되면 완전한 Snapshot을 반환한다.
        """
        if chunk.offset != self.expected_offset:
            # 순서가 맞지 않음 — 리셋
            self.buffer = bytearray()
            self.expected_offset = 0
            return None

        self.buffer.extend(chunk.data)
        self.expected_offset = chunk.offset + len(chunk.data)
        self.last_included_index = chunk.last_included_index
        self.last_included_term = chunk.last_included_term
        self.chunks_received += 1

        if chunk.done:
            # 스냅샷 조립
            data = json.loads(self.buffer.decode())
            snapshot = Snapshot(
                last_included_index=self.last_included_index,
                last_included_term=self.last_included_term,
                data=data,
                config=set(),
            )
            self._reset()
            return snapshot

        return None

    def _reset(self):
        """다음 전송을 위해 수신기 상태를 리셋한다."""
        self.buffer = bytearray()
        self.expected_offset = 0
        self.chunks_received = 0


def demonstrate_snapshot_transfer():
    """청크 스냅샷 전송을 시연한다."""
    print("=== 스냅샷 전송 ===\n")

    # 비교적 큰 스냅샷 생성
    data = {f"key_{i}": f"value_{i}" for i in range(1000)}
    snapshot = Snapshot(
        last_included_index=5000,
        last_included_term=3,
        data=data,
        config=set(),
    )

    # 송신자
    transfer = SnapshotTransfer(snapshot, "leader", term=3)
    transfer.CHUNK_SIZE = 4096  # 시연을 위해 더 작은 청크

    # 수신자
    receiver = SnapshotReceiver()

    print(f"스냅샷 크기: {transfer.total_size} 바이트")
    print(f"청크 크기: {transfer.CHUNK_SIZE} 바이트")

    result = None
    while result is None:
        chunk = transfer.next_chunk()
        if chunk is None:
            break
        result = receiver.receive_chunk(chunk)
        print(f"  청크 {transfer.chunks_sent}: offset={chunk.offset}, "
              f"size={len(chunk.data)}, done={chunk.done}, "
              f"progress={transfer.progress():.1f}%")

    if result:
        print(f"\n스냅샷 수신 성공!")
        print(f"  인덱스: {result.last_included_index}")
        print(f"  텀: {result.last_included_term}")
        print(f"  키 수: {len(result.data)}")


demonstrate_snapshot_transfer()
```

---

## 6. 선형화 가능 읽기

### 6.1 순진한 읽기의 문제

리더의 상태 머신에서 순진하게 읽는 것은 선형화 가능(linearizable)하지 않다. 리더가 오래된(파티션에서 과반수로부터 격리된) 상태일 수 있기 때문이다. 리더는 읽기를 제공하기 전에 자신이 여전히 리더인지 확인해야 한다.

```
시간 ──────────────────────────────────────►
       S1 (이전 리더)            S2 (새 리더)
       ┌──────────┐           ┌──────────┐
       │ x = 1    │           │ x = 2    │  ← 클라이언트가 S2에 x=2 기록
       └──────────┘           └──────────┘
              │
     클라이언트가 S1에서 x를 읽음
     1을 반환 ← 오래됨! 선형화 가능하지 않음.
```

### 6.2 ReadIndex

ReadIndex는 과반수를 확인하여 리더십을 확인한다:

```python
class LinearizableReader:
    """
    Raft를 위한 선형화 가능 읽기를 구현한다.

    세 가지 접근법:
    1. 로그 읽기(Log Read): 읽기를 로그 항목으로 처리 (간단하지만 느림)
    2. ReadIndex: 하트비트 라운드로 리더십 확인
    3. LeaseRead: 시간 기반 리스로 하트비트 건너뛰기
    """

    def __init__(self, node_id: str, peers: list[str]):
        self.node_id = node_id
        self.peers = peers
        self.commit_index = 0
        self.last_applied = 0
        self.state_machine: dict = {}
        self.pending_reads: list[dict] = []

        # 리스 기반
        self.lease_expiry: float = 0.0
        self.LEASE_DURATION: float = 0.1  # 100ms 리스

    def read_via_log(self, key: str) -> dict:
        """
        접근법 1: 로그 읽기 — 읽기를 로그 항목으로 처리한다.

        읽기 명령이 Raft 합의를 거쳐, 이전에 커밋된 모든
        쓰기를 볼 수 있도록 보장한다. 정확하지만 전체 합의
        라운드의 지연시간이 추가된다.
        """
        return {
            "method": "log_read",
            "action": "읽기를 로그 항목으로 제안 → 커밋 대기 → 적용",
            "latency": "1 RTT (합의 라운드)",
            "key": key,
        }

    def read_via_read_index(self, key: str) -> dict:
        """
        접근법 2: ReadIndex — 리더십을 확인한 후 commitIndex에서 읽기.

        단계:
        1. 현재 commitIndex를 readIndex로 기록
        2. 과반수에 하트비트 전송
        3. 과반수가 확인하면 우리가 여전히 리더
        4. 상태 머신이 readIndex까지 진행할 때까지 대기
        5. 상태 머신에서 읽기 실행

        지연시간: 1 하트비트 RTT (합의보다 훨씬 짧음)
        """
        read_index = self.commit_index

        # 하트비트 확인 시뮬레이션
        acks = self._send_heartbeats()
        majority = len(self.peers) // 2 + 1

        if acks >= majority:
            # 리더로 확인됨; 적용이 따라잡을 때까지 대기
            while self.last_applied < read_index:
                pass  # 실제로는: 비동기 대기

            value = self.state_machine.get(key)
            return {
                "method": "read_index",
                "key": key,
                "value": value,
                "read_at_index": read_index,
                "latency": "1 하트비트 RTT",
            }
        else:
            return {"method": "read_index", "error": "리더가 아님 (하트비트 실패)"}

    def read_via_lease(self, key: str) -> dict:
        """
        접근법 3: LeaseRead — 시간 기반 리스로 하트비트를 건너뛴다.

        리더의 리스가 만료되지 않았으면, 리더십을 확인하지 않고
        읽기를 제공할 수 있다. 이것은 제한된 시계 드리프트를 가정한다.

        경고: LeaseRead는 시계 정확도에 의존한다. 시계가 범위를
        넘어 드리프트하면 선형화 가능성이 위반될 수 있다.
        """
        now = time.time()

        if now < self.lease_expiry:
            # 리스가 유효 — 읽기를 즉시 제공
            value = self.state_machine.get(key)
            return {
                "method": "lease_read",
                "key": key,
                "value": value,
                "lease_remaining_ms": round((self.lease_expiry - now) * 1000, 1),
                "latency": "0 RTT (로컬 읽기)",
            }
        else:
            # 리스 만료 — ReadIndex로 폴백
            return self.read_via_read_index(key)

    def renew_lease(self, heartbeat_acks: int):
        """
        과반수의 하트비트가 성공하면 리스를 갱신한다.

        리스는 LEASE_DURATION 초 후에 만료되도록 설정된다.
        리더는 이 윈도우 내에 확인을 받아야 갱신할 수 있다.
        """
        majority = len(self.peers) // 2 + 1
        if heartbeat_acks >= majority:
            self.lease_expiry = time.time() + self.LEASE_DURATION

    def _send_heartbeats(self) -> int:
        """하트비트를 전송하고 확인을 수신하는 것을 시뮬레이션한다."""
        # 프로덕션에서는 타임아웃이 있는 비동기 방식
        return len(self.peers)  # 시연을 위해 모두 확인한다고 가정


def compare_read_approaches():
    """세 가지 선형화 가능 읽기 접근법을 비교한다."""
    print("=== 선형화 가능 읽기 접근법 ===\n")

    reader = LinearizableReader("leader", ["f1", "f2", "f3", "f4"])
    reader.commit_index = 100
    reader.last_applied = 100
    reader.state_machine = {"x": "42", "y": "hello"}
    reader.lease_expiry = time.time() + 1.0  # 활성 리스

    approaches = [
        ("로그 읽기", reader.read_via_log("x")),
        ("ReadIndex", reader.read_via_read_index("x")),
        ("LeaseRead", reader.read_via_lease("x")),
    ]

    for name, result in approaches:
        print(f"{name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print()

    # 비교 표
    print("비교:")
    print(f"  {'접근법':<15} {'지연시간':<20} {'안전성':<20} {'요구사항'}")
    print(f"  {'로그 읽기':<15} {'1 합의 RTT':<20} {'항상 안전':<20} {'없음'}")
    print(f"  {'ReadIndex':<15} {'1 하트비트 RTT':<20} {'항상 안전':<20} {'과반수 하트비트'}")
    print(f"  {'LeaseRead':<15} {'0 RTT (로컬)':<20} {'시계 의존':<20} {'제한된 시계 드리프트'}")


compare_read_approaches()
```

---

## 7. 성능 튜닝

### 7.1 핵심 성능 조절 매개변수

```python
@dataclass
class RaftPerformanceConfig:
    """
    Raft 구현의 튜닝 가능한 성능 매개변수.

    이 매개변수는 지연시간, 처리량, 리소스 소비
    사이의 트레이드오프를 조절한다.
    """
    # 선출 타이밍
    election_timeout_min_ms: int = 150
    election_timeout_max_ms: int = 300
    heartbeat_interval_ms: int = 50

    # 로그 배칭
    max_entries_per_append: int = 1000  # AppendEntries당 최대 항목
    max_batch_size_bytes: int = 1024 * 1024  # 1MB 최대 배치
    batch_wait_ms: int = 1  # 배치 축적을 위해 최대 1ms 대기

    # 스냅샷
    snapshot_threshold: int = 10000  # 스냅샷 전 항목 수
    snapshot_chunk_size: int = 1024 * 1024  # 1MB 청크

    # 파이프라인
    max_inflight_messages: int = 256  # 피어당 진행 중인 최대 AppendEntries
    pipeline_enabled: bool = True

    # 디스크
    sync_on_apply: bool = False  # 상태 머신 변경 시 fsync
    wal_sync_mode: str = "fdatasync"  # "none", "fdatasync", "fsync"

    def validate(self):
        """매개변수 관계를 검증한다."""
        errors = []

        # 하트비트 << 선출 타임아웃
        if self.heartbeat_interval_ms >= self.election_timeout_min_ms / 3:
            errors.append(
                f"하트비트 ({self.heartbeat_interval_ms}ms)가 "
                f"선출 타임아웃 최소 ({self.election_timeout_min_ms}ms)보다 훨씬 작아야 함"
            )

        # 선출 타임아웃 범위
        if self.election_timeout_max_ms <= self.election_timeout_min_ms:
            errors.append("선출 타임아웃 최대가 최소보다 커야 함")

        return errors


def analyze_performance_config():
    """다양한 성능 구성을 분석한다."""
    print("=== 성능 구성 분석 ===\n")

    configs = {
        "저지연": RaftPerformanceConfig(
            election_timeout_min_ms=100,
            election_timeout_max_ms=200,
            heartbeat_interval_ms=20,
            batch_wait_ms=0,
            max_inflight_messages=512,
        ),
        "고처리량": RaftPerformanceConfig(
            election_timeout_min_ms=500,
            election_timeout_max_ms=1000,
            heartbeat_interval_ms=100,
            batch_wait_ms=5,
            max_entries_per_append=5000,
            max_batch_size_bytes=4 * 1024 * 1024,
        ),
        "WAN 배포": RaftPerformanceConfig(
            election_timeout_min_ms=5000,
            election_timeout_max_ms=10000,
            heartbeat_interval_ms=1000,
            batch_wait_ms=10,
            snapshot_chunk_size=256 * 1024,
        ),
    }

    for name, config in configs.items():
        errors = config.validate()
        print(f"{name}:")
        print(f"  선출 타임아웃: [{config.election_timeout_min_ms}, "
              f"{config.election_timeout_max_ms}] ms")
        print(f"  하트비트: {config.heartbeat_interval_ms} ms")
        print(f"  배치 대기: {config.batch_wait_ms} ms")
        print(f"  최대 배치: {config.max_entries_per_append} 항목 / "
              f"{config.max_batch_size_bytes / 1024:.0f} KB")
        if errors:
            for e in errors:
                print(f"  경고: {e}")
        print()


analyze_performance_config()
```

---

## 8. 배칭과 파이프라이닝

### 8.1 요청 배칭(Request Batching)

여러 클라이언트 요청을 축적하여 단일 AppendEntries로 복제한다:

```python
class RequestBatcher:
    """
    효율적인 복제를 위해 클라이언트 요청을 배칭한다.

    한 번에 하나의 항목을 복제하는 대신, 배처는 `max_wait`까지
    요청을 축적하고 단일 AppendEntries 배치로 전송한다.
    """

    def __init__(self, max_size: int = 100, max_wait_ms: float = 1.0):
        self.max_size = max_size
        self.max_wait_ms = max_wait_ms
        self.batch: list[dict] = []
        self.batch_start_time: Optional[float] = None
        self.batches_flushed: int = 0
        self.total_entries: int = 0

    def add(self, entry: dict) -> Optional[list[dict]]:
        """
        배치에 항목을 추가한다.

        플러시해야 할 때 (크기 또는 시간 제한) 배치를 반환한다.
        """
        if not self.batch:
            self.batch_start_time = time.time()

        self.batch.append(entry)

        if len(self.batch) >= self.max_size:
            return self._flush()

        return None

    def check_timeout(self) -> Optional[list[dict]]:
        """타임아웃으로 인해 배치를 플러시해야 하는지 확인한다."""
        if not self.batch or self.batch_start_time is None:
            return None

        elapsed_ms = (time.time() - self.batch_start_time) * 1000
        if elapsed_ms >= self.max_wait_ms:
            return self._flush()

        return None

    def _flush(self) -> list[dict]:
        """현재 배치를 플러시한다."""
        batch = self.batch
        self.batch = []
        self.batch_start_time = None
        self.batches_flushed += 1
        self.total_entries += len(batch)
        return batch

    def stats(self) -> dict:
        return {
            "batches_flushed": self.batches_flushed,
            "total_entries": self.total_entries,
            "avg_batch_size": (
                self.total_entries / self.batches_flushed
                if self.batches_flushed > 0 else 0
            ),
            "pending": len(self.batch),
        }


def demonstrate_batching():
    """처리량 개선을 위한 요청 배칭을 시연한다."""
    print("=== 요청 배칭 ===\n")

    batcher = RequestBatcher(max_size=10, max_wait_ms=5.0)

    # 요청 폭주 시뮬레이션
    flushed_batches = []
    for i in range(50):
        entry = {"op": "put", "key": f"k{i}", "value": f"v{i}"}
        batch = batcher.add(entry)
        if batch:
            flushed_batches.append(batch)
            print(f"  {len(batch)}개 항목의 배치 플러시 (크기 제한)")

    # 남은 항목 확인
    remaining = batcher.check_timeout()
    if remaining:
        flushed_batches.append(remaining)

    print(f"\n배칭 통계: {batcher.stats()}")
    print(f"  배칭 없이: {50}개 AppendEntries RPC")
    print(f"  배칭 사용: {len(flushed_batches)}개 AppendEntries RPC")
    print(f"  감소: {(1 - len(flushed_batches)/50)*100:.0f}%")


demonstrate_batching()
```

### 8.2 파이프라인 복제(Pipeline Replication)

```python
class PipelinedReplicator:
    """
    지연시간 감소를 위한 파이프라인 로그 복제.

    각 AppendEntries 응답을 기다린 후 다음 배치를 전송하는
    대신, 여러 배치를 동시에 진행 중으로 파이프라인한다.
    """

    def __init__(self, peer_id: str, max_inflight: int = 8):
        self.peer_id = peer_id
        self.max_inflight = max_inflight
        self.inflight: list[dict] = []
        self.next_index: int = 1
        self.match_index: int = 0
        self.messages_sent: int = 0
        self.messages_acked: int = 0

    def can_send(self) -> bool:
        """다른 배치를 전송할 수 있는지 확인한다."""
        return len(self.inflight) < self.max_inflight

    def send_batch(self, entries: list[dict]) -> dict:
        """전송을 위해 배치를 큐에 넣는다."""
        msg = {
            "type": "AppendEntries",
            "to": self.peer_id,
            "prev_log_index": self.next_index - 1,
            "entries": entries,
            "batch_id": self.messages_sent,
        }
        self.inflight.append(msg)
        self.next_index += len(entries)
        self.messages_sent += 1
        return msg

    def ack(self, batch_id: int, success: bool, match_index: int):
        """확인을 처리한다."""
        self.inflight = [m for m in self.inflight if m["batch_id"] != batch_id]
        self.messages_acked += 1

        if success:
            self.match_index = max(self.match_index, match_index)
        else:
            # 재시도 필요 — 파이프라인 리셋
            self.next_index = match_index + 1
            self.inflight.clear()

    def stats(self) -> dict:
        return {
            "peer": self.peer_id,
            "inflight": len(self.inflight),
            "sent": self.messages_sent,
            "acked": self.messages_acked,
            "match_index": self.match_index,
        }


def demonstrate_pipelining():
    """순차적 vs 파이프라인 복제를 비교한다."""
    print("=== 파이프라인 복제 ===\n")

    rtt_ms = 1.0  # 1ms RTT
    num_batches = 20

    # 순차적: 한 번에 하나의 배치
    sequential_time = num_batches * rtt_ms
    print(f"순차적: {num_batches} 배치 × {rtt_ms}ms RTT = {sequential_time:.0f}ms")

    # 파이프라인: 여러 배치가 동시에 진행 중
    pipeline_depth = 8
    pipeline_time = rtt_ms + (num_batches - 1) * (rtt_ms / pipeline_depth)
    print(f"파이프라인 (깊이={pipeline_depth}): ~{pipeline_time:.1f}ms")
    print(f"속도 향상: {sequential_time / pipeline_time:.1f}배")


demonstrate_pipelining()
```

---

## 9. 완전한 구현

### 9.1 통합 테스트

```python
def integration_test():
    """
    전체 통합 테스트: 멤버십 변경 + 스냅샷 + 선형화 가능 읽기.
    """
    print("=== 통합 테스트 ===\n")

    # 1. 3개 노드 클러스터로 시작
    membership = RaftMembership({"n1", "n2", "n3"})
    compactor = LogCompactor()
    compactor.SNAPSHOT_THRESHOLD = 50

    reader = LinearizableReader("n1", ["n2", "n3"])

    # 2. 200개 항목 쓰기
    for i in range(1, 201):
        compactor.append({
            "term": 1,
            "index": i,
            "command": {"op": "put", "key": f"k{i % 20}", "value": f"v{i}"},
        })
        compactor.apply_up_to(i)

        if compactor.should_snapshot():
            snap = compactor.create_snapshot()
            print(f"  인덱스 {snap.last_included_index}에서 스냅샷, "
                  f"로그가 {len(compactor.log)} 항목으로 축소")

    # 3. 새 노드 추가
    print(f"\n  노드 n4 추가 중...")
    membership.propose_add("n4")
    membership.commit_change()

    # 4. 선형화 가능 읽기
    reader.state_machine = dict(compactor.state_machine)
    reader.commit_index = compactor.commit_index
    reader.last_applied = compactor.commit_index
    reader.lease_expiry = time.time() + 1.0

    result = reader.read_via_lease("k5")
    print(f"\n  LeaseRead('k5'): {result}")

    # 5. 요약
    print(f"\n  클러스터: {sorted(membership.current_config)}")
    print(f"  로그: {compactor.stats()}")
    print(f"  상태 머신: {len(compactor.state_machine)} 키")


integration_test()
```

---

## 10. 요약과 핵심 정리

### Raft Part 2 체크리스트

> **RAFT 구현 PART 2 체크리스트**
>
> ☐ 멤버십 변경: 한 번에 하나만 보류
> ☐ 새 서버가 투표자가 되기 전에 캐치업
> ☐ 다중 서버 변경을 위한 joint consensus
> ☐ 설정 가능한 임계값에서 스냅샷 트리거
> ☐ 청크 전송이 포함된 InstallSnapshot RPC
> ☐ 로그 오버헤드 없는 선형화 가능 읽기를 위한 ReadIndex
> ☐ 제로 RTT 읽기를 위한 LeaseRead (시계 가정 포함)
> ☐ 처리량을 위한 요청 배칭
> ☐ 지연시간을 위한 파이프라인 복제

### 핵심 통찰

1. **멤버십 변경은 위험하다**: 여기서 버그가 발생하면 스플릿 브레인(split-brain)이 생긴다. 단일 서버 변경이 더 간단하고 안전하다.
2. **스냅샷은 필수적이다**: 스냅샷 없이는 로그가 무한히 증가하고 새 노드가 합류하는 데 영원히 걸린다.
3. **선형화 가능 읽기는 작업이 필요하다**: 순진한 리더 읽기는 선형화 가능하지 않다. ReadIndex 또는 LeaseRead가 필요하다.
4. **성능 튜닝은 배포 환경에 따라 다르다**: LAN vs WAN은 다른 타임아웃이 필요하다. 배칭 vs 지연시간은 트레이드오프이다.

---

## 11. 연습 문제

### 문제 1: 멤버십 변경 안전성

단일 서버 멤버십 변경이 두 개의 분리된 과반수를 만들 수 없음을 증명하라. 커밋을 기다리지 않고 안전하게 연속으로 추가할 수 있는 최대 서버 수는 얼마인가?

### 문제 2: 스냅샷 최적화

스냅샷이 디스크에 직렬화되는 동안 상태 머신이 읽기를 계속 제공할 수 있는 copy-on-write 스냅샷 메커니즘을 설계하라. 메모리 오버헤드와 일관성을 고려하라.

### 문제 3: ReadIndex 지연시간

5개 노드 클러스터의 네트워크 RTT가 2ms일 때 다음의 읽기 지연시간을 계산하라:
1. 로그 읽기
2. ReadIndex
3. LeaseRead (유효한 리스 포함)
4. LeaseRead (만료된 리스 포함)

### 문제 4: 배칭 트레이드오프

1ms 네트워크 RTT에서 10,000 요청/초의 워크로드가 주어질 때:
- 최적 배치 크기와 대기 시간을 계산하라
- 배칭 사용/미사용 시 p99 지연시간은 얼마인가?
- 처리량 개선은 얼마인가?

### 문제 5: 구현 과제

리더로부터 스냅샷을 수신하는 `RaftNode.install_snapshot()`을 구현하라:
- 충돌하는 로그 항목 폐기
- 상태 머신 복원
- 커밋/적용 인덱스 업데이트

---

## 12. 참고 문헌

1. Ongaro, D. (2014). "Consensus: Bridging Theory and Practice." PhD Dissertation, Stanford University.
2. Ongaro, D. (2015). "Raft Membership Changes." (Raft 개발자 가이드)
3. Howard, H. et al. (2015). "Raft Refloated: Do We Have Consensus?" *Operating Systems Review*.
4. etcd 문서: Learner 모드와 멤버십 변경.
5. TiKV 블로그: "Raft Optimization" — 배칭, 파이프라이닝, 비동기 적용.
6. CockroachDB 블로그: "Living Without Atomic Clocks" — 실제 LeaseRead.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 9. O'Reilly Media.

---

[다음: 레슨 20 — 분산 해시 테이블](./20_Distributed_Hash_Tables.md)
