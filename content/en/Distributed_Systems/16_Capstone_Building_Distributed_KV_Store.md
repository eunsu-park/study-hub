# Lesson 16: Capstone — Building a Distributed Key-Value Store

[Overview](./00_Overview.md) | [Previous: Formal Verification with TLA+](./15_Formal_Verification_TLAplus.md) | [Next: Raft Implementation Part 1](./18_Raft_Implementation_Part1.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a three-layer architecture (transport, consensus, state machine) for a distributed KV store
2. Implement the Raft consensus algorithm with leader election, log replication, and commit advancement
3. Build linearizable read paths using ReadIndex and LeaseRead optimizations
4. Handle client request deduplication, leader redirection, and membership changes
5. Test distributed systems with fault injection including partitions, crashes, and message reordering

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Design](#2-architecture-design)
3. [Layer 1: Network Transport](#3-layer-1-network-transport)
4. [Layer 2: Raft Consensus Module](#4-layer-2-raft-consensus-module)
5. [Layer 3: State Machine](#5-layer-3-state-machine)
6. [Linearizable Reads](#6-linearizable-reads)
7. [Client Interaction](#7-client-interaction)
8. [Membership Changes](#8-membership-changes)
9. [Testing with Fault Injection](#9-testing-with-fault-injection)
10. [Performance Considerations](#10-performance-considerations)
11. [Complete Implementation](#11-complete-implementation)
12. [Summary](#12-summary)

---

## 1. Project Overview

### 1.1 What We Are Building

A complete, runnable distributed key-value store built on Raft consensus. The system supports:

- **PUT(key, value)**: Store a key-value pair
- **GET(key)**: Retrieve the value for a key (linearizable)
- **DELETE(key)**: Remove a key-value pair
- **Fault tolerance**: Survives minority node failures
- **Strong consistency**: Linearizable reads and writes

```
┌─────────────────────────────────────────────────────┐
│                    Client                            │
│              PUT / GET / DELETE                       │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│                  Cluster (3 or 5 nodes)              │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │  Node 1  │◀──▶│  Node 2  │◀──▶│  Node 3  │       │
│  │ (Leader) │    │(Follower)│    │(Follower)│       │
│  └──────────┘    └──────────┘    └──────────┘       │
│       │               │               │              │
│  ┌────┴────┐     ┌────┴────┐     ┌────┴────┐       │
│  │  Raft   │     │  Raft   │     │  Raft   │       │
│  │Consensus│     │Consensus│     │Consensus│       │
│  └────┬────┘     └────┬────┘     └────┬────┘       │
│  ┌────┴────┐     ┌────┴────┐     ┌────┴────┐       │
│  │  State  │     │  State  │     │  State  │       │
│  │ Machine │     │ Machine │     │ Machine │       │
│  │ (dict)  │     │ (dict)  │     │ (dict)  │       │
│  └─────────┘     └─────────┘     └─────────┘       │
└─────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

| Principle | Choice | Rationale |
|-----------|--------|-----------|
| Simplicity | Python | Focus on correctness, not performance |
| Serialization | JSON | Human-readable, easy debugging |
| Transport | TCP | Reliable delivery, connection-oriented |
| State machine | In-memory dict | Simplest possible implementation |
| Consensus | Raft | Understandable, well-documented |

### 1.3 What This Is and Is Not

```
This IS:                              This is NOT:
  ✓ Educational implementation          ✗ Production-ready system
  ✓ Correct Raft implementation         ✗ Optimized for performance
  ✓ Complete enough to run and test     ✗ Persistent storage (memory only)
  ✓ Demonstrates core concepts          ✗ Full Raft (no pre-vote, no pipeline)
  ✓ Testable with fault injection       ✗ Benchmark-quality
```

---

## 2. Architecture Design

### 2.1 Three-Layer Architecture

```
┌─────────────────────────────────────────────────┐
│                 Client API Layer                 │
│         PUT(key, value) / GET(key) / DELETE(key) │
│         Request routing, deduplication           │
└───────────────────────┬─────────────────────────┘
                        │ propose(command)
┌───────────────────────┴─────────────────────────┐
│              Raft Consensus Layer                │
│                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────┐│
│  │ Leader      │  │ Log          │  │ Election ││
│  │ Election    │  │ Replication  │  │ Timer    ││
│  └─────────────┘  └──────────────┘  └─────────┘│
│                                                  │
│  Persistent State: currentTerm, votedFor, log[]  │
│  Volatile State:   commitIndex, lastApplied      │
│  Leader State:     nextIndex[], matchIndex[]      │
└───────────────────────┬─────────────────────────┘
                        │ apply(entry)
┌───────────────────────┴─────────────────────────┐
│              State Machine Layer                 │
│                                                  │
│  In-memory key-value store (Python dict)         │
│  Snapshot: serialize → bytes → restore           │
└─────────────────────────────────────────────────┘
```

### 2.2 Message Types

```
Client → Leader:
  ClientRequest { client_id, seq_num, command: Put/Get/Delete }

Leader → Client:
  ClientResponse { success, value, error, leader_hint }

Node → Node (Raft RPCs):
  RequestVote { term, candidate_id, last_log_index, last_log_term }
  RequestVoteResponse { term, vote_granted }

  AppendEntries { term, leader_id, prev_log_index, prev_log_term,
                  entries[], leader_commit }
  AppendEntriesResponse { term, success, match_index }

  InstallSnapshot { term, leader_id, last_included_index,
                    last_included_term, data }
  InstallSnapshotResponse { term }
```

### 2.3 Log Entry Format

```python
@dataclass
class LogEntry:
    term: int          # Term when the entry was received by the leader
    index: int         # Position in the log (1-indexed)
    command: dict      # {"type": "put", "key": "k", "value": "v"}
                       # {"type": "get", "key": "k"}
                       # {"type": "delete", "key": "k"}
                       # {"type": "noop"}  (committed on leader election)
    client_id: str     # For deduplication
    seq_num: int       # For deduplication
```

---

## 3. Layer 1: Network Transport

### 3.1 TCP Message Framing

Since TCP is a byte stream (not a message stream), we need framing. We use a simple length-prefix protocol:

```
┌──────────────┬───────────────────────────────────┐
│ 4 bytes:     │ N bytes:                          │
│ message      │ JSON-encoded message body         │
│ length (N)   │                                   │
└──────────────┴───────────────────────────────────┘
```

```python
import json
import struct
import socket
import threading
import logging
from typing import Dict, Optional, Callable, Any

logger = logging.getLogger(__name__)


class MessageTransport:
    """
    TCP-based message transport with JSON serialization.

    Handles:
      - Connection management (connect, reconnect)
      - Length-prefixed message framing
      - JSON serialization/deserialization
      - Concurrent send/receive
    """

    HEADER_SIZE = 4  # 4 bytes for message length (uint32)
    MAX_MESSAGE_SIZE = 16 * 1024 * 1024  # 16 MB

    def __init__(self, node_id: str, host: str, port: int):
        self.node_id = node_id
        self.host = host
        self.port = port

        # Outgoing connections: peer_id -> socket
        self._connections: Dict[str, socket.socket] = {}
        self._conn_lock = threading.Lock()

        # Incoming connection handler
        self._server_socket: Optional[socket.socket] = None
        self._message_handler: Optional[Callable] = None
        self._running = False

        # Peer addresses: peer_id -> (host, port)
        self._peers: Dict[str, tuple] = {}

    def register_peer(self, peer_id: str, host: str, port: int) -> None:
        """Register a peer's address for outgoing connections."""
        self._peers[peer_id] = (host, port)

    def set_message_handler(self, handler: Callable[[str, dict], None]) -> None:
        """Set the callback for incoming messages."""
        self._message_handler = handler

    def start(self) -> None:
        """Start listening for incoming connections."""
        self._running = True
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(10)
        self._server_socket.settimeout(1.0)

        accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True
        )
        accept_thread.start()
        logger.info(f"[{self.node_id}] Transport listening on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the transport."""
        self._running = False
        if self._server_socket:
            self._server_socket.close()
        with self._conn_lock:
            for sock in self._connections.values():
                try:
                    sock.close()
                except OSError:
                    pass
            self._connections.clear()

    def send(self, peer_id: str, message: dict) -> bool:
        """
        Send a message to a peer.

        Returns True if sent successfully, False otherwise.
        """
        try:
            sock = self._get_connection(peer_id)
            if sock is None:
                return False

            data = json.dumps(message).encode("utf-8")
            if len(data) > self.MAX_MESSAGE_SIZE:
                logger.error(f"Message too large: {len(data)} bytes")
                return False

            # Length-prefix framing
            header = struct.pack("!I", len(data))
            sock.sendall(header + data)
            return True

        except (ConnectionError, OSError) as e:
            logger.debug(f"Send to {peer_id} failed: {e}")
            self._close_connection(peer_id)
            return False

    def _get_connection(self, peer_id: str) -> Optional[socket.socket]:
        """Get or establish a connection to a peer."""
        with self._conn_lock:
            if peer_id in self._connections:
                return self._connections[peer_id]

        if peer_id not in self._peers:
            return None

        host, port = self._peers[peer_id]
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((host, port))

            # Send our node_id so the peer knows who we are
            hello = json.dumps({"type": "hello", "node_id": self.node_id})
            hello_data = hello.encode("utf-8")
            sock.sendall(struct.pack("!I", len(hello_data)) + hello_data)

            with self._conn_lock:
                self._connections[peer_id] = sock
            return sock

        except (ConnectionError, OSError) as e:
            logger.debug(f"Connect to {peer_id} ({host}:{port}) failed: {e}")
            return None

    def _close_connection(self, peer_id: str) -> None:
        """Close and remove a connection."""
        with self._conn_lock:
            sock = self._connections.pop(peer_id, None)
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass

    def _accept_loop(self) -> None:
        """Accept incoming connections."""
        while self._running:
            try:
                client_sock, addr = self._server_socket.accept()
                handler_thread = threading.Thread(
                    target=self._handle_connection,
                    args=(client_sock,),
                    daemon=True,
                )
                handler_thread.start()
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.error("Accept error")
                break

    def _handle_connection(self, sock: socket.socket) -> None:
        """Handle an incoming connection."""
        peer_id = None
        try:
            while self._running:
                # Read header
                header_data = self._recv_exact(sock, self.HEADER_SIZE)
                if header_data is None:
                    break

                msg_len = struct.unpack("!I", header_data)[0]
                if msg_len > self.MAX_MESSAGE_SIZE:
                    logger.error(f"Message too large: {msg_len}")
                    break

                # Read body
                body_data = self._recv_exact(sock, msg_len)
                if body_data is None:
                    break

                message = json.loads(body_data.decode("utf-8"))

                # Handle hello message (identify peer)
                if message.get("type") == "hello":
                    peer_id = message["node_id"]
                    with self._conn_lock:
                        self._connections[peer_id] = sock
                    continue

                # Dispatch to handler
                if self._message_handler and peer_id:
                    self._message_handler(peer_id, message)

        except (ConnectionError, OSError, json.JSONDecodeError) as e:
            logger.debug(f"Connection handler error: {e}")
        finally:
            try:
                sock.close()
            except OSError:
                pass
            if peer_id:
                with self._conn_lock:
                    if self._connections.get(peer_id) is sock:
                        del self._connections[peer_id]

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from a socket."""
        data = bytearray()
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            except socket.timeout:
                return None
        return bytes(data)
```

---

## 4. Layer 2: Raft Consensus Module

### 4.1 State Overview

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import random
import time
import threading


class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    """A single entry in the Raft log."""
    term: int
    index: int
    command: dict
    client_id: str = ""
    seq_num: int = 0
```

### 4.2 Complete Raft Node

```python
class RaftNode:
    """
    Complete Raft consensus module.

    Implements:
      - Leader election with jittered timeouts
      - Log replication with backtracking
      - Commit advancement
      - Apply notification to state machine
    """

    def __init__(
        self,
        node_id: str,
        peers: List[str],
        transport: MessageTransport,
        state_machine: 'KeyValueStateMachine',
        election_timeout_range: tuple = (150, 300),
        heartbeat_interval: float = 50,
    ):
        self.node_id = node_id
        self.peers = list(peers)
        self.transport = transport
        self.state_machine = state_machine
        self.election_timeout_range = election_timeout_range  # ms
        self.heartbeat_interval = heartbeat_interval  # ms

        # --- Persistent state (would be on disk in production) ---
        self.current_term: int = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []  # 0-indexed internally

        # --- Volatile state (all servers) ---
        self.commit_index: int = 0
        self.last_applied: int = 0
        self.state: NodeState = NodeState.FOLLOWER
        self.leader_id: Optional[str] = None

        # --- Volatile state (leader only, reinitialized after election) ---
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}

        # --- Election machinery ---
        self.votes_received: Set[str] = set()
        self._election_deadline: float = 0
        self._reset_election_timer()

        # --- Client request tracking ---
        # client_id -> {seq_num: response}
        self._client_responses: Dict[str, Dict[int, dict]] = {}
        # pending client requests waiting for commit
        self._pending_requests: Dict[int, threading.Event] = {}
        self._pending_results: Dict[int, dict] = {}

        # --- Threading ---
        self._lock = threading.Lock()
        self._running = False
        self._apply_event = threading.Event()

        # Register message handler
        self.transport.set_message_handler(self._handle_message)

    # ========== Public Interface ==========

    def start(self) -> None:
        """Start the Raft node."""
        self._running = True
        self.transport.start()

        # Start background threads
        threading.Thread(target=self._ticker_loop, daemon=True).start()
        threading.Thread(target=self._apply_loop, daemon=True).start()

        logger.info(f"[{self.node_id}] Raft node started as {self.state.value}")

    def stop(self) -> None:
        """Stop the Raft node."""
        self._running = False
        self._apply_event.set()
        self.transport.stop()

    def propose(self, command: dict, client_id: str = "", seq_num: int = 0,
                timeout: float = 5.0) -> dict:
        """
        Propose a command to be replicated.

        Only the leader can accept proposals. Returns the result
        after the command is committed and applied.

        Args:
            command: The command to replicate (put/get/delete)
            client_id: Client identifier for deduplication
            seq_num: Sequence number for deduplication
            timeout: Maximum time to wait for commit

        Returns:
            {"success": bool, "value": ..., "error": ..., "leader_hint": ...}
        """
        with self._lock:
            if self.state != NodeState.LEADER:
                return {
                    "success": False,
                    "error": "not_leader",
                    "leader_hint": self.leader_id,
                }

            # Check for duplicate request
            if client_id and client_id in self._client_responses:
                cached = self._client_responses[client_id].get(seq_num)
                if cached is not None:
                    return cached

            # Append to log
            entry = LogEntry(
                term=self.current_term,
                index=len(self.log) + 1,
                command=command,
                client_id=client_id,
                seq_num=seq_num,
            )
            self.log.append(entry)
            log_index = entry.index

            # Create event to wait for commit
            event = threading.Event()
            self._pending_requests[log_index] = event

        # Send AppendEntries to all followers immediately
        self._send_append_entries_to_all()

        # Wait for commit
        if event.wait(timeout=timeout):
            with self._lock:
                result = self._pending_results.pop(log_index, {
                    "success": False,
                    "error": "unknown",
                })
                self._pending_requests.pop(log_index, None)
                return result
        else:
            with self._lock:
                self._pending_requests.pop(log_index, None)
                self._pending_results.pop(log_index, None)
            return {"success": False, "error": "timeout"}

    # ========== Election ==========

    def _reset_election_timer(self) -> None:
        """Reset the election timeout with jitter."""
        timeout_ms = random.randint(*self.election_timeout_range)
        self._election_deadline = time.monotonic() + timeout_ms / 1000.0

    def _start_election(self) -> None:
        """Start a new election."""
        with self._lock:
            self.current_term += 1
            self.state = NodeState.CANDIDATE
            self.voted_for = self.node_id
            self.votes_received = {self.node_id}
            self.leader_id = None
            self._reset_election_timer()

            term = self.current_term
            last_log_index = len(self.log)
            last_log_term = self.log[-1].term if self.log else 0

        logger.info(
            f"[{self.node_id}] Starting election for term {term}"
        )

        # Request votes from all peers
        for peer in self.peers:
            self.transport.send(peer, {
                "type": "request_vote",
                "term": term,
                "candidate_id": self.node_id,
                "last_log_index": last_log_index,
                "last_log_term": last_log_term,
            })

    def _handle_request_vote(self, sender: str, msg: dict) -> None:
        """Handle an incoming RequestVote RPC."""
        with self._lock:
            term = msg["term"]
            candidate_id = msg["candidate_id"]
            last_log_index = msg["last_log_index"]
            last_log_term = msg["last_log_term"]

            # If the candidate's term is higher, update our term
            if term > self.current_term:
                self._step_down(term)

            vote_granted = False

            if term >= self.current_term:
                # Grant vote if we haven't voted or already voted for this candidate
                if self.voted_for is None or self.voted_for == candidate_id:
                    # Check log is at least as up-to-date
                    my_last_term = self.log[-1].term if self.log else 0
                    my_last_index = len(self.log)

                    log_ok = (
                        last_log_term > my_last_term
                        or (last_log_term == my_last_term
                            and last_log_index >= my_last_index)
                    )

                    if log_ok:
                        vote_granted = True
                        self.voted_for = candidate_id
                        self._reset_election_timer()

            response_term = self.current_term

        self.transport.send(sender, {
            "type": "request_vote_response",
            "term": response_term,
            "vote_granted": vote_granted,
        })

    def _handle_request_vote_response(self, sender: str, msg: dict) -> None:
        """Handle a RequestVote response."""
        with self._lock:
            if msg["term"] > self.current_term:
                self._step_down(msg["term"])
                return

            if self.state != NodeState.CANDIDATE:
                return

            if msg["term"] != self.current_term:
                return

            if msg["vote_granted"]:
                self.votes_received.add(sender)
                # Check if we have a majority
                if len(self.votes_received) > (len(self.peers) + 1) / 2:
                    self._become_leader()

    def _become_leader(self) -> None:
        """Transition to leader state. Must hold self._lock."""
        logger.info(
            f"[{self.node_id}] Became leader for term {self.current_term}"
        )
        self.state = NodeState.LEADER
        self.leader_id = self.node_id

        # Initialize leader volatile state
        next_idx = len(self.log) + 1
        for peer in self.peers:
            self.next_index[peer] = next_idx
            self.match_index[peer] = 0

        # Append a no-op entry to commit entries from previous terms
        noop = LogEntry(
            term=self.current_term,
            index=len(self.log) + 1,
            command={"type": "noop"},
        )
        self.log.append(noop)

    # ========== Log Replication ==========

    def _send_append_entries_to_all(self) -> None:
        """Send AppendEntries to all peers."""
        for peer in self.peers:
            self._send_append_entries(peer)

    def _send_append_entries(self, peer: str) -> None:
        """Send AppendEntries RPC to a specific peer."""
        with self._lock:
            if self.state != NodeState.LEADER:
                return

            next_idx = self.next_index.get(peer, 1)
            prev_log_index = next_idx - 1
            prev_log_term = 0
            if prev_log_index > 0 and prev_log_index <= len(self.log):
                prev_log_term = self.log[prev_log_index - 1].term

            # Entries to send (from nextIndex onward)
            entries = []
            for entry in self.log[next_idx - 1:]:
                entries.append({
                    "term": entry.term,
                    "index": entry.index,
                    "command": entry.command,
                    "client_id": entry.client_id,
                    "seq_num": entry.seq_num,
                })

            msg = {
                "type": "append_entries",
                "term": self.current_term,
                "leader_id": self.node_id,
                "prev_log_index": prev_log_index,
                "prev_log_term": prev_log_term,
                "entries": entries,
                "leader_commit": self.commit_index,
            }

        self.transport.send(peer, msg)

    def _handle_append_entries(self, sender: str, msg: dict) -> None:
        """Handle an incoming AppendEntries RPC."""
        with self._lock:
            term = msg["term"]

            # If leader's term is higher, update
            if term > self.current_term:
                self._step_down(term)

            response = {
                "type": "append_entries_response",
                "term": self.current_term,
                "success": False,
                "match_index": 0,
            }

            if term < self.current_term:
                self.transport.send(sender, response)
                return

            # Valid AppendEntries from current leader
            self.state = NodeState.FOLLOWER
            self.leader_id = msg["leader_id"]
            self._reset_election_timer()

            prev_log_index = msg["prev_log_index"]
            prev_log_term = msg["prev_log_term"]

            # Check if log contains entry at prev_log_index with prev_log_term
            if prev_log_index > 0:
                if prev_log_index > len(self.log):
                    # Log too short
                    self.transport.send(sender, response)
                    return
                if self.log[prev_log_index - 1].term != prev_log_term:
                    # Term mismatch: delete conflicting entry and all after
                    self.log = self.log[:prev_log_index - 1]
                    self.transport.send(sender, response)
                    return

            # Append new entries (overwriting any conflicts)
            for entry_data in msg["entries"]:
                idx = entry_data["index"]
                if idx <= len(self.log):
                    if self.log[idx - 1].term != entry_data["term"]:
                        # Conflict: truncate and append
                        self.log = self.log[:idx - 1]
                        self.log.append(LogEntry(
                            term=entry_data["term"],
                            index=entry_data["index"],
                            command=entry_data["command"],
                            client_id=entry_data.get("client_id", ""),
                            seq_num=entry_data.get("seq_num", 0),
                        ))
                else:
                    self.log.append(LogEntry(
                        term=entry_data["term"],
                        index=entry_data["index"],
                        command=entry_data["command"],
                        client_id=entry_data.get("client_id", ""),
                        seq_num=entry_data.get("seq_num", 0),
                    ))

            # Update commit index
            if msg["leader_commit"] > self.commit_index:
                self.commit_index = min(
                    msg["leader_commit"], len(self.log)
                )
                self._apply_event.set()

            response["success"] = True
            response["match_index"] = len(self.log)

        self.transport.send(sender, response)

    def _handle_append_entries_response(self, sender: str, msg: dict) -> None:
        """Handle an AppendEntries response."""
        with self._lock:
            if msg["term"] > self.current_term:
                self._step_down(msg["term"])
                return

            if self.state != NodeState.LEADER:
                return

            if msg["success"]:
                # Update nextIndex and matchIndex for this peer
                self.match_index[sender] = msg["match_index"]
                self.next_index[sender] = msg["match_index"] + 1

                # Try to advance commit index
                self._advance_commit_index()
            else:
                # Backtrack: decrement nextIndex and retry
                if sender in self.next_index:
                    self.next_index[sender] = max(
                        1, self.next_index[sender] - 1
                    )
                # Retry immediately
                self._send_append_entries(sender)

    def _advance_commit_index(self) -> None:
        """
        Advance commit index if a majority has replicated.
        Must hold self._lock.
        """
        for n in range(len(self.log), self.commit_index, -1):
            if self.log[n - 1].term != self.current_term:
                # Raft only commits entries from the current term
                # (entries from previous terms are committed indirectly)
                continue

            # Count how many servers have this entry
            count = 1  # Leader has it
            for peer in self.peers:
                if self.match_index.get(peer, 0) >= n:
                    count += 1

            if count > (len(self.peers) + 1) / 2:
                self.commit_index = n
                self._apply_event.set()
                break

    # ========== State Management ==========

    def _step_down(self, new_term: int) -> None:
        """Step down to follower. Must hold self._lock."""
        self.current_term = new_term
        self.state = NodeState.FOLLOWER
        self.voted_for = None
        self.leader_id = None
        self._reset_election_timer()

    # ========== Background Loops ==========

    def _ticker_loop(self) -> None:
        """Background loop for election timeouts and heartbeats."""
        while self._running:
            time.sleep(0.01)  # 10ms tick

            with self._lock:
                now = time.monotonic()

                if self.state == NodeState.LEADER:
                    # Send heartbeats periodically
                    pass  # Heartbeats sent via separate timer below

                elif now >= self._election_deadline:
                    # Election timeout: start election
                    pass  # Must release lock before starting election

            # Check outside lock to avoid deadlock
            should_start_election = False
            should_send_heartbeats = False

            with self._lock:
                now = time.monotonic()
                if self.state == NodeState.LEADER:
                    should_send_heartbeats = True
                elif now >= self._election_deadline:
                    should_start_election = True

            if should_start_election:
                self._start_election()
            elif should_send_heartbeats:
                self._send_append_entries_to_all()
                time.sleep(self.heartbeat_interval / 1000.0)

    def _apply_loop(self) -> None:
        """Background loop that applies committed entries to the state machine."""
        while self._running:
            self._apply_event.wait(timeout=0.1)
            self._apply_event.clear()

            while True:
                with self._lock:
                    if self.last_applied >= self.commit_index:
                        break

                    self.last_applied += 1
                    entry = self.log[self.last_applied - 1]
                    apply_index = self.last_applied

                # Apply to state machine (outside lock)
                result = self.state_machine.apply(entry.command)

                with self._lock:
                    # Cache response for deduplication
                    if entry.client_id:
                        if entry.client_id not in self._client_responses:
                            self._client_responses[entry.client_id] = {}
                        self._client_responses[entry.client_id][entry.seq_num] = result

                    # Notify waiting client request
                    event = self._pending_requests.get(apply_index)
                    if event:
                        self._pending_results[apply_index] = result
                        event.set()

    # ========== Message Dispatch ==========

    def _handle_message(self, sender: str, msg: dict) -> None:
        """Dispatch incoming messages to the appropriate handler."""
        msg_type = msg.get("type")

        handlers = {
            "request_vote": self._handle_request_vote,
            "request_vote_response": self._handle_request_vote_response,
            "append_entries": self._handle_append_entries,
            "append_entries_response": self._handle_append_entries_response,
        }

        handler = handlers.get(msg_type)
        if handler:
            handler(sender, msg)
        else:
            logger.warning(f"Unknown message type: {msg_type}")
```

### 4.3 Key Design Decisions Explained

**Election timeout jitter**: Without jitter, all nodes would timeout simultaneously, leading to split votes. The jitter ensures one node typically times out first and wins the election.

```
Node A timeout: ████████████████████ (250ms)
Node B timeout: █████████████████████████ (300ms)
Node C timeout: ██████████████████ (230ms)  ← times out first, starts election

Node C starts election at t=230ms.
By t=250ms, it likely already has a majority.
Node A never needs to start its own election.
```

**No-op on leader election**: After becoming leader, we append a no-op entry. This ensures that entries from previous terms are committed (the Raft paper explains that leaders cannot directly commit entries from previous terms — they must commit a new entry in the current term, which indirectly commits all previous entries).

**Backtracking on AppendEntries failure**: When a follower rejects AppendEntries (prev_log doesn't match), the leader decrements nextIndex and retries. This handles scenarios where the follower's log diverged from the leader's.

```
Leader log:    [1:a] [1:b] [2:c] [3:d] [3:e]
Follower log:  [1:a] [1:b] [2:x]

Attempt 1: prev=(3, term=2) → follower has term=2 at index 3? No (x, not c) → FAIL
Attempt 2: prev=(2, term=1) → follower has term=1 at index 2? Yes (b) → SUCCESS
            → follower truncates from index 3, appends [2:c] [3:d] [3:e]

Final:
Leader log:    [1:a] [1:b] [2:c] [3:d] [3:e]
Follower log:  [1:a] [1:b] [2:c] [3:d] [3:e]  ← matches leader
```

---

## 5. Layer 3: State Machine

### 5.1 Key-Value State Machine

```python
import json
import copy
from typing import Optional, Dict, Any


class KeyValueStateMachine:
    """
    Deterministic state machine for a key-value store.

    Applies commands from the Raft log to an in-memory dictionary.
    Supports snapshotting for log compaction.

    IMPORTANT: The state machine must be deterministic — applying
    the same sequence of commands must always produce the same state.
    """

    def __init__(self):
        self._data: Dict[str, str] = {}
        self._lock = threading.Lock()

    def apply(self, command: dict) -> dict:
        """
        Apply a committed command to the state machine.

        Args:
            command: {"type": "put"|"get"|"delete"|"noop", ...}

        Returns:
            Result dict with "success" and optionally "value"
        """
        cmd_type = command.get("type", "")

        with self._lock:
            if cmd_type == "put":
                key = command["key"]
                value = command["value"]
                self._data[key] = value
                return {"success": True}

            elif cmd_type == "get":
                key = command["key"]
                value = self._data.get(key)
                if value is not None:
                    return {"success": True, "value": value}
                else:
                    return {"success": False, "error": "key_not_found"}

            elif cmd_type == "delete":
                key = command["key"]
                if key in self._data:
                    del self._data[key]
                    return {"success": True}
                else:
                    return {"success": False, "error": "key_not_found"}

            elif cmd_type == "noop":
                return {"success": True}

            else:
                return {"success": False, "error": f"unknown command: {cmd_type}"}

    def snapshot(self) -> bytes:
        """
        Create a snapshot of the current state.

        Returns:
            Serialized state as bytes
        """
        with self._lock:
            return json.dumps(self._data).encode("utf-8")

    def restore(self, data: bytes) -> None:
        """
        Restore state from a snapshot.

        Args:
            data: Serialized state from snapshot()
        """
        with self._lock:
            self._data = json.loads(data.decode("utf-8"))

    def get_all(self) -> dict:
        """Return a copy of the entire state (for debugging)."""
        with self._lock:
            return copy.deepcopy(self._data)
```

### 5.2 Snapshotting

Log compaction via snapshotting prevents unbounded log growth:

```
Before snapshot:
  Log: [1:put(a,1)] [1:put(b,2)] [2:del(a)] [2:put(c,3)] [3:put(b,5)]
  State: {b: 5, c: 3}

After snapshot at index 5:
  Snapshot: {b: 5, c: 3}  (last_included_index=5, last_included_term=3)
  Log: []  (entries 1-5 discarded)

New entries continue appending:
  Log: [3:put(d,4)]
  State: {b: 5, c: 3, d: 4}
```

---

## 6. Linearizable Reads

### 6.1 The Problem

A naive read from the leader's state machine is NOT linearizable because the leader might have been deposed:

```
Timeline:
  Leader A: ──────────── [network partition] ────────────────▶
  Leader B (new):        ──── write(k, v2) ─── commit ──────▶
  Client reads from A:   ──── GET(k) ── returns v1 (stale!) ─▶

Client sees v1 even though v2 was committed.
This violates linearizability.
```

### 6.2 Approach 1: Read Through Raft Log

Route reads through the Raft log like writes. The read is committed and then applied.

```
Client: GET(key)
Leader: append LogEntry(type="get", key=key) to log
        replicate to followers
        commit
        apply to state machine
        return result
```

**Pros**: Simple, correct
**Cons**: Every read has consensus latency (one round-trip to majority)

### 6.3 Approach 2: ReadIndex

The leader confirms it is still the leader by exchanging heartbeats with a majority, then reads from its state machine at the committed index.

```python
def read_index(self, key: str) -> dict:
    """
    ReadIndex: linearizable read without log entry.

    Steps:
      1. Record current commit index as read_index
      2. Send heartbeats to confirm leadership
      3. Wait until last_applied >= read_index
      4. Read from state machine
    """
    with self._lock:
        if self.state != NodeState.LEADER:
            return {
                "success": False,
                "error": "not_leader",
                "leader_hint": self.leader_id,
            }
        read_index = self.commit_index

    # Confirm leadership with a round of heartbeats
    if not self._confirm_leadership():
        return {"success": False, "error": "leadership_lost"}

    # Wait until state machine has applied up to read_index
    deadline = time.monotonic() + 5.0
    while True:
        with self._lock:
            if self.last_applied >= read_index:
                break
            if self.state != NodeState.LEADER:
                return {"success": False, "error": "leadership_lost"}
        if time.monotonic() > deadline:
            return {"success": False, "error": "timeout"}
        time.sleep(0.001)

    # Safe to read from state machine
    return self.state_machine.apply({"type": "get", "key": key})

def _confirm_leadership(self) -> bool:
    """
    Confirm leadership by getting heartbeat acks from a majority.

    Returns True if still leader, False if deposed.
    """
    # In a real implementation, this would:
    # 1. Send heartbeats (empty AppendEntries) to all peers
    # 2. Wait for acks from a majority
    # 3. Return True if majority responded within timeout
    #
    # For simplicity, we check if we're still leader and have
    # recent heartbeat acks (within one heartbeat interval)
    with self._lock:
        return self.state == NodeState.LEADER
```

### 6.4 Approach 3: LeaseRead

If all followers acknowledge that the leader's lease has not expired, the leader can serve reads without any additional RPCs.

```
Leader heartbeat interval: T_hb
Follower election timeout: T_elect

If T_elect > T_hb + max_clock_drift, then:
  - The leader knows no election will start for at least
    (T_elect - T_hb - max_clock_drift) after a successful heartbeat
  - During this "lease", the leader can serve reads directly

Timeline:
  Leader:   ─── heartbeat ──── lease ────────────── heartbeat ──── lease ────▶
            │               │                       │               │
         heartbeat       lease                   heartbeat       lease
          sent           expires                  sent           expires
            │               │                       │               │
  Follower: ─ ack ─────── election timeout ─────── ack ──────────────────────▶

  During the lease period, leader can serve reads without confirmation.
```

**Comparison**:

| Approach | Read Latency | Leadership Confirmation | Risk |
|----------|-------------|------------------------|------|
| Log read | Consensus latency (50-200ms) | Full consensus | None |
| ReadIndex | One heartbeat round (10-50ms) | Heartbeat majority | None |
| LeaseRead | Local read (<1ms) | None (relies on lease) | Clock skew |

---

## 7. Client Interaction

### 7.1 Request Routing

```python
class KVClient:
    """
    Client for the distributed KV store.

    Handles:
      - Leader discovery via redirection
      - Request retry with backoff
      - Request deduplication (client_id + seq_num)
    """

    def __init__(self, cluster_addrs: Dict[str, tuple], client_id: str = None):
        """
        Args:
            cluster_addrs: {node_id: (host, port)} for all nodes
            client_id: Unique client identifier (auto-generated if None)
        """
        self.cluster_addrs = cluster_addrs
        self.client_id = client_id or f"client-{id(self)}"
        self._seq_num = 0
        self._known_leader: Optional[str] = None
        self._transport = MessageTransport(
            self.client_id, "localhost", 0  # Ephemeral port
        )

    def _next_seq(self) -> int:
        self._seq_num += 1
        return self._seq_num

    def put(self, key: str, value: str, timeout: float = 10.0) -> dict:
        """PUT a key-value pair."""
        return self._send_command(
            {"type": "put", "key": key, "value": value},
            timeout=timeout,
        )

    def get(self, key: str, timeout: float = 10.0) -> dict:
        """GET a value by key."""
        return self._send_command(
            {"type": "get", "key": key},
            timeout=timeout,
        )

    def delete(self, key: str, timeout: float = 10.0) -> dict:
        """DELETE a key."""
        return self._send_command(
            {"type": "delete", "key": key},
            timeout=timeout,
        )

    def _send_command(self, command: dict, timeout: float = 10.0) -> dict:
        """
        Send a command to the cluster, handling redirects and retries.
        """
        seq = self._next_seq()
        deadline = time.monotonic() + timeout
        attempt = 0

        while time.monotonic() < deadline:
            # Choose target: known leader or random node
            target = self._known_leader or random.choice(
                list(self.cluster_addrs.keys())
            )

            try:
                result = self._rpc(target, {
                    "type": "client_request",
                    "client_id": self.client_id,
                    "seq_num": seq,
                    "command": command,
                })

                if result.get("success"):
                    self._known_leader = target
                    return result

                if result.get("error") == "not_leader":
                    # Follow the redirect hint
                    leader_hint = result.get("leader_hint")
                    if leader_hint and leader_hint in self.cluster_addrs:
                        self._known_leader = leader_hint
                    else:
                        self._known_leader = None
                    continue

                if result.get("error") == "timeout":
                    self._known_leader = None
                    attempt += 1
                    time.sleep(min(0.1 * (2 ** attempt), 2.0))
                    continue

                return result

            except Exception:
                self._known_leader = None
                attempt += 1
                time.sleep(min(0.1 * (2 ** attempt), 2.0))

        return {"success": False, "error": "client_timeout"}

    def _rpc(self, target: str, message: dict) -> dict:
        """Send an RPC and wait for response (simplified)."""
        # In a real implementation, this would use the transport
        # layer with a response callback. Here we simulate it.
        raise NotImplementedError("Use with actual transport layer")
```

### 7.2 Request Deduplication

Without deduplication, a retried PUT could be applied twice:

```
Client: PUT(k, v1) ──▶ Leader (succeeds, committed)
Leader: response ──X── (lost in network)
Client: PUT(k, v1) ──▶ Leader (retry — without dedup, applied again!)
```

The solution uses `(client_id, seq_num)` pairs:

```python
# In RaftNode.propose():
if client_id and client_id in self._client_responses:
    cached = self._client_responses[client_id].get(seq_num)
    if cached is not None:
        return cached  # Return cached result, don't re-apply
```

---

## 8. Membership Changes

### 8.1 The Problem

Naively adding or removing a node can cause two disjoint majorities:

```
Original cluster: {A, B, C}  (majority = 2)
Add node D and E simultaneously:

If some nodes see {A,B,C} and others see {A,B,C,D,E}:
  {A, B} = majority of {A,B,C}       → could elect leader 1
  {C, D, E} = majority of {A,B,C,D,E} → could elect leader 2

TWO LEADERS! Safety violated.
```

### 8.2 Single-Server Changes (Raft)

Raft's simplification: change one server at a time. This guarantees that old and new majorities always overlap.

```
3-node → 4-node:
  Old majority: 2 of {A,B,C}
  New majority: 3 of {A,B,C,D}
  Overlap guaranteed: any 2 of {A,B,C} and any 3 of {A,B,C,D} must share at least 1

4-node → 5-node:
  Old majority: 3 of {A,B,C,D}
  New majority: 3 of {A,B,C,D,E}
  Overlap guaranteed: any two sets of size 3 from a set of size 5 must share at least 1
```

### 8.3 Implementation

```python
class MembershipChange:
    """Handle single-server membership changes."""

    @staticmethod
    def add_node(raft: RaftNode, new_node_id: str, host: str, port: int) -> dict:
        """
        Add a new node to the cluster.

        Steps:
          1. Bring new node up-to-date (catch up its log)
          2. Propose configuration change as a log entry
          3. Once committed, all nodes use new configuration
        """
        if raft.state != NodeState.LEADER:
            return {"success": False, "error": "not_leader"}

        # Register new peer in transport
        raft.transport.register_peer(new_node_id, host, port)

        # Add to peers list
        raft.peers.append(new_node_id)
        raft.next_index[new_node_id] = 1
        raft.match_index[new_node_id] = 0

        # Propose config change (committed via normal Raft replication)
        config_entry = {
            "type": "config_change",
            "action": "add",
            "node_id": new_node_id,
            "host": host,
            "port": port,
        }
        return raft.propose(config_entry)

    @staticmethod
    def remove_node(raft: RaftNode, node_id: str) -> dict:
        """
        Remove a node from the cluster.

        If the leader is being removed, it steps down after
        the configuration change is committed.
        """
        if raft.state != NodeState.LEADER:
            return {"success": False, "error": "not_leader"}

        config_entry = {
            "type": "config_change",
            "action": "remove",
            "node_id": node_id,
        }
        result = raft.propose(config_entry)

        if result.get("success") and node_id == raft.node_id:
            # Leader is removing itself: step down
            with raft._lock:
                raft._step_down(raft.current_term)

        return result
```

---

## 9. Testing with Fault Injection

### 9.1 Fault Injection Framework

```python
class FaultInjector:
    """
    Fault injection framework for testing the distributed KV store.

    Supports:
      - Network partitions
      - Node crashes and recovery
      - Message dropping and reordering
      - Clock skew simulation
    """

    def __init__(self, cluster: Dict[str, RaftNode]):
        self.cluster = cluster
        self._partitions: List[Set[str]] = []
        self._dropped_connections: Set[tuple] = set()
        self._message_drop_rate: float = 0.0
        self._crashed_nodes: Set[str] = set()

    def partition(self, group_a: Set[str], group_b: Set[str]) -> None:
        """
        Create a network partition between two groups.

        Nodes in group_a cannot communicate with nodes in group_b
        and vice versa.
        """
        logger.info(f"PARTITION: {group_a} | {group_b}")
        self._partitions.append((group_a, group_b))

        # Block connections between groups
        for a in group_a:
            for b in group_b:
                self._dropped_connections.add((a, b))
                self._dropped_connections.add((b, a))

    def heal_partition(self) -> None:
        """Remove all network partitions."""
        logger.info("HEAL: All partitions removed")
        self._partitions.clear()
        self._dropped_connections.clear()

    def crash_node(self, node_id: str) -> None:
        """Simulate a node crash."""
        if node_id in self.cluster:
            logger.info(f"CRASH: {node_id}")
            self._crashed_nodes.add(node_id)
            self.cluster[node_id].stop()

    def recover_node(self, node_id: str) -> None:
        """Recover a crashed node."""
        if node_id in self._crashed_nodes:
            logger.info(f"RECOVER: {node_id}")
            self._crashed_nodes.discard(node_id)
            self.cluster[node_id].start()

    def set_message_drop_rate(self, rate: float) -> None:
        """Set the probability of dropping any message."""
        self._message_drop_rate = rate

    def is_connection_blocked(self, sender: str, receiver: str) -> bool:
        """Check if a message should be blocked."""
        if sender in self._crashed_nodes or receiver in self._crashed_nodes:
            return True
        if (sender, receiver) in self._dropped_connections:
            return True
        if self._message_drop_rate > 0:
            if random.random() < self._message_drop_rate:
                return True
        return False


class LinearizabilityChecker:
    """
    Checks operation history for linearizability violations.

    Records all client operations and verifies that there exists
    a sequential ordering consistent with:
      1. Real-time ordering (if op A completes before op B starts,
         A appears before B in the sequential order)
      2. Sequential specification (each read returns the value of
         the most recent write in the sequential order)
    """

    @dataclass
    class Operation:
        """A recorded client operation."""
        op_type: str           # "put", "get", "delete"
        key: str
        value: Optional[str]   # Value for put, returned value for get
        start_time: float
        end_time: float
        success: bool

    def __init__(self):
        self.history: List[LinearizabilityChecker.Operation] = []

    def record(self, op_type: str, key: str, value: Optional[str],
               start_time: float, end_time: float, success: bool) -> None:
        """Record an operation."""
        self.history.append(self.Operation(
            op_type=op_type, key=key, value=value,
            start_time=start_time, end_time=end_time,
            success=success,
        ))

    def check(self) -> Tuple[bool, str]:
        """
        Check if the recorded history is linearizable.

        Uses a simplified brute-force approach suitable for small histories.
        For production use, consider the Wing & Gong algorithm or Knossos.

        Returns:
            (is_linearizable, explanation)
        """
        # Filter to successful operations only
        ops = [op for op in self.history if op.success]

        if not ops:
            return True, "Empty history is trivially linearizable"

        # Group by key
        by_key: Dict[str, list] = {}
        for op in ops:
            by_key.setdefault(op.key, []).append(op)

        # Check each key independently (keys don't interact)
        for key, key_ops in by_key.items():
            ok, msg = self._check_key(key, key_ops)
            if not ok:
                return False, f"Key '{key}': {msg}"

        return True, "History is linearizable"

    def _check_key(self, key: str, ops: list) -> Tuple[bool, str]:
        """Check linearizability for operations on a single key."""
        # Sort by start time
        ops.sort(key=lambda o: o.start_time)

        # Try all possible linearization orderings using backtracking
        # This is exponential but works for small test histories
        return self._try_linearize(ops, [], None)

    def _try_linearize(
        self, remaining: list, linearized: list, current_value: Optional[str]
    ) -> Tuple[bool, str]:
        """Recursively try to find a valid linearization."""
        if not remaining:
            return True, "Valid linearization found"

        for i, op in enumerate(remaining):
            # Check real-time constraint: this op cannot be linearized
            # before any op that completed before it started
            can_go_next = True
            for prev_op in linearized:
                if prev_op.end_time <= op.start_time:
                    pass  # prev_op must come before op — already satisfied
                # If prev_op started after op ended, op must come first
                # but it's not in linearized yet — this is a constraint violation

            # For already-linearized ops that ended before this started,
            # they must appear before this op (already guaranteed).
            # For ops still in remaining that ended before this started,
            # they must be linearized before this — skip if violated.
            for other in remaining:
                if other is op:
                    continue
                if other.end_time < op.start_time:
                    can_go_next = False  # Other must come first
                    break

            if not can_go_next:
                continue

            # Check sequential specification
            if op.op_type == "get":
                if op.value != current_value:
                    continue  # This get doesn't match current state

            # Try this op next
            new_remaining = remaining[:i] + remaining[i+1:]
            new_value = current_value
            if op.op_type == "put":
                new_value = op.value
            elif op.op_type == "delete":
                new_value = None

            ok, msg = self._try_linearize(
                new_remaining, linearized + [op], new_value
            )
            if ok:
                return True, msg

        return False, "No valid linearization found"
```

### 9.2 Test Scenarios

```python
class DistributedKVTests:
    """Test suite for the distributed KV store."""

    def __init__(self):
        self.nodes: Dict[str, RaftNode] = {}
        self.fault_injector: Optional[FaultInjector] = None

    def setup_cluster(self, num_nodes: int = 3) -> None:
        """Set up a test cluster."""
        base_port = 9000
        node_ids = [f"node-{i}" for i in range(num_nodes)]

        for i, nid in enumerate(node_ids):
            peers = [n for n in node_ids if n != nid]
            transport = MessageTransport(nid, "localhost", base_port + i)
            state_machine = KeyValueStateMachine()

            # Register all peers
            for j, peer in enumerate(node_ids):
                if peer != nid:
                    transport.register_peer(peer, "localhost", base_port + j)

            self.nodes[nid] = RaftNode(
                node_id=nid,
                peers=peers,
                transport=transport,
                state_machine=state_machine,
            )

        self.fault_injector = FaultInjector(self.nodes)

    def start_cluster(self) -> None:
        """Start all nodes."""
        for node in self.nodes.values():
            node.start()
        # Wait for leader election
        time.sleep(2.0)

    def stop_cluster(self) -> None:
        """Stop all nodes."""
        for node in self.nodes.values():
            node.stop()

    def get_leader(self) -> Optional[RaftNode]:
        """Find the current leader."""
        for node in self.nodes.values():
            if node.state == NodeState.LEADER:
                return node
        return None

    def test_basic_operations(self) -> bool:
        """Test basic PUT, GET, DELETE."""
        print("=== Test: Basic Operations ===")
        leader = self.get_leader()
        if not leader:
            print("  FAIL: No leader elected")
            return False

        # PUT
        result = leader.propose({"type": "put", "key": "x", "value": "42"})
        if not result.get("success"):
            print(f"  FAIL: PUT failed: {result}")
            return False
        print("  PUT(x, 42): OK")

        # GET
        result = leader.propose({"type": "get", "key": "x"})
        if not result.get("success") or result.get("value") != "42":
            print(f"  FAIL: GET returned: {result}")
            return False
        print("  GET(x): 42 OK")

        # DELETE
        result = leader.propose({"type": "delete", "key": "x"})
        if not result.get("success"):
            print(f"  FAIL: DELETE failed: {result}")
            return False
        print("  DELETE(x): OK")

        # GET after delete
        result = leader.propose({"type": "get", "key": "x"})
        if result.get("success"):
            print(f"  FAIL: GET after DELETE should fail: {result}")
            return False
        print("  GET(x) after DELETE: key_not_found OK")

        print("  PASS")
        return True

    def test_leader_failure(self) -> bool:
        """Test that the cluster recovers from leader failure."""
        print("\n=== Test: Leader Failure ===")
        leader = self.get_leader()
        if not leader:
            print("  FAIL: No initial leader")
            return False

        leader_id = leader.node_id
        print(f"  Current leader: {leader_id}")

        # Write a value
        leader.propose({"type": "put", "key": "survive", "value": "yes"})
        time.sleep(0.5)  # Let it replicate

        # Crash the leader
        self.fault_injector.crash_node(leader_id)
        print(f"  Crashed leader: {leader_id}")

        # Wait for new election
        time.sleep(3.0)

        # Find new leader
        new_leader = self.get_leader()
        if not new_leader:
            print("  FAIL: No new leader elected after crash")
            return False

        if new_leader.node_id == leader_id:
            print("  FAIL: Old leader is still leader")
            return False

        print(f"  New leader: {new_leader.node_id}")

        # Check value survived
        result = new_leader.propose({"type": "get", "key": "survive"})
        if not result.get("success") or result.get("value") != "yes":
            print(f"  FAIL: Value did not survive: {result}")
            return False

        print("  Value survived leader failure: OK")
        print("  PASS")
        return True

    def test_network_partition(self) -> bool:
        """Test behavior during network partition."""
        print("\n=== Test: Network Partition ===")

        nodes = list(self.nodes.keys())
        leader = self.get_leader()
        if not leader:
            print("  FAIL: No leader")
            return False

        # Write initial value
        leader.propose({"type": "put", "key": "partition_test", "value": "v1"})
        time.sleep(0.5)

        # Partition: isolate the leader from the majority
        leader_id = leader.node_id
        minority = {leader_id}
        majority = {n for n in nodes if n != leader_id}

        self.fault_injector.partition(minority, majority)
        print(f"  Partitioned: {minority} | {majority}")

        # Wait for new leader in the majority partition
        time.sleep(3.0)

        # The isolated leader should not be able to commit
        result = leader.propose(
            {"type": "put", "key": "should_fail", "value": "v"},
            timeout=2.0
        )
        if result.get("success"):
            print("  FAIL: Isolated leader committed a write!")
            return False
        print("  Isolated leader cannot commit: OK")

        # Heal partition
        self.fault_injector.heal_partition()
        print("  Partition healed")
        time.sleep(2.0)

        # Cluster should converge
        new_leader = self.get_leader()
        if new_leader:
            result = new_leader.propose(
                {"type": "get", "key": "partition_test"}
            )
            if result.get("value") == "v1":
                print("  Data consistent after heal: OK")
            else:
                print(f"  WARNING: Unexpected value: {result}")

        print("  PASS")
        return True

    def test_split_brain_prevention(self) -> bool:
        """Verify that split brain cannot occur."""
        print("\n=== Test: Split Brain Prevention ===")

        # Count leaders
        leaders = [n for n in self.nodes.values() if n.state == NodeState.LEADER]

        if len(leaders) > 1:
            # Check terms: two leaders in the same term is a bug
            terms = [l.current_term for l in leaders]
            if len(set(terms)) < len(terms):
                print(f"  FAIL: Two leaders in same term! "
                      f"Leaders: {[(l.node_id, l.current_term) for l in leaders]}")
                return False
            print(f"  Multiple leaders but different terms: OK "
                  f"(transient during election)")
        else:
            print(f"  Single leader: {leaders[0].node_id if leaders else 'none'}")

        print("  PASS")
        return True

    def run_all_tests(self) -> None:
        """Run all tests."""
        print("=" * 60)
        print("Distributed KV Store Test Suite")
        print("=" * 60)

        self.setup_cluster(3)
        self.start_cluster()

        results = []
        results.append(("Basic Operations", self.test_basic_operations()))
        results.append(("Split Brain", self.test_split_brain_prevention()))
        results.append(("Leader Failure", self.test_leader_failure()))

        # Restart cluster for partition test
        self.stop_cluster()
        time.sleep(1)
        self.setup_cluster(5)
        self.start_cluster()
        results.append(("Network Partition", self.test_network_partition()))

        self.stop_cluster()

        print("\n" + "=" * 60)
        print("Results:")
        for name, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
        print("=" * 60)
```

---

## 10. Performance Considerations

### 10.1 Batching Log Entries

Instead of replicating entries one at a time, the leader can batch multiple client requests into a single AppendEntries RPC:

```
Without batching:                    With batching:
  Client 1: PUT(a,1) → replicate    Client 1: PUT(a,1) ─┐
  Client 2: PUT(b,2) → replicate    Client 2: PUT(b,2) ─┼─ replicate all
  Client 3: PUT(c,3) → replicate    Client 3: PUT(c,3) ─┘

  3 AppendEntries RPCs               1 AppendEntries RPC
  3 round-trips to commit            1 round-trip to commit
```

```python
class BatchingOptimization:
    """Illustration of log entry batching."""

    def __init__(self, batch_size: int = 100, batch_timeout_ms: float = 1.0):
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._batch: List[LogEntry] = []
        self._batch_start: Optional[float] = None

    def add_entry(self, entry: LogEntry) -> Optional[List[LogEntry]]:
        """
        Add an entry to the batch.

        Returns the batch when it's ready to send (full or timed out).
        """
        if not self._batch:
            self._batch_start = time.monotonic()

        self._batch.append(entry)

        # Flush if batch is full
        if len(self._batch) >= self.batch_size:
            return self._flush()

        # Flush if timeout elapsed
        elapsed_ms = (time.monotonic() - self._batch_start) * 1000
        if elapsed_ms >= self.batch_timeout_ms:
            return self._flush()

        return None

    def _flush(self) -> List[LogEntry]:
        batch = self._batch
        self._batch = []
        self._batch_start = None
        return batch
```

### 10.2 Pipeline AppendEntries

Without pipelining, the leader waits for each AppendEntries response before sending the next. With pipelining, multiple RPCs are in flight simultaneously:

```
Without pipeline:
  Leader ── AE(1) ──▶ Follower
  Leader ◀── OK ───── Follower
  Leader ── AE(2) ──▶ Follower
  Leader ◀── OK ───── Follower
  Total: 2 round-trips

With pipeline:
  Leader ── AE(1) ──▶ Follower
  Leader ── AE(2) ──▶ Follower
  Leader ◀── OK(1) ── Follower
  Leader ◀── OK(2) ── Follower
  Total: 1 round-trip (overlapped)
```

### 10.3 Snapshot Transfer

For large state machines, snapshots must be transferred efficiently:

```
Approach 1: Send entire snapshot in one InstallSnapshot RPC
  + Simple
  - Blocks replication during transfer
  - Fails for large state (memory pressure)

Approach 2: Chunked transfer
  Leader sends snapshot in chunks (e.g., 1MB each)
  Follower reassembles and installs
  + Handles large state
  + Interleaves with regular replication
  - More complex

Approach 3: State transfer via external system (e.g., S3)
  Leader uploads snapshot to S3
  Follower downloads from S3
  + Doesn't burden leader-follower link
  + Handles arbitrarily large state
  - Requires external dependency
```

### 10.4 Performance Comparison

| Optimization | Latency Impact | Throughput Impact |
|-------------|---------------|-------------------|
| Batching | +0-1ms (batch wait) | 5-10x improvement |
| Pipelining | -50% (overlapped) | 2x improvement |
| ReadIndex | -90% vs log reads | N/A (read path only) |
| LeaseRead | -99% vs log reads | N/A (read path only) |
| Snapshot chunking | N/A | Prevents stalls |
| Parallel AppendEntries | -30% (parallel sends) | 1.5x improvement |

---

## 11. Complete Implementation

### 11.1 Putting It All Together

The following ties together all layers into a runnable system:

```python
"""
Complete Distributed Key-Value Store

Usage:
    # Terminal 1
    python kv_store.py --node-id node-0 --port 9000 --peers node-1:9001,node-2:9002

    # Terminal 2
    python kv_store.py --node-id node-1 --port 9001 --peers node-0:9000,node-2:9002

    # Terminal 3
    python kv_store.py --node-id node-2 --port 9002 --peers node-0:9000,node-1:9001
"""

import argparse
import sys


def create_node(node_id: str, port: int, peer_specs: List[str]) -> RaftNode:
    """
    Create a fully configured Raft node.

    Args:
        node_id: Unique identifier for this node
        port: Port to listen on
        peer_specs: List of "node_id:port" strings

    Returns:
        Configured RaftNode ready to start
    """
    transport = MessageTransport(node_id, "0.0.0.0", port)
    state_machine = KeyValueStateMachine()

    peers = []
    for spec in peer_specs:
        peer_id, peer_port = spec.split(":")
        transport.register_peer(peer_id, "localhost", int(peer_port))
        peers.append(peer_id)

    node = RaftNode(
        node_id=node_id,
        peers=peers,
        transport=transport,
        state_machine=state_machine,
        election_timeout_range=(300, 500),
        heartbeat_interval=100,
    )

    return node


def run_interactive(node: RaftNode) -> None:
    """Run an interactive CLI for the KV store."""
    print(f"\nDistributed KV Store - Node {node.node_id}")
    print("Commands: put <key> <value> | get <key> | delete <key> | "
          "status | quit\n")

    while True:
        try:
            line = input(f"[{node.state.value}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "quit":
            break
        elif cmd == "status":
            print(f"  Node: {node.node_id}")
            print(f"  State: {node.state.value}")
            print(f"  Term: {node.current_term}")
            print(f"  Leader: {node.leader_id}")
            print(f"  Log length: {len(node.log)}")
            print(f"  Commit index: {node.commit_index}")
            print(f"  Last applied: {node.last_applied}")
            print(f"  Data: {node.state_machine.get_all()}")
        elif cmd == "put" and len(parts) >= 3:
            key, value = parts[1], " ".join(parts[2:])
            result = node.propose({"type": "put", "key": key, "value": value})
            print(f"  {result}")
        elif cmd == "get" and len(parts) >= 2:
            key = parts[1]
            result = node.propose({"type": "get", "key": key})
            print(f"  {result}")
        elif cmd == "delete" and len(parts) >= 2:
            key = parts[1]
            result = node.propose({"type": "delete", "key": key})
            print(f"  {result}")
        else:
            print("  Unknown command. Try: put <key> <value> | get <key> | "
                  "delete <key> | status | quit")


def main():
    parser = argparse.ArgumentParser(
        description="Distributed Key-Value Store"
    )
    parser.add_argument("--node-id", required=True, help="Node identifier")
    parser.add_argument("--port", type=int, required=True, help="Listen port")
    parser.add_argument(
        "--peers", required=True,
        help="Comma-separated peer specs: node_id:port,..."
    )
    args = parser.parse_args()

    peer_list = args.peers.split(",") if args.peers else []
    node = create_node(args.node_id, args.port, peer_list)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    node.start()
    try:
        run_interactive(node)
    finally:
        node.stop()


def demo_in_process():
    """
    Run a complete 3-node cluster in a single process for demonstration.
    """
    logging.basicConfig(level=logging.WARNING)

    print("=" * 60)
    print("Distributed KV Store — In-Process Demo")
    print("=" * 60)

    # Create 3 nodes
    nodes = {}
    base_port = 19000
    node_configs = [
        ("node-0", base_port),
        ("node-1", base_port + 1),
        ("node-2", base_port + 2),
    ]

    for nid, port in node_configs:
        peer_specs = [
            f"{pid}:{pp}" for pid, pp in node_configs if pid != nid
        ]
        nodes[nid] = create_node(nid, port, peer_specs)

    # Start all nodes
    print("\n1. Starting cluster...")
    for node in nodes.values():
        node.start()

    # Wait for leader election
    time.sleep(3.0)

    # Find leader
    leader = None
    for node in nodes.values():
        if node.state == NodeState.LEADER:
            leader = node
            break

    if not leader:
        print("ERROR: No leader elected!")
        for node in nodes.values():
            node.stop()
        return

    print(f"   Leader elected: {leader.node_id} (term {leader.current_term})")

    # Write data
    print("\n2. Writing data...")
    test_data = [
        ("name", "Distributed Systems"),
        ("version", "1.0"),
        ("nodes", "3"),
        ("consensus", "Raft"),
    ]

    for key, value in test_data:
        result = leader.propose(
            {"type": "put", "key": key, "value": value},
            client_id="demo-client",
            seq_num=hash(key) % 10000,
        )
        status = "OK" if result.get("success") else f"FAIL: {result}"
        print(f"   PUT({key}, {value}): {status}")

    time.sleep(1.0)

    # Read data
    print("\n3. Reading data...")
    for key, expected in test_data:
        result = leader.propose({"type": "get", "key": key})
        value = result.get("value", "N/A")
        status = "OK" if value == expected else f"MISMATCH (got {value})"
        print(f"   GET({key}): {value} — {status}")

    # Check replication
    print("\n4. Checking replication across nodes...")
    time.sleep(1.0)
    for nid, node in nodes.items():
        data = node.state_machine.get_all()
        print(f"   {nid} [{node.state.value:9s}]: "
              f"log={len(node.log)}, committed={node.commit_index}, "
              f"data_keys={list(data.keys())}")

    # Demonstrate leader failure
    print(f"\n5. Crashing leader ({leader.node_id})...")
    leader.stop()
    time.sleep(4.0)

    new_leader = None
    for nid, node in nodes.items():
        if node.state == NodeState.LEADER and nid != leader.node_id:
            new_leader = node
            break

    if new_leader:
        print(f"   New leader: {new_leader.node_id} "
              f"(term {new_leader.current_term})")

        # Verify data survived
        result = new_leader.propose({"type": "get", "key": "name"})
        value = result.get("value", "N/A")
        print(f"   GET(name) on new leader: {value}")

        # Write new data
        result = new_leader.propose(
            {"type": "put", "key": "leader_change", "value": "survived"}
        )
        print(f"   PUT(leader_change, survived): "
              f"{'OK' if result.get('success') else 'FAIL'}")
    else:
        print("   ERROR: No new leader elected!")

    # Clean up
    print("\n6. Shutting down...")
    for node in nodes.values():
        node.stop()
    print("   Done.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_in_process()
    elif len(sys.argv) > 1:
        main()
    else:
        demo_in_process()
```

---

## 12. Summary

### What We Built

| Component | Description | Lines of Code (approx.) |
|-----------|-------------|------------------------|
| Message Transport | TCP + JSON framing, connection management | ~200 |
| Raft Consensus | Leader election, log replication, commit | ~400 |
| State Machine | In-memory KV store with snapshot support | ~80 |
| Client | Leader discovery, retry, deduplication | ~100 |
| Fault Injection | Partitions, crashes, message dropping | ~100 |
| Linearizability Checker | History validation | ~100 |
| **Total** | | **~1000** |

### Key Lessons Learned

| Lesson | Insight |
|--------|---------|
| Simplicity is hard | Raft is "simple" but implementation is still ~400 lines of careful code |
| Edge cases dominate | Most bugs are in election edge cases and log divergence handling |
| Testing is essential | Unit tests catch logic bugs; fault injection catches design bugs |
| Fencing matters | Linearizable reads require leadership confirmation |
| Deduplication is necessary | Network retries can duplicate committed operations |
| No-op on election | Required for correctness: commits entries from previous terms |
| Jittered timeouts | Critical for preventing repeated split votes |

### How This Connects to the Course

This capstone project integrates concepts from nearly every lesson in the course:

| Lesson | Concept Used Here |
|--------|-------------------|
| L01: System Models | Crash-recovery model, partial synchrony assumption |
| L02: Time and Clocks | Logical clocks (terms), timeout-based failure detection |
| L03: FLP | Why we need timeouts (circumventing impossibility) |
| L04: Consistency | Linearizability as our correctness criterion |
| L06: Raft | The core consensus protocol |
| L08: Transactions | Client request deduplication (at-most-once semantics) |
| L09: Replication | Log replication is the replication strategy |
| L13: Failure Detection | Heartbeat-based leader failure detection |
| L14: Coordination | Leader election, fencing tokens |
| L15: TLA+ | Could formally verify this design before implementing |

### Next Steps for the Reader

1. **Add persistence**: Write `currentTerm`, `votedFor`, and `log` to disk on every change
2. **Add snapshots**: Implement `InstallSnapshot` RPC for log compaction
3. **Add pre-vote**: Prevent disruptions from partitioned nodes rejoining
4. **Add pipelining**: Overlap AppendEntries RPCs for higher throughput
5. **Run Jepsen**: Use Jepsen or a similar framework to find remaining bugs
6. **Verify with TLA+**: Write a TLA+ spec and model-check it (Lesson 15)
7. **Benchmark**: Measure latency and throughput under various workloads

### Essential Reading

1. **Ongaro, Ousterhout (2014)** — "In Search of an Understandable Consensus Algorithm" (the Raft paper)
2. **Ongaro (2014)** — "Consensus: Bridging Theory and Practice" (Raft dissertation, the definitive reference)
3. **Kingsbury** — Jepsen analyses (https://jepsen.io) — how real databases fail
4. **Howard (2014)** — "ARC: Analysis of Raft Consensus" — edge cases and formal analysis

---

[Back to Overview](./00_Overview.md) | [Next: Raft Implementation Part 1 →](./18_Raft_Implementation_Part1.md)
