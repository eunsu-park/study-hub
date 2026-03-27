"""
TCP State Machine

Demonstrates:
- TCP connection states (CLOSED → ESTABLISHED → CLOSED)
- Three-way handshake (SYN → SYN+ACK → ACK)
- Four-way termination (FIN → ACK → FIN → ACK)
- Sequence number tracking
- Connection timeout handling

Theory:
- TCP is connection-oriented: state machine manages lifecycle.
- 3-way handshake: client SYN, server SYN+ACK, client ACK.
  Establishes sequence numbers for both directions.
- Connection teardown: each side sends FIN + receives ACK.
  TIME_WAIT prevents old packets from being misinterpreted.
- States: CLOSED, LISTEN, SYN_SENT, SYN_RCVD, ESTABLISHED,
  FIN_WAIT_1, FIN_WAIT_2, CLOSE_WAIT, LAST_ACK, TIME_WAIT.

Adapted from Networking Lesson 10.
"""

from enum import Enum
from dataclasses import dataclass, field


class TCPState(Enum):
    CLOSED = "CLOSED"
    LISTEN = "LISTEN"
    SYN_SENT = "SYN_SENT"
    SYN_RCVD = "SYN_RCVD"
    ESTABLISHED = "ESTABLISHED"
    FIN_WAIT_1 = "FIN_WAIT_1"
    FIN_WAIT_2 = "FIN_WAIT_2"
    CLOSE_WAIT = "CLOSE_WAIT"
    LAST_ACK = "LAST_ACK"
    TIME_WAIT = "TIME_WAIT"
    CLOSING = "CLOSING"


@dataclass
class TCPSegment:
    src_port: int
    dst_port: int
    seq: int
    ack: int
    flags: set[str]  # SYN, ACK, FIN, RST
    data: bytes = b""

    def __str__(self) -> str:
        flags_str = "+".join(sorted(self.flags))
        return (f"[{self.src_port}→{self.dst_port}] "
                f"seq={self.seq} ack={self.ack} {flags_str}")


# Why: Modeling each endpoint separately (not as a pair) reflects how TCP
# actually works — each side independently tracks its own state, sequence
# numbers, and transitions. Connection state is not shared; it emerges from
# the exchange of segments between two autonomous state machines.
class TCPEndpoint:
    """One side of a TCP connection."""

    def __init__(self, name: str, port: int):
        self.name = name
        self.port = port
        self.state = TCPState.CLOSED
        self.seq = 0        # Our next sequence number
        self.ack = 0        # Next expected from peer
        self.initial_seq = 0
        self.log: list[str] = []

    def _transition(self, new_state: TCPState) -> None:
        old = self.state.value
        self.state = new_state
        self.log.append(f"  {self.name}: {old} → {new_state.value}")

    def send_syn(self, dst_port: int, isn: int = 1000) -> TCPSegment:
        """Active open: send SYN."""
        self.initial_seq = isn
        self.seq = isn
        self._transition(TCPState.SYN_SENT)
        seg = TCPSegment(self.port, dst_port, self.seq, 0, {"SYN"})
        # Why: SYN and FIN each consume one sequence number even though they
        # carry no data. This ensures the peer can ACK them unambiguously.
        # Without this, there would be no way to distinguish "SYN received"
        # from "SYN lost" based on the acknowledgment number alone.
        self.seq += 1
        self.log.append(f"  {self.name} → {seg}")
        return seg

    def listen(self) -> None:
        """Passive open: start listening."""
        self._transition(TCPState.LISTEN)

    def receive(self, seg: TCPSegment, isn: int = 2000) -> TCPSegment | None:
        """Process received segment, return response if needed."""
        self.log.append(f"  {self.name} ← {seg}")

        if self.state == TCPState.LISTEN and "SYN" in seg.flags:
            # Why: The server's SYN+ACK serves double duty — it acknowledges
            # the client's SYN and simultaneously sends its own SYN (with its
            # own ISN). This pipelining is why it's a 3-way handshake, not 4.
            self.initial_seq = isn
            self.seq = isn
            self.ack = seg.seq + 1
            self._transition(TCPState.SYN_RCVD)
            resp = TCPSegment(self.port, seg.src_port, self.seq,
                              self.ack, {"SYN", "ACK"})
            self.seq += 1
            self.log.append(f"  {self.name} → {resp}")
            return resp

        if self.state == TCPState.SYN_SENT and seg.flags == {"SYN", "ACK"}:
            # Received SYN+ACK: send ACK, connection established
            self.ack = seg.seq + 1
            self._transition(TCPState.ESTABLISHED)
            resp = TCPSegment(self.port, seg.src_port, self.seq,
                              self.ack, {"ACK"})
            self.log.append(f"  {self.name} → {resp}")
            return resp

        if self.state == TCPState.SYN_RCVD and "ACK" in seg.flags:
            # Received ACK for our SYN+ACK: established
            self._transition(TCPState.ESTABLISHED)
            return None

        if self.state == TCPState.ESTABLISHED and "FIN" in seg.flags:
            # Why: TCP close is half-duplex — receiving FIN means the peer
            # has no more data to send, but we can still send. CLOSE_WAIT
            # lets the application finish transmitting before sending its own FIN.
            self.ack = seg.seq + 1
            self._transition(TCPState.CLOSE_WAIT)
            resp = TCPSegment(self.port, seg.src_port, self.seq,
                              self.ack, {"ACK"})
            self.log.append(f"  {self.name} → {resp}")
            return resp

        if self.state == TCPState.FIN_WAIT_1 and "ACK" in seg.flags:
            if "FIN" in seg.flags:
                # Simultaneous close
                self.ack = seg.seq + 1
                self._transition(TCPState.TIME_WAIT)
                resp = TCPSegment(self.port, seg.src_port, self.seq,
                                  self.ack, {"ACK"})
                self.log.append(f"  {self.name} → {resp}")
                return resp
            self._transition(TCPState.FIN_WAIT_2)
            return None

        if self.state == TCPState.FIN_WAIT_2 and "FIN" in seg.flags:
            self.ack = seg.seq + 1
            self._transition(TCPState.TIME_WAIT)
            resp = TCPSegment(self.port, seg.src_port, self.seq,
                              self.ack, {"ACK"})
            self.log.append(f"  {self.name} → {resp}")
            return resp

        if self.state == TCPState.LAST_ACK and "ACK" in seg.flags:
            self._transition(TCPState.CLOSED)
            return None

        return None

    def send_fin(self, dst_port: int) -> TCPSegment:
        """Initiate connection close."""
        seg = TCPSegment(self.port, dst_port, self.seq, self.ack, {"FIN", "ACK"})
        self.seq += 1  # FIN consumes a sequence number
        if self.state == TCPState.ESTABLISHED:
            self._transition(TCPState.FIN_WAIT_1)
        elif self.state == TCPState.CLOSE_WAIT:
            self._transition(TCPState.LAST_ACK)
        self.log.append(f"  {self.name} → {seg}")
        return seg

    # Why: TIME_WAIT lasts 2×MSL (Maximum Segment Lifetime, typically 60s)
    # to ensure two things: (1) the final ACK reaches the peer (if lost, the
    # peer will retransmit FIN), and (2) old duplicate segments from this
    # connection expire before the same port pair can be reused.
    def timeout_time_wait(self) -> None:
        """TIME_WAIT timeout (2×MSL)."""
        if self.state == TCPState.TIME_WAIT:
            self._transition(TCPState.CLOSED)
            self.log.append(f"  {self.name}: TIME_WAIT expired (2×MSL)")


# ── Demos ──────────────────────────────────────────────────────────────

def demo_three_way_handshake():
    print("=" * 60)
    print("TCP THREE-WAY HANDSHAKE")
    print("=" * 60)

    client = TCPEndpoint("Client", 12345)
    server = TCPEndpoint("Server", 80)

    # Server listens
    server.listen()

    # Client → SYN
    syn = client.send_syn(80, isn=1000)

    # Server ← SYN, → SYN+ACK
    syn_ack = server.receive(syn, isn=2000)

    # Client ← SYN+ACK, → ACK
    ack = client.receive(syn_ack)

    # Server ← ACK
    server.receive(ack)

    print(f"\n  Three-way handshake:")
    for entry in client.log + server.log:
        print(entry)

    print(f"\n  Final states:")
    print(f"    Client: {client.state.value}")
    print(f"    Server: {server.state.value}")
    print(f"\n  Sequence numbers:")
    print(f"    Client: seq={client.seq}, ack={client.ack}")
    print(f"    Server: seq={server.seq}, ack={server.ack}")


def demo_connection_teardown():
    print("\n" + "=" * 60)
    print("TCP FOUR-WAY TEARDOWN")
    print("=" * 60)

    # Set up established connection
    client = TCPEndpoint("Client", 12345)
    server = TCPEndpoint("Server", 80)
    server.listen()
    syn = client.send_syn(80, isn=1000)
    syn_ack = server.receive(syn, isn=2000)
    ack = client.receive(syn_ack)
    server.receive(ack)
    client.log.clear()
    server.log.clear()

    print(f"\n  Starting from ESTABLISHED state")

    # Client initiates close
    fin1 = client.send_fin(80)

    # Server receives FIN, sends ACK
    ack1 = server.receive(fin1)

    # Client receives ACK
    client.receive(ack1)

    # Server sends its FIN
    fin2 = server.send_fin(12345)

    # Client receives FIN, sends ACK
    ack2 = client.receive(fin2)

    # Server receives ACK
    server.receive(ack2)

    # Client TIME_WAIT timeout
    client.timeout_time_wait()

    print(f"\n  Four-way teardown:")
    all_logs = []
    for entry in client.log:
        all_logs.append(entry)
    for entry in server.log:
        all_logs.append(entry)
    for entry in sorted(set(all_logs)):
        print(entry)

    print(f"\n  Final states:")
    print(f"    Client: {client.state.value}")
    print(f"    Server: {server.state.value}")


def demo_state_diagram():
    print("\n" + "=" * 60)
    print("TCP STATE TRANSITIONS SUMMARY")
    print("=" * 60)

    transitions = [
        ("CLOSED", "LISTEN", "passive open"),
        ("CLOSED", "SYN_SENT", "active open, send SYN"),
        ("LISTEN", "SYN_RCVD", "recv SYN, send SYN+ACK"),
        ("SYN_SENT", "ESTABLISHED", "recv SYN+ACK, send ACK"),
        ("SYN_RCVD", "ESTABLISHED", "recv ACK"),
        ("ESTABLISHED", "FIN_WAIT_1", "close, send FIN"),
        ("ESTABLISHED", "CLOSE_WAIT", "recv FIN, send ACK"),
        ("FIN_WAIT_1", "FIN_WAIT_2", "recv ACK"),
        ("FIN_WAIT_2", "TIME_WAIT", "recv FIN, send ACK"),
        ("CLOSE_WAIT", "LAST_ACK", "close, send FIN"),
        ("LAST_ACK", "CLOSED", "recv ACK"),
        ("TIME_WAIT", "CLOSED", "timeout (2×MSL)"),
    ]

    print(f"\n  {'From':<15} {'To':<15} {'Trigger'}")
    print(f"  {'-'*15} {'-'*15} {'-'*30}")
    for from_st, to_st, trigger in transitions:
        print(f"  {from_st:<15} {to_st:<15} {trigger}")

    print(f"""
  Connection Establishment:
    Client              Server
      |  ── SYN ──→       |
      |  ←─ SYN+ACK ──   |
      |  ── ACK ──→       |
      |   ESTABLISHED     |

  Connection Teardown:
    Client              Server
      |  ── FIN ──→       |
      |  ←─ ACK ──        |
      |  ←─ FIN ──        |
      |  ── ACK ──→       |
      | (TIME_WAIT)       |""")


if __name__ == "__main__":
    demo_three_way_handshake()
    demo_connection_teardown()
    demo_state_diagram()
