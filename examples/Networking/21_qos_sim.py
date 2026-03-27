"""
Example: Quality of Service (QoS) Simulator
Topic: Networking – Lesson 21

Demonstrates core QoS mechanisms:
  1. Token bucket traffic shaper
  2. Weighted Fair Queuing (WFQ)
  3. DSCP classification and Per-Hop Behavior (PHB)
  4. Comparison: FIFO vs priority queuing vs WFQ

Run: python 21_qos_sim.py
"""

import random
from collections import deque


# ============================================================
# Traffic Models
# ============================================================
class Packet:
    """Represents a network packet with QoS metadata."""
    _id_counter = 0

    def __init__(self, src: str, dst: str, size: int = 1500,
                 dscp: int = 0, flow_id: int = 0):
        Packet._id_counter += 1
        self.id = Packet._id_counter
        self.src = src
        self.dst = dst
        self.size = size          # bytes
        self.dscp = dscp          # DiffServ Code Point (0-63)
        self.flow_id = flow_id
        self.arrival_time = 0.0
        self.departure_time = 0.0

    @property
    def delay(self):
        return self.departure_time - self.arrival_time

    def __repr__(self):
        return (f"Pkt(id={self.id}, flow={self.flow_id}, "
                f"dscp={self.dscp}, {self.size}B)")


# DSCP value → PHB mapping (simplified)
DSCP_MAP = {
    46: "EF",       # Expedited Forwarding (voice)
    34: "AF41",     # Assured Forwarding class 4 (video)
    26: "AF31",     # Assured Forwarding class 3 (critical data)
    0:  "BE",       # Best Effort
}


def classify_dscp(dscp: int) -> str:
    """Map DSCP value to PHB name."""
    return DSCP_MAP.get(dscp, f"DSCP-{dscp}")


# ============================================================
# Demo 1: Token Bucket Shaper
# ============================================================
class TokenBucket:
    """Token bucket algorithm for traffic shaping.

    Tokens accumulate at `rate` tokens/second, up to `burst` max.
    A packet of size S consumes S tokens. If insufficient tokens,
    the packet is delayed (shaped) or dropped (policed).
    """

    def __init__(self, rate: float, burst: int):
        self.rate = rate          # bytes per second
        self.burst = burst        # max tokens (bytes)
        self.tokens = burst       # start full
        self.last_time = 0.0

    def _refill(self, current_time: float):
        """Add tokens based on elapsed time."""
        elapsed = current_time - self.last_time
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_time = current_time

    def admit(self, packet_size: int, current_time: float) -> bool:
        """Check if a packet can pass (True) or must be delayed/dropped."""
        self._refill(current_time)
        if self.tokens >= packet_size:
            self.tokens -= packet_size
            return True
        return False


def demo_token_bucket():
    """Demonstrate token bucket shaping on a bursty traffic source."""
    print("=" * 60)
    print("Demo 1: Token Bucket Traffic Shaper")
    print("=" * 60)

    # 1 Mbps rate, 3000 byte burst (2 full packets)
    bucket = TokenBucket(rate=125_000, burst=3000)  # 1 Mbps = 125,000 B/s

    rng = random.Random(42)
    admitted = 0
    dropped = 0

    print(f"\n  Rate: 1 Mbps, Burst: 3000 B")
    print(f"  Sending 20 packets in bursty pattern...\n")
    print(f"  {'Time':>6} | {'Size':>5} | {'Tokens':>7} | {'Result':>8}")
    print(f"  {'-'*36}")

    time = 0.0
    for i in range(20):
        # Bursty: 5 packets close together, then a gap
        if i % 5 == 0 and i > 0:
            time += 0.02   # 20ms gap between bursts
        else:
            time += 0.001  # 1ms between packets in a burst

        size = rng.choice([64, 512, 1500])
        bucket._refill(time)
        tokens_before = bucket.tokens

        if bucket.admit(size, time):
            admitted += 1
            result = "PASS"
        else:
            dropped += 1
            result = "DROP"

        print(f"  {time:6.3f} | {size:5d} | {tokens_before:7.0f} | {result:>8}")

    print(f"\n  Results: {admitted} admitted, {dropped} dropped")
    print(f"  Token bucket smooths bursts while allowing short-term excess.")
    print()


# ============================================================
# Demo 2: Queuing Disciplines
# ============================================================
class FIFOQueue:
    """Simple First-In-First-Out queue."""

    def __init__(self, capacity: int = 50):
        self.queue = deque()
        self.capacity = capacity
        self.drops = 0

    def enqueue(self, packet: Packet) -> bool:
        if len(self.queue) >= self.capacity:
            self.drops += 1
            return False
        self.queue.append(packet)
        return True

    def dequeue(self) -> Packet | None:
        return self.queue.popleft() if self.queue else None

    def __len__(self):
        return len(self.queue)


class StrictPriorityQueue:
    """Strict priority queuing (SPQ).

    Higher priority queues are always served first.
    Risk: low-priority starvation.
    """

    def __init__(self, num_queues: int = 4, capacity: int = 20):
        # Queue 0 = highest priority
        self.queues = [deque() for _ in range(num_queues)]
        self.capacity = capacity
        self.drops = 0

    def enqueue(self, packet: Packet, priority: int) -> bool:
        q = self.queues[priority]
        if len(q) >= self.capacity:
            self.drops += 1
            return False
        q.append(packet)
        return True

    def dequeue(self) -> Packet | None:
        for q in self.queues:  # serve highest priority first
            if q:
                return q.popleft()
        return None


class WeightedFairQueue:
    """Weighted Fair Queuing (WFQ).

    Each queue gets bandwidth proportional to its weight.
    Uses virtual finish time to decide scheduling order.
    """

    def __init__(self, weights: dict, capacity: int = 20):
        self.weights = weights  # queue_id -> weight
        self.queues = {qid: deque() for qid in weights}
        self.virtual_time = {qid: 0.0 for qid in weights}
        self.capacity = capacity
        self.drops = 0

    def enqueue(self, packet: Packet, queue_id: int) -> bool:
        if len(self.queues[queue_id]) >= self.capacity:
            self.drops += 1
            return False
        # Compute virtual finish time: size / weight
        vft = self.virtual_time[queue_id] + packet.size / self.weights[queue_id]
        self.queues[queue_id].append((vft, packet))
        self.virtual_time[queue_id] = vft
        return True

    def dequeue(self) -> Packet | None:
        """Serve the queue whose head packet has the smallest finish time."""
        best_qid = None
        best_vft = float("inf")

        for qid, q in self.queues.items():
            if q and q[0][0] < best_vft:
                best_vft = q[0][0]
                best_qid = qid

        if best_qid is not None:
            _, packet = self.queues[best_qid].popleft()
            return packet
        return None


def demo_queuing_disciplines():
    """Compare FIFO, strict priority, and WFQ scheduling."""
    print("=" * 60)
    print("Demo 2: Queuing Disciplines Comparison")
    print("=" * 60)

    rng = random.Random(42)
    Packet._id_counter = 0

    # Generate mixed traffic: voice (EF), video (AF41), data (BE)
    packets = []
    for i in range(60):
        flow_type = rng.choices(
            ["voice", "video", "data"], weights=[1, 2, 4]
        )[0]
        if flow_type == "voice":
            p = Packet("A", "B", size=160, dscp=46, flow_id=1)
        elif flow_type == "video":
            p = Packet("A", "B", size=1200, dscp=34, flow_id=2)
        else:
            p = Packet("A", "B", size=1500, dscp=0, flow_id=3)
        p.arrival_time = i * 0.001
        packets.append(p)

    # --- FIFO ---
    fifo = FIFOQueue(capacity=30)
    fifo_served = {1: 0, 2: 0, 3: 0}
    for p in packets:
        fifo.enqueue(p)
    time = 0.0
    link_speed = 10_000_000  # 10 Mbps
    while fifo:
        p = fifo.dequeue()
        if p is None:
            break
        time += p.size * 8 / link_speed
        p.departure_time = time
        fifo_served[p.flow_id] += 1

    # --- Strict Priority ---
    spq = StrictPriorityQueue(num_queues=3, capacity=20)
    spq_served = {1: 0, 2: 0, 3: 0}
    dscp_to_prio = {46: 0, 34: 1, 0: 2}
    for p in packets:
        prio = dscp_to_prio.get(p.dscp, 2)
        spq.enqueue(p, prio)
    time = 0.0
    while True:
        p = spq.dequeue()
        if p is None:
            break
        time += p.size * 8 / link_speed
        p.departure_time = time
        spq_served[p.flow_id] += 1

    # --- WFQ ---
    wfq = WeightedFairQueue(weights={0: 4, 1: 3, 2: 1}, capacity=20)
    wfq_served = {1: 0, 2: 0, 3: 0}
    dscp_to_queue = {46: 0, 34: 1, 0: 2}
    for p in packets:
        qid = dscp_to_queue.get(p.dscp, 2)
        wfq.enqueue(p, qid)
    time = 0.0
    while True:
        p = wfq.dequeue()
        if p is None:
            break
        time += p.size * 8 / link_speed
        p.departure_time = time
        wfq_served[p.flow_id] += 1

    flow_names = {1: "Voice(EF)", 2: "Video(AF41)", 3: "Data(BE)"}
    print(f"\n  60 packets: voice=~{sum(1 for p in packets if p.flow_id==1)}, "
          f"video=~{sum(1 for p in packets if p.flow_id==2)}, "
          f"data=~{sum(1 for p in packets if p.flow_id==3)}")
    print(f"\n  {'Method':>20} | {'Voice':>6} | {'Video':>6} | {'Data':>6} | {'Drops':>5}")
    print(f"  {'-'*55}")
    print(f"  {'FIFO':>20} | {fifo_served[1]:6d} | {fifo_served[2]:6d} | "
          f"{fifo_served[3]:6d} | {fifo.drops:5d}")
    print(f"  {'Strict Priority':>20} | {spq_served[1]:6d} | {spq_served[2]:6d} | "
          f"{spq_served[3]:6d} | {spq.drops:5d}")
    print(f"  {'WFQ (4:3:1)':>20} | {wfq_served[1]:6d} | {wfq_served[2]:6d} | "
          f"{wfq_served[3]:6d} | {wfq.drops:5d}")

    print(f"\n  FIFO: no differentiation — all flows treated equally")
    print(f"  SPQ:  voice always first, but data may starve")
    print(f"  WFQ:  proportional sharing — fair yet prioritized")
    print()


# ============================================================
# Demo 3: DiffServ DSCP Classification Pipeline
# ============================================================
def demo_dscp_classification():
    """Demonstrate DiffServ classification, marking, and PHB assignment."""
    print("=" * 60)
    print("Demo 3: DiffServ DSCP Classification")
    print("=" * 60)

    # Policy: classify packets by application and mark DSCP
    policies = [
        {"match": {"proto": "rtp"},    "dscp": 46, "phb": "EF",
         "desc": "VoIP → Expedited Forwarding"},
        {"match": {"proto": "rtsp"},   "dscp": 34, "phb": "AF41",
         "desc": "Video → Assured Forwarding 4.1"},
        {"match": {"dst_port": 22},    "dscp": 26, "phb": "AF31",
         "desc": "SSH → Assured Forwarding 3.1"},
        {"match": {"dst_port": 80},    "dscp": 0,  "phb": "BE",
         "desc": "HTTP → Best Effort"},
    ]

    print(f"\n  Classification policies:")
    for p in policies:
        print(f"    {p['match']} → DSCP {p['dscp']:2d} ({p['phb']:>5}) "
              f"  {p['desc']}")

    # Simulate traffic
    traffic = [
        {"src": "10.0.0.1", "dst": "10.0.0.2", "proto": "rtp",
         "dst_port": 5004},
        {"src": "10.0.0.3", "dst": "10.0.0.4", "proto": "rtsp",
         "dst_port": 554},
        {"src": "10.0.0.5", "dst": "10.0.0.6", "proto": "tcp",
         "dst_port": 22},
        {"src": "10.0.0.7", "dst": "10.0.0.8", "proto": "tcp",
         "dst_port": 80},
        {"src": "10.0.0.9", "dst": "10.0.0.10", "proto": "udp",
         "dst_port": 12345},
    ]

    print(f"\n  Packet classification results:")
    print(f"  {'Src':>12} → {'Dst':>12} | {'Proto':>5} | {'Port':>5} | "
          f"{'DSCP':>4} | {'PHB':>5}")
    print(f"  {'-'*60}")

    for pkt in traffic:
        dscp = 0  # default: best effort
        for policy in policies:
            match = True
            for field, value in policy["match"].items():
                if pkt.get(field) != value:
                    match = False
                    break
            if match:
                dscp = policy["dscp"]
                break

        phb = classify_dscp(dscp)
        print(f"  {pkt['src']:>12} → {pkt['dst']:>12} | "
              f"{pkt.get('proto', '?'):>5} | {pkt['dst_port']:>5} | "
              f"{dscp:>4} | {phb:>5}")

    print(f"\n  DiffServ marks packets at the edge; core routers just honor PHB.")
    print(f"  This scales better than per-flow IntServ (no per-flow state in core).")
    print()


if __name__ == "__main__":
    demo_token_bucket()
    demo_queuing_disciplines()
    demo_dscp_classification()
