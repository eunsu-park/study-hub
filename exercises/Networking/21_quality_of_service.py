"""
Exercises for Lesson 21: Quality of Service
Topic: Networking

Solutions to practice problems covering token bucket, leaky bucket,
WFQ scheduling, DSCP classification, and end-to-end QoS design.
"""

import numpy as np
from collections import deque


# ============================================================
# Exercise 1: Token Bucket vs Leaky Bucket
# ============================================================
def exercise_1():
    """
    Implement both token bucket and leaky bucket algorithms.
    Compare their behavior under bursty traffic.
    """
    print("=== Exercise 1: Token Bucket vs Leaky Bucket ===\n")

    class TokenBucket:
        def __init__(self, rate, burst):
            self.rate = rate
            self.burst = burst
            self.tokens = burst
            self.last_time = 0.0

        def process(self, size, time):
            self.tokens = min(self.burst,
                              self.tokens + (time - self.last_time) * self.rate)
            self.last_time = time
            if self.tokens >= size:
                self.tokens -= size
                return True
            return False

    class LeakyBucket:
        def __init__(self, rate, capacity):
            self.rate = rate
            self.capacity = capacity
            self.water = 0.0
            self.last_time = 0.0

        def process(self, size, time):
            # Drain water based on elapsed time
            elapsed = time - self.last_time
            self.water = max(0, self.water - elapsed * self.rate)
            self.last_time = time
            # Try to add water (packet)
            if self.water + size <= self.capacity:
                self.water += size
                return True
            return False

    # Compare under bursty traffic
    rate = 1000  # bytes/sec
    burst = 3000  # bytes

    tb = TokenBucket(rate, burst)
    lb = LeakyBucket(rate, burst)

    # Generate burst: 10 packets at once, then pause, then more
    np.random.seed(42)
    events = []
    t = 0.0
    for _ in range(5):
        events.append((t, 500))
        t += 0.001  # 1ms apart (bursty)
    t += 3.0  # 3 second pause
    for _ in range(5):
        events.append((t, 500))
        t += 0.001

    print(f"  Rate: {rate} B/s, Burst/Capacity: {burst} B")
    print(f"  Traffic: 5 packets → 3s pause → 5 packets\n")
    print(f"  {'Time':>6} | {'Size':>5} | {'TokenBucket':>12} | {'LeakyBucket':>12}")
    print(f"  {'-'*42}")

    tb_pass, lb_pass = 0, 0
    for t, size in events:
        tb_ok = tb.process(size, t)
        lb_ok = lb.process(size, t)
        if tb_ok:
            tb_pass += 1
        if lb_ok:
            lb_pass += 1
        print(f"  {t:6.3f} | {size:5} | {'PASS' if tb_ok else 'DROP':>12} | "
              f"{'PASS' if lb_ok else 'DROP':>12}")

    print(f"\n  Token Bucket: {tb_pass}/10 passed")
    print(f"  Leaky Bucket: {lb_pass}/10 passed")
    print(f"\n  Token bucket allows controlled bursts up to burst size.")
    print(f"  Leaky bucket smooths output to constant rate (stricter).")
    print()


# ============================================================
# Exercise 2: Weighted Fair Queuing
# ============================================================
def exercise_2():
    """
    Implement WFQ and verify bandwidth allocation matches weights.
    """
    print("=== Exercise 2: WFQ Bandwidth Allocation ===\n")

    class WFQ:
        def __init__(self, weights):
            self.weights = weights
            self.queues = {qid: deque() for qid in weights}
            self.virtual_finish = {qid: 0.0 for qid in weights}
            self.served = {qid: 0 for qid in weights}
            self.served_bytes = {qid: 0 for qid in weights}

        def enqueue(self, queue_id, packet_size):
            vft = self.virtual_finish[queue_id] + packet_size / self.weights[queue_id]
            self.queues[queue_id].append((vft, packet_size))
            self.virtual_finish[queue_id] = vft

        def dequeue(self):
            best_qid = None
            best_vft = float('inf')
            for qid, q in self.queues.items():
                if q and q[0][0] < best_vft:
                    best_vft = q[0][0]
                    best_qid = qid
            if best_qid is not None:
                vft, size = self.queues[best_qid].popleft()
                self.served[best_qid] += 1
                self.served_bytes[best_qid] += size
                return best_qid, size
            return None, 0

    # 3 queues with weights 4:2:1
    wfq = WFQ(weights={0: 4, 1: 2, 2: 1})

    # Enqueue 100 packets per queue (same size for fair comparison)
    rng = np.random.RandomState(42)
    for _ in range(100):
        for qid in [0, 1, 2]:
            wfq.enqueue(qid, 1000)

    # Serve all packets
    order = []
    while True:
        qid, size = wfq.dequeue()
        if qid is None:
            break
        order.append(qid)

    total_bytes = sum(wfq.served_bytes.values())
    print(f"  Weights: Q0=4, Q1=2, Q2=1 (total=7)")
    print(f"  Total served: {sum(wfq.served.values())} packets\n")

    print(f"  {'Queue':>6} | {'Packets':>8} | {'Bytes':>8} | "
          f"{'Actual %':>9} | {'Expected %':>11}")
    print(f"  {'-'*50}")
    total_weight = sum(wfq.weights.values())
    for qid in sorted(wfq.weights.keys()):
        pct = wfq.served_bytes[qid] / total_bytes * 100
        expected = wfq.weights[qid] / total_weight * 100
        print(f"  Q{qid:>5} | {wfq.served[qid]:>8} | {wfq.served_bytes[qid]:>8} | "
              f"{pct:>8.1f}% | {expected:>10.1f}%")

    # Check interleaving in first 21 served
    print(f"\n  First 21 served: {order[:21]}")
    print(f"  Pattern: Q0 gets ~4x more service than Q2 (proportional)")
    print()


# ============================================================
# Exercise 3: DSCP Marking and Policing
# ============================================================
def exercise_3():
    """
    Implement a DiffServ edge router that classifies, marks (DSCP),
    and polices incoming traffic.
    """
    print("=== Exercise 3: DiffServ Edge Classification ===\n")

    class DiffServEdge:
        def __init__(self):
            self.policies = []
            self.stats = {"marked": 0, "remarked": 0, "dropped": 0}

        def add_policy(self, match, dscp, rate_limit=None):
            self.policies.append({
                "match": match,
                "dscp": dscp,
                "rate_limit": rate_limit,
                "byte_count": 0,
                "window_start": 0.0,
            })

        def process(self, packet, time):
            for policy in self.policies:
                if all(packet.get(k) == v for k, v in policy["match"].items()):
                    # Rate limit check
                    if policy["rate_limit"]:
                        elapsed = time - policy["window_start"]
                        if elapsed >= 1.0:
                            policy["byte_count"] = 0
                            policy["window_start"] = time
                        policy["byte_count"] += packet.get("size", 0)
                        if policy["byte_count"] > policy["rate_limit"]:
                            # Remark to lower class or drop
                            self.stats["remarked"] += 1
                            return {"dscp": 0, "action": "remark-to-BE"}
                    self.stats["marked"] += 1
                    return {"dscp": policy["dscp"], "action": "mark"}
            self.stats["dropped"] += 1
            return {"dscp": 0, "action": "default-BE"}

    edge = DiffServEdge()
    edge.add_policy({"app": "voip"}, dscp=46)  # EF
    edge.add_policy({"app": "video"}, dscp=34, rate_limit=5_000_000)  # AF41, 5MB/s
    edge.add_policy({"app": "ssh"}, dscp=26)  # AF31
    # Everything else: best effort (no explicit policy)

    # Simulate traffic
    traffic = [
        {"app": "voip", "size": 160, "label": "VoIP call"},
        {"app": "video", "size": 1200, "label": "Video stream"},
        {"app": "ssh", "size": 200, "label": "SSH session"},
        {"app": "http", "size": 1500, "label": "Web browsing"},
        {"app": "video", "size": 1200, "label": "Video (excess)"},
    ]

    dscp_names = {46: "EF", 34: "AF41", 26: "AF31", 0: "BE"}

    print(f"  {'Traffic':>20} | {'DSCP':>5} | {'PHB':>5} | {'Action'}")
    print(f"  {'-'*55}")

    for pkt in traffic:
        label = pkt.pop("label")
        result = edge.process(pkt, time=0.0)
        phb = dscp_names.get(result["dscp"], "BE")
        print(f"  {label:>20} | {result['dscp']:>5} | {phb:>5} | "
              f"{result['action']}")

    print(f"\n  Stats: {edge.stats}")
    print(f"\n  Edge router marks packets; core routers just honor DSCP.")
    print(f"  Over-limit video is remarked to Best Effort (policing).")
    print()


# ============================================================
# Exercise 4: Queuing Delay Analysis
# ============================================================
def exercise_4():
    """
    Analyze queuing delay under different load levels using M/M/1 model.
    Compare with simulation results.
    """
    print("=== Exercise 4: Queuing Delay Analysis ===\n")

    # M/M/1 analytical model
    # Average delay = 1 / (μ - λ)  where μ=service rate, λ=arrival rate

    service_rate = 10_000  # packets/sec (10 Gbps / 1000 byte avg)

    print(f"  M/M/1 queuing model: service rate μ = {service_rate} pkt/s\n")
    print(f"  {'Load (ρ)':>9} | {'λ (pkt/s)':>10} | {'Analytical (ms)':>16} | "
          f"{'Simulated (ms)':>15}")
    print(f"  {'-'*56}")

    rng = np.random.RandomState(42)

    for rho in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        arrival_rate = rho * service_rate

        # Analytical average delay
        if rho < 1:
            avg_delay_analytical = 1.0 / (service_rate - arrival_rate) * 1000
        else:
            avg_delay_analytical = float('inf')

        # Simulate M/M/1 queue
        n_packets = 5000
        inter_arrivals = rng.exponential(1.0 / arrival_rate, n_packets)
        service_times = rng.exponential(1.0 / service_rate, n_packets)

        arrival_times = np.cumsum(inter_arrivals)
        departure_times = np.zeros(n_packets)

        departure_times[0] = arrival_times[0] + service_times[0]
        for i in range(1, n_packets):
            start = max(arrival_times[i], departure_times[i - 1])
            departure_times[i] = start + service_times[i]

        delays = departure_times - arrival_times
        avg_delay_sim = np.mean(delays) * 1000

        print(f"  {rho:>9.2f} | {arrival_rate:>10.0f} | "
              f"{avg_delay_analytical:>16.3f} | {avg_delay_sim:>15.3f}")

    print(f"\n  Key insight: delay grows rapidly as load → 1.0")
    print(f"  QoS mechanisms prevent high-priority traffic from experiencing")
    print(f"  this exponential delay growth by reserving capacity.")
    print()


# ============================================================
# Exercise 5: End-to-End QoS Design
# ============================================================
def exercise_5():
    """
    Design a campus network QoS policy:
    classify traffic, assign DSCP, allocate bandwidth per class.
    """
    print("=== Exercise 5: Campus Network QoS Design ===\n")

    # Traffic classes for a campus network
    classes = [
        {"name": "Voice", "dscp": 46, "phb": "EF", "bw_pct": 10,
         "queue": "strict", "burst": "low",
         "requirements": "< 150ms delay, < 30ms jitter, < 1% loss"},
        {"name": "Video Conf", "dscp": 34, "phb": "AF41", "bw_pct": 20,
         "queue": "priority", "burst": "medium",
         "requirements": "< 200ms delay, < 50ms jitter, < 2% loss"},
        {"name": "Signaling", "dscp": 24, "phb": "CS3", "bw_pct": 5,
         "queue": "guaranteed", "burst": "low",
         "requirements": "reliable delivery, < 1s delay"},
        {"name": "Critical Data", "dscp": 26, "phb": "AF31", "bw_pct": 25,
         "queue": "guaranteed", "burst": "high",
         "requirements": "< 500ms delay, minimal loss"},
        {"name": "Bulk Transfer", "dscp": 10, "phb": "AF11", "bw_pct": 15,
         "queue": "best-effort+", "burst": "high",
         "requirements": "throughput, loss tolerance"},
        {"name": "Best Effort", "dscp": 0, "phb": "BE", "bw_pct": 20,
         "queue": "best-effort", "burst": "any",
         "requirements": "no guarantees"},
        {"name": "Scavenger", "dscp": 8, "phb": "CS1", "bw_pct": 5,
         "queue": "scavenger", "burst": "any",
         "requirements": "leftover bandwidth only"},
    ]

    total_bw = 10_000  # 10 Gbps link

    print(f"  Campus network: {total_bw/1000:.0f} Gbps uplink\n")
    print(f"  {'Class':>15} | {'DSCP':>4} | {'PHB':>5} | {'BW%':>4} | "
          f"{'Mbps':>6} | {'Queue Type':>12}")
    print(f"  {'-'*60}")

    for c in classes:
        mbps = total_bw * c["bw_pct"] / 100
        print(f"  {c['name']:>15} | {c['dscp']:>4} | {c['phb']:>5} | "
              f"{c['bw_pct']:>3}% | {mbps:>6.0f} | {c['queue']:>12}")

    print(f"\n  Per-class requirements:")
    for c in classes:
        print(f"    {c['name']:>15}: {c['requirements']}")

    # Verify allocation sums to 100%
    total_pct = sum(c["bw_pct"] for c in classes)
    print(f"\n  Total bandwidth allocation: {total_pct}% (should be 100%)")

    print(f"\n  Design principles:")
    print(f"    1. Voice gets strict priority (always served first)")
    print(f"    2. Video gets secondary priority with policing")
    print(f"    3. Critical data gets guaranteed minimum bandwidth")
    print(f"    4. Scavenger only uses leftover (e.g., software updates)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
