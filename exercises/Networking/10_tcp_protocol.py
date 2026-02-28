"""
Exercises for Lesson 10: TCP Protocol
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem 1: 3-Way Handshake Analysis
    Packet 1: 192.168.1.10:50000 -> 10.0.0.5:443, SYN, Seq=1000000000
    Packet 2: 10.0.0.5:443 -> 192.168.1.10:50000, SYN-ACK, Seq=2000000000, Ack=?
    Packet 3: 192.168.1.10:50000 -> 10.0.0.5:443, ACK, Seq=?, Ack=?

    Reasoning: The SYN flag consumes one sequence number, so the ACK value
    is always the peer's Seq + 1 during the handshake.
    """
    client_isn = 1000000000
    server_isn = 2000000000

    handshake = [
        {"pkt": 1, "src": "Client", "flags": "SYN",
         "seq": client_isn, "ack": None},
        {"pkt": 2, "src": "Server", "flags": "SYN-ACK",
         "seq": server_isn, "ack": client_isn + 1},
        {"pkt": 3, "src": "Client", "flags": "ACK",
         "seq": client_isn + 1, "ack": server_isn + 1},
    ]

    print("3-Way Handshake Analysis:")
    for step in handshake:
        ack_str = f"Ack={step['ack']}" if step["ack"] else "No Ack"
        print(f"\n  Packet {step['pkt']} ({step['src']}):")
        print(f"    Flags: {step['flags']}, Seq={step['seq']}, {ack_str}")

    print(f"\n  Answers:")
    print(f"    a) Packet 2 Ack = {client_isn + 1} (Client Seq + 1)")
    print(f"    b) Packet 3 Seq = {client_isn + 1} (SYN counts as 1 byte)")
    print(f"    c) Packet 3 Ack = {server_isn + 1} (Server Seq + 1)")


def exercise_2():
    """
    Problem 2: Sequence Number Calculation
    Client sends 5000 bytes, MSS=1000 bytes, initial Seq=10000.
    Calculate Seq and expected ACK for each segment.

    Reasoning: TCP uses byte-stream sequence numbers. Each segment's Seq
    is the byte offset of the first byte in that segment. The expected
    ACK is the next byte the receiver expects (Seq + data_size).
    """
    initial_seq = 10000
    mss = 1000
    total_data = 5000
    num_segments = total_data // mss

    print("Sequence Number Calculation:")
    print(f"  Initial Seq: {initial_seq}, MSS: {mss}, Total: {total_data} bytes")
    print(f"\n  {'Segment':>8s} {'Data Size':>10s} {'Seq':>10s} {'Expected ACK':>14s} {'Bytes':>15s}")
    print(f"  {'-'*60}")

    for i in range(num_segments):
        seq = initial_seq + i * mss
        expected_ack = seq + mss
        byte_range = f"{seq}-{seq + mss - 1}"
        print(f"  {i + 1:>8d} {mss:>10d} {seq:>10d} {expected_ack:>14d} {byte_range:>15s}")

    print(f"\n  After all segments: next expected Seq = {initial_seq + total_data}")


def exercise_3():
    """
    Problem 3: Flow Control
    Receiver buffer = 10000 bytes, currently 2000 bytes in buffer.
    a) Advertised window size?
    b) After sender transmits 4000 bytes?
    c) After application reads 3000 bytes?

    Reasoning: TCP flow control uses the receive window (rwnd) to prevent
    the sender from overwhelming the receiver. The window shrinks as data
    arrives and grows as the application consumes data.
    """
    buffer_size = 10000
    initial_used = 2000

    print("Flow Control Analysis:")
    print(f"  Receive buffer: {buffer_size} bytes")
    print(f"  Initial buffer usage: {initial_used} bytes")

    # a) Initial window
    window_a = buffer_size - initial_used
    print(f"\n  a) Advertised window = {buffer_size} - {initial_used} = {window_a} bytes")

    # b) After receiving 4000 bytes
    new_used = initial_used + 4000
    window_b = buffer_size - new_used
    print(f"\n  b) After receiving 4000 bytes:")
    print(f"     Buffer used: {initial_used} + 4000 = {new_used}")
    print(f"     New window = {buffer_size} - {new_used} = {window_b} bytes")

    # c) After application reads 3000 bytes
    after_read = new_used - 3000
    window_c = buffer_size - after_read
    print(f"\n  c) After application reads 3000 bytes:")
    print(f"     Buffer used: {new_used} - 3000 = {after_read}")
    print(f"     New window = {buffer_size} - {after_read} = {window_c} bytes")

    # Diagram
    print(f"\n  Buffer state visualization:")
    states = [
        ("Initial", initial_used, window_a),
        ("After recv 4000", new_used, window_b),
        ("After app reads 3000", after_read, window_c),
    ]
    for label, used, free in states:
        bar_used = int(used / buffer_size * 40)
        bar_free = 40 - bar_used
        print(f"    {label:25s}: [{'#' * bar_used}{'.' * bar_free}] "
              f"used={used}, free={free}")


def exercise_4():
    """
    Problem 4: Congestion Control
    Starting with ssthresh=16 MSS, cwnd=1 MSS.
    a) What is cwnd after 4 RTTs? (no loss)
    b) When cwnd=32 MSS, timeout occurs. New ssthresh and cwnd?
    c) When cwnd=24 MSS, 3 duplicate ACKs occur. New ssthresh and cwnd?

    Reasoning: TCP's congestion control has two phases (slow start and
    congestion avoidance) and two loss responses (timeout vs 3 dup ACKs).
    """
    ssthresh = 16
    cwnd = 1

    print("Congestion Control Analysis:")
    print(f"  Initial: ssthresh={ssthresh} MSS, cwnd={cwnd} MSS")

    # a) cwnd after 4 RTTs
    print(f"\n  a) cwnd after 4 RTTs (no loss):")
    print(f"     Phase: Slow Start (cwnd < ssthresh, doubling each RTT)")
    cwnd_trace = [cwnd]
    temp_cwnd = cwnd
    for rtt in range(1, 5):
        temp_cwnd *= 2  # Slow start: double each RTT
        phase = "Slow Start" if temp_cwnd <= ssthresh else "Congestion Avoidance"
        print(f"     RTT {rtt}: cwnd = {temp_cwnd} MSS ({phase})")
        cwnd_trace.append(temp_cwnd)
    print(f"     Answer: cwnd = {temp_cwnd} MSS after 4 RTTs")

    # b) Timeout at cwnd=32
    cwnd_at_loss = 32
    print(f"\n  b) Timeout at cwnd={cwnd_at_loss} MSS:")
    new_ssthresh = cwnd_at_loss // 2
    new_cwnd = 1
    print(f"     Timeout response: Reset to slow start")
    print(f"     New ssthresh = {cwnd_at_loss} / 2 = {new_ssthresh} MSS")
    print(f"     New cwnd = {new_cwnd} MSS (back to slow start)")

    # c) 3 duplicate ACKs at cwnd=24
    cwnd_at_dup = 24
    print(f"\n  c) 3 duplicate ACKs at cwnd={cwnd_at_dup} MSS:")
    dup_ssthresh = cwnd_at_dup // 2
    dup_cwnd = dup_ssthresh + 3  # Fast Recovery
    print(f"     Fast Retransmit + Fast Recovery:")
    print(f"     New ssthresh = {cwnd_at_dup} / 2 = {dup_ssthresh} MSS")
    print(f"     New cwnd = {dup_ssthresh} + 3 = {dup_cwnd} MSS (Fast Recovery)")
    print(f"\n     Note: 3 dup ACKs indicate network still delivering packets,")
    print(f"     so recovery is less aggressive than timeout.")


if __name__ == "__main__":
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
