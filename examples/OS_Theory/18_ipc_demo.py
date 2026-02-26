"""
Inter-Process Communication (IPC) Demonstration

Demonstrates IPC mechanisms using Python:
- Pipe (unidirectional communication)
- Named pipe / message passing (via multiprocessing Queue)
- Shared memory (via multiprocessing shared ctypes)
- Message queue patterns (request-reply, pub-sub)

Theory:
- Pipes: unidirectional byte stream between related processes.
  Used for producer-consumer patterns (e.g., shell pipelines).
- Message queues: structured messages with types/priorities.
  Decouples sender and receiver timing.
- Shared memory: fastest IPC — processes access same physical memory.
  Requires explicit synchronization to avoid races.

Adapted from OS Theory Lesson 18.
"""

import multiprocessing as mp
import os
import time
import struct
from multiprocessing import Queue, Value, Array, Lock


# ── Pipe Communication ──────────────────────────────────────────────────

def pipe_writer(conn, messages: list[str]) -> None:
    """Write messages to pipe."""
    for msg in messages:
        conn.send(msg)
        print(f"    Writer (PID {os.getpid()}): sent '{msg}'")
        time.sleep(0.1)
    conn.send("DONE")
    conn.close()


def pipe_reader(conn) -> None:
    """Read messages from pipe."""
    while True:
        msg = conn.recv()
        if msg == "DONE":
            break
        print(f"    Reader (PID {os.getpid()}): received '{msg}'")
    conn.close()


def demo_pipe():
    """Demonstrate pipe-based IPC."""
    print("=" * 60)
    print("PIPE COMMUNICATION")
    print("=" * 60)

    # Pipe() returns two Connection objects — each end is bound to one process.
    # Pipes are unidirectional by convention (one end writes, the other reads),
    # which mirrors the Unix pipe(2) syscall used for shell pipelines (cmd1 | cmd2).
    parent_conn, child_conn = mp.Pipe()
    messages = ["Hello", "World", "from", "pipe"]

    print(f"\n  Parent PID: {os.getpid()}")
    print(f"  Sending {len(messages)} messages via pipe:\n")

    writer = mp.Process(target=pipe_writer, args=(child_conn, messages))
    reader = mp.Process(target=pipe_reader, args=(parent_conn,))

    writer.start()
    reader.start()
    writer.join()
    reader.join()

    print("\n  Pipe communication complete.")


# ── Message Queue (via multiprocessing.Queue) ───────────────────────────

def mq_producer(queue: Queue, items: list[tuple[str, int]]) -> None:
    """Produce typed messages into queue."""
    for msg_type, value in items:
        message = {"type": msg_type, "value": value, "pid": os.getpid()}
        queue.put(message)
        print(f"    Producer: sent {msg_type}={value}")
        time.sleep(0.05)
    # Sentinel message signals the consumer to exit its receive loop —
    # without a shutdown protocol, the consumer would block on get() forever
    # after the producer finishes, causing the program to hang
    queue.put({"type": "SHUTDOWN", "value": 0, "pid": os.getpid()})


def mq_consumer(queue: Queue, name: str) -> None:
    """Consume messages from queue, filtering by type."""
    count = 0
    while True:
        message = queue.get()
        if message["type"] == "SHUTDOWN":
            break
        count += 1
        print(f"    {name}: received {message['type']}={message['value']}")
    print(f"    {name}: processed {count} messages")


def demo_message_queue():
    """Demonstrate message queue IPC."""
    print("\n" + "=" * 60)
    print("MESSAGE QUEUE")
    print("=" * 60)

    queue = Queue()
    items = [
        ("TEMP", 23),
        ("PRESSURE", 1013),
        ("TEMP", 24),
        ("HUMIDITY", 65),
        ("PRESSURE", 1012),
        ("TEMP", 22),
    ]

    print(f"\n  {len(items)} sensor messages via queue:\n")

    producer = mp.Process(target=mq_producer, args=(queue, items))
    consumer = mp.Process(target=mq_consumer, args=(queue, "Consumer"))

    producer.start()
    consumer.start()
    producer.join()
    consumer.join()

    print("\n  Message queue complete.")


# ── Shared Memory ───────────────────────────────────────────────────────

def shared_mem_writer(shared_counter: Value, shared_array: Array, lock: Lock) -> None:
    """Increment shared counter and write to shared array."""
    pid = os.getpid()
    for i in range(5):
        with lock:
            shared_counter.value += 1
            idx = min(i, len(shared_array) - 1)
            shared_array[idx] = pid % 100  # store last 2 digits of PID
        print(f"    Writer (PID {pid}): counter={shared_counter.value}")
        time.sleep(0.05)


def shared_mem_reader(shared_counter: Value, shared_array: Array, lock: Lock) -> None:
    """Read from shared memory."""
    time.sleep(0.15)  # let writer get ahead
    pid = os.getpid()
    with lock:
        print(f"    Reader (PID {pid}): counter={shared_counter.value}")
        vals = list(shared_array)
        print(f"    Reader: array={vals}")


def demo_shared_memory():
    """Demonstrate shared memory IPC."""
    print("\n" + "=" * 60)
    print("SHARED MEMORY")
    print("=" * 60)

    # Shared counter (integer) and array — these use shared memory (mmap'd
    # regions) under the hood, making them the fastest IPC mechanism since
    # no kernel buffer copying occurs. The trade-off: we must provide our
    # own synchronization (the Lock below) to avoid data races.
    counter = Value("i", 0)
    array = Array("i", [0] * 5)
    lock = Lock()

    print(f"\n  Shared counter (int) + array (5 ints):")
    print(f"  Writer increments counter 5 times.\n")

    writer = mp.Process(target=shared_mem_writer, args=(counter, array, lock))
    reader = mp.Process(target=shared_mem_reader, args=(counter, array, lock))

    writer.start()
    reader.start()
    writer.join()
    reader.join()

    print(f"\n  Final counter: {counter.value}")
    print(f"  Final array: {list(array)}")
    print(f"  Shared memory complete.")


# ── Request-Reply Pattern ───────────────────────────────────────────────

def server(request_q: Queue, reply_q: Queue) -> None:
    """Simple request-reply server."""
    pid = os.getpid()
    print(f"    Server (PID {pid}) started")

    while True:
        request = request_q.get()
        if request["op"] == "SHUTDOWN":
            break

        # Process request
        op = request["op"]
        a, b = request["a"], request["b"]
        if op == "ADD":
            result = a + b
        elif op == "MUL":
            result = a * b
        else:
            result = None

        reply_q.put({"result": result, "op": op})
        print(f"    Server: {op}({a}, {b}) = {result}")

    print(f"    Server shutting down")


def client(request_q: Queue, reply_q: Queue) -> None:
    """Simple request-reply client."""
    pid = os.getpid()
    requests = [
        {"op": "ADD", "a": 10, "b": 20},
        {"op": "MUL", "a": 6, "b": 7},
        {"op": "ADD", "a": 100, "b": 200},
    ]

    for req in requests:
        request_q.put(req)
        reply = reply_q.get()
        print(f"    Client: {req['op']}({req['a']}, {req['b']}) → {reply['result']}")

    request_q.put({"op": "SHUTDOWN"})


def demo_request_reply():
    """Demonstrate request-reply IPC pattern."""
    print("\n" + "=" * 60)
    print("REQUEST-REPLY PATTERN")
    print("=" * 60)

    # Two separate queues implement the request-reply pattern — a single
    # queue would require message filtering to distinguish requests from
    # replies, and the client could accidentally dequeue its own request
    request_q = Queue()
    reply_q = Queue()

    print(f"\n  Client sends math operations, server computes results:\n")

    srv = mp.Process(target=server, args=(request_q, reply_q))
    cli = mp.Process(target=client, args=(request_q, reply_q))

    srv.start()
    cli.start()
    cli.join()
    srv.join()

    print("\n  Request-reply complete.")


# ── IPC Comparison ──────────────────────────────────────────────────────

def demo_comparison():
    """Summarize IPC mechanism trade-offs."""
    print("\n" + "=" * 60)
    print("IPC MECHANISM COMPARISON")
    print("=" * 60)

    comparison = [
        ("Pipe", "Unidirectional", "Stream", "Fast", "Related processes"),
        ("Named Pipe", "Unidirectional", "Stream", "Fast", "Any processes"),
        ("Message Queue", "Bidirectional", "Messages", "Medium", "Structured data"),
        ("Shared Memory", "Bidirectional", "Raw bytes", "Fastest", "High throughput"),
        ("Socket", "Bidirectional", "Stream/Dgram", "Slow", "Network/local"),
        ("Signal", "Unidirectional", "Notification", "Fast", "Simple events"),
    ]

    print(f"\n  {'Mechanism':<16} {'Direction':<16} {'Data':<12} {'Speed':<10} {'Best For'}")
    print(f"  {'-'*16} {'-'*16} {'-'*12} {'-'*10} {'-'*20}")
    for row in comparison:
        print(f"  {row[0]:<16} {row[1]:<16} {row[2]:<12} {row[3]:<10} {row[4]}")


if __name__ == "__main__":
    demo_pipe()
    demo_message_queue()
    demo_shared_memory()
    demo_request_reply()
    demo_comparison()
