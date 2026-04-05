"""
Exercises for Lesson 01: System Models and Failure Modes
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import time
import random
import threading
from enum import Enum
from typing import Dict, List, Optional, Any


# === Exercise 1: Classify System Models ===
# Problem: Given a set of distributed system scenarios, classify each
# as synchronous, asynchronous, or partially synchronous. Justify the
# classification based on timing assumptions about message delivery,
# process speed, and clock drift.

class SystemModel(Enum):
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    PARTIALLY_SYNCHRONOUS = "partially_synchronous"


def classify_system_model(
    has_message_bound: bool,
    has_processing_bound: bool,
    has_clock_bound: bool,
    bounds_known_after_gst: bool = False,
) -> SystemModel:
    """
    Classify a system model based on its timing properties.

    Args:
        has_message_bound: Messages are delivered within a known time bound.
        has_processing_bound: Processes execute steps within a known time bound.
        has_clock_bound: Clock drift is bounded.
        bounds_known_after_gst: Bounds hold only after some unknown Global
            Stabilization Time (GST).

    Returns:
        The system model classification.
    """
    if has_message_bound and has_processing_bound and has_clock_bound:
        if bounds_known_after_gst:
            return SystemModel.PARTIALLY_SYNCHRONOUS
        return SystemModel.SYNCHRONOUS
    elif not has_message_bound and not has_processing_bound:
        return SystemModel.ASYNCHRONOUS
    elif bounds_known_after_gst:
        return SystemModel.PARTIALLY_SYNCHRONOUS
    else:
        return SystemModel.ASYNCHRONOUS


def exercise_1():
    """
    Classify several real-world distributed system scenarios.
    """
    scenarios = [
        {
            "name": "Hard real-time embedded controller (CAN bus)",
            "has_message_bound": True,
            "has_processing_bound": True,
            "has_clock_bound": True,
            "bounds_known_after_gst": False,
            "expected": SystemModel.SYNCHRONOUS,
        },
        {
            "name": "Internet-based microservices",
            "has_message_bound": False,
            "has_processing_bound": False,
            "has_clock_bound": False,
            "bounds_known_after_gst": False,
            "expected": SystemModel.ASYNCHRONOUS,
        },
        {
            "name": "Datacenter with occasional network partition",
            "has_message_bound": True,
            "has_processing_bound": True,
            "has_clock_bound": True,
            "bounds_known_after_gst": True,
            "expected": SystemModel.PARTIALLY_SYNCHRONOUS,
        },
        {
            "name": "Peer-to-peer file sharing network",
            "has_message_bound": False,
            "has_processing_bound": False,
            "has_clock_bound": False,
            "bounds_known_after_gst": False,
            "expected": SystemModel.ASYNCHRONOUS,
        },
        {
            "name": "LAN cluster with heartbeat monitoring",
            "has_message_bound": True,
            "has_processing_bound": True,
            "has_clock_bound": True,
            "bounds_known_after_gst": True,
            "expected": SystemModel.PARTIALLY_SYNCHRONOUS,
        },
    ]

    print("=== Exercise 1: Classify System Models ===\n")
    for s in scenarios:
        result = classify_system_model(
            s["has_message_bound"],
            s["has_processing_bound"],
            s["has_clock_bound"],
            s.get("bounds_known_after_gst", False),
        )
        status = "PASS" if result == s["expected"] else "FAIL"
        print(f"[{status}] {s['name']}")
        print(f"       Classification: {result.value}")
    print()


# === Exercise 2: Crash-Recovery Simulator ===
# Problem: Implement a process that can crash and recover. On recovery
# it must resume from its last persisted state. Demonstrate that
# volatile state is lost but durable state survives.

class CrashRecoveryProcess:
    """
    Simulates a process with crash-recovery semantics.

    - Volatile state (in-memory counter) is lost on crash.
    - Durable state (written to a dict acting as stable storage) survives.
    """

    def __init__(self, process_id: str, stable_storage: Dict[str, Any]):
        self.process_id = process_id
        self.stable_storage = stable_storage
        self.volatile_counter = 0
        self.is_alive = True
        # Recover durable state if it exists
        self.durable_counter = self.stable_storage.get(
            f"{self.process_id}_counter", 0
        )

    def do_work(self) -> Optional[int]:
        """Perform a unit of work: increment both counters."""
        if not self.is_alive:
            return None
        self.volatile_counter += 1
        self.durable_counter += 1
        # Persist durable state
        self.stable_storage[f"{self.process_id}_counter"] = self.durable_counter
        return self.durable_counter

    def crash(self):
        """Simulate a crash: volatile state is lost."""
        self.is_alive = False
        self.volatile_counter = 0  # volatile state gone

    def recover(self):
        """Recover from crash: reload durable state from stable storage."""
        self.is_alive = True
        self.volatile_counter = 0  # volatile state starts fresh
        self.durable_counter = self.stable_storage.get(
            f"{self.process_id}_counter", 0
        )


def exercise_2():
    """
    Demonstrate crash-recovery behavior: volatile state is lost,
    durable state is preserved across crashes.
    """
    print("=== Exercise 2: Crash-Recovery Simulator ===\n")

    stable_storage: Dict[str, Any] = {}
    proc = CrashRecoveryProcess("P1", stable_storage)

    # Do 5 units of work
    for _ in range(5):
        proc.do_work()
    print(f"After 5 operations:")
    print(f"  Volatile counter: {proc.volatile_counter}")
    print(f"  Durable counter:  {proc.durable_counter}")

    # Crash
    proc.crash()
    print(f"\nAfter crash:")
    print(f"  Volatile counter: {proc.volatile_counter} (lost!)")
    print(f"  Process alive:    {proc.is_alive}")

    # Recover
    proc.recover()
    print(f"\nAfter recovery:")
    print(f"  Volatile counter: {proc.volatile_counter} (reset to 0)")
    print(f"  Durable counter:  {proc.durable_counter} (restored to 5)")

    # Continue work
    for _ in range(3):
        proc.do_work()
    print(f"\nAfter 3 more operations:")
    print(f"  Volatile counter: {proc.volatile_counter}")
    print(f"  Durable counter:  {proc.durable_counter}")

    assert proc.durable_counter == 8, "Durable counter should be 8"
    assert proc.volatile_counter == 3, "Volatile counter should be 3"
    print("\nAll assertions passed.")
    print()


# === Exercise 3: Byzantine Process Simulator ===
# Problem: Implement a Byzantine process that sends conflicting messages
# to different receivers. Show how honest processes receive different
# values from the same sender, violating agreement.

class Message:
    """A message in the distributed system."""

    def __init__(self, sender: str, receiver: str, value: Any):
        self.sender = sender
        self.receiver = receiver
        self.value = value

    def __repr__(self):
        return f"Msg({self.sender}->{self.receiver}: {self.value})"


class ByzantineProcess:
    """
    A process that may behave honestly or in a Byzantine fashion.
    Byzantine behavior: sends different values to different receivers.
    """

    def __init__(self, process_id: str, is_byzantine: bool = False):
        self.process_id = process_id
        self.is_byzantine = is_byzantine
        self.received_messages: List[Message] = []

    def send(self, value: Any, receivers: List[str]) -> List[Message]:
        """
        Send a value to all receivers.
        If Byzantine, send a different value to each receiver.
        """
        messages = []
        if self.is_byzantine:
            # Send conflicting messages
            for i, r in enumerate(receivers):
                conflicting_value = f"{value}_variant_{i}"
                messages.append(Message(self.process_id, r, conflicting_value))
        else:
            for r in receivers:
                messages.append(Message(self.process_id, r, value))
        return messages

    def receive(self, msg: Message):
        """Receive and store a message."""
        self.received_messages.append(msg)


def exercise_3():
    """
    Demonstrate Byzantine behavior: a faulty process sends conflicting
    messages to honest receivers, breaking agreement.
    """
    print("=== Exercise 3: Byzantine Process Simulator ===\n")

    # Create processes: general (Byzantine) + 3 honest lieutenants
    general = ByzantineProcess("General", is_byzantine=True)
    lieutenants = [
        ByzantineProcess(f"Lieutenant_{i}", is_byzantine=False) for i in range(3)
    ]
    lt_ids = [lt.process_id for lt in lieutenants]

    # General sends "ATTACK" but is Byzantine
    messages = general.send("ATTACK", lt_ids)

    print("General (Byzantine) sends 'ATTACK' command to 3 lieutenants:\n")
    for msg in messages:
        # Deliver message
        for lt in lieutenants:
            if lt.process_id == msg.receiver:
                lt.receive(msg)
        print(f"  {msg}")

    print("\nWhat each lieutenant received:")
    values_seen = set()
    for lt in lieutenants:
        for msg in lt.received_messages:
            print(f"  {lt.process_id} got: {msg.value}")
            values_seen.add(msg.value)

    print(f"\nDistinct values received: {len(values_seen)}")
    print(f"Agreement violated: {len(values_seen) > 1}")
    assert len(values_seen) > 1, "Byzantine process should cause disagreement"
    print("Byzantine fault demonstration successful.")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
