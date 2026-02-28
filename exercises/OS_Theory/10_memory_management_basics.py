"""
Exercises for Lesson 10: Memory Management Basics
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers address translation with relocation and limit registers.
"""


# === Exercise 1: Address Translation ===
# Problem: Translate logical addresses using relocation and limit registers.

def exercise_1():
    """Address translation with relocation and limit registers."""
    relocation = 0x4000
    limit = 0x3000

    print(f"Given: Relocation register = 0x{relocation:04X}")
    print(f"       Limit register = 0x{limit:04X}\n")

    # Test case 1
    logical_addr_1 = 0x1500
    physical_addr_1 = relocation + logical_addr_1
    within_limit_1 = logical_addr_1 < limit

    print(f"Q1: Logical address 0x{logical_addr_1:04X}")
    print(f"  Check: 0x{logical_addr_1:04X} < 0x{limit:04X}? {within_limit_1}")
    if within_limit_1:
        print(f"  Physical address = 0x{relocation:04X} + 0x{logical_addr_1:04X} = 0x{physical_addr_1:04X}")
        print(f"  Result: Valid access, physical address is 0x{physical_addr_1:04X}")
    else:
        print(f"  Result: Segmentation Fault! Address exceeds limit.")
    print()

    # Test case 2
    logical_addr_2 = 0x3500
    within_limit_2 = logical_addr_2 < limit

    print(f"Q2: Logical address 0x{logical_addr_2:04X}")
    print(f"  Check: 0x{logical_addr_2:04X} < 0x{limit:04X}? {within_limit_2}")
    if within_limit_2:
        physical_addr_2 = relocation + logical_addr_2
        print(f"  Physical address = 0x{relocation:04X} + 0x{logical_addr_2:04X} = 0x{physical_addr_2:04X}")
    else:
        print(f"  0x{logical_addr_2:04X} >= 0x{limit:04X} -- EXCEEDS LIMIT!")
        print(f"  Result: Segmentation Fault (trap to OS)")
        print(f"  The OS will typically terminate the process or send SIGSEGV.")

    print("\n  Address translation diagram:")
    print(f"  Logical addr -> [< Limit?] --YES--> [+ Relocation] -> Physical addr")
    print(f"                      |")
    print(f"                      NO -> Segmentation Fault (trap)")

    # Additional examples for deeper understanding
    print("\n\n--- Additional address translation examples ---\n")

    test_addrs = [0x0000, 0x1000, 0x2FFF, 0x3000, 0x3001, 0xFFFF]
    print(f"  {'Logical':<12} {'< Limit?':<12} {'Physical':<14} {'Result'}")
    print("  " + "-" * 50)
    for addr in test_addrs:
        valid = addr < limit
        if valid:
            phys = relocation + addr
            print(f"  0x{addr:04X}      {'Yes':<12} 0x{phys:04X}        Valid")
        else:
            print(f"  0x{addr:04X}      {'No':<12} {'N/A':<14} Seg Fault")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Address Translation ===")
    print("=" * 70)
    exercise_1()

    print("\nAll exercises completed!")
