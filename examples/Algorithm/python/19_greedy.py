"""
Greedy Algorithm
Greedy Algorithms

An algorithmic technique that makes the locally optimal choice at each step to find the global optimum.
"""

from typing import List, Tuple
import heapq


# =============================================================================
# 1. Activity Selection Problem
# =============================================================================

def activity_selection(activities: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Select the maximum number of non-overlapping activities
    activities: [(start, end), ...]
    Greedy strategy: select activities with earliest end time first
    Time Complexity: O(n log n)
    """
    # Sort by end time
    sorted_activities = sorted(activities, key=lambda x: x[1])

    selected = []
    last_end = 0

    for start, end in sorted_activities:
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected


# =============================================================================
# 2. Fractional Knapsack Problem
# =============================================================================

def fractional_knapsack(capacity: int, items: List[Tuple[int, int]]) -> float:
    """
    Fractional knapsack problem
    items: [(value, weight), ...]
    Greedy strategy: select items with highest value-to-weight ratio first
    Time Complexity: O(n log n)
    """
    # Sort by value-to-weight ratio (descending)
    sorted_items = sorted(items, key=lambda x: x[0] / x[1], reverse=True)

    total_value = 0
    remaining = capacity

    for value, weight in sorted_items:
        if remaining >= weight:
            total_value += value
            remaining -= weight
        else:
            # Take a fraction
            total_value += value * (remaining / weight)
            break

    return total_value


# =============================================================================
# 3. Meeting Rooms
# =============================================================================

def min_meeting_rooms(intervals: List[Tuple[int, int]]) -> int:
    """
    Minimum number of meeting rooms required
    Time Complexity: O(n log n)
    """
    if not intervals:
        return 0

    # Separate into start/end events
    events = []
    for start, end in intervals:
        events.append((start, 1))   # Start: +1
        events.append((end, -1))    # End: -1

    events.sort()

    max_rooms = current_rooms = 0
    for _, delta in events:
        current_rooms += delta
        max_rooms = max(max_rooms, current_rooms)

    return max_rooms


# =============================================================================
# 4. Job Scheduling
# =============================================================================

def job_scheduling(jobs: List[Tuple[int, int, int]]) -> Tuple[int, List[int]]:
    """
    Schedule jobs for maximum profit within deadlines
    jobs: [(job_id, deadline, profit), ...]
    Greedy strategy: sort by profit descending, assign to latest available slot
    """
    # Sort by profit descending
    sorted_jobs = sorted(jobs, key=lambda x: x[2], reverse=True)

    max_deadline = max(job[1] for job in jobs)
    slots = [None] * (max_deadline + 1)  # Time slots
    total_profit = 0
    scheduled = []

    for job_id, deadline, profit in sorted_jobs:
        # Find an empty slot from deadline backwards
        for slot in range(deadline, 0, -1):
            if slots[slot] is None:
                slots[slot] = job_id
                total_profit += profit
                scheduled.append(job_id)
                break

    return total_profit, scheduled


# =============================================================================
# 5. Huffman Coding
# =============================================================================

class HuffmanNode:
    def __init__(self, char: str = None, freq: int = 0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_encoding(text: str) -> Tuple[dict, str]:
    """
    Huffman encoding
    Returns: (character->code dictionary, encoded string)
    Time Complexity: O(n log n)
    """
    if not text:
        return {}, ""

    # Calculate frequencies
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    # Build tree using priority queue
    heap = [HuffmanNode(char, f) for char, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged = HuffmanNode(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(heap, merged)

    # Generate codes
    codes = {}

    def generate_codes(node: HuffmanNode, code: str):
        if node is None:
            return

        if node.char is not None:
            codes[node.char] = code if code else "0"
            return

        generate_codes(node.left, code + "0")
        generate_codes(node.right, code + "1")

    if heap:
        generate_codes(heap[0], "")

    # Encode
    encoded = ''.join(codes[char] for char in text)

    return codes, encoded


def huffman_decoding(encoded: str, codes: dict) -> str:
    """Huffman decoding"""
    reverse_codes = {v: k for k, v in codes.items()}
    decoded = []
    current = ""

    for bit in encoded:
        current += bit
        if current in reverse_codes:
            decoded.append(reverse_codes[current])
            current = ""

    return ''.join(decoded)


# =============================================================================
# 6. Coin Change (Greedy)
# =============================================================================

def coin_change_greedy(coins: List[int], amount: int) -> List[int]:
    """
    Coin change (greedy, optimal only for certain coin sets)
    coins: coins sorted in descending order
    Note: DP is required for general coin combinations
    """
    coins = sorted(coins, reverse=True)
    result = []

    for coin in coins:
        while amount >= coin:
            result.append(coin)
            amount -= coin

    return result if amount == 0 else []


# =============================================================================
# 7. Interval Scheduling (Partitioning)
# =============================================================================

def interval_partitioning(intervals: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """
    Partition intervals into minimum number of groups (no overlap within each group)
    Time Complexity: O(n log n)
    """
    if not intervals:
        return []

    # Sort by start time
    sorted_intervals = sorted(enumerate(intervals), key=lambda x: x[1][0])

    # Manage end times of each group using a heap
    groups = []  # [(end_time, group_index)]
    assignment = [[] for _ in range(len(intervals))]

    for idx, (start, end) in sorted_intervals:
        if groups and groups[0][0] <= start:
            # Assign to existing group
            _, group_idx = heapq.heappop(groups)
            assignment[group_idx].append(intervals[idx])
            heapq.heappush(groups, (end, group_idx))
        else:
            # Create new group
            new_group = len(groups)
            assignment.append([intervals[idx]])
            heapq.heappush(groups, (end, new_group))

    return [g for g in assignment if g]


# =============================================================================
# 8. Jump Game
# =============================================================================

def can_jump(nums: List[int]) -> bool:
    """
    Whether the last index is reachable
    nums[i] = maximum jump distance from i
    Time Complexity: O(n)
    """
    max_reach = 0

    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)

    return True


def min_jumps(nums: List[int]) -> int:
    """
    Minimum number of jumps to reach the last index
    Time Complexity: O(n)
    """
    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])

        if i == current_end:
            jumps += 1
            current_end = farthest

            if current_end >= len(nums) - 1:
                break

    return jumps


# =============================================================================
# 9. Gas Station
# =============================================================================

def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """
    Find starting station to complete a circular route
    gas[i] = fuel gained at station i
    cost[i] = fuel cost to travel from i to i+1
    Time Complexity: O(n)
    """
    total_tank = 0
    current_tank = 0
    start = 0

    for i in range(len(gas)):
        gain = gas[i] - cost[i]
        total_tank += gain
        current_tank += gain

        if current_tank < 0:
            start = i + 1
            current_tank = 0

    return start if total_tank >= 0 else -1


# =============================================================================
# 10. Minimum Arrows to Burst Balloons
# =============================================================================

def min_arrows(points: List[List[int]]) -> int:
    """
    Minimum number of arrows to burst all horizontal balloons
    points[i] = [start, end] range
    Time Complexity: O(n log n)
    """
    if not points:
        return 0

    # Sort by end point
    points.sort(key=lambda x: x[1])

    arrows = 1
    arrow_pos = points[0][1]

    for start, end in points[1:]:
        if start > arrow_pos:
            arrows += 1
            arrow_pos = end

    return arrows


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Greedy Algorithm Examples")
    print("=" * 60)

    # 1. Activity Selection
    print("\n[1] Activity Selection Problem")
    activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11)]
    selected = activity_selection(activities)
    print(f"    Activities: {activities}")
    print(f"    Selected: {selected} ({len(selected)} activities)")

    # 2. Fractional Knapsack
    print("\n[2] Fractional Knapsack")
    items = [(60, 10), (100, 20), (120, 30)]  # (value, weight)
    capacity = 50
    value = fractional_knapsack(capacity, items)
    print(f"    Items (value, weight): {items}")
    print(f"    Capacity: {capacity}, Max value: {value}")

    # 3. Meeting Rooms
    print("\n[3] Minimum Meeting Rooms")
    meetings = [(0, 30), (5, 10), (15, 20)]
    rooms = min_meeting_rooms(meetings)
    print(f"    Meetings: {meetings}")
    print(f"    Rooms needed: {rooms}")

    # 4. Job Scheduling
    print("\n[4] Job Scheduling")
    jobs = [(1, 4, 20), (2, 1, 10), (3, 1, 40), (4, 1, 30)]  # (ID, deadline, profit)
    profit, scheduled = job_scheduling(jobs)
    print(f"    Jobs (ID, deadline, profit): {jobs}")
    print(f"    Schedule: {scheduled}, Total profit: {profit}")

    # 5. Huffman Coding
    print("\n[5] Huffman Coding")
    text = "abracadabra"
    codes, encoded = huffman_encoding(text)
    decoded = huffman_decoding(encoded, codes)
    print(f"    Original: '{text}'")
    print(f"    Codes: {codes}")
    print(f"    Encoded: {encoded} ({len(encoded)} bits)")
    print(f"    Original size: {len(text) * 8} bits")
    print(f"    Decoded: '{decoded}'")

    # 6. Coin Change
    print("\n[6] Coin Change")
    coins = [500, 100, 50, 10]
    amount = 1260
    result = coin_change_greedy(coins, amount)
    print(f"    Coins: {coins}, Amount: {amount}")
    print(f"    Result: {result} ({len(result)} coins)")

    # 7. Jump Game
    print("\n[7] Jump Game")
    nums1 = [2, 3, 1, 1, 4]
    nums2 = [3, 2, 1, 0, 4]
    print(f"    {nums1}: reachable = {can_jump(nums1)}, min jumps = {min_jumps(nums1)}")
    print(f"    {nums2}: reachable = {can_jump(nums2)}")

    # 8. Gas Station
    print("\n[8] Gas Station Problem")
    gas = [1, 2, 3, 4, 5]
    cost = [3, 4, 5, 1, 2]
    start = can_complete_circuit(gas, cost)
    print(f"    gas: {gas}, cost: {cost}")
    print(f"    Starting station: {start}")

    # 9. Burst Balloons
    print("\n[9] Minimum Arrows to Burst Balloons")
    points = [[10, 16], [2, 8], [1, 6], [7, 12]]
    arrows = min_arrows(points)
    print(f"    Balloons: {points}")
    print(f"    Arrows needed: {arrows}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
