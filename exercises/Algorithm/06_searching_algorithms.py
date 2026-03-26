"""
Exercises for Lesson 06: Searching Algorithms
Topic: Algorithm

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Integer Square Root ===
# Problem: Return the integer square root of x (floor).
#   Input: 8
#   Output: 2 (floor of 2.828...)
# Approach: Binary search for the maximum mid where mid*mid <= x.

def exercise_1():
    """Solution: Binary search for floor(sqrt(x))."""
    def my_sqrt(x):
        if x < 2:
            return x

        left, right = 1, x // 2
        answer = 1

        while left <= right:
            mid = (left + right) // 2

            if mid * mid <= x:
                answer = mid      # mid might be the answer
                left = mid + 1    # try a larger value
            else:
                right = mid - 1   # mid is too large

        return answer

    # Test cases
    tests = [
        (0, 0),
        (1, 1),
        (4, 2),
        (8, 2),
        (9, 3),
        (15, 3),
        (16, 4),
        (100, 10),
        (2147483647, 46340),  # Large input (near INT_MAX)
    ]

    for x, expected in tests:
        result = my_sqrt(x)
        print(f"sqrt({x}) = {result}")
        assert result == expected, f"Expected {expected}, got {result}"

    print("All Square Root tests passed!")


# === Exercise 2: Find Peak Element ===
# Problem: A peak element is greater than its neighbors. Find any peak index.
#   Input: [1, 2, 3, 1]
#   Output: 2 (element 3 at index 2 is a peak)
# Approach: Binary search - if arr[mid] > arr[mid+1], a peak exists in [left, mid],
#           otherwise in [mid+1, right]. This works because the boundary condition
#           guarantees arr[-1] = arr[n] = -infinity.

def exercise_2():
    """Solution: Binary search for peak in O(log n)."""
    def find_peak_element(arr):
        left, right = 0, len(arr) - 1

        while left < right:
            mid = (left + right) // 2

            if arr[mid] > arr[mid + 1]:
                # Peak is at mid or to the left (arr is descending at mid)
                right = mid
            else:
                # Peak is to the right (arr is ascending at mid)
                left = mid + 1

        return left

    # Test cases
    tests = [
        ([1, 2, 3, 1], [2]),           # peak at index 2
        ([1, 2, 1, 3, 5, 6, 4], [1, 5]),  # peak at index 1 or 5
        ([1], [0]),                     # single element is a peak
        ([1, 2], [1]),                  # ascending pair
        ([2, 1], [0]),                  # descending pair
        ([1, 3, 2, 4, 1], [1, 3]),     # multiple peaks possible
    ]

    for arr, valid_peaks in tests:
        result = find_peak_element(arr)
        print(f"Array: {arr} -> peak index: {result} (value: {arr[result]})")
        assert result in valid_peaks, f"Expected one of {valid_peaks}, got {result}"
        # Verify it's actually a peak
        if len(arr) > 1:
            if result == 0:
                assert arr[0] > arr[1]
            elif result == len(arr) - 1:
                assert arr[-1] > arr[-2]
            else:
                assert arr[result] > arr[result - 1] and arr[result] > arr[result + 1]

    print("All Peak Element tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Integer Square Root ===")
    exercise_1()
    print("\n=== Exercise 2: Find Peak Element ===")
    exercise_2()
    print("\nAll exercises completed!")
