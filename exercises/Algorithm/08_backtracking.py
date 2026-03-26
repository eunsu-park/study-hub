"""
Exercises for Lesson 08: Backtracking
Topic: Algorithm

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Unique Permutations ===
# Problem: Generate all permutations without duplicates when input has duplicate elements.
#   Input: [1, 1, 2]
#   Output: [[1,1,2], [1,2,1], [2,1,1]]
# Approach: Sort the input, then use backtracking with duplicate skipping.

def exercise_1():
    """Solution: Backtracking with duplicate avoidance."""
    def permute_unique(nums):
        result = []
        nums.sort()  # Sort so duplicates are adjacent

        def backtrack(current, remaining):
            if not remaining:
                result.append(current[:])
                return

            for i in range(len(remaining)):
                # Skip duplicates: if this element equals the previous one
                # at the same recursion level, skip it to avoid duplicate permutations.
                if i > 0 and remaining[i] == remaining[i - 1]:
                    continue

                backtrack(
                    current + [remaining[i]],
                    remaining[:i] + remaining[i + 1:]
                )

        backtrack([], nums)
        return result

    # Test case 1
    result1 = permute_unique([1, 1, 2])
    print(f"permute_unique([1,1,2]):")
    for p in result1:
        print(f"  {p}")
    assert len(result1) == 3
    assert sorted(result1) == sorted([[1, 1, 2], [1, 2, 1], [2, 1, 1]])

    # Test case 2
    result2 = permute_unique([1, 2, 3])
    print(f"\npermute_unique([1,2,3]):")
    for p in result2:
        print(f"  {p}")
    assert len(result2) == 6  # 3! = 6 unique permutations

    # Test case 3
    result3 = permute_unique([2, 2, 2])
    print(f"\npermute_unique([2,2,2]):")
    for p in result3:
        print(f"  {p}")
    assert len(result3) == 1  # all same -> only 1 permutation
    assert result3 == [[2, 2, 2]]

    print("All Unique Permutations tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Unique Permutations ===")
    exercise_1()
    print("\nAll exercises completed!")
