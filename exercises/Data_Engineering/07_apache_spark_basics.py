"""
Exercise Solutions: Lesson 07 - Apache Spark Basics

Covers:
  - Problem 1: Basic RDD Operations (sum of squares of even numbers 1-100)
  - Problem 2: Pair RDD (aggregate log counts by error level)

Note: Pure Python simulation of Spark RDD operations.
      No Spark cluster required.
"""

from functools import reduce
from collections import Counter


# ---------------------------------------------------------------------------
# Simulated RDD class
# ---------------------------------------------------------------------------

class SimulatedRDD:
    """Mimics the core Spark RDD API: lazy transformations + eager actions.

    In Spark:
    - Transformations (map, filter, flatMap, reduceByKey) are lazy — they
      build a DAG of operations but don't execute until an action is called.
    - Actions (collect, reduce, count, take) trigger execution.

    This simulation evaluates eagerly for simplicity but preserves the API.
    """
    def __init__(self, data: list):
        self._data = list(data)

    def filter(self, func) -> "SimulatedRDD":
        """Keep only elements where func(element) is True."""
        return SimulatedRDD([x for x in self._data if func(x)])

    def map(self, func) -> "SimulatedRDD":
        """Apply func to each element."""
        return SimulatedRDD([func(x) for x in self._data])

    def flatMap(self, func) -> "SimulatedRDD":
        """Apply func to each element and flatten the results."""
        result = []
        for x in self._data:
            result.extend(func(x))
        return SimulatedRDD(result)

    def reduceByKey(self, func) -> "SimulatedRDD":
        """For pair RDDs: group by key and reduce values.

        Spark partitions data by key and applies the reduce function
        within each partition (combiner) and across partitions (reducer).
        This is more efficient than groupByKey because it reduces data
        movement over the network (shuffle).
        """
        groups: dict = {}
        for key, value in self._data:
            if key in groups:
                groups[key] = func(groups[key], value)
            else:
                groups[key] = value
        return SimulatedRDD(list(groups.items()))

    def sortBy(self, func, ascending: bool = True) -> "SimulatedRDD":
        """Sort elements by the given key function."""
        return SimulatedRDD(sorted(self._data, key=func, reverse=not ascending))

    def reduce(self, func):
        """Reduce all elements to a single value (Action)."""
        return reduce(func, self._data)

    def collect(self) -> list:
        """Return all elements as a Python list (Action)."""
        return self._data

    def count(self) -> int:
        """Return the number of elements (Action)."""
        return len(self._data)

    def take(self, n: int) -> list:
        """Return the first n elements (Action)."""
        return self._data[:n]


class SimulatedSparkContext:
    """Simulates SparkContext for creating RDDs."""
    def parallelize(self, data: list) -> SimulatedRDD:
        return SimulatedRDD(data)

    def textFile(self, path: str) -> SimulatedRDD:
        """Simulate reading a text file. Uses in-memory data."""
        raise NotImplementedError("Use parallelize() with sample data instead")


# ---------------------------------------------------------------------------
# Problem 1: Basic RDD Operations
# Find the sum of squares of even numbers from 1 to 100.
# ---------------------------------------------------------------------------

def problem1_sum_of_squares():
    """
    PySpark equivalent:

        sc = spark.sparkContext
        result = (
            sc.parallelize(range(1, 101))
              .filter(lambda x: x % 2 == 0)
              .map(lambda x: x ** 2)
              .reduce(lambda a, b: a + b)
        )
        print(result)  # 171700

    Execution plan:
    1. parallelize(range(1,101)) -> distribute 100 integers across partitions
    2. filter(even) -> keep 50 numbers: [2, 4, 6, ..., 100]
    3. map(square) -> [4, 16, 36, ..., 10000]
    4. reduce(sum) -> 171700

    Verification: sum of squares of first n even numbers = 2n(n+1)(2n+1)/3
    For n=50: 2*50*51*101/3 = 171700
    """
    sc = SimulatedSparkContext()

    result = (
        sc.parallelize(range(1, 101))
          .filter(lambda x: x % 2 == 0)      # Keep even numbers
          .map(lambda x: x ** 2)              # Square each
          .reduce(lambda a, b: a + b)         # Sum all
    )

    # Verification using closed-form formula
    n = 50  # There are 50 even numbers in [1, 100]
    expected = 2 * n * (n + 1) * (2 * n + 1) // 3

    print(f"  Sum of squares of even numbers from 1 to 100:")
    print(f"    RDD result:  {result}")
    print(f"    Formula:     {expected}")
    print(f"    Match:       {result == expected}")

    # Also show the intermediate results for understanding
    evens = sc.parallelize(range(1, 101)).filter(lambda x: x % 2 == 0).collect()
    print(f"\n  Even numbers: {evens[:10]}... ({len(evens)} total)")
    squares = sc.parallelize(evens).map(lambda x: x ** 2).take(10)
    print(f"  Squares:      {squares}...")

    return result


# ---------------------------------------------------------------------------
# Problem 2: Pair RDD
# Aggregate log counts by error level from a log file.
# ---------------------------------------------------------------------------

def problem2_pair_rdd():
    """
    PySpark equivalent:

        logs = sc.textFile("logs.txt")
        error_counts = (
            logs.map(lambda line: line.split()[1].replace(":", ""))
                .map(lambda level: (level, 1))
                .reduceByKey(lambda a, b: a + b)
                .collect()
        )

    Execution plan:
    1. Read lines from file (or parallelize sample data)
    2. map: extract the log level from each line (e.g., "ERROR")
    3. map: create key-value pairs -> (level, 1)
    4. reduceByKey: sum counts per level (combiners reduce shuffle data)
    5. collect: bring results to driver
    """
    # Simulated log data (what sc.textFile would read)
    sample_logs = [
        "2024-01-01 ERROR: Connection failed",
        "2024-01-01 INFO: Server started",
        "2024-01-01 WARN: High memory usage",
        "2024-01-01 ERROR: Timeout on request /api/users",
        "2024-01-01 INFO: Request processed in 120ms",
        "2024-01-01 ERROR: Database connection lost",
        "2024-01-01 DEBUG: Cache hit for key=user:123",
        "2024-01-01 INFO: Health check passed",
        "2024-01-01 WARN: Disk usage at 85%",
        "2024-01-01 INFO: New connection from 10.0.0.5",
        "2024-01-01 ERROR: NullPointerException in PaymentService",
        "2024-01-01 DEBUG: Query executed in 45ms",
        "2024-01-01 INFO: Batch job completed",
        "2024-01-01 WARN: Response time > 2s for /api/orders",
        "2024-01-01 INFO: Shutting down gracefully",
    ]

    sc = SimulatedSparkContext()

    # Method 1: Using Pair RDD with reduceByKey
    error_counts = (
        sc.parallelize(sample_logs)
          .map(lambda line: line.split()[1].replace(":", ""))  # Extract level
          .map(lambda level: (level, 1))                       # Create pairs
          .reduceByKey(lambda a, b: a + b)                     # Sum by key
          .sortBy(lambda x: -x[1])                             # Sort by count desc
          .collect()
    )

    print("  Log Level Counts (Pair RDD + reduceByKey):")
    print(f"  {'Level':<10} {'Count':>6}")
    print(f"  {'-'*10} {'-'*6}")
    for level, count in error_counts:
        print(f"  {level:<10} {count:>6}")
    print(f"  {'':10} {'-----':>6}")
    print(f"  {'Total':<10} {sum(c for _, c in error_counts):>6}")

    # Verify with Python Counter
    verification = Counter(
        line.split()[1].replace(":", "") for line in sample_logs
    )
    print(f"\n  Verification (Counter): {dict(verification)}")

    return error_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Basic RDD Operations (Sum of Squares of Evens)")
    print("=" * 70)
    problem1_sum_of_squares()

    print()
    print("=" * 70)
    print("Problem 2: Pair RDD (Log Counts by Error Level)")
    print("=" * 70)
    problem2_pair_rdd()
