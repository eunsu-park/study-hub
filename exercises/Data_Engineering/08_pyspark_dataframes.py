"""
Exercise Solutions: Lesson 08 - PySpark DataFrames

Covers:
  - Problem 1: Data Transformation (total and avg sales by month and category)
  - Problem 2: Window Functions (rank employees by salary per department)
  - Problem 3: UDF Writing (extract domain from email)

Note: Pure Python simulation of PySpark DataFrame operations.
"""

from datetime import date
import random
from collections import defaultdict


# ---------------------------------------------------------------------------
# Simulated DataFrame helpers
# ---------------------------------------------------------------------------

class SimpleDataFrame:
    """Minimal DataFrame simulation for demonstrating PySpark-style operations.

    In PySpark you would use:
        df = spark.createDataFrame(data, schema)
        df.groupBy("month", "category").agg(sum("amount"), avg("amount"))
    """
    def __init__(self, data: list[dict], columns: list[str] | None = None):
        self.data = data
        self.columns = columns or (list(data[0].keys()) if data else [])

    def show(self, n: int = 20, header: bool = True) -> None:
        """Display the DataFrame in tabular format."""
        if not self.data:
            print("(empty DataFrame)")
            return
        widths = {}
        for col in self.columns:
            max_width = len(str(col))
            for row in self.data[:n]:
                max_width = max(max_width, len(str(row.get(col, ""))))
            widths[col] = min(max_width + 2, 30)

        if header:
            header_str = "".join(str(col).ljust(widths[col]) for col in self.columns)
            print(f"  {header_str}")
            print(f"  {''.join('-' * widths[col] for col in self.columns)}")
        for i, row in enumerate(self.data[:n]):
            row_str = "".join(str(row.get(col, "")).ljust(widths[col]) for col in self.columns)
            print(f"  {row_str}")
        if len(self.data) > n:
            print(f"  ... ({len(self.data) - n} more rows)")

    def count(self) -> int:
        return len(self.data)


# ---------------------------------------------------------------------------
# Problem 1: Data Transformation
# Calculate total sales and average sales by month and category.
# ---------------------------------------------------------------------------

def generate_sales_data(n: int = 200) -> list[dict]:
    """Generate sample sales data."""
    categories = ["Electronics", "Clothing", "Books", "Food"]
    data = []
    for i in range(n):
        month = random.randint(1, 12)
        data.append({
            "sale_id": i + 1,
            "sale_date": f"2024-{month:02d}-{random.randint(1,28):02d}",
            "month": f"2024-{month:02d}",
            "category": random.choice(categories),
            "amount": round(random.uniform(10, 500), 2),
            "quantity": random.randint(1, 5),
        })
    return data


def problem1_data_transformation():
    """
    PySpark equivalent:

        from pyspark.sql import functions as F

        sales_df = spark.createDataFrame(data)

        result = (
            sales_df
            .withColumn("month", F.date_format("sale_date", "yyyy-MM"))
            .groupBy("month", "category")
            .agg(
                F.sum("amount").alias("total_sales"),
                F.avg("amount").alias("avg_sales"),
                F.count("*").alias("order_count"),
            )
            .orderBy("month", "category")
        )
        result.show()
    """
    sales = generate_sales_data(200)

    # GROUP BY month, category
    agg: dict[tuple[str, str], dict] = {}
    for s in sales:
        key = (s["month"], s["category"])
        if key not in agg:
            agg[key] = {"total_sales": 0.0, "count": 0, "amounts": []}
        agg[key]["total_sales"] += s["amount"]
        agg[key]["count"] += 1
        agg[key]["amounts"].append(s["amount"])

    results = []
    for (month, category), stats in sorted(agg.items()):
        results.append({
            "month": month,
            "category": category,
            "total_sales": round(stats["total_sales"], 2),
            "avg_sales": round(stats["total_sales"] / stats["count"], 2),
            "order_count": stats["count"],
        })

    df = SimpleDataFrame(results, ["month", "category", "total_sales", "avg_sales", "order_count"])
    print(f"\n  Sales by Month and Category ({df.count()} rows):\n")
    df.show(20)
    return results


# ---------------------------------------------------------------------------
# Problem 2: Window Functions
# Rank employees by salary within each department and extract top 3.
# ---------------------------------------------------------------------------

def generate_employee_data() -> list[dict]:
    """Generate sample employee data with known salaries for verification."""
    employees = [
        {"emp_id": 1, "name": "Alice", "department": "Engineering", "salary": 120000},
        {"emp_id": 2, "name": "Bob", "department": "Engineering", "salary": 115000},
        {"emp_id": 3, "name": "Carol", "department": "Engineering", "salary": 130000},
        {"emp_id": 4, "name": "Dave", "department": "Engineering", "salary": 110000},
        {"emp_id": 5, "name": "Eve", "department": "Engineering", "salary": 125000},
        {"emp_id": 6, "name": "Frank", "department": "Sales", "salary": 90000},
        {"emp_id": 7, "name": "Grace", "department": "Sales", "salary": 95000},
        {"emp_id": 8, "name": "Hank", "department": "Sales", "salary": 85000},
        {"emp_id": 9, "name": "Ivy", "department": "Sales", "salary": 92000},
        {"emp_id": 10, "name": "Jack", "department": "Marketing", "salary": 88000},
        {"emp_id": 11, "name": "Kate", "department": "Marketing", "salary": 105000},
        {"emp_id": 12, "name": "Leo", "department": "Marketing", "salary": 98000},
        {"emp_id": 13, "name": "Mia", "department": "Marketing", "salary": 102000},
    ]
    return employees


def problem2_window_functions():
    """
    PySpark equivalent:

        from pyspark.sql import Window
        from pyspark.sql import functions as F

        window_spec = Window.partitionBy("department").orderBy(F.desc("salary"))

        ranked = (
            emp_df
            .withColumn("rank", F.row_number().over(window_spec))
            .withColumn("dept_avg", F.avg("salary").over(
                Window.partitionBy("department")
            ))
        )

        top3 = ranked.filter(F.col("rank") <= 3)
        top3.show()

    Window functions are powerful because they compute aggregates OVER a
    partition without collapsing rows (unlike GROUP BY).
    """
    employees = generate_employee_data()

    # Partition by department, order by salary descending
    by_dept: dict[str, list[dict]] = defaultdict(list)
    for emp in employees:
        by_dept[emp["department"]].append(emp)

    ranked_results = []
    for dept, emps in sorted(by_dept.items()):
        sorted_emps = sorted(emps, key=lambda e: -e["salary"])
        dept_avg = sum(e["salary"] for e in emps) / len(emps)
        for rank, emp in enumerate(sorted_emps, 1):
            ranked_results.append({
                "department": dept,
                "rank": rank,
                "name": emp["name"],
                "salary": emp["salary"],
                "dept_avg_salary": round(dept_avg, 0),
            })

    # All ranked
    print("\n  All Employees Ranked by Salary per Department:\n")
    all_df = SimpleDataFrame(ranked_results, ["department", "rank", "name", "salary", "dept_avg_salary"])
    all_df.show(20)

    # Top 3 per department
    top3 = [r for r in ranked_results if r["rank"] <= 3]
    print(f"\n  Top 3 per Department ({len(top3)} rows):\n")
    top3_df = SimpleDataFrame(top3, ["department", "rank", "name", "salary", "dept_avg_salary"])
    top3_df.show()

    return top3


# ---------------------------------------------------------------------------
# Problem 3: UDF Writing
# Write a UDF that extracts the domain from an email address.
# ---------------------------------------------------------------------------

def extract_email_domain(email: str) -> str | None:
    """UDF: Extract the domain from an email address.

    PySpark equivalent:

        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType

        @udf(returnType=StringType())
        def extract_domain(email):
            if email and '@' in email:
                return email.split('@')[1].lower()
            return None

        df_with_domain = df.withColumn("domain", extract_domain(F.col("email")))

    Why UDFs should be used sparingly:
    - UDFs serialize/deserialize data between JVM and Python (for PySpark).
    - This kills Catalyst optimizer benefits (no predicate pushdown, etc.).
    - Prefer built-in functions when possible:
        df.withColumn("domain", F.split(F.col("email"), "@")[1])
    """
    if email and "@" in email:
        return email.split("@")[1].lower()
    return None


def problem3_udf():
    """Apply the email domain UDF to a sample dataset."""
    users = [
        {"user_id": 1, "name": "Alice", "email": "alice@gmail.com"},
        {"user_id": 2, "name": "Bob", "email": "bob@company.org"},
        {"user_id": 3, "name": "Carol", "email": "carol@university.edu"},
        {"user_id": 4, "name": "Dave", "email": "dave@gmail.com"},
        {"user_id": 5, "name": "Eve", "email": None},
        {"user_id": 6, "name": "Frank", "email": "frank@company.org"},
        {"user_id": 7, "name": "Grace", "email": "grace.lee@startup.io"},
        {"user_id": 8, "name": "Hank", "email": "invalid-email"},
    ]

    # Apply UDF
    for user in users:
        user["domain"] = extract_email_domain(user.get("email"))

    print("\n  Users with Extracted Domains:\n")
    df = SimpleDataFrame(users, ["user_id", "name", "email", "domain"])
    df.show()

    # Domain count aggregation
    domain_counts: dict[str, int] = defaultdict(int)
    for user in users:
        domain = user["domain"]
        if domain:
            domain_counts[domain] += 1

    print("\n  Domain Distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"    {domain:<25} {count}")

    return users


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Data Transformation (Sales by Month & Category)")
    print("=" * 70)
    problem1_data_transformation()

    print()
    print("=" * 70)
    print("Problem 2: Window Functions (Top 3 Employees per Department)")
    print("=" * 70)
    problem2_window_functions()

    print()
    print("=" * 70)
    print("Problem 3: UDF Writing (Extract Email Domain)")
    print("=" * 70)
    problem3_udf()
