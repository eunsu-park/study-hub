"""
Exercises for Lesson 04: Pandas Data Manipulation
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import pandas as pd


# === Exercise 1: Group Statistics ===
# Problem: Calculate average salary and employee count by department.
def exercise_1():
    """Solution using groupby + agg for multi-metric group summaries."""
    df = pd.DataFrame({
        'department': ['Sales', 'IT', 'IT', 'HR', 'Sales'],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'salary': [50000, 60000, 70000, 55000, 65000]
    })
    print("Employee data:")
    print(df)

    # .agg() with named aggregations produces clean, descriptive column names
    # Each tuple: (source_column, aggregation_function)
    result = df.groupby('department').agg(
        avg_salary=('salary', 'mean'),
        count=('name', 'count')
    )
    print("\nDepartment statistics:")
    print(result)

    # Alternative: multiple aggregations on the same column
    detailed = df.groupby('department')['salary'].agg(['mean', 'std', 'min', 'max'])
    print("\nDetailed salary statistics:")
    print(detailed)


# === Exercise 2: Data Merging ===
# Problem: Join two DataFrames to include department names for employees.
def exercise_2():
    """Solution using pd.merge for relational-style joins."""
    employees = pd.DataFrame({
        'emp_id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'dept_id': [10, 20, 10]
    })

    departments = pd.DataFrame({
        'dept_id': [10, 20],
        'dept_name': ['Sales', 'IT']
    })

    print("Employees:")
    print(employees)
    print("\nDepartments:")
    print(departments)

    # Inner join on dept_id: only rows with matching keys are kept
    # This is similar to SQL: SELECT * FROM employees JOIN departments ON dept_id
    result = pd.merge(employees, departments, on='dept_id')
    print("\nMerged result (inner join):")
    print(result)

    # Demonstrate left join to show what happens when departments are missing
    employees_extra = pd.DataFrame({
        'emp_id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Frank'],
        'dept_id': [10, 20, 10, 30]  # dept_id 30 does not exist
    })
    result_left = pd.merge(employees_extra, departments, on='dept_id', how='left')
    print("\nLeft join (Frank has no matching department):")
    print(result_left)


# === Exercise 3: Pivot Table ===
# Problem: Create a pivot table with sales totals by month and category.
def exercise_3():
    """Solution using pd.pivot_table for cross-tabulation with aggregation."""
    sales = pd.DataFrame({
        'month': ['Jan', 'Jan', 'Feb', 'Feb', 'Jan', 'Feb'],
        'category': ['A', 'B', 'A', 'B', 'A', 'A'],
        'amount': [100, 150, 120, 180, 110, 130]
    })
    print("Sales data:")
    print(sales)

    # pivot_table aggregates values when there are duplicate index/column combinations
    # Here, Jan-A has two entries (100, 110) which get summed to 210
    pivot = pd.pivot_table(
        sales,
        values='amount',
        index='month',
        columns='category',
        aggfunc='sum'
    )
    print("\nPivot table (sum of sales):")
    print(pivot)

    # Add row and column totals using margins parameter
    pivot_with_totals = pd.pivot_table(
        sales,
        values='amount',
        index='month',
        columns='category',
        aggfunc='sum',
        margins=True,
        margins_name='Total'
    )
    print("\nPivot table with totals:")
    print(pivot_with_totals)

    # Reverse operation: melt (unpivot) converts wide format back to long format
    melted = pivot.reset_index().melt(
        id_vars='month',
        var_name='category',
        value_name='total_amount'
    )
    print("\nMelted (unpivoted) back to long format:")
    print(melted)


if __name__ == "__main__":
    print("=== Exercise 1: Group Statistics ===")
    exercise_1()
    print("\n=== Exercise 2: Data Merging ===")
    exercise_2()
    print("\n=== Exercise 3: Pivot Table ===")
    exercise_3()
    print("\nAll exercises completed!")
