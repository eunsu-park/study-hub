"""
Exercises for Lesson 03: Pandas Basics
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import pandas as pd


# === Exercise 1: Data Loading and Exploration ===
# Problem: Create a DataFrame from the given data and check basic information.
def exercise_1():
    """Solution demonstrating DataFrame creation and basic exploration."""
    data = {
        'product': ['Apple', 'Banana', 'Cherry', 'Date'],
        'price': [1000, 500, 2000, 1500],
        'quantity': [50, 100, 30, 45]
    }

    df = pd.DataFrame(data)
    print("DataFrame:")
    print(df)

    # .info() provides column names, dtypes, non-null counts, and memory usage
    print("\n--- DataFrame Info ---")
    print(df.info())

    # .describe() gives summary statistics for numerical columns
    # count, mean, std, min, 25%, 50%, 75%, max
    print("\n--- Descriptive Statistics ---")
    print(df.describe())

    # Average price across all products
    avg_price = df['price'].mean()
    print(f"\nAverage price: {avg_price:.0f}")

    # Additional exploration: total quantity and price range
    print(f"Total quantity: {df['quantity'].sum()}")
    print(f"Price range: {df['price'].min()} - {df['price'].max()}")


# === Exercise 2: Data Selection ===
# Problem: Select product names and quantities where price is 1000 or more.
def exercise_2():
    """Solution demonstrating boolean indexing with .loc for label-based selection."""
    data = {
        'product': ['Apple', 'Banana', 'Cherry', 'Date'],
        'price': [1000, 500, 2000, 1500],
        'quantity': [50, 100, 30, 45]
    }
    df = pd.DataFrame(data)

    # .loc[row_condition, column_list] selects rows matching the condition
    # and only the specified columns
    # This is label-based selection (as opposed to .iloc which is position-based)
    result = df.loc[df['price'] >= 1000, ['product', 'quantity']]
    print("Products with price >= 1000:")
    print(result)

    # Alternative using boolean indexing + column selection
    mask = df['price'] >= 1000
    result_alt = df[mask][['product', 'quantity']]
    print("\nSame result (alternative syntax):")
    print(result_alt)

    # Using .query() for SQL-like filtering
    result_query = df.query('price >= 1000')[['product', 'quantity']]
    print("\nSame result (query method):")
    print(result_query)


# === Exercise 3: Add Column ===
# Problem: Add a total amount column (price * quantity).
def exercise_3():
    """Solution demonstrating vectorized column creation."""
    data = {
        'product': ['Apple', 'Banana', 'Cherry', 'Date'],
        'price': [1000, 500, 2000, 1500],
        'quantity': [50, 100, 30, 45]
    }
    df = pd.DataFrame(data)

    # Vectorized operation: Pandas multiplies element-wise without explicit loops
    # This is both concise and performant compared to iterating over rows
    df['total'] = df['price'] * df['quantity']
    print("DataFrame with total column:")
    print(df)

    # Additional insight: which product generates the most revenue?
    top_product = df.loc[df['total'].idxmax()]
    print(f"\nHighest revenue product: {top_product['product']} "
          f"(total: {top_product['total']:,})")

    # Revenue share per product
    df['revenue_pct'] = (df['total'] / df['total'].sum() * 100).round(1)
    print("\nRevenue distribution:")
    print(df[['product', 'total', 'revenue_pct']])


if __name__ == "__main__":
    print("=== Exercise 1: Data Loading and Exploration ===")
    exercise_1()
    print("\n=== Exercise 2: Data Selection ===")
    exercise_2()
    print("\n=== Exercise 3: Add Column ===")
    exercise_3()
    print("\nAll exercises completed!")
