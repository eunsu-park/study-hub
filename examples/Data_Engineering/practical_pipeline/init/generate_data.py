"""
Synthetic e-commerce data generator.

Generates customers, products, and orders for the practical pipeline project.
Run standalone or import as a module.

Usage:
    python init/generate_data.py                          # default: 200 customers, 50 products, 2000 orders
    python init/generate_data.py --orders 5000            # custom order count
    python init/generate_data.py --db-url postgresql://ecommerce:ecommerce_pass@localhost:5433/ecommerce
"""

import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# --- Configuration ---

REGIONS = ["North", "South", "East", "West", "Central"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
CATEGORIES = {
    "Electronics": ["Smartphones", "Laptops", "Tablets", "Headphones", "Cameras"],
    "Clothing":    ["Shirts", "Pants", "Shoes", "Jackets", "Accessories"],
    "Home":        ["Furniture", "Kitchen", "Lighting", "Bedding", "Decor"],
    "Books":       ["Fiction", "Non-Fiction", "Technical", "Children", "Comics"],
    "Sports":      ["Fitness", "Outdoor", "Team Sports", "Water Sports", "Cycling"],
}
STATUSES = ["pending", "processing", "shipped", "delivered", "cancelled"]
STATUS_WEIGHTS = [0.05, 0.10, 0.15, 0.60, 0.10]

DEFAULT_DB_URL = "postgresql://ecommerce:ecommerce_pass@localhost:5433/ecommerce"


def generate_customers(n: int = 200) -> pd.DataFrame:
    """Generate synthetic customer data."""
    from faker import Faker
    fake = Faker()

    customers = []
    for i in range(1, n + 1):
        customers.append({
            "customer_id": i,
            "customer_name": fake.name(),
            "email": f"user{i}@{fake.free_email_domain()}",
            "region": random.choice(REGIONS),
            "segment": random.choices(SEGMENTS, weights=[0.5, 0.3, 0.2])[0],
            "created_at": fake.date_time_between(start_date="-2y", end_date="-30d"),
        })
    df = pd.DataFrame(customers)
    df["updated_at"] = df["created_at"]
    return df


def generate_products(n: int = 50) -> pd.DataFrame:
    """Generate synthetic product catalog."""
    products = []
    pid = 1
    for _ in range(n):
        category = random.choice(list(CATEGORIES.keys()))
        sub_category = random.choice(CATEGORIES[category])
        unit_cost = round(random.uniform(5, 500), 2)
        margin = random.uniform(0.15, 0.60)
        products.append({
            "product_id": pid,
            "product_name": f"{sub_category} - Model {pid:03d}",
            "category": category,
            "sub_category": sub_category,
            "unit_cost": unit_cost,
            "unit_price": round(unit_cost * (1 + margin), 2),
        })
        pid += 1
    return pd.DataFrame(products)


def generate_orders(
    n: int = 2000,
    n_customers: int = 200,
    n_products: int = 50,
    products_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate synthetic order data with realistic distribution."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)

    # Skew customer activity: 20% of customers generate 80% of orders
    active_customers = list(range(1, n_customers + 1))
    weights = np.array([1.0] * n_customers)
    top_20 = int(n_customers * 0.2)
    weights[:top_20] = 4.0
    weights = weights / weights.sum()

    orders = []
    for i in range(1, n + 1):
        cid = np.random.choice(active_customers, p=weights)
        pid = random.randint(1, n_products)
        qty = random.choices([1, 2, 3, 5, 10], weights=[0.5, 0.25, 0.15, 0.07, 0.03])[0]

        if products_df is not None:
            price = products_df.loc[products_df["product_id"] == pid, "unit_price"].values[0]
        else:
            price = round(random.uniform(10, 800), 2)

        amount = round(price * qty, 2)
        order_date = start_date + timedelta(days=random.randint(0, 90))

        orders.append({
            "order_id": i,
            "customer_id": int(cid),
            "product_id": pid,
            "order_date": order_date,
            "quantity": qty,
            "amount": amount,
            "status": random.choices(STATUSES, weights=STATUS_WEIGHTS)[0],
            "updated_at": datetime.combine(order_date, datetime.min.time())
                          + timedelta(hours=random.randint(0, 72)),
        })
    return pd.DataFrame(orders)


def load_to_db(db_url: str, customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame):
    """Load generated data into PostgreSQL."""
    engine = create_engine(db_url)

    with engine.begin() as conn:
        # Truncate in correct order (respect FK constraints)
        conn.execute(text("TRUNCATE orders, customers, products RESTART IDENTITY CASCADE"))

    customers.to_sql("customers", engine, if_exists="append", index=False)
    products.to_sql("products", engine, if_exists="append", index=False)
    orders.to_sql("orders", engine, if_exists="append", index=False)

    print(f"Loaded: {len(customers)} customers, {len(products)} products, {len(orders)} orders")


def main():
    parser = argparse.ArgumentParser(description="Generate e-commerce sample data")
    parser.add_argument("--customers", type=int, default=200)
    parser.add_argument("--products", type=int, default=50)
    parser.add_argument("--orders", type=int, default=2000)
    parser.add_argument("--db-url", default=DEFAULT_DB_URL)
    parser.add_argument("--csv", action="store_true", help="Also export to CSV")
    args = parser.parse_args()

    print(f"Generating data: {args.customers} customers, {args.products} products, {args.orders} orders")

    customers = generate_customers(args.customers)
    products = generate_products(args.products)
    orders = generate_orders(args.orders, args.customers, args.products, products)

    load_to_db(args.db_url, customers, products, orders)

    if args.csv:
        customers.to_csv("customers.csv", index=False)
        products.to_csv("products.csv", index=False)
        orders.to_csv("orders.csv", index=False)
        print("Exported CSV files")


if __name__ == "__main__":
    main()
