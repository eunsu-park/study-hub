"""
Daily aggregation: Bronze/Silver → Silver aggregates.

Adapted from Data_Engineering Lesson 14 §4.3.
S3 paths replaced with local data-lake filesystem.

Usage:
    spark-submit aggregate_daily.py --date 2024-01-15 \
        --output /opt/data-lake/silver/daily_aggregates/2024-01-15/
"""

import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, countDistinct, sum


DATA_LAKE = os.environ.get("DATA_LAKE_PATH", "/opt/data-lake")


def main():
    parser = argparse.ArgumentParser(description="Daily sales aggregation")
    parser.add_argument("--date", required=True, help="Aggregation date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output path for aggregated Parquet")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("Daily Aggregation")
        .getOrCreate()
    )

    # Read Bronze/Silver layer data (local filesystem)
    orders = spark.read.parquet(f"{DATA_LAKE}/bronze/orders/{args.date}/")
    customers = spark.read.parquet(f"{DATA_LAKE}/bronze/customers/")
    products = spark.read.parquet(f"{DATA_LAKE}/bronze/products/")

    # Daily sales aggregation by category and region
    daily_sales = (
        orders
        .filter(col("order_date") == args.date)
        .join(products, "product_id")
        .groupBy(
            col("order_date"),
            col("category"),
            col("region"),
        )
        .agg(
            count("order_id").alias("order_count"),
            sum("amount").alias("total_revenue"),
            avg("amount").alias("avg_order_value"),
            countDistinct("customer_id").alias("unique_customers"),
        )
    )

    # Customer segment aggregation
    customer_segments = (
        orders
        .filter(col("order_date") == args.date)
        .join(customers, "customer_id")
        .groupBy("customer_segment")
        .agg(
            count("order_id").alias("orders"),
            sum("amount").alias("revenue"),
        )
    )

    # Save aggregated results
    daily_sales.write.mode("overwrite").parquet(f"{args.output}/daily_sales/")
    customer_segments.write.mode("overwrite").parquet(f"{args.output}/customer_segments/")

    print(f"Aggregated data for {args.date} -> {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
