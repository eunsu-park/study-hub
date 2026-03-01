"""
Extract data from PostgreSQL source database to Bronze layer (Parquet).

Adapted from Data_Engineering Lesson 14 §4.1.
Cloud references (S3) replaced with local filesystem.

Usage:
    spark-submit extract_postgres.py --table orders --date 2024-01-15 \
        --output /opt/data-lake/bronze/orders/2024-01-15/
"""

import argparse
import os

from pyspark.sql import SparkSession


def main():
    parser = argparse.ArgumentParser(description="Extract table from PostgreSQL")
    parser.add_argument("--table", required=True, help="Source table name")
    parser.add_argument("--date", required=False, help="Date for incremental extract (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output path for Parquet files")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName(f"Extract {args.table}")
        .getOrCreate()
    )

    # JDBC connection to source database
    jdbc_url = "jdbc:postgresql://{host}:{port}/{db}".format(
        host=os.environ.get("SOURCE_DB_HOST", "source-db"),
        port=os.environ.get("SOURCE_DB_PORT", "5432"),
        db=os.environ.get("SOURCE_DB_NAME", "ecommerce"),
    )
    properties = {
        "user": os.environ.get("SOURCE_DB_USER", "ecommerce"),
        "password": os.environ.get("SOURCE_DB_PASSWORD", "ecommerce_pass"),
        "driver": "org.postgresql.Driver",
    }

    # Incremental extraction when date is specified
    if args.date:
        query = f"""
            (SELECT * FROM {args.table}
             WHERE DATE(updated_at) = '{args.date}') AS t
        """
    else:
        query = args.table

    # Read from PostgreSQL
    df = spark.read.jdbc(url=jdbc_url, table=query, properties=properties)

    # Write to Bronze layer as Parquet
    df.write.mode("overwrite").parquet(args.output)

    print(f"Extracted {df.count()} rows from {args.table} -> {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
