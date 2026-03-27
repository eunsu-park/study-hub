"""
Process raw clickstream JSON into cleaned Bronze-layer Parquet.

Adapted from Data_Engineering Lesson 14 §4.2.

Usage:
    spark-submit process_clickstream.py --date 2024-01-15 \
        --input /opt/data-lake/raw/clickstream/2024-01-15/ \
        --output /opt/data-lake/bronze/clickstream/
"""

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, to_date
from pyspark.sql.types import (
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def main():
    parser = argparse.ArgumentParser(description="Process clickstream JSON")
    parser.add_argument("--date", required=True, help="Processing date (YYYY-MM-DD)")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output Parquet path")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("Process Clickstream")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    # Define expected JSON schema
    schema = StructType([
        StructField("event_id", StringType()),
        StructField("user_id", StringType()),
        StructField("session_id", StringType()),
        StructField("event_type", StringType()),
        StructField("page_url", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("properties", MapType(StringType(), StringType())),
    ])

    # Read raw JSON
    df = spark.read.schema(schema).json(args.input)

    # Clean and transform
    processed_df = (
        df
        .filter(col("event_id").isNotNull())
        .filter(col("user_id").isNotNull())
        .withColumn("event_date", to_date(col("timestamp")))
        .withColumn("event_hour", hour(col("timestamp")))
        .withColumn("product_id", col("properties").getItem("product_id"))
        .dropDuplicates(["event_id"])
        .select(
            "event_id",
            "user_id",
            "session_id",
            "event_type",
            "page_url",
            "product_id",
            "event_date",
            "event_hour",
            "timestamp",
        )
    )

    # Write partitioned Parquet
    (
        processed_df.write
        .mode("overwrite")
        .partitionBy("event_date", "event_hour")
        .parquet(args.output)
    )

    print(f"Processed {processed_df.count()} events -> {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
