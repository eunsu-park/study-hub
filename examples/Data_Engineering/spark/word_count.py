"""
PySpark Word Count Example

A classic Word Count example demonstrating the basic operations of Spark.

Run:
  spark-submit word_count.py
  or
  python word_count.py (local mode)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, regexp_replace, col, desc


def word_count_rdd(spark, text_data):
    """Word Count using RDD API"""
    print("=" * 50)
    print("RDD API Word Count")
    print("=" * 50)

    sc = spark.sparkContext

    # Create RDD
    lines_rdd = sc.parallelize(text_data)

    # Word Count processing
    word_counts = (
        lines_rdd
        .flatMap(lambda line: line.lower().split())  # Split into words
        .map(lambda word: (word, 1))                  # Create (word, 1) pairs
        .reduceByKey(lambda a, b: a + b)              # Sum by word
        .sortBy(lambda x: x[1], ascending=False)      # Sort by frequency
    )

    # Print results
    print("\nTop 10 words (RDD):")
    for word, count in word_counts.take(10):
        print(f"  {word}: {count}")

    return word_counts


def word_count_df(spark, text_data):
    """Word Count using DataFrame API"""
    print("\n" + "=" * 50)
    print("DataFrame API Word Count")
    print("=" * 50)

    # Create DataFrame
    df = spark.createDataFrame([(line,) for line in text_data], ["line"])

    # Word Count processing
    word_counts = (
        df
        .select(explode(split(lower(col("line")), r"\s+")).alias("word"))
        # Remove special characters
        .withColumn("word", regexp_replace(col("word"), r"[^a-z0-9]", ""))
        # Exclude empty strings
        .filter(col("word") != "")
        # Group and count
        .groupBy("word")
        .count()
        # Sort
        .orderBy(desc("count"))
    )

    print("\nTop 10 words (DataFrame):")
    word_counts.show(10, truncate=False)

    return word_counts


def word_count_sql(spark, text_data):
    """Word Count using Spark SQL"""
    print("\n" + "=" * 50)
    print("Spark SQL Word Count")
    print("=" * 50)

    # Create DataFrame and register as view
    df = spark.createDataFrame([(line,) for line in text_data], ["line"])
    df.createOrReplaceTempView("lines")

    # Word Count with SQL
    word_counts = spark.sql("""
        WITH words AS (
            SELECT explode(split(lower(line), '\\\\s+')) AS word
            FROM lines
        ),
        cleaned_words AS (
            SELECT regexp_replace(word, '[^a-z0-9]', '') AS word
            FROM words
            WHERE word != ''
        )
        SELECT
            word,
            COUNT(*) AS count
        FROM cleaned_words
        WHERE word != ''
        GROUP BY word
        ORDER BY count DESC
    """)

    print("\nTop 10 words (SQL):")
    word_counts.show(10, truncate=False)

    return word_counts


def main():
    # Create SparkSession
    spark = SparkSession.builder \
        .appName("Word Count Example") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    # Set log level
    spark.sparkContext.setLogLevel("WARN")

    # Sample text data
    text_data = [
        "Apache Spark is a unified analytics engine for large-scale data processing",
        "Spark provides high-level APIs in Java, Scala, Python and R",
        "Spark also supports a rich set of higher-level tools",
        "Spark SQL for structured data processing",
        "MLlib for machine learning",
        "GraphX for graph processing",
        "Spark Streaming for real-time data processing",
        "Spark is designed to be fast and general-purpose",
        "Spark extends the popular MapReduce model",
        "Spark supports in-memory computing which can improve performance",
        "Data processing with Spark is efficient and scalable",
        "Spark can process data from various sources like HDFS, S3, Kafka",
    ]

    print("Sample Data:")
    for line in text_data[:3]:
        print(f"  {line}")
    print(f"  ... (total {len(text_data)} lines)")

    # Run Word Count using three different approaches
    word_count_rdd(spark, text_data)
    word_count_df(spark, text_data)
    word_count_sql(spark, text_data)

    # Example of saving results
    print("\n" + "=" * 50)
    print("Saving Results")
    print("=" * 50)

    output_path = "/tmp/word_count_output"
    word_counts = word_count_df(spark, text_data)

    # Save in Parquet format
    word_counts.write.mode("overwrite").parquet(f"{output_path}/parquet")
    print(f"Saved to {output_path}/parquet")

    # Save in CSV format (single file)
    word_counts.coalesce(1).write.mode("overwrite").csv(
        f"{output_path}/csv",
        header=True
    )
    print(f"Saved to {output_path}/csv")

    # Print statistics
    print("\n" + "=" * 50)
    print("Statistics")
    print("=" * 50)
    print(f"Total unique words: {word_counts.count()}")
    print(f"Total word occurrences: {word_counts.agg({'count': 'sum'}).collect()[0][0]}")

    # Stop SparkSession
    spark.stop()


if __name__ == "__main__":
    main()
