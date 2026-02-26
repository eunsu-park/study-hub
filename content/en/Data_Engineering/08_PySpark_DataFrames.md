# PySpark DataFrame

## Learning Objectives

After completing this lesson, you will be able to:

1. Create Spark DataFrames from various sources including Python lists, Pandas DataFrames, CSV files, and Parquet files, with explicit schema definitions
2. Apply core DataFrame transformations including select, filter, withColumn, groupBy, agg, and join operations using the PySpark API
3. Use built-in Spark SQL functions and window functions to perform complex column-level computations
4. Handle missing data using fillna, dropna, and imputation strategies in distributed DataFrames
5. Explain how the Catalyst optimizer generates and optimizes logical and physical execution plans
6. Write DataFrames to various output formats and storage systems while configuring partition strategies for performance

---

## Overview

Spark DataFrame is a high-level API that represents distributed data in table format. It provides SQL-like operations and is automatically optimized through the Catalyst optimizer.

---

## 1. SparkSession and DataFrame Creation

### 1.1 SparkSession Initialization

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

# SparkSession is the unified entry point since Spark 2.0 — replaces separate SQLContext/HiveContext
spark = SparkSession.builder \
    .appName("PySpark DataFrame Tutorial") \
    .config("spark.sql.shuffle.partitions", 100) \  # 200 default shuffle partitions often too many for small datasets — tune to ~2-3x core count
    .config("spark.sql.adaptive.enabled", True) \    # AQE auto-adjusts partitions at runtime — essential for varying data volumes
    .getOrCreate()

# Check Spark version
print(f"Spark Version: {spark.version}")
```

### 1.2 DataFrame Creation Methods

```python
# Method 1: From Python list — convenient for testing but requires Spark to
# infer schema by scanning data, which is slow for large datasets
data = [
    ("Alice", 30, "Engineering"),
    ("Bob", 25, "Marketing"),
    ("Charlie", 35, "Engineering"),
]
df1 = spark.createDataFrame(data, ["name", "age", "department"])

# Method 2: Explicit schema avoids the schema inference scan and prevents type
# mismatches (e.g., nulls causing wrong type inference). Always use in production.
# nullable=False adds a NOT NULL constraint that Spark enforces at write time.
schema = StructType([
    StructField("name", StringType(), nullable=False),
    StructField("age", IntegerType(), nullable=True),
    StructField("department", StringType(), nullable=True),
])
df2 = spark.createDataFrame(data, schema)

# Method 3: From list of dictionaries — Spark infers schema from dict keys.
# More readable than tuples but slightly slower due to dict overhead.
dict_data = [
    {"name": "Alice", "age": 30, "department": "Engineering"},
    {"name": "Bob", "age": 25, "department": "Marketing"},
]
df3 = spark.createDataFrame(dict_data)

# Method 4: From Pandas — uses Arrow for efficient transfer if available
# (spark.sql.execution.arrow.pyspark.enabled=true). Only for data that fits
# in Driver memory; for large data, read directly from distributed storage.
import pandas as pd
pdf = pd.DataFrame(data, columns=["name", "age", "department"])
df4 = spark.createDataFrame(pdf)

# Method 5: From RDD — useful when migrating legacy RDD code to DataFrames.
# toDF() infers types from Python objects which can be unreliable; prefer
# createDataFrame(rdd, schema) with explicit schema.
rdd = spark.sparkContext.parallelize(data)
df5 = rdd.toDF(["name", "age", "department"])
```

### 1.3 Reading DataFrames from Files

```python
# CSV file
df_csv = spark.read.csv(
    "data.csv",
    header=True,           # First row as header
    inferSchema=True,      # Scans entire file to detect types — doubles read time.
                           # Use explicit schema in production to avoid this overhead.
    sep=",",               # Delimiter
    nullValue="NA",        # Maps "NA" strings to Spark nulls — prevents treating them as valid strings
    dateFormat="yyyy-MM-dd"
)

# Explicit schema skips the full-file inference scan and guarantees consistent
# types across runs (inferSchema may detect int vs long differently based on data).
schema = StructType([
    StructField("id", IntegerType()),
    StructField("name", StringType()),
    StructField("amount", DoubleType()),
    StructField("date", DateType()),
])
df_csv = spark.read.csv("data.csv", header=True, schema=schema)

# Parquet embeds its schema in file metadata — no inference needed. Columnar format
# enables column pruning (only read requested columns) and predicate pushdown
# (skip entire row groups via min/max statistics).
df_parquet = spark.read.parquet("data.parquet")

# JSON file
df_json = spark.read.json("data.json")

# ORC file
df_orc = spark.read.orc("data.orc")

# JDBC reads run queries on the source DB — by default uses a single partition
# (single JDBC connection), creating a bottleneck. For large tables, add
# partitionColumn/lowerBound/upperBound/numPartitions for parallel reads.
df_jdbc = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "public.users") \
    .option("user", "user") \
    .option("password", "password") \
    .option("driver", "org.postgresql.Driver") \
    .load()

# Delta Lake adds ACID transactions and time travel on top of Parquet —
# schema enforcement and evolution prevent silent data corruption
df_delta = spark.read.format("delta").load("path/to/delta")
```

---

## 2. Basic DataFrame Operations

### 2.1 Data Inspection

```python
# show() triggers execution but limits output — safe for any data size unlike collect()
df.show()           # Top 20 rows
df.show(5)          # Top 5 rows
df.show(truncate=False)  # No column truncation — useful when values are long strings

# printSchema() reads metadata only (no data scan) — always fast
df.printSchema()
df.dtypes           # [(column_name, type), ...]
df.columns          # Column list

# describe() triggers a full scan — computes count, mean, stddev, min, max.
# summary() adds percentiles (25%, 50%, 75%) but is more expensive.
df.describe().show()        # Descriptive statistics
df.summary().show()         # Extended statistics

# count() is an action that scans all partitions — cache the result if called
# multiple times to avoid redundant scans
df.count()

# distinct() requires a shuffle to deduplicate — expensive for high-cardinality columns
df.select("department").distinct().count()

# first/head fetch only from the first partition — much cheaper than full scans
df.first()
df.head(5)

# toPandas() collects ALL data to the Driver as a Pandas DataFrame — will OOM
# if the DataFrame is larger than Driver memory. Use only for small result sets.
pdf = df.toPandas()
```

### 2.2 Column Selection

```python
from pyspark.sql.functions import col, lit

# Single column
df.select("name")
df.select(col("name"))
df.select(df.name)
df.select(df["name"])

# Multiple columns
df.select("name", "age")
df.select(["name", "age"])
df.select(col("name"), col("age"))

# All columns + additional column
df.select("*", lit(1).alias("constant"))

# Drop column
df.drop("department")

# Rename column
df.withColumnRenamed("name", "full_name")

# Rename multiple columns
df.toDF("name_new", "age_new", "dept_new")

# Using alias
df.select(col("name").alias("employee_name"))
```

### 2.3 Filtering

```python
from pyspark.sql.functions import col

# Basic filter
df.filter(col("age") > 30)
df.filter(df.age > 30)
df.filter("age > 30")           # SQL expression
df.where(col("age") > 30)       # Same as filter

# Compound conditions
df.filter((col("age") > 25) & (col("department") == "Engineering"))
df.filter((col("age") < 25) | (col("department") == "Marketing"))
df.filter(~(col("age") > 30))   # NOT

# String filters
df.filter(col("name").startswith("A"))
df.filter(col("name").endswith("e"))
df.filter(col("name").contains("li"))
df.filter(col("name").like("%li%"))
df.filter(col("name").rlike("^[A-C].*"))  # Regex

# IN condition
df.filter(col("department").isin(["Engineering", "Marketing"]))

# NULL handling
df.filter(col("age").isNull())
df.filter(col("age").isNotNull())

# BETWEEN
df.filter(col("age").between(25, 35))
```

---

## 3. Transformations

### 3.1 Adding/Modifying Columns

```python
from pyspark.sql.functions import col, lit, when, concat, upper, lower, length

# withColumn returns a NEW DataFrame — DataFrames are immutable. Each call adds
# a projection to the logical plan; Catalyst fuses consecutive withColumn calls.
df.withColumn("bonus", col("salary") * 0.1)

# lit() wraps a Python scalar as a Spark Column — needed because Spark expressions
# operate on distributed Column objects, not local Python values
df.withColumn("country", lit("USA"))

# Using the same column name replaces it in-place (in the logical plan, not mutating)
df.withColumn("name", upper(col("name")))

# when/otherwise maps to SQL CASE WHEN — evaluated lazily as part of the query plan.
# Conditions are checked in order; first match wins.
df.withColumn("age_group",
    when(col("age") < 30, "Young")
    .when(col("age") < 50, "Middle")
    .otherwise("Senior")
)

# withColumns (Spark 3.3+) applies multiple transformations in one call — cleaner
# than chaining withColumn and may help Catalyst optimize together
df.withColumns({
    "name_upper": upper(col("name")),
    "age_plus_10": col("age") + 10,
})

# String concatenation
df.withColumn("full_info", concat(col("name"), lit(" - "), col("department")))

# cast() changes column type in the logical plan — Spark handles conversion at
# execution time. Invalid casts (e.g., "abc" to int) produce nulls, not errors.
df.withColumn("age_double", col("age").cast("double"))
df.withColumn("age_string", col("age").cast(StringType()))
```

### 3.2 Aggregation Operations

```python
from pyspark.sql.functions import (
    count, sum as _sum, avg, min as _min, max as _max,
    countDistinct, collect_list, collect_set,
    first, last, stddev, variance
)

# Aggregate without groupBy computes over the entire DataFrame — produces a
# single-row result (like SQL SELECT without GROUP BY)
df.agg(
    count("*").alias("total_count"),
    _sum("salary").alias("total_salary"),
    avg("salary").alias("avg_salary"),
    _min("salary").alias("min_salary"),
    _max("salary").alias("max_salary"),
).show()

# groupBy shuffles data by key and applies aggregations within each group.
# Catalyst uses partial aggregation (hash-based) within partitions before
# the shuffle to reduce network transfer.
df.groupBy("department").agg(
    count("*").alias("employee_count"),
    avg("salary").alias("avg_salary"),
    _sum("salary").alias("total_salary"),
    countDistinct("name").alias("unique_names"),  # Requires tracking distinct values — more memory than count
)

# Multi-column grouping creates a cross-product of group keys — can produce
# many groups if cardinalities are high, potentially causing OOM
df.groupBy("department", "age_group").count()

# collect_list/collect_set gather all values into an array per group — WARNING:
# if a group has millions of values, this can OOM the executor. Use only when
# groups are guaranteed to have bounded sizes.
df.groupBy("department").agg(
    collect_list("name").alias("employee_names"),  # Preserves duplicates and insertion order
    collect_set("age").alias("unique_ages"),        # Deduplicates values
)

# pivot() converts row values into columns — specify allowed values explicitly
# to avoid a full pre-scan of the pivot column and to control output schema
df.groupBy("department") \
    .pivot("age_group", ["Young", "Middle", "Senior"]) \
    .agg(count("*"))
```

### 3.3 Sorting

```python
from pyspark.sql.functions import col, asc, desc

# Single column sort
df.orderBy("age")                    # Ascending (default)
df.orderBy(col("age").desc())        # Descending
df.orderBy(desc("age"))

# Multiple column sort
df.orderBy(["department", "age"])
df.orderBy(col("department").asc(), col("age").desc())

# NULL handling
df.orderBy(col("age").asc_nulls_first())
df.orderBy(col("age").desc_nulls_last())

# sort is same as orderBy
df.sort("age")
```

### 3.4 Joins

```python
# Test data
employees = spark.createDataFrame([
    (1, "Alice", 101),
    (2, "Bob", 102),
    (3, "Charlie", 101),
], ["id", "name", "dept_id"])

departments = spark.createDataFrame([
    (101, "Engineering"),
    (102, "Marketing"),
    (103, "Finance"),
], ["dept_id", "dept_name"])

# Inner Join — Spark auto-selects strategy: broadcast if one side < 10MB,
# otherwise sort-merge join. Use explain() to verify the chosen strategy.
employees.join(departments, employees.dept_id == departments.dept_id)
employees.join(departments, "dept_id")  # String form auto-deduplicates the join column

# Left Join — preserves all left rows; use when you need to detect missing
# relationships (e.g., employees without valid departments)
employees.join(departments, "dept_id", "left")

# Right Join
employees.join(departments, "dept_id", "right")

# Full Outer Join — most expensive: must materialize all rows from both sides
employees.join(departments, "dept_id", "full")

# Cross Join produces N*M rows — use with extreme caution on large tables.
# Spark requires explicit crossJoin() to prevent accidental Cartesian products.
employees.crossJoin(departments)

# Semi Join returns left rows that have a match but does NOT include right columns —
# more efficient than inner join + drop because right side is only probed, not materialized
employees.join(departments, "dept_id", "left_semi")

# Anti Join returns left rows with NO match — useful for finding orphan records
# (e.g., orders referencing deleted products)
employees.join(departments, "dept_id", "left_anti")

# Compound conditions — note: this disables the single-column join optimization
# and may produce duplicate join columns in the output
employees.join(
    departments,
    (employees.dept_id == departments.dept_id) & (employees.id > 1),
    "inner"
)
```

---

## 4. Actions

### 4.1 Data Collection

```python
# collect() pulls ALL data to Driver memory — will OOM if data exceeds Driver RAM.
# Prefer take()/show() for inspection, write() for large outputs.
result = df.collect()           # All data (caution: memory)
result = df.take(10)            # Only processes partitions until 10 rows found — cheap
result = df.first()             # First row
result = df.head(5)             # Top 5 rows

# .rdd conversion breaks Catalyst optimization — avoid in production pipelines.
# Use df.select("age").collect() + list comprehension as a DataFrame-native alternative.
ages = df.select("age").rdd.flatMap(lambda x: x).collect()

# toPandas() materializes the full dataset in Driver memory — enable Arrow
# (spark.sql.execution.arrow.pyspark.enabled=true) for ~10x faster transfer
pdf = df.toPandas()             # Small data only

# toLocalIterator() fetches one partition at a time — bounded memory usage
# but much slower than collect() due to sequential partition fetching
for row in df.toLocalIterator():
    print(row)
```

### 4.2 File Writing

```python
# Parquet is the recommended format — columnar storage with compression, schema
# metadata, and predicate pushdown support. Reads are ~10x faster than CSV.
df.write.parquet("output/data.parquet")

# "overwrite" replaces the entire directory — NOT atomic without Delta Lake.
# "append" is safer for incremental writes but can create duplicate data on retries.
df.write.mode("overwrite").parquet("output/data.parquet")
# overwrite: Overwrite existing
# append: Append to existing
# ignore: Ignore if exists
# error: Error if exists (default)

# partitionBy creates a directory hierarchy (year=2024/month=01/) — enables
# partition pruning so queries filtering on these columns skip entire directories.
# Choose low-cardinality columns (date, region) — high cardinality creates too many small files.
df.write.partitionBy("date", "department").parquet("output/partitioned")

# CSV
df.write.csv("output/data.csv", header=True)

# JSON
df.write.json("output/data.json")

# coalesce(1) merges all partitions into one — produces a single output file but
# loses write parallelism. Only use for small outputs; large data should keep
# multiple files and let readers parallelize.
df.coalesce(1).write.csv("output/single_file.csv", header=True)

# JDBC writes use one connection per partition — for many partitions, this can
# overwhelm the database. Use coalesce() or repartition() to control connection count.
df.write.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "public.output_table") \
    .option("user", "user") \
    .option("password", "password") \
    .mode("overwrite") \
    .save()
```

---

## 5. UDF (User Defined Functions)

### 5.1 Basic UDF

```python
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType

# UDFs are a last resort — they disable Catalyst optimization because Spark
# cannot inspect Python function internals. Prefer built-in functions (when/otherwise)
# for this categorization pattern. Use UDFs only for logic that cannot be expressed
# with built-in Spark functions.
def categorize_age(age):
    if age is None:
        return "Unknown"
    elif age < 30:
        return "Young"
    elif age < 50:
        return "Middle"
    else:
        return "Senior"

# Decorator style — cleaner for functions used only as UDFs.
# returnType is required because Spark cannot infer Python function return types.
@udf(returnType=StringType())
def categorize_age_udf(age):
    if age is None:
        return "Unknown"
    elif age < 30:
        return "Young"
    elif age < 50:
        return "Middle"
    else:
        return "Senior"

# Function style — useful when you also need the plain Python function for testing
categorize_udf = udf(categorize_age, StringType())

# Both approaches produce identical execution plans — each row is serialized to
# Python, processed, then serialized back. This ser/de overhead makes UDFs
# 10-100x slower than equivalent built-in functions.
df.withColumn("age_category", categorize_udf(col("age")))
df.withColumn("age_category", categorize_age_udf(col("age")))
```

### 5.2 Pandas UDF (Performance Improvement)

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd

# Pandas UDFs use Apache Arrow for vectorized data transfer between JVM and Python —
# processes data in batches (not row-by-row), achieving 3-100x speedup over regular UDFs.
# Still slower than native Spark functions, but the best option for complex Python logic.
@pandas_udf(StringType())
def categorize_pandas_udf(age_series: pd.Series) -> pd.Series:
    return age_series.apply(
        lambda x: "Unknown" if x is None
        else "Young" if x < 30
        else "Middle" if x < 50
        else "Senior"
    )

# Use
df.withColumn("age_category", categorize_pandas_udf(col("age")))

# GROUPED_MAP receives all rows per group as a Pandas DataFrame — useful for
# complex per-group analytics (regression, custom aggregations) that cannot be
# expressed with built-in Spark aggregation functions.
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Output schema must be declared explicitly because Spark needs it to plan the
# query before executing the Python function
result_schema = StructType([
    StructField("department", StringType()),
    StructField("avg_salary", DoubleType()),
    StructField("employee_count", IntegerType()),
])

@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def analyze_department(pdf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "department": [pdf["department"].iloc[0]],
        "avg_salary": [pdf["salary"].mean()],
        "employee_count": [len(pdf)],
    })

# apply() sends each group's data to Python — entire group must fit in executor
# memory. For very large groups, consider pre-aggregating with built-in functions.
df.groupby("department").apply(analyze_department)
```

### 5.3 Using UDF in SQL

```python
# Register UDF for SQL
spark.udf.register("categorize_age", categorize_age, StringType())

# Use in SQL
df.createOrReplaceTempView("employees")
spark.sql("""
    SELECT name, age, categorize_age(age) as age_category
    FROM employees
""").show()
```

---

## 6. Window Functions

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    row_number, rank, dense_rank,
    lead, lag, sum as _sum, avg,
    first, last, ntile
)

# Window definitions separate the "what to compute" from the "over which rows" —
# reuse the same window spec across multiple calculations to keep code DRY.
# partitionBy determines independent groups; orderBy defines row ordering within each group.
window_dept = Window.partitionBy("department").orderBy("salary")
window_all = Window.orderBy("salary")  # No partition = single global window (expensive on large data)

# row_number gives unique sequential numbers (no ties) — useful for top-N queries.
# rank/dense_rank handle ties differently: rank skips numbers (1,2,2,4), dense_rank doesn't (1,2,2,3).
df.withColumn("row_num", row_number().over(window_dept))
df.withColumn("rank", rank().over(window_dept))
df.withColumn("dense_rank", dense_rank().over(window_dept))
df.withColumn("ntile_4", ntile(4).over(window_dept))  # Splits into 4 roughly equal buckets

# lag/lead access adjacent rows without self-joins — much more efficient for
# computing differences between consecutive records (e.g., day-over-day change)
df.withColumn("prev_salary", lag("salary", 1).over(window_dept))
df.withColumn("next_salary", lead("salary", 1).over(window_dept))

# rowsBetween defines the frame: unboundedPreceding to currentRow = running total.
# Without explicit frame bounds, Spark uses RANGE (value-based) which may include
# ties unexpectedly. Use ROWS (position-based) for deterministic cumulative sums.
window_cumsum = Window.partitionBy("department") \
    .orderBy("date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df.withColumn("cumsum_salary", _sum("salary").over(window_cumsum))

# Moving average with fixed window: current row + 2 preceding rows.
# rowsBetween(-2, 0) means "from 2 rows before to current row" (3-row window).
window_moving = Window.partitionBy("department") \
    .orderBy("date") \
    .rowsBetween(-2, 0)

df.withColumn("moving_avg", avg("salary").over(window_moving))

# first/last within window — useful for forward-fill or back-fill patterns
df.withColumn("first_name", first("name").over(window_dept))
df.withColumn("last_name", last("name").over(window_dept))
```

---

## Practice Problems

### Problem 1: Data Transformation
Calculate total sales and average sales by month and category from sales data.

### Problem 2: Window Functions
Rank employees by salary within each department and extract the top 3 highest paid employees per department.

### Problem 3: UDF Writing
Write a UDF that extracts the domain from an email address and apply it.

---

## Summary

| Operation | Description | Example |
|-----------|-------------|---------|
| **select** | Column selection | `df.select("name", "age")` |
| **filter** | Row filtering | `df.filter(col("age") > 30)` |
| **groupBy** | Grouping | `df.groupBy("dept").agg(...)` |
| **join** | Table join | `df1.join(df2, "key")` |
| **orderBy** | Sorting | `df.orderBy(desc("salary"))` |
| **withColumn** | Add/modify column | `df.withColumn("new", ...)` |

---

## References

- [PySpark DataFrame Guide](https://spark.apache.org/docs/latest/sql-getting-started.html)
- [PySpark Functions](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html)
