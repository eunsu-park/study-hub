# Data Engineering Learning Guide

## Introduction

Data Engineering is the field of designing and building systems that collect, store, transform, and deliver data. Through data pipelines, raw data is transformed into analyzable formats for data analysts and scientists to utilize.

**Target Audience**: Data pipeline beginners to intermediate level (practical fundamentals)

---

## Learning Roadmap

```
Data Engineering Overview → Data Modeling → ETL/ELT Concepts
                                          ↓
    Prefect ← Airflow Advanced ← Airflow Basics ←┘
       ↓
    Spark Basics → PySpark DataFrame → Spark Optimization
                                          ↓
    Practical Project ← Data Quality ← dbt ← Data Lake/Warehouse ← Kafka Streaming
```

---

## File List

| Filename | Topic | Difficulty | Key Content |
|--------|------|--------|----------|
| [01_Data_Engineering_Overview.md](./01_Data_Engineering_Overview.md) | Data Engineering Overview | ⭐ | Roles, pipelines, batch vs streaming, architecture patterns |
| [02_Data_Modeling_Basics.md](./02_Data_Modeling_Basics.md) | Data Modeling Basics | ⭐⭐ | Dimensional modeling, star/snowflake schema, SCD |
| [03_ETL_vs_ELT.md](./03_ETL_vs_ELT.md) | ETL vs ELT | ⭐⭐ | Traditional ETL, modern ELT, tool comparison, use cases |
| [04_Apache_Airflow_Basics.md](./04_Apache_Airflow_Basics.md) | Airflow Basics | ⭐⭐ | Architecture, DAG, Task, Operator, scheduling |
| [05_Airflow_Advanced.md](./05_Airflow_Advanced.md) | Airflow Advanced | ⭐⭐⭐ | XCom, dynamic DAG, Sensor, Hook, TaskGroup |
| [06_Prefect_Modern_Orchestration.md](./06_Prefect_Modern_Orchestration.md) | Prefect Modern Orchestration | ⭐⭐ | Flow, Task, Airflow comparison, Deployment |
| [07_Apache_Spark_Basics.md](./07_Apache_Spark_Basics.md) | Apache Spark Basics | ⭐⭐⭐ | Architecture, RDD, cluster modes, installation |
| [08_PySpark_DataFrames.md](./08_PySpark_DataFrames.md) | PySpark DataFrame | ⭐⭐⭐ | SparkSession, DataFrame, transformations, actions, UDF |
| [09_Spark_SQL_Optimization.md](./09_Spark_SQL_Optimization.md) | Spark SQL Optimization | ⭐⭐⭐ | Catalyst, partitioning, caching, join strategies, tuning |
| [10_Kafka_Streaming.md](./10_Kafka_Streaming.md) | Kafka Streaming | ⭐⭐⭐ | Kafka overview, Topic, Producer/Consumer, real-time processing |
| [11_Data_Lake_Warehouse.md](./11_Data_Lake_Warehouse.md) | Data Lake and Warehouse | ⭐⭐ | Lake, Warehouse, Lakehouse, Delta Lake, Iceberg |
| [12_dbt_Transformation.md](./12_dbt_Transformation.md) | dbt Transformation Tool | ⭐⭐⭐ | Models, sources, tests, documentation, Jinja templates |
| [13_Data_Quality_Governance.md](./13_Data_Quality_Governance.md) | Data Quality and Governance | ⭐⭐⭐ | Quality dimensions, Great Expectations, catalog, lineage |
| [14_Practical_Pipeline_Project.md](./14_Practical_Pipeline_Project.md) | Practical Pipeline Project | ⭐⭐⭐⭐ | E2E design, Airflow+Spark+dbt, quality validation, monitoring |
| [15_Airflow_TaskFlow_API.md](./15_Airflow_TaskFlow_API.md) | Airflow TaskFlow API | ⭐⭐⭐ | @task decorator, dynamic mapping, TaskGroups, testing |
| [16_Kafka_Streams_ksqlDB.md](./16_Kafka_Streams_ksqlDB.md) | Kafka Streams & ksqlDB | ⭐⭐⭐ | Faust stream processing, windowed aggregation, ksqlDB push/pull queries |
| [17_Spark_Structured_Streaming.md](./17_Spark_Structured_Streaming.md) | Spark Structured Streaming | ⭐⭐⭐⭐ | Unbounded tables, watermarking, stateful processing, Kafka integration |
| [18_CDC_Debezium.md](./18_CDC_Debezium.md) | CDC with Debezium | ⭐⭐⭐ | Log-based CDC, Debezium+Kafka Connect, event format, outbox pattern |
| [19_Lakehouse_Practical_Patterns.md](./19_Lakehouse_Practical_Patterns.md) | Lakehouse Practical Patterns | ⭐⭐⭐⭐ | Medallion architecture, MERGE/SCD2, compaction, time travel, Iceberg |

**Total Lessons**: 19

---

## Environment Setup

### Docker-based Environment (Recommended)

```bash
# Build complete environment with Docker Compose
# After creating docker-compose.yml file:

# Start Airflow environment
docker compose up -d airflow-webserver airflow-scheduler

# Start Spark environment
docker compose up -d spark-master spark-worker

# Start Kafka environment
docker compose up -d zookeeper kafka
```

### Individual Tool Installation

```bash
# Apache Airflow
pip install apache-airflow

# PySpark
pip install pyspark

# Kafka Python client
pip install confluent-kafka

# dbt
pip install dbt-core dbt-postgres

# Great Expectations
pip install great_expectations

# Prefect
pip install prefect
```

### Recommended Versions

| Tool | Version |
|------|------|
| Python | 3.9+ |
| Apache Airflow | 2.7+ |
| Apache Spark | 3.4+ |
| Apache Kafka | 3.5+ |
| dbt-core | 1.6+ |
| Prefect | 2.x |

---

## Learning Sequence

### Phase 1: Fundamental Concepts (01-03)
1. **Data Engineering Overview**: Roles, pipeline concepts
2. **Data Modeling**: Dimensional modeling, schema design
3. **ETL vs ELT**: Understanding data processing patterns

### Phase 2: Workflow Orchestration (04-06)
4. **Airflow Basics**: Writing DAGs, scheduling
5. **Airflow Advanced**: XCom, Sensor, dynamic DAGs
6. **Prefect**: Modern orchestration tool

### Phase 3: Large-scale Data Processing (07-09)
7. **Spark Basics**: RDD, cluster architecture
8. **PySpark DataFrame**: Using DataFrame API
9. **Spark Optimization**: Performance tuning, join strategies

### Phase 4: Streaming and Storage (10-11)
10. **Kafka**: Real-time data streaming
11. **Data Lake/Warehouse**: Storage architecture

### Phase 5: Transformation and Quality (12-13)
12. **dbt**: SQL-based data transformation
13. **Data Quality**: Great Expectations, governance

### Phase 6: Practical Project (14)
14. **E2E Pipeline**: Airflow + Spark + dbt integration

### Phase 7: Advanced Topics (15-19)
15. **Airflow TaskFlow API**: Modern Airflow patterns
16. **Kafka Streams & ksqlDB**: Stream processing with SQL
17. **Spark Structured Streaming**: Real-time Spark
18. **CDC with Debezium**: Change data capture
19. **Lakehouse Patterns**: Production Delta Lake / Iceberg

---

## References

### Official Documentation
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [dbt Documentation](https://docs.getdbt.com/)
- [Prefect Documentation](https://docs.prefect.io/)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)

### Related Folders
- [Data_Analysis/](../Data_Analysis/) - NumPy, Pandas basics
- [PostgreSQL/](../PostgreSQL/) - SQL, database fundamentals
- [Docker/](../Docker/) - Container environment setup
- [Cloud_Computing/](../Cloud_Computing/) - Cloud services (S3, BigQuery)
