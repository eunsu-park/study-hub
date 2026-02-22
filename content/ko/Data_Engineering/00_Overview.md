# 데이터 엔지니어링 (Data Engineering) 학습 가이드

## 소개

데이터 엔지니어링은 데이터를 수집, 저장, 변환, 전달하는 시스템을 설계하고 구축하는 분야입니다. 데이터 파이프라인을 통해 원시 데이터를 분석 가능한 형태로 변환하여 데이터 분석가와 과학자가 활용할 수 있도록 합니다.

**대상 독자**: 데이터 파이프라인 입문자 ~ 중급자 (실무 기초)

---

## 학습 로드맵

```
데이터 엔지니어링 개요 → 데이터 모델링 → ETL/ELT 개념
                                          ↓
    Prefect ← Airflow 심화 ← Airflow 기초 ←┘
       ↓
    Spark 기초 → PySpark DataFrame → Spark 최적화
                                          ↓
    실전 프로젝트 ← 데이터 품질 ← dbt ← Data Lake/Warehouse ← Kafka 스트리밍
```

---

## 파일 목록

| 파일명 | 주제 | 난이도 | 핵심 내용 |
|--------|------|--------|----------|
| [01_Data_Engineering_Overview.md](./01_Data_Engineering_Overview.md) | 데이터 엔지니어링 개요 | ⭐ | 역할, 파이프라인, 배치 vs 스트리밍, 아키텍처 패턴 |
| [02_Data_Modeling_Basics.md](./02_Data_Modeling_Basics.md) | 데이터 모델링 기초 | ⭐⭐ | 차원 모델링, 스타/스노우플레이크 스키마, SCD |
| [03_ETL_vs_ELT.md](./03_ETL_vs_ELT.md) | ETL vs ELT | ⭐⭐ | 전통 ETL, 모던 ELT, 도구 비교, 사용 사례 |
| [04_Apache_Airflow_Basics.md](./04_Apache_Airflow_Basics.md) | Airflow 기초 | ⭐⭐ | 아키텍처, DAG, Task, Operator, 스케줄링 |
| [05_Airflow_Advanced.md](./05_Airflow_Advanced.md) | Airflow 심화 | ⭐⭐⭐ | XCom, 동적 DAG, Sensor, Hook, TaskGroup |
| [06_Prefect_Modern_Orchestration.md](./06_Prefect_Modern_Orchestration.md) | Prefect 모던 오케스트레이션 | ⭐⭐ | Flow, Task, Airflow 비교, Deployment |
| [07_Apache_Spark_Basics.md](./07_Apache_Spark_Basics.md) | Apache Spark 기초 | ⭐⭐⭐ | 아키텍처, RDD, 클러스터 모드, 설치 |
| [08_PySpark_DataFrames.md](./08_PySpark_DataFrames.md) | PySpark DataFrame | ⭐⭐⭐ | SparkSession, DataFrame, 변환, 액션, UDF |
| [09_Spark_SQL_Optimization.md](./09_Spark_SQL_Optimization.md) | Spark SQL 최적화 | ⭐⭐⭐ | Catalyst, 파티셔닝, 캐싱, 조인 전략, 튜닝 |
| [10_Kafka_Streaming.md](./10_Kafka_Streaming.md) | Kafka 스트리밍 | ⭐⭐⭐ | Kafka 개요, Topic, Producer/Consumer, 실시간 처리 |
| [11_Data_Lake_Warehouse.md](./11_Data_Lake_Warehouse.md) | Data Lake와 Warehouse | ⭐⭐ | Lake, Warehouse, Lakehouse, Delta Lake, Iceberg |
| [12_dbt_Transformation.md](./12_dbt_Transformation.md) | dbt 변환 도구 | ⭐⭐⭐ | 모델, 소스, 테스트, 문서화, Jinja 템플릿 |
| [13_Data_Quality_Governance.md](./13_Data_Quality_Governance.md) | 데이터 품질과 거버넌스 | ⭐⭐⭐ | 품질 차원, Great Expectations, 카탈로그, 리니지 |
| [14_Practical_Pipeline_Project.md](./14_Practical_Pipeline_Project.md) | 실전 파이프라인 프로젝트 | ⭐⭐⭐⭐ | E2E 설계, Airflow+Spark+dbt, 품질 검증, 모니터링 |
| [15_Airflow_TaskFlow_API.md](./15_Airflow_TaskFlow_API.md) | Airflow TaskFlow API | ⭐⭐⭐ | @task 데코레이터, 동적 매핑, TaskGroups, 테스트 |
| [16_Kafka_Streams_ksqlDB.md](./16_Kafka_Streams_ksqlDB.md) | Kafka Streams와 ksqlDB | ⭐⭐⭐ | Faust 스트림 처리, 윈도우 집계, ksqlDB 푸시/풀 쿼리 |
| [17_Spark_Structured_Streaming.md](./17_Spark_Structured_Streaming.md) | Spark Structured Streaming | ⭐⭐⭐⭐ | 무한 테이블, 워터마킹, 상태 기반 처리, Kafka 통합 |
| [18_CDC_Debezium.md](./18_CDC_Debezium.md) | CDC와 Debezium | ⭐⭐⭐ | 로그 기반 CDC, Debezium+Kafka Connect, 이벤트 형식, 아웃박스 패턴 |
| [19_Lakehouse_Practical_Patterns.md](./19_Lakehouse_Practical_Patterns.md) | Lakehouse 실전 패턴 | ⭐⭐⭐⭐ | 메달리온 아키텍처, MERGE/SCD2, 컴팩션, 타임 트래블, Iceberg |

**총 레슨**: 19개

---

## 환경 설정

### Docker 기반 환경 (권장)

```bash
# Docker Compose로 전체 환경 구축
# docker-compose.yml 파일 생성 후:

# Airflow 환경 시작
docker compose up -d airflow-webserver airflow-scheduler

# Spark 환경 시작
docker compose up -d spark-master spark-worker

# Kafka 환경 시작
docker compose up -d zookeeper kafka
```

### 개별 도구 설치

```bash
# Apache Airflow
pip install apache-airflow

# PySpark
pip install pyspark

# Kafka Python 클라이언트
pip install confluent-kafka

# dbt
pip install dbt-core dbt-postgres

# Great Expectations
pip install great_expectations

# Prefect
pip install prefect
```

### 권장 버전

| 도구 | 버전 |
|------|------|
| Python | 3.9+ |
| Apache Airflow | 2.7+ |
| Apache Spark | 3.4+ |
| Apache Kafka | 3.5+ |
| dbt-core | 1.6+ |
| Prefect | 2.x |

---

## 학습 순서

### Phase 1: 기초 개념 (01-03)
1. **데이터 엔지니어링 개요**: 역할, 파이프라인 개념
2. **데이터 모델링**: 차원 모델링, 스키마 설계
3. **ETL vs ELT**: 데이터 처리 패턴 이해

### Phase 2: 워크플로우 오케스트레이션 (04-06)
4. **Airflow 기초**: DAG 작성, 스케줄링
5. **Airflow 심화**: XCom, Sensor, 동적 DAG
6. **Prefect**: 모던 오케스트레이션 도구

### Phase 3: 대규모 데이터 처리 (07-09)
7. **Spark 기초**: RDD, 클러스터 아키텍처
8. **PySpark DataFrame**: DataFrame API 활용
9. **Spark 최적화**: 성능 튜닝, 조인 전략

### Phase 4: 스트리밍과 저장소 (10-11)
10. **Kafka**: 실시간 데이터 스트리밍
11. **Data Lake/Warehouse**: 스토리지 아키텍처

### Phase 5: 변환과 품질 (12-13)
12. **dbt**: SQL 기반 데이터 변환
13. **데이터 품질**: Great Expectations, 거버넌스

### Phase 6: 실전 프로젝트 (14)
14. **E2E 파이프라인**: Airflow + Spark + dbt 통합

### Phase 7: 심화 주제 (15-19)
15. **Airflow TaskFlow API**: 모던 Airflow 패턴
16. **Kafka Streams와 ksqlDB**: SQL로 스트림 처리
17. **Spark Structured Streaming**: 실시간 Spark
18. **CDC와 Debezium**: 변경 데이터 캡처
19. **Lakehouse 패턴**: 프로덕션 Delta Lake / Iceberg

---

## 참고 자료

### 공식 문서
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [dbt Documentation](https://docs.getdbt.com/)
- [Prefect Documentation](https://docs.prefect.io/)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)

### 관련 폴더
- [Data_Analysis/](../Data_Analysis/) - NumPy, Pandas 기초
- [PostgreSQL/](../PostgreSQL/) - SQL, 데이터베이스 기초
- [Docker/](../Docker/) - 컨테이너 환경 구축
- [Cloud_Computing/](../Cloud_Computing/) - 클라우드 서비스 (S3, BigQuery)
