# PostgreSQL Learning Guide

## Introduction

This folder contains learning materials for the PostgreSQL relational database management system. You can learn step by step from SQL basics to advanced features and operations.

**Target Audience**: SQL beginners ~ intermediate learners, backend developers

---

## Learning Roadmap

```
[Basics]              [Intermediate]           [Advanced]
  │                       │                        │
  ▼                       ▼                        ▼
PostgreSQL Basics ──▶ JOIN ───────────▶ Functions/Procedures
  │                       │                        │
  ▼                       ▼                        ▼
DB Management ────────▶ Aggregation ──────▶ Transactions
  │                       │                        │
  ▼                       ▼                        ▼
Tables/Types ─────────▶ Subqueries/CTE ───▶ Triggers
  │                       │                        │
  ▼                       ▼                        ▼
CRUD Basics ──────────▶ Views & Indexes ───▶ Backup/Operations
  │
  ▼
Conditions & Sorting
```

---

## Prerequisites

- Basic computer skills
- Terminal/command line experience
- (Optional) Basic Docker knowledge

---

## File List

| File | Difficulty | Key Topics |
|------|------------|------------|
| [01_PostgreSQL_Basics.md](./01_PostgreSQL_Basics.md) | ⭐ | Concepts, installation, psql basics |
| [02_Database_Management.md](./02_Database_Management.md) | ⭐ | DB creation/deletion, users, permissions |
| [03_Tables_and_Data_Types.md](./03_Tables_and_Data_Types.md) | ⭐⭐ | CREATE TABLE, data types, constraints |
| [04_CRUD_Basics.md](./04_CRUD_Basics.md) | ⭐ | SELECT, INSERT, UPDATE, DELETE |
| [05_Conditions_and_Sorting.md](./05_Conditions_and_Sorting.md) | ⭐⭐ | WHERE, ORDER BY, LIMIT |
| [06_JOIN.md](./06_JOIN.md) | ⭐⭐ | INNER, LEFT, RIGHT, FULL JOIN |
| [07_Aggregation_and_Grouping.md](./07_Aggregation_and_Grouping.md) | ⭐⭐ | COUNT, SUM, GROUP BY, HAVING |
| [08_Subqueries_and_CTE.md](./08_Subqueries_and_CTE.md) | ⭐⭐⭐ | Subqueries, WITH clause |
| [09_Views_and_Indexes.md](./09_Views_and_Indexes.md) | ⭐⭐⭐ | VIEW, INDEX, EXPLAIN |
| [10_Functions_and_Procedures.md](./10_Functions_and_Procedures.md) | ⭐⭐⭐ | PL/pgSQL, user-defined functions |
| [11_Transactions.md](./11_Transactions.md) | ⭐⭐⭐ | ACID, BEGIN, COMMIT, isolation levels |
| [12_Triggers.md](./12_Triggers.md) | ⭐⭐⭐ | Trigger creation and usage |
| [13_Backup_and_Operations.md](./13_Backup_and_Operations.md) | ⭐⭐⭐⭐ | pg_dump, monitoring, operations |
| [14_JSON_JSONB.md](./14_JSON_JSONB.md) | ⭐⭐⭐ | JSON operators, indexing, schema validation |
| [15_Query_Optimization.md](./15_Query_Optimization.md) | ⭐⭐⭐⭐ | EXPLAIN ANALYZE, index strategies |
| [16_Replication_HA.md](./16_Replication_HA.md) | ⭐⭐⭐⭐⭐ | Streaming replication, logical replication, failover |
| [17_Window_Functions.md](./17_Window_Functions.md) | ⭐⭐⭐ | OVER, ROW_NUMBER, RANK, LEAD/LAG |
| [18_Table_Partitioning.md](./18_Table_Partitioning.md) | ⭐⭐⭐⭐ | Range/List/Hash partitioning |
| [19_Full_Text_Search.md](./19_Full_Text_Search.md) | ⭐⭐⭐ | tsvector/tsquery, GIN index, ranking, pg_trgm |
| [20_Security_Access_Control.md](./20_Security_Access_Control.md) | ⭐⭐⭐⭐ | Roles, RLS, pg_hba.conf, SSL, audit logging |

---

## Recommended Learning Order

### Beginner (SQL Introduction)
1. PostgreSQL Basics → DB Management → Tables/Types → CRUD → Conditions/Sorting

### Intermediate (Data Analysis)
2. JOIN → Aggregation & Grouping → Subqueries/CTE → Views & Indexes

### Advanced (DBA/Backend)
3. Functions/Procedures → Transactions → Triggers → Backup/Operations

### Expert (Specialist)
4. JSON/JSONB → Query Optimization → Window Functions → Partitioning → Full-Text Search → Security → Replication & HA

---

## Practice Environment

### Docker (Recommended)

```bash
# Run PostgreSQL container
docker run --name postgres-study \
  -e POSTGRES_PASSWORD=mypassword \
  -p 5432:5432 \
  -d postgres:16

# Connect with psql
docker exec -it postgres-study psql -U postgres
```

### macOS (Homebrew)

```bash
brew install postgresql@16
brew services start postgresql@16
psql postgres
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql
```

---

## Basic psql Commands

| Command | Description |
|---------|-------------|
| `\l` | List databases |
| `\c dbname` | Connect to database |
| `\dt` | List tables |
| `\d tablename` | Describe table structure |
| `\q` | Quit psql |

---

## Related Resources

- [Docker Learning](../Docker/00_Overview.md) - Run PostgreSQL in containers
- [Official Documentation](https://www.postgresql.org/docs/)
