# Introduction to Database Systems

**Previous**: [Overview](./00_Overview.md) | **Next**: [Relational Model](./02_Relational_Model.md)

---

A database system is one of the most important pieces of software infrastructure in modern computing. From banking transactions and airline reservations to social media feeds and scientific research, databases underpin virtually every application that manages persistent, shared data. This lesson introduces the fundamental concepts, architecture, and terminology that form the foundation of database theory.

## Table of Contents

1. [What Is a Database?](#1-what-is-a-database)
2. [Why Not Just Use Files?](#2-why-not-just-use-files)
3. [Database Management Systems](#3-database-management-systems)
4. [Brief History of Database Systems](#4-brief-history-of-database-systems)
5. [Three-Schema Architecture](#5-three-schema-architecture)
6. [Data Independence](#6-data-independence)
7. [ANSI/SPARC Architecture](#7-ansisparc-architecture)
8. [Data Models](#8-data-models)
9. [Database Users and Roles](#9-database-users-and-roles)
10. [Database System Architecture](#10-database-system-architecture)
11. [Exercises](#11-exercises)

---

## 1. What Is a Database?

A **database** is an organized collection of logically related data, designed to meet the information needs of multiple users in an organization. More precisely:

> **Database**: A shared, integrated collection of persistent data that provides a controlled, reliable, and efficient mechanism for definition, construction, manipulation, and sharing of data among various users and applications.

Key properties of a database:

- **Persistent**: Data survives beyond the process that created it
- **Shared**: Multiple users and applications can access data concurrently
- **Integrated**: Data is collected in a unified structure, minimizing redundancy
- **Managed**: Access is controlled by software that enforces rules and constraints

### Database vs. Data

It is important to distinguish between raw **data** and a **database**:

```
Data:       Individual facts (e.g., "Alice", "29", "Engineering")

Information: Data with context and meaning
             ("Alice is 29 years old and works in Engineering")

Database:    Organized collection of related data with
             structure, constraints, and access control
```

### A Simple Example

Consider a university that needs to track students, courses, and enrollments:

```
STUDENT table:
+--------+-----------+------+--------+
| Stu_ID | Name      | Year | GPA    |
+--------+-----------+------+--------+
| S001   | Alice Kim | 3    | 3.85   |
| S002   | Bob Park  | 2    | 3.42   |
| S003   | Carol Lee | 4    | 3.91   |
+--------+-----------+------+--------+

COURSE table:
+-----------+---------------------+---------+
| Course_ID | Title               | Credits |
+-----------+---------------------+---------+
| CS101     | Intro to CS         | 3       |
| CS301     | Database Theory     | 3       |
| MA201     | Linear Algebra      | 4       |
+-----------+---------------------+---------+

ENROLLMENT table:
+--------+-----------+-------+
| Stu_ID | Course_ID | Grade |
+--------+-----------+-------+
| S001   | CS101     | A     |
| S001   | CS301     | A+    |
| S002   | CS101     | B+    |
| S003   | MA201     | A     |
+--------+-----------+-------+
```

This structured representation allows us to answer queries like:
- "What courses is Alice enrolled in?"
- "How many students are taking CS101?"
- "What is the average GPA of students in CS301?"

---

## 2. Why Not Just Use Files?

Before databases existed, applications stored data in flat files. While files are simple, they have fundamental limitations that motivated the development of database systems.

### The File-Based Approach

```
                    ┌──────────────┐
                    │  Application │
                    │   Program 1  │──────► student_records.dat
                    └──────────────┘
                    ┌──────────────┐
                    │  Application │
                    │   Program 2  │──────► course_records.dat
                    └──────────────┘
                    ┌──────────────┐
                    │  Application │
                    │   Program 3  │──────► enrollment.dat
                    └──────────────┘

    Each application manages its own files independently.
```

### Problems with File-Based Systems

| Problem | Description | Example |
|---------|-------------|---------|
| **Data Redundancy** | Same data stored in multiple files | Student name in student file AND enrollment file |
| **Data Inconsistency** | Redundant copies become out of sync | Name changed in one file but not another |
| **Program-Data Dependence** | File format changes require program changes | Adding a field to a record breaks existing code |
| **Limited Data Sharing** | Each application has its own files | Registrar and financial aid cannot share data |
| **No Concurrent Access** | Multiple users cannot safely update simultaneously | Two registrars editing the same record |
| **No Recovery Mechanism** | Data loss from crashes is permanent | Power failure during file write corrupts data |
| **No Security Control** | File-level access only, no fine-grained control | Cannot restrict access to specific fields |
| **No Integrity Enforcement** | No centralized rules for valid data | GPA of 5.0 or negative credits can be stored |

### A Concrete Example of Redundancy and Inconsistency

Suppose the Registrar and Financial Aid offices each maintain their own files:

```
Registrar's file (students.txt):
S001, Alice Kim, Computer Science, 3.85

Financial Aid's file (financial.txt):
S001, Alice Kim, Computer Science, Need-Based

Alice changes her major to Data Science...
The Registrar updates their file, but Financial Aid's file still says
"Computer Science." Now the data is INCONSISTENT.
```

### The Database Approach

A DBMS solves these problems by centralizing data management:

```
    ┌──────────────┐
    │  Application  │───┐
    │   Program 1   │   │
    └──────────────┘   │    ┌──────────┐    ┌──────────────┐
    ┌──────────────┐   ├───►│          │    │              │
    │  Application  │───┤   │   DBMS   │───►│   Database   │
    │   Program 2   │   ├───►│          │    │              │
    └──────────────┘   │    └──────────┘    └──────────────┘
    ┌──────────────┐   │
    │  Application  │───┘
    │   Program 3   │
    └──────────────┘

    All applications go through the DBMS to access a single database.
```

---

## 3. Database Management Systems

A **Database Management System (DBMS)** is the software that sits between users/applications and the stored data. It provides a systematic way to create, retrieve, update, and manage data.

### Core Functions of a DBMS

1. **Data Definition**: Define the structure (schema) of the database
2. **Data Manipulation**: Insert, update, delete, and retrieve data
3. **Data Control**: Manage security, integrity, and concurrent access
4. **Data Administration**: Backup, recovery, performance tuning

### DBMS Components

```
┌─────────────────────────────────────────────────────────┐
│                    DBMS Software                         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │    Query      │  │  Transaction │  │   Storage    │  │
│  │  Processor    │  │   Manager    │  │   Manager    │  │
│  │              │  │              │  │              │  │
│  │ - Parser     │  │ - Scheduler  │  │ - Buffer Mgr │  │
│  │ - Optimizer  │  │ - Lock Mgr   │  │ - File Mgr   │  │
│  │ - Executor   │  │ - Recovery   │  │ - Disk Mgr   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Catalog    │  │  Authorization│  │  Integrity   │  │
│  │   Manager    │  │   Manager    │  │   Manager    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │    Stored Data    │
                │  (Files on Disk)  │
                └──────────────────┘
```

### Advantages of the DBMS Approach

| Advantage | Description |
|-----------|-------------|
| **Data Abstraction** | Users see logical structure, not physical storage details |
| **Minimal Redundancy** | Centralized storage with controlled redundancy |
| **Consistency** | Integrity constraints enforced by DBMS |
| **Data Sharing** | Multiple users/apps access same data |
| **Concurrency Control** | Safe simultaneous access via locking/MVCC |
| **Recovery** | Automatic recovery from failures |
| **Security** | Fine-grained access control (table, row, column level) |
| **Standards Enforcement** | SQL standard, naming conventions, data formats |

### When NOT to Use a DBMS

A DBMS is not always the right choice. Consider simpler alternatives when:

- The database and applications are simple, well-defined, and not expected to change
- There are stringent real-time requirements that a DBMS overhead cannot meet
- There is no need for multi-user access
- The data volume is very small (a few kilobytes)
- The application requires direct, low-level access to storage

Examples: embedded sensor firmware, simple configuration files, small scripts.

---

## 4. Brief History of Database Systems

Understanding how databases evolved helps appreciate why current systems work the way they do.

### Timeline

```
1960s         1970s         1980s         1990s         2000s         2010s+
  │             │             │             │             │             │
  ▼             ▼             ▼             ▼             ▼             ▼
Flat Files  Relational    SQL Standard  Object-       NoSQL         NewSQL
Hierarchical Model        Commercial    Relational    Movement      Distributed
Network     (Codd 1970)   RDBMS Boom    DBMS          (2009+)       HTAP
(IMS, IDMS)  System R      Oracle,       PostgreSQL    MongoDB       CockroachDB
             INGRES        DB2, SQL      Informix      Cassandra     TiDB
                           Server                      Redis         Google Spanner
```

### Era 1: Pre-Relational (1960s)

**Hierarchical Model** (IBM IMS, 1966):
- Data organized as tree structures (parent-child relationships)
- Fast for predefined queries along the hierarchy
- Inflexible: changing the tree structure required rewriting applications

```
        Department
       /          \
  Employee      Project
    |
  Dependent
```

**Network Model** (CODASYL, 1969):
- Generalization of hierarchical model: records can have multiple parents
- More flexible, but complex pointer-based navigation
- Programmer must know the exact access path

```
  Student ──────── Course
     │    \  /  \    │
     │     \/    \   │
     │     /\     \  │
     ▼    /  \     ▼ ▼
  Advisor    Enrollment
```

### Era 2: The Relational Revolution (1970s)

**Edgar F. Codd** published "A Relational Model of Data for Large Shared Data Banks" in 1970, proposing that data be represented as mathematical relations (tables).

Key innovations:
- **Declarative queries**: Specify *what* data you want, not *how* to get it
- **Data independence**: Physical storage changes do not affect applications
- **Mathematical foundation**: Relational algebra and relational calculus
- **Simplicity**: All data represented uniformly as tables

**System R** (IBM, 1974-1979): First implementation of the relational model. Introduced SQL (originally SEQUEL).

**INGRES** (UC Berkeley, 1973-1979): Concurrent independent implementation. Used QUEL query language.

### Era 3: Commercial RDBMS (1980s-1990s)

The relational model proved its worth and commercial systems proliferated:

| System | Year | Notable Features |
|--------|------|------------------|
| Oracle | 1979 | First commercial RDBMS |
| IBM DB2 | 1983 | System R successor |
| SQL Server | 1989 | Microsoft's RDBMS |
| PostgreSQL | 1996 | Open-source, extensible |
| MySQL | 1995 | Open-source, web-friendly |

**SQL standardization**:
- SQL-86: First ANSI standard
- SQL-92: Major revision (subqueries, JOINs, CASE)
- SQL:1999: Recursive queries, triggers, object-relational features
- SQL:2003: XML, window functions, sequences
- SQL:2011: Temporal data
- SQL:2016: JSON support
- SQL:2023: Property graph queries, multi-dimensional arrays

### Era 4: Object-Relational and Beyond (1990s-2000s)

**Object-Relational DBMS** extended the relational model with:
- User-defined types and functions
- Inheritance
- Complex objects (arrays, nested tables)
- PostgreSQL is a prime example

**Object-Oriented DBMS** (OODBMS):
- Store objects directly (no impedance mismatch)
- Never achieved mainstream adoption
- Examples: ObjectStore, db4o

### Era 5: NoSQL Movement (2009+)

Driven by web-scale companies needing to handle massive data volumes:

| Type | Examples | Best For |
|------|----------|----------|
| Key-Value | Redis, DynamoDB | Caching, sessions |
| Document | MongoDB, CouchDB | Semi-structured data |
| Column-Family | Cassandra, HBase | Time-series, analytics |
| Graph | Neo4j, JanusGraph | Connected data, social networks |

Key concepts:
- **CAP Theorem**: Cannot simultaneously guarantee Consistency, Availability, and Partition tolerance
- **BASE**: Basically Available, Soft state, Eventually consistent (vs. ACID)
- **Schema flexibility**: Schema-on-read vs schema-on-write

### Era 6: NewSQL and Distributed SQL (2010s+)

Combining the scalability of NoSQL with the guarantees of relational systems:

| System | Approach |
|--------|----------|
| Google Spanner | Globally distributed, TrueTime |
| CockroachDB | Distributed PostgreSQL-compatible |
| TiDB | MySQL-compatible, HTAP |
| YugabyteDB | PostgreSQL-compatible distributed |
| Vitess | MySQL sharding middleware |

---

## 5. Three-Schema Architecture

The **three-schema architecture** (also called the three-level architecture) separates a database system into three abstraction levels. This separation is the foundation of data independence.

### The Three Levels

```
┌──────────────────────────────────────────────────────┐
│                  External Level                       │
│            (Individual User Views)                    │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  View 1  │  │  View 2  │  │  View 3  │  ...     │
│  │(Students)│  │(Faculty) │  │(Finance) │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│         │            │             │                 │
│         └────────────┼─────────────┘                 │
│                      │                               │
│            External/Conceptual Mapping               │
└──────────────────────┼───────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────┐
│                      ▼                               │
│              Conceptual Level                         │
│         (Community User View)                         │
│                                                      │
│  Describes the WHAT:                                 │
│  - All entities, attributes, relationships           │
│  - Integrity constraints                             │
│  - Security and authorization rules                  │
│                                                      │
│            Conceptual/Internal Mapping               │
└──────────────────────┼───────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────┐
│                      ▼                               │
│               Internal Level                          │
│          (Physical Storage View)                      │
│                                                      │
│  Describes the HOW:                                  │
│  - File organization (heap, sorted, hashed)          │
│  - Index structures (B+ tree, hash index)            │
│  - Record layout and compression                     │
│  - Buffer management policies                        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### External Level (View Level)

The **external level** describes the part of the database that is relevant to a particular user or application. Different users see different views of the same underlying data.

```sql
-- View for the Registrar (sees academic info)
CREATE VIEW registrar_view AS
SELECT s.student_id, s.name, s.major, s.gpa,
       c.course_id, c.title, e.grade
FROM students s
JOIN enrollments e ON s.student_id = e.student_id
JOIN courses c ON e.course_id = c.course_id;

-- View for Financial Aid (sees financial info)
CREATE VIEW financial_aid_view AS
SELECT s.student_id, s.name, s.financial_status,
       s.scholarship_amount, s.loan_balance
FROM students s;

-- View for the Student Portal (limited self-view)
CREATE VIEW student_portal_view AS
SELECT s.name, s.gpa, c.title, e.grade
FROM students s
JOIN enrollments e ON s.student_id = e.student_id
JOIN courses c ON e.course_id = c.course_id
WHERE s.student_id = CURRENT_USER_ID();
```

### Conceptual Level (Logical Level)

The **conceptual level** describes the logical structure of the entire database for all users. It includes:

- All entity types and their attributes
- Relationships between entities
- Integrity constraints
- Security and authorization information

```
Conceptual Schema:

STUDENT(student_id PK, name, major, gpa, financial_status,
        scholarship_amount, loan_balance)

COURSE(course_id PK, title, credits, department)

ENROLLMENT(student_id FK, course_id FK, grade, semester)
    PK(student_id, course_id, semester)

INSTRUCTOR(instructor_id PK, name, department, salary)

TEACHES(instructor_id FK, course_id FK, semester)
    PK(instructor_id, course_id, semester)

Constraints:
  - gpa BETWEEN 0.0 AND 4.0
  - credits > 0
  - grade IN ('A+','A','A-','B+','B','B-','C+','C','C-','D','F')
```

### Internal Level (Physical Level)

The **internal level** describes how data is physically stored on disk:

```
Internal Schema (conceptual representation):

STUDENT table:
  - Storage: Heap file with overflow pages
  - Primary index: B+ tree on student_id (clustered)
  - Secondary index: Hash index on name
  - Record format: Fixed-length (student_id: 8 bytes,
                   name: 50 bytes VARCHAR, major: 30 bytes, ...)
  - Compression: Dictionary encoding on 'major' column
  - Partition: Range partition on student_id

ENROLLMENT table:
  - Storage: Sorted file on (student_id, course_id)
  - Index: Composite B+ tree on (student_id, course_id, semester)
  - Record format: Fixed-length, 32 bytes per record
```

---

## 6. Data Independence

**Data independence** is the capacity to change the schema at one level without having to change the schema at the next higher level. This is the primary benefit of the three-schema architecture.

### Logical Data Independence

The ability to change the **conceptual schema** without changing the external schema or application programs.

```
Example: Adding a column to the STUDENT table

Before:
  STUDENT(student_id, name, major, gpa)

After:
  STUDENT(student_id, name, major, gpa, email, phone)

Impact:
  - External views that don't use email/phone: NO CHANGE needed
  - Applications querying only name and gpa: NO CHANGE needed
  - Only applications that need email/phone must be updated
```

Changes that benefit from logical data independence:
- Adding or removing attributes from a table
- Splitting a table into two (with a view to maintain the original appearance)
- Combining two tables into one
- Adding new relationships or entity types

### Physical Data Independence

The ability to change the **internal schema** without changing the conceptual schema or external views.

```
Example: Changing the index structure

Before:
  STUDENT.name indexed with B+ tree

After:
  STUDENT.name indexed with hash index

Impact:
  - Conceptual schema: NO CHANGE (still STUDENT table with name column)
  - External views: NO CHANGE
  - Application programs: NO CHANGE
  - Only query performance characteristics change
```

Changes that benefit from physical data independence:
- Changing file organization (heap to sorted)
- Adding or removing indexes
- Moving data to different storage devices
- Changing buffer management strategies
- Compressing or partitioning data differently

### Why Data Independence Matters

```
Without Data Independence:          With Data Independence:

  App 1 ──► Physical Storage         App 1 ──► View 1 ─┐
  App 2 ──► Physical Storage         App 2 ──► View 2 ─┤
  App 3 ──► Physical Storage         App 3 ──► View 3 ─┤
                                                        ▼
  Change storage format?             Conceptual Schema
  → Rewrite ALL applications!              │
                                           ▼
                                     Internal Schema

                                     Change storage format?
                                     → Applications unaffected!
```

### Practical Reality

In practice, achieving complete data independence is difficult:

- **Logical independence** is harder to achieve than physical independence
- Performance considerations often leak through abstraction layers
- ORM frameworks provide partial logical independence
- Views provide external-level independence but can have performance implications

---

## 7. ANSI/SPARC Architecture

The **ANSI/SPARC** (American National Standards Institute / Standards Planning and Requirements Committee) architecture, proposed in 1975, formalized the three-schema approach. While no DBMS implements it exactly, it remains the conceptual blueprint for all modern database systems.

### Full Architecture Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                         USERS                                  │
│                                                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │ Casual   │  │ App     │  │ Parametric│  │ DBA            │ │
│  │ User     │  │ Prog    │  │ User     │  │                │ │
│  └────┬─────┘  └────┬────┘  └─────┬────┘  └───────┬────────┘ │
│       │              │             │               │           │
└───────┼──────────────┼─────────────┼───────────────┼──────────┘
        │              │             │               │
        ▼              ▼             ▼               ▼
┌───────────────────────────────────────────────────────────────┐
│                 EXTERNAL LEVEL                                 │
│                                                               │
│  External     External     External     DDL              │
│  Schema 1     Schema 2     Schema 3     Compiler             │
│       │            │            │            │                │
│       └────────────┴────────────┘            │                │
│                    │                         │                │
│          External/Conceptual                 │                │
│              Mapping                         │                │
└────────────────────┼─────────────────────────┼────────────────┘
                     │                         │
                     ▼                         ▼
┌───────────────────────────────────────────────────────────────┐
│                CONCEPTUAL LEVEL                                │
│                                                               │
│            Conceptual Schema                                  │
│            (defined by DBA)                                   │
│                    │                                          │
│          Conceptual/Internal                                  │
│              Mapping                                          │
└────────────────────┼──────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────────┐
│                 INTERNAL LEVEL                                 │
│                                                               │
│            Internal Schema                                    │
│       (storage structures, indexes)                           │
│                    │                                          │
│             Internal/Physical                                 │
│                Mapping                                        │
└────────────────────┼──────────────────────────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │   STORED     │
              │   DATABASE   │
              └──────────────┘
```

### Mappings Between Levels

The architecture defines **mappings** that translate requests and results between levels:

1. **External/Conceptual Mapping**: Translates between a user's view and the conceptual schema
2. **Conceptual/Internal Mapping**: Translates between the logical structure and physical storage
3. **Internal/Physical Mapping**: Translates between DBMS internal structures and OS file system

```python
# Conceptual illustration of mapping (not real DBMS code)

# External Schema: User sees a "student_summary" view
class StudentSummaryView:
    student_name: str      # Maps to STUDENT.first_name + ' ' + STUDENT.last_name
    course_count: int      # Maps to COUNT(*) from ENROLLMENT
    average_grade: float   # Maps to AVG(ENROLLMENT.grade_points)

# Conceptual Schema: Actual tables
class Student:
    student_id: int        # PK
    first_name: str
    last_name: str
    major: str

class Enrollment:
    student_id: int        # FK -> Student
    course_id: str         # FK -> Course
    grade_points: float

# Internal Schema: Physical storage
class StudentStorage:
    file_type: str = "B+ tree clustered on student_id"
    record_size: int = 128  # bytes
    page_size: int = 8192   # bytes
    records_per_page: int = 64
```

### Key Interfaces

| Interface | Between | Purpose |
|-----------|---------|---------|
| **DDL** (Data Definition Language) | DBA and Conceptual Level | Define schema: CREATE TABLE, ALTER TABLE |
| **VDL** (View Definition Language) | Users and External Level | Define views: CREATE VIEW |
| **SDL** (Storage Definition Language) | DBA and Internal Level | Define storage: CREATE INDEX, TABLESPACE |
| **DML** (Data Manipulation Language) | Users and Data | Manipulate data: SELECT, INSERT, UPDATE, DELETE |

---

## 8. Data Models

A **data model** is a collection of concepts for describing the structure, operations, and constraints of a database. Different data models provide different levels of abstraction.

### Categories of Data Models

```
High-level                                              Low-level
(Conceptual)                                           (Physical)
    │                                                      │
    ▼                                                      ▼
┌──────────┐    ┌──────────────┐    ┌──────────────┐   ┌──────────┐
│ ER Model │    │  Relational  │    │  Record-based│   │ Physical │
│ UML      │    │  Model       │    │  Models      │   │ Data     │
│ ORM      │    │              │    │  (Network,   │   │ Model    │
│          │    │              │    │   Hierarch.) │   │          │
└──────────┘    └──────────────┘    └──────────────┘   └──────────┘
  Conceptual      Representational     Implementation     Physical
  (what)          (what + some how)    (how)              (how exactly)
```

### Summary of Major Data Models

| Data Model | Structure | Query Language | Era |
|------------|-----------|---------------|-----|
| **Hierarchical** | Trees | DL/1 | 1960s |
| **Network** | Graphs (CODASYL) | Navigational | 1960s |
| **Relational** | Tables (relations) | SQL | 1970s+ |
| **Entity-Relationship** | Entities & relationships | N/A (design tool) | 1976 |
| **Object-Oriented** | Objects, classes | OQL | 1990s |
| **Object-Relational** | Extended tables | SQL + extensions | 1990s |
| **Document** | JSON/BSON documents | MongoDB Query | 2000s |
| **Key-Value** | Key-value pairs | GET/SET | 2000s |
| **Column-Family** | Column groups | CQL | 2000s |
| **Graph** | Nodes & edges | Cypher, SPARQL | 2000s |

### The Relational Model (Preview)

Since the relational model is the focus of this course, here is a brief preview:

```
Relation (Table):
  - A set of tuples (rows)
  - Each tuple has the same set of attributes (columns)
  - Each attribute has a domain (allowed values)
  - Order of tuples does not matter
  - Order of attributes does not matter
  - No duplicate tuples
  - Each cell contains an atomic value

    ┌─────────────────────────────────────────┐
    │           EMPLOYEE (Relation)            │
    ├──────────┬──────────┬───────┬───────────┤
    │  emp_id  │   name   │  age  │   dept    │   ← Attributes
    ├──────────┼──────────┼───────┼───────────┤
    │  E001    │  Alice   │  29   │  CS       │   ← Tuple 1
    │  E002    │  Bob     │  35   │  EE       │   ← Tuple 2
    │  E003    │  Carol   │  42   │  CS       │   ← Tuple 3
    └──────────┴──────────┴───────┴───────────┘

    Domain(emp_id) = {E001, E002, ..., E999}
    Domain(age) = positive integers
    Domain(dept) = {CS, EE, ME, CE, ...}
```

---

## 9. Database Users and Roles

A database system serves many types of users, each with different needs and levels of technical expertise.

### User Classification

```
┌─────────────────────────────────────────────────────────┐
│                     Database Users                       │
│                                                         │
│  ┌─────────────────────┐  ┌──────────────────────────┐ │
│  │    Actors on the     │  │   Actors Behind the      │ │
│  │      Scene           │  │      Scene               │ │
│  │                     │  │                          │ │
│  │  - Database Admin   │  │  - DBMS Designers        │ │
│  │  - Database Designer│  │  - Tool Developers       │ │
│  │  - End Users        │  │  - System Administrators │ │
│  │  - App Programmers  │  │                          │ │
│  └─────────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Actors on the Scene

| Role | Responsibilities | Tools Used |
|------|-----------------|------------|
| **Database Administrator (DBA)** | Schema definition, storage structure, access control, backup/recovery, performance tuning | DDL, DCL, monitoring tools |
| **Database Designer** | Conceptual and logical design, ER modeling, normalization, view definition | ER tools, UML, CASE tools |
| **Application Programmer** | Write programs that access the database, embed SQL in host language | SQL, ORM, APIs |
| **End Users** | Query and update the database through applications or ad-hoc queries | Forms, reports, SQL |

### Types of End Users

```
End Users
    │
    ├── Casual Users
    │     Access database occasionally
    │     Use ad-hoc queries (SQL or GUI)
    │     Example: Manager running monthly report
    │
    ├── Naive (Parametric) Users
    │     Use pre-written applications repeatedly
    │     Do not write queries
    │     Example: Bank teller, airline reservation agent
    │
    ├── Sophisticated Users
    │     Familiar with DBMS facilities
    │     Write complex queries
    │     Example: Data analyst, scientist
    │
    └── Standalone Users
          Personal database
          Use off-the-shelf software
          Example: Tax preparation software user
```

### The DBA in Detail

The Database Administrator is the central authority for managing the database:

```python
# Typical DBA responsibilities (conceptual)

class DBA:
    """Database Administrator responsibilities"""

    def schema_management(self):
        """Define and modify database schema"""
        # CREATE TABLE, ALTER TABLE, CREATE INDEX
        # Define views for different user groups
        # Manage schema migrations

    def security_management(self):
        """Control access to the database"""
        # GRANT/REVOKE privileges
        # Create roles and assign users
        # Audit access logs

    def performance_tuning(self):
        """Optimize database performance"""
        # Analyze query execution plans
        # Create/drop indexes based on workload
        # Configure buffer pool, cache sizes
        # Partition large tables

    def backup_and_recovery(self):
        """Ensure data durability"""
        # Schedule regular backups (full, incremental)
        # Test recovery procedures
        # Manage transaction logs
        # Handle disaster recovery

    def capacity_planning(self):
        """Plan for growth"""
        # Monitor disk usage trends
        # Estimate future storage needs
        # Plan hardware upgrades
```

---

## 10. Database System Architecture

Modern database systems can be deployed in various architectural configurations.

### Centralized Architecture

```
┌────────────────────────────────────────┐
│           Centralized System           │
│                                        │
│  ┌────────────────────────────────┐   │
│  │         Application +          │   │
│  │         DBMS Software          │   │
│  └────────────────────────────────┘   │
│                  │                     │
│  ┌────────────────────────────────┐   │
│  │          Database              │   │
│  └────────────────────────────────┘   │
│                                        │
│  Users access via dumb terminals       │
└────────────────────────────────────────┘
```

### Client-Server Architecture

**Two-Tier:**
```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Client 1 │  │ Client 2 │  │ Client 3 │
│ (App +   │  │ (App +   │  │ (App +   │
│  UI)     │  │  UI)     │  │  UI)     │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │
     └──────────────┼──────────────┘
                    │  Network (SQL over TCP)
                    ▼
           ┌──────────────┐
           │   Database   │
           │    Server    │
           │  (DBMS +     │
           │   Database)  │
           └──────────────┘
```

**Three-Tier (Web Architecture):**
```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Browser  │  │ Browser  │  │ Mobile   │
│ (Thin    │  │ (Thin    │  │ App      │
│  Client) │  │  Client) │  │          │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │
     └──────────────┼──────────────┘
                    │  HTTP/HTTPS
                    ▼
           ┌──────────────┐
           │ Application  │    Tier 2: Business Logic
           │   Server     │    (Web server, API server)
           │ (Flask/      │
           │  Django/     │
           │  Express)    │
           └──────┬───────┘
                  │  SQL/Protocol
                  ▼
           ┌──────────────┐
           │   Database   │    Tier 3: Data Management
           │    Server    │
           │ (PostgreSQL/ │
           │  MySQL)      │
           └──────────────┘
```

### Distributed Architecture

```
                    ┌──────────────┐
                    │   Client     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Router /   │
                    │   Coordinator│
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐
      │  Node 1  │  │  Node 2  │  │  Node 3  │
      │ (Shard A)│  │ (Shard B)│  │ (Shard C)│
      │ +Replica │  │ +Replica │  │ +Replica │
      └──────────┘  └──────────┘  └──────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    Replication &
                    Consensus Protocol
```

### Cloud Architecture

```
┌─────────────────────────────────────────────┐
│              Cloud Provider                  │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │         Managed Database            │   │
│  │  (RDS, Cloud SQL, Aurora, etc.)     │   │
│  │                                     │   │
│  │  ┌──────────┐  ┌──────────┐        │   │
│  │  │ Primary  │  │ Replica  │        │   │
│  │  │ Instance │──│ Instance │        │   │
│  │  └──────────┘  └──────────┘        │   │
│  │       │                             │   │
│  │  ┌──────────────────────────┐      │   │
│  │  │  Shared Storage Layer    │      │   │
│  │  │  (Distributed, Durable) │      │   │
│  │  └──────────────────────────┘      │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  Features: Auto-backup, scaling,            │
│  monitoring, patching, HA failover          │
└─────────────────────────────────────────────┘
```

---

## 11. Exercises

### Conceptual Questions

**Exercise 1.1**: List five disadvantages of the file-based approach to data management and explain how a DBMS addresses each one.

**Exercise 1.2**: Explain the difference between the three-schema architecture levels. For a university database, give an example of what each level would describe.

**Exercise 1.3**: Define the following terms:
- (a) Data independence
- (b) Data abstraction
- (c) Data definition language
- (d) Data manipulation language
- (e) Schema vs. instance

**Exercise 1.4**: A company currently stores all employee data in spreadsheets. The CEO wants to migrate to a database system. Write a brief memo (5-7 bullet points) explaining the benefits of this migration and potential challenges.

### Analysis Questions

**Exercise 1.5**: Classify each of the following changes as requiring a modification to the (i) external, (ii) conceptual, or (iii) internal schema. Explain whether data independence is preserved.

- (a) A new index is added to the STUDENT table
- (b) A new column `email` is added to the STUDENT table
- (c) The STUDENT table is split into STUDENT_PERSONAL and STUDENT_ACADEMIC
- (d) The database file is moved from HDD to SSD
- (e) A new view is created for the financial aid office

**Exercise 1.6**: For each pair of database models, explain two advantages and two disadvantages of the first model compared to the second:
- (a) Relational vs. Hierarchical
- (b) Document (NoSQL) vs. Relational
- (c) Graph vs. Relational

**Exercise 1.7**: Classify the following users of a hospital database system by their role (DBA, database designer, application programmer, or end user type):
- (a) A person who designs the ER diagram for the patient records system
- (b) A nurse entering patient vital signs through a tablet application
- (c) The IT staff member who performs nightly backups and monitors query performance
- (d) A doctor querying the database to find all patients with a specific diagnosis
- (e) A programmer building the patient portal web application

### Practical Questions

**Exercise 1.8**: Research and compare three modern DBMS (one relational, one document, one graph). For each, identify:
- Primary data model
- Query language
- ACID support (full, partial, or none)
- Typical use case
- Three-schema architecture support level

**Exercise 1.9**: Consider the following scenario:

> An e-commerce company currently uses flat files to store product catalog data (product_catalog.csv), customer data (customers.csv), and order data (orders.csv). They process about 1,000 orders per day and have 50,000 products.

Design a brief plan for migrating this data to a relational database. Include:
- What tables would you create?
- What are the relationships between tables?
- What problems from the file-based approach would this solve?
- What additional features would a DBMS provide?

**Exercise 1.10**: Consider the ANSI/SPARC architecture. Explain what happens at each level when a user executes the following SQL query:

```sql
SELECT name, gpa FROM student_summary WHERE gpa > 3.5;
```

Assume `student_summary` is a view defined as:
```sql
CREATE VIEW student_summary AS
SELECT student_id, first_name || ' ' || last_name AS name, gpa
FROM students;
```

Trace the query through: external level (view resolution) -> conceptual level (logical plan) -> internal level (physical access).

---

**Next**: [Relational Model](./02_Relational_Model.md)
