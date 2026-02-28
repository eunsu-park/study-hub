-- Exercises for Lesson 01: PostgreSQL Basics
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.

-- === Exercise 1: Verify Environment ===
-- Problem: Check PostgreSQL version, current user, database, time, and server config.

-- Solution:

-- 1. Check PostgreSQL version
SELECT version();

-- 2. Check current user
SELECT current_user;

-- 3. Check current database
SELECT current_database();

-- 4. Check current time
SELECT NOW();

-- 5. Check server configuration
SHOW server_version;
SHOW data_directory;


-- === Exercise 2: Create First Database ===
-- Problem: Create a study database, list databases, switch to it, and verify.

-- Solution:

-- 1. Create study database
CREATE DATABASE study_db;

-- 2. List databases (psql meta-command)
-- \l

-- 3. Switch to new database (psql meta-command)
-- \c study_db

-- 4. Check connection info (psql meta-command)
-- \conninfo


-- === Exercise 3: Create Simple Table ===
-- Problem: Create a "hello" table, insert data, query it, and check structure.

-- Solution:

-- 1. Create table with auto-incrementing ID and timestamp
CREATE TABLE hello (
    id SERIAL PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 2. Insert sample data
INSERT INTO hello (message) VALUES ('Hello, PostgreSQL!');
INSERT INTO hello (message) VALUES ('My first table!');

-- 3. Query all data
SELECT * FROM hello;

-- 4. Check table structure (psql meta-command)
-- \d hello
-- Alternatively, query information_schema:
SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'hello'
ORDER BY ordinal_position;
