-- Exercises for Lesson 04: CRUD Basics
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.

-- Setup: Create and populate the users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    age INTEGER,
    city VARCHAR(50)
);

TRUNCATE TABLE users RESTART IDENTITY;

INSERT INTO users (name, email, age, city) VALUES
('John Kim', 'kim@email.com', 30, 'Seoul'),
('Jane Lee', 'lee@email.com', 25, 'Busan'),
('Michael Park', 'park@email.com', 35, 'Seoul'),
('Sarah Choi', 'choi@email.com', 28, 'Daejeon'),
('Emma Jung', 'jung@email.com', 32, 'Seoul'),
('David Hong', 'hong@email.com', 40, 'Incheon'),
('Kevin Kang', 'kang@email.com', 27, 'Busan'),
('Lisa Son', 'son@email.com', 33, 'Seoul');


-- === Exercise 1: Basic CRUD ===
-- Problem: Insert a new user, query Seoul users, update ages, delete by city.

-- Solution:

-- 1. Add new user — RETURNING shows the inserted row immediately
INSERT INTO users (name, email, age, city)
VALUES ('New User', 'new@email.com', 22, 'Gwangju')
RETURNING *;

-- 2. Query all Seoul users
SELECT * FROM users WHERE city = 'Seoul';

-- 3. Change city to 'Metropolitan' for users age 30+
-- RETURNING lets us verify which rows were updated without a separate SELECT
UPDATE users
SET city = 'Metropolitan'
WHERE age >= 30
RETURNING name, age, city;

-- 4. Delete users in Gwangju
DELETE FROM users
WHERE city = 'Gwangju'
RETURNING *;


-- === Exercise 2: UPSERT ===
-- Problem: Use ON CONFLICT to update existing records or insert new ones.

-- Solution:

-- Attempt to insert 'John Kim' with new age/city.
-- Since kim@email.com already exists, the DO UPDATE clause fires instead,
-- updating only the specified columns while preserving the original name.
INSERT INTO users (name, email, age, city)
VALUES ('John Kim', 'kim@email.com', 31, 'Gyeonggi')
ON CONFLICT (email)
DO UPDATE SET
    age = EXCLUDED.age,
    city = EXCLUDED.city
RETURNING *;

-- Insert a genuinely new user — no conflict, so normal INSERT happens
INSERT INTO users (name, email, age, city)
VALUES ('New Member', 'newuser@email.com', 29, 'Jeju')
ON CONFLICT (email)
DO UPDATE SET age = EXCLUDED.age, city = EXCLUDED.city
RETURNING *;


-- === Exercise 3: Bulk Data Processing ===
-- Problem: Create a backup table and copy data into it.

-- Solution:

-- Create backup table with identical structure but no data.
-- WHERE 1=0 ensures zero rows are copied — only the column definitions.
CREATE TABLE users_backup AS
SELECT * FROM users WHERE 1=0;

-- Copy all current users into the backup
INSERT INTO users_backup
SELECT * FROM users;

-- Also create a filtered backup (Seoul and Busan residents only)
INSERT INTO users_backup
SELECT * FROM users WHERE city IN ('Seoul', 'Busan');

-- Verify backup contents
SELECT COUNT(*) AS total_backup_rows FROM users_backup;

-- Show breakdown by city in the backup
SELECT city, COUNT(*) AS cnt
FROM users_backup
GROUP BY city
ORDER BY cnt DESC;
