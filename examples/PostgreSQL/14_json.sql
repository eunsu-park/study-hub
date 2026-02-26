-- ============================================================================
-- PostgreSQL JSON/JSONB Operations
-- ============================================================================
-- Demonstrates:
--   - JSON vs JSONB data types
--   - JSON creation and insertion
--   - Operators: ->, ->>, #>, @>, ?
--   - JSONB indexing (GIN)
--   - JSON aggregation functions
--   - JSON path queries (PostgreSQL 12+)
--   - JSON manipulation (set, delete, merge)
--
-- Prerequisites: PostgreSQL 12+ (14+ recommended for JSON path subscripting)
-- Usage: psql -U postgres -d your_database -f 14_json.sql
-- ============================================================================

-- Clean up
DROP TABLE IF EXISTS api_events CASCADE;
DROP TABLE IF EXISTS user_profiles CASCADE;
DROP TABLE IF EXISTS product_catalog CASCADE;

-- ============================================================================
-- 1. JSON vs JSONB
-- ============================================================================

-- JSON: stores exact text, preserves whitespace and key order
-- JSONB: binary format, faster operations, supports indexing
-- Rule of thumb: always use JSONB unless you need exact text preservation

-- ============================================================================
-- 2. Basic JSONB Table
-- ============================================================================

CREATE TABLE user_profiles (
    user_id SERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    profile JSONB NOT NULL DEFAULT '{}',
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO user_profiles (username, profile, settings) VALUES
(
    'alice',
    '{
        "name": "Alice Johnson",
        "age": 30,
        "email": "alice@example.com",
        "address": {
            "city": "San Francisco",
            "state": "CA",
            "zip": "94102"
        },
        "skills": ["Python", "PostgreSQL", "Docker"],
        "experience_years": 8
    }',
    '{"theme": "dark", "notifications": true, "language": "en"}'
),
(
    'bob',
    '{
        "name": "Bob Smith",
        "age": 25,
        "email": "bob@example.com",
        "address": {
            "city": "New York",
            "state": "NY",
            "zip": "10001"
        },
        "skills": ["JavaScript", "React", "Node.js"],
        "experience_years": 3
    }',
    '{"theme": "light", "notifications": false, "language": "en"}'
),
(
    'charlie',
    '{
        "name": "Charlie Brown",
        "age": 35,
        "email": "charlie@example.com",
        "address": {
            "city": "Chicago",
            "state": "IL",
            "zip": "60601"
        },
        "skills": ["Python", "Machine Learning", "TensorFlow", "PostgreSQL"],
        "experience_years": 12
    }',
    '{"theme": "dark", "notifications": true, "language": "ko"}'
);

-- ============================================================================
-- 3. JSON Access Operators
-- ============================================================================

-- -> returns JSON object/array element
-- ->> returns TEXT value
-- #> returns JSON at path
-- #>> returns TEXT at path

-- Get name as JSON vs TEXT
SELECT
    profile -> 'name' AS name_json,       -- "Alice Johnson" (with quotes)
    profile ->> 'name' AS name_text       -- Alice Johnson (no quotes)
FROM user_profiles
WHERE username = 'alice';

-- Nested access
SELECT
    profile -> 'address' -> 'city' AS city_json,
    profile -> 'address' ->> 'city' AS city_text,
    profile #>> '{address,city}' AS city_path
FROM user_profiles;

-- Array access (0-indexed)
SELECT
    username,
    profile -> 'skills' -> 0 AS first_skill,
    profile -> 'skills' -> -1 AS last_skill
FROM user_profiles;

-- ============================================================================
-- 4. JSONB Containment and Existence
-- ============================================================================

-- @> containment: left contains right?
SELECT username, profile ->> 'name' AS name
FROM user_profiles
WHERE profile @> '{"address": {"state": "CA"}}';

-- ? key existence: does the key exist?
SELECT username
FROM user_profiles
WHERE profile ? 'experience_years';

-- ?| any key exists
SELECT username
FROM user_profiles
WHERE profile -> 'skills' ?| ARRAY['Python', 'Rust'];

-- ?& all keys exist (for objects)
SELECT username
FROM user_profiles
WHERE settings ?& ARRAY['theme', 'notifications'];

-- ============================================================================
-- 5. JSONB Indexing (GIN)
-- ============================================================================

-- Why: Three different index strategies for JSONB depending on query patterns:
-- 1. Default GIN: supports all JSONB operators (@>, ?, ?|, ?&) but is larger
-- 2. jsonb_path_ops: ~3x smaller, but only supports @> containment queries
-- 3. Expression index: most efficient for queries on a single known key path,
--    because it indexes only that one value instead of the entire document.
CREATE INDEX idx_profile_gin ON user_profiles USING GIN (profile);

CREATE INDEX idx_profile_path ON user_profiles
    USING GIN (profile jsonb_path_ops);

CREATE INDEX idx_profile_city ON user_profiles
    ((profile -> 'address' ->> 'city'));

-- This query uses the expression index:
SELECT username FROM user_profiles
WHERE profile -> 'address' ->> 'city' = 'San Francisco';

-- ============================================================================
-- 6. JSONB Modification
-- ============================================================================

-- Why: The || operator performs a shallow merge — top-level keys from the right
-- operand overwrite the left. This is the simplest way to add or update a
-- top-level key. For nested updates, jsonb_set is needed instead because ||
-- would replace the entire nested object.
UPDATE user_profiles
SET profile = profile || '{"verified": true}'
WHERE username = 'alice';

-- Set nested value using jsonb_set
UPDATE user_profiles
SET profile = jsonb_set(profile, '{address,country}', '"US"')
WHERE username = 'alice';

-- Delete a key using -
UPDATE user_profiles
SET settings = settings - 'language'
WHERE username = 'bob';

-- Delete nested key using #-
UPDATE user_profiles
SET profile = profile #- '{address,zip}'
WHERE username = 'charlie';

-- Add to array
UPDATE user_profiles
SET profile = jsonb_set(
    profile,
    '{skills}',
    (profile -> 'skills') || '"Kubernetes"'
)
WHERE username = 'alice';

-- Check results
SELECT username, profile -> 'skills' AS skills,
       profile -> 'address' AS address
FROM user_profiles;

-- ============================================================================
-- 7. JSON Aggregation
-- ============================================================================

-- Build JSON from query results
SELECT jsonb_agg(
    jsonb_build_object(
        'username', username,
        'name', profile ->> 'name',
        'city', profile -> 'address' ->> 'city'
    )
) AS users_json
FROM user_profiles;

-- Object aggregation (key-value pairs)
SELECT jsonb_object_agg(
    username,
    profile ->> 'name'
) AS username_to_name
FROM user_profiles;

-- Array aggregation with filtering
SELECT jsonb_agg(DISTINCT skill)
FROM user_profiles,
     jsonb_array_elements_text(profile -> 'skills') AS skill
ORDER BY skill;

-- ============================================================================
-- 8. JSON Path Queries (PostgreSQL 12+)
-- ============================================================================

-- jsonb_path_query: extract values using SQL/JSON path
SELECT jsonb_path_query(profile, '$.skills[*]') AS skill
FROM user_profiles
WHERE username = 'alice';

-- jsonb_path_exists: check if path matches
SELECT username
FROM user_profiles
WHERE jsonb_path_exists(profile, '$.skills[*] ? (@ == "Python")');

-- jsonb_path_query with filter
SELECT
    username,
    jsonb_path_query_first(profile, '$.address.city') AS city
FROM user_profiles
WHERE jsonb_path_exists(profile, '$.experience_years ? (@ > 5)');

-- ============================================================================
-- 9. Expanding JSON: Arrays and Objects
-- ============================================================================

-- Why: jsonb_array_elements_text is a set-returning function that "unnests" a
-- JSON array into one row per element. This is how you join relational data
-- with JSON arrays — essential for filtering, grouping, or aggregating
-- individual array elements using standard SQL.
SELECT
    username,
    skill
FROM user_profiles,
     jsonb_array_elements_text(profile -> 'skills') AS skill
ORDER BY username, skill;

-- Expand object key-value pairs
SELECT
    username,
    key,
    value
FROM user_profiles,
     jsonb_each(settings) AS kv(key, value);

-- Count skills per user
SELECT
    username,
    jsonb_array_length(profile -> 'skills') AS skill_count
FROM user_profiles
ORDER BY skill_count DESC;

-- ============================================================================
-- 10. Practical Example: Event Store
-- ============================================================================

-- Why: An event store with JSONB payload is a practical pattern for logging
-- heterogeneous events (login, order, logout) in one table. Each event type
-- has different fields, making a rigid schema impractical. The B-tree index on
-- event_type handles type-based filtering, while GIN on payload enables queries
-- into the flexible event data without knowing the schema upfront.
CREATE TABLE api_events (
    event_id SERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_type ON api_events (event_type);
CREATE INDEX idx_events_payload ON api_events USING GIN (payload);

INSERT INTO api_events (event_type, payload, metadata) VALUES
    ('user.login', '{"user_id": 1, "ip": "192.168.1.1", "method": "password"}',
     '{"source": "web", "version": "2.0"}'),
    ('user.login', '{"user_id": 2, "ip": "10.0.0.5", "method": "oauth"}',
     '{"source": "mobile", "version": "1.5"}'),
    ('order.created', '{"order_id": 101, "user_id": 1, "total": 99.99, "items": 3}',
     '{"source": "web"}'),
    ('order.created', '{"order_id": 102, "user_id": 2, "total": 249.99, "items": 1}',
     '{"source": "api"}'),
    ('user.logout', '{"user_id": 1, "session_duration": 3600}',
     '{"source": "web"}');

-- Query events by type and payload
SELECT
    event_type,
    payload ->> 'user_id' AS user_id,
    payload -> 'total' AS order_total,
    metadata ->> 'source' AS source,
    created_at
FROM api_events
WHERE event_type = 'order.created'
  AND (payload -> 'total')::NUMERIC > 100;

-- Event statistics
SELECT
    event_type,
    COUNT(*) AS count,
    MIN(created_at) AS first_seen,
    MAX(created_at) AS last_seen
FROM api_events
GROUP BY event_type
ORDER BY count DESC;
