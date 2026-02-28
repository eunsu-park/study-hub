-- Exercises for Lesson 14: JSON/JSONB
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.


-- === Exercise 1: User Settings Storage ===
-- Problem: Create a table to store user settings in JSONB,
-- with a function to merge default settings and query/update specific keys.

-- Solution:

-- Table stores per-user settings as a JSONB blob.
-- This is more flexible than adding a column for each setting.
CREATE TABLE user_settings (
    user_id INTEGER PRIMARY KEY,
    settings JSONB NOT NULL DEFAULT '{}'::JSONB,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Function: merge user settings with defaults so missing keys get filled in.
-- The || operator: right side overrides left, so user values take precedence.
CREATE OR REPLACE FUNCTION get_settings_with_defaults(p_user_id INTEGER)
RETURNS JSONB AS $$
DECLARE
    defaults JSONB := '{
        "theme": "light",
        "language": "en",
        "notifications": true,
        "items_per_page": 20,
        "timezone": "UTC"
    }'::JSONB;
    user_prefs JSONB;
BEGIN
    SELECT settings INTO user_prefs
    FROM user_settings
    WHERE user_id = p_user_id;

    -- Merge: defaults first, then user overrides on top
    RETURN defaults || COALESCE(user_prefs, '{}'::JSONB);
END;
$$ LANGUAGE plpgsql STABLE;

-- Function: read a single setting (returns text)
CREATE OR REPLACE FUNCTION get_setting(p_user_id INTEGER, p_key TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN (
        SELECT settings ->> p_key
        FROM user_settings
        WHERE user_id = p_user_id
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- Function: update a single setting using jsonb_set
CREATE OR REPLACE FUNCTION update_setting(
    p_user_id INTEGER,
    p_key TEXT,
    p_value JSONB
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO user_settings (user_id, settings, updated_at)
    VALUES (p_user_id, jsonb_build_object(p_key, p_value), NOW())
    ON CONFLICT (user_id)
    DO UPDATE SET
        settings = user_settings.settings || jsonb_build_object(p_key, p_value),
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Test
SELECT update_setting(1, 'theme', '"dark"'::JSONB);
SELECT update_setting(1, 'language', '"ko"'::JSONB);
SELECT get_settings_with_defaults(1);
SELECT get_setting(1, 'theme');


-- === Exercise 2: JSON Aggregate Report ===
-- Problem: Generate a comprehensive report from orders in a single JSON object.

-- Solution:

-- Build the entire report as one JSONB value using subqueries.
-- json_build_object creates key-value pairs; json_agg creates arrays.
SELECT json_build_object(
    'total_orders', (SELECT COUNT(*) FROM orders),
    'total_revenue', (SELECT COALESCE(SUM(amount), 0) FROM orders),
    'by_status', (
        -- Aggregate status counts into a single JSON object
        SELECT json_object_agg(status, cnt)
        FROM (
            SELECT status, COUNT(*) AS cnt
            FROM orders
            GROUP BY status
        ) s
    ),
    'top_products', (
        -- Top 5 products as a JSON array of objects
        SELECT json_agg(row_to_json(t))
        FROM (
            SELECT product_id AS id, COUNT(*) AS count
            FROM order_items
            GROUP BY product_id
            ORDER BY count DESC
            LIMIT 5
        ) t
    ),
    'generated_at', NOW()
) AS report;


-- === Exercise 3: JSON Search Optimization ===
-- Problem: Generate 1M rows of event data, compare different JSON index types.

-- Solution:

-- Create an events table with JSONB metadata
CREATE TABLE events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50),
    payload JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Generate 1 million rows of realistic event data
INSERT INTO events (event_type, payload, created_at)
SELECT
    (ARRAY['click', 'view', 'purchase', 'signup', 'logout'])[1 + (random()*4)::int],
    jsonb_build_object(
        'user_id', (random() * 10000)::int,
        'page', '/page/' || (random() * 100)::int,
        'duration_ms', (random() * 5000)::int,
        'metadata', jsonb_build_object(
            'browser', (ARRAY['Chrome', 'Firefox', 'Safari'])[1 + (random()*2)::int],
            'os', (ARRAY['Windows', 'macOS', 'Linux'])[1 + (random()*2)::int]
        )
    ),
    NOW() - (random() * 365 || ' days')::interval
FROM generate_series(1, 1000000);

-- Baseline: no index — expect sequential scan
EXPLAIN ANALYZE
SELECT * FROM events WHERE payload @> '{"user_id": 42}';

-- Strategy 1: GIN index on entire JSONB column
-- Supports @>, ?, ?|, ?& operators
CREATE INDEX idx_events_payload_gin ON events USING GIN (payload);

EXPLAIN ANALYZE
SELECT * FROM events WHERE payload @> '{"user_id": 42}';

-- Strategy 2: GIN with jsonb_path_ops (smaller index, faster @> only)
DROP INDEX idx_events_payload_gin;
CREATE INDEX idx_events_payload_pathops ON events USING GIN (payload jsonb_path_ops);

EXPLAIN ANALYZE
SELECT * FROM events WHERE payload @> '{"user_id": 42}';

-- Strategy 3: B-tree index on a specific extracted key (most efficient for equality)
CREATE INDEX idx_events_user_id ON events ((payload->>'user_id'));

EXPLAIN ANALYZE
SELECT * FROM events WHERE payload->>'user_id' = '42';

-- Compare index sizes
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE tablename = 'events';


-- === Exercise 4: Hierarchical JSON Processing ===
-- Problem: Flatten a nested org chart JSON using a recursive CTE.

-- Solution:

-- The org chart is a tree stored as nested JSON:
-- {"name": "CEO", "title": "...", "children": [{"name": "CTO", ...}]}
WITH RECURSIVE org_tree AS (
    -- Base case: the root node
    SELECT
        (org->>'name') AS name,
        (org->>'title') AS title,
        org->'children' AS children,
        0 AS depth,
        (org->>'name') AS path
    FROM (
        SELECT '{
            "name": "Alice",
            "title": "CEO",
            "children": [
                {
                    "name": "Bob",
                    "title": "CTO",
                    "children": [
                        {"name": "Dave", "title": "Senior Engineer", "children": []},
                        {"name": "Eve", "title": "Engineer", "children": []}
                    ]
                },
                {
                    "name": "Carol",
                    "title": "CFO",
                    "children": [
                        {"name": "Frank", "title": "Accountant", "children": []}
                    ]
                }
            ]
        }'::JSONB AS org
    ) root

    UNION ALL

    -- Recursive case: expand each child node
    SELECT
        (child->>'name'),
        (child->>'title'),
        child->'children',
        depth + 1,
        path || ' > ' || (child->>'name')
    FROM org_tree,
         jsonb_array_elements(children) AS child
    WHERE jsonb_array_length(children) > 0
)
SELECT
    REPEAT('  ', depth) || name AS hierarchy,
    title,
    depth,
    path
FROM org_tree
ORDER BY path;
