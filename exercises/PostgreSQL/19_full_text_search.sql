-- Exercises for Lesson 19: Full-Text Search
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.


-- === Exercise 1: Product Search ===
-- Problem: Build a full-text search system for an e-commerce product catalog
-- with weighted fields and category filtering.

-- Solution:

-- The search_vector column is a generated column that combines multiple
-- text fields with different weights:
--   A (highest) = product name — exact name matches should rank highest
--   B (medium)  = description — content matches are important
--   C (lower)   = category — provides contextual relevance
CREATE TABLE products_fts (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT,
    price NUMERIC(10,2),
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(name, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(category, '')), 'C')
    ) STORED
);

-- GIN index makes full-text search queries fast (inverted index structure)
CREATE INDEX idx_products_fts_search ON products_fts USING GIN (search_vector);

-- Insert sample products
INSERT INTO products_fts (name, description, category, price) VALUES
('Wireless Bluetooth Headphones', 'Premium noise-cancelling headphones with 30-hour battery life', 'Electronics', 149.99),
('USB-C Charging Cable', 'Fast charging cable compatible with all USB-C devices', 'Accessories', 12.99),
('Mechanical Gaming Keyboard', 'RGB backlit keyboard with Cherry MX switches', 'Electronics', 89.99),
('Laptop Stand', 'Adjustable aluminum stand for laptops up to 17 inches', 'Accessories', 39.99),
('Bluetooth Speaker', 'Portable wireless speaker with deep bass and 12-hour battery', 'Electronics', 59.99);

-- Search function with optional category filter and result limit.
-- websearch_to_tsquery handles natural language input ("wireless bluetooth")
-- better than plainto_tsquery (which doesn't support operators like OR).
CREATE OR REPLACE FUNCTION search_products(
    search_term TEXT,
    category_filter TEXT DEFAULT NULL,
    max_results INT DEFAULT 20
)
RETURNS TABLE(id INT, name TEXT, price NUMERIC, rank REAL) AS $$
BEGIN
    RETURN QUERY
    SELECT p.id, p.name, p.price,
           ts_rank(p.search_vector, websearch_to_tsquery('english', search_term)) AS rank
    FROM products_fts p
    WHERE p.search_vector @@ websearch_to_tsquery('english', search_term)
      AND (category_filter IS NULL OR p.category = category_filter)
    ORDER BY rank DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Test searches
SELECT * FROM search_products('wireless bluetooth');
SELECT * FROM search_products('bluetooth', 'Electronics');
SELECT * FROM search_products('cable charging');


-- === Exercise 2: Search with Highlighting ===
-- Problem: Display search results with matched terms highlighted using HTML tags.

-- Solution:

-- ts_headline wraps matched terms in <mark> tags (configurable).
-- MaxFragments=2 shows up to 2 matching fragments from longer text.
-- FragmentDelimiter separates fragments with " ... " for readability.
SELECT
    id,
    ts_headline('english', name, query,
        'StartSel=<mark>, StopSel=</mark>') AS highlighted_name,
    ts_headline('english', description, query,
        'StartSel=<mark>, StopSel=</mark>, MaxFragments=2, FragmentDelimiter= ... ')
        AS highlighted_desc,
    ts_rank(search_vector, query) AS relevance
FROM products_fts,
     websearch_to_tsquery('english', 'wireless bluetooth') AS query
WHERE search_vector @@ query
ORDER BY relevance DESC;

-- Bonus: highlight with different styling options
SELECT
    id,
    ts_headline('english', name, query,
        'StartSel=**, StopSel=**, MaxWords=10, MinWords=5') AS highlighted_name,
    ts_headline('english', description, query,
        'StartSel=**, StopSel=**, MaxFragments=3, FragmentDelimiter=\n...\n')
        AS highlighted_desc
FROM products_fts,
     websearch_to_tsquery('english', 'battery bluetooth') AS query
WHERE search_vector @@ query
ORDER BY ts_rank(search_vector, query) DESC;


-- === Exercise 3: Multilingual Search ===
-- Problem: Design a search system supporting both English (stemmed)
-- and simple text (no stemming, exact token match).

-- Solution:

-- Two separate tsvector columns, each using a different text search config:
--   'english' config: applies stemming (running -> run, batteries -> batteri)
--   'simple' config: no stemming, just lowercases and splits on whitespace
-- This supports searching by either stemmed English or exact tokens.
CREATE TABLE multilingual_docs (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content_en TEXT,
    content_raw TEXT,
    search_en tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(content_en, ''))
    ) STORED,
    search_simple tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(content_raw, ''))
    ) STORED
);

-- Separate GIN indexes for each language configuration
CREATE INDEX idx_ml_en ON multilingual_docs USING GIN (search_en);
CREATE INDEX idx_ml_simple ON multilingual_docs USING GIN (search_simple);

-- Insert sample data
INSERT INTO multilingual_docs (title, content_en, content_raw) VALUES
('Getting Started', 'Learn how to set up your development environment', 'setup guide beginner tutorial'),
('Advanced Features', 'Explore advanced configuration and performance tuning', 'advanced config performance'),
('API Reference', 'Complete API documentation with examples', 'API REST endpoints reference');

-- Search across both configurations with OR
-- A match in either language is considered a hit
SELECT id, title, content_en
FROM multilingual_docs
WHERE search_en @@ websearch_to_tsquery('english', 'development environment')
   OR search_simple @@ websearch_to_tsquery('simple', 'development environment');

-- Bonus: combined relevance score from both search vectors
SELECT
    id,
    title,
    GREATEST(
        ts_rank(search_en, websearch_to_tsquery('english', 'setup guide')),
        ts_rank(search_simple, websearch_to_tsquery('simple', 'setup guide'))
    ) AS best_rank
FROM multilingual_docs
WHERE search_en @@ websearch_to_tsquery('english', 'setup guide')
   OR search_simple @@ websearch_to_tsquery('simple', 'setup guide')
ORDER BY best_rank DESC;
