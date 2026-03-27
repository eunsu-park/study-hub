-- Exercises for Lesson 03: Tables and Data Types
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.

-- === Exercise 1: Online Shopping Mall Table Design ===
-- Problem: Design a complete e-commerce schema with users, categories,
-- products, orders, and order_items tables with proper constraints
-- and foreign key relationships.

-- Solution:

-- 1. Users table
-- Stores customer accounts with unique email constraint
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. Categories table
-- Self-referencing FK allows hierarchical categories (e.g., Electronics > Laptops)
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES categories(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3. Products table
-- Uses JSONB for flexible attributes (brand, color, specs vary by product)
-- CHECK constraints prevent negative prices and stock
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES categories(id),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price NUMERIC(12, 2) NOT NULL CHECK (price >= 0),
    stock INTEGER DEFAULT 0 CHECK (stock >= 0),
    attributes JSONB,
    is_available BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 4. Orders table
-- CHECK constraint restricts status to a known set of valid values
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    status VARCHAR(20) DEFAULT 'pending' CHECK (
        status IN ('pending', 'paid', 'shipped', 'delivered', 'cancelled')
    ),
    total_amount NUMERIC(12, 2) NOT NULL,
    shipping_address TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 5. Order items table
-- ON DELETE CASCADE: if an order is deleted, its line items are removed too
-- because order_items are meaningless without their parent order
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Verify table structure
-- \dt
-- \d products

-- Insert sample data to verify constraints work
INSERT INTO categories (name) VALUES ('Electronics'), ('Clothing');
INSERT INTO categories (name, parent_id) VALUES ('Laptops', 1), ('T-Shirts', 2);

INSERT INTO users (email, password_hash, name) VALUES
('alice@example.com', '$2b$12$hash1', 'Alice'),
('bob@example.com', '$2b$12$hash2', 'Bob');

INSERT INTO products (category_id, name, price, stock, attributes) VALUES
(3, 'MacBook Pro 16"', 2499.99, 50, '{"brand": "Apple", "ram": 32, "storage": "1TB"}'),
(4, 'Basic Tee', 19.99, 200, '{"color": "white", "sizes": ["S", "M", "L", "XL"]}');

-- Verify: this INSERT should fail due to CHECK constraint (negative price)
-- INSERT INTO products (category_id, name, price) VALUES (3, 'Bad Product', -10.00);
-- ERROR: new row violates check constraint "products_price_check"
