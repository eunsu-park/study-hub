-- E-Commerce Source Database Schema
-- Automatically executed on first `docker compose up`

-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id   SERIAL PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    email         VARCHAR(100) UNIQUE NOT NULL,
    region        VARCHAR(50)  NOT NULL,
    segment       VARCHAR(30)  NOT NULL,  -- 'Consumer', 'Corporate', 'Home Office'
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    product_id   SERIAL PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    category     VARCHAR(50)  NOT NULL,
    sub_category VARCHAR(50)  NOT NULL,
    unit_cost    DECIMAL(10,2) NOT NULL,
    unit_price   DECIMAL(10,2) NOT NULL,
    created_at   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id     SERIAL PRIMARY KEY,
    customer_id  INTEGER      NOT NULL REFERENCES customers(customer_id),
    product_id   INTEGER      NOT NULL REFERENCES products(product_id),
    order_date   DATE         NOT NULL,
    quantity     INTEGER      NOT NULL CHECK (quantity > 0),
    amount       DECIMAL(12,2) NOT NULL CHECK (amount >= 0),
    status       VARCHAR(20)  NOT NULL DEFAULT 'pending',
    updated_at   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_status CHECK (status IN ('pending','processing','shipped','delivered','cancelled'))
);

-- Indexes for extraction queries
CREATE INDEX IF NOT EXISTS idx_orders_date     ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_updated  ON orders(updated_at);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_orders_updated
    BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE OR REPLACE TRIGGER trg_customers_updated
    BEFORE UPDATE ON customers
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();
