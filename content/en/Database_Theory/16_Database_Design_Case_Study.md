# Database Design Case Study

**Previous**: [15. NewSQL and Modern Trends](./15_NewSQL_and_Modern_Trends.md)

---

This capstone lesson brings together everything from the preceding fifteen lessons into a comprehensive, end-to-end database design exercise. We will walk through the complete design lifecycle of an e-commerce platform -- from gathering requirements to writing optimized queries -- applying the theory of ER modeling, normalization, indexing, transactions, and concurrency control to a realistic scenario. A second mini case study (social media platform) provides additional practice, and the lesson concludes with a design review checklist and a catalog of common mistakes.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Conduct requirements analysis for a database system
2. Produce a complete ER diagram from business requirements
3. Map an ER model to a relational schema and normalize to BCNF
4. Make informed decisions about physical design (indexing, partitioning, denormalization)
5. Analyze query plans and apply optimization techniques
6. Design transaction strategies with appropriate isolation levels
7. Evaluate design tradeoffs and document decisions
8. Recognize and avoid common database design mistakes

---

## Table of Contents

1. [Overview of the Design Lifecycle](#1-overview-of-the-design-lifecycle)
2. [Phase 1: Requirements Analysis](#2-phase-1-requirements-analysis)
3. [Phase 2: Conceptual Design (ER Diagram)](#3-phase-2-conceptual-design-er-diagram)
4. [Phase 3: Logical Design (Relational Schema)](#4-phase-3-logical-design-relational-schema)
5. [Phase 4: Physical Design](#5-phase-4-physical-design)
6. [Phase 5: Query Optimization](#6-phase-5-query-optimization)
7. [Phase 6: Transaction Design](#7-phase-6-transaction-design)
8. [Alternative Case Study: Social Media Platform](#8-alternative-case-study-social-media-platform)
9. [Design Review Checklist](#9-design-review-checklist)
10. [Common Design Mistakes and How to Avoid Them](#10-common-design-mistakes-and-how-to-avoid-them)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. Overview of the Design Lifecycle

Database design is not a one-shot process. It follows a structured lifecycle with feedback loops:

```
┌──────────────────────────────────────────────────────────────┐
│                    Database Design Lifecycle                   │
│                                                              │
│  Phase 1            Phase 2            Phase 3               │
│  Requirements  ──▶  Conceptual    ──▶  Logical               │
│  Analysis           Design (ER)        Design (Relational)   │
│                                                              │
│       │                                    │                 │
│       │              ┌─────────────────────┘                 │
│       │              │                                       │
│       │              ▼                                       │
│       │         Phase 4            Phase 5                   │
│       │         Physical      ──▶  Query                     │
│       └────────▶Design             Optimization              │
│                                                              │
│                      │                 │                      │
│                      ▼                 ▼                      │
│                 Phase 6                                       │
│                 Transaction Design                            │
│                                                              │
│                      │                                       │
│                      ▼                                       │
│              Deploy, Monitor, Iterate                         │
└──────────────────────────────────────────────────────────────┘
```

Each phase produces artifacts that feed into the next:

| Phase | Input | Output |
|-------|-------|--------|
| **1. Requirements** | Business needs, interviews | Functional & data requirements |
| **2. Conceptual** | Requirements | ER diagram |
| **3. Logical** | ER diagram | Normalized relational schema (DDL) |
| **4. Physical** | Relational schema + query patterns | Indexes, partitions, denormalization |
| **5. Query** | Physical schema + common queries | Optimized queries with explain plans |
| **6. Transaction** | Business rules + concurrency needs | Isolation levels, locking strategy |

---

## 2. Phase 1: Requirements Analysis

### 2.1 The Business Context

We are designing the database for **ShopWave**, an online e-commerce platform that sells physical products. The platform supports multiple sellers, product reviews, promotions, and a standard checkout flow.

### 2.2 Functional Requirements

After interviewing stakeholders, we identify these key business functions:

```
FR-01: Customer registration and authentication
FR-02: Product catalog browsing with search and filtering
FR-03: Product detail page with images, specifications, and reviews
FR-04: Shopping cart management (add, remove, update quantities)
FR-05: Checkout with multiple payment methods
FR-06: Order tracking (status updates from placement to delivery)
FR-07: Seller management (sellers list products, manage inventory)
FR-08: Product reviews and ratings (1-5 stars, text review)
FR-09: Promotions and discount codes
FR-10: Wishlist (customers save products for later)
FR-11: Address management (multiple shipping addresses per customer)
FR-12: Product categories with hierarchy (e.g., Electronics > Phones > Smartphones)
```

### 2.3 Data Requirements

From the functional requirements, we extract the data entities and their key attributes:

```
DR-01: Customers
  - Name, email, password hash, phone, registration date
  - Multiple shipping addresses
  - One or more payment methods

DR-02: Products
  - Name, description, SKU, price, weight, dimensions
  - Multiple images (URLs)
  - Belongs to one or more categories
  - Listed by exactly one seller
  - Has a stock quantity (inventory)

DR-03: Categories
  - Name, description
  - Hierarchical (parent-child relationship)

DR-04: Orders
  - Customer, shipping address, order date, status
  - One or more order items (product, quantity, unit price at time of order)
  - Payment information
  - Shipping method and tracking number

DR-05: Reviews
  - Customer, product, rating (1-5), text, date
  - Each customer can review a product at most once

DR-06: Sellers
  - Company name, contact email, commission rate
  - Bank account for payouts

DR-07: Promotions
  - Code, discount type (percentage/fixed), amount, valid dates
  - Applicable to specific products, categories, or site-wide
  - Usage limit (total and per customer)
```

### 2.4 Volumetric Analysis

Understanding data volumes helps with physical design decisions:

| Entity | Expected Volume | Growth Rate |
|--------|-----------------|-------------|
| Customers | 1 million | 50K/month |
| Products | 500,000 | 10K/month |
| Orders | 10 million | 500K/month |
| Order Items | 25 million | 1.25M/month |
| Reviews | 2 million | 100K/month |
| Categories | 1,000 | Rarely changes |
| Sellers | 5,000 | 200/month |
| Promotions | 500 active at a time | Seasonal peaks |

### 2.5 Query Patterns

We identify the most frequent and critical queries:

```
QP-01: [High freq] Product search by keyword + category + price range
QP-02: [High freq] Product detail page (product + images + reviews + seller)
QP-03: [High freq] Shopping cart contents for a customer
QP-04: [Critical]  Place order (decrement inventory, create order + items)
QP-05: [High freq] Order history for a customer (with status)
QP-06: [Medium]    Seller dashboard (orders, revenue, inventory)
QP-07: [Medium]    Top-rated products in a category
QP-08: [Low freq]  Admin: sales report by category, date range
QP-09: [Critical]  Apply promotion code (validate, calculate discount)
QP-10: [High freq] Customer wishlist display
```

### 2.6 Non-Functional Requirements

```
NFR-01: Response time: < 200ms for read queries, < 500ms for writes
NFR-02: Availability: 99.9% uptime
NFR-03: Data durability: no data loss for orders and payments
NFR-04: Concurrency: support 10,000 concurrent users
NFR-05: Scalability: handle 10x growth over 3 years
NFR-06: Security: PCI-DSS compliance for payment data
```

---

## 3. Phase 2: Conceptual Design (ER Diagram)

### 3.1 Entity Identification

From the data requirements, we identify these entities:

```
Strong Entities:        Weak Entities:         Associative Entities:
├── Customer            ├── OrderItem          ├── Review
├── Product             ├── CartItem           ├── ProductCategory
├── Category            ├── ProductImage       ├── WishlistItem
├── Order               └── Address            └── PromotionUsage
├── Seller
├── Promotion
├── Cart
└── PaymentMethod
```

### 3.2 ER Diagram

```
                                    ┌──────────────┐
                              ┌─────│   Category   │─────┐
                              │     │ id           │     │
                              │     │ name         │     │ parent_id
                              │     │ description  │     │ (self-ref)
                              │     └──────────────┘─────┘
                              │           │
                              │     ┌─────┴──────┐
                              │     │ Product     │
                              │     │ Category    │
                              │     │ (M:N)       │
                              │     └─────┬──────┘
                              │           │
┌──────────────┐        ┌─────┴──────────┴──┐        ┌──────────────┐
│   Seller     │ 1   M  │     Product       │  1   M │ ProductImage │
│ id           │────────│ id                │────────│ id           │
│ company_name │        │ name              │        │ product_id   │
│ email        │        │ description       │        │ url          │
│ commission   │        │ sku               │        │ alt_text     │
│ bank_account │        │ price             │        │ position     │
└──────────────┘        │ weight            │        └──────────────┘
                        │ stock_quantity    │
                        │ seller_id (FK)    │
                        └────────┬──────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
        ┌─────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐
        │  Review    │   │  WishlistItem│   │  CartItem   │
        │ id         │   │ id           │   │ id          │
        │ customer_id│   │ customer_id  │   │ cart_id     │
        │ product_id │   │ product_id   │   │ product_id  │
        │ rating     │   │ added_at     │   │ quantity    │
        │ text       │   └─────────────┘   │ added_at    │
        │ created_at │                      └──────┬──────┘
        └─────┬──────┘                             │
              │                                    │
              │                              ┌─────┴──────┐
              │                              │    Cart     │
              │                              │ id          │
              │                              │ customer_id │
              │                              │ created_at  │
              │                              └─────┬──────┘
              │                                    │
              │                                    │
        ┌─────┴──────────────────────────────────┴──┐
        │              Customer                      │
        │ id                                         │
        │ email                                      │
        │ password_hash                              │
        │ first_name, last_name                      │
        │ phone                                      │
        │ created_at                                 │
        └────────────┬─────────────┬────────────────┘
                     │             │
              ┌──────┴──────┐ ┌───┴────────────┐
              │   Address   │ │ PaymentMethod  │
              │ id          │ │ id             │
              │ customer_id │ │ customer_id    │
              │ label       │ │ type           │
              │ street      │ │ last_four      │
              │ city        │ │ token          │
              │ state       │ │ is_default     │
              │ zip_code    │ └────────────────┘
              │ country     │
              │ is_default  │
              └──────┬──────┘
                     │
              ┌──────┴──────────────────────────────┐
              │              Order                    │
              │ id                                   │
              │ customer_id                          │
              │ shipping_address_id                  │
              │ payment_method_id                    │
              │ status                               │
              │ subtotal, discount, tax, total       │
              │ promotion_id (nullable)              │
              │ tracking_number                      │
              │ created_at, updated_at               │
              └──────────┬──────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │     OrderItem       │
              │ id                  │
              │ order_id            │
              │ product_id          │
              │ quantity            │
              │ unit_price          │ ← price at time of order
              │ subtotal            │   (not current price!)
              └─────────────────────┘

┌──────────────┐
│  Promotion   │
│ id           │
│ code         │
│ discount_type│  (percentage / fixed_amount)
│ amount       │
│ min_purchase │
│ max_uses     │
│ uses_count   │
│ valid_from   │
│ valid_until  │
│ is_active    │
└──────────────┘
```

### 3.3 Cardinality Summary

| Relationship | Cardinality | Participation |
|-------------|-------------|---------------|
| Customer -- Address | 1:M | Total (at least one address required for ordering) |
| Customer -- PaymentMethod | 1:M | Partial (can register without payment method) |
| Customer -- Order | 1:M | Partial |
| Customer -- Review | 1:M | Partial |
| Customer -- Cart | 1:1 | Partial |
| Customer -- WishlistItem | 1:M | Partial |
| Order -- OrderItem | 1:M | Total (order must have at least one item) |
| Product -- ProductImage | 1:M | Partial (images optional) |
| Product -- Review | 1:M | Partial |
| Product -- Category | M:N | Total (product must be in at least one category) |
| Seller -- Product | 1:M | Total |
| Category -- Category (self) | 1:M | Partial (root categories have no parent) |
| Order -- Promotion | M:1 | Partial (order may have no promotion) |
| Customer + Product -- Review | Unique (one review per customer per product) |

---

## 4. Phase 3: Logical Design (Relational Schema)

### 4.1 ER-to-Relational Mapping

We apply the standard mapping rules from [Lesson 4](./04_ER_Modeling.md):

**Rule 1: Strong entities become tables.**
**Rule 2: Weak entities become tables with the owner's PK as part of their PK or as a FK.**
**Rule 3: 1:M relationships become FKs in the "many" side.**
**Rule 4: M:N relationships become junction tables.**

### 4.2 Complete DDL

```sql
-- ============================================================
-- CUSTOMERS AND AUTHENTICATION
-- ============================================================

CREATE TABLE customers (
    id              BIGSERIAL PRIMARY KEY,
    email           VARCHAR(255) NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    first_name      VARCHAR(100) NOT NULL,
    last_name       VARCHAR(100) NOT NULL,
    phone           VARCHAR(20),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE addresses (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    label           VARCHAR(50) NOT NULL DEFAULT 'Home',  -- 'Home', 'Work', etc.
    street_line1    VARCHAR(255) NOT NULL,
    street_line2    VARCHAR(255),
    city            VARCHAR(100) NOT NULL,
    state           VARCHAR(100),
    zip_code        VARCHAR(20) NOT NULL,
    country         VARCHAR(2) NOT NULL,    -- ISO 3166-1 alpha-2
    is_default      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE payment_methods (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    type            VARCHAR(20) NOT NULL CHECK (type IN ('credit_card', 'debit_card', 'paypal')),
    last_four       CHAR(4) NOT NULL,
    token           VARCHAR(255) NOT NULL,  -- tokenized by payment processor
    expires_at      DATE,
    is_default      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- SELLERS
-- ============================================================

CREATE TABLE sellers (
    id              BIGSERIAL PRIMARY KEY,
    company_name    VARCHAR(255) NOT NULL,
    contact_email   VARCHAR(255) NOT NULL UNIQUE,
    commission_rate DECIMAL(5,4) NOT NULL DEFAULT 0.1500,  -- 15.00%
    bank_account    VARCHAR(255),  -- encrypted or tokenized
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- PRODUCT CATALOG
-- ============================================================

CREATE TABLE categories (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    slug            VARCHAR(100) NOT NULL UNIQUE,
    description     TEXT,
    parent_id       INT REFERENCES categories(id) ON DELETE SET NULL,
    position        INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE products (
    id              BIGSERIAL PRIMARY KEY,
    seller_id       BIGINT NOT NULL REFERENCES sellers(id),
    name            VARCHAR(255) NOT NULL,
    slug            VARCHAR(255) NOT NULL UNIQUE,
    description     TEXT,
    sku             VARCHAR(50) NOT NULL UNIQUE,
    price           DECIMAL(10,2) NOT NULL CHECK (price >= 0),
    weight_grams    INT,
    stock_quantity  INT NOT NULL DEFAULT 0 CHECK (stock_quantity >= 0),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE product_categories (
    product_id      BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    category_id     INT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    PRIMARY KEY (product_id, category_id)
);

CREATE TABLE product_images (
    id              BIGSERIAL PRIMARY KEY,
    product_id      BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    url             VARCHAR(500) NOT NULL,
    alt_text        VARCHAR(255),
    position        INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- SHOPPING CART
-- ============================================================

CREATE TABLE carts (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT REFERENCES customers(id) ON DELETE SET NULL,
    session_id      VARCHAR(255),  -- for anonymous carts
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT cart_owner CHECK (customer_id IS NOT NULL OR session_id IS NOT NULL)
);

CREATE TABLE cart_items (
    id              BIGSERIAL PRIMARY KEY,
    cart_id         BIGINT NOT NULL REFERENCES carts(id) ON DELETE CASCADE,
    product_id      BIGINT NOT NULL REFERENCES products(id),
    quantity        INT NOT NULL CHECK (quantity > 0),
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (cart_id, product_id)
);

-- ============================================================
-- PROMOTIONS
-- ============================================================

CREATE TABLE promotions (
    id              BIGSERIAL PRIMARY KEY,
    code            VARCHAR(50) NOT NULL UNIQUE,
    discount_type   VARCHAR(20) NOT NULL CHECK (discount_type IN ('percentage', 'fixed_amount')),
    amount          DECIMAL(10,2) NOT NULL CHECK (amount > 0),
    min_purchase    DECIMAL(10,2) DEFAULT 0,
    max_uses        INT,                     -- NULL = unlimited
    uses_count      INT NOT NULL DEFAULT 0,
    per_customer    INT NOT NULL DEFAULT 1,  -- max uses per customer
    valid_from      TIMESTAMPTZ NOT NULL,
    valid_until     TIMESTAMPTZ NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (valid_from < valid_until)
);

-- ============================================================
-- ORDERS
-- ============================================================

CREATE TYPE order_status AS ENUM (
    'pending', 'confirmed', 'processing', 'shipped',
    'delivered', 'cancelled', 'refunded'
);

CREATE TABLE orders (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL REFERENCES customers(id),
    shipping_address_id BIGINT NOT NULL REFERENCES addresses(id),
    payment_method_id   BIGINT NOT NULL REFERENCES payment_methods(id),
    promotion_id    BIGINT REFERENCES promotions(id),
    status          order_status NOT NULL DEFAULT 'pending',
    subtotal        DECIMAL(10,2) NOT NULL,
    discount_amount DECIMAL(10,2) NOT NULL DEFAULT 0,
    tax_amount      DECIMAL(10,2) NOT NULL DEFAULT 0,
    shipping_cost   DECIMAL(10,2) NOT NULL DEFAULT 0,
    total           DECIMAL(10,2) NOT NULL,
    tracking_number VARCHAR(100),
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE order_items (
    id              BIGSERIAL PRIMARY KEY,
    order_id        BIGINT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id      BIGINT NOT NULL REFERENCES products(id),
    quantity        INT NOT NULL CHECK (quantity > 0),
    unit_price      DECIMAL(10,2) NOT NULL,  -- price at time of order
    subtotal        DECIMAL(10,2) NOT NULL,  -- quantity * unit_price
    UNIQUE (order_id, product_id)
);

-- ============================================================
-- REVIEWS
-- ============================================================

CREATE TABLE reviews (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL REFERENCES customers(id),
    product_id      BIGINT NOT NULL REFERENCES products(id),
    rating          SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    title           VARCHAR(255),
    body            TEXT,
    is_verified     BOOLEAN NOT NULL DEFAULT FALSE,  -- verified purchase
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (customer_id, product_id)  -- one review per customer per product
);

-- ============================================================
-- WISHLISTS
-- ============================================================

CREATE TABLE wishlist_items (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    product_id      BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (customer_id, product_id)
);
```

### 4.3 Normalization Verification

Let us verify that our schema is in BCNF by analyzing the functional dependencies of key tables.

**customers table**:
```
FDs: id → email, password_hash, first_name, last_name, phone, created_at, updated_at
     email → id  (candidate key)

Candidate keys: {id}, {email}
Both determinants are superkeys → BCNF ✓
```

**products table**:
```
FDs: id → seller_id, name, slug, description, sku, price, weight_grams,
          stock_quantity, is_active, created_at, updated_at
     sku → id (candidate key)
     slug → id (candidate key)

Candidate keys: {id}, {sku}, {slug}
All determinants are superkeys → BCNF ✓
```

**order_items table**:
```
FDs: id → order_id, product_id, quantity, unit_price, subtotal
     (order_id, product_id) → id, quantity, unit_price, subtotal

Candidate keys: {id}, {order_id, product_id}
All determinants are superkeys → BCNF ✓
```

**reviews table**:
```
FDs: id → customer_id, product_id, rating, title, body, is_verified, created_at
     (customer_id, product_id) → id, rating, title, body, is_verified, created_at

Candidate keys: {id}, {customer_id, product_id}
All determinants are superkeys → BCNF ✓
```

**categories table** (self-referential hierarchy):
```
FDs: id → name, slug, description, parent_id, position, created_at
     slug → id

Candidate keys: {id}, {slug}
All determinants are superkeys → BCNF ✓
```

**The schema is in BCNF** -- no non-trivial FD has a determinant that is not a superkey.

### 4.4 Referential Integrity Summary

```
┌─────────────────┐     ┌─────────────────┐
│    customers     │     │     sellers     │
│    (PK: id)      │     │    (PK: id)     │
└───────┬─────────┘     └───────┬─────────┘
   │    │    │    │              │
   │    │    │    │              │
   │    │    │    │    ┌─────────┴──────────┐
   │    │    │    └───▶│     products       │
   │    │    │         │  (FK: seller_id)   │
   │    │    │         └──┬────────┬────────┘
   │    │    │            │        │
   │    │    │     ┌──────┘        └──────┐
   │    │    │     ▼                      ▼
   │    │    │  ┌────────────┐    ┌──────────────┐
   │    │    │  │product_cats│    │product_images│
   │    │    │  └────────────┘    └──────────────┘
   │    │    │
   │    │    └────▶┌──────────┐
   │    │          │  reviews  │ (FK: customer_id, product_id)
   │    │          └──────────┘
   │    │
   │    └─────────▶┌──────────┐     ┌─────────────┐
   │               │  orders   │────▶│ order_items  │
   │               └──────────┘     └─────────────┘
   │
   ├──────────────▶┌──────────┐
   │               │ addresses│
   ├──────────────▶┌──────────┐
   │               │pay_methods│
   ├──────────────▶┌──────────┐
   │               │   carts   │────▶ cart_items
   └──────────────▶┌──────────┐
                   │ wishlist  │
                   └──────────┘
```

---

## 5. Phase 4: Physical Design

### 5.1 Indexing Strategy

We design indexes based on the query patterns identified in Phase 1.

**Primary key indexes** (automatically created):

```sql
-- All tables already have PK indexes from BIGSERIAL PRIMARY KEY
-- Unique constraints also create indexes automatically:
--   customers(email), products(sku), products(slug), promotions(code)
```

**Secondary indexes for frequent queries**:

```sql
-- QP-01: Product search by category + price range
CREATE INDEX idx_products_active_price ON products(price)
    WHERE is_active = TRUE;

-- For category-based search (through junction table)
CREATE INDEX idx_product_categories_category ON product_categories(category_id);

-- QP-02: Product detail page (reviews for a product)
CREATE INDEX idx_reviews_product_rating ON reviews(product_id, rating);

-- QP-03: Shopping cart for a customer
CREATE INDEX idx_carts_customer ON carts(customer_id);

-- QP-04: Inventory check (stock_quantity for a product)
-- PK on products(id) is sufficient

-- QP-05: Order history for a customer
CREATE INDEX idx_orders_customer_date ON orders(customer_id, created_at DESC);

-- QP-06: Seller dashboard
CREATE INDEX idx_products_seller ON products(seller_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);

-- QP-07: Top-rated products in a category
-- Requires joining product_categories + reviews
-- The idx_reviews_product_rating and idx_product_categories_category help

-- QP-09: Promotion validation
CREATE INDEX idx_promotions_code_active ON promotions(code)
    WHERE is_active = TRUE;

-- QP-10: Wishlist display
CREATE INDEX idx_wishlist_customer ON wishlist_items(customer_id, added_at DESC);

-- Full-text search for product names and descriptions
CREATE INDEX idx_products_search ON products
    USING GIN (to_tsvector('english', name || ' ' || COALESCE(description, '')));
```

### 5.2 Partitioning Strategy

For tables expected to grow into hundreds of millions of rows, we apply table partitioning.

```sql
-- Partition orders by month (range partitioning)
-- This helps with:
--   1. Order history queries (scan only relevant months)
--   2. Archiving old orders (drop entire partitions)
--   3. Maintenance operations (VACUUM on smaller partitions)

CREATE TABLE orders (
    id              BIGSERIAL,
    customer_id     BIGINT NOT NULL,
    -- ... other columns ...
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)  -- partition key must be in PK
) PARTITION BY RANGE (created_at);

-- Create partitions
CREATE TABLE orders_2024_01 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE orders_2024_02 PARTITION OF orders
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- ... (automate with pg_partman extension)

-- Similarly partition order_items by order date
-- Or use foreign key to orders which is already partitioned

-- Partition reviews (if they grow very large)
CREATE TABLE reviews (
    -- ...
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);
```

### 5.3 Denormalization Decisions

Pure normalization optimizes for storage and update consistency but can hurt read performance. We strategically denormalize where the read performance benefit outweighs the update complexity.

**Decision 1: Store unit_price in order_items**

```sql
-- ALREADY IN SCHEMA: order_items.unit_price captures price at order time
-- This is NOT denormalization for performance -- it's a business requirement!
-- Product price may change after the order is placed.
-- Without this, we cannot reconstruct historical order totals.
```

**Decision 2: Materialized review statistics**

```sql
-- Instead of computing AVG(rating) and COUNT(*) for every product page load,
-- store pre-computed values on the products table.

ALTER TABLE products ADD COLUMN avg_rating DECIMAL(3,2) DEFAULT NULL;
ALTER TABLE products ADD COLUMN review_count INT NOT NULL DEFAULT 0;

-- Update via trigger or application code when a review is added/modified:
-- UPDATE products SET
--   avg_rating = (SELECT AVG(rating) FROM reviews WHERE product_id = $1),
--   review_count = (SELECT COUNT(*) FROM reviews WHERE product_id = $1)
-- WHERE id = $1;
```

**Decision 3: Store total in orders**

```sql
-- ALREADY IN SCHEMA: orders.total is stored rather than computed
-- Computing SUM(subtotal) from order_items on every read is wasteful
-- The total is fixed once the order is placed
```

**Decision 4: Category path materialization**

```sql
-- For deep category hierarchies, recursive queries are slow
-- Materialize the full path for display:

ALTER TABLE categories ADD COLUMN path TEXT;  -- e.g., "Electronics/Phones/Smartphones"
ALTER TABLE categories ADD COLUMN depth INT NOT NULL DEFAULT 0;

-- Update on category creation/move:
-- path = parent.path || '/' || name
-- depth = parent.depth + 1
```

**Denormalization decision matrix**:

| Denormalization | Benefit | Cost | Decision |
|-----------------|---------|------|----------|
| avg_rating on products | Avoid aggregate on every product page | Update on review change | YES (high read:write ratio) |
| total on orders | Avoid SUM on every order display | Must match order_items | YES (immutable after creation) |
| category path | Avoid recursive queries | Update on category move | YES (categories change rarely) |
| customer name on orders | Avoid JOIN for order list | Update on name change | NO (JOIN is fast with index) |

### 5.4 Storage Considerations

```sql
-- Use appropriate data types for storage efficiency

-- GOOD: DECIMAL(10,2) for money (exact arithmetic)
-- BAD:  FLOAT for money (rounding errors!)

-- GOOD: SMALLINT for rating (1-5, uses 2 bytes)
-- BAD:  INT for rating (uses 4 bytes, wastes space)

-- GOOD: VARCHAR(2) for country codes
-- BAD:  TEXT for country codes (no length constraint)

-- GOOD: TIMESTAMPTZ for dates (timezone-aware)
-- BAD:  TIMESTAMP for dates (ambiguous timezone)

-- GOOD: ENUM for fixed status values
-- BAD:  VARCHAR for status (allows typos)

-- Estimate table sizes:
-- customers: 1M rows x ~300 bytes ≈ 300 MB
-- products:  500K rows x ~500 bytes ≈ 250 MB
-- orders:    10M rows x ~200 bytes ≈ 2 GB
-- order_items: 25M rows x ~100 bytes ≈ 2.5 GB
-- reviews:   2M rows x ~500 bytes ≈ 1 GB
-- TOTAL: ~6 GB (easily fits in RAM for caching)
```

---

## 6. Phase 5: Query Optimization

### 6.1 Product Search (QP-01)

```sql
-- Query: Search products by keyword in a category with price filter
-- This is the most frequent query on the platform

SELECT p.id, p.name, p.price, p.avg_rating, p.review_count,
       (SELECT url FROM product_images pi
        WHERE pi.product_id = p.id ORDER BY pi.position LIMIT 1) AS thumbnail
FROM products p
JOIN product_categories pc ON p.id = pc.product_id
WHERE pc.category_id = 42
  AND p.price BETWEEN 10.00 AND 100.00
  AND p.is_active = TRUE
  AND to_tsvector('english', p.name || ' ' || COALESCE(p.description, ''))
      @@ plainto_tsquery('english', 'wireless bluetooth headphones')
ORDER BY p.avg_rating DESC NULLS LAST
LIMIT 20 OFFSET 0;
```

**Explain plan analysis**:

```
EXPLAIN (ANALYZE, BUFFERS)
-- Expected plan:
-- Limit  (cost=... rows=20)
--   -> Sort  (cost=... key: avg_rating DESC)
--     -> Nested Loop  (cost=...)
--       -> Index Scan on product_categories pc (category_id = 42)
--       -> Index Scan on products p (id = pc.product_id)
--            Filter: is_active AND price BETWEEN 10 AND 100
--            Filter: tsvector @@ tsquery
```

**Optimization**: Create a composite GIN index that covers both full-text search and category:

```sql
-- If category-based full-text search is the dominant pattern:
-- Consider a materialized view that pre-joins products and categories
CREATE MATERIALIZED VIEW product_search AS
SELECT p.id, p.name, p.description, p.price, p.avg_rating,
       p.review_count, p.is_active, pc.category_id,
       to_tsvector('english', p.name || ' ' || COALESCE(p.description, '')) AS search_vector
FROM products p
JOIN product_categories pc ON p.id = pc.product_id;

CREATE INDEX idx_product_search_category ON product_search(category_id)
    WHERE is_active = TRUE;
CREATE INDEX idx_product_search_fts ON product_search USING GIN(search_vector);
CREATE INDEX idx_product_search_price ON product_search(category_id, price)
    WHERE is_active = TRUE;

-- Refresh periodically or on product changes
REFRESH MATERIALIZED VIEW CONCURRENTLY product_search;
```

### 6.2 Product Detail Page (QP-02)

```sql
-- Fetch product with images, seller, and review summary in one query
SELECT
  p.id, p.name, p.description, p.price, p.sku,
  p.stock_quantity, p.avg_rating, p.review_count,
  s.company_name AS seller_name,
  json_agg(DISTINCT jsonb_build_object(
    'url', pi.url, 'alt_text', pi.alt_text, 'position', pi.position
  )) AS images
FROM products p
JOIN sellers s ON p.seller_id = s.id
LEFT JOIN product_images pi ON p.id = pi.product_id
WHERE p.id = $1
GROUP BY p.id, s.id;

-- Fetch reviews separately (paginated)
SELECT r.rating, r.title, r.body, r.created_at, r.is_verified,
       c.first_name || ' ' || LEFT(c.last_name, 1) || '.' AS reviewer
FROM reviews r
JOIN customers c ON r.customer_id = c.id
WHERE r.product_id = $1
ORDER BY r.created_at DESC
LIMIT 10 OFFSET $2;
```

### 6.3 Order History (QP-05)

```sql
-- Customer's recent orders with item summary
SELECT o.id, o.status, o.total, o.created_at,
       COUNT(oi.id) AS item_count,
       STRING_AGG(LEFT(p.name, 30), ', ' ORDER BY oi.id LIMIT 3) AS item_preview
FROM orders o
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE o.customer_id = $1
GROUP BY o.id
ORDER BY o.created_at DESC
LIMIT 20 OFFSET $2;

-- Uses index: idx_orders_customer_date (customer_id, created_at DESC)
-- Partition pruning: if querying recent orders, only scans recent partitions
```

### 6.4 Seller Dashboard (QP-06)

```sql
-- Seller's monthly revenue and order count
SELECT
  DATE_TRUNC('month', o.created_at) AS month,
  COUNT(DISTINCT o.id) AS order_count,
  SUM(oi.subtotal) AS revenue,
  SUM(oi.subtotal * s.commission_rate) AS commission
FROM sellers s
JOIN products p ON s.id = p.seller_id
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
WHERE s.id = $1
  AND o.created_at >= $2  -- start date
  AND o.created_at < $3   -- end date
  AND o.status NOT IN ('cancelled', 'refunded')
GROUP BY month
ORDER BY month DESC;
```

### 6.5 Query Performance Summary

| Query | Expected Latency | Key Index | Notes |
|-------|-----------------|-----------|-------|
| Product search | < 50ms | GIN (FTS) + category | Materialized view helps |
| Product detail | < 10ms | PK + seller FK | Single row + images |
| Cart contents | < 5ms | carts(customer_id) | Small result set |
| Place order | < 100ms | Inventory check + insert | Transaction (see Phase 6) |
| Order history | < 20ms | orders(customer_id, date) | Partition pruning |
| Seller dashboard | < 200ms | products(seller_id) + order_items(product_id) | Aggregate query |
| Top rated products | < 30ms | product_categories + avg_rating | Uses denormalized rating |

---

## 7. Phase 6: Transaction Design

### 7.1 Critical Transaction: Place Order

This is the most complex and important transaction in the system. It must:
1. Validate cart items are in stock
2. Calculate totals (with potential promotion)
3. Decrement inventory
4. Create the order and order items
5. Clear the cart
6. Process payment (external API call)

```sql
-- Isolation level: SERIALIZABLE or READ COMMITTED with explicit locking
-- We use READ COMMITTED + explicit locking for better performance

BEGIN;

-- Step 1: Lock cart items to prevent concurrent modifications
-- Also lock the product rows to prevent overselling
SELECT p.id, p.name, p.price, p.stock_quantity, ci.quantity
FROM cart_items ci
JOIN products p ON ci.product_id = p.id
WHERE ci.cart_id = $cart_id
FOR UPDATE OF p;  -- locks product rows

-- Step 2: Validate stock (application checks: stock_quantity >= ci.quantity)
-- If any item is out of stock, ROLLBACK and inform the customer

-- Step 3: Apply promotion (if any)
SELECT id, discount_type, amount, min_purchase, max_uses, uses_count
FROM promotions
WHERE code = $promo_code
  AND is_active = TRUE
  AND valid_from <= NOW()
  AND valid_until > NOW()
FOR UPDATE;  -- lock to prevent concurrent usage beyond max_uses

-- Step 4: Create order
INSERT INTO orders (customer_id, shipping_address_id, payment_method_id,
                    promotion_id, subtotal, discount_amount, tax_amount,
                    shipping_cost, total, status)
VALUES ($customer_id, $address_id, $payment_id,
        $promo_id, $subtotal, $discount, $tax, $shipping, $total, 'pending')
RETURNING id AS order_id;

-- Step 5: Create order items and decrement inventory
INSERT INTO order_items (order_id, product_id, quantity, unit_price, subtotal)
SELECT $order_id, ci.product_id, ci.quantity, p.price, ci.quantity * p.price
FROM cart_items ci
JOIN products p ON ci.product_id = p.id
WHERE ci.cart_id = $cart_id;

UPDATE products p
SET stock_quantity = stock_quantity - ci.quantity,
    updated_at = NOW()
FROM cart_items ci
WHERE p.id = ci.product_id
  AND ci.cart_id = $cart_id;

-- Step 6: Update promotion usage count
UPDATE promotions SET uses_count = uses_count + 1
WHERE id = $promo_id;

-- Step 7: Clear cart
DELETE FROM cart_items WHERE cart_id = $cart_id;

-- Step 8: Process payment (external API call)
-- NOTE: This should happen OUTSIDE the transaction or use the Saga pattern
-- If payment fails, the entire transaction rolls back

COMMIT;
```

**Important design decision**: The payment API call should NOT be inside the database transaction because:
1. External API calls can be slow (seconds), holding locks too long
2. If the DB commits but the payment fails, we have an inconsistent state

**Better approach: Saga pattern**

```
1. BEGIN; create order with status='pending_payment'; decrement stock; COMMIT;
2. Call payment API
3. If payment succeeds: UPDATE orders SET status='confirmed' WHERE id=$id;
4. If payment fails: UPDATE orders SET status='payment_failed' WHERE id=$id;
                     Restore inventory (compensating transaction)
```

### 7.2 Isolation Level Decisions

| Transaction | Isolation Level | Reason |
|------------|-----------------|--------|
| Place order | READ COMMITTED + FOR UPDATE | Explicit locking prevents overselling; RC is faster than Serializable |
| Browse products | READ COMMITTED | Slight staleness is acceptable for catalog |
| View order history | READ COMMITTED | Historical data, no write conflicts |
| Submit review | READ COMMITTED + UNIQUE constraint | UNIQUE(customer_id, product_id) prevents duplicates |
| Apply promotion | READ COMMITTED + FOR UPDATE | Lock promotion row to prevent exceeding max_uses |
| Admin reports | REPEATABLE READ | Consistent snapshot for long-running aggregation queries |

### 7.3 Locking Strategy

```sql
-- Prevent overselling: use SELECT FOR UPDATE on product rows
-- This acquires row-level exclusive locks

-- Ordered locking to prevent deadlocks:
-- Always lock products in id order when updating multiple products

-- Example: if cart has products [42, 17, 88], lock in order [17, 42, 88]
SELECT id, stock_quantity
FROM products
WHERE id IN (17, 42, 88)
ORDER BY id    -- consistent ordering prevents deadlocks
FOR UPDATE;
```

### 7.4 Concurrency Scenarios

**Scenario 1: Two customers order the last item**

```
Customer A                              Customer B
BEGIN                                   BEGIN
SELECT ... FOR UPDATE (stock=1) ✓       SELECT ... FOR UPDATE → BLOCKS
UPDATE stock = 0                        (waiting for A's lock)
INSERT order_item                       ...
COMMIT → releases lock                  → unblocked, reads stock=0
                                        → stock < quantity → ROLLBACK
                                        → "Item out of stock" message
```

**Scenario 2: Customer updates cart while checking out**

```
Checkout Transaction              Cart Update
BEGIN                             BEGIN
SELECT cart_items FOR UPDATE ✓    UPDATE cart_items → BLOCKS
...                               (waiting for checkout's lock)
...processes order...             ...
DELETE cart_items                  ...
COMMIT → releases lock            → row deleted, update affects 0 rows
                                  → COMMIT (no error, but cart is empty)
```

---

## 8. Alternative Case Study: Social Media Platform

### 8.1 Requirements Summary

**SocialBuzz** is a social media platform supporting posts, follows, likes, and a personalized feed.

### 8.2 Key Entities

```sql
CREATE TABLE users (
    id          BIGSERIAL PRIMARY KEY,
    username    VARCHAR(50) NOT NULL UNIQUE,
    email       VARCHAR(255) NOT NULL UNIQUE,
    display_name VARCHAR(100) NOT NULL,
    bio         TEXT,
    avatar_url  VARCHAR(500),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE follows (
    follower_id BIGINT NOT NULL REFERENCES users(id),
    followee_id BIGINT NOT NULL REFERENCES users(id),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (follower_id, followee_id),
    CHECK (follower_id != followee_id)
);

CREATE TABLE posts (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id),
    content     TEXT NOT NULL,
    media_urls  TEXT[],  -- array of media URLs
    like_count  INT NOT NULL DEFAULT 0,      -- denormalized
    comment_count INT NOT NULL DEFAULT 0,    -- denormalized
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE likes (
    user_id     BIGINT NOT NULL REFERENCES users(id),
    post_id     BIGINT NOT NULL REFERENCES posts(id),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, post_id)
);

CREATE TABLE comments (
    id          BIGSERIAL PRIMARY KEY,
    post_id     BIGINT NOT NULL REFERENCES posts(id),
    user_id     BIGINT NOT NULL REFERENCES users(id),
    parent_id   BIGINT REFERENCES comments(id),  -- threaded comments
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 8.3 The Feed Problem

The most challenging query is the personalized feed: "Show recent posts from people I follow."

**Pull model** (fan-out on read):

```sql
-- At read time, query posts from all followed users
SELECT p.*, u.username, u.display_name, u.avatar_url
FROM posts p
JOIN follows f ON p.user_id = f.followee_id
JOIN users u ON p.user_id = u.id
WHERE f.follower_id = $current_user_id
ORDER BY p.created_at DESC
LIMIT 20;

-- Problem: If a user follows 1000 people, this JOINs a large follows list
-- with a large posts table. Slow for active users.
```

**Push model** (fan-out on write):

```sql
-- Create a feed table (denormalized timeline)
CREATE TABLE feed_items (
    user_id     BIGINT NOT NULL,  -- the user whose feed this is
    post_id     BIGINT NOT NULL,
    author_id   BIGINT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (user_id, created_at DESC, post_id)
);

-- When user X creates a post:
-- Insert a feed_item for EVERY follower of X
INSERT INTO feed_items (user_id, post_id, author_id, created_at)
SELECT f.follower_id, $post_id, $author_id, $created_at
FROM follows f
WHERE f.followee_id = $author_id;

-- Reading the feed is now trivial:
SELECT fi.post_id, fi.author_id, fi.created_at,
       p.content, p.media_urls, p.like_count,
       u.username, u.avatar_url
FROM feed_items fi
JOIN posts p ON fi.post_id = p.id
JOIN users u ON fi.author_id = u.id
WHERE fi.user_id = $current_user_id
ORDER BY fi.created_at DESC
LIMIT 20;

-- Problem: If a celebrity has 10 million followers, creating a post
-- requires 10 million INSERT operations. Slow for popular users.
```

**Hybrid model** (used by Twitter/X):

```
Regular users (< 10K followers): Push model (fan-out on write)
Celebrity users (> 10K followers): Pull model (fan-out on read)

Feed = Merge(feed_items table, live query for followed celebrities)
```

### 8.4 Design Tradeoffs

| Aspect | E-Commerce | Social Media |
|--------|-----------|-------------|
| **Critical transaction** | Place order (ACID required) | Post creation (eventual consistency OK) |
| **Primary read pattern** | Product lookup by ID | Timeline (feed) |
| **Write pattern** | Low frequency, high value | High frequency, low value |
| **Denormalization** | Moderate (avg_rating, totals) | Heavy (feed_items, counters) |
| **Consistency** | Strong (inventory, payments) | Eventual (like counts, feeds) |
| **Scale challenge** | Inventory contention | Fan-out (popular users) |

---

## 9. Design Review Checklist

Use this checklist before deploying any database design to production:

### 9.1 Schema Design

```
[ ] Every table has a clear primary key
[ ] Foreign keys are defined for all relationships
[ ] ON DELETE / ON UPDATE actions are specified (CASCADE, SET NULL, RESTRICT)
[ ] NOT NULL constraints on required fields
[ ] CHECK constraints for domain validation (e.g., price >= 0, rating BETWEEN 1 AND 5)
[ ] UNIQUE constraints for natural keys (email, username, SKU)
[ ] Appropriate data types (DECIMAL for money, TIMESTAMPTZ for dates)
[ ] No redundant columns unless denormalization is documented and justified
[ ] Schema is in at least 3NF (preferably BCNF)
[ ] Self-referential FKs for hierarchies (categories, comments)
```

### 9.2 Indexing

```
[ ] Indexes exist for all foreign keys (PostgreSQL does NOT auto-create FK indexes!)
[ ] Indexes cover the WHERE, JOIN, and ORDER BY columns of frequent queries
[ ] Composite indexes follow the left-prefix rule
[ ] Partial indexes used where appropriate (WHERE is_active = TRUE)
[ ] GIN indexes for full-text search and array columns
[ ] No redundant indexes (index on (a, b) makes index on (a) redundant)
[ ] Index bloat monitoring plan in place
```

### 9.3 Data Integrity

```
[ ] All money values use DECIMAL, never FLOAT
[ ] Timestamps use TIMESTAMPTZ (timezone-aware)
[ ] Enum types or CHECK constraints for status fields
[ ] Immutable fields are protected (order totals, historical prices)
[ ] Soft delete vs hard delete decision documented for each table
[ ] Data retention policy defined
```

### 9.4 Performance

```
[ ] EXPLAIN ANALYZE run on all critical queries
[ ] No sequential scans on large tables for indexed queries
[ ] Pagination uses keyset pagination, not OFFSET
[ ] N+1 query patterns eliminated (use JOINs or batch fetching)
[ ] Connection pooling configured (PgBouncer)
[ ] Appropriate work_mem and shared_buffers settings
```

### 9.5 Transactions

```
[ ] Isolation level chosen for each transaction type
[ ] Locking strategy documented (which rows/tables are locked)
[ ] Lock ordering defined to prevent deadlocks
[ ] Long-running transactions avoided (no API calls inside transactions)
[ ] Retry logic for serialization failures
[ ] Deadlock detection timeout configured
```

### 9.6 Security

```
[ ] No plaintext passwords (use bcrypt/argon2 hashes)
[ ] Payment tokens stored, not card numbers (PCI-DSS)
[ ] Row-Level Security (RLS) for multi-tenant data
[ ] Application user has minimal privileges (no SUPERUSER)
[ ] SQL injection prevented (parameterized queries only)
[ ] Audit logging for sensitive operations
```

### 9.7 Operations

```
[ ] Backup strategy defined (pg_dump schedule, WAL archiving)
[ ] Point-in-time recovery (PITR) tested
[ ] Monitoring: query latency, connection count, table bloat, replication lag
[ ] Schema migration tool in use (Flyway, Alembic, golang-migrate)
[ ] Runbook for common failure scenarios (disk full, replication lag, deadlocks)
```

---

## 10. Common Design Mistakes and How to Avoid Them

### Mistake 1: Using FLOAT for Money

```sql
-- BAD: Floating-point arithmetic causes rounding errors
CREATE TABLE orders (
    total FLOAT  -- 0.1 + 0.2 = 0.30000000000000004
);

-- GOOD: Use exact decimal types
CREATE TABLE orders (
    total DECIMAL(10,2)  -- 0.1 + 0.2 = 0.30
);
```

### Mistake 2: Missing Foreign Key Indexes

```sql
-- PostgreSQL creates indexes for PRIMARY KEY and UNIQUE, but NOT for FOREIGN KEYS!
-- This means JOIN queries on FK columns do sequential scans.

CREATE TABLE order_items (
    order_id BIGINT REFERENCES orders(id),  -- NO automatic index!
    product_id BIGINT REFERENCES products(id)  -- NO automatic index!
);

-- FIX: Always create indexes on FK columns
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
```

### Mistake 3: Storing Derived Data Without a Maintenance Strategy

```sql
-- Denormalization is fine, but you MUST maintain consistency
ALTER TABLE products ADD COLUMN avg_rating DECIMAL(3,2);

-- BAD: Update manually and hope developers remember
-- GOOD: Use a trigger
CREATE OR REPLACE FUNCTION update_product_rating()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE products SET
        avg_rating = (SELECT AVG(rating) FROM reviews WHERE product_id = NEW.product_id),
        review_count = (SELECT COUNT(*) FROM reviews WHERE product_id = NEW.product_id)
    WHERE id = NEW.product_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_product_rating
AFTER INSERT OR UPDATE OR DELETE ON reviews
FOR EACH ROW EXECUTE FUNCTION update_product_rating();
```

### Mistake 4: OFFSET-Based Pagination

```sql
-- BAD: OFFSET scans and discards rows. Page 1000 reads 20,000 rows!
SELECT * FROM products ORDER BY created_at DESC LIMIT 20 OFFSET 19980;
-- ↑ Reads 20,000 rows, discards 19,980, returns 20.

-- GOOD: Keyset pagination (cursor-based)
SELECT * FROM products
WHERE created_at < $last_seen_created_at
ORDER BY created_at DESC
LIMIT 20;
-- ↑ Uses index, reads exactly 20 rows regardless of page number.
```

### Mistake 5: The God Table (EAV Anti-Pattern)

```sql
-- BAD: Entity-Attribute-Value pattern
CREATE TABLE properties (
    entity_type VARCHAR(50),   -- 'product', 'user', etc.
    entity_id   BIGINT,
    key         VARCHAR(100),  -- 'color', 'size', 'price'
    value       TEXT           -- everything is a string!
);

-- Problems:
-- 1. No type safety (price stored as text)
-- 2. No constraints (can't enforce "price >= 0")
-- 3. No foreign keys
-- 4. Terrible query performance (pivot queries are slow)
-- 5. No schema documentation

-- GOOD: Use proper tables with typed columns
-- If attributes vary by category, consider:
--   1. PostgreSQL JSONB column for flexible attributes
--   2. Separate tables per category (product_electronics, product_clothing)
--   3. Hybrid: common columns in products + JSONB for category-specific
```

### Mistake 6: Not Capturing Historical Data

```sql
-- BAD: Only storing current state
-- If a product's price changes, all historical orders show the new price!

-- GOOD: Capture state at the time of the event
CREATE TABLE order_items (
    product_id  BIGINT REFERENCES products(id),
    quantity    INT,
    unit_price  DECIMAL(10,2),  -- price AT TIME OF ORDER, not current price
    product_name VARCHAR(255)    -- optional: snapshot of product name at order time
);
```

### Mistake 7: Premature Denormalization

```
Phase: Before you have any users
Developer: "Let's denormalize everything for performance!"

Reality check:
- A properly indexed PostgreSQL can handle most workloads up to 100M rows
- Start normalized, measure, then denormalize where measurements show problems
- Premature denormalization creates maintenance nightmares with no measurable benefit

Rule of thumb: If you have fewer than 10M rows and your queries use indexes,
you probably don't need denormalization.
```

### Mistake 8: Ignoring NULL Semantics

```sql
-- Surprise: NULL comparisons don't work as expected
SELECT * FROM products WHERE category_id != 5;
-- Does NOT return rows where category_id IS NULL!

-- FIX: Handle NULLs explicitly
SELECT * FROM products WHERE category_id != 5 OR category_id IS NULL;
-- Or use: WHERE category_id IS DISTINCT FROM 5;

-- Also: COUNT(column) ignores NULLs, COUNT(*) counts all rows
SELECT COUNT(phone) FROM customers;  -- only counts non-NULL phones
SELECT COUNT(*) FROM customers;      -- counts all customers
```

### Mistake 9: Storing Comma-Separated Values

```sql
-- BAD: Storing multiple values in a single column
CREATE TABLE products (
    tags VARCHAR(500)  -- "electronics,wireless,bluetooth"
);

-- Problems:
-- 1. Can't index individual tags efficiently
-- 2. Can't enforce referential integrity
-- 3. "Find products with tag 'wireless'" requires LIKE '%wireless%' (slow, incorrect)
-- 4. Can't JOIN with a tags table
-- 5. Violates 1NF

-- GOOD: Use a junction table
CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE product_tags (
    product_id BIGINT REFERENCES products(id),
    tag_id INT REFERENCES tags(id),
    PRIMARY KEY (product_id, tag_id)
);

-- Or use PostgreSQL arrays with GIN index
CREATE TABLE products (
    tags TEXT[]
);
CREATE INDEX idx_products_tags ON products USING GIN(tags);
SELECT * FROM products WHERE tags @> ARRAY['wireless'];
```

### Mistake 10: Not Planning for Schema Evolution

```sql
-- BAD: No migration tool, manual ALTER TABLEs in production

-- GOOD: Use a migration tool (Flyway, Alembic, golang-migrate)
-- Each migration is a versioned, reversible SQL file:

-- V001__create_customers.sql
CREATE TABLE customers (...);

-- V002__add_phone_to_customers.sql
ALTER TABLE customers ADD COLUMN phone VARCHAR(20);

-- V003__create_orders.sql
CREATE TABLE orders (...);

-- Benefits:
-- 1. Every schema change is version-controlled
-- 2. Reproducible across environments (dev, staging, prod)
-- 3. Rollback capability
-- 4. Team collaboration (no conflicting manual changes)
```

---

## 11. Exercises

### Exercise 1: Complete the E-Commerce Schema

The ShopWave schema above is missing several features. Extend it to support:

1. **Product variants**: Products can have variants (e.g., "Blue, Large", "Red, Small") each with its own price, SKU, and stock quantity. Design the tables.
2. **Order status history**: Instead of a single status field, track every status change with a timestamp and the user who made the change.
3. **Seller payouts**: Track commission calculations and payout history for sellers.

Write the CREATE TABLE statements for each extension and explain how they integrate with the existing schema.

### Exercise 2: Normalization Analysis

Consider this denormalized order table:

```sql
CREATE TABLE flat_orders (
    order_id        INT,
    customer_name   VARCHAR(100),
    customer_email  VARCHAR(255),
    product_name    VARCHAR(255),
    product_sku     VARCHAR(50),
    product_price   DECIMAL(10,2),
    quantity        INT,
    order_total     DECIMAL(10,2),
    order_date      DATE,
    shipping_street VARCHAR(255),
    shipping_city   VARCHAR(100),
    shipping_zip    VARCHAR(20)
);
```

1. List all functional dependencies.
2. Identify the candidate key(s).
3. Determine whether the table is in 1NF, 2NF, 3NF, and BCNF.
4. Decompose into BCNF, showing each decomposition step.
5. Verify that your decomposition is lossless-join.

### Exercise 3: Index Design Challenge

Given the ShopWave schema and these query patterns:

```sql
-- Q1: Products on sale (price < original_price) in a category
SELECT * FROM products
WHERE category_id = $1 AND sale_price < price AND is_active = TRUE
ORDER BY (price - sale_price) DESC
LIMIT 20;

-- Q2: Customer's order with specific status in date range
SELECT * FROM orders
WHERE customer_id = $1 AND status = $2
  AND created_at BETWEEN $3 AND $4
ORDER BY created_at DESC;

-- Q3: Products reviewed by a specific customer
SELECT p.name, r.rating, r.created_at
FROM reviews r
JOIN products p ON r.product_id = p.id
WHERE r.customer_id = $1
ORDER BY r.created_at DESC;

-- Q4: Sellers with the most orders this month
SELECT s.company_name, COUNT(DISTINCT o.id) AS order_count
FROM sellers s
JOIN products p ON s.id = p.seller_id
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
WHERE o.created_at >= DATE_TRUNC('month', CURRENT_DATE)
GROUP BY s.id
ORDER BY order_count DESC
LIMIT 10;
```

For each query:
1. Design the optimal index(es).
2. Write the expected EXPLAIN plan.
3. Estimate the selectivity of each predicate.

### Exercise 4: Transaction Design

Design the transaction for the following business operation:

**Apply a bulk discount**: An admin wants to reduce the price of all products in category "Electronics" by 15%, capping the discount at $50 per product, and logging the change in an audit table.

Requirements:
- No product should be visible to customers at an intermediate state (partially discounted)
- The operation should not block product reads for more than 1 second
- All changes must be logged with before/after values

Write the transaction SQL and specify the isolation level.

### Exercise 5: Design Your Own Database

Choose ONE of the following scenarios and complete a full database design (Phases 1-6):

**Option A: Library Management System**
- Patrons, books (multiple copies), checkouts, reservations, fines
- Patrons can check out up to 5 books for 14 days
- Late returns incur $0.25/day fine
- Books can be reserved if all copies are checked out
- Librarians can add/remove books and manage patron accounts

**Option B: Restaurant Reservation System**
- Restaurants, tables, reservations, customers, waitlists
- Restaurants have tables of different sizes (2, 4, 6, 8 seats)
- Reservations are for a specific date/time and party size
- Waitlist when no tables are available
- Customers can leave reviews for restaurants

**Option C: Online Learning Platform**
- Courses, lessons, students, enrollments, progress tracking, quizzes
- Courses have multiple lessons in a defined order
- Students track progress (completed lessons, quiz scores)
- Instructors create and manage courses
- Certificates issued upon course completion

For your chosen scenario, produce:
1. Functional requirements (at least 8)
2. Data requirements with volumetric estimates
3. ER diagram with cardinalities
4. Normalized relational schema (CREATE TABLE statements)
5. Indexing strategy with justification
6. Two critical transactions with isolation level and locking strategy
7. Three representative queries with expected explain plans

### Exercise 6: Critique This Design

Find and fix at least 8 problems in the following schema:

```sql
CREATE TABLE users (
    id INT,
    name TEXT,
    email TEXT,
    password TEXT,
    created DATE
);

CREATE TABLE products (
    id INT,
    title TEXT,
    price FLOAT,
    category TEXT,
    tags TEXT,  -- comma-separated: "electronics,sale,new"
    stock INT
);

CREATE TABLE orders (
    id INT,
    user_email TEXT,
    product_ids TEXT,  -- comma-separated: "1,5,12"
    quantities TEXT,   -- comma-separated: "2,1,3"
    total FLOAT,
    status TEXT,
    date TEXT
);
```

For each problem:
1. Identify the issue.
2. Explain why it is problematic.
3. Provide the corrected SQL.

### Exercise 7: Scaling the E-Commerce Design

The ShopWave platform has grown to:
- 50 million customers
- 10 million products
- 500 million orders
- 1.5 billion order items
- 100 million reviews

The current single-node PostgreSQL can no longer handle the load. Propose a scaling strategy:

1. Which tables should be partitioned? By what key?
2. Which tables need read replicas?
3. Which data should be moved to Redis (caching)?
4. Should any data be moved to a different database type (e.g., Elasticsearch for search, Cassandra for order history)? Justify.
5. How would you handle the transition from a single database to a distributed architecture without downtime?

### Exercise 8: Social Media Feed Design

Using the SocialBuzz schema from Section 8:

1. Implement the hybrid feed model: write the SQL for both the push path (regular users creating posts) and the pull path (merging celebrity posts at read time).
2. A celebrity with 5 million followers changes their username. How do you update the feed_items table efficiently?
3. Design a "trending posts" feature: show the 50 posts with the most likes in the last 24 hours. What indexes do you need? How do you handle the write load on like_count?
4. A user blocks another user. What changes do you need to make to the feed system?

### Exercise 9: Migration Planning

You need to add a "product variants" feature to the live ShopWave database (see Exercise 1). Write a zero-downtime migration plan:

1. List the migration steps in order.
2. For each step, specify whether it requires a lock and for how long.
3. Describe the backward-compatible application changes needed between migration steps.
4. What is your rollback plan if something goes wrong mid-migration?

### Exercise 10: Design Review

Review the complete ShopWave schema (Phase 3 DDL) against the Design Review Checklist (Section 9). For each checklist item that is NOT yet satisfied, describe the gap and provide the SQL fix. Aim to find at least 5 gaps.

---

## 12. References

1. Elmasri, R. & Navathe, S. (2016). *Fundamentals of Database Systems*, 7th Edition. Pearson. Chapters 3-9.
2. Garcia-Molina, H., Ullman, J., & Widom, J. (2008). *Database Systems: The Complete Book*, 2nd Edition. Pearson.
3. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media. Chapters 2-3.
4. Winand, M. (2012). *SQL Performance Explained*. https://use-the-index-luke.com/
5. PostgreSQL Documentation. "Partitioning." https://www.postgresql.org/docs/current/ddl-partitioning.html
6. PostgreSQL Documentation. "Concurrency Control." https://www.postgresql.org/docs/current/mvcc.html
7. Karwin, B. (2010). *SQL Antipatterns*. Pragmatic Bookshelf.
8. Kleppmann, M. (2012). "Designing Data-Intensive Applications: Partitioning." Chapter 6.
9. Sadalage, P. & Fowler, M. (2012). *NoSQL Distilled*. Addison-Wesley. (For polyglot persistence comparisons.)
10. Twitter Engineering Blog. (2013). "How Twitter Stores 250 Million Tweets a Day."

---

**Previous**: [15. NewSQL and Modern Trends](./15_NewSQL_and_Modern_Trends.md)
