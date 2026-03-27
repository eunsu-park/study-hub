-- Fact table: orders enriched with customer and product dimensions.
-- Adapted from Data_Engineering Lesson 14 §5.2.

{{
    config(
        materialized='incremental',
        unique_key='order_id',
        schema='marts'
    )
}}

WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

customers AS (
    SELECT * FROM {{ source('bronze', 'customers') }}
),

products AS (
    SELECT * FROM {{ source('bronze', 'products') }}
)

SELECT
    o.order_id,
    o.order_date,

    -- Customer information
    o.customer_id,
    c.name        AS customer_name,
    c.segment     AS customer_segment,
    c.region,

    -- Product information
    o.product_id,
    p.name        AS product_name,
    p.category,

    -- Metrics
    o.quantity,
    o.amount                          AS order_amount,
    p.price * o.quantity              AS cost_amount,
    o.amount - (p.price * o.quantity) AS profit_amount,

    -- Status
    o.status,

    -- Metadata
    o.loaded_at

FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN products p  ON o.product_id  = p.product_id

{% if is_incremental() %}
WHERE o.order_date > (SELECT MAX(order_date) FROM {{ this }})
{% endif %}
