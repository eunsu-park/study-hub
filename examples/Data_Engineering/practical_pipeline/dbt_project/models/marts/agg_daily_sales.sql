-- Aggregate daily sales by category, region, and customer segment.
-- Adapted from Data_Engineering Lesson 14 §5.2.

{{
    config(
        materialized='table',
        schema='marts'
    )
}}

WITH orders AS (
    SELECT * FROM {{ ref('fct_orders') }}
)

SELECT
    order_date,
    category,
    region,
    customer_segment,

    -- Order metrics
    COUNT(DISTINCT order_id)   AS total_orders,
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(quantity)               AS total_quantity,

    -- Revenue metrics
    SUM(order_amount)           AS total_revenue,
    AVG(order_amount)           AS avg_order_value,
    SUM(profit_amount)          AS total_profit,

    -- Profit margin
    ROUND(
        SUM(profit_amount) / NULLIF(SUM(order_amount), 0) * 100, 2
    ) AS profit_margin_pct,

    CURRENT_TIMESTAMP AS updated_at

FROM orders
GROUP BY
    order_date,
    category,
    region,
    customer_segment
