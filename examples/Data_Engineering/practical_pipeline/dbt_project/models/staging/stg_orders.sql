-- Staging model: clean and type-cast raw orders.
-- Adapted from Data_Engineering Lesson 14 §5.1.

{{
    config(
        materialized='view',
        schema='staging'
    )
}}

WITH source AS (
    SELECT * FROM {{ source('bronze', 'orders') }}
),

cleaned AS (
    SELECT
        order_id,
        customer_id,
        product_id,
        CAST(order_date AS DATE) AS order_date,
        CAST(amount AS DECIMAL(12, 2)) AS amount,
        CAST(quantity AS INT) AS quantity,
        status,
        CURRENT_TIMESTAMP AS loaded_at
    FROM source
    WHERE order_id IS NOT NULL
      AND amount > 0
)

SELECT * FROM cleaned
