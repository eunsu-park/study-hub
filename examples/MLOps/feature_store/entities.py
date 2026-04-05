"""
Feast entity definitions — User and Product entities.

Adapted from MLOps Lesson 11 §3.1.
Demonstrates multiple entities (unlike practical_project/ which uses only User).
"""

from feast import Entity, ValueType

# Primary entity: customer
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="Unique customer identifier",
)

# Secondary entity: product (for product-level features)
product = Entity(
    name="product_id",
    value_type=ValueType.INT64,
    description="Product identifier",
)
