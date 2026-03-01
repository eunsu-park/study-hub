"""
Feast entity definitions for the churn prediction project.

Adapted from MLOps Lesson 11 (Feature Stores) and Lesson 12 (Practical Project).
"""

from feast import Entity, ValueType

# Primary entity: identifies a unique customer
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="Unique customer identifier",
)
