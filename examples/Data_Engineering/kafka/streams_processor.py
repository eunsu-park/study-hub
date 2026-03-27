"""
Kafka Streams Processor (Faust)
================================
Real-time stream processing with Faust (Python Kafka Streams).

Demonstrates:
- Stream filtering and mapping
- Stateful processing with Tables
- Windowed aggregations (tumbling, hopping)
- Stream-Table joins for enrichment
- HTTP API for querying state

Requirements:
    pip install faust-streaming

Usage:
    faust -A streams_processor worker -l info
"""

import faust
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# ── App Configuration ──────────────────────────────────────────────

app = faust.App(
    'ecommerce_streams',
    broker='kafka://localhost:9092',
    store='memory://',  # Use 'rocksdb://' in production
    topic_replication_factor=1,
    stream_wait_empty=False,
)

# ── Data Models ────────────────────────────────────────────────────

class Order(faust.Record):
    order_id: str
    user_id: str
    product_id: str
    amount: float
    currency: str = 'USD'
    timestamp: float = None


class UserProfile(faust.Record):
    user_id: str
    name: str
    tier: str  # bronze, silver, gold, platinum
    country: str


class OrderAlert(faust.Record):
    order_id: str
    user_id: str
    amount: float
    reason: str
    severity: str  # low, medium, high


# ── Topics ─────────────────────────────────────────────────────────

orders_topic = app.topic('orders', value_type=Order)
profiles_topic = app.topic('user_profiles', value_type=UserProfile)
alerts_topic = app.topic('order_alerts', value_type=OrderAlert)
high_value_topic = app.topic('high_value_orders', value_type=Order)

# ── State Tables ───────────────────────────────────────────────────

# User profiles table (latest profile per user_id)
user_profiles_table = app.Table(
    'user_profiles_table',
    default=None,
    partitions=4,
)

# Order counts per user (tumbling 5-minute window)
order_counts_5min = app.Table(
    'order_counts_5min',
    default=int,
).tumbling(
    size=timedelta(minutes=5),
    expires=timedelta(hours=1),
)

# Revenue per user (hopping window: 1-hour size, 10-minute step)
revenue_hourly = app.Table(
    'revenue_hourly',
    default=float,
).hopping(
    size=timedelta(hours=1),
    step=timedelta(minutes=10),
    expires=timedelta(hours=24),
)

# Global counters
total_orders = app.Table('total_orders', default=int)
total_revenue = app.Table('total_revenue', default=float)

# ── Stream Processors ─────────────────────────────────────────────

@app.agent(profiles_topic)
async def update_profiles(profiles):
    """Maintain user profiles table from changelog stream."""
    async for profile in profiles:
        user_profiles_table[profile.user_id] = profile
        logger.info(f"Updated profile for {profile.user_id}: {profile.name}")


@app.agent(orders_topic)
async def process_orders(orders):
    """Main order processing pipeline."""
    async for order in orders:
        # 1. Update windowed counters
        order_counts_5min[order.user_id] += 1
        revenue_hourly[order.user_id] += order.amount

        # 2. Update global counters
        total_orders['all'] += 1
        total_revenue['all'] += order.amount

        # 3. Enrich with user profile (stream-table join)
        profile = user_profiles_table.get(order.user_id)
        user_name = profile.name if profile else "Unknown"
        user_tier = profile.tier if profile else "unknown"

        # 4. High-value order detection
        if order.amount > 1000:
            await high_value_topic.send(value=order)
            alert = OrderAlert(
                order_id=order.order_id,
                user_id=order.user_id,
                amount=order.amount,
                reason=f"High-value order: ${order.amount:.2f}",
                severity="high" if order.amount > 5000 else "medium",
            )
            await alerts_topic.send(value=alert)
            logger.warning(f"ALERT: {alert.reason} by {user_name} ({user_tier})")

        # 5. Velocity detection (too many orders in short time)
        current_count = order_counts_5min[order.user_id].current()
        if current_count > 3:
            alert = OrderAlert(
                order_id=order.order_id,
                user_id=order.user_id,
                amount=order.amount,
                reason=f"High velocity: {current_count} orders in 5 min",
                severity="high",
            )
            await alerts_topic.send(value=alert)
            logger.warning(f"VELOCITY ALERT: {alert.reason}")

        logger.info(
            f"Order {order.order_id}: {user_name} ({user_tier}) "
            f"- ${order.amount:.2f} [total: {total_orders['all']}]"
        )


@app.agent(alerts_topic)
async def log_alerts(alerts):
    """Process and log all alerts."""
    async for alert in alerts:
        logger.error(
            f"ALERT [{alert.severity.upper()}] "
            f"Order {alert.order_id}: {alert.reason}"
        )


# ── HTTP API ───────────────────────────────────────────────────────

@app.page('/stats/')
async def get_stats(self, request):
    """GET /stats/ - Global statistics."""
    return self.json({
        'total_orders': total_orders.get('all', 0),
        'total_revenue': total_revenue.get('all', 0.0),
    })


@app.page('/user/{user_id}/')
async def get_user_stats(self, request, user_id):
    """GET /user/<user_id>/ - User-specific statistics."""
    profile = user_profiles_table.get(user_id)
    return self.json({
        'user_id': user_id,
        'profile': profile.to_representation() if profile else None,
        'orders_5min': order_counts_5min[user_id].current(),
    })


# ── Timer Tasks ────────────────────────────────────────────────────

@app.timer(interval=60.0)
async def periodic_report():
    """Print summary every 60 seconds."""
    orders = total_orders.get('all', 0)
    revenue = total_revenue.get('all', 0.0)
    if orders > 0:
        logger.info(
            f"[PERIODIC] Orders: {orders}, "
            f"Revenue: ${revenue:.2f}, "
            f"Avg: ${revenue/orders:.2f}"
        )


if __name__ == '__main__':
    app.main()
