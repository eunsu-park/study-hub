"""
Exercise Solutions: Lesson 02 - Data Modeling Basics

Covers:
  - Problem 1: Star Schema Design (online bookstore)
  - Problem 2: SCD Type 2 (customer tier changes)
"""

from datetime import datetime, date
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Problem 1: Star Schema Design
# Design a star schema for sales analysis of an online bookstore.
# Define the necessary fact and dimension tables.
# ---------------------------------------------------------------------------

@dataclass
class DimBook:
    """Book dimension: descriptive attributes about each book.

    Why a separate dimension?
    - Books rarely change (title, author, genre, publisher stay the same).
    - Queries can filter/group by any attribute without touching the fact table.
    """
    book_key: int
    book_id: str
    title: str
    author: str
    genre: str
    publisher: str
    publish_year: int
    price: float  # list price at time of dimension load


@dataclass
class DimCustomer:
    """Customer dimension: who bought the book."""
    customer_key: int
    customer_id: str
    name: str
    email: str
    city: str
    state: str
    country: str
    registration_date: str
    tier: str  # Bronze / Silver / Gold


@dataclass
class DimDate:
    """Date dimension: enables time-based analysis.

    Pre-generated rows (one per calendar day) allow SQL GROUP BY on
    year, quarter, month, day_of_week without date functions.
    """
    date_key: int  # YYYYMMDD integer for fast joins
    full_date: str
    year: int
    quarter: int
    month: int
    month_name: str
    day_of_week: int
    day_name: str
    is_weekend: bool
    is_holiday: bool


@dataclass
class DimPromotion:
    """Promotion dimension: discount and campaign context."""
    promotion_key: int
    promotion_id: str
    promotion_name: str
    discount_percent: float
    start_date: str
    end_date: str
    channel: str  # email, social, in-app


@dataclass
class FactBookSales:
    """Fact table: one row per book sale (line item grain).

    Design decisions:
    - Grain = one line item per order. This is the lowest useful grain
      because it lets us aggregate by any combination of dimensions.
    - Surrogate keys (xxx_key) reference dimension tables.
    - Measures are additive (quantity, revenue, discount_amount).
    """
    sale_key: int
    order_id: str
    date_key: int          # FK -> DimDate
    book_key: int          # FK -> DimBook
    customer_key: int      # FK -> DimCustomer
    promotion_key: int     # FK -> DimPromotion (0 = no promo)
    quantity: int
    unit_price: float
    discount_amount: float
    revenue: float         # quantity * unit_price - discount_amount


def build_star_schema_example():
    """Populate a small star schema with sample data and run analytic queries."""

    # -- Dimension data -------------------------------------------------------
    dates = [
        DimDate(20240101, "2024-01-01", 2024, 1, 1, "January", 1, "Monday", False, True),
        DimDate(20240102, "2024-01-02", 2024, 1, 1, "January", 2, "Tuesday", False, False),
        DimDate(20240115, "2024-01-15", 2024, 1, 1, "January", 1, "Monday", False, False),
        DimDate(20240201, "2024-02-01", 2024, 1, 2, "February", 4, "Thursday", False, False),
    ]

    books = [
        DimBook(1, "B001", "Clean Code", "Robert C. Martin", "Technology", "Prentice Hall", 2008, 39.99),
        DimBook(2, "B002", "Dune", "Frank Herbert", "Science Fiction", "Ace Books", 1965, 14.99),
        DimBook(3, "B003", "Sapiens", "Yuval Noah Harari", "Non-Fiction", "Harper", 2015, 22.99),
    ]

    customers = [
        DimCustomer(1, "C001", "Alice Kim", "alice@example.com", "Seoul", "Seoul", "KR", "2023-01-10", "Gold"),
        DimCustomer(2, "C002", "Bob Smith", "bob@example.com", "NYC", "NY", "US", "2023-06-20", "Silver"),
        DimCustomer(3, "C003", "Carol Lee", "carol@example.com", "London", "ENG", "UK", "2024-01-01", "Bronze"),
    ]

    promotions = [
        DimPromotion(0, "NONE", "No Promotion", 0.0, "", "", ""),
        DimPromotion(1, "P001", "New Year Sale", 10.0, "2024-01-01", "2024-01-05", "email"),
    ]

    # -- Fact data ------------------------------------------------------------
    facts = [
        FactBookSales(1, "ORD-001", 20240101, 1, 1, 1, 2, 39.99, 8.00, 71.98),
        FactBookSales(2, "ORD-001", 20240101, 2, 1, 1, 1, 14.99, 1.50, 13.49),
        FactBookSales(3, "ORD-002", 20240102, 3, 2, 0, 1, 22.99, 0.00, 22.99),
        FactBookSales(4, "ORD-003", 20240115, 1, 3, 0, 3, 39.99, 0.00, 119.97),
        FactBookSales(5, "ORD-004", 20240201, 2, 2, 0, 2, 14.99, 0.00, 29.98),
    ]

    # -- Analytic queries (simulated in Python) -------------------------------
    # Query 1: Total revenue by genre
    genre_revenue: dict[str, float] = {}
    book_map = {b.book_key: b for b in books}
    for f in facts:
        genre = book_map[f.book_key].genre
        genre_revenue[genre] = genre_revenue.get(genre, 0.0) + f.revenue

    print("Star Schema: Online Bookstore")
    print("-" * 50)
    print("\nQuery 1: Revenue by Genre")
    for genre, rev in sorted(genre_revenue.items(), key=lambda x: -x[1]):
        print(f"  {genre:20s}  ${rev:>10.2f}")

    # Query 2: Monthly revenue
    date_map = {d.date_key: d for d in dates}
    monthly_rev: dict[str, float] = {}
    for f in facts:
        month = date_map[f.date_key].month_name + f" {date_map[f.date_key].year}"
        monthly_rev[month] = monthly_rev.get(month, 0.0) + f.revenue

    print("\nQuery 2: Revenue by Month")
    for month, rev in monthly_rev.items():
        print(f"  {month:20s}  ${rev:>10.2f}")

    # Query 3: Top customers by revenue
    cust_map = {c.customer_key: c for c in customers}
    cust_rev: dict[str, float] = {}
    for f in facts:
        name = cust_map[f.customer_key].name
        cust_rev[name] = cust_rev.get(name, 0.0) + f.revenue

    print("\nQuery 3: Revenue by Customer")
    for name, rev in sorted(cust_rev.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}  ${rev:>10.2f}")

    return facts


# ---------------------------------------------------------------------------
# Problem 2: SCD Type 2
# Write SQL (and a pure-Python simulation) for preserving history when a
# customer's tier (Bronze -> Silver -> Gold) changes.
# ---------------------------------------------------------------------------

@dataclass
class SCDCustomerRecord:
    """A single SCD Type 2 row.

    SCD Type 2 stores full history by closing the old row (setting
    effective_end and is_current=False) and inserting a new current row.
    """
    surrogate_key: int
    customer_id: str
    name: str
    tier: str
    effective_start: str
    effective_end: str | None
    is_current: bool


class SCDType2Table:
    """In-memory simulation of an SCD Type 2 dimension table.

    Equivalent SQL for the MERGE logic:

        -- Close the old row
        UPDATE dim_customer
        SET effective_end = CURRENT_DATE - INTERVAL '1 day',
            is_current = FALSE
        WHERE customer_id = :cid
          AND is_current = TRUE
          AND tier != :new_tier;

        -- Insert the new row
        INSERT INTO dim_customer (customer_id, name, tier, effective_start, effective_end, is_current)
        SELECT :cid, :name, :new_tier, CURRENT_DATE, NULL, TRUE
        WHERE EXISTS (
            SELECT 1 FROM dim_customer
            WHERE customer_id = :cid AND is_current = FALSE
              AND effective_end = CURRENT_DATE - INTERVAL '1 day'
        );
    """

    def __init__(self):
        self.records: list[SCDCustomerRecord] = []
        self._next_key = 1

    def _get_next_key(self) -> int:
        key = self._next_key
        self._next_key += 1
        return key

    def initial_load(self, customers: list[dict], load_date: str) -> None:
        """Bulk-insert initial customer records."""
        for c in customers:
            self.records.append(SCDCustomerRecord(
                surrogate_key=self._get_next_key(),
                customer_id=c["customer_id"],
                name=c["name"],
                tier=c["tier"],
                effective_start=load_date,
                effective_end=None,
                is_current=True,
            ))

    def apply_change(self, customer_id: str, new_tier: str, change_date: str) -> None:
        """Apply a tier change using SCD Type 2 logic.

        Steps:
        1. Find the current row for this customer.
        2. If the tier has not changed, do nothing (no-op).
        3. Close the current row (set effective_end, is_current=False).
        4. Insert a new current row with the updated tier.
        """
        current_row = None
        for r in self.records:
            if r.customer_id == customer_id and r.is_current:
                current_row = r
                break

        if current_row is None:
            print(f"  [WARN] Customer {customer_id} not found -- skipping")
            return

        if current_row.tier == new_tier:
            print(f"  [SKIP] Customer {customer_id} tier unchanged ({new_tier})")
            return

        # Close the old row
        current_row.effective_end = change_date
        current_row.is_current = False

        # Insert the new row
        self.records.append(SCDCustomerRecord(
            surrogate_key=self._get_next_key(),
            customer_id=customer_id,
            name=current_row.name,
            tier=new_tier,
            effective_start=change_date,
            effective_end=None,
            is_current=True,
        ))
        print(f"  [SCD2] {customer_id}: {current_row.tier} -> {new_tier} on {change_date}")

    def display(self) -> None:
        print(f"\n{'SK':>3} {'CustID':<8} {'Name':<12} {'Tier':<8} {'Start':<12} {'End':<12} {'Current'}")
        print("-" * 75)
        for r in self.records:
            end = r.effective_end or "NULL"
            print(f"{r.surrogate_key:>3} {r.customer_id:<8} {r.name:<12} {r.tier:<8} "
                  f"{r.effective_start:<12} {end:<12} {r.is_current}")

    def query_as_of(self, customer_id: str, as_of_date: str) -> SCDCustomerRecord | None:
        """Point-in-time query: what was the customer's tier on a given date?"""
        for r in self.records:
            if r.customer_id != customer_id:
                continue
            end = r.effective_end or "9999-12-31"
            if r.effective_start <= as_of_date <= end:
                return r
        return None


def scd_type2_demo():
    """Demonstrate SCD Type 2 with tier changes."""
    table = SCDType2Table()

    # Initial load
    table.initial_load([
        {"customer_id": "C001", "name": "Alice", "tier": "Bronze"},
        {"customer_id": "C002", "name": "Bob", "tier": "Bronze"},
        {"customer_id": "C003", "name": "Carol", "tier": "Silver"},
    ], load_date="2024-01-01")

    print("\n--- Initial State ---")
    table.display()

    # Apply tier changes
    print("\n--- Applying Changes ---")
    table.apply_change("C001", "Silver", "2024-03-15")
    table.apply_change("C001", "Gold", "2024-07-01")
    table.apply_change("C002", "Silver", "2024-06-01")
    table.apply_change("C003", "Silver", "2024-08-01")  # no-op

    print("\n--- Final State ---")
    table.display()

    # Point-in-time queries
    print("\n--- Point-in-Time Queries ---")
    for cid, qdate in [("C001", "2024-02-01"), ("C001", "2024-05-01"), ("C001", "2024-09-01")]:
        rec = table.query_as_of(cid, qdate)
        tier = rec.tier if rec else "NOT FOUND"
        print(f"  {cid} on {qdate}: tier = {tier}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Star Schema Design (Online Bookstore)")
    print("=" * 70)
    build_star_schema_example()

    print()
    print("=" * 70)
    print("Problem 2: SCD Type 2 (Customer Tier Changes)")
    print("=" * 70)
    scd_type2_demo()
