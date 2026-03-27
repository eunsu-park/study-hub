"""
Exercises for Lesson 08: Query Processing
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers join cost calculation, selectivity estimation, heuristic optimization,
join algorithm selection, execution plan reading, and cost-based optimization.
"""

import math


# === Exercise 1: Cost Calculation ===
# Problem: Calculate join costs using different algorithms.

def exercise_1():
    """Calculate join costs for different algorithms."""
    # Given
    n_emp = 10_000    # employee tuples
    b_emp = 500       # employee blocks
    n_dept = 200      # department tuples
    b_dept = 10       # department blocks
    M = 12            # memory pages

    print("Given:")
    print(f"  employees: n={n_emp:,}, b={b_emp}, B+Tree index on emp_id (height=3)")
    print(f"  departments: n={n_dept}, b={b_dept}")
    print(f"  Memory: M={M} pages")
    print()

    # 1. Block Nested Loop Join (employees as outer)
    # Cost = ceil(b_outer / (M-2)) * b_inner + b_outer
    chunks_emp = math.ceil(b_emp / (M - 2))
    cost_bnlj_emp_outer = chunks_emp * b_dept + b_emp
    print("1. Block Nested Loop Join (employees as outer):")
    print(f"   Outer chunks: ceil({b_emp} / ({M}-2)) = ceil({b_emp}/{M-2}) = {chunks_emp}")
    print(f"   Cost: {chunks_emp} x {b_dept} + {b_emp} = {cost_bnlj_emp_outer} block transfers")
    print()

    # 2. Block Nested Loop Join (departments as outer)
    chunks_dept = math.ceil(b_dept / (M - 2))
    cost_bnlj_dept_outer = chunks_dept * b_emp + b_dept
    print("2. Block Nested Loop Join (departments as outer):")
    print(f"   Outer chunks: ceil({b_dept} / ({M}-2)) = ceil({b_dept}/{M-2}) = {chunks_dept}")
    print(f"   Cost: {chunks_dept} x {b_emp} + {b_dept} = {cost_bnlj_dept_outer} block transfers")
    print()

    # 3. Hash Join (departments as build)
    # departments (10 blocks) fits in M=12 pages
    cost_hash = b_dept + b_emp  # single pass: build + probe
    print("3. Hash Join (departments as build):")
    print(f"   departments ({b_dept} blocks) fits in {M}-page memory: {'YES' if b_dept <= M else 'NO'}")
    print(f"   Cost: {b_dept} + {b_emp} = {cost_hash} block transfers (single-pass)")
    print()

    # Summary
    print("Summary:")
    print(f"  BNLJ (emp outer):   {cost_bnlj_emp_outer:,} transfers")
    print(f"  BNLJ (dept outer):  {cost_bnlj_dept_outer:,} transfers")
    print(f"  Hash join:          {cost_hash:,} transfers")
    print()
    print("  Hash join and dept-outer BNLJ are comparable.")
    print("  Hash join preferred due to better cache behavior.")


# === Exercise 2: Selectivity Estimation ===
# Problem: Estimate result sizes using selectivity formulas.

def exercise_2():
    """Estimate selectivity and result sizes."""
    n = 10_000        # total rows
    salary_min = 30_000
    salary_max = 150_000
    v_salary = 2_000  # distinct salary values
    v_dept = 50       # distinct dept_id values
    v_city = 100      # distinct city values

    print("Given: employees table, n=10,000")
    print(f"  salary: min={salary_min:,}, max={salary_max:,}, V(salary)={v_salary:,}")
    print(f"  dept_id: V(dept_id)={v_dept}")
    print(f"  city: V(city)={v_city}")
    print()

    # 1. sigma_{salary = 75000}(employees)
    sel_1 = 1 / v_salary
    result_1 = n * sel_1
    print("1. sigma_{salary = 75000}:")
    print(f"   Selectivity = 1/V(salary) = 1/{v_salary} = {sel_1:.6f}")
    print(f"   Result = {n} x {sel_1:.6f} = {result_1:.0f} tuples")
    print()

    # 2. sigma_{salary > 100000}(employees)
    sel_2 = (salary_max - 100_000) / (salary_max - salary_min)
    result_2 = n * sel_2
    print("2. sigma_{salary > 100000}:")
    print(f"   Selectivity = (max - 100000) / (max - min) = ({salary_max}-100000)/({salary_max}-{salary_min})")
    print(f"              = {salary_max - 100_000}/{salary_max - salary_min} = {sel_2:.4f}")
    print(f"   Result = {n} x {sel_2:.4f} = {result_2:.0f} tuples")
    print()

    # 3. sigma_{dept_id = 5 AND city = 'Boston'}(employees)
    sel_dept = 1 / v_dept
    sel_city = 1 / v_city
    sel_3 = sel_dept * sel_city  # independence assumption
    result_3 = n * sel_3
    print("3. sigma_{dept_id = 5 AND city = 'Boston'} (assuming independence):")
    print(f"   sel(dept_id=5) = 1/{v_dept} = {sel_dept:.4f}")
    print(f"   sel(city='Boston') = 1/{v_city} = {sel_city:.4f}")
    print(f"   Combined selectivity = {sel_dept:.4f} x {sel_city:.4f} = {sel_3:.6f}")
    print(f"   Result = {n} x {sel_3:.6f} = {result_3:.0f} tuples")
    print()
    print("Note: Independence assumption may not hold (correlated attributes).")
    print("Histograms provide more accurate estimates in practice.")


# === Exercise 3: Heuristic Optimization ===
# Problem: Optimize a query tree using heuristic rules.

def exercise_3():
    """Heuristic query optimization."""
    print("Query:")
    print("  SELECT p.product_name, c.category_name")
    print("  FROM products p, categories c, order_items oi")
    print("  WHERE p.category_id = c.category_id")
    print("    AND p.product_id = oi.product_id")
    print("    AND oi.quantity > 10")
    print("    AND c.category_name = 'Electronics';")
    print()

    print("Initial (unoptimized) tree:")
    print("""
    pi_{product_name, category_name}
        |
    sigma_{all conditions}
        |
        x  (Cartesian product)
       / \\
      x   oi
     / \\
    p   c
""")

    print("Optimized tree:")
    print("""
    pi_{product_name, category_name}
        |
    |><|_{p.cat_id = c.cat_id}
       / \\
      /   sigma_{cat_name='Electronics'}(c)
     |
    |><|_{p.prod_id = oi.prod_id}
       / \\
      p   sigma_{qty > 10}(oi)
""")

    print("Optimization rules applied:")
    rules = [
        ("1. Decompose conjunctive selection",
         "Break sigma_{A AND B AND C AND D} into individual selections."),
        ("2. Push selections down",
         "sigma_{qty>10} pushed to order_items (before any join)."),
        ("3. Push selections down",
         "sigma_{cat_name='Electronics'} pushed to categories (before join)."),
        ("4. Replace Cartesian products with joins",
         "x + equality condition -> natural join / equi-join."),
        ("5. Project early",
         "Only needed columns pass through each join."),
    ]
    for rule, desc in rules:
        print(f"  {rule}: {desc}")

    print()
    print("Key gain:")
    print("  categories filtered to ~1 row ('Electronics') before join.")
    print("  order_items filtered to subset (qty>10) before join.")
    print("  This drastically reduces intermediate result sizes.")


# === Exercise 4: Join Algorithm Selection ===
# Problem: Choose the best join algorithm for each scenario.

def exercise_4():
    """Join algorithm selection for different scenarios."""
    scenarios = [
        {
            "scenario": "1. 100-row lookup table JOIN 10M-row fact table (index on fact table join column)",
            "algorithm": "Indexed Nested Loop Join",
            "reason": [
                "Lookup table (100 rows) as outer -- tiny number of iterations.",
                "Each outer row uses the index on fact table for O(log n) lookup.",
                "Cost: 100 x ~4 I/Os (tree height) = 400 I/Os.",
                "Much better than scanning 10M rows."
            ]
        },
        {
            "scenario": "2. Two 1M-row tables, neither sorted, 1GB buffer pool",
            "algorithm": "Hash Join",
            "reason": [
                "Plenty of memory: one table's hash table fits entirely in RAM.",
                "Cost: read both tables once -- optimal I/O.",
                "No index needed, no sorting overhead.",
                "Hash join is the workhorse for unsorted equi-joins."
            ]
        },
        {
            "scenario": "3. Two 1M-row tables, range condition (r.date BETWEEN s.start AND s.end)",
            "algorithm": "Sort-Merge Join (or Block Nested Loop)",
            "reason": [
                "Hash join does NOT work for range conditions (can't hash ranges).",
                "Sort-merge: sort both on date columns, then merge with range check.",
                "Block NLJ is fallback if sorting is too expensive.",
                "Indexed NLJ possible if index exists on date columns."
            ]
        },
        {
            "scenario": "4. Two tables both already sorted on the join column",
            "algorithm": "Sort-Merge Join (skip sort phase)",
            "reason": [
                "Sort phase already done! Only merge phase needed.",
                "Cost: b_r + b_s (single pass through each table).",
                "This is the optimal case for sort-merge join.",
                "No additional memory needed beyond streaming buffers."
            ]
        }
    ]

    for s in scenarios:
        print(f"{s['scenario']}")
        print(f"  Best algorithm: {s['algorithm']}")
        for r in s["reason"]:
            print(f"    - {r}")
        print()


# === Exercise 5: Reading Execution Plans ===
# Problem: Analyze a PostgreSQL EXPLAIN output.

def exercise_5():
    """Analyze PostgreSQL EXPLAIN output."""
    plan = """
Nested Loop  (cost=0.29..8.33 rows=1 width=64)
  ->  Index Scan using idx_emp_id on employees  (cost=0.29..4.30 rows=1 width=40)
        Index Cond: (emp_id = 42)
  ->  Seq Scan on departments  (cost=0.00..1.62 rows=1 width=24)
        Filter: (dept_id = employees.dept_id)
        Rows Removed by Filter: 49
"""
    print("EXPLAIN output:")
    print(plan)

    answers = [
        ("1. Join algorithm?", "Nested Loop Join"),
        ("2. Outer (driving) table?",
         "employees (listed first under Nested Loop). It drives the loop."),
        ("3. Why index scan on employees?",
         "emp_id = 42 is a highly selective equality predicate. "
         "The index finds exactly 1 row (rows=1). Full scan would be wasteful."),
        ("4. Why sequential scan on departments?",
         "departments is small (~50 rows, ~2 pages). With only 1 outer row, "
         "the seq scan runs once. An index lookup has overhead for such a tiny table."),
        ("5. Total estimated cost?",
         "8.33 (in PostgreSQL cost units, where 1.0 ~ a sequential page read). "
         "Very cheap: 1 index lookup + 1 small table scan."),
    ]

    print("Answers:")
    for q, a in answers:
        print(f"  {q}")
        print(f"    {a}")
        print()


# === Exercise 6: Equivalence Rules ===
# Problem: Show that pushing selection through join is equivalent.

def exercise_6():
    """Prove equivalence of selection push-down through join."""
    print("Expression A: sigma_{dept='CS'}(employees |><| departments)")
    print("Expression B: employees |><| sigma_{dept='CS'}(departments)")
    print()

    print("Proof of equivalence (Rule 6: Push selection through join):")
    print("  The predicate dept='CS' involves only attributes of 'departments'.")
    print("  By the selection-join commutativity rule:")
    print("    sigma_{c}(R |><| S) = R |><| sigma_{c}(S)")
    print("  when predicate c references only S's attributes.")
    print()

    # Demonstrate with concrete numbers
    print("Efficiency comparison:")
    n_emp = 10_000
    n_dept = 50
    dept_cs = 1  # 'CS' is one department
    emp_cs = n_emp // n_dept  # ~200 CS employees

    print(f"  Expression A (filter after join):")
    print(f"    Join: {n_emp:,} employees x {n_dept} departments -> {n_emp:,} matched rows")
    print(f"    Filter: {n_emp:,} rows -> {emp_cs} CS rows")
    print(f"    Intermediate result size: {n_emp:,} rows")
    print()

    print(f"  Expression B (filter before join):")
    print(f"    Filter departments: {n_dept} -> {dept_cs} row (only CS)")
    print(f"    Join: {n_emp:,} employees x {dept_cs} department -> {emp_cs} rows")
    print(f"    Intermediate result size: {dept_cs} row(s)")
    print()

    print(f"  Expression B is MUCH more efficient:")
    print(f"    Intermediate result: {dept_cs} vs {n_emp:,} rows")
    print(f"    Join comparisons: ~{n_emp:,} vs ~{n_emp * n_dept:,}")


# === Exercise 7: Cost-Based Optimization ===
# Problem: Compare join orderings for three-way join.

def exercise_7():
    """Cost-based optimization: compare join orderings."""
    # Given
    n_o, b_o = 100_000, 5_000
    n_c, b_c = 10_000, 500
    n_p, b_p = 1_000, 50
    M = 100

    print("Given:")
    print(f"  orders (o):    n={n_o:,}, b={b_o:,}")
    print(f"  customers (c): n={n_c:,}, b={b_c}")
    print(f"  products (p):  n={n_p:,}, b={b_p}")
    print(f"  Memory: M={M} pages")
    print(f"  Join predicates: o.cust_id = c.cust_id AND o.prod_id = p.prod_id")
    print()

    # Plan A: (orders |><| customers) |><| products
    print("Plan A: (orders |><| customers) |><| products")
    print()

    # Step 1: orders |><| customers
    # customers (500 blocks) > 100 pages -> Grace hash join
    cost_a1 = 3 * (b_o + b_c)
    result_a1_rows = n_o  # each order has one customer
    result_a1_blocks = b_o
    print(f"  Step 1: orders |><| customers (Grace hash join, customers build)")
    print(f"    customers ({b_c} blocks) > {M} pages -> Grace hash join")
    print(f"    Cost: 3 x ({b_o} + {b_c}) = {cost_a1:,} transfers")
    print(f"    Result: ~{result_a1_rows:,} rows, ~{result_a1_blocks:,} blocks")
    print()

    # Step 2: result |><| products
    # products (50 blocks) < 100 pages -> in-memory hash join
    cost_a2 = result_a1_blocks + b_p
    print(f"  Step 2: result |><| products (in-memory hash join, products build)")
    print(f"    products ({b_p} blocks) < {M} pages -> in-memory hash join")
    print(f"    Cost: {result_a1_blocks:,} + {b_p} = {cost_a2:,} transfers")
    print()

    cost_a_total = cost_a1 + cost_a2
    print(f"  Total Plan A: {cost_a1:,} + {cost_a2:,} = {cost_a_total:,} transfers")
    print()

    # Plan B: (orders |><| products) |><| customers
    print("Plan B: (orders |><| products) |><| customers")
    print()

    # Step 1: orders |><| products
    # products (50 blocks) < 100 pages -> in-memory hash join
    cost_b1 = b_o + b_p
    result_b1_rows = n_o
    result_b1_blocks = b_o
    print(f"  Step 1: orders |><| products (in-memory hash join, products build)")
    print(f"    products ({b_p} blocks) < {M} pages -> in-memory hash join")
    print(f"    Cost: {b_o:,} + {b_p} = {cost_b1:,} transfers")
    print(f"    Result: ~{result_b1_rows:,} rows, ~{result_b1_blocks:,} blocks")
    print()

    # Step 2: result |><| customers
    # customers (500 blocks) > 100 pages -> Grace hash join
    cost_b2 = 3 * (result_b1_blocks + b_c)
    print(f"  Step 2: result |><| customers (Grace hash join, customers build)")
    print(f"    customers ({b_c} blocks) > {M} pages -> Grace hash join")
    print(f"    Cost: 3 x ({result_b1_blocks:,} + {b_c}) = {cost_b2:,} transfers")
    print()

    cost_b_total = cost_b1 + cost_b2
    print(f"  Total Plan B: {cost_b1:,} + {cost_b2:,} = {cost_b_total:,} transfers")
    print()

    # Comparison
    print("Comparison:")
    print(f"  Plan A: {cost_a_total:,} transfers")
    print(f"  Plan B: {cost_b_total:,} transfers")
    print(f"  Difference: {abs(cost_a_total - cost_b_total)} transfers")
    print()

    if cost_a_total == cost_b_total:
        print("  Same total I/O! But Plan B is slightly better because:")
        print("    - Step 1 uses in-memory hash join (fewer seeks, better cache)")
        print("    - Intermediate result from Step 1 may be pipelined into Step 2")
    else:
        winner = "A" if cost_a_total < cost_b_total else "B"
        print(f"  Plan {winner} wins.")

    print()
    print("  Ideal Plan C: If M >= 550 pages (both small tables in memory):")
    cost_c = b_o + b_c + b_p
    print(f"    Scan orders once, probe both hash tables.")
    print(f"    Cost: {b_o:,} + {b_c} + {b_p} = {cost_c:,} transfers")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Cost Calculation ===")
    print("=" * 70)
    exercise_1()

    print("=" * 70)
    print("=== Exercise 2: Selectivity Estimation ===")
    print("=" * 70)
    exercise_2()

    print("=" * 70)
    print("=== Exercise 3: Heuristic Optimization ===")
    print("=" * 70)
    exercise_3()

    print("=" * 70)
    print("=== Exercise 4: Join Algorithm Selection ===")
    print("=" * 70)
    exercise_4()

    print("=" * 70)
    print("=== Exercise 5: Reading Execution Plans ===")
    print("=" * 70)
    exercise_5()

    print("=" * 70)
    print("=== Exercise 6: Equivalence Rules ===")
    print("=" * 70)
    exercise_6()

    print("=" * 70)
    print("=== Exercise 7: Cost-Based Optimization ===")
    print("=" * 70)
    exercise_7()

    print("\nAll exercises completed!")
