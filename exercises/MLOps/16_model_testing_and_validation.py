"""
Exercise Solutions: Model Testing and Validation
===========================================
Lesson 16 from MLOps topic.

Exercises
---------
1. Data Validation Suite — Implement comprehensive data validation with
   schema checks, statistical tests, and anomaly detection.
2. Behavioral Test Suite — Build behavioral tests using the MFT/INV/DIR
   framework (Minimum Functionality, Invariance, Directional Expectation).
3. Fairness Audit Report — Conduct a fairness audit across demographic
   groups with multiple fairness metrics.
4. CI/CD Integration — Design automated model testing gates for a CI/CD
   pipeline with pass/fail criteria.
5. Shadow Deployment Simulator — Simulate shadow deployment comparing a
   new model against production with statistical significance testing.
"""

import math
import random
import json
import statistics
from datetime import datetime, timedelta


# ============================================================
# Exercise 1: Data Validation Suite
# ============================================================

def exercise_1_data_validation():
    """Implement comprehensive data validation.

    Validation layers:
    1. Schema validation (columns, types, required fields)
    2. Statistical validation (distribution properties)
    3. Cross-feature validation (logical constraints between features)
    4. Temporal validation (data freshness, ordering)
    """

    class DataValidationSuite:
        def __init__(self):
            self.checks = []

        def _add_result(self, category, name, passed, details):
            self.checks.append({
                "category": category,
                "name": name,
                "passed": passed,
                "details": details,
            })

        def validate_schema(self, data, schema):
            """Validate data schema."""
            if not data:
                self._add_result("schema", "non_empty", False, "Dataset is empty")
                return

            self._add_result("schema", "non_empty", True, f"{len(data)} rows")

            # Column presence
            actual_cols = set(data[0].keys())
            expected_cols = set(schema.keys())
            missing = expected_cols - actual_cols
            extra = actual_cols - expected_cols
            self._add_result("schema", "columns_present",
                             len(missing) == 0,
                             f"Missing: {missing}" if missing else "All columns present")

            # Type checking
            type_errors = 0
            for col, spec in schema.items():
                if col not in actual_cols:
                    continue
                expected_type = spec.get("type")
                sample_vals = [row[col] for row in data[:100] if row.get(col) is not None]
                wrong_type = [v for v in sample_vals if not isinstance(v, expected_type)]
                if wrong_type:
                    type_errors += 1
                    self._add_result("schema", f"type_{col}", False,
                                     f"Expected {expected_type.__name__}, "
                                     f"got {type(wrong_type[0]).__name__}")
                else:
                    self._add_result("schema", f"type_{col}", True, "Type OK")

        def validate_statistics(self, data, reference_stats, tolerance_sigma=3):
            """Validate statistical properties against reference."""
            for col, ref in reference_stats.items():
                values = [row[col] for row in data if row.get(col) is not None
                          and isinstance(row[col], (int, float))]
                if not values:
                    self._add_result("statistics", f"stats_{col}", False, "No values")
                    continue

                actual_mean = sum(values) / len(values)
                actual_std = math.sqrt(sum((v - actual_mean) ** 2 for v in values)
                                       / max(len(values) - 1, 1))

                # Mean drift in terms of reference standard deviation
                mean_drift = abs(actual_mean - ref["mean"]) / max(ref["std"], 1e-8)
                mean_ok = mean_drift < tolerance_sigma

                # Variance ratio test
                std_ratio = actual_std / max(ref["std"], 1e-8)
                std_ok = 0.3 < std_ratio < 3.0

                passed = mean_ok and std_ok
                self._add_result("statistics", f"stats_{col}", passed,
                                 f"mean_drift={mean_drift:.2f}σ, std_ratio={std_ratio:.2f}")

                # Outlier check
                n_outliers = sum(1 for v in values
                                 if abs(v - actual_mean) > 4 * actual_std)
                outlier_rate = n_outliers / len(values)
                self._add_result("statistics", f"outliers_{col}",
                                 outlier_rate < 0.01,
                                 f"{n_outliers} outliers ({outlier_rate:.2%})")

        def validate_cross_feature(self, data, rules):
            """Validate logical constraints between features."""
            for rule_name, rule_fn in rules.items():
                violations = 0
                for row in data:
                    if not rule_fn(row):
                        violations += 1
                violation_rate = violations / max(len(data), 1)
                self._add_result("cross_feature", rule_name,
                                 violation_rate < 0.01,
                                 f"{violations} violations ({violation_rate:.2%})")

        def validate_completeness(self, data, max_null_rate=0.05):
            """Check null rates per column."""
            for col in data[0].keys() if data else []:
                null_count = sum(1 for row in data if row.get(col) is None)
                null_rate = null_count / len(data) if data else 0
                self._add_result("completeness", f"null_{col}",
                                 null_rate <= max_null_rate,
                                 f"{null_count} nulls ({null_rate:.2%})")

        def summary(self):
            total = len(self.checks)
            passed = sum(1 for c in self.checks if c["passed"])
            by_category = {}
            for c in self.checks:
                cat = c["category"]
                if cat not in by_category:
                    by_category[cat] = {"passed": 0, "failed": 0}
                if c["passed"]:
                    by_category[cat]["passed"] += 1
                else:
                    by_category[cat]["failed"] += 1
            return {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "by_category": by_category,
            }

    # --- Generate test data ---
    random.seed(42)
    data = []
    for _ in range(1000):
        age = random.randint(18, 85)
        income = max(0, round(20000 + age * 500 + random.gauss(0, 15000), 2))
        tenure = random.randint(0, min(age - 18, 40))
        charges = round(30 + tenure * 1.5 + random.gauss(20, 10), 2)
        row = {
            "age": age,
            "income": income,
            "tenure_years": tenure,
            "monthly_charges": charges,
            "churned": random.choice([0, 0, 0, 1]),
        }
        if random.random() < 0.015:
            row["income"] = None  # Inject some nulls
        data.append(row)

    # --- Run validation ---
    suite = DataValidationSuite()

    schema = {
        "age": {"type": int},
        "income": {"type": float},
        "tenure_years": {"type": int},
        "monthly_charges": {"type": float},
        "churned": {"type": int},
    }
    suite.validate_schema(data, schema)

    ref_stats = {
        "age": {"mean": 50, "std": 18},
        "income": {"mean": 45000, "std": 15000},
        "monthly_charges": {"mean": 65, "std": 15},
    }
    suite.validate_statistics(data, ref_stats)

    cross_rules = {
        "tenure_less_than_age": lambda r: (r.get("tenure_years", 0) <=
                                           r.get("age", 100) - 18),
        "charges_positive": lambda r: (r.get("monthly_charges", 0) > 0),
        "valid_churn_label": lambda r: r.get("churned") in (0, 1),
    }
    suite.validate_cross_feature(data, cross_rules)
    suite.validate_completeness(data)

    # Display
    print("Data Validation Suite")
    print("=" * 60)

    for check in suite.checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] ({check['category']}) {check['name']}: {check['details']}")

    summary = suite.summary()
    print(f"\n  Summary: {summary['passed']}/{summary['total']} passed")
    for cat, counts in summary["by_category"].items():
        print(f"    {cat}: {counts['passed']} passed, {counts['failed']} failed")

    return suite


# ============================================================
# Exercise 2: Behavioral Test Suite
# ============================================================

def exercise_2_behavioral_tests():
    """Build behavioral tests using MFT/INV/DIR framework.

    - MFT (Minimum Functionality Test): Simple cases the model MUST get right
    - INV (Invariance Test): Output should NOT change for these input changes
    - DIR (Directional Expectation Test): Output should change in a specific
      direction for these input changes
    """

    class BehavioralTestSuite:
        def __init__(self, predict_fn):
            self.predict_fn = predict_fn
            self.results = {"MFT": [], "INV": [], "DIR": []}

        def mft(self, name, inputs, expected_outputs):
            """Minimum Functionality Test."""
            passed = 0
            total = len(inputs)
            failures = []

            for inp, expected in zip(inputs, expected_outputs):
                actual = self.predict_fn(inp)
                if actual == expected:
                    passed += 1
                else:
                    failures.append({"input": inp, "expected": expected, "actual": actual})

            result = {
                "name": name,
                "passed": passed,
                "total": total,
                "pass_rate": round(passed / total, 4) if total > 0 else 0,
                "failures": failures[:3],  # Show first 3 failures
            }
            self.results["MFT"].append(result)
            return result

        def inv(self, name, original_inputs, perturbed_inputs):
            """Invariance Test: predictions should be the same."""
            same = 0
            total = len(original_inputs)
            failures = []

            for orig, pert in zip(original_inputs, perturbed_inputs):
                orig_pred = self.predict_fn(orig)
                pert_pred = self.predict_fn(pert)
                if orig_pred == pert_pred:
                    same += 1
                else:
                    failures.append({
                        "original": orig,
                        "perturbed": pert,
                        "orig_pred": orig_pred,
                        "pert_pred": pert_pred,
                    })

            result = {
                "name": name,
                "invariant": same,
                "total": total,
                "invariance_rate": round(same / total, 4) if total > 0 else 0,
                "failures": failures[:3],
            }
            self.results["INV"].append(result)
            return result

        def dir(self, name, pairs, expected_direction):
            """Directional Expectation Test.
            expected_direction: 'increase' or 'decrease'
            """
            correct = 0
            total = len(pairs)

            for base_input, modified_input in pairs:
                base_score = self.predict_fn(base_input, return_prob=True)
                mod_score = self.predict_fn(modified_input, return_prob=True)

                if expected_direction == "increase":
                    if mod_score > base_score:
                        correct += 1
                elif expected_direction == "decrease":
                    if mod_score < base_score:
                        correct += 1

            result = {
                "name": name,
                "correct": correct,
                "total": total,
                "direction_rate": round(correct / total, 4) if total > 0 else 0,
                "expected_direction": expected_direction,
            }
            self.results["DIR"].append(result)
            return result

        def summary(self):
            total_tests = sum(len(v) for v in self.results.values())
            all_pass = True
            for category, tests in self.results.items():
                for test in tests:
                    rate_key = {
                        "MFT": "pass_rate",
                        "INV": "invariance_rate",
                        "DIR": "direction_rate",
                    }[category]
                    if test[rate_key] < 0.80:
                        all_pass = False
            return {"total_test_groups": total_tests, "all_above_80pct": all_pass}

    # --- Simple churn prediction model ---
    random.seed(42)
    weights = [0.02, -0.00001, 0.01, -0.01, 0.5]  # age, income, charges, tenure, support_calls
    bias = -2.0

    def predict(features, return_prob=False):
        z = sum(w * f for w, f in zip(weights, features)) + bias
        z = max(-500, min(500, z))
        prob = 1 / (1 + math.exp(-z))
        if return_prob:
            return prob
        return 1 if prob >= 0.5 else 0

    suite = BehavioralTestSuite(predict)

    print("Behavioral Test Suite")
    print("=" * 60)

    # MFT: Obvious churn cases
    print("\n  MFT — Minimum Functionality Tests")
    print("-" * 40)
    suite.mft("obvious_churn", [
        [70, 30000, 100, 1, 5],   # Old, low income, high charges, short tenure, many calls
        [65, 25000, 90, 2, 4],
    ], [1, 1])

    suite.mft("obvious_retain", [
        [30, 80000, 40, 15, 0],   # Young, high income, low charges, long tenure, no calls
        [35, 90000, 35, 20, 0],
    ], [0, 0])

    for test in suite.results["MFT"]:
        print(f"    {test['name']}: {test['passed']}/{test['total']} "
              f"({test['pass_rate']:.0%})")

    # INV: Name change should not affect prediction
    print("\n  INV — Invariance Tests")
    print("-" * 40)
    # Income rounding should not change prediction
    suite.inv("income_rounding", [
        [40, 50000, 60, 10, 2],
        [40, 50100, 60, 10, 2],
    ], [
        [40, 50001, 60, 10, 2],
        [40, 50099, 60, 10, 2],
    ])

    for test in suite.results["INV"]:
        print(f"    {test['name']}: {test['invariant']}/{test['total']} invariant "
              f"({test['invariance_rate']:.0%})")

    # DIR: More support calls should increase churn probability
    print("\n  DIR — Directional Expectation Tests")
    print("-" * 40)
    suite.dir("more_support_calls_increases_churn", [
        ([40, 50000, 60, 10, 1], [40, 50000, 60, 10, 4]),
        ([50, 40000, 70, 5, 0], [50, 40000, 70, 5, 3]),
        ([30, 60000, 50, 8, 2], [30, 60000, 50, 8, 5]),
    ], "increase")

    suite.dir("longer_tenure_decreases_churn", [
        ([40, 50000, 60, 2, 2], [40, 50000, 60, 15, 2]),
        ([50, 40000, 70, 1, 1], [50, 40000, 70, 10, 1]),
    ], "decrease")

    for test in suite.results["DIR"]:
        print(f"    {test['name']}: {test['correct']}/{test['total']} correct "
              f"({test['direction_rate']:.0%}, expected: {test['expected_direction']})")

    summary = suite.summary()
    print(f"\n  Overall: {summary['total_test_groups']} test groups, "
          f"all >80%: {summary['all_above_80pct']}")

    return suite


# ============================================================
# Exercise 3: Fairness Audit Report
# ============================================================

def exercise_3_fairness_audit():
    """Conduct a fairness audit with multiple fairness metrics.

    Metrics:
    - Demographic Parity: P(Y=1|A=a) should be equal across groups
    - Equal Opportunity: TPR should be equal across groups
    - Equalized Odds: Both TPR and FPR should be equal
    - Calibration: P(Y=1|score=s, A=a) should be equal
    """

    random.seed(42)

    # Generate predictions with built-in bias
    groups = {
        "group_A": {"n": 300, "base_score": 0.4, "label_rate": 0.3},
        "group_B": {"n": 300, "base_score": 0.5, "label_rate": 0.3},
        "group_C": {"n": 200, "base_score": 0.6, "label_rate": 0.3},
    }

    data = []
    for group_name, config in groups.items():
        for _ in range(config["n"]):
            score = max(0, min(1, config["base_score"] + random.gauss(0, 0.25)))
            pred = 1 if score >= 0.5 else 0
            label = 1 if random.random() < config["label_rate"] else 0
            # Bias: model is more accurate for group_A
            if group_name == "group_A" and label == 1:
                score = max(0, min(1, score + 0.15))
                pred = 1 if score >= 0.5 else 0

            data.append({
                "group": group_name,
                "score": round(score, 4),
                "prediction": pred,
                "label": label,
            })

    # --- Calculate fairness metrics ---
    def calc_group_metrics(data, group):
        group_data = [d for d in data if d["group"] == group]
        n = len(group_data)
        tp = sum(1 for d in group_data if d["prediction"] == 1 and d["label"] == 1)
        fp = sum(1 for d in group_data if d["prediction"] == 1 and d["label"] == 0)
        tn = sum(1 for d in group_data if d["prediction"] == 0 and d["label"] == 0)
        fn = sum(1 for d in group_data if d["prediction"] == 0 and d["label"] == 1)

        positive_rate = (tp + fp) / n if n > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        accuracy = (tp + tn) / n if n > 0 else 0

        return {
            "n": n,
            "positive_rate": round(positive_rate, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "accuracy": round(accuracy, 4),
        }

    print("Fairness Audit Report")
    print("=" * 60)

    group_metrics = {}
    for group in groups:
        group_metrics[group] = calc_group_metrics(data, group)

    # Display per-group metrics
    print(f"\n  {'Group':<12s} {'N':>5s} {'Pos Rate':>9s} {'TPR':>6s} "
          f"{'FPR':>6s} {'Accuracy':>9s}")
    print(f"  {'-'*50}")
    for group, m in group_metrics.items():
        print(f"  {group:<12s} {m['n']:>5d} {m['positive_rate']:>9.4f} "
              f"{m['tpr']:>6.4f} {m['fpr']:>6.4f} {m['accuracy']:>9.4f}")

    # Fairness metrics
    print(f"\n  Fairness Metrics:")
    print(f"  {'-'*50}")

    # Demographic Parity
    pos_rates = [m["positive_rate"] for m in group_metrics.values()]
    dp_diff = max(pos_rates) - min(pos_rates)
    dp_pass = dp_diff < 0.1
    print(f"  Demographic Parity Difference: {dp_diff:.4f} "
          f"{'PASS' if dp_pass else 'FAIL'} (threshold: 0.1)")

    # Equal Opportunity (TPR parity)
    tprs = [m["tpr"] for m in group_metrics.values()]
    eo_diff = max(tprs) - min(tprs)
    eo_pass = eo_diff < 0.1
    print(f"  Equal Opportunity Difference:  {eo_diff:.4f} "
          f"{'PASS' if eo_pass else 'FAIL'} (threshold: 0.1)")

    # Equalized Odds (TPR + FPR parity)
    fprs = [m["fpr"] for m in group_metrics.values()]
    fpr_diff = max(fprs) - min(fprs)
    eq_odds_pass = eo_diff < 0.1 and fpr_diff < 0.1
    print(f"  Equalized Odds (FPR Diff):     {fpr_diff:.4f} "
          f"{'PASS' if eq_odds_pass else 'FAIL'} (threshold: 0.1)")

    # Disparate Impact Ratio
    min_pos = min(pos_rates)
    max_pos = max(pos_rates)
    di_ratio = min_pos / max_pos if max_pos > 0 else 0
    di_pass = di_ratio >= 0.8
    print(f"  Disparate Impact Ratio:        {di_ratio:.4f} "
          f"{'PASS' if di_pass else 'FAIL'} (threshold: 0.8)")

    # Overall
    overall = dp_pass and eo_pass and eq_odds_pass and di_pass
    print(f"\n  Overall Fairness: {'PASS' if overall else 'FAIL'}")

    if not overall:
        print(f"\n  Recommendations:")
        if not dp_pass:
            print(f"    - Investigate positive rate disparity across groups")
        if not eo_pass:
            print(f"    - TPR varies significantly — consider rebalancing training data")
        if not di_pass:
            print(f"    - Disparate impact detected — may need post-processing calibration")

    return group_metrics


# ============================================================
# Exercise 4: CI/CD Integration
# ============================================================

def exercise_4_cicd_integration():
    """Design automated model testing gates for CI/CD."""

    class ModelTestingGate:
        def __init__(self, name):
            self.name = name
            self.gates = []
            self.results = []

        def add_gate(self, name, check_fn, blocking=True, description=""):
            self.gates.append({
                "name": name,
                "check": check_fn,
                "blocking": blocking,
                "description": description,
            })

        def execute(self, context):
            self.results = []
            all_blocking_passed = True

            for gate in self.gates:
                result = gate["check"](context)
                self.results.append({
                    "gate": gate["name"],
                    "passed": result["passed"],
                    "blocking": gate["blocking"],
                    "details": result.get("details", ""),
                })
                if gate["blocking"] and not result["passed"]:
                    all_blocking_passed = False

            return all_blocking_passed

    gate = ModelTestingGate("churn_model_v3")

    # Define gates
    gate.add_gate("data_validation", lambda ctx: {
        "passed": True, "details": "Schema OK, stats within bounds"
    }, description="Validate training data quality")

    gate.add_gate("model_accuracy", lambda ctx: {
        "passed": ctx.get("accuracy", 0) >= 0.80,
        "details": f"accuracy={ctx.get('accuracy', 0):.4f} (threshold: 0.80)"
    }, description="Minimum accuracy threshold")

    gate.add_gate("model_f1", lambda ctx: {
        "passed": ctx.get("f1", 0) >= 0.72,
        "details": f"f1={ctx.get('f1', 0):.4f} (threshold: 0.72)"
    }, description="Minimum F1 threshold")

    gate.add_gate("no_regression", lambda ctx: {
        "passed": ctx.get("f1", 0) >= ctx.get("prod_f1", 0) - 0.02,
        "details": f"new={ctx.get('f1', 0):.4f} vs prod={ctx.get('prod_f1', 0):.4f}"
    }, description="No regression vs production")

    gate.add_gate("fairness_check", lambda ctx: {
        "passed": ctx.get("di_ratio", 0) >= 0.8,
        "details": f"DI ratio={ctx.get('di_ratio', 0):.4f}"
    }, blocking=False, description="Fairness audit (warning only)")

    gate.add_gate("latency_check", lambda ctx: {
        "passed": ctx.get("p99_latency_ms", 0) <= 50,
        "details": f"p99={ctx.get('p99_latency_ms', 0):.1f}ms (max: 50ms)"
    }, description="Inference latency requirement")

    # Run
    print("CI/CD Model Testing Gates")
    print("=" * 60)

    context = {
        "accuracy": 0.86,
        "f1": 0.80,
        "prod_f1": 0.77,
        "di_ratio": 0.75,
        "p99_latency_ms": 35,
    }

    overall = gate.execute(context)

    for r in gate.results:
        status = "PASS" if r["passed"] else "FAIL"
        block = " (BLOCKING)" if r["blocking"] and not r["passed"] else ""
        warn = " (WARNING)" if not r["blocking"] and not r["passed"] else ""
        print(f"  [{status}] {r['gate']:<25s} {r['details']}{block}{warn}")

    print(f"\n  Pipeline: {'PROCEED' if overall else 'BLOCKED'}")
    return gate


# ============================================================
# Exercise 5: Shadow Deployment Simulator
# ============================================================

def exercise_5_shadow_deployment():
    """Simulate shadow deployment with statistical significance testing."""

    random.seed(42)

    class ShadowDeployment:
        def __init__(self, production_model, shadow_model):
            self.production = production_model
            self.shadow = shadow_model
            self.production_results = []
            self.shadow_results = []

        def process_request(self, features, true_label=None):
            # Production model serves the response
            prod_pred = self.production(features)
            shadow_pred = self.shadow(features)

            self.production_results.append({
                "prediction": prod_pred,
                "label": true_label,
                "correct": prod_pred == true_label if true_label is not None else None,
            })
            self.shadow_results.append({
                "prediction": shadow_pred,
                "label": true_label,
                "correct": shadow_pred == true_label if true_label is not None else None,
            })

        def analyze(self):
            # Agreement rate
            n = len(self.production_results)
            agree = sum(1 for p, s in zip(self.production_results, self.shadow_results)
                        if p["prediction"] == s["prediction"])
            agreement_rate = agree / n if n > 0 else 0

            # Accuracy comparison (where labels available)
            prod_correct = [r for r in self.production_results if r["correct"] is not None]
            shadow_correct = [r for r in self.shadow_results if r["correct"] is not None]

            prod_acc = (sum(1 for r in prod_correct if r["correct"]) /
                        len(prod_correct) if prod_correct else 0)
            shadow_acc = (sum(1 for r in shadow_correct if r["correct"]) /
                          len(shadow_correct) if shadow_correct else 0)

            # Statistical significance (simplified z-test for proportions)
            n_labeled = len(prod_correct)
            if n_labeled > 0:
                p_hat = (prod_acc + shadow_acc) / 2
                se = math.sqrt(2 * p_hat * (1 - p_hat) / max(n_labeled, 1))
                z_stat = (shadow_acc - prod_acc) / max(se, 1e-8)
                p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_stat) / math.sqrt(2))))
            else:
                z_stat = 0
                p_value = 1.0

            return {
                "n_requests": n,
                "n_labeled": n_labeled,
                "agreement_rate": round(agreement_rate, 4),
                "production_accuracy": round(prod_acc, 4),
                "shadow_accuracy": round(shadow_acc, 4),
                "accuracy_diff": round(shadow_acc - prod_acc, 4),
                "z_statistic": round(z_stat, 4),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
            }

    # Models
    weights_prod = [0.5, -0.3, 0.8, 0.1, -0.6]
    bias_prod = -0.5
    weights_shadow = [0.55, -0.28, 0.85, 0.15, -0.55]
    bias_shadow = -0.45

    def make_model(weights, bias):
        def model(features):
            z = sum(w * f for w, f in zip(weights, features)) + bias
            z = max(-500, min(500, z))
            return 1 if (1 / (1 + math.exp(-z))) >= 0.5 else 0
        return model

    shadow = ShadowDeployment(
        make_model(weights_prod, bias_prod),
        make_model(weights_shadow, bias_shadow),
    )

    # Simulate traffic
    for _ in range(2000):
        features = [random.gauss(0, 1) for _ in range(5)]
        true_label = 1 if sum(w * f for w, f in
                               zip([0.5, -0.3, 0.8, 0.1, -0.6], features)) > 0 else 0
        # Label available for 30% of requests (delayed feedback)
        label = true_label if random.random() < 0.3 else None
        shadow.process_request(features, label)

    results = shadow.analyze()

    print("Shadow Deployment Analysis")
    print("=" * 60)
    print(f"\n  Requests processed: {results['n_requests']}")
    print(f"  Labeled requests:   {results['n_labeled']}")
    print(f"  Agreement rate:     {results['agreement_rate']:.2%}")
    print(f"\n  Production accuracy: {results['production_accuracy']:.4f}")
    print(f"  Shadow accuracy:     {results['shadow_accuracy']:.4f}")
    print(f"  Difference:          {results['accuracy_diff']:+.4f}")
    print(f"\n  Z-statistic: {results['z_statistic']:.4f}")
    print(f"  P-value:     {results['p_value']:.4f}")
    print(f"  Significant: {results['significant']} (alpha=0.05)")

    if results["significant"] and results["accuracy_diff"] > 0:
        print(f"\n  Recommendation: Shadow model is significantly better. "
              f"Proceed to canary deployment.")
    elif results["significant"] and results["accuracy_diff"] < 0:
        print(f"\n  Recommendation: Shadow model is significantly worse. "
              f"Do not deploy.")
    else:
        print(f"\n  Recommendation: No significant difference. "
              f"Continue shadow testing or increase sample size.")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: Data Validation Suite")
    print("=" * 60)
    exercise_1_data_validation()

    print("\n\n")
    print("Exercise 2: Behavioral Test Suite")
    print("=" * 60)
    exercise_2_behavioral_tests()

    print("\n\n")
    print("Exercise 3: Fairness Audit Report")
    print("=" * 60)
    exercise_3_fairness_audit()

    print("\n\n")
    print("Exercise 4: CI/CD Integration")
    print("=" * 60)
    exercise_4_cicd_integration()

    print("\n\n")
    print("Exercise 5: Shadow Deployment Simulator")
    print("=" * 60)
    exercise_5_shadow_deployment()
