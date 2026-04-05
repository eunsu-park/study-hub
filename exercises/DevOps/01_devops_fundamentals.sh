#!/bin/bash
# Exercises for Lesson 01: DevOps Fundamentals
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: CALMS Framework Analysis ===
# Problem: Given a team scenario, identify which CALMS pillar is weakest
# and propose improvements.
exercise_1() {
    echo "=== Exercise 1: CALMS Framework Analysis ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# CALMS Framework Assessment Tool
# C = Culture, A = Automation, L = Lean, M = Measurement, S = Sharing

def assess_calms(team_practices: dict[str, list[str]]) -> dict:
    """Assess a team's DevOps maturity using the CALMS framework.

    Each pillar is scored 0-5 based on observed practices.
    """
    pillar_weights = {
        "Culture": {
            "blameless_postmortems": 1,
            "shared_on_call": 1,
            "cross_functional_teams": 1,
            "psychological_safety": 1,
            "experimentation_encouraged": 1,
        },
        "Automation": {
            "ci_pipeline": 1,
            "cd_pipeline": 1,
            "infrastructure_as_code": 1,
            "automated_testing": 1,
            "automated_rollback": 1,
        },
        "Lean": {
            "small_batch_sizes": 1,
            "wip_limits": 1,
            "value_stream_mapped": 1,
            "waste_identified": 1,
            "continuous_improvement": 1,
        },
        "Measurement": {
            "deployment_frequency_tracked": 1,
            "lead_time_tracked": 1,
            "mttr_tracked": 1,
            "change_failure_rate_tracked": 1,
            "business_metrics_dashboards": 1,
        },
        "Sharing": {
            "internal_tech_talks": 1,
            "runbooks_documented": 1,
            "knowledge_base_maintained": 1,
            "code_reviews_practiced": 1,
            "cross_team_collaboration": 1,
        },
    }

    scores = {}
    for pillar, criteria in pillar_weights.items():
        practices = team_practices.get(pillar, [])
        score = sum(1 for c in criteria if c in practices)
        scores[pillar] = {"score": score, "max": len(criteria)}

    weakest = min(scores, key=lambda p: scores[p]["score"])
    return {"scores": scores, "weakest_pillar": weakest}

# Example assessment
team = {
    "Culture": ["blameless_postmortems", "cross_functional_teams"],
    "Automation": ["ci_pipeline", "cd_pipeline", "automated_testing",
                   "infrastructure_as_code", "automated_rollback"],
    "Lean": ["small_batch_sizes"],
    "Measurement": ["deployment_frequency_tracked", "mttr_tracked"],
    "Sharing": ["code_reviews_practiced", "runbooks_documented",
                "knowledge_base_maintained"],
}

result = assess_calms(team)
for pillar, data in result["scores"].items():
    print(f"  {pillar}: {data['score']}/{data['max']}")
print(f"\n  Weakest pillar: {result['weakest_pillar']}")
# Lean is weakest (1/5) — team should adopt WIP limits and value stream mapping
SOLUTION
}

# === Exercise 2: Value Stream Mapping ===
# Problem: Map the deployment pipeline from code commit to production,
# identify wait times and bottlenecks.
exercise_2() {
    echo "=== Exercise 2: Value Stream Mapping ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from dataclasses import dataclass

@dataclass
class PipelineStage:
    name: str
    process_time_min: float   # Active work time
    wait_time_min: float      # Idle/queue time before this stage
    automation_level: str     # manual, semi-auto, fully-auto

    @property
    def total_time(self) -> float:
        return self.process_time_min + self.wait_time_min

    @property
    def efficiency(self) -> float:
        if self.total_time == 0:
            return 0
        return self.process_time_min / self.total_time

# Map current deployment pipeline
pipeline = [
    PipelineStage("Code commit",          5,    0, "fully-auto"),
    PipelineStage("Code review",         30,  120, "manual"),       # 2h wait for reviewer
    PipelineStage("CI build & test",     15,    5, "fully-auto"),
    PipelineStage("Security scan",       10,   60, "semi-auto"),    # Manual approval
    PipelineStage("Staging deploy",       5,   30, "semi-auto"),
    PipelineStage("QA validation",       60,  240, "manual"),       # 4h wait for QA
    PipelineStage("Change approval",     10, 1440, "manual"),       # 24h CAB meeting
    PipelineStage("Production deploy",   10,   30, "semi-auto"),
]

total_process = sum(s.process_time_min for s in pipeline)
total_wait = sum(s.wait_time_min for s in pipeline)
total_lead = total_process + total_wait

print("Value Stream Map — Deployment Pipeline")
print(f"{'Stage':<25} {'Process':>8} {'Wait':>8} {'Eff':>6} {'Auto':<12}")
print("-" * 65)
for stage in pipeline:
    print(f"{stage.name:<25} {stage.process_time_min:>6.0f}m "
          f"{stage.wait_time_min:>6.0f}m {stage.efficiency:>5.0%} "
          f"{stage.automation_level}")
print("-" * 65)
print(f"{'Total':<25} {total_process:>6.0f}m {total_wait:>6.0f}m "
      f"{total_process/total_lead:>5.0%}")
print(f"\nLead time:     {total_lead/60:.1f} hours ({total_lead/1440:.1f} days)")
print(f"Process time:  {total_process/60:.1f} hours")
print(f"Efficiency:    {total_process/total_lead:.1%}")

# Bottleneck: Change approval (24h wait) — 74% of total wait time
# Recommendation: Replace manual CAB with automated policy checks
SOLUTION
}

# === Exercise 3: DORA Metrics Calculation ===
# Problem: Calculate the four DORA metrics from deployment log data
# and classify the team's performance level.
exercise_3() {
    echo "=== Exercise 3: DORA Metrics Calculation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from datetime import datetime, timedelta

def classify_dora_metrics(
    deployments_per_day: float,
    lead_time_hours: float,
    change_failure_pct: float,
    mttr_hours: float,
) -> dict:
    """Classify team performance using DORA metrics (2023 benchmarks).

    Categories: Elite, High, Medium, Low
    """
    # Deployment Frequency
    if deployments_per_day >= 1:
        df_level = "Elite"       # On-demand (multiple per day)
    elif deployments_per_day >= 1/7:
        df_level = "High"        # Weekly to daily
    elif deployments_per_day >= 1/30:
        df_level = "Medium"      # Monthly to weekly
    else:
        df_level = "Low"         # Less than monthly

    # Lead Time for Changes
    if lead_time_hours < 24:
        lt_level = "Elite"       # Less than one day
    elif lead_time_hours < 168:
        lt_level = "High"        # One day to one week
    elif lead_time_hours < 720:
        lt_level = "Medium"      # One week to one month
    else:
        lt_level = "Low"         # More than one month

    # Change Failure Rate
    if change_failure_pct <= 5:
        cfr_level = "Elite"
    elif change_failure_pct <= 10:
        cfr_level = "High"
    elif change_failure_pct <= 15:
        cfr_level = "Medium"
    else:
        cfr_level = "Low"

    # Mean Time to Recovery
    if mttr_hours < 1:
        mttr_level = "Elite"     # Less than one hour
    elif mttr_hours < 24:
        mttr_level = "High"      # Less than one day
    elif mttr_hours < 168:
        mttr_level = "Medium"    # Less than one week
    else:
        mttr_level = "Low"       # More than one week

    levels = [df_level, lt_level, cfr_level, mttr_level]
    rank = {"Elite": 4, "High": 3, "Medium": 2, "Low": 1}
    avg_rank = sum(rank[l] for l in levels) / len(levels)
    overall = "Elite" if avg_rank >= 3.5 else (
        "High" if avg_rank >= 2.5 else "Medium" if avg_rank >= 1.5 else "Low"
    )

    return {
        "deployment_frequency": {"value": deployments_per_day, "level": df_level},
        "lead_time": {"value": lead_time_hours, "level": lt_level},
        "change_failure_rate": {"value": change_failure_pct, "level": cfr_level},
        "mttr": {"value": mttr_hours, "level": mttr_level},
        "overall": overall,
    }

# Example: calculate from deployment log
result = classify_dora_metrics(
    deployments_per_day=3.5,      # 3-4 deploys per day
    lead_time_hours=4.2,          # ~4 hours from commit to production
    change_failure_pct=8.0,       # 8% of changes cause incidents
    mttr_hours=0.5,               # 30 minutes average recovery
)

for metric, data in result.items():
    if metric == "overall":
        print(f"\nOverall: {data}")
    else:
        print(f"  {metric}: {data['value']} -> {data['level']}")
SOLUTION
}

# === Exercise 4: Three Ways Principles ===
# Problem: For each of the Three Ways (Flow, Feedback, Continual Learning),
# identify one practice your team does NOT have and design an implementation plan.
exercise_4() {
    echo "=== Exercise 4: Three Ways Principles ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# The Three Ways of DevOps (from "The Phoenix Project")

three_ways = {
    "First Way - Flow": {
        "principle": "Accelerate flow from Dev to Ops to Customer",
        "practices": [
            "Continuous integration / continuous delivery",
            "Small batch sizes and frequent releases",
            "Limit work in progress (WIP)",
            "Reduce handoffs between teams",
            "Automate manual processes",
        ],
        "anti_patterns": [
            "Large, infrequent releases",
            "Manual deployment steps",
            "Separate Dev and Ops teams with ticket-based handoffs",
        ],
    },
    "Second Way - Feedback": {
        "principle": "Enable fast feedback from right to left",
        "practices": [
            "Automated testing at every stage",
            "Monitoring and alerting in production",
            "Telemetry visible to developers",
            "A/B testing and feature flags",
            "Fast rollback mechanisms",
        ],
        "anti_patterns": [
            "Ops discovers bugs only after deployment",
            "No production monitoring dashboards for developers",
            "Post-release testing only",
        ],
    },
    "Third Way - Continual Learning": {
        "principle": "Foster experimentation and learning from failure",
        "practices": [
            "Blameless postmortems",
            "Chaos engineering experiments",
            "20% time for improvement work",
            "Internal tech talks and knowledge sharing",
            "Game days and disaster recovery drills",
        ],
        "anti_patterns": [
            "Blame-oriented incident reviews",
            "No time allocated for improvement",
            "Fear of making changes to production",
        ],
    },
}

for way, details in three_ways.items():
    print(f"\n{way}")
    print(f"  Principle: {details['principle']}")
    print(f"  Key practices:")
    for p in details["practices"]:
        print(f"    - {p}")
    print(f"  Anti-patterns to avoid:")
    for a in details["anti_patterns"]:
        print(f"    x {a}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 01: DevOps Fundamentals"
echo "======================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
