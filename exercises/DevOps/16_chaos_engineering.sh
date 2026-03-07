#!/bin/bash
# Exercises for Lesson 16: Chaos Engineering
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Steady-State Hypothesis ===
# Problem: Define steady-state hypotheses for a microservices application,
# including measurable thresholds and probes.
exercise_1() {
    echo "=== Exercise 1: Steady-State Hypothesis ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from dataclasses import dataclass

@dataclass
class SteadyStateProbe:
    name: str
    metric: str
    operator: str       # "lt", "gt", "between"
    threshold: float | tuple
    source: str          # "prometheus", "logs", "healthcheck"

hypotheses = {
    "Order API Availability": {
        "description": "The order API maintains 99.9% success rate under normal conditions",
        "probes": [
            SteadyStateProbe("success_rate", "http_success_ratio", "gt", 0.999, "prometheus"),
            SteadyStateProbe("p99_latency", "http_request_duration_p99", "lt", 0.5, "prometheus"),
            SteadyStateProbe("error_count", "http_5xx_count_5m", "lt", 10.0, "prometheus"),
        ],
    },
    "Database Resilience": {
        "description": "Application handles database failover without user-visible errors",
        "probes": [
            SteadyStateProbe("db_connections", "pg_active_connections", "lt", 80.0, "prometheus"),
            SteadyStateProbe("query_latency", "db_query_duration_p99", "lt", 0.1, "prometheus"),
            SteadyStateProbe("circuit_breaker", "circuit_breaker_state", "lt", 1.0, "healthcheck"),
        ],
    },
    "Queue Processing": {
        "description": "Message queue consumers keep up with incoming messages",
        "probes": [
            SteadyStateProbe("queue_depth", "rabbitmq_queue_messages", "lt", 1000.0, "prometheus"),
            SteadyStateProbe("consumer_lag", "consumer_lag_seconds", "lt", 30.0, "prometheus"),
            SteadyStateProbe("dlq_count", "dead_letter_queue_size", "lt", 5.0, "prometheus"),
        ],
    },
}

for name, hypothesis in hypotheses.items():
    print(f"\nHypothesis: {name}")
    print(f"  Description: {hypothesis['description']}")
    print(f"  Probes:")
    for p in hypothesis["probes"]:
        print(f"    - {p.name}: {p.metric} {p.operator} {p.threshold} (via {p.source})")

# A hypothesis is DISPROVED when any probe fails under fault conditions.
# This reveals a weakness that needs fixing (circuit breaker, retry, fallback).
SOLUTION
}

# === Exercise 2: Fault Injection Catalog ===
# Problem: Create a catalog of fault injection experiments categorized
# by failure domain (network, compute, application, dependency).
exercise_2() {
    echo "=== Exercise 2: Fault Injection Catalog ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
fault_catalog = {
    "Network Faults": [
        {
            "name": "Latency injection",
            "tool": "tc netem / Istio fault injection",
            "description": "Add 500ms delay to all traffic between service A and B",
            "blast_radius": "Single service pair",
            "expected_behavior": "Timeouts trigger, circuit breaker opens, fallback activates",
        },
        {
            "name": "Packet loss",
            "tool": "tc netem / Chaos Mesh",
            "description": "Drop 10% of packets on the target pod's network interface",
            "blast_radius": "Single pod",
            "expected_behavior": "Retries succeed, no user-visible errors (if retry logic works)",
        },
        {
            "name": "DNS failure",
            "tool": "Chaos Mesh / CoreDNS manipulation",
            "description": "Inject NXDOMAIN responses for a specific service",
            "blast_radius": "All consumers of the target service",
            "expected_behavior": "Cached DNS entries used, graceful degradation",
        },
    ],
    "Compute Faults": [
        {
            "name": "Pod kill",
            "tool": "Chaos Monkey / Litmus / kubectl delete pod",
            "description": "Randomly kill one pod of a deployment",
            "blast_radius": "Single pod (deployment auto-recreates it)",
            "expected_behavior": "Traffic shifts to remaining pods, zero user impact",
        },
        {
            "name": "CPU stress",
            "tool": "stress-ng / Chaos Mesh",
            "description": "Consume 90% CPU on a target pod",
            "blast_radius": "Single pod (may affect co-located pods on same node)",
            "expected_behavior": "HPA scales up, pod evicted if resource limit hit",
        },
        {
            "name": "Node drain",
            "tool": "kubectl drain / Chaos Mesh",
            "description": "Drain all pods from a node (simulates hardware failure)",
            "blast_radius": "All pods on the node",
            "expected_behavior": "Pods reschedule to other nodes within seconds",
        },
    ],
    "Application Faults": [
        {
            "name": "Error injection (HTTP 500)",
            "tool": "Istio fault injection / app-level feature flag",
            "description": "Return 500 for 10% of requests to a specific endpoint",
            "blast_radius": "10% of requests to one endpoint",
            "expected_behavior": "Client retries succeed, error rate stays below SLO",
        },
        {
            "name": "Memory leak simulation",
            "tool": "Custom code / stress-ng",
            "description": "Gradually consume memory until OOM",
            "blast_radius": "Single pod",
            "expected_behavior": "OOMKilled -> restart -> healthy within 30 seconds",
        },
    ],
    "Dependency Faults": [
        {
            "name": "Database failover",
            "tool": "RDS failover / pg_ctl promote",
            "description": "Trigger primary-to-replica failover",
            "blast_radius": "All database consumers",
            "expected_behavior": "Brief connection errors, then auto-reconnect",
        },
        {
            "name": "Cache unavailability",
            "tool": "Redis SHUTDOWN / Chaos Mesh pod kill",
            "description": "Kill Redis cache entirely",
            "blast_radius": "All cache consumers",
            "expected_behavior": "Cache miss -> DB fallback, higher latency but no errors",
        },
    ],
}

for domain, faults in fault_catalog.items():
    print(f"\n{domain}:")
    for fault in faults:
        print(f"  {fault['name']}")
        print(f"    Tool:     {fault['tool']}")
        print(f"    Blast:    {fault['blast_radius']}")
        print(f"    Expected: {fault['expected_behavior']}")
SOLUTION
}

# === Exercise 3: Chaos Experiment Design ===
# Problem: Design a complete chaos experiment with pre-checks,
# fault injection, monitoring, and rollback plan.
exercise_3() {
    echo "=== Exercise 3: Chaos Experiment Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
experiment = {
    "name": "Redis Cache Failure Resilience",
    "objective": "Verify that the order-api degrades gracefully when Redis is unavailable",
    "hypothesis": "Order API maintains >95% success rate and <2s p99 latency without Redis",
    "blast_radius": "Single Redis instance in staging environment",
    "duration": "5 minutes",
    "abort_conditions": [
        "Error rate exceeds 20% (indicates no fallback)",
        "p99 latency exceeds 5 seconds",
        "Any downstream service cascading failure",
    ],

    "steps": [
        {
            "phase": "Pre-check",
            "actions": [
                "Verify steady state: error rate < 1%, p99 < 500ms",
                "Confirm Redis is healthy: redis-cli ping -> PONG",
                "Confirm rollback mechanism: Chaos Mesh abort works",
                "Notify team in #chaos-experiments Slack channel",
            ],
        },
        {
            "phase": "Inject fault",
            "actions": [
                "Apply Chaos Mesh PodChaos to kill Redis pod",
                "Start timer (5 minute experiment window)",
            ],
        },
        {
            "phase": "Observe",
            "actions": [
                "Monitor Grafana dashboard: error rate, latency, cache hit rate",
                "Watch application logs for cache fallback messages",
                "Verify database query rate increases (expected behavior)",
                "Check if circuit breaker opens for Redis",
            ],
        },
        {
            "phase": "Analyze",
            "actions": [
                "Compare metrics during fault vs steady state",
                "Was the hypothesis upheld? (>95% success, <2s p99)",
                "Were abort conditions triggered?",
            ],
        },
        {
            "phase": "Rollback",
            "actions": [
                "Delete Chaos Mesh experiment (Redis pod auto-restarts)",
                "Verify Redis is healthy and cache is warming up",
                "Confirm metrics return to steady state within 2 minutes",
            ],
        },
    ],
}

print(f"Experiment: {experiment['name']}")
print(f"Hypothesis: {experiment['hypothesis']}")
print(f"Blast radius: {experiment['blast_radius']}")
print(f"Duration: {experiment['duration']}")
print()
for step in experiment["steps"]:
    print(f"  Phase: {step['phase']}")
    for action in step["actions"]:
        print(f"    - {action}")
    print()
SOLUTION
}

# === Exercise 4: Game Day Planning ===
# Problem: Plan a game day exercise that tests the team's incident
# response capability under realistic conditions.
exercise_4() {
    echo "=== Exercise 4: Game Day Planning ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
game_day = {
    "title": "Game Day: Regional Outage Simulation",
    "date": "2025-02-15, 10:00-12:00 UTC",
    "environment": "Staging (mirrors production architecture)",
    "participants": ["Platform team", "Backend team", "SRE on-call"],

    "scenario": "Simulate AWS us-east-1a availability zone failure",
    "injected_faults": [
        "Terminate all pods scheduled on us-east-1a nodes",
        "Block network traffic to us-east-1a subnets",
        "Fail health checks for RDS primary in us-east-1a",
    ],

    "success_criteria": [
        "Service recovers to >99% availability within 5 minutes",
        "Incident commander declared within 2 minutes of alert",
        "Status page updated within 5 minutes",
        "No data loss during database failover",
        "Runbooks were followed and were accurate",
    ],

    "preparation": [
        "Brief all participants on the game day format (no surprises)",
        "Ensure staging environment mirrors production topology",
        "Pre-position observers to take notes on team behavior",
        "Set up dedicated Slack channel for the exercise",
        "Have a kill switch to abort the experiment instantly",
    ],

    "debrief_questions": [
        "When did you first realize something was wrong?",
        "Were the runbooks helpful? What was missing?",
        "Where did communication break down?",
        "What would you do differently next time?",
        "What automated response should we build?",
    ],
}

print(f"Game Day: {game_day['title']}")
print(f"Date: {game_day['date']}")
print(f"Scenario: {game_day['scenario']}")
print()
print("Success Criteria:")
for c in game_day["success_criteria"]:
    print(f"  [ ] {c}")
print()
print("Debrief Questions:")
for q in game_day["debrief_questions"]:
    print(f"  - {q}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 16: Chaos Engineering"
echo "====================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
