"""
Exercise Solutions: MLOps Overview
===========================================
Lesson 01 from MLOps topic.

Exercises
---------
1. MLOps Maturity Assessment — Evaluate where an organization falls on the MLOps
   maturity scale (Levels 0-4) and create an improvement roadmap.
2. Tool Selection — Given project constraints, select appropriate tools for each
   MLOps component and justify the choices.
"""


# ============================================================
# Exercise 1: MLOps Maturity Assessment
# ============================================================

def exercise_1_maturity_assessment():
    """Assess an organization's MLOps maturity level and create an improvement roadmap.

    MLOps Maturity Levels:
      Level 0 — No MLOps: Manual, ad-hoc processes
      Level 1 — DevOps but no MLOps: Basic CI/CD but no ML-specific automation
      Level 2 — Automated Training: Training pipelines automated, manual deployment
      Level 3 — Automated Deployment: Full CI/CD for ML, automated model deployment
      Level 4 — Full MLOps: Automated retraining, monitoring, and feedback loops

    We simulate this by defining a rubric across several dimensions and scoring
    an organization against it.
    """

    # --- Maturity rubric: each dimension scored 0-4 ---
    maturity_dimensions = {
        "data_management": {
            0: "No versioning; data scattered across local machines",
            1: "Shared storage; manual backups",
            2: "Version-controlled datasets; automated ingestion",
            3: "Feature store; data lineage tracking",
            4: "Automated data quality checks; drift detection",
        },
        "model_development": {
            0: "Jupyter notebooks; no reproducibility",
            1: "Shared code repo; basic experiment notes",
            2: "Experiment tracking (MLflow/W&B); reproducible training",
            3: "Automated hyperparameter tuning; model registry",
            4: "AutoML integration; NAS; continuous training",
        },
        "model_deployment": {
            0: "No deployment; models shared via email/files",
            1: "Manual deployment scripts; single environment",
            2: "Containerized serving; staging environment",
            3: "CI/CD for models; canary/blue-green deployment",
            4: "Automated rollback; A/B testing; shadow mode",
        },
        "monitoring": {
            0: "No monitoring",
            1: "Basic health checks (latency, errors)",
            2: "Prediction logging; manual review",
            3: "Automated drift detection; performance alerts",
            4: "Automated retraining triggers; feedback loops",
        },
        "governance": {
            0: "No documentation or audit trail",
            1: "Basic model documentation",
            2: "Model cards; approval workflows",
            3: "Full lineage tracking; compliance automation",
            4: "Bias auditing; explainability; regulatory compliance",
        },
    }

    # --- Simulate an organization's scores ---
    # This represents a team that has good development practices but
    # weak deployment and monitoring — a common pattern.
    org_scores = {
        "data_management": 2,
        "model_development": 3,
        "model_deployment": 1,
        "monitoring": 1,
        "governance": 2,
    }

    # --- Calculate overall maturity ---
    # Overall level = minimum of all dimensions (the weakest link determines
    # the effective maturity, since a chain is only as strong as its weakest link).
    # Average gives a more nuanced picture.
    avg_score = sum(org_scores.values()) / len(org_scores)
    min_score = min(org_scores.values())
    max_score = max(org_scores.values())

    level_names = {
        0: "No MLOps",
        1: "DevOps but no MLOps",
        2: "Automated Training",
        3: "Automated Deployment",
        4: "Full MLOps",
    }

    print("=" * 60)
    print("MLOps Maturity Assessment Report")
    print("=" * 60)
    print()

    for dim, score in org_scores.items():
        label = maturity_dimensions[dim][score]
        bar = "█" * score + "░" * (4 - score)
        print(f"  {dim:<25s} [{bar}] Level {score}: {label}")
    print()
    print(f"  Average Score:  {avg_score:.1f}")
    print(f"  Minimum Score:  {min_score} ({level_names[min_score]})")
    print(f"  Maximum Score:  {max_score} ({level_names[max_score]})")
    print(f"  Effective Level: {min_score} — {level_names[min_score]}")
    print()

    # --- Generate improvement roadmap ---
    # Prioritize dimensions with the lowest scores, targeting +1 improvement
    # per dimension in priority order.
    print("Improvement Roadmap (prioritized by lowest score):")
    print("-" * 60)

    sorted_dims = sorted(org_scores.items(), key=lambda x: x[1])
    for priority, (dim, current_score) in enumerate(sorted_dims, 1):
        if current_score < 4:
            target_score = current_score + 1
            current_desc = maturity_dimensions[dim][current_score]
            target_desc = maturity_dimensions[dim][target_score]
            print(f"\n  Priority {priority}: {dim}")
            print(f"    Current (L{current_score}): {current_desc}")
            print(f"    Target  (L{target_score}): {target_desc}")

            # Concrete action items per dimension
            actions = {
                "model_deployment": [
                    "Containerize models with Docker",
                    "Set up a staging environment for pre-production testing",
                    "Create deployment scripts with rollback capability",
                ],
                "monitoring": [
                    "Implement prediction logging to a centralized store",
                    "Add latency and error-rate dashboards",
                    "Set up periodic model performance review process",
                ],
                "data_management": [
                    "Adopt DVC or similar for dataset versioning",
                    "Implement automated data quality checks",
                    "Build data lineage documentation",
                ],
                "governance": [
                    "Create model card templates",
                    "Implement approval workflow for production models",
                    "Document data sources and processing steps",
                ],
                "model_development": [
                    "Integrate automated hyperparameter search",
                    "Set up a model registry with staging/production stages",
                    "Implement model comparison dashboards",
                ],
            }
            if dim in actions:
                print(f"    Actions:")
                for action in actions[dim]:
                    print(f"      - {action}")

    print()
    return org_scores, avg_score


# ============================================================
# Exercise 2: Tool Selection
# ============================================================

def exercise_2_tool_selection():
    """Select appropriate tools for each MLOps component given project constraints.

    Given a project profile (team size, budget, cloud provider, model type),
    recommend tools for each MLOps component and justify each choice.
    """

    # --- Define project profiles ---
    project_profiles = [
        {
            "name": "Startup — Small Team, Limited Budget",
            "team_size": 3,
            "budget": "low",
            "cloud": "AWS",
            "model_type": "tabular classification",
            "compliance": "none",
            "scale": "100 predictions/day",
        },
        {
            "name": "Enterprise — Large Team, Regulated Industry",
            "team_size": 25,
            "budget": "high",
            "cloud": "multi-cloud",
            "model_type": "NLP + computer vision",
            "compliance": "HIPAA/SOC2",
            "scale": "1M predictions/day",
        },
    ]

    # --- Tool catalog with metadata ---
    tool_catalog = {
        "experiment_tracking": {
            "MLflow": {
                "cost": "free",
                "complexity": "low",
                "cloud_lock_in": "none",
                "features": ["experiment tracking", "model registry", "model serving"],
            },
            "Weights & Biases": {
                "cost": "freemium",
                "complexity": "low",
                "cloud_lock_in": "none",
                "features": ["experiment tracking", "sweeps", "artifacts", "reports"],
            },
            "SageMaker Experiments": {
                "cost": "pay-per-use",
                "complexity": "medium",
                "cloud_lock_in": "AWS",
                "features": ["experiment tracking", "tight AWS integration"],
            },
        },
        "orchestration": {
            "Airflow": {
                "cost": "free",
                "complexity": "high",
                "cloud_lock_in": "none",
                "features": ["DAG scheduling", "monitoring", "extensible"],
            },
            "Kubeflow Pipelines": {
                "cost": "free",
                "complexity": "high",
                "cloud_lock_in": "none",
                "features": ["ML-native", "caching", "visualization"],
            },
            "Dagster": {
                "cost": "free",
                "complexity": "medium",
                "cloud_lock_in": "none",
                "features": ["software-defined assets", "type checking", "observability"],
            },
        },
        "model_serving": {
            "FastAPI + Docker": {
                "cost": "free",
                "complexity": "low",
                "cloud_lock_in": "none",
                "features": ["custom logic", "simple", "flexible"],
            },
            "TorchServe": {
                "cost": "free",
                "complexity": "medium",
                "cloud_lock_in": "none",
                "features": ["PyTorch-native", "batching", "versioning"],
            },
            "Triton Inference Server": {
                "cost": "free",
                "complexity": "high",
                "cloud_lock_in": "none",
                "features": ["multi-framework", "GPU optimization", "dynamic batching"],
            },
            "SageMaker Endpoints": {
                "cost": "pay-per-use",
                "complexity": "low",
                "cloud_lock_in": "AWS",
                "features": ["auto-scaling", "A/B testing", "managed"],
            },
        },
        "monitoring": {
            "Evidently": {
                "cost": "free",
                "complexity": "low",
                "cloud_lock_in": "none",
                "features": ["drift detection", "reports", "dashboards"],
            },
            "Whylogs": {
                "cost": "free",
                "complexity": "low",
                "cloud_lock_in": "none",
                "features": ["data profiling", "lightweight", "streaming"],
            },
            "Prometheus + Grafana": {
                "cost": "free",
                "complexity": "medium",
                "cloud_lock_in": "none",
                "features": ["metrics", "alerting", "dashboards"],
            },
        },
    }

    def select_tools(profile):
        """Simple rule-based tool recommendation engine."""
        recommendations = {}

        # Experiment tracking
        if profile["budget"] == "low":
            recommendations["experiment_tracking"] = (
                "MLflow",
                "Free, self-hosted, low complexity — ideal for small teams on a budget.",
            )
        else:
            recommendations["experiment_tracking"] = (
                "Weights & Biases",
                "Superior collaboration features (reports, team dashboards) justify "
                "cost for larger teams. Enterprise plan supports SSO and compliance.",
            )

        # Orchestration
        if profile["team_size"] <= 5:
            recommendations["orchestration"] = (
                "Dagster",
                "Lower learning curve than Airflow/Kubeflow. Software-defined assets "
                "map well to ML workflows. Easier to maintain with a small team.",
            )
        else:
            recommendations["orchestration"] = (
                "Kubeflow Pipelines",
                "ML-native pipeline abstraction. Supports parallel execution and "
                "caching. Scales well for large teams with many concurrent experiments.",
            )

        # Model serving
        if profile["scale"] == "100 predictions/day":
            recommendations["model_serving"] = (
                "FastAPI + Docker",
                "Simple, flexible, minimal overhead. Sufficient for low-volume serving. "
                "Easy for a small team to maintain and debug.",
            )
        elif "GPU" in profile.get("model_type", "").upper() or "vision" in profile["model_type"]:
            recommendations["model_serving"] = (
                "Triton Inference Server",
                "GPU optimization with dynamic batching is critical at 1M predictions/day "
                "with vision models. Multi-framework support allows serving diverse models.",
            )
        else:
            recommendations["model_serving"] = (
                "SageMaker Endpoints",
                "Managed auto-scaling reduces operational burden. Native A/B testing "
                "supports safe deployment. Pay-per-use is cost-effective at scale.",
            )

        # Monitoring
        if profile["compliance"] != "none":
            recommendations["monitoring"] = (
                "Evidently + Prometheus/Grafana",
                "Evidently provides ML-specific drift reports (required for compliance audits). "
                "Prometheus/Grafana adds infrastructure monitoring and alerting.",
            )
        else:
            recommendations["monitoring"] = (
                "Evidently",
                "Simple setup, comprehensive drift detection. Generates HTML reports "
                "for stakeholder review without extra infrastructure.",
            )

        return recommendations

    # --- Evaluate each profile ---
    for profile in project_profiles:
        print("=" * 60)
        print(f"Project: {profile['name']}")
        print("=" * 60)
        print(f"  Team Size:   {profile['team_size']}")
        print(f"  Budget:      {profile['budget']}")
        print(f"  Cloud:       {profile['cloud']}")
        print(f"  Model Type:  {profile['model_type']}")
        print(f"  Compliance:  {profile['compliance']}")
        print(f"  Scale:       {profile['scale']}")
        print()

        recs = select_tools(profile)
        for component, (tool, justification) in recs.items():
            print(f"  {component}:")
            print(f"    Recommended: {tool}")
            details = tool_catalog.get(component, {}).get(tool.split(" + ")[0], {})
            if details:
                print(f"    Cost: {details.get('cost', 'N/A')}")
                print(f"    Complexity: {details.get('complexity', 'N/A')}")
            print(f"    Justification: {justification}")
            print()

    return project_profiles


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: MLOps Maturity Assessment")
    print("=" * 60)
    exercise_1_maturity_assessment()

    print("\n")
    print("Exercise 2: Tool Selection")
    print("=" * 60)
    exercise_2_tool_selection()
