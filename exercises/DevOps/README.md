# DevOps Exercises

Practice problem solutions for the DevOps topic (18 lessons). Each file corresponds to a lesson and contains working solutions with Python/YAML/HCL/bash code displayed via heredoc.

## Exercise Files

| # | File | Lesson | Description |
|---|------|--------|-------------|
| 01 | `01_devops_fundamentals.sh` | DevOps Fundamentals | DevOps culture, CALMS framework, value stream mapping |
| 02 | `02_version_control_workflows.sh` | Version Control Workflows | Git branching strategies, trunk-based dev, merge vs rebase |
| 03 | `03_ci_cd_fundamentals.sh` | CI/CD Fundamentals | Pipeline design, build stages, artifact management |
| 04 | `04_containerization.sh` | Containerization | Docker builds, image optimization, container orchestration |
| 05 | `05_infrastructure_as_code.sh` | Infrastructure as Code | Terraform, state management, modules, drift detection |
| 06 | `06_configuration_management.sh` | Configuration Management | Ansible playbooks, idempotency, inventory management |
| 07 | `07_cloud_platforms.sh` | Cloud Platforms | AWS/GCP/Azure core services, multi-cloud strategies |
| 08 | `08_kubernetes_operations.sh` | Kubernetes Operations | Deployments, services, ingress, resource management |
| 09 | `09_monitoring_and_observability.sh` | Monitoring and Observability | Metrics, tracing, alerting, dashboards, three pillars |
| 10 | `10_logging_and_log_management.sh` | Logging and Log Management | Structured logging, aggregation, ELK/Loki, correlation |
| 11 | `11_security_in_devops.sh` | Security in DevOps (DevSecOps) | SAST, DAST, secret management, supply chain security |
| 12 | `12_deployment_strategies.sh` | Deployment Strategies | Blue-green, canary, rolling, feature flags |
| 13 | `13_networking_and_service_mesh.sh` | Networking and Service Mesh | DNS, load balancing, Istio/Envoy, mTLS |
| 14 | `14_gitops.sh` | GitOps | ArgoCD, Flux, reconciliation, declarative infrastructure |
| 15 | `15_incident_management.sh` | Incident Management | On-call, runbooks, postmortems, incident response |
| 16 | `16_chaos_engineering.sh` | Chaos Engineering | Fault injection, steady-state hypothesis, blast radius |
| 17 | `17_slis_slos_and_error_budgets.sh` | SLIs, SLOs, and Error Budgets | Error budget computation, burn rate alerts, reliability targets |
| 18 | `18_sre_practices.sh` | SRE Practices | Toil reduction, capacity planning, release engineering |

## How to Use

1. Study the lesson in `content/en/DevOps/` or `content/ko/DevOps/`
2. Attempt the exercises at the end of each lesson on your own
3. Run an exercise file to view the solutions: `bash exercises/DevOps/01_devops_fundamentals.sh`
4. Each exercise function prints its solution as Python/YAML/bash code

## File Structure

Each `.sh` file follows this pattern:

```bash
#!/bin/bash
# Exercises for Lesson XX: Title
# Topic: DevOps
# Solutions to practice problems from the lesson.

exercise_1() {
    echo "=== Exercise 1: Title ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
    # Python/YAML/HCL/bash solution code here
SOLUTION
}

# Run all exercises
exercise_1
exercise_2
```
