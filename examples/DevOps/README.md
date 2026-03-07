# DevOps Examples

Example code for the DevOps topic.

| File | Description |
|------|-------------|
| [01_github_actions_workflow.py](01_github_actions_workflow.py) | Generate and validate GitHub Actions workflow YAML files |
| [02_terraform_basics.py](02_terraform_basics.py) | Terraform HCL generation and state file parsing |
| [03_ansible_playbook.py](03_ansible_playbook.py) | Generate Ansible playbooks and inventory files |
| [04_prometheus_metrics.py](04_prometheus_metrics.py) | Flask app with Prometheus metrics instrumentation |
| [05_structured_logging.py](05_structured_logging.py) | Structured logging with JSON formatter and correlation IDs |
| [06_deployment_strategies.py](06_deployment_strategies.py) | Blue-green and canary deployment logic simulation |
| [07_sli_slo_calculator.py](07_sli_slo_calculator.py) | SLI/SLO calculator with error budget and burn rate alerts |
| [08_chaos_experiment.py](08_chaos_experiment.py) | Chaos experiment framework with fault injection and steady-state validation |
| [09_opentelemetry_tracing.py](09_opentelemetry_tracing.py) | OpenTelemetry instrumentation for a Flask app |
| [10_gitops_reconciler.py](10_gitops_reconciler.py) | Simplified GitOps reconciliation loop simulation |

## Running

```bash
# Install dependencies
pip install flask prometheus_client pyyaml opentelemetry-api opentelemetry-sdk

# Run individual examples
python examples/DevOps/01_github_actions_workflow.py
python examples/DevOps/04_prometheus_metrics.py   # starts Flask on :5000
python examples/DevOps/05_structured_logging.py
python examples/DevOps/07_sli_slo_calculator.py

# Run examples that produce terminal output
python examples/DevOps/06_deployment_strategies.py
python examples/DevOps/08_chaos_experiment.py
python examples/DevOps/10_gitops_reconciler.py
```

**License**: CC BY-NC 4.0
