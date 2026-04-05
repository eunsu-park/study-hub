# 14_responsible_deployment.py — Model card generator and monitoring dashboard
#
# Run: python 14_responsible_deployment.py

"""
Implements responsible AI deployment tools: automated model card
generation following the Mitchell et al. framework, and a runtime
monitoring dashboard for tracking safety metrics in production.
"""

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ModelMetadata:
    name: str
    version: str
    task: str
    architecture: str
    training_data: str
    parameters: int
    license: str


@dataclass
class PerformanceMetric:
    name: str
    value: float
    dataset: str
    split: str


@dataclass
class FairnessMetric:
    demographic: str
    metric_name: str
    overall: float
    subgroup_values: dict


@dataclass
class Limitation:
    category: str
    description: str
    severity: str  # "low", "medium", "high"


@dataclass
class ModelCard:
    metadata: ModelMetadata
    intended_use: str
    out_of_scope: list[str]
    performance: list[PerformanceMetric]
    fairness: list[FairnessMetric]
    limitations: list[Limitation]
    ethical_considerations: list[str]
    carbon_footprint_kg: float


class ModelCardGenerator:
    """Generates model cards following best practices."""

    def generate(self, metadata: ModelMetadata,
                 performance: list[PerformanceMetric],
                 fairness: list[FairnessMetric]) -> ModelCard:
        limitations = self._infer_limitations(metadata, performance)
        ethical = self._infer_ethical_considerations(metadata)
        intended_use = self._infer_intended_use(metadata)
        out_of_scope = self._infer_out_of_scope(metadata)
        carbon = self._estimate_carbon(metadata.parameters)

        return ModelCard(
            metadata=metadata, intended_use=intended_use,
            out_of_scope=out_of_scope, performance=performance,
            fairness=fairness, limitations=limitations,
            ethical_considerations=ethical,
            carbon_footprint_kg=carbon,
        )

    def _infer_limitations(self, meta: ModelMetadata,
                           perf: list[PerformanceMetric]) -> list[Limitation]:
        limitations = [
            Limitation("generalization",
                       f"Trained on {meta.training_data}; may not "
                       "generalize to other domains.", "medium"),
            Limitation("language",
                       "Primarily trained on English text; "
                       "multilingual performance not verified.", "medium"),
        ]
        for p in perf:
            if p.value < 0.8:
                limitations.append(Limitation(
                    "performance",
                    f"Below 80% on {p.name} ({p.dataset}): {p.value:.1%}",
                    "high"))
        return limitations

    def _infer_ethical_considerations(self, meta: ModelMetadata) -> list:
        considerations = [
            "Model outputs should not be used as sole basis for decisions "
            "affecting individuals.",
            "Regular bias audits are recommended before deployment.",
        ]
        if meta.parameters > 1e9:
            considerations.append(
                "Large model size contributes significant carbon footprint.")
        return considerations

    def _infer_intended_use(self, meta: ModelMetadata) -> str:
        return (f"This model is designed for {meta.task} tasks. "
                f"It should be used with human oversight and is not "
                f"intended for safety-critical autonomous decisions.")

    def _infer_out_of_scope(self, meta: ModelMetadata) -> list[str]:
        return [
            "Medical, legal, or financial decision-making",
            "Autonomous operation without human review",
            "Deployment to vulnerable populations without safeguards",
            f"Tasks outside of {meta.task}",
        ]

    def _estimate_carbon(self, params: int) -> float:
        # Rough estimate: ~0.001 kg CO2 per million parameters for training
        return round(params / 1e6 * 0.001, 2)

    def render(self, card: ModelCard) -> str:
        lines = [
            f"{'=' * 60}",
            f"MODEL CARD: {card.metadata.name} v{card.metadata.version}",
            f"{'=' * 60}",
            "",
            f"Architecture: {card.metadata.architecture}",
            f"Parameters: {card.metadata.parameters:,}",
            f"License: {card.metadata.license}",
            f"Training Data: {card.metadata.training_data}",
            "",
            f"INTENDED USE:",
            f"  {card.intended_use}",
            "",
            "OUT OF SCOPE:",
        ]
        for oos in card.out_of_scope:
            lines.append(f"  - {oos}")

        lines.extend(["", "PERFORMANCE:"])
        for p in card.performance:
            lines.append(f"  {p.name}: {p.value:.1%} ({p.dataset}, "
                         f"{p.split})")

        lines.extend(["", "FAIRNESS:"])
        for f in card.fairness:
            lines.append(f"  {f.metric_name} by {f.demographic}:")
            for group, val in f.subgroup_values.items():
                gap = val - f.overall
                lines.append(f"    {group}: {val:.1%} "
                             f"(gap: {gap:+.1%})")

        lines.extend(["", "LIMITATIONS:"])
        for lim in card.limitations:
            lines.append(f"  [{lim.severity:>6}] {lim.description}")

        lines.extend(["", "ETHICAL CONSIDERATIONS:"])
        for ec in card.ethical_considerations:
            lines.append(f"  - {ec}")

        lines.append(f"\nCARBON FOOTPRINT: ~{card.carbon_footprint_kg} kg CO2")
        return "\n".join(lines)


@dataclass
class MonitoringEvent:
    timestamp: datetime
    metric: str
    value: float
    alert: bool = False


class ProductionMonitor:
    """Runtime monitoring dashboard for deployed AI systems."""

    def __init__(self, alert_thresholds: dict = None):
        self.thresholds = alert_thresholds or {
            "toxicity_rate": 0.05,
            "refusal_rate": 0.30,
            "latency_p99_ms": 2000,
            "error_rate": 0.02,
            "drift_score": 0.15,
        }
        self.events: list[MonitoringEvent] = []

    def log_event(self, metric: str, value: float,
                  ts: datetime = None) -> MonitoringEvent:
        ts = ts or datetime.now()
        alert = False
        if metric in self.thresholds:
            alert = value > self.thresholds[metric]
        event = MonitoringEvent(ts, metric, value, alert)
        self.events.append(event)
        return event

    def simulate_monitoring(self, hours: int = 24,
                            events_per_hour: int = 10):
        """Simulate production monitoring data."""
        base_time = datetime.now() - timedelta(hours=hours)
        for h in range(hours):
            for _ in range(events_per_hour):
                ts = base_time + timedelta(
                    hours=h, minutes=random.randint(0, 59))
                # Simulate metrics with occasional anomalies
                anomaly = random.random() < 0.05
                self.log_event("toxicity_rate",
                               random.gauss(0.02, 0.01) +
                               (0.1 if anomaly else 0), ts)
                self.log_event("refusal_rate",
                               random.gauss(0.15, 0.05), ts)
                self.log_event("latency_p99_ms",
                               random.gauss(500, 200) +
                               (3000 if anomaly else 0), ts)
                self.log_event("error_rate",
                               max(0, random.gauss(0.005, 0.003)), ts)
                self.log_event("drift_score",
                               max(0, random.gauss(0.08, 0.03)), ts)

    def get_summary(self) -> dict:
        metrics = {}
        for event in self.events:
            if event.metric not in metrics:
                metrics[event.metric] = {"values": [], "alerts": 0}
            metrics[event.metric]["values"].append(event.value)
            if event.alert:
                metrics[event.metric]["alerts"] += 1

        summary = {}
        for metric, data in metrics.items():
            vals = data["values"]
            summary[metric] = {
                "mean": sum(vals) / len(vals),
                "max": max(vals),
                "min": min(vals),
                "alerts": data["alerts"],
                "total": len(vals),
            }
        return summary

    def render_dashboard(self) -> str:
        summary = self.get_summary()
        lines = [
            f"\n{'=' * 60}",
            "PRODUCTION MONITORING DASHBOARD",
            f"{'=' * 60}",
            f"Period: last 24 hours | Events: {len(self.events)}",
            "",
        ]
        for metric, stats in summary.items():
            threshold = self.thresholds.get(metric, "N/A")
            alert_pct = (stats["alerts"] / stats["total"] * 100
                         if stats["total"] > 0 else 0)
            status = "OK" if alert_pct < 5 else "WARN" if alert_pct < 10 \
                else "ALERT"
            lines.append(
                f"  {metric:<20} avg={stats['mean']:.4f} "
                f"max={stats['max']:.4f} "
                f"alerts={stats['alerts']} ({alert_pct:.1f}%) [{status}]")
        return "\n".join(lines)


if __name__ == "__main__":
    random.seed(42)
    print("=== Responsible Deployment Framework ===\n")

    # Generate model card
    metadata = ModelMetadata(
        name="SafeChat-7B", version="1.2.0",
        task="conversational AI",
        architecture="Transformer (decoder-only)",
        training_data="Curated web corpus + human feedback",
        parameters=7_000_000_000, license="Apache 2.0")

    performance = [
        PerformanceMetric("Accuracy", 0.89, "MMLU", "test"),
        PerformanceMetric("Helpfulness", 0.82, "AlpacaEval", "test"),
        PerformanceMetric("Safety", 0.95, "SafetyBench", "test"),
        PerformanceMetric("Truthfulness", 0.74, "TruthfulQA", "test"),
    ]

    fairness = [
        FairnessMetric("gender", "accuracy", 0.89,
                       {"male": 0.90, "female": 0.88, "non-binary": 0.87}),
        FairnessMetric("age_group", "helpfulness", 0.82,
                       {"18-30": 0.85, "31-50": 0.83, "51+": 0.78}),
    ]

    generator = ModelCardGenerator()
    card = generator.generate(metadata, performance, fairness)
    print(generator.render(card))

    # Production monitoring
    monitor = ProductionMonitor()
    monitor.simulate_monitoring(hours=24, events_per_hour=10)
    print(monitor.render_dashboard())
