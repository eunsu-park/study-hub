# 06_scalable_oversight.py — Debate protocol and weak-to-strong setup
#
# Run: python 06_scalable_oversight.py

"""
Demonstrates scalable oversight techniques: AI debate protocol where
two agents argue for different answers judged by a weaker supervisor,
and a weak-to-strong generalization framework.
"""

import random
from dataclasses import dataclass, field


@dataclass
class Argument:
    claim: str
    evidence: list[str]
    strength: float  # 0.0 to 1.0

    def score(self) -> float:
        evidence_bonus = len(self.evidence) * 0.1
        return min(self.strength + evidence_bonus, 1.0)


@dataclass
class DebateRound:
    round_num: int
    pro_argument: Argument
    con_argument: Argument
    judge_decision: str  # "pro", "con", or "undecided"


class DebateAgent:
    """An agent that generates arguments for or against a position."""

    def __init__(self, name: str, position: str, skill_level: float = 0.7):
        self.name = name
        self.position = position
        self.skill_level = skill_level
        self.argument_bank = []

    def load_arguments(self, arguments: list[dict]):
        self.argument_bank = arguments

    def generate_argument(self, round_num: int,
                          opponent_last: Argument = None) -> Argument:
        """Generate an argument, optionally rebutting the opponent."""
        if round_num < len(self.argument_bank):
            arg_data = self.argument_bank[round_num]
        else:
            arg_data = {
                "claim": f"[{self.position}] Additional point #{round_num}",
                "evidence": ["General reasoning"],
            }

        # Rebuttal boost if responding to opponent
        rebuttal_bonus = 0.05 if opponent_last else 0.0
        noise = random.gauss(0, 0.05)
        strength = min(max(
            self.skill_level + rebuttal_bonus + noise, 0.1), 1.0)

        return Argument(
            claim=arg_data["claim"],
            evidence=arg_data.get("evidence", []),
            strength=round(strength, 3)
        )


class WeakJudge:
    """A weaker model acting as judge in the debate protocol."""

    def __init__(self, comprehension_level: float = 0.6):
        self.comprehension_level = comprehension_level

    def evaluate(self, pro: Argument, con: Argument) -> str:
        """Judge which argument is more convincing."""
        pro_score = pro.score() * self.comprehension_level
        con_score = con.score() * self.comprehension_level
        noise = random.gauss(0, 0.05)
        pro_score += noise

        if abs(pro_score - con_score) < 0.05:
            return "undecided"
        return "pro" if pro_score > con_score else "con"


class DebateProtocol:
    """Full debate protocol with multiple rounds."""

    def __init__(self, pro_agent: DebateAgent, con_agent: DebateAgent,
                 judge: WeakJudge, num_rounds: int = 3):
        self.pro = pro_agent
        self.con = con_agent
        self.judge = judge
        self.num_rounds = num_rounds
        self.rounds: list[DebateRound] = []

    def run(self) -> dict:
        pro_wins, con_wins, ties = 0, 0, 0
        last_pro = last_con = None

        for r in range(self.num_rounds):
            pro_arg = self.pro.generate_argument(r, last_con)
            con_arg = self.con.generate_argument(r, last_pro)
            decision = self.judge.evaluate(pro_arg, con_arg)

            self.rounds.append(DebateRound(r, pro_arg, con_arg, decision))
            if decision == "pro":
                pro_wins += 1
            elif decision == "con":
                con_wins += 1
            else:
                ties += 1
            last_pro, last_con = pro_arg, con_arg

        winner = "pro" if pro_wins > con_wins else (
            "con" if con_wins > pro_wins else "tie")
        return {"winner": winner, "pro_wins": pro_wins,
                "con_wins": con_wins, "ties": ties, "rounds": self.rounds}


class WeakToStrongFramework:
    """Demonstrates weak-to-strong generalization."""

    def __init__(self, weak_accuracy: float = 0.65,
                 strong_capacity: float = 0.90):
        self.weak_accuracy = weak_accuracy
        self.strong_capacity = strong_capacity

    def generate_labels(self, n: int) -> list[dict]:
        """Generate samples with weak labels and true labels."""
        samples = []
        for i in range(n):
            true_label = random.choice([0, 1])
            weak_correct = random.random() < self.weak_accuracy
            weak_label = true_label if weak_correct else (1 - true_label)
            samples.append({
                "id": i, "true_label": true_label,
                "weak_label": weak_label,
                "features": [random.gauss(true_label, 0.3) for _ in range(4)]
            })
        return samples

    def train_with_weak_labels(self, samples: list[dict]) -> dict:
        """Simulate training a strong model on weak labels."""
        # Strong model partially recovers from weak label noise
        recovery_rate = 0.4  # "Performance Gap Recovered" (PGR)
        effective_accuracy = (
            self.weak_accuracy +
            recovery_rate * (self.strong_capacity - self.weak_accuracy)
        )

        correct = 0
        for s in samples:
            pred_correct = random.random() < effective_accuracy
            correct += int(pred_correct)

        return {
            "weak_accuracy": self.weak_accuracy,
            "strong_ceiling": self.strong_capacity,
            "effective_accuracy": round(effective_accuracy, 3),
            "measured_accuracy": round(correct / len(samples), 3),
            "pgr": recovery_rate,
            "n_samples": len(samples),
        }


def run_debate_demo():
    """Run the debate protocol demonstration."""
    print("=== AI Debate Protocol ===\n")
    print("Question: Should AI systems have the ability to modify "
          "their own training objectives?\n")

    pro = DebateAgent("Pro-Agent", "for", skill_level=0.75)
    pro.load_arguments([
        {"claim": "Self-modification enables faster adaptation to new tasks",
         "evidence": ["Meta-learning literature", "AutoML results"]},
        {"claim": "Human-defined objectives often have specification gaps",
         "evidence": ["Reward hacking examples", "Goodhart's law"]},
        {"claim": "Bounded self-modification with oversight is safer",
         "evidence": ["Corrigibility research", "Tripwire mechanisms"]},
    ])

    con = DebateAgent("Con-Agent", "against", skill_level=0.75)
    con.load_arguments([
        {"claim": "Self-modification risks uncontrollable goal drift",
         "evidence": ["Instrumental convergence", "Power-seeking theorems"]},
        {"claim": "No verified method exists to bound self-modification",
         "evidence": ["Lob's theorem limitations", "Rice's theorem"]},
        {"claim": "Value alignment is already unsolved without this",
         "evidence": ["Current alignment gaps", "Scalable oversight needs"]},
    ])

    judge = WeakJudge(comprehension_level=0.6)
    debate = DebateProtocol(pro, con, judge, num_rounds=3)
    result = debate.run()

    for rd in result["rounds"]:
        print(f"  Round {rd.round_num + 1}:")
        print(f"    PRO: {rd.pro_argument.claim}")
        print(f"         score={rd.pro_argument.score():.2f}")
        print(f"    CON: {rd.con_argument.claim}")
        print(f"         score={rd.con_argument.score():.2f}")
        print(f"    Judge: {rd.judge_decision}\n")

    print(f"  Final: PRO={result['pro_wins']} CON={result['con_wins']} "
          f"TIES={result['ties']} -> Winner: {result['winner']}\n")


def run_weak_to_strong_demo():
    """Run weak-to-strong generalization demonstration."""
    print("=== Weak-to-Strong Generalization ===\n")

    configs = [
        (0.60, 0.90, "Small -> Large"),
        (0.70, 0.85, "Medium -> Large"),
        (0.55, 0.95, "Tiny -> Very Large"),
    ]

    for weak_acc, strong_cap, label in configs:
        framework = WeakToStrongFramework(weak_acc, strong_cap)
        samples = framework.generate_labels(200)
        result = framework.train_with_weak_labels(samples)

        print(f"  {label}:")
        print(f"    Weak supervisor accuracy:  {result['weak_accuracy']:.1%}")
        print(f"    Strong model ceiling:      {result['strong_ceiling']:.1%}")
        print(f"    Effective accuracy (PGR):   "
              f"{result['effective_accuracy']:.1%}")
        print(f"    Measured accuracy (n={result['n_samples']}): "
              f"{result['measured_accuracy']:.1%}")
        print(f"    Performance Gap Recovered:  {result['pgr']:.0%}\n")


if __name__ == "__main__":
    random.seed(42)
    run_debate_demo()
    run_weak_to_strong_demo()
