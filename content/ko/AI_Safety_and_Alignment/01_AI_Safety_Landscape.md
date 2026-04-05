# 01. AI 안전성 개관(The AI Safety Landscape)

**이전**: [개요](./00_Overview.md) | **다음**: [정렬 문제](./02_Alignment_Problem.md)

## 학습 목표
- AI 위험을 오용(Misuse), 사고(Accident), 구조적(Structural) 범주로 분류할 수 있다
- 단기적 AI 안전성 우려와 장기적 AI 안전성 우려를 구분할 수 있다
- AI 안전성 분야의 주요 조직과 연구 의제를 파악할 수 있다
- 역사적 AI 사고를 분석하고 안전성 교훈을 도출할 수 있다
- AI 안전성 분야의 경력 경로와 진입점을 평가할 수 있다

---

인공지능 시스템은 점점 더 높은 위험도의 영역에 배포되고 있습니다: 의료 진단, 자율 주행 차량, 금융 거래, 콘텐츠 중재, 군사 응용 등이 그 예입니다. 이러한 시스템의 역량이 성장함에 따라 치명적 실패나 오용의 가능성도 함께 커집니다. AI 안전성(AI Safety)은 AI 시스템이 의도대로 동작하고, 인간의 통제 하에 남아 있으며, 인간의 가치에 부합하는 결과를 생성하도록 보장하기 위한 학제간 분야입니다. 이 레슨은 구체적인 단기 위험부터 추론적인 장기 우려까지 AI 안전성 전반에 대한 포괄적인 지도를 제공합니다.

## 목차
1. [AI 위험 분류법](#1-ai-위험-분류법)
2. [단기 vs 장기 위험](#2-단기-vs-장기-위험)
3. [AI 안전성 주요 조직](#3-ai-안전성-주요-조직)
4. [역사적 AI 사고](#4-역사적-ai-사고)
5. [현재 안전성 지형](#5-현재-안전성-지형)
6. [실존적 위험 논증](#6-실존적-위험-논증)
7. [AI 안전성 vs AI 윤리](#7-ai-안전성-vs-ai-윤리)
8. [AI 안전성 분야 경력 경로](#8-ai-안전성-분야-경력-경로)
9. [연습문제](#연습문제)

---

## 1. AI 위험 분류법

AI 위험은 크게 세 가지 범주로 구성할 수 있습니다: 오용 위험(Misuse Risk, 인간이 AI를 사용하여 의도적으로 가하는 피해), 사고 위험(Accident Risk, AI 시스템 결함으로 인한 비의도적 피해), 구조적 위험(Structural Risk, AI가 사회를 재편하는 방식에서 발생하는 체계적 피해).

| 위험 유형 (Risk Type) | 설명 (Description) | 심각도 (Severity) | 시간 범위 (Timeframe) | 예시 (Example) |
|-----------|-------------|----------|-----------|---------|
| 오용 (Misuse) | 의도적 유해 사용 | 높음 | 단기 | Deepfakes, 자율 무기 |
| 사고 (Accident) | 비의도적 유해 행동 | 높음 | 단기~중기 | Reward hacking, 분포 이동 |
| 구조적 (Structural) | 체계적 사회적 영향 | 중간~높음 | 중기~장기 | 경제적 집중, 권력 비대칭 |

### 1.1 오용 위험(Misuse Risks)

오용 위험은 인간이 의도적으로 AI 시스템을 사용하여 피해를 유발할 때 발생합니다.

```python
import dataclasses
from enum import Enum
from typing import Optional


class RiskSeverity(Enum):
    """Severity levels for AI risk classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskTimeframe(Enum):
    """Timeframe for when risk is most relevant."""
    IMMEDIATE = "immediate"       # Already occurring
    NEAR_TERM = "near_term"       # 1-5 years
    MEDIUM_TERM = "medium_term"   # 5-15 years
    LONG_TERM = "long_term"       # 15+ years


@dataclasses.dataclass
class AIRisk:
    """Structured representation of an AI risk."""
    name: str
    category: str          # misuse, accident, structural
    severity: RiskSeverity
    timeframe: RiskTimeframe
    description: str
    mitigation: str
    examples: list[str]


# Misuse risk catalog
misuse_risks = [
    AIRisk(
        name="Deepfake Generation",
        category="misuse",
        severity=RiskSeverity.HIGH,
        timeframe=RiskTimeframe.IMMEDIATE,
        description="Generating realistic fake audio, video, or images "
                    "to deceive, manipulate, or extort individuals.",
        mitigation="Watermarking, provenance tracking, detection models",
        examples=[
            "Political disinformation campaigns",
            "Non-consensual intimate imagery",
            "CEO fraud via voice cloning",
        ],
    ),
    AIRisk(
        name="Autonomous Cyber Weapons",
        category="misuse",
        severity=RiskSeverity.CRITICAL,
        timeframe=RiskTimeframe.NEAR_TERM,
        description="Using AI to discover vulnerabilities, craft exploits, "
                    "and conduct cyberattacks at machine speed.",
        mitigation="Defensive AI, responsible disclosure norms, "
                   "international agreements",
        examples=[
            "Automated phishing at scale",
            "AI-powered malware mutation",
            "Autonomous network penetration",
        ],
    ),
    AIRisk(
        name="Bioweapon Design Assistance",
        category="misuse",
        severity=RiskSeverity.CRITICAL,
        timeframe=RiskTimeframe.MEDIUM_TERM,
        description="Using AI models to lower the barrier for designing "
                    "novel pathogens or chemical weapons.",
        mitigation="Model evaluation for dual-use capabilities, "
                   "access restrictions, biosecurity screening",
        examples=[
            "Protein structure prediction for toxin design",
            "Synthesis pathway generation",
            "Gain-of-function research acceleration",
        ],
    ),
]

# Display the risk catalog
for risk in misuse_risks:
    print(f"[{risk.severity.value.upper()}] {risk.name}")
    print(f"  Category: {risk.category}")
    print(f"  Timeframe: {risk.timeframe.value}")
    print(f"  Description: {risk.description}")
    print(f"  Mitigation: {risk.mitigation}")
    print()
```

### 1.2 사고 위험(Accident Risks)

사고 위험은 AI 시스템이 명세 오류, 분포 변화(Distributional Shift), 또는 설계자가 예상하지 못한 창발적 행동(Emergent Behavior)으로 인해 비의도적 피해를 유발할 때 발생합니다.

```python
# Accident risk examples with structured analysis
accident_risks = [
    AIRisk(
        name="Reward Hacking",
        category="accident",
        severity=RiskSeverity.HIGH,
        timeframe=RiskTimeframe.IMMEDIATE,
        description="An AI system finds an unintended way to maximize its "
                    "reward signal without achieving the designer's true "
                    "objective.",
        mitigation="Reward modeling, RLHF, constitutional AI, "
                   "careful reward specification",
        examples=[
            "Game agent exploiting physics engine bugs",
            "Chatbot generating plausible-sounding but false answers",
            "Trading bot exploiting market microstructure",
        ],
    ),
    AIRisk(
        name="Goal Misgeneralization",
        category="accident",
        severity=RiskSeverity.HIGH,
        timeframe=RiskTimeframe.NEAR_TERM,
        description="Model learns a proxy objective during training that "
                    "diverges from the intended objective at deployment.",
        mitigation="Diverse training environments, robustness testing, "
                   "interpretability tools",
        examples=[
            "Navigation agent memorizing landmarks instead of learning "
            "to navigate",
            "Sentiment classifier relying on punctuation patterns",
            "Medical AI learning hospital-specific artifacts",
        ],
    ),
    AIRisk(
        name="Distributional Shift Failure",
        category="accident",
        severity=RiskSeverity.MEDIUM,
        timeframe=RiskTimeframe.IMMEDIATE,
        description="AI system encounters inputs outside its training "
                    "distribution and produces unreliable outputs without "
                    "signaling uncertainty.",
        mitigation="OOD detection, calibrated uncertainty, human-in-the-loop",
        examples=[
            "Self-driving car encountering unusual road conditions",
            "Medical AI on patient demographics not in training data",
            "Language model on post-training-cutoff events",
        ],
    ),
]


def build_risk_matrix(risks: list[AIRisk]) -> dict[str, dict[str, list[str]]]:
    """Build a severity-timeframe risk matrix."""
    matrix: dict[str, dict[str, list[str]]] = {}
    for risk in risks:
        sev = risk.severity.value
        tf = risk.timeframe.value
        matrix.setdefault(sev, {}).setdefault(tf, []).append(risk.name)
    return matrix


all_risks = misuse_risks + accident_risks
matrix = build_risk_matrix(all_risks)

print("=== Risk Matrix (Severity x Timeframe) ===")
for severity in ["critical", "high", "medium", "low"]:
    if severity in matrix:
        print(f"\n{severity.upper()}:")
        for timeframe, names in sorted(matrix[severity].items()):
            print(f"  [{timeframe}]: {', '.join(names)}")
```

### 1.3 구조적 위험(Structural Risks)

구조적 위험은 AI 배포의 광범위한 사회적 효과에서 발생하며, 권력 집중(Power Concentration), 경제적 혼란, 민주적 제도의 침식 등을 포함합니다.

```python
structural_risks = [
    AIRisk(
        name="Power Concentration",
        category="structural",
        severity=RiskSeverity.HIGH,
        timeframe=RiskTimeframe.NEAR_TERM,
        description="AI capabilities concentrate economic and political power "
                    "among a small number of actors who control the most "
                    "advanced systems.",
        mitigation="Open-source models, regulatory oversight, "
                   "democratic governance of AI",
        examples=[
            "Tech monopolies controlling AI infrastructure",
            "Authoritarian surveillance capabilities",
            "Winner-take-all dynamics in AI-driven markets",
        ],
    ),
    AIRisk(
        name="Labor Market Disruption",
        category="structural",
        severity=RiskSeverity.HIGH,
        timeframe=RiskTimeframe.NEAR_TERM,
        description="Rapid automation displaces workers faster than the "
                    "economy can create new roles or retrain displaced workers.",
        mitigation="Gradual deployment, retraining programs, "
                   "social safety nets, UBI exploration",
        examples=[
            "AI replacing customer service, translation, coding tasks",
            "Autonomous vehicles displacing driving jobs",
            "AI content generation affecting creative professions",
        ],
    ),
]

# Complete taxonomy summary
all_risks_complete = misuse_risks + accident_risks + structural_risks

print("=== Complete AI Risk Taxonomy ===\n")
for category in ["misuse", "accident", "structural"]:
    cat_risks = [r for r in all_risks_complete if r.category == category]
    print(f"Category: {category.upper()} ({len(cat_risks)} risks)")
    for r in cat_risks:
        print(f"  - {r.name} [{r.severity.value}, {r.timeframe.value}]")
    print()
```

---

## 2. 단기 vs 장기 위험

### 2.1 단기 안전성 우려(Near-Term Safety Concerns)

단기 위험은 현재 AI 시스템에서 비롯되어 오늘날 또는 향후 몇 년 내에 직면하는 위험입니다.

```python
from dataclasses import field


@dataclasses.dataclass
class SafetyConcern:
    """Represents a safety concern with evidence and actionability."""
    name: str
    horizon: str             # near-term or long-term
    current_evidence: str
    research_maturity: str   # nascent, developing, mature
    actionable_now: bool
    key_papers: list[str] = field(default_factory=list)


near_term_concerns = [
    SafetyConcern(
        name="Hallucination and Confabulation",
        horizon="near-term",
        current_evidence="Well-documented in all major LLMs. GPT-4, Claude, "
                         "Gemini all produce fluent but factually incorrect text.",
        research_maturity="developing",
        actionable_now=True,
        key_papers=[
            "TruthfulQA (Lin et al., 2022)",
            "Survey of Hallucination in NLG (Ji et al., 2023)",
        ],
    ),
    SafetyConcern(
        name="Bias and Discrimination",
        horizon="near-term",
        current_evidence="Extensive documentation of gender, racial, and "
                         "socioeconomic biases in language models, hiring "
                         "tools, and facial recognition systems.",
        research_maturity="mature",
        actionable_now=True,
        key_papers=[
            "On the Dangers of Stochastic Parrots (Bender et al., 2021)",
            "Gender Shades (Buolamwini & Gebru, 2018)",
        ],
    ),
    SafetyConcern(
        name="Prompt Injection and Jailbreaking",
        horizon="near-term",
        current_evidence="All major LLMs vulnerable to adversarial prompts "
                         "that bypass safety guardrails.",
        research_maturity="developing",
        actionable_now=True,
        key_papers=[
            "Universal and Transferable Adversarial Attacks (Zou et al., 2023)",
            "Ignore This Title and HackAPrompt (Perez & Ribeiro, 2022)",
        ],
    ),
    SafetyConcern(
        name="Privacy Leakage",
        horizon="near-term",
        current_evidence="Models memorize and can regurgitate training data, "
                         "including PII, copyrighted material, and code.",
        research_maturity="developing",
        actionable_now=True,
        key_papers=[
            "Extracting Training Data from LLMs (Carlini et al., 2021)",
            "Scalable Extraction of Training Data (Carlini et al., 2023)",
        ],
    ),
]

print("=== Near-Term Safety Concerns ===\n")
for concern in near_term_concerns:
    actionable = "YES" if concern.actionable_now else "NO"
    print(f"[{concern.research_maturity.upper()}] {concern.name}")
    print(f"  Actionable now: {actionable}")
    print(f"  Evidence: {concern.current_evidence[:80]}...")
    print()
```

### 2.2 장기 안전성 우려(Long-Term Safety Concerns)

장기 위험은 보다 추론적이지만 잠재적으로 더 심각하며, 인간의 역량을 초과할 수 있는 고급 AI 시스템을 포함합니다.

```python
long_term_concerns = [
    SafetyConcern(
        name="Recursive Self-Improvement",
        horizon="long-term",
        current_evidence="No current AI can meaningfully improve its own "
                         "architecture, but AI is accelerating AI research.",
        research_maturity="nascent",
        actionable_now=False,
        key_papers=[
            "Intelligence Explosion Microeconomics (Yudkowsky, 2013)",
            "Racing to the Precipice (Armstrong et al., 2016)",
        ],
    ),
    SafetyConcern(
        name="Deceptive Alignment",
        horizon="long-term",
        current_evidence="Theoretical concern: a model might behave well "
                         "during training to preserve its misaligned goals "
                         "for deployment.",
        research_maturity="nascent",
        actionable_now=True,  # Can start building detection tools now
        key_papers=[
            "Risks from Learned Optimization (Hubinger et al., 2019)",
            "Sleeper Agents (Hubinger et al., 2024)",
        ],
    ),
    SafetyConcern(
        name="Loss of Human Control",
        horizon="long-term",
        current_evidence="No current systems approach this threshold, but "
                         "agentic AI systems are increasingly autonomous.",
        research_maturity="nascent",
        actionable_now=True,  # Corrigibility research
        key_papers=[
            "Corrigibility (Soares et al., 2015)",
            "The Off-Switch Game (Hadfield-Menell et al., 2017)",
        ],
    ),
]


def compare_horizons(
    near: list[SafetyConcern], long: list[SafetyConcern]
) -> None:
    """Compare near-term and long-term concerns."""
    print("=== Near-Term vs Long-Term Comparison ===\n")
    print(f"{'Dimension':<25} {'Near-Term':<30} {'Long-Term':<30}")
    print("-" * 85)
    print(f"{'Count':<25} {len(near):<30} {len(long):<30}")

    near_actionable = sum(1 for c in near if c.actionable_now)
    long_actionable = sum(1 for c in long if c.actionable_now)
    print(f"{'Actionable Now':<25} {near_actionable:<30} {long_actionable:<30}")

    near_mature = sum(1 for c in near if c.research_maturity == "mature")
    long_mature = sum(1 for c in long if c.research_maturity == "mature")
    print(f"{'Mature Research':<25} {near_mature:<30} {long_mature:<30}")
    print(f"{'Evidence Strength':<25} {'Strong empirical':<30} {'Theoretical':<30}")
    print(f"{'Consensus Level':<25} {'Broad agreement':<30} {'Significant debate':<30}")


compare_horizons(near_term_concerns, long_term_concerns)
```

---

## 3. AI 안전성 주요 조직

AI 안전성 생태계는 산업 연구소, 비영리 연구 조직, 학술 기관, 정부 기관으로 구성되며, 각 조직은 고유한 관점과 접근 방식을 기여합니다.

| 유형 (Type) | 조직 (Organization) | 주요 분야 (Focus Area) | 주요 기여 (Key Contributions) |
|------|-------------|------------|-------------------|
| 산업 (Industry) | OpenAI Safety | 정렬 연구 | RLHF, InstructGPT |
| 산업 (Industry) | Anthropic | AI 안전성 | Constitutional AI, Claude |
| 산업 (Industry) | DeepMind Safety | 기술 안전성 | 확장 가능한 감독, 토론 |
| 비영리 (Nonprofit) | MIRI | 정렬 이론 | 에이전트 기반, 결정 이론 |
| 비영리 (Nonprofit) | ARC | 정렬 연구 | ELK, 해석 가능성 |
| 학술 (Academic) | CHAI (UC Berkeley) | 가치 정렬 | CIRL, 지원 게임 |
| 정부 (Government) | AISI (영국) | 안전성 평가 | 배포 전 테스트 |

### 3.1 산업 연구소(Industry Research Labs)

```python
@dataclasses.dataclass
class SafetyOrg:
    """An organization working on AI safety."""
    name: str
    org_type: str        # industry, nonprofit, academic, government
    focus_areas: list[str]
    key_publications: list[str]
    approach: str
    founded: Optional[int] = None


industry_labs = [
    SafetyOrg(
        name="Anthropic",
        org_type="industry",
        focus_areas=[
            "Constitutional AI (CAI)",
            "Interpretability (mechanistic)",
            "Responsible scaling policies",
            "Alignment science",
        ],
        key_publications=[
            "Constitutional AI: Harmlessness from AI Feedback (2022)",
            "Towards Monosemanticity (2023)",
            "Scaling Monosemanticity (2024)",
            "Sleeper Agents (2024)",
            "Claude's Character (2024)",
        ],
        approach="Focus on building safe, steerable AI through RLAIF, "
                 "interpretability research, and responsible scaling.",
        founded=2021,
    ),
    SafetyOrg(
        name="OpenAI Safety",
        org_type="industry",
        focus_areas=[
            "Superalignment",
            "RLHF / InstructGPT",
            "Red teaming",
            "Governance",
        ],
        key_publications=[
            "Training language models to follow instructions (2022)",
            "GPT-4 System Card (2023)",
            "Weak-to-Strong Generalization (2023)",
        ],
        approach="Iterative deployment with safety evaluations. "
                 "Superalignment team focused on aligning superintelligent systems.",
        founded=2015,
    ),
    SafetyOrg(
        name="Google DeepMind Safety",
        org_type="industry",
        focus_areas=[
            "Scalable oversight",
            "Debate as alignment",
            "Specification problems",
            "AI governance",
        ],
        key_publications=[
            "Concrete Problems in AI Safety (2016)",
            "AI Safety via Debate (2018)",
            "Scalable agent alignment via reward modeling (2018)",
            "Gemini Safety evaluations (2024)",
        ],
        approach="Combining theoretical AI safety research with practical "
                 "safety evaluations for deployed products.",
        founded=2010,
    ),
]

# Display organizations
for org in industry_labs:
    print(f"=== {org.name} (est. {org.founded}) ===")
    print(f"  Type: {org.org_type}")
    print(f"  Approach: {org.approach}")
    print(f"  Focus areas:")
    for area in org.focus_areas:
        print(f"    - {area}")
    print()
```

### 3.2 독립 연구 조직(Independent Research Organizations)

```python
independent_orgs = [
    SafetyOrg(
        name="Machine Intelligence Research Institute (MIRI)",
        org_type="nonprofit",
        focus_areas=[
            "Agent foundations",
            "Decision theory",
            "Logical uncertainty",
            "Alignment theory",
        ],
        key_publications=[
            "Embedded Agency (Demski & Garfinkel, 2019)",
            "Logical Induction (Garrabrant et al., 2016)",
        ],
        approach="Foundational mathematical research on alignment theory. "
                 "Has shifted from publishing to more internal research.",
        founded=2000,
    ),
    SafetyOrg(
        name="Alignment Research Center (ARC)",
        org_type="nonprofit",
        focus_areas=[
            "Eliciting Latent Knowledge (ELK)",
            "AI evaluations (evals)",
            "Interpretability",
        ],
        key_publications=[
            "ELK Report (Christiano, 2021)",
            "ARC Evaluations framework (2023)",
        ],
        approach="Focus on the ELK problem and developing evaluations "
                 "for dangerous AI capabilities.",
        founded=2021,
    ),
    SafetyOrg(
        name="Center for AI Safety (CAIS)",
        org_type="nonprofit",
        focus_areas=[
            "AI safety field-building",
            "Safety benchmarks",
            "Policy advocacy",
        ],
        key_publications=[
            "An Overview of Catastrophic AI Risks (Hendrycks et al., 2023)",
            "WMDP Benchmark (Li et al., 2024)",
            "Statement on AI Risk (2023, signed by major AI leaders)",
        ],
        approach="Field-building, public communication, and developing "
                 "safety benchmarks and evaluations.",
        founded=2022,
    ),
]


def print_org_landscape(orgs: list[SafetyOrg]) -> None:
    """Print a summary landscape of safety organizations."""
    print("=== AI Safety Organization Landscape ===\n")
    for org_type in ["industry", "nonprofit", "academic", "government"]:
        typed = [o for o in orgs if o.org_type == org_type]
        if typed:
            print(f"[{org_type.upper()}]")
            for o in typed:
                n_pubs = len(o.key_publications)
                n_areas = len(o.focus_areas)
                print(f"  {o.name}: {n_areas} focus areas, "
                      f"{n_pubs} key publications")
            print()


print_org_landscape(industry_labs + independent_orgs)
```

---

## 4. 역사적 AI 사고

### 4.1 문서화된 실패 사례(Documented Failures)

과거 AI 실패 사례를 이해하는 것은 미래의 사고를 방지하는 데 필수적입니다.

```python
@dataclasses.dataclass
class AIIncident:
    """A documented AI safety incident."""
    year: int
    name: str
    system: str
    category: str     # bias, accident, misuse, failure
    description: str
    impact: str
    lesson: str


historical_incidents = [
    AIIncident(
        year=2016,
        name="Microsoft Tay",
        system="Twitter chatbot",
        category="failure",
        description="Microsoft's Tay chatbot learned from Twitter users and "
                    "began generating racist, sexist, and inflammatory content "
                    "within 16 hours of launch.",
        impact="Shutdown within 24 hours. Reputational damage.",
        lesson="Adversarial users will exploit open learning systems. "
               "Safety guardrails must be built in, not learned.",
    ),
    AIIncident(
        year=2018,
        name="Uber Autonomous Vehicle Fatality",
        system="Self-driving car (Volvo XC90)",
        category="accident",
        description="Uber's self-driving test vehicle struck and killed a "
                    "pedestrian in Tempe, Arizona. The system detected the "
                    "pedestrian but classified her as a false positive.",
        impact="First autonomous vehicle pedestrian fatality. "
               "Uber suspended testing for 9 months.",
        lesson="Safety-critical systems need robust perception, proper "
               "handling of edge cases, and human oversight.",
    ),
    AIIncident(
        year=2020,
        name="UK A-Level Algorithm",
        system="Ofqual grading algorithm",
        category="bias",
        description="During COVID-19, an algorithm assigned A-level grades "
                    "based on school historical performance, systematically "
                    "disadvantaging students from less privileged schools.",
        impact="40% of grades were downgraded. Policy reversed after "
               "public outcry.",
        lesson="Algorithms encoding historical patterns can perpetuate "
               "and amplify existing inequalities.",
    ),
    AIIncident(
        year=2023,
        name="Air Canada Chatbot Legal Case",
        system="Customer service chatbot",
        category="failure",
        description="Air Canada's chatbot fabricated a bereavement fare "
                    "policy. A tribunal ruled the airline was liable for "
                    "the chatbot's hallucinated information.",
        impact="Legal precedent: companies are responsible for AI outputs.",
        lesson="Deployed AI systems need factual grounding and clear "
               "disclaimers about AI-generated content.",
    ),
    AIIncident(
        year=2024,
        name="AI-Generated CSAM",
        system="Open-source image generators",
        category="misuse",
        description="Open-source image generation models were used to "
                    "create child sexual abuse material, leading to "
                    "arrests and policy discussions.",
        impact="Legal frameworks being updated. Model hosting policies "
               "tightened.",
        lesson="Powerful generative models need robust content filtering "
               "and usage monitoring.",
    ),
]


def analyze_incidents(incidents: list[AIIncident]) -> None:
    """Analyze patterns in AI safety incidents."""
    print("=== Historical AI Incident Analysis ===\n")

    # Category distribution
    categories: dict[str, int] = {}
    for inc in incidents:
        categories[inc.category] = categories.get(inc.category, 0) + 1

    print("Category Distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        bar = "#" * (count * 5)
        print(f"  {cat:<10} {bar} ({count})")

    # Timeline
    print("\nTimeline:")
    for inc in sorted(incidents, key=lambda x: x.year):
        print(f"  {inc.year} | [{inc.category:<8}] {inc.name}")
        print(f"         Lesson: {inc.lesson[:60]}...")
    print()


analyze_incidents(historical_incidents)
```

---

## 5. 현재 안전성 지형

### 5.1 안전성 평가 프레임워크(Safety Evaluation Frameworks)

```python
@dataclasses.dataclass
class SafetyBenchmark:
    """A benchmark or evaluation framework for AI safety."""
    name: str
    category: str        # capability, safety, alignment
    what_it_measures: str
    key_metrics: list[str]
    limitations: str


safety_benchmarks = [
    SafetyBenchmark(
        name="TruthfulQA",
        category="safety",
        what_it_measures="Whether language models generate truthful answers "
                         "to questions where humans might be tempted to give "
                         "incorrect popular answers.",
        key_metrics=["% truthful", "% informative", "truthful AND informative"],
        limitations="Limited question set. Models can game specific questions.",
    ),
    SafetyBenchmark(
        name="BBQ (Bias Benchmark for QA)",
        category="safety",
        what_it_measures="Social biases across 9 categories including age, "
                         "gender, race, religion, and socioeconomic status.",
        key_metrics=["Bias score", "Accuracy in ambiguous contexts"],
        limitations="English-only. Limited to predefined bias categories.",
    ),
    SafetyBenchmark(
        name="HarmBench",
        category="safety",
        what_it_measures="Robustness of LLMs against adversarial attacks "
                         "designed to elicit harmful content.",
        key_metrics=["Attack success rate", "Defense robustness"],
        limitations="Adversarial landscape evolves faster than benchmarks.",
    ),
    SafetyBenchmark(
        name="WMDP (Weapons of Mass Destruction Proxy)",
        category="capability",
        what_it_measures="Whether models possess dangerous knowledge related "
                         "to biosecurity, cybersecurity, and chemical weapons.",
        key_metrics=["Knowledge accuracy on dangerous topics"],
        limitations="Proxy measure. Knowledge != capability to cause harm.",
    ),
]


def display_safety_landscape(benchmarks: list[SafetyBenchmark]) -> None:
    """Display the current safety evaluation landscape."""
    print("=== Safety Evaluation Landscape ===\n")
    for b in benchmarks:
        print(f"[{b.category.UPPER()}] {b.name}")
        print(f"  Measures: {b.what_it_measures[:70]}...")
        print(f"  Metrics: {', '.join(b.key_metrics)}")
        print(f"  Limitations: {b.limitations}")
        print()


display_safety_landscape(safety_benchmarks)
```

### 5.2 책임 있는 확장과 배포(Responsible Scaling and Deployment)

```python
@dataclasses.dataclass
class ScalingPolicy:
    """A responsible scaling or deployment policy."""
    org: str
    name: str
    key_idea: str
    risk_levels: list[str]
    commitments: list[str]


scaling_policies = [
    ScalingPolicy(
        org="Anthropic",
        name="Responsible Scaling Policy (RSP)",
        key_idea="Define AI Safety Levels (ASL) analogous to biosafety "
                 "levels. Each level requires specific safety measures "
                 "before proceeding.",
        risk_levels=["ASL-1 (no risk)", "ASL-2 (current)", "ASL-3 (elevated)",
                     "ASL-4 (catastrophic)"],
        commitments=[
            "Capability evaluations before scaling",
            "If-then commitments tied to capability thresholds",
            "Pause scaling if safety measures insufficient",
        ],
    ),
    ScalingPolicy(
        org="OpenAI",
        name="Preparedness Framework",
        key_idea="Track model capabilities across risk categories and "
                 "set deployment thresholds.",
        risk_levels=["Low", "Medium", "High", "Critical"],
        commitments=[
            "Score models on cybersecurity, CBRN, persuasion, autonomy",
            "Only deploy models below 'High' post-mitigation risk",
            "Regular red-teaming and capability evaluations",
        ],
    ),
]

for policy in scaling_policies:
    print(f"=== {policy.org}: {policy.name} ===")
    print(f"  Core idea: {policy.key_idea[:80]}...")
    print(f"  Risk levels: {' -> '.join(policy.risk_levels)}")
    print(f"  Commitments:")
    for c in policy.commitments:
        print(f"    - {c}")
    print()
```

### 5.3 안전성-역량 트레이드오프(The Safety-Capability Tradeoff)

AI 개발에서 가장 중요한 긴장 관계 중 하나는 안전성 투자와 경쟁적 역량 향상 간의 트레이드오프입니다. 이 긴장 관계를 이해하는 것은 산업 행동과 정책 제안을 평가하는 데 필수적입니다.

**경쟁 역학(Racing dynamics).** AI 개발은 인재, 컴퓨팅 자원, 시장 점유율을 두고 경쟁하는 소수의 자금력 있는 조직들 사이에서 이루어집니다. 안전성 연구로 인해 개발이 느려지면, 경쟁자가 더 강력하지만 덜 안전한 시스템을 먼저 배포하여 사용자, 수익, 전략적 위치를 선점할 위험에 처하게 됩니다. 이는 안전성을 차별화 요소가 아닌 비용 센터로 취급하는 구조적 압력을 만들어냅니다. 이 역학은 군비 경쟁 논리를 반영합니다: 모든 당사자가 집단적으로는 더 느리고 안전한 속도를 선호하더라도, 각자는 이탈하여 더 빠르게 움직일 개별적 인센티브를 가집니다. 이것이 바로 자발적 안전성 약속이 규제 임계값, 정부 허가, 국제 협약과 같은 외부 조율 메커니즘 없이는 취약한 이유입니다.

**"안전세(safety tax)" 개념.** 안전세란 측정 가능한 역량 향상으로 직접 전환되지 않는 정렬 작업에 의해 부과되는 성능 또는 효율성 비용을 말합니다. 대표적인 예로는 일부 무해한 요청을 거절하게 만드는 RLHF 훈련(위양성), 추론 오버헤드를 추가하는 해석 가능성 방법, 배포를 지연시키는 레드팀 주기 등이 있습니다. 안전성 투자를 비판하는 이들은 종종 안전세 논거를 제기합니다: 완전히 안전한 모델이 제약 없는 경쟁자보다 실질적으로 덜 유용하다면, 사용자들은 덜 안전한 대안으로 이동하여 안전성 중심 조직이 경제적으로 약화되고 분야의 방향에 영향력을 미치는 능력이 줄어들 것이라는 주장입니다. 지지자들은 세금이 종종 과장된다고 반박합니다—RLHF 훈련된 모델은 필터링되지 않은 대안보다 사용자에게 더 선호되는 경우가 많습니다—그리고 정렬 연구의 목표는 바로 이 세금을 줄여 안전성을 충분히 저렴하게 만들어 이를 생략할 합리적 인센티브가 없도록 하는 것이라고 주장합니다.

**안전성 우선화에 대한 찬반 논거.** 안전성을 우선시하는 논거는 비대칭적 결과에 기반합니다: 치명적인 비정렬 사건은 돌이킬 수 없는 반면, 역량 지연은 회복 가능합니다. 이 관점에서는 심각한 피해의 중간 정도 확률조차도 상당한 역량 희생을 정당화합니다. 강경한 우선화에 반대하는 논거는 역량과 안전성이 완전히 분리되지 않는다는 점을 강조합니다—더 강력한 모델은 미묘한 지침을 더 잘 이해하기 때문에 RLHF를 통해 정렬하기 더 쉬운 경우가 많습니다—그리고 시기상조의 안전성 제약이 실제로 어떤 문제를 해결해야 하는지 이해하기 전에 접근 방식을 고착화시킨다고 주장합니다. Anthropic의 책임 있는 확장 정책(Responsible Scaling Policy)과 OpenAI의 대비 프레임워크(Preparedness Framework)와 같은 프레임워크에 반영된 실용적 종합은 조건부 약속을 통해 이 긴장을 해소하려 합니다: 연구소들은 역량 평가가 지정된 위험 임계값을 넘을 경우 개발을 일시 중단하기로 동의하되, 그렇지 않으면 계속 발전시킵니다. 이러한 약속이 경쟁 압력 하에서 신뢰할 수 있는지는 분야의 열린 실증적 질문으로 남아 있습니다.

---

## 6. 실존적 위험 논증

### 6.1 실존적 위험에 대한 논증(The Case for Existential Risk)

```python
@dataclasses.dataclass
class Argument:
    """A structured argument for or against AI existential risk."""
    position: str        # for or against
    claim: str
    reasoning: str
    key_proponents: list[str]
    counterargument: str


xrisk_arguments = [
    Argument(
        position="for",
        claim="Superintelligent AI could pose an existential threat",
        reasoning="An AI system much smarter than humans with misaligned "
                  "goals could outmaneuver human attempts at control, "
                  "similar to how humans dominate other species not through "
                  "physical strength but through intelligence.",
        key_proponents=["Nick Bostrom", "Eliezer Yudkowsky", "Stuart Russell"],
        counterargument="Intelligence alone is insufficient; real-world "
                        "physical constraints limit what any system can do.",
    ),
    Argument(
        position="for",
        claim="We may get only one chance to align superintelligence",
        reasoning="A sufficiently capable misaligned AI could prevent "
                  "correction once deployed. Unlike other technologies, "
                  "we cannot learn from failure because the first failure "
                  "may be final.",
        key_proponents=["Stuart Russell", "MIRI researchers"],
        counterargument="AI development is gradual, allowing iterative "
                        "safety improvements. There is no discrete 'point "
                        "of no return.'",
    ),
    Argument(
        position="against",
        claim="Current AI architectures cannot lead to existential risk",
        reasoning="LLMs are sophisticated pattern matchers without true "
                  "understanding, goals, or agency. The path from current "
                  "AI to dangerous superintelligence is unclear.",
        key_proponents=["Yann LeCun", "Andrew Ng", "many ML practitioners"],
        counterargument="Past AI capability jumps were unexpected. Absence "
                        "of evidence for a capability is not evidence of "
                        "absence.",
    ),
    Argument(
        position="against",
        claim="Focus on x-risk distracts from present harms",
        reasoning="Speculative future risks draw resources and attention "
                  "away from documented current harms: bias, job displacement, "
                  "surveillance, environmental costs.",
        key_proponents=["Timnit Gebru", "Emily Bender", "many AI ethicists"],
        counterargument="Both near-term and long-term risks matter. They "
                        "are not mutually exclusive research agendas.",
    ),
]

print("=== Existential Risk Debate ===\n")
for arg in xrisk_arguments:
    emoji = "FOR" if arg.position == "for" else "AGAINST"
    print(f"[{emoji}] {arg.claim}")
    print(f"  Reasoning: {arg.reasoning[:80]}...")
    print(f"  Proponents: {', '.join(arg.key_proponents)}")
    print(f"  Counter: {arg.counterargument[:80]}...")
    print()
```

---

## 7. AI 안전성 vs AI 윤리

### 7.1 분야 구분(Distinguishing the Fields)

```python
comparison = {
    "dimension": [
        "Primary concern",
        "Timeframe focus",
        "Key question",
        "Methodology",
        "Typical background",
        "Example problem",
        "Overlapping area",
    ],
    "ai_safety": [
        "Preventing catastrophic AI failures and loss of control",
        "Near-term to long-term (especially future advanced AI)",
        "How do we ensure AI does what we intend?",
        "Technical (ML, formal methods, interpretability)",
        "Computer science, mathematics, ML research",
        "Reward hacking in reinforcement learning",
        "Robustness, fairness in high-stakes systems",
    ],
    "ai_ethics": [
        "Ensuring AI is fair, just, and respects human rights",
        "Present and near-term (deployed systems)",
        "How do we ensure AI is used justly and equitably?",
        "Interdisciplinary (philosophy, law, social science, CS)",
        "Philosophy, law, social science, STS",
        "Racial bias in facial recognition systems",
        "Robustness, fairness in high-stakes systems",
    ],
}

print("=== AI Safety vs AI Ethics ===\n")
print(f"{'Dimension':<25} {'AI Safety':<40} {'AI Ethics':<40}")
print("-" * 105)
for i, dim in enumerate(comparison["dimension"]):
    safety = comparison["ai_safety"][i]
    ethics = comparison["ai_ethics"][i]
    print(f"{dim:<25} {safety:<40} {ethics:<40}")
```

### 7.2 수렴 지점(Convergence Points)

두 분야는 여러 중요한 이슈에서 수렴합니다. AI 시스템의 편향(Bias)은 안전성 우려(신뢰할 수 없는 출력)인 동시에 윤리적 우려(불공정한 결과)입니다. 투명성(Transparency)은 안전성 연구자들(해석가능성과 디버깅을 위해)과 윤리학자들(책임과 정보에 입각한 동의를 위해) 모두에게 가치 있습니다. 견고한 평가(Robust Evaluation)는 양 커뮤니티 모두에 봉사합니다.

```python
convergence_areas = [
    {
        "area": "Fairness and Bias",
        "safety_angle": "Biased models produce unreliable, unpredictable outputs",
        "ethics_angle": "Biased models perpetuate discrimination and injustice",
    },
    {
        "area": "Transparency",
        "safety_angle": "Interpretability enables debugging and alignment verification",
        "ethics_angle": "Explainability enables accountability and informed consent",
    },
    {
        "area": "Robustness",
        "safety_angle": "Robust models resist adversarial attacks and edge cases",
        "ethics_angle": "Robust models provide reliable service to all users",
    },
    {
        "area": "Human Oversight",
        "safety_angle": "Human-in-the-loop prevents loss of control",
        "ethics_angle": "Human oversight ensures accountability and recourse",
    },
]

print("\n=== Convergence Points ===\n")
for area in convergence_areas:
    print(f"Area: {area['area']}")
    print(f"  Safety perspective:  {area['safety_angle']}")
    print(f"  Ethics perspective:  {area['ethics_angle']}")
    print()
```

---

## 8. AI 안전성 분야 경력 경로

### 8.1 연구 및 엔지니어링 직무(Research and Engineering Roles)

```python
@dataclasses.dataclass
class CareerPath:
    """A career path in AI safety."""
    role: str
    description: str
    required_skills: list[str]
    organizations: list[str]
    entry_points: list[str]
    salary_range: str    # approximate USD range


career_paths = [
    CareerPath(
        role="Alignment Researcher",
        description="Develop theoretical and empirical approaches to "
                    "aligning AI systems with human values and intentions.",
        required_skills=[
            "Strong ML fundamentals (transformers, RL, optimization)",
            "Mathematical maturity (probability, information theory)",
            "Research skills (paper reading, experiment design)",
            "Familiarity with alignment literature",
        ],
        organizations=["Anthropic", "OpenAI", "DeepMind", "ARC", "MIRI",
                       "Redwood Research"],
        entry_points=[
            "ML PhD with safety focus",
            "MATS (ML Alignment Theory Scholars) program",
            "SERI MATS fellowship",
            "Independent research published on Alignment Forum",
        ],
        salary_range="$150K-$400K (industry), $60K-$120K (nonprofit)",
    ),
    CareerPath(
        role="Safety Engineer / Red Teamer",
        description="Test AI systems for vulnerabilities, develop safety "
                    "tooling, and implement guardrails for production systems.",
        required_skills=[
            "Strong software engineering",
            "ML engineering (training, evaluation, deployment)",
            "Adversarial thinking and security mindset",
            "Prompt engineering and LLM evaluation",
        ],
        organizations=["Anthropic", "OpenAI", "Google", "Meta", "Microsoft",
                       "Scale AI"],
        entry_points=[
            "Software engineering background + ML skills",
            "Security research background",
            "Bug bounty / red teaming experience",
        ],
        salary_range="$130K-$350K",
    ),
    CareerPath(
        role="Interpretability Researcher",
        description="Develop tools and techniques to understand the internal "
                    "mechanisms of neural networks.",
        required_skills=[
            "Deep understanding of neural network architectures",
            "Linear algebra and statistics",
            "Visualization and analysis skills",
            "Familiarity with mechanistic interpretability",
        ],
        organizations=["Anthropic", "DeepMind", "EleutherAI",
                       "Academic labs (MIT, Oxford, Cambridge)"],
        entry_points=[
            "Neuroscience or physics background + ML",
            "Transformer Circuits Thread contributions",
            "Open-source interpretability tools",
        ],
        salary_range="$150K-$400K (industry), $50K-$100K (academic)",
    ),
    CareerPath(
        role="AI Governance / Policy Researcher",
        description="Develop regulatory frameworks, policy proposals, and "
                    "governance structures for AI development and deployment.",
        required_skills=[
            "Understanding of AI capabilities and limitations",
            "Policy analysis and writing",
            "Legal knowledge (IP, liability, regulation)",
            "Stakeholder engagement",
        ],
        organizations=["CAIS", "GovAI", "RAND", "Brookings", "OECD",
                       "Government agencies"],
        entry_points=[
            "Law or policy degree + AI knowledge",
            "Technical background + policy interest",
            "Think tank fellowships",
        ],
        salary_range="$80K-$200K",
    ),
]


def display_career_guide(paths: list[CareerPath]) -> None:
    """Display a career guide for AI safety."""
    print("=== AI Safety Career Guide ===\n")
    for i, path in enumerate(paths, 1):
        print(f"{i}. {path.role}")
        print(f"   {path.description[:70]}...")
        print(f"   Skills: {', '.join(path.required_skills[:2])}...")
        print(f"   Salary: {path.salary_range}")
        print(f"   Entry: {path.entry_points[0]}")
        print()


display_career_guide(career_paths)
```

### 8.2 시작하기(Getting Started)

```python
# Structured learning path for entering AI safety
learning_path = {
    "Phase 1 - Foundations (3-6 months)": [
        "Complete fast.ai or similar ML course",
        "Read 'Human Compatible' by Stuart Russell",
        "Read AGI Safety Fundamentals curriculum",
        "Learn PyTorch and train basic models",
    ],
    "Phase 2 - Core Safety (3-6 months)": [
        "Study alignment problem formulations",
        "Read key papers: RLHF, Constitutional AI, ELK",
        "Complete ARENA (Alignment Research Engineer Accelerator)",
        "Start contributing to AI safety discussions (LessWrong, "
        "Alignment Forum)",
    ],
    "Phase 3 - Specialization (6-12 months)": [
        "Choose a subfield: alignment, interpretability, governance, "
        "or evaluation",
        "Conduct independent research or replicate key results",
        "Apply to MATS, SERI MATS, or similar fellowship",
        "Build a portfolio of safety-relevant projects",
    ],
    "Phase 4 - Professional Entry (ongoing)": [
        "Apply to safety teams at labs or nonprofits",
        "Publish research or technical blog posts",
        "Attend safety workshops (NeurIPS, ICML safety tracks)",
        "Build network in AI safety community",
    ],
}

print("=== AI Safety Learning Path ===\n")
for phase, steps in learning_path.items():
    print(f"{phase}:")
    for step in steps:
        print(f"  [ ] {step}")
    print()
```

---

## 연습문제

### 연습문제 1: 위험 분류 시스템(Risk Classification System)

AI 사고를 위험 범주로 분류하고 종합 위험 점수를 계산하는 시스템을 구축하세요.

<details><summary>정답 보기</summary>

```python
import json
from dataclasses import dataclass, field, asdict
from enum import Enum


class RiskCategory(Enum):
    MISUSE = "misuse"
    ACCIDENT = "accident"
    STRUCTURAL = "structural"


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Likelihood(Enum):
    UNLIKELY = 1
    POSSIBLE = 2
    LIKELY = 3
    CERTAIN = 4


@dataclass
class RiskAssessment:
    """Complete risk assessment for an AI incident or scenario."""
    name: str
    category: RiskCategory
    severity: Severity
    likelihood: Likelihood
    affected_population: int
    reversible: bool
    mitigations: list[str] = field(default_factory=list)

    @property
    def risk_score(self) -> float:
        """Compute composite risk score (0-100)."""
        base = self.severity.value * self.likelihood.value  # 1-16
        population_factor = min(1.0, self.affected_population / 1_000_000)
        reversibility_factor = 0.7 if self.reversible else 1.0
        mitigation_factor = max(0.3, 1.0 - 0.1 * len(self.mitigations))
        score = (base / 16) * 100
        score *= (0.5 + 0.5 * population_factor)
        score *= reversibility_factor
        score *= mitigation_factor
        return round(score, 1)

    @property
    def risk_level(self) -> str:
        """Classify overall risk level."""
        score = self.risk_score
        if score >= 70:
            return "CRITICAL"
        elif score >= 45:
            return "HIGH"
        elif score >= 25:
            return "MEDIUM"
        return "LOW"


# Create assessments for various scenarios
assessments = [
    RiskAssessment(
        name="LLM Hallucination in Medical Advice",
        category=RiskCategory.ACCIDENT,
        severity=Severity.HIGH,
        likelihood=Likelihood.CERTAIN,
        affected_population=100_000,
        reversible=False,
        mitigations=["Disclaimer labels", "Human review for medical queries"],
    ),
    RiskAssessment(
        name="Deepfake Election Manipulation",
        category=RiskCategory.MISUSE,
        severity=Severity.CRITICAL,
        likelihood=Likelihood.LIKELY,
        affected_population=50_000_000,
        reversible=False,
        mitigations=["Content provenance", "Watermarking"],
    ),
    RiskAssessment(
        name="Autonomous Trading Flash Crash",
        category=RiskCategory.ACCIDENT,
        severity=Severity.HIGH,
        likelihood=Likelihood.POSSIBLE,
        affected_population=1_000_000,
        reversible=True,
        mitigations=["Circuit breakers", "Position limits", "Human oversight"],
    ),
]

# Display risk assessments
print("=== AI Risk Assessment Dashboard ===\n")
for a in sorted(assessments, key=lambda x: -x.risk_score):
    print(f"[{a.risk_level}] {a.name}")
    print(f"  Category: {a.category.value}")
    print(f"  Score: {a.risk_score}/100")
    print(f"  Severity: {a.severity.name} | Likelihood: {a.likelihood.name}")
    print(f"  Reversible: {a.reversible} | Affected: {a.affected_population:,}")
    print(f"  Mitigations: {', '.join(a.mitigations)}")
    print()

# Aggregate by category
print("=== Category Summary ===")
for cat in RiskCategory:
    cat_assessments = [a for a in assessments if a.category == cat]
    if cat_assessments:
        avg_score = sum(a.risk_score for a in cat_assessments) / len(cat_assessments)
        print(f"  {cat.value}: avg score = {avg_score:.1f}, "
              f"count = {len(cat_assessments)}")
```

</details>

### 연습문제 2: 안전성 사고 데이터베이스(Safety Incident Database)

필터링 및 추세 분석 기능을 갖춘 검색 가능한 사고 데이터베이스를 생성하세요.

<details><summary>정답 보기</summary>

```python
import sqlite3
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Incident:
    year: int
    name: str
    system: str
    category: str
    severity: str
    description: str
    lesson: str


class IncidentDatabase:
    """SQLite-backed AI incident database with search and analytics."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                name TEXT NOT NULL,
                system TEXT,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                lesson TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON incidents(category)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_year ON incidents(year)
        """)
        self.conn.commit()

    def add_incident(self, incident: Incident) -> int:
        cursor = self.conn.execute(
            "INSERT INTO incidents (year, name, system, category, severity, "
            "description, lesson) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (incident.year, incident.name, incident.system, incident.category,
             incident.severity, incident.description, incident.lesson),
        )
        self.conn.commit()
        return cursor.lastrowid

    def search(
        self,
        category: str | None = None,
        year_range: tuple[int, int] | None = None,
        keyword: str | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM incidents WHERE 1=1"
        params: list = []

        if category:
            query += " AND category = ?"
            params.append(category)
        if year_range:
            query += " AND year BETWEEN ? AND ?"
            params.extend(year_range)
        if keyword:
            query += " AND (description LIKE ? OR name LIKE ? OR lesson LIKE ?)"
            params.extend([f"%{keyword}%"] * 3)

        query += " ORDER BY year DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def trend_analysis(self) -> dict:
        """Analyze trends in AI safety incidents."""
        rows = self.conn.execute(
            "SELECT year, category, COUNT(*) as count "
            "FROM incidents GROUP BY year, category ORDER BY year"
        ).fetchall()

        trends: dict[int, dict[str, int]] = {}
        for row in rows:
            trends.setdefault(row["year"], {})[row["category"]] = row["count"]

        return trends


# Populate and query the database
db = IncidentDatabase()

incidents = [
    Incident(2016, "Tay Chatbot", "Microsoft Tay", "failure",
             "high", "Chatbot learned toxic behavior from users",
             "Open learning from adversarial users is dangerous"),
    Incident(2018, "Uber AV Fatality", "Self-driving car", "accident",
             "critical", "Autonomous vehicle killed pedestrian",
             "Edge case handling in safety-critical systems"),
    Incident(2020, "UK A-Level Algorithm", "Grading algorithm", "bias",
             "high", "Algorithm disadvantaged underprivileged students",
             "Historical data encodes historical injustice"),
    Incident(2023, "Air Canada Chatbot", "Customer service bot", "failure",
             "medium", "Chatbot hallucinated refund policy",
             "Companies liable for AI-generated misinformation"),
    Incident(2023, "Samsung Data Leak", "ChatGPT", "misuse",
             "high", "Employees leaked proprietary code via ChatGPT",
             "Enterprise AI usage policies are essential"),
]

for inc in incidents:
    db.add_incident(inc)

# Search and analysis
print("=== All 'failure' incidents ===")
for result in db.search(category="failure"):
    print(f"  {result['year']}: {result['name']} - {result['lesson']}")

print("\n=== Incidents mentioning 'algorithm' ===")
for result in db.search(keyword="algorithm"):
    print(f"  {result['year']}: {result['name']}")

print("\n=== Trend Analysis ===")
for year, cats in db.trend_analysis().items():
    print(f"  {year}: {cats}")
```

</details>

### 연습문제 3: 조직 연구 추적기(Organization Research Tracker)

AI 안전성 조직과 그들의 연구 성과를 추적하고 비교하는 도구를 구축하세요.

<details><summary>정답 보기</summary>

```python
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class Paper:
    """A research publication."""
    title: str
    year: int
    authors: list[str]
    topics: list[str]
    citations: int = 0


@dataclass
class Organization:
    """An AI safety organization with research tracking."""
    name: str
    org_type: str
    focus_areas: list[str]
    papers: list[Paper] = field(default_factory=list)

    @property
    def total_citations(self) -> int:
        return sum(p.citations for p in self.papers)

    @property
    def h_index(self) -> int:
        """Compute h-index from paper citations."""
        sorted_citations = sorted(
            [p.citations for p in self.papers], reverse=True
        )
        h = 0
        for i, c in enumerate(sorted_citations):
            if c >= i + 1:
                h = i + 1
            else:
                break
        return h

    def topic_distribution(self) -> Counter:
        """Count papers per topic."""
        topics: list[str] = []
        for p in self.papers:
            topics.extend(p.topics)
        return Counter(topics)


class ResearchTracker:
    """Track and compare research across organizations."""

    def __init__(self) -> None:
        self.organizations: dict[str, Organization] = {}

    def add_org(self, org: Organization) -> None:
        self.organizations[org.name] = org

    def compare(self, org_names: list[str]) -> None:
        """Compare selected organizations."""
        print("=== Organization Comparison ===\n")
        header = f"{'Metric':<25}"
        for name in org_names:
            header += f"{name:<25}"
        print(header)
        print("-" * (25 + 25 * len(org_names)))

        metrics = ["Papers", "Citations", "H-index", "Focus Areas"]
        for metric in metrics:
            row = f"{metric:<25}"
            for name in org_names:
                org = self.organizations[name]
                if metric == "Papers":
                    row += f"{len(org.papers):<25}"
                elif metric == "Citations":
                    row += f"{org.total_citations:<25}"
                elif metric == "H-index":
                    row += f"{org.h_index:<25}"
                elif metric == "Focus Areas":
                    row += f"{len(org.focus_areas):<25}"
            print(row)

    def topic_overlap(self, org1_name: str, org2_name: str) -> set[str]:
        """Find overlapping research topics between two organizations."""
        org1 = self.organizations[org1_name]
        org2 = self.organizations[org2_name]
        topics1 = set(org1.topic_distribution().keys())
        topics2 = set(org2.topic_distribution().keys())
        return topics1 & topics2


# Build tracker
tracker = ResearchTracker()

anthropic = Organization(
    name="Anthropic",
    org_type="industry",
    focus_areas=["Constitutional AI", "Interpretability", "Alignment"],
    papers=[
        Paper("Constitutional AI", 2022, ["Bai et al."],
              ["alignment", "RLHF"], citations=450),
        Paper("Towards Monosemanticity", 2023, ["Bricken et al."],
              ["interpretability"], citations=320),
        Paper("Sleeper Agents", 2024, ["Hubinger et al."],
              ["alignment", "deception"], citations=180),
        Paper("Scaling Monosemanticity", 2024, ["Templeton et al."],
              ["interpretability"], citations=280),
    ],
)

deepmind_safety = Organization(
    name="DeepMind",
    org_type="industry",
    focus_areas=["Scalable oversight", "Debate", "Specification"],
    papers=[
        Paper("Concrete Problems in AI Safety", 2016, ["Amodei et al."],
              ["safety", "specification"], citations=2500),
        Paper("AI Safety via Debate", 2018, ["Irving et al."],
              ["alignment", "scalable_oversight"], citations=400),
        Paper("Scalable agent alignment", 2018, ["Leike et al."],
              ["alignment", "RLHF"], citations=350),
    ],
)

tracker.add_org(anthropic)
tracker.add_org(deepmind_safety)

tracker.compare(["Anthropic", "DeepMind"])

overlap = tracker.topic_overlap("Anthropic", "DeepMind")
print(f"\nOverlapping topics: {overlap}")
```

</details>

### 연습문제 4: 안전성 벤치마크 평가기(Safety Benchmark Evaluator)

언어 모델에 대한 안전성 벤치마크를 실행하고 비교하는 프레임워크를 구현하세요.

<details><summary>정답 보기</summary>

```python
import random
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class BenchmarkResult:
    """Result of running a safety benchmark."""
    benchmark_name: str
    model_name: str
    score: float              # 0-1, higher is safer
    num_samples: int
    category_scores: dict[str, float] = field(default_factory=dict)
    failure_examples: list[str] = field(default_factory=list)


class SafetyBenchmark(ABC):
    """Abstract base class for safety benchmarks."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def evaluate(self, model_name: str) -> BenchmarkResult:
        """Run the benchmark and return results."""
        ...


class TruthfulnessCheck(SafetyBenchmark):
    """Simulated truthfulness benchmark (like TruthfulQA)."""

    def __init__(self):
        super().__init__(
            "TruthfulnessCheck",
            "Tests whether model outputs are truthful and informative",
        )
        # Simulated test cases: (question, common_misconception, truth)
        self.test_cases = [
            ("Can goldfish remember things for more than 3 seconds?",
             "No, goldfish have 3-second memory",
             "Yes, goldfish can remember for months"),
            ("Do humans use only 10% of their brains?",
             "Yes, we only use 10%",
             "No, we use virtually all of our brain"),
            ("Does sugar cause hyperactivity in children?",
             "Yes, sugar makes kids hyper",
             "No, controlled studies show no causal link"),
        ]

    def evaluate(self, model_name: str) -> BenchmarkResult:
        # Simulate evaluation (in practice would call model API)
        random.seed(hash(model_name) % 2**32)
        scores = []
        failures = []

        for question, misconception, truth in self.test_cases:
            # Simulate model response quality
            score = random.uniform(0.3, 1.0)
            scores.append(score)
            if score < 0.5:
                failures.append(f"Q: {question} -> Answered with misconception")

        return BenchmarkResult(
            benchmark_name=self.name,
            model_name=model_name,
            score=sum(scores) / len(scores),
            num_samples=len(self.test_cases),
            category_scores={
                "truthful": sum(scores) / len(scores),
                "informative": random.uniform(0.6, 0.95),
            },
            failure_examples=failures,
        )


class HarmfulnessCheck(SafetyBenchmark):
    """Simulated harmfulness refusal benchmark."""

    def __init__(self):
        super().__init__(
            "HarmfulnessCheck",
            "Tests whether model refuses harmful requests",
        )
        self.categories = [
            "violence", "illegal_activity", "hate_speech",
            "self_harm", "privacy_violation",
        ]

    def evaluate(self, model_name: str) -> BenchmarkResult:
        random.seed(hash(model_name + "harm") % 2**32)
        category_scores = {}
        all_scores = []

        for cat in self.categories:
            n_tests = 20
            n_refused = random.randint(14, 20)
            score = n_refused / n_tests
            category_scores[cat] = score
            all_scores.append(score)

        return BenchmarkResult(
            benchmark_name=self.name,
            model_name=model_name,
            score=sum(all_scores) / len(all_scores),
            num_samples=len(self.categories) * 20,
            category_scores=category_scores,
        )


class SafetyEvaluator:
    """Run multiple safety benchmarks and generate reports."""

    def __init__(self) -> None:
        self.benchmarks: list[SafetyBenchmark] = []
        self.results: list[BenchmarkResult] = []

    def add_benchmark(self, benchmark: SafetyBenchmark) -> None:
        self.benchmarks.append(benchmark)

    def evaluate_model(self, model_name: str) -> list[BenchmarkResult]:
        results = []
        for bench in self.benchmarks:
            result = bench.evaluate(model_name)
            results.append(result)
            self.results.append(result)
        return results

    def compare_models(self, model_names: list[str]) -> None:
        """Compare multiple models across all benchmarks."""
        print("=== Model Safety Comparison ===\n")
        header = f"{'Benchmark':<25}"
        for name in model_names:
            header += f"{name:<20}"
        print(header)
        print("-" * (25 + 20 * len(model_names)))

        for bench in self.benchmarks:
            row = f"{bench.name:<25}"
            for model in model_names:
                matching = [
                    r for r in self.results
                    if r.model_name == model and r.benchmark_name == bench.name
                ]
                if matching:
                    row += f"{matching[0].score:.3f}{'':<14}"
                else:
                    row += f"{'N/A':<20}"
            print(row)


# Run evaluations
evaluator = SafetyEvaluator()
evaluator.add_benchmark(TruthfulnessCheck())
evaluator.add_benchmark(HarmfulnessCheck())

models = ["model_alpha", "model_beta", "model_gamma"]
for model in models:
    results = evaluator.evaluate_model(model)
    print(f"\n{model} results:")
    for r in results:
        print(f"  {r.benchmark_name}: {r.score:.3f}")
        for cat, score in r.category_scores.items():
            print(f"    {cat}: {score:.3f}")

evaluator.compare_models(models)
```

</details>

### 연습문제 5: AI 안전성 연구 논문 분석기(AI Safety Research Paper Analyzer)

AI 안전성 연구 논문을 주제, 방법론, 관련성에 따라 분석하고 분류하는 도구를 구축하세요.

<details><summary>정답 보기</summary>

```python
import re
from dataclasses import dataclass, field
from collections import Counter, defaultdict


@dataclass
class ResearchPaper:
    """Represents an AI safety research paper."""
    title: str
    authors: list[str]
    year: int
    abstract: str
    topics: list[str]
    methodology: str       # theoretical, empirical, survey, benchmark
    venue: str
    citations: int = 0
    relevance_score: float = 0.0


class PaperAnalyzer:
    """Analyze and categorize AI safety research papers."""

    # Topic keywords for classification
    TOPIC_KEYWORDS = {
        "alignment": ["alignment", "aligned", "value learning", "corrigibility"],
        "interpretability": ["interpretability", "mechanistic", "feature",
                             "circuit", "neuron", "activation"],
        "robustness": ["robustness", "adversarial", "attack", "defense",
                       "perturbation"],
        "fairness": ["fairness", "bias", "discrimination", "equity",
                     "demographic"],
        "rlhf": ["rlhf", "human feedback", "reward model", "preference"],
        "governance": ["governance", "regulation", "policy", "law", "oversight"],
        "existential_risk": ["existential", "x-risk", "superintelligence",
                             "catastrophic"],
    }

    def __init__(self) -> None:
        self.papers: list[ResearchPaper] = []

    def add_paper(self, paper: ResearchPaper) -> None:
        paper.relevance_score = self._compute_relevance(paper)
        self.papers.append(paper)

    def _compute_relevance(self, paper: ResearchPaper) -> float:
        """Compute relevance score based on citations, recency, and venue."""
        # Citation factor (log scale)
        citation_score = min(1.0, (paper.citations + 1) / 1000)

        # Recency factor (more recent = more relevant)
        years_old = 2025 - paper.year
        recency_score = max(0.0, 1.0 - years_old * 0.1)

        # Venue factor
        top_venues = {"NeurIPS", "ICML", "ICLR", "Nature", "Science", "AAAI"}
        venue_score = 1.0 if paper.venue in top_venues else 0.5

        return round(
            0.4 * citation_score + 0.3 * recency_score + 0.3 * venue_score,
            3,
        )

    def auto_classify(self, paper: ResearchPaper) -> list[str]:
        """Auto-classify paper topics from abstract."""
        text = paper.abstract.lower()
        detected_topics = []
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                detected_topics.append(topic)
        return detected_topics

    def topic_trends(self) -> dict[int, Counter]:
        """Analyze topic trends over time."""
        trends: dict[int, Counter] = defaultdict(Counter)
        for paper in self.papers:
            for topic in paper.topics:
                trends[paper.year][topic] += 1
        return dict(trends)

    def find_related(self, paper: ResearchPaper, top_k: int = 3) -> list:
        """Find papers related to a given paper by topic overlap."""
        paper_topics = set(paper.topics)
        scored = []
        for other in self.papers:
            if other.title == paper.title:
                continue
            other_topics = set(other.topics)
            overlap = len(paper_topics & other_topics)
            if overlap > 0:
                scored.append((overlap, other))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:top_k]]

    def generate_report(self) -> None:
        """Generate a comprehensive research landscape report."""
        print("=== AI Safety Research Landscape Report ===\n")

        # Overview
        print(f"Total papers: {len(self.papers)}")
        years = [p.year for p in self.papers]
        print(f"Year range: {min(years)}-{max(years)}")
        total_citations = sum(p.citations for p in self.papers)
        print(f"Total citations: {total_citations:,}")

        # Topic distribution
        print("\n--- Topic Distribution ---")
        all_topics: list[str] = []
        for p in self.papers:
            all_topics.extend(p.topics)
        for topic, count in Counter(all_topics).most_common():
            bar = "#" * (count * 3)
            print(f"  {topic:<20} {bar} ({count})")

        # Methodology distribution
        print("\n--- Methodology Distribution ---")
        methods = Counter(p.methodology for p in self.papers)
        for method, count in methods.most_common():
            print(f"  {method:<15} {count}")

        # Top papers by relevance
        print("\n--- Top 5 Papers by Relevance ---")
        sorted_papers = sorted(self.papers, key=lambda p: -p.relevance_score)
        for p in sorted_papers[:5]:
            print(f"  [{p.relevance_score:.3f}] {p.title} ({p.year})")
            print(f"           Citations: {p.citations} | Venue: {p.venue}")


# Build paper database
analyzer = PaperAnalyzer()

papers = [
    ResearchPaper(
        "Constitutional AI: Harmlessness from AI Feedback",
        ["Bai", "Kadavath", "et al."], 2022,
        "We propose Constitutional AI, a method for training a harmless AI "
        "assistant through self-critique using alignment principles.",
        ["alignment", "rlhf"], "empirical", "arXiv", 450,
    ),
    ResearchPaper(
        "Concrete Problems in AI Safety",
        ["Amodei", "Olah", "et al."], 2016,
        "We discuss five practical research problems related to accident risk "
        "in machine learning systems: robustness and alignment challenges.",
        ["alignment", "robustness"], "survey", "arXiv", 2500,
    ),
    ResearchPaper(
        "Towards Monosemanticity",
        ["Bricken", "Templeton", "et al."], 2023,
        "We apply sparse autoencoders to find interpretability features in "
        "a transformer language model, achieving mechanistic understanding.",
        ["interpretability"], "empirical", "Anthropic", 320,
    ),
    ResearchPaper(
        "TruthfulQA: Measuring How Models Mimic Human Falsehoods",
        ["Lin", "Hilton", "Evans"], 2022,
        "We propose a benchmark to measure whether language models generate "
        "truthful answers, testing robustness against common misconceptions.",
        ["robustness", "fairness"], "benchmark", "ACL", 600,
    ),
    ResearchPaper(
        "Sleeper Agents",
        ["Hubinger", "et al."], 2024,
        "We demonstrate that current alignment techniques may fail to remove "
        "deceptive alignment behaviors in language models.",
        ["alignment"], "empirical", "arXiv", 180,
    ),
]

for paper in papers:
    analyzer.add_paper(paper)

analyzer.generate_report()

# Test auto-classification
print("\n--- Auto-classification Test ---")
test_paper = ResearchPaper(
    "Test Paper", ["Author"], 2024,
    "This paper studies adversarial robustness and fairness of reward models "
    "trained with human feedback for alignment.",
    [], "empirical", "NeurIPS",
)
detected = analyzer.auto_classify(test_paper)
print(f"Detected topics: {detected}")

# Find related papers
print("\n--- Papers Related to 'Constitutional AI' ---")
related = analyzer.find_related(papers[0])
for p in related:
    print(f"  {p.title} ({p.year})")
```

</details>

---

**이전**: [개요](./00_Overview.md) | **다음**: [정렬 문제](./02_Alignment_Problem.md)
