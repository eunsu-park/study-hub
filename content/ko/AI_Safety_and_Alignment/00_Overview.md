# AI 안전성과 정렬(AI Safety and Alignment)

## 토픽 개요

AI 안전성과 정렬(AI Safety and Alignment)은 고급 AI 시스템이 유익하고, 예측 가능하며, 인간의 가치와 의도에 부합하는 방식으로 동작하도록 보장하는 학제간 분야입니다. AI 역량이 급속히 성장함에 따라 — 특히 대규모 언어 모델(Large Language Model)과 자율 에이전트(Autonomous Agent) 분야에서 — AI 시스템이 *할 수 있는* 것과 우리가 *보장할 수 있는* 것 사이의 간극이 기술 분야에서 가장 시급한 과제 중 하나가 되었습니다.

이 토픽은 AI 안전성의 전체 스펙트럼을 다룹니다: 단기적 실용 문제(프롬프트 인젝션(Prompt Injection), 편향(Bias), 환각(Hallucination))부터 장기적 정렬 과제(기만적 정렬(Deceptive Alignment), 메사 최적화(Mesa-Optimization), 확장 가능한 감독(Scalable Oversight))까지 포괄합니다. 기술적 방법론(RLHF, 헌법적 AI(Constitutional AI), DPO, 레드 티밍(Red Teaming), 표현 공학(Representation Engineering)), 평가 프레임워크(안전성 벤치마크(Safety Benchmark), 자동화 감사(Automated Auditing)), 거버넌스 구조(조직 안전성, 프론티어 모델 정책), 사회적 영향을 아우릅니다.

이 토픽은 딥러닝(Deep Learning), NLP/LLM 개념, 강화학습(Reinforcement Learning) 기초에 대한 사전 지식을 전제합니다. 이러한 이론적 기반 위에서 다음 질문을 다룹니다: *강력한 AI 시스템을 어떻게 안전하고 신뢰할 수 있게 만들 수 있는가?*

## 학습 경로

```
기초                            정렬 방법론                       평가 및 방어
─────────────────              ─────────────────                ─────────────────
01 안전성 개관          ★★      03 정렬을 위한 RLHF    ★★★      07 레드 티밍          ★★★
02 정렬 문제           ★★★     04 헌법적 AI           ★★★      08 안전성 평가         ★★★
                                05 DPO 방법론          ★★★      09 강건성              ★★★
                                06 확장 가능한 감독    ★★★★     10 표현 공학           ★★★★

배포 및 방어                    거버넌스 및 사회                  프로젝트
─────────────────              ─────────────────                ─────────────────
11 가드레일            ★★★     13 거버넌스            ★★★      17 캡스톤              ★★★★
12 기만적 정렬         ★★★★    14 책임 있는 배포      ★★★
                                15 사회적 영향         ★★
                                16 미해결 문제         ★★★★
```

## 레슨 목록

| # | 레슨 | 난이도 | 핵심 개념 |
|---|--------|------------|--------------|
| 01 | [AI 안전성 개관](./01_AI_Safety_Landscape.md) | ⭐⭐ | 위험 분류법, 단기 vs 장기, 주요 조직 |
| 02 | [정렬 문제](./02_Alignment_Problem.md) | ⭐⭐⭐ | 내부/외부 정렬, 굿하트의 법칙(Goodhart's Law), 보상 해킹(Reward Hacking) |
| 03 | [정렬을 위한 RLHF](./03_RLHF_for_Alignment.md) | ⭐⭐⭐ | RLHF 파이프라인, 보상 모델링, 정렬을 위한 PPO |
| 04 | [헌법적 AI](./04_Constitutional_AI.md) | ⭐⭐⭐ | RLAIF, 원칙 계층 구조, 자기 비평(Self-Critique) |
| 05 | [직접 선호도 최적화](./05_Direct_Preference_Optimization.md) | ⭐⭐⭐ | DPO, KTO, IPO — 보상 없는 정렬 방법론 |
| 06 | [확장 가능한 감독](./06_Scalable_Oversight.md) | ⭐⭐⭐⭐ | 디베이트(Debate), 재귀적 보상 모델링, 약한-강한 일반화(Weak-to-Strong Generalization) |
| 07 | [레드 티밍](./07_Red_Teaming.md) | ⭐⭐⭐ | 체계적 레드 티밍, 자동화 레드 티밍, 평가 |
| 08 | [안전성 평가](./08_Safety_Evaluation.md) | ⭐⭐⭐ | TruthfulQA, BBQ, 독성 벤치마크, 커스텀 평가 하네스 |
| 09 | [강건성과 적대적 공격](./09_Robustness_and_Adversarial.md) | ⭐⭐⭐ | LLM에 대한 적대적 공격, 강건성 학습, 입력 필터링 |
| 10 | [표현 공학](./10_Representation_Engineering.md) | ⭐⭐⭐⭐ | 활성화 조향(Activation Steering), 표현 읽기, 안전성 프로브 |
| 11 | [가드레일과 필터](./11_Guardrails_and_Filters.md) | ⭐⭐⭐ | NeMo Guardrails, Guardrails AI, 입출력 필터링 |
| 12 | [기만적 정렬](./12_Deceptive_Alignment.md) | ⭐⭐⭐⭐ | 메사 최적화(Mesa-Optimization), 기만, 목표 오일반화(Goal Misgeneralization) |
| 13 | [거버넌스 프레임워크](./13_Governance_Frameworks.md) | ⭐⭐⭐ | 조직 안전성, AI 정책, 프론티어 모델 거버넌스 |
| 14 | [책임 있는 배포](./14_Responsible_Deployment.md) | ⭐⭐⭐ | 단계적 릴리스, 모니터링, AI 사고 대응 |
| 15 | [사회적 영향](./15_Societal_Impact.md) | ⭐⭐ | 노동 대체, 이중 용도, 권력 집중 |
| 16 | [미해결 문제](./16_Open_Problems.md) | ⭐⭐⭐⭐ | 현재 연구 최전선, 미해결 정렬 문제 |
| 17 | [캡스톤: 안전성 감사](./17_Capstone_Safety_Audit.md) | ⭐⭐⭐⭐ | 모델/애플리케이션에 대한 전체 안전성 감사 수행 |

## 선수 과목

- 딥러닝 기초 (신경망, 학습, 최적화)
- LLM 개념 (트랜스포머, 토큰화, 파인튜닝, RLHF 기초)
- 강화학습 기초 (보상, 정책, 가치 함수)
- 추천: [NLP and LLM](../NLP_and_LLM/00_Overview.md), [Reinforcement Learning](../Reinforcement_Learning/00_Overview.md), [Interpretable AI](../Interpretable_AI/00_Overview.md)

## 환경 설정

```bash
# Python environment
python -m venv ai-safety
source ai-safety/bin/activate

# Core libraries
pip install torch transformers datasets accelerate
pip install anthropic openai

# Safety-specific tools
pip install nemoguardrails guardrails-ai
pip install evaluate  # HuggingFace evaluation library

# For representation engineering (Lesson 10)
pip install baukit  # activation analysis

# Verify
python -c "import torch; import transformers; print('Ready')"
```

## 추천 자료

- [Anthropic Research](https://www.anthropic.com/research) — 헌법적 AI(Constitutional AI), 해석가능성(Interpretability), 정렬 연구
- [Alignment Forum](https://www.alignmentforum.org/) — 기술적 AI 정렬 토론
- Ngo et al., "The Alignment Problem from a Deep Learning Perspective" (2023)
- Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (2022)
- Rafailov et al., "Direct Preference Optimization" (2023)
- Burns et al., "Weak-to-Strong Generalization" (2023)
- Zou et al., "Representation Engineering" (2023)
