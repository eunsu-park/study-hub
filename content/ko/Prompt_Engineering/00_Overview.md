# 프롬프트 엔지니어링(Prompt Engineering)

## 토픽 개요

프롬프트 엔지니어링(Prompt Engineering)은 대규모 언어 모델(LLM)에 대한 입력을 설계, 최적화, 평가하여 원하는 출력을 안정적으로 생성하는 학문입니다. LLM이 소프트웨어 개발, 데이터 분석, 콘텐츠 제작, 의사결정 지원의 핵심이 되면서, 이러한 모델과 효과적으로 소통하는 능력은 AI 연구자뿐만 아니라 업무에 LLM을 사용하는 모든 사람에게 핵심 전문 역량이 되었습니다.

이 토픽은 기본적인 "좋은 프롬프트 작성법" 수준을 훨씬 넘어갑니다. 특정 프롬프팅 기법이 작동하는 이론적 기반, 프롬프트 설계 및 최적화를 위한 체계적 방법론, 자동화된 프롬프트 엔지니어링 도구, 평가 프레임워크, 적대적 견고성(Adversarial Robustness), 프로덕션 배포 패턴까지 다룹니다. 목표는 모델과 사용 사례를 넘어 전이 가능한 원칙적 이해를 구축하는 것입니다.

이 토픽은 LLM 개념(토큰화, 온도, 컨텍스트 윈도우)과 기본 API 사용에 대한 친숙함을 전제합니다. NLP_and_LLM 토픽을 먼저 수강하는 것을 권장합니다.

## 학습 경로

```
기초                            추론 & 구조                      응용 분야
─────────────────              ─────────────────                ─────────────────
01 프롬프트 기초         ★      05 구조화된 출력       ★★       08 멀티모달            ★★★
02 제로/퓨샷            ★★     06 시스템 프롬프트     ★★       09 코드 생성            ★★★
03 사고의 연쇄          ★★     07 다중 턴             ★★★      10 RAG 패턴             ★★★
04 고급 추론            ★★★                                     14 도메인 특화          ★★

최적화 & 안전                   프로덕션                          프로젝트
─────────────────              ─────────────────                ─────────────────
11 최적화               ★★★    15 프로덕션 관리       ★★★      17 캡스톤               ★★★★
12 평가                 ★★★    16 에이전트 프롬프팅   ★★★★
13 적대적               ★★★
```

## 레슨 목록

| # | 레슨 | 난이도 | 핵심 개념 |
|---|------|--------|----------|
| 01 | [프롬프트 기초](./01_Prompt_Fundamentals.md) | ⭐ | 프롬프트 구조, 역할/과제/형식/맥락, 멘탈 모델 |
| 02 | [제로샷과 퓨샷](./02_Zero_Shot_and_Few_Shot.md) | ⭐⭐ | 예시 선택, 순서 효과, 동적 퓨샷(Few-shot) |
| 03 | [사고의 연쇄](./03_Chain_of_Thought.md) | ⭐⭐ | CoT, 제로샷(Zero-shot) CoT, 자동(Auto) CoT, CoT가 도움이 되는/해가 되는 경우 |
| 04 | [고급 추론 프롬프트](./04_Advanced_Reasoning_Prompts.md) | ⭐⭐⭐ | 사고의 나무(Tree-of-Thought), 자기 일관성(Self-Consistency), 사고의 그래프(Graph-of-Thought), 메타 프롬프팅(Meta-prompting) |
| 05 | [구조화된 출력 프롬프팅](./05_Structured_Output_Prompting.md) | ⭐⭐ | JSON/XML 출력, 스키마 제약 생성, 문법 기반 디코딩 |
| 06 | [시스템 프롬프트 설계](./06_System_Prompt_Design.md) | ⭐⭐ | 페르소나 설계, 명령어 계층, 행동 가드레일 |
| 07 | [다중 턴 대화](./07_Multi_Turn_Conversation.md) | ⭐⭐⭐ | 컨텍스트 관리, 메모리 주입, 대화 조향 |
| 08 | [멀티모달 프롬프팅](./08_Multimodal_Prompting.md) | ⭐⭐⭐ | 비전 프롬프트, 이미지+텍스트 추론, 문서 이해 |
| 09 | [코드 생성 프롬프팅](./09_Code_Generation_Prompting.md) | ⭐⭐⭐ | 코딩 프롬프트, 테스트 주도 프롬프팅, 디버깅 프롬프트 |
| 10 | [RAG 프롬프트 패턴](./10_RAG_Prompt_Patterns.md) | ⭐⭐⭐ | 검색 증강 프롬프팅(Retrieval-augmented Prompting), 인용, 그라운딩, 충실도 |
| 11 | [프롬프트 최적화](./11_Prompt_Optimization.md) | ⭐⭐⭐ | DSPy, OPRO, 자동 프롬프트 튜닝, 그래디언트 프리 최적화 |
| 12 | [평가와 메트릭](./12_Evaluation_and_Metrics.md) | ⭐⭐⭐ | 프롬프트 품질 메트릭, A/B 테스팅, 회귀 테스팅, 벤치마크 |
| 13 | [적대적 프롬프팅](./13_Adversarial_Prompting.md) | ⭐⭐⭐ | 탈옥(Jailbreak), 프롬프트 인젝션, 방어적 설계, 입력 정제(Sanitization) |
| 14 | [도메인 특화 프롬프팅](./14_Domain_Specific_Prompting.md) | ⭐⭐ | 데이터 추출, 분석, 요약, 번역, 교육 |
| 15 | [프로덕션 프롬프트 관리](./15_Prompt_Management_in_Production.md) | ⭐⭐⭐ | 버전 관리, 템플릿화, 프롬프트 레지스트리, 프롬프트를 위한 CI/CD |
| 16 | [에이전트 프롬프팅 패턴](./16_Agent_Prompting_Patterns.md) | ⭐⭐⭐⭐ | 도구 사용 프롬프트, 계획 프롬프트, 반성(Reflection), 오케스트레이션 |
| 17 | [캡스톤: 프롬프트 라이브러리](./17_Capstone_Prompt_Library.md) | ⭐⭐⭐⭐ | 평가 모음을 갖춘 재사용 가능한 프롬프트 라이브러리 구축 |

## 선수 과목

- LLM 개념(토큰화, 온도, 컨텍스트 윈도우)에 대한 친숙함
- 기본 Python 프로그래밍 및 API 사용
- 권장: [NLP와 LLM](../NLP_and_LLM/00_Overview.md) 토픽

## 환경 설정

```bash
# Python environment
python -m venv prompt-eng
source prompt-eng/bin/activate

# Core libraries
pip install anthropic openai tiktoken

# For optimization lessons (Lesson 11)
pip install dspy-ai

# For evaluation lessons (Lesson 12)
pip install promptfoo  # or install via npm: npm install -g promptfoo

# Verify
python -c "import anthropic; print('Ready')"
```

## 추천 자료

- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — 공식 Claude 프롬프팅 가이드
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — GPT 중심 전략
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — 커뮤니티 관리 종합 가이드
- [DSPy Documentation](https://dspy-docs.vercel.app/) — 프로그래밍 방식의 프롬프트 최적화
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023)
