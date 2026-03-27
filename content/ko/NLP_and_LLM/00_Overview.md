# NLP & LLM 학습 가이드

## 소개

이 폴더는 자연어 처리(NLP) 기초부터 실용적인 대규모 언어 모델(LLM) 활용까지 다룹니다. 고전 NLP에서 시작하여 최신 LLM 기법과 프로덕션급 애플리케이션 패턴까지 단계적으로 학습합니다.

**대상 독자**: Deep_Learning 폴더를 완료한 학습자 (Transformer와 Attention 이해 필수)

---

## 학습 로드맵

```
[NLP 기초]                 [사전학습 모델]              [LLM 응용]
    │                          │                          │
    ▼                          ▼                          ▼
토큰화/임베딩 ────────▶ BERT 이해 ──────────▶ 프롬프트 엔지니어링
    │                          │                          │
    ▼                          ▼                          ▼
Word2Vec/GloVe ───────▶ GPT 이해 ───────────▶ RAG 시스템
    │                          │                          │
    ▼                          ▼                          ▼
Transformer 복습 ─────▶ HuggingFace ────────▶ 에이전트 & 도구
                               │                          │
                               ▼                          ▼
                          파인튜닝/PEFT ─────▶ 프로덕션 패턴
```

---

## 파일 목록

### 섹션 1: NLP 기초 (01-03)

| 파일 | 난이도 | 핵심 주제 |
|------|--------|-----------|
| [01_NLP_Basics.md](./01_NLP_Basics.md) | ⭐⭐ | 토큰화, 정규화, 어휘 구축 |
| [02_Word2Vec_GloVe.md](./02_Word2Vec_GloVe.md) | ⭐⭐ | 단어 임베딩, Skip-gram, CBOW |
| [03_Transformer_Review.md](./03_Transformer_Review.md) | ⭐⭐⭐ | 어텐션, 인코더-디코더 |

### 섹션 2: 사전학습 모델 (04-08)

| 파일 | 난이도 | 핵심 주제 |
|------|--------|-----------|
| [04_BERT_Understanding.md](./04_BERT_Understanding.md) | ⭐⭐⭐ | MLM, NSP, 양방향 인코더 |
| [05_GPT_Understanding.md](./05_GPT_Understanding.md) | ⭐⭐⭐ | 자기회귀 모델, 텍스트 생성 |
| [06_HuggingFace_Basics.md](./06_HuggingFace_Basics.md) | ⭐⭐ | Transformers 라이브러리, Pipeline |
| [07_Fine_Tuning.md](./07_Fine_Tuning.md) | ⭐⭐⭐ | 분류, QA, 요약 파인튜닝 |
| [08_PEFT_and_QLoRA.md](./08_PEFT_and_QLoRA.md) | ⭐⭐⭐ | LoRA, QLoRA, DoRA, Adapters, IA3 |

### 섹션 3: 검색 & RAG (09-12)

| 파일 | 난이도 | 핵심 주제 |
|------|--------|-----------|
| [09_Prompt_Engineering.md](./09_Prompt_Engineering.md) | ⭐⭐ | 프롬프트 설계, Few-shot, CoT |
| [10_RAG_Fundamentals.md](./10_RAG_Fundamentals.md) | ⭐⭐⭐ | 검색 증강 생성, 청킹 전략 |
| [11_Vector_Search_for_RAG.md](./11_Vector_Search_for_RAG.md) | ⭐⭐⭐ | Chroma, Pinecone, FAISS, 임베딩 모델 |
| [12_Advanced_RAG.md](./12_Advanced_RAG.md) | ⭐⭐⭐⭐ | Agentic RAG, HyDE, RAPTOR, ColBERT |

### 섹션 4: 에이전트 & 오케스트레이션 (13-16)

| 파일 | 난이도 | 핵심 주제 |
|------|--------|-----------|
| [13_LangChain_Basics.md](./13_LangChain_Basics.md) | ⭐⭐⭐ | Chains, Agents, Memory |
| [14_LLM_Agents.md](./14_LLM_Agents.md) | ⭐⭐⭐⭐ | ReAct, Tool Use, LangChain Agent |
| [15_Multi_Agent_Systems.md](./15_Multi_Agent_Systems.md) | ⭐⭐⭐⭐ | CrewAI, AutoGen, LangGraph, 에이전트 오케스트레이션 |
| [16_Practical_Chatbot.md](./16_Practical_Chatbot.md) | ⭐⭐⭐⭐ | 대화형 AI 시스템 구축 |

### 섹션 5: 최적화 & 정렬 (17-20)

| 파일 | 난이도 | 핵심 주제 |
|------|--------|-----------|
| [17_Model_Quantization.md](./17_Model_Quantization.md) | ⭐⭐⭐ | INT8/INT4, GPTQ, AWQ, bitsandbytes |
| [18_Inference_Optimization.md](./18_Inference_Optimization.md) | ⭐⭐⭐ | vLLM, TGI, Speculative Decoding, PagedAttention |
| [19_RLHF_Alignment.md](./19_RLHF_Alignment.md) | ⭐⭐⭐⭐ | PPO, Reward Model, DPO, Constitutional AI |
| [20_Evaluation_Metrics.md](./20_Evaluation_Metrics.md) | ⭐⭐⭐ | BLEU, ROUGE, BERTScore, Human Eval, 벤치마크 |

### 섹션 6: 프로덕션 LLM 엔지니어링 (21-24)

| 파일 | 난이도 | 핵심 주제 |
|------|--------|-----------|
| [21_LLM_Security_Safety.md](./21_LLM_Security_Safety.md) | ⭐⭐⭐ | 프롬프트 인젝션, 가드레일, 레드 팀, PII 삭제 |
| [22_Structured_Output.md](./22_Structured_Output.md) | ⭐⭐⭐ | JSON 모드, Pydantic 파싱, instructor, 스키마 설계 |
| [23_Function_Calling_Tools.md](./23_Function_Calling_Tools.md) | ⭐⭐⭐⭐ | OpenAI/Anthropic 도구 사용, MCP, 도구 오케스트레이션 |
| [24_Production_LLM_Patterns.md](./24_Production_LLM_Patterns.md) | ⭐⭐⭐⭐ | 캐싱, 비용 최적화, 옵저버빌리티, 배포 |

### 섹션 7: 고급 에이전트 엔지니어링 (25-27)

| 파일 | 난이도 | 핵심 주제 |
|------|--------|-----------|
| [25_Agent_Memory_and_Planning.md](./25_Agent_Memory_and_Planning.md) | ⭐⭐⭐⭐ | 메모리 아키텍처, 계획 프레임워크, 태스크 분해 |
| [26_Agent_Evaluation_and_Benchmarks.md](./26_Agent_Evaluation_and_Benchmarks.md) | ⭐⭐⭐⭐ | AgentBench, SWE-bench, 실패 분석, 커스텀 평가 |
| [27_Agent_Design_Patterns.md](./27_Agent_Design_Patterns.md) | ⭐⭐⭐⭐ | 오케스트레이터-워커, 라우터, HITL, 가드레일 에이전트 |

---

## 선수 과목

- Deep_Learning 폴더 (필수)
  - 어텐션 메커니즘
  - Transformer 아키텍처
- Python 고급
- PyTorch 기초

---

## 환경 설정

### 필수 패키지

```bash
# PyTorch
pip install torch torchvision torchaudio

# HuggingFace
pip install transformers datasets tokenizers accelerate

# PEFT
pip install peft bitsandbytes

# LangChain
pip install langchain langchain-community langchain-openai langgraph

# 벡터 데이터베이스
pip install chromadb faiss-cpu sentence-transformers

# 에이전트
pip install crewai autogen

# 프로덕션
pip install instructor guardrails-ai vllm

# 기타
pip install openai anthropic tiktoken numpy pandas
```

### API 키 설정

```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
export HUGGINGFACE_TOKEN="your-token"
```

---

## 추천 학습 순서

1. **NLP 기초 (3일)**: 01 → 02 → 03
2. **사전학습 모델 (5일)**: 04 → 05 → 06 → 07 → 08
3. **RAG 시스템 (5일)**: 09 → 10 → 11 → 12
4. **에이전트 (5일)**: 13 → 14 → 15 → 16
5. **최적화 (4일)**: 17 → 18 → 19 → 20
6. **프로덕션 (4일)**: 21 → 22 → 23 → 24
7. **고급 에이전트 (3일)**: 25 → 26 → 27

---

## 관련 자료

- [Deep_Learning/](../Deep_Learning/00_Overview.md) - 선수 과목 (Transformer)
- [Foundation_Models/](../Foundation_Models/00_Overview.md) - FM 아키텍처 이론 및 연구
- [Python/](../Python/00_Overview.md) - Python 고급

---

## 참고 링크

- [HuggingFace 문서](https://huggingface.co/docs)
- [LangChain 문서](https://python.langchain.com/docs)
- [OpenAI API 레퍼런스](https://platform.openai.com/docs)
- [Anthropic API 레퍼런스](https://docs.anthropic.com)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
