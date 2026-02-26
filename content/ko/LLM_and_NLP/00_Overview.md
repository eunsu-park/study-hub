# LLM & NLP 학습 가이드

## 소개

이 폴더는 자연어 처리(NLP)와 대규모 언어 모델(LLM)을 학습하기 위한 자료입니다. 기초 NLP부터 최신 LLM 활용까지 단계별로 구성했습니다.

**대상 독자**: Deep_Learning 폴더를 완료한 학습자 (Transformer, Attention 이해 필수)

---

## 학습 로드맵

```
[NLP 기초]                [사전학습 모델]              [LLM 활용]
    │                          │                          │
    ▼                          ▼                          ▼
토큰화/임베딩 ────────▶ BERT 이해 ─────────▶ 프롬프트 엔지니어링
    │                          │                          │
    ▼                          ▼                          ▼
Word2Vec/GloVe ────────▶ GPT 이해 ─────────▶ RAG 시스템
    │                          │                          │
    ▼                          ▼                          ▼
Transformer 복습 ──────▶ HuggingFace ──────▶ LangChain
                               │                          │
                               ▼                          ▼
                          파인튜닝 ─────────▶ 실전 챗봇
```

---

## 파일 목록

### NLP 기초

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [NLP 기초](./01_NLP_Basics.md) | ⭐⭐ | 토큰화, 정규화, 어휘 구축 |
| [Word2Vec과 GloVe](./02_Word2Vec_GloVe.md) | ⭐⭐ | 단어 임베딩, Skip-gram, CBOW |
| [Transformer 복습](./03_Transformer_Review.md) | ⭐⭐⭐ | Attention, Encoder-Decoder |

### 사전학습 모델

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [BERT 이해](./04_BERT_Understanding.md) | ⭐⭐⭐ | MLM, NSP, 양방향 인코더 |
| [GPT 이해](./05_GPT_Understanding.md) | ⭐⭐⭐ | 자기회귀 모델, 텍스트 생성 |
| [HuggingFace 기초](./06_HuggingFace_Basics.md) | ⭐⭐ | Transformers 라이브러리, Pipeline |
| [파인튜닝](./07_Fine_Tuning.md) | ⭐⭐⭐ | 분류, QA, 요약 파인튜닝 |

### LLM 활용

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [프롬프트 엔지니어링](./08_Prompt_Engineering.md) | ⭐⭐ | 프롬프트 설계, Few-shot, CoT |
| [RAG 기초](./09_RAG_Basics.md) | ⭐⭐⭐ | 검색 증강 생성, 청킹 전략 |
| [LangChain 기초](./10_LangChain_Basics.md) | ⭐⭐⭐ | 체인, 에이전트, 메모리 |
| [벡터 데이터베이스](./11_Vector_Databases.md) | ⭐⭐⭐ | Chroma, Pinecone, FAISS |
| [실전 챗봇 프로젝트](./12_Practical_Chatbot.md) | ⭐⭐⭐⭐ | 대화형 AI 시스템 구축 |

### LLM 심화

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [모델 양자화 (Model Quantization)](./13_Model_Quantization.md) | ⭐⭐⭐ | INT8/INT4, GPTQ, AWQ, bitsandbytes, QLoRA |
| [RLHF와 LLM 정렬 (Alignment)](./14_RLHF_Alignment.md) | ⭐⭐⭐⭐ | PPO, Reward Model, DPO, Constitutional AI |
| [LLM 에이전트 (LLM Agents)](./15_LLM_Agents.md) | ⭐⭐⭐⭐ | ReAct, Tool Use, AutoGPT, LangChain Agent |
| [LLM 평가 지표 (Evaluation Metrics)](./16_Evaluation_Metrics.md) | ⭐⭐⭐ | BLEU, ROUGE, BERTScore, Human Eval, Benchmarks |

---

## 핵심 개념 미리보기

### NLP 파이프라인

```python
# 기본 NLP 파이프라인
텍스트 → 토큰화 → 임베딩 → 모델 → 출력

# HuggingFace 파이프라인
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
```

### BERT vs GPT

| 항목 | BERT | GPT |
|------|------|-----|
| 방향 | 양방향 (인코더) | 단방향 (디코더) |
| 학습 | MLM + NSP | 다음 토큰 예측 |
| 용도 | 분류, QA, NER | 생성, 대화 |
| 특징 | 문맥 이해 | 텍스트 생성 |

### RAG 시스템

```
질문 → 검색 (벡터 DB) → 관련 문서 → LLM + 문서 → 답변
```

---

## 선수 지식

- Deep_Learning 폴더 (필수)
  - Attention 메커니즘
  - Transformer 아키텍처
  - 텍스트 분류 기초
- Python 고급
- PyTorch 기본

---

## 환경 설정

### 필수 패키지

```bash
# PyTorch
pip install torch torchvision torchaudio

# HuggingFace
pip install transformers datasets tokenizers accelerate

# LangChain
pip install langchain langchain-community langchain-openai

# 벡터 데이터베이스
pip install chromadb faiss-cpu sentence-transformers

# 기타
pip install openai tiktoken numpy pandas
```

### API 키 설정

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# HuggingFace (모델 다운로드용)
export HUGGINGFACE_TOKEN="your-token"
```

---

## 추천 학습 순서

1. **NLP 기초 (3일)**: 01 → 02 → 03
   - 토큰화, 임베딩 개념 확실히 이해
2. **사전학습 모델 (5일)**: 04 → 05 → 06 → 07
   - HuggingFace 실습 중심
3. **LLM 활용 (7일)**: 08 → 09 → 10 → 11 → 12
   - 프로젝트 기반 학습
4. **LLM 심화 (5일)**: 13 → 14 → 15 → 16
   - 양자화, RLHF, Agent, 평가 지표

---

## 관련 자료

- [Deep_Learning/](../Deep_Learning/00_Overview.md) - 선수 과목 (Transformer)
- [Python/](../Python/00_Overview.md) - 고급 Python
- [Data_Analysis/](../Data_Analysis/00_Overview.md) - 데이터 처리

---

## 참고 링크

- [HuggingFace Documentation](https://huggingface.co/docs)
- [LangChain Documentation](https://python.langchain.com/docs)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
