# 해석 가능한 AI(Interpretable AI)

해석 가능한 AI(Interpretable AI)는 머신러닝 모델의 의사결정을 이해하고, 설명하며, 감사하는 방법을 연구합니다. AI 시스템이 채용, 대출, 의료, 형사 사법에 점점 더 많은 영향을 미치면서, 모델이 *왜* 특정 예측을 내렸는지 설명할 수 있는 능력은 더 이상 선택 사항이 아닙니다 — 이는 법적, 윤리적, 공학적 요구사항입니다. 이 토픽은 머신러닝 Lesson 16에서 소개된 기초를 넘어 그래디언트 기반 어트리뷰션(Gradient-based Attribution), 개념 기반 설명(Concept-based Explanations), 인과 추론 기반 설명 가능성(Causal Inference for Explainability), 고급 알고리즘 공정성(Advanced Algorithmic Fairness), AI 규제, 그리고 신흥 분야인 메커니즘 해석 가능성(Mechanistic Interpretability) 등 고급 해석 가능성 기법을 다룹니다.

## 대상 독자

- 모델 예측을 이해관계자에게 설명해야 하는 ML 엔지니어
- 규제 산업(금융, 의료, 보험)에서 일하는 데이터 과학자
- 모델 이해와 안전에 관심 있는 AI 연구자
- 프로덕션 설명 시스템을 구축하는 소프트웨어 엔지니어
- AI 거버넌스 프레임워크를 평가하는 정책 전문가

## 선수 과목

- **[Machine_Learning](../Machine_Learning/00_Overview.md)**: 특히 Lesson 16 (모델 설명 가능성) — SHAP, LIME, PDP/ICE 기초
- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: 신경망 아키텍처, 역전파(Backpropagation), CNN, 트랜스포머(Transformer)
- **[Python](../Python/00_Overview.md)**: PyTorch, NumPy, scikit-learn 활용에 익숙할 것

## 학습 로드맵

```
Section 1: 기초
  L01 해석 가능성 기초 ──► L02 그래디언트 어트리뷰션
                                           │
Section 2: 딥러닝 설명                      ▼
  L03 클래스 활성화 매핑 ──► L04 어텐션 해석 ──► L05 프로빙 & 표현 분석
                                                                          │
Section 3: 고급 기법                                                       ▼
  L06 고급 SHAP ──► L07 개념 기반 설명 ──► L08 반사실적 설명
                                                                  │
Section 4: 인과 & 이론                                             ▼
  L09 해석 가능성을 위한 인과 추론 ──► L10 설명 평가
                                                       │
Section 5: 공정성                                       ▼
  L11 고급 알고리즘 공정성 ──► L12 공정성 완화
                                              │
Section 6: 규제 & 프로덕션                    ▼
  L13 AI 규제 & 거버넌스 ──► L14 프로덕션 해석 가능성
                                           │
Section 7: 도메인 & 프론티어                ▼
  L15 도메인 특화 해석 가능성 ──► L16 메커니즘 해석 가능성
```

## 파일 목록

| 레슨 | 파일 | 난이도 | 설명 |
|--------|------|-----------|-------------|
| L01 | [01_Interpretability_Foundations.md](./01_Interpretability_Foundations.md) | ⭐⭐ | Lipton의 분류법, 설명 요구사항, 연구 동향 |
| L02 | [02_Gradient_Attribution.md](./02_Gradient_Attribution.md) | ⭐⭐⭐ | 현저성 맵(Saliency Maps), 적분 그래디언트(Integrated Gradients), SmoothGrad, 정상성 검사 |
| L03 | [03_Class_Activation_Mapping.md](./03_Class_Activation_Mapping.md) | ⭐⭐⭐ | CAM, GradCAM, GradCAM++, Score-CAM, Eigen-CAM |
| L04 | [04_Attention_Interpretation.md](./04_Attention_Interpretation.md) | ⭐⭐⭐⭐ | BertViz, 어텐션 롤아웃(Attention Rollout), "어텐션은 설명이 아니다(Attention is not Explanation)" 논쟁 |
| L05 | [05_Probing_and_Representation_Analysis.md](./05_Probing_and_Representation_Analysis.md) | ⭐⭐⭐⭐ | 프로빙 분류기(Probing Classifiers), 네트워크 해부(Network Dissection), 로짓 렌즈(Logit Lens), CKA |
| L06 | [06_Advanced_SHAP.md](./06_Advanced_SHAP.md) | ⭐⭐⭐ | DeepSHAP, SHAP 상호작용, 인과 SHAP(Causal SHAP), 최적화 |
| L07 | [07_Concept_Based_Explanations.md](./07_Concept_Based_Explanations.md) | ⭐⭐⭐⭐ | TCAV, 개념 병목 모델(Concept Bottleneck Models), ACE |
| L08 | [08_Counterfactual_Explanations.md](./08_Counterfactual_Explanations.md) | ⭐⭐⭐ | Wachter 공식화, DiCE, 실행 가능성 제약 조건 |
| L09 | [09_Causal_Inference_for_Interpretability.md](./09_Causal_Inference_for_Interpretability.md) | ⭐⭐⭐⭐ | 구조적 인과 모델(SCM), do-연산, 인과적 특성 중요도, DoWhy |
| L10 | [10_Evaluating_Explanations.md](./10_Evaluating_Explanations.md) | ⭐⭐⭐ | 충실성(Faithfulness), 안정성(Stability), ROAR 벤치마크, 인간 평가 |
| L11 | [11_Advanced_Algorithmic_Fairness.md](./11_Advanced_Algorithmic_Fairness.md) | ⭐⭐⭐⭐ | 개인 공정성(Individual Fairness), 반사실적 공정성(Counterfactual Fairness), 불가능성 정리 |
| L12 | [12_Fairness_Mitigation.md](./12_Fairness_Mitigation.md) | ⭐⭐⭐⭐ | 전처리/학습 중/후처리, 파레토 프론티어, Fairlearn, AIF360 |
| L13 | [13_AI_Regulation_and_Governance.md](./13_AI_Regulation_and_Governance.md) | ⭐⭐⭐ | EU AI Act, GDPR Art. 22, NIST AI RMF, 모델 카드 |
| L14 | [14_Production_Interpretability.md](./14_Production_Interpretability.md) | ⭐⭐⭐⭐ | 설명 서빙, 캐싱, 드리프트 모니터링, MLOps 통합 |
| L15 | [15_Domain_Specific_Interpretability.md](./15_Domain_Specific_Interpretability.md) | ⭐⭐⭐ | 의료, 금융, NLP, 컴퓨터 비전 응용 |
| L16 | [16_Mechanistic_Interpretability.md](./16_Mechanistic_Interpretability.md) | ⭐⭐⭐⭐ | 중첩(Superposition), 희소 오토인코더(Sparse Autoencoders), 회로 발견(Circuit Discovery), 활성화 패칭(Activation Patching) |

## 난이도 가이드

- ⭐⭐ ML L16 기초 위에 더 깊은 분류법과 프레임워크를 구축
- ⭐⭐⭐ 능숙한 PyTorch/수학 능력 필요; 구현 중심
- ⭐⭐⭐⭐ 연구 수준의 기법; 강한 수학 및 DL 배경 지식 필요

## 환경 설정

```bash
# Core
pip install torch>=2.0 torchvision transformers>=4.36

# Explainability libraries
pip install shap>=0.43 lime captum

# Visualization
pip install matplotlib seaborn

# Fairness toolkits
pip install fairlearn aif360

# Counterfactual explanations
pip install dice-ml

# Causal inference
pip install dowhy

# Attention visualization
pip install bertviz

# Mechanistic interpretability (L16)
pip install transformer-lens
```

## 관련 토픽

- **[Machine_Learning](../Machine_Learning/00_Overview.md)**: Lesson 16에서 SHAP/LIME/PDP 기초를 다룸 (선수 과목)
- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: 여기서 설명하는 CNN과 트랜스포머 아키텍처
- **[Foundation_Models](../Foundation_Models/00_Overview.md)**: 메커니즘 해석 가능성을 위한 스케일링 및 파인튜닝 맥락
- **[MLOps](../MLOps/00_Overview.md)**: 설명 서빙을 위한 프로덕션 배포 맥락
- **[Probability_and_Statistics](../Probability_and_Statistics/00_Overview.md)**: 평가 및 공정성에 사용되는 통계적 검정

## 학습 팁

1. **먼저 ML L16을 완료하세요** — 이 토픽은 SHAP, LIME, PDP 기초를 이미 이해하고 있다고 가정합니다
2. **코드를 실행하세요** — 해석 가능성은 실제 모델의 설명을 시각화함으로써 가장 잘 이해됩니다
3. **방법들을 비교하세요** — 동일한 예측에 여러 방법을 적용하고 일치/불일치하는 부분을 확인하세요
4. **대상 독자를 생각하세요** — 서로 다른 이해관계자는 서로 다른 유형의 설명이 필요합니다
5. **최신 동향을 따르세요** — 메커니즘 해석 가능성은 빠르게 발전하고 있습니다; 최신 논문을 확인하세요
6. **공정성 수학을 연습하세요** — 불가능성 정리는 증명을 직접 해보기 전까지는 직관에 반합니다

## 학습 성과

이 토픽을 완료하면 다음을 할 수 있습니다:

- PyTorch에서 그래디언트 기반 어트리뷰션 방법(적분 그래디언트, GradCAM)을 처음부터 구현
- 어텐션 가중치가 유효한 설명을 구성하는지 비판적으로 평가
- 표 형식 데이터 및 이미지 데이터에 대한 반사실적 설명 생성 및 평가
- 인과 추론을 적용하여 진정한 특성 효과와 허위 상관관계를 구별
- 다양한 정의를 사용하여 ML 모델의 공정성을 감사하고 완화 전략을 구현
- 예측과 함께 설명을 제공하는 프로덕션 시스템 설계
- AI 규제 프레임워크(EU AI Act, GDPR)를 탐색하고 규정 준수 문서를 작성
- 메커니즘 해석 가능성 기법을 적용하여 신경망 내부를 이해

## 다음 단계

- **[Foundation_Models](../Foundation_Models/00_Overview.md)**: 해석 가능성이 대규모 모델에서 어떻게 확장되는지 탐구
- **[MLOps](../MLOps/00_Overview.md)**: 설명 시스템을 ML 파이프라인에 통합
- **연구**: Anthropic의 메커니즘 해석 가능성 연구, Google DeepMind의 XAI 연구를 따라가세요

---

**라이선스**: CC BY-NC 4.0

[시작: Lesson 01 — 해석 가능성 기초](./01_Interpretability_Foundations.md)
