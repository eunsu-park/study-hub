# Interpretable AI

Interpretable AI studies how to understand, explain, and audit machine learning model decisions. As AI systems increasingly affect hiring, lending, healthcare, and criminal justice, the ability to explain *why* a model made a prediction is no longer optional — it is a legal, ethical, and engineering requirement. This topic covers advanced interpretability methods beyond the fundamentals introduced in Machine Learning Lesson 16, including gradient-based attribution, concept-based explanations, causal inference for explainability, advanced algorithmic fairness, AI regulation, and the emerging field of mechanistic interpretability.

## Target Audience

- ML engineers who need to explain model predictions to stakeholders
- Data scientists working in regulated industries (finance, healthcare, insurance)
- AI researchers interested in model understanding and safety
- Software engineers building production explanation systems
- Policy professionals evaluating AI governance frameworks

## Prerequisites

- **[Machine_Learning](../Machine_Learning/00_Overview.md)**: Especially Lesson 16 (Model Explainability) — SHAP, LIME, PDP/ICE basics
- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: Neural network architectures, backpropagation, CNNs, Transformers
- **[Python](../Python/00_Overview.md)**: Comfortable with PyTorch, NumPy, scikit-learn

## Learning Roadmap

```
Section 1: Foundations
  L01 Interpretability Foundations ──► L02 Gradient Attribution
                                           │
Section 2: Deep Learning Explanations      ▼
  L03 Class Activation Mapping ──► L04 Attention Interpretation ──► L05 Probing & Representations
                                                                          │
Section 3: Advanced Methods                                               ▼
  L06 Advanced SHAP ──► L07 Concept-Based Explanations ──► L08 Counterfactual Explanations
                                                                  │
Section 4: Causal & Theory                                        ▼
  L09 Causal Inference for Interpretability ──► L10 Evaluating Explanations
                                                       │
Section 5: Fairness                                    ▼
  L11 Advanced Algorithmic Fairness ──► L12 Fairness Mitigation
                                              │
Section 6: Regulation & Production            ▼
  L13 AI Regulation & Governance ──► L14 Production Interpretability
                                           │
Section 7: Domain & Frontier                ▼
  L15 Domain-Specific Interpretability ──► L16 Mechanistic Interpretability
```

## File List

| Lesson | File | Difficulty | Description |
|--------|------|-----------|-------------|
| L01 | [01_Interpretability_Foundations.md](./01_Interpretability_Foundations.md) | ⭐⭐ | Lipton's taxonomy, explanation desiderata, research landscape |
| L02 | [02_Gradient_Attribution.md](./02_Gradient_Attribution.md) | ⭐⭐⭐ | Saliency maps, Integrated Gradients, SmoothGrad, sanity checks |
| L03 | [03_Class_Activation_Mapping.md](./03_Class_Activation_Mapping.md) | ⭐⭐⭐ | CAM, GradCAM, GradCAM++, Score-CAM, Eigen-CAM |
| L04 | [04_Attention_Interpretation.md](./04_Attention_Interpretation.md) | ⭐⭐⭐⭐ | BertViz, attention rollout, "Attention is not Explanation" debate |
| L05 | [05_Probing_and_Representation_Analysis.md](./05_Probing_and_Representation_Analysis.md) | ⭐⭐⭐⭐ | Probing classifiers, Network Dissection, logit lens, CKA |
| L06 | [06_Advanced_SHAP.md](./06_Advanced_SHAP.md) | ⭐⭐⭐ | DeepSHAP, SHAP interactions, Causal SHAP, optimization |
| L07 | [07_Concept_Based_Explanations.md](./07_Concept_Based_Explanations.md) | ⭐⭐⭐⭐ | TCAV, Concept Bottleneck Models, ACE |
| L08 | [08_Counterfactual_Explanations.md](./08_Counterfactual_Explanations.md) | ⭐⭐⭐ | Wachter formulation, DiCE, actionability constraints |
| L09 | [09_Causal_Inference_for_Interpretability.md](./09_Causal_Inference_for_Interpretability.md) | ⭐⭐⭐⭐ | SCMs, do-calculus, causal feature importance, DoWhy |
| L10 | [10_Evaluating_Explanations.md](./10_Evaluating_Explanations.md) | ⭐⭐⭐ | Faithfulness, stability, ROAR benchmark, human evaluation |
| L11 | [11_Advanced_Algorithmic_Fairness.md](./11_Advanced_Algorithmic_Fairness.md) | ⭐⭐⭐⭐ | Individual fairness, counterfactual fairness, impossibility theorem |
| L12 | [12_Fairness_Mitigation.md](./12_Fairness_Mitigation.md) | ⭐⭐⭐⭐ | Pre/in/post-processing, Pareto frontiers, Fairlearn, AIF360 |
| L13 | [13_AI_Regulation_and_Governance.md](./13_AI_Regulation_and_Governance.md) | ⭐⭐⭐ | EU AI Act, GDPR Art. 22, NIST AI RMF, model cards |
| L14 | [14_Production_Interpretability.md](./14_Production_Interpretability.md) | ⭐⭐⭐⭐ | Explanation serving, caching, drift monitoring, MLOps integration |
| L15 | [15_Domain_Specific_Interpretability.md](./15_Domain_Specific_Interpretability.md) | ⭐⭐⭐ | Healthcare, finance, NLP, computer vision applications |
| L16 | [16_Mechanistic_Interpretability.md](./16_Mechanistic_Interpretability.md) | ⭐⭐⭐⭐ | Superposition, sparse autoencoders, circuit discovery, activation patching |

## Difficulty Guide

- ⭐⭐ Builds on ML L16 foundations with deeper taxonomy and frameworks
- ⭐⭐⭐ Requires comfortable PyTorch/math skills; implementation-focused
- ⭐⭐⭐⭐ Research-level methods; requires strong math and DL background

## Environment Setup

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

## Related Topics

- **[Machine_Learning](../Machine_Learning/00_Overview.md)**: Lesson 16 covers SHAP/LIME/PDP fundamentals (prerequisite)
- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: CNN and Transformer architectures explained here
- **[Foundation_Models](../Foundation_Models/00_Overview.md)**: Scaling and fine-tuning context for mechanistic interpretability
- **[MLOps](../MLOps/00_Overview.md)**: Production deployment context for explanation serving
- **[Probability_and_Statistics](../Probability_and_Statistics/00_Overview.md)**: Statistical testing used in evaluation and fairness

## Study Tips

1. **Complete ML L16 first** — this topic assumes you already understand SHAP, LIME, and PDP basics
2. **Run the code** — interpretability is best understood by visualizing explanations on real models
3. **Compare methods** — apply multiple methods to the same prediction and note where they agree/disagree
4. **Think about the audience** — different stakeholders need different types of explanations
5. **Stay current** — mechanistic interpretability is evolving rapidly; check recent papers
6. **Practice the fairness math** — impossibility theorems are counterintuitive until you work through proofs

## Learning Outcomes

After completing this topic, you will be able to:

- Implement gradient-based attribution methods (Integrated Gradients, GradCAM) from scratch in PyTorch
- Critically evaluate whether attention weights constitute valid explanations
- Generate and evaluate counterfactual explanations for tabular and image data
- Apply causal reasoning to distinguish genuine feature effects from spurious correlations
- Audit ML models for fairness using multiple definitions and implement mitigation strategies
- Design production systems that serve explanations alongside predictions
- Navigate AI regulatory frameworks (EU AI Act, GDPR) and create compliant documentation
- Apply mechanistic interpretability techniques to understand neural network internals

## Next Steps

- **[Foundation_Models](../Foundation_Models/00_Overview.md)**: Explore how interpretability scales to large models
- **[MLOps](../MLOps/00_Overview.md)**: Integrate explanation systems into ML pipelines
- **Research**: Follow Anthropic's mechanistic interpretability research, Google DeepMind's XAI work

---

**License**: CC BY-NC 4.0

[Start: Lesson 01 — Interpretability Foundations](./01_Interpretability_Foundations.md)
