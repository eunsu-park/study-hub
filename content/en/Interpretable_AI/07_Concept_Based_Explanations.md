# Lesson 7: Concept-Based Explanations

[Previous: Advanced SHAP](./06_Advanced_SHAP.md) | [Next: Counterfactual Explanations](./08_Counterfactual_Explanations.md)

---

## Learning Objectives

- Understand Testing with Concept Activation Vectors (TCAV) and implement it to quantify model sensitivity to human-defined concepts
- Build Concept Bottleneck Models (CBMs) that enforce explicit concept prediction as an intermediate step, enabling concept-level interventions at test time
- Apply Automated Concept-based Explanations (ACE) to discover concepts from data without manual annotation
- Evaluate concept completeness to determine whether discovered concepts fully explain model behavior
- Design practical TCAV experiments with custom concept sets for real-world model auditing

---

Feature-level explanations (SHAP, LIME, saliency maps) tell you which pixels or which tabular features matter, but they do not tell you **why** in terms that humans naturally reason about. When a doctor looks at a skin lesion, they think in concepts: "irregular border", "asymmetric shape", "dark coloration" — not in pixel coordinates. Concept-based explanations bridge this gap by explaining model behavior in terms of high-level, human-meaningful concepts.

This lesson covers four major approaches to concept-based explanations, from manually defined concept testing (TCAV) to fully automated concept discovery (ACE), and from post-hoc explanation (applied to any model) to concept-by-design architectures (Concept Bottleneck Models).

---

## 1. Testing with Concept Activation Vectors (TCAV)

### 1.1 The Key Idea

```python
"""
TCAV (Kim et al. 2018): Testing with Concept Activation Vectors

Central question: "How important is the concept 'striped' to the
model's prediction of 'zebra'?"

The method:
1. Collect a set of images that exemplify the concept (e.g., 30 images
   of striped textures) and a set of random images.
2. Extract the model's internal activations for both sets at a chosen layer.
3. Train a linear classifier to separate concept examples from random
   examples in activation space. The normal vector to the decision
   boundary is the Concept Activation Vector (CAV).
4. Compute the directional derivative of the model's output with
   respect to the CAV direction. This measures how much the model's
   prediction would change if the internal representation moved
   toward the concept.

Why linear? Because we want concepts to be directions in activation
space. A linear separator defines a direction (its normal vector),
and we can meaningfully talk about "moving toward" or "away from"
that direction. Non-linear separators don't define a single direction.

Why this matters:
- Explanations in human concepts, not pixel features
- Can test for bias: "Is 'female' a significant concept for 'nurse'?"
- Works on any differentiable model (CNNs, Transformers)
- Provides statistical testing (not just a single number)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_1samp
from typing import Optional
```

### 1.2 Complete TCAV Implementation

```python
class TCAV:
    """
    Full TCAV implementation for PyTorch models.

    Usage:
        tcav = TCAV(model, target_layer="layer4")
        cav = tcav.train_cav(concept_images, random_images)
        score = tcav.compute_tcav_score(test_images, class_idx=340, cav=cav)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: str = "cpu"
    ):
        """
        Parameters:
            model: Pre-trained model (e.g., InceptionV3, ResNet50)
            target_layer: Name of the layer to extract activations from.
                         Middle layers work best: early layers capture
                         low-level features, late layers are too task-specific.
            device: "cpu" or "cuda"
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks for activation extraction and gradient capture
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks at the target layer."""

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        layer = dict(self.model.named_modules())[self.target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def get_activations(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract activations at the target layer for a batch of images.

        Returns:
            Flattened activations, shape (n_images, activation_dim)
        """
        self.model.eval()
        all_activations = []

        with torch.no_grad():
            # Process in batches to manage memory
            batch_size = 32
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].to(self.device)
                _ = self.model(batch)

                # Flatten spatial dimensions: (B, C, H, W) → (B, C*H*W)
                act = self.activations.detach().cpu()
                if act.dim() == 4:
                    act = act.flatten(start_dim=1)
                all_activations.append(act.numpy())

        return np.concatenate(all_activations, axis=0)

    def train_cav(
        self,
        concept_images: torch.Tensor,
        random_images: torch.Tensor,
        regularization: float = 0.01
    ) -> dict:
        """
        Train a Concept Activation Vector (CAV).

        The CAV is the normal vector to the linear decision boundary
        separating concept examples from random examples in activation space.

        Parameters:
            concept_images: Images exemplifying the concept
            random_images: Random images (should be diverse, not from target class)
            regularization: L2 regularization strength for the linear classifier.
                          Higher values → simpler boundary → potentially less
                          faithful to the concept, but more generalizable.

        Returns:
            Dict with 'cav' (the vector), 'accuracy' (classifier accuracy),
            and 'classifier' (the trained model)
        """
        # Step 1: Extract activations
        concept_acts = self.get_activations(concept_images)
        random_acts = self.get_activations(random_images)

        # Step 2: Create binary labels
        # 1 = concept present, 0 = random (concept absent)
        X = np.concatenate([concept_acts, random_acts], axis=0)
        y = np.array([1] * len(concept_acts) + [0] * len(random_acts))

        # Step 3: Train linear classifier
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # SGDClassifier with hinge loss = linear SVM
        classifier = SGDClassifier(
            loss="hinge",
            alpha=regularization,
            max_iter=1000,
            random_state=42
        )
        classifier.fit(X_train, y_train)

        # Validation accuracy
        val_accuracy = accuracy_score(y_val, classifier.predict(X_val))

        # Step 4: Extract the CAV
        # The CAV is the weight vector of the linear classifier,
        # normalized to unit length. This is the direction in activation
        # space that points "toward" the concept.
        cav = classifier.coef_[0]
        cav = cav / np.linalg.norm(cav)

        print(f"CAV trained: validation accuracy = {val_accuracy:.3f}")

        # Sanity check: if accuracy ≈ 0.5, the concept is not linearly
        # separable at this layer, and the CAV is unreliable.
        if val_accuracy < 0.6:
            print("  WARNING: Low classifier accuracy. The concept may not be "
                  "linearly separable at this layer. Try a different layer.")

        return {
            "cav": cav,
            "accuracy": val_accuracy,
            "classifier": classifier
        }

    def compute_directional_derivative(
        self,
        image: torch.Tensor,
        class_idx: int,
        cav: np.ndarray
    ) -> float:
        """
        Compute the directional derivative of the model's output
        with respect to the CAV direction.

        This is the core TCAV computation:
            S_C,k,l(x) = lim_{epsilon→0} [h_l,k(f_l(x) + epsilon*v_C) - h_l,k(f_l(x))] / epsilon

        Where:
            v_C = the CAV direction
            f_l(x) = activations at layer l
            h_l,k = function from activations to class k logit

        In practice, this equals: gradient of class_k logit w.r.t. activations,
        dotted with the CAV.

            S_C,k,l(x) = ∇_{f_l(x)} h_l,k(x) · v_C
        """
        image = image.to(self.device).unsqueeze(0)
        image.requires_grad_(False)

        # Forward pass
        output = self.model(image)

        # Backward pass: gradient of class_idx logit w.r.t. activations
        self.model.zero_grad()
        class_logit = output[0, class_idx]
        class_logit.backward()

        # Get gradient at the target layer
        grad = self.gradients.detach().cpu()
        if grad.dim() == 4:
            grad = grad.flatten(start_dim=1)
        grad = grad.numpy().squeeze()

        # Directional derivative = dot product of gradient and CAV
        directional_deriv = np.dot(grad, cav)

        return directional_deriv

    def compute_tcav_score(
        self,
        test_images: torch.Tensor,
        class_idx: int,
        cav: np.ndarray
    ) -> dict:
        """
        Compute the TCAV score for a set of test images.

        TCAV score = fraction of test images for which the directional
        derivative is positive (i.e., moving toward the concept
        increases the model's prediction for the target class).

        TCAV score interpretation:
        - 0.5 = concept has no consistent effect (random direction)
        - > 0.5 = concept positively influences the prediction
        - < 0.5 = concept negatively influences the prediction
        - 1.0 = concept always increases the prediction (very strong)

        Returns:
            Dict with 'score', 'directional_derivatives', and 'p_value'
        """
        derivatives = []

        for i in range(len(test_images)):
            dd = self.compute_directional_derivative(
                test_images[i], class_idx, cav
            )
            derivatives.append(dd)

        derivatives = np.array(derivatives)

        # TCAV score = fraction of positive directional derivatives
        score = (derivatives > 0).mean()

        # Statistical testing: is the TCAV score significantly different
        # from 0.5 (random chance)?
        # We use a one-sample t-test on the directional derivatives
        # against 0 (if mean derivative is significantly > 0, the concept
        # positively influences the class prediction)
        t_stat, p_value = ttest_1samp(derivatives, 0)

        print(f"TCAV Score: {score:.3f}")
        print(f"Mean directional derivative: {derivatives.mean():.6f}")
        print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.6f}")

        if p_value < 0.05:
            direction = "positively" if derivatives.mean() > 0 else "negatively"
            print(f"Result: Concept SIGNIFICANTLY {direction} influences "
                  f"class {class_idx} (p < 0.05)")
        else:
            print(f"Result: Concept does NOT significantly influence "
                  f"class {class_idx}")

        return {
            "score": score,
            "directional_derivatives": derivatives,
            "p_value": p_value,
            "t_statistic": t_stat
        }
```

### 1.3 Statistical Testing with Multiple CAVs

```python
def tcav_with_statistical_testing(
    model: nn.Module,
    target_layer: str,
    concept_images: torch.Tensor,
    random_image_sets: list[torch.Tensor],
    test_images: torch.Tensor,
    class_idx: int,
    n_runs: int = 10,
    significance_level: float = 0.05
) -> dict:
    """
    Robust TCAV with multiple random sets for statistical testing.

    Kim et al. (2018) recommend:
    1. Train multiple CAVs using different random sets
    2. Compute TCAV scores for each CAV
    3. Use a statistical test to determine if TCAV scores are
       consistently above 0.5 across different random sets

    Why multiple random sets? A single CAV might be an artifact of
    the particular random images chosen. By training many CAVs with
    different random sets and checking consistency, we ensure the
    concept's influence is genuine.
    """
    tcav = TCAV(model, target_layer)
    all_scores = []
    all_accuracies = []

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        # Use a different random set for each run
        random_idx = run % len(random_image_sets)
        random_images = random_image_sets[random_idx]

        # Train CAV
        cav_result = tcav.train_cav(concept_images, random_images)
        all_accuracies.append(cav_result["accuracy"])

        # Compute TCAV score
        score_result = tcav.compute_tcav_score(
            test_images, class_idx, cav_result["cav"]
        )
        all_scores.append(score_result["score"])

    # Statistical test: are TCAV scores consistently > 0.5?
    all_scores = np.array(all_scores)
    t_stat, p_value = ttest_1samp(all_scores, 0.5)

    print(f"\n{'='*60}")
    print(f"TCAV Statistical Summary ({n_runs} runs)")
    print(f"{'='*60}")
    print(f"  Mean TCAV score: {all_scores.mean():.3f} "
          f"(+/- {all_scores.std():.3f})")
    print(f"  Mean CAV accuracy: {np.mean(all_accuracies):.3f}")
    print(f"  t-statistic vs 0.5: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")

    is_significant = p_value < significance_level
    if is_significant and all_scores.mean() > 0.5:
        print(f"  CONCLUSION: Concept SIGNIFICANTLY influences class {class_idx}")
    elif is_significant and all_scores.mean() < 0.5:
        print(f"  CONCLUSION: Concept SIGNIFICANTLY inhibits class {class_idx}")
    else:
        print(f"  CONCLUSION: No significant concept influence detected")

    return {
        "scores": all_scores,
        "mean_score": all_scores.mean(),
        "p_value": p_value,
        "is_significant": is_significant,
        "cav_accuracies": all_accuracies
    }
```

---

## 2. Concept Bottleneck Models (CBMs)

### 2.1 Architecture and Motivation

```python
"""
Concept Bottleneck Models (Koh et al. 2020)

Unlike TCAV (which is post-hoc), CBMs build concepts INTO the model
architecture as an explicit intermediate layer.

Architecture:
    Input → Feature Extractor → Concept Layer → Label Predictor → Output
                                    |
                                    └── Interpretable: each neuron
                                        in this layer corresponds to
                                        a known concept

Benefits:
1. INHERENT interpretability: the model MUST use concepts
2. CONCEPT INTERVENTION: at test time, a human can correct
   mispredicted concepts and see the effect on the final output
3. DEBUGGING: if the model fails, you can check which concept
   went wrong

Limitations:
1. Requires concept annotations (expensive to collect)
2. Concept bottleneck may limit model performance
3. Incomplete concepts = missing information

Training strategies:
- Independent: train concept predictor and label predictor separately
- Sequential: train concept predictor first, then label predictor on top
- Joint: train both end-to-end (best performance, less interpretable)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class ConceptBottleneckModel(nn.Module):
    """
    A Concept Bottleneck Model with a ResNet backbone.

    The model has two stages:
    1. Concept Predictor: ResNet features → concept predictions
    2. Label Predictor: concept predictions → final class label

    The concept layer is the "bottleneck" — all information must
    flow through human-interpretable concept predictions.
    """

    def __init__(
        self,
        n_concepts: int,
        n_classes: int,
        backbone: str = "resnet18",
        use_sigmoid: bool = True
    ):
        """
        Parameters:
            n_concepts: Number of binary concepts (e.g., 112 for CUB birds)
            n_classes: Number of output classes (e.g., 200 for CUB birds)
            backbone: Feature extractor architecture
            use_sigmoid: If True, concept layer uses sigmoid (binary concepts).
                        If False, concepts are continuous (soft bottleneck).
        """
        super().__init__()

        # Stage 1: Feature extractor (pre-trained, fine-tuned)
        resnet = models.resnet18(pretrained=True)
        feature_dim = resnet.fc.in_features
        # Remove the original classification head
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Stage 2: Concept predictor
        # Maps features to concept predictions
        # Each concept is predicted independently (multi-label)
        self.concept_predictor = nn.Linear(feature_dim, n_concepts)
        self.use_sigmoid = use_sigmoid

        # Stage 3: Label predictor
        # Maps concept predictions to final class label
        # This is deliberately kept simple (linear) for interpretability:
        # the weight matrix directly shows which concepts influence which class
        self.label_predictor = nn.Linear(n_concepts, n_classes)

        self.n_concepts = n_concepts
        self.n_classes = n_classes

    def forward(
        self,
        x: torch.Tensor,
        concept_intervention: dict[int, float] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional concept intervention.

        Parameters:
            x: Input images, shape (B, 3, H, W)
            concept_intervention: Optional dict mapping concept_index → value.
                                 If provided, overrides the model's concept
                                 predictions with human-specified values.
                                 This is the key CBM feature: humans can
                                 correct mistakes at the concept level.

        Returns:
            (class_logits, concept_predictions)
        """
        # Extract features
        features = self.feature_extractor(x)
        features = features.flatten(start_dim=1)  # (B, feature_dim)

        # Predict concepts
        concept_logits = self.concept_predictor(features)

        if self.use_sigmoid:
            concept_preds = torch.sigmoid(concept_logits)
        else:
            concept_preds = concept_logits

        # Apply concept interventions (if any)
        if concept_intervention is not None:
            concept_preds = concept_preds.clone()
            for concept_idx, value in concept_intervention.items():
                concept_preds[:, concept_idx] = value

        # Predict final label from concepts
        class_logits = self.label_predictor(concept_preds)

        return class_logits, concept_preds

    def get_concept_influence(self) -> np.ndarray:
        """
        Get the weight matrix showing how each concept influences each class.

        Returns:
            Matrix of shape (n_classes, n_concepts) where entry (i, j)
            is the influence of concept j on class i.

        This is one of the key interpretability benefits: you can directly
        read off which concepts matter for which classes.
        """
        return self.label_predictor.weight.detach().cpu().numpy()
```

### 2.2 Training Strategies

```python
class CBMTrainer:
    """
    Train a Concept Bottleneck Model with different strategies.

    Three strategies from Koh et al. (2020):

    1. Independent: Train concept predictor and label predictor SEPARATELY.
       - Concept predictor: minimize concept prediction loss
       - Label predictor: train on GROUND TRUTH concepts (not predicted)
       - Most interpretable, but concept predictor errors don't backprop

    2. Sequential: Train concept predictor first, THEN label predictor
       on the predicted concepts (not ground truth).
       - Better calibrated: label predictor learns to handle concept errors
       - Still no end-to-end gradient flow

    3. Joint: Train both end-to-end with a combined loss.
       - Best task accuracy (gradient flows through concepts)
       - But concepts may become less interpretable (they optimize
         for the task, not necessarily for human-meaningful concepts)
    """

    def __init__(self, model: ConceptBottleneckModel, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def train_independent(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
        learning_rate: float = 1e-3
    ) -> dict:
        """
        Independent training strategy.

        Phase 1: Train concept predictor to predict concepts from images.
        Phase 2: Train label predictor to predict labels from TRUE concepts.
        """
        # Phase 1: Concept predictor
        print("Phase 1: Training concept predictor...")
        concept_optimizer = torch.optim.Adam(
            list(self.model.feature_extractor.parameters()) +
            list(self.model.concept_predictor.parameters()),
            lr=learning_rate
        )
        concept_criterion = nn.BCEWithLogitsLoss()

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            n_batches = 0

            for images, concepts, labels in train_loader:
                images = images.to(self.device)
                concepts = concepts.to(self.device).float()

                concept_optimizer.zero_grad()

                features = self.model.feature_extractor(images)
                features = features.flatten(start_dim=1)
                concept_logits = self.model.concept_predictor(features)

                loss = concept_criterion(concept_logits, concepts)
                loss.backward()
                concept_optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: concept loss = "
                      f"{total_loss/n_batches:.4f}")

        # Phase 2: Label predictor (using TRUE concepts)
        print("\nPhase 2: Training label predictor on true concepts...")

        # Freeze everything except label predictor
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.model.concept_predictor.parameters():
            param.requires_grad = False

        label_optimizer = torch.optim.Adam(
            self.model.label_predictor.parameters(), lr=learning_rate
        )
        label_criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            n_batches = 0

            for images, concepts, labels in train_loader:
                concepts = concepts.to(self.device).float()
                labels = labels.to(self.device)

                label_optimizer.zero_grad()
                # Use TRUE concepts, not predicted ones
                class_logits = self.model.label_predictor(concepts)
                loss = label_criterion(class_logits, labels)
                loss.backward()
                label_optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: label loss = "
                      f"{total_loss/n_batches:.4f}")

        return self._evaluate(val_loader)

    def train_joint(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
        learning_rate: float = 1e-3,
        concept_loss_weight: float = 1.0
    ) -> dict:
        """
        Joint training strategy: end-to-end with combined loss.

        Loss = label_loss + concept_loss_weight * concept_loss

        The concept_loss_weight controls the tradeoff:
        - High weight: concepts are more accurate, but task accuracy may suffer
        - Low weight: better task accuracy, but concepts may lose meaning
        """
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        concept_criterion = nn.BCEWithLogitsLoss()
        label_criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.model.train()
            total_concept_loss = 0
            total_label_loss = 0
            n_batches = 0

            for images, concepts, labels in train_loader:
                images = images.to(self.device)
                concepts = concepts.to(self.device).float()
                labels = labels.to(self.device)

                optimizer.zero_grad()

                class_logits, concept_preds = self.model(images)

                # Combined loss
                c_loss = concept_criterion(
                    self.model.concept_predictor(
                        self.model.feature_extractor(images).flatten(1)
                    ),
                    concepts
                )
                l_loss = label_criterion(class_logits, labels)

                total_loss = l_loss + concept_loss_weight * c_loss
                total_loss.backward()
                optimizer.step()

                total_concept_loss += c_loss.item()
                total_label_loss += l_loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: concept_loss = "
                      f"{total_concept_loss/n_batches:.4f}, "
                      f"label_loss = {total_label_loss/n_batches:.4f}")

        return self._evaluate(val_loader)

    def _evaluate(self, val_loader: DataLoader) -> dict:
        """Evaluate model on concept accuracy and label accuracy."""
        self.model.eval()
        all_concept_preds = []
        all_concept_true = []
        all_label_preds = []
        all_label_true = []

        with torch.no_grad():
            for images, concepts, labels in val_loader:
                images = images.to(self.device)
                class_logits, concept_preds = self.model(images)

                all_concept_preds.append((concept_preds > 0.5).cpu().numpy())
                all_concept_true.append(concepts.numpy())
                all_label_preds.append(class_logits.argmax(1).cpu().numpy())
                all_label_true.append(labels.numpy())

        concept_preds = np.concatenate(all_concept_preds)
        concept_true = np.concatenate(all_concept_true)
        label_preds = np.concatenate(all_label_preds)
        label_true = np.concatenate(all_label_true)

        concept_acc = (concept_preds == concept_true).mean()
        label_acc = (label_preds == label_true).mean()

        print(f"\nEvaluation Results:")
        print(f"  Concept accuracy: {concept_acc:.4f}")
        print(f"  Label accuracy:   {label_acc:.4f}")

        return {
            "concept_accuracy": concept_acc,
            "label_accuracy": label_acc
        }
```

### 2.3 Concept Intervention at Test Time

```python
def demonstrate_concept_intervention(
    model: ConceptBottleneckModel,
    test_image: torch.Tensor,
    concept_names: list[str],
    true_concepts: np.ndarray
):
    """
    Demonstrate the power of concept intervention in CBMs.

    Scenario: A bird classification model predicts the wrong species.
    A domain expert can examine the predicted concepts, correct any
    mistakes, and see if the corrected concepts fix the prediction.

    This is CBM's killer feature: interactive debugging and correction.
    """
    model.eval()

    # Step 1: Normal prediction (no intervention)
    print("Step 1: Prediction WITHOUT intervention")
    print("-" * 50)
    with torch.no_grad():
        class_logits, concept_preds = model(test_image.unsqueeze(0))

    predicted_class = class_logits.argmax(1).item()
    predicted_concepts = (concept_preds > 0.5).squeeze().numpy()

    print(f"Predicted class: {predicted_class}")
    print(f"\nConcept predictions vs ground truth:")
    mismatches = []
    for i, name in enumerate(concept_names):
        pred = predicted_concepts[i]
        true = true_concepts[i]
        match = "OK" if pred == true else "WRONG"
        if pred != true:
            mismatches.append(i)
        print(f"  {name:25s}: predicted={int(pred)}, "
              f"true={int(true)}  [{match}]")

    # Step 2: Intervene on incorrect concepts
    if mismatches:
        print(f"\nStep 2: Prediction WITH intervention")
        print(f"  Correcting {len(mismatches)} concept(s)...")
        print("-" * 50)

        # Create intervention dict: correct the wrong concepts
        interventions = {}
        for idx in mismatches:
            interventions[idx] = float(true_concepts[idx])
            print(f"  Correcting '{concept_names[idx]}': "
                  f"{int(predicted_concepts[idx])} → {int(true_concepts[idx])}")

        with torch.no_grad():
            corrected_logits, _ = model(
                test_image.unsqueeze(0),
                concept_intervention=interventions
            )

        corrected_class = corrected_logits.argmax(1).item()
        print(f"\nCorrected class: {corrected_class}")

        if corrected_class != predicted_class:
            print(f"Intervention CHANGED the prediction: "
                  f"{predicted_class} → {corrected_class}")
        else:
            print(f"Intervention did NOT change the prediction.")
            print(f"  (Other concepts might also be wrong, or the "
                  f"label predictor is robust to these corrections)")

    # Step 3: Sensitivity analysis — which concepts matter most?
    print(f"\nStep 3: Concept Sensitivity Analysis")
    print("-" * 50)

    sensitivities = []
    base_prob = torch.softmax(class_logits, dim=1)[0, predicted_class].item()

    for i in range(len(concept_names)):
        # Flip concept i and measure the effect
        flip_intervention = {i: 1.0 - predicted_concepts[i]}
        with torch.no_grad():
            flipped_logits, _ = model(
                test_image.unsqueeze(0),
                concept_intervention=flip_intervention
            )
        flipped_prob = torch.softmax(flipped_logits, dim=1)[0, predicted_class].item()
        sensitivity = abs(base_prob - flipped_prob)
        sensitivities.append((concept_names[i], sensitivity))

    sensitivities.sort(key=lambda x: x[1], reverse=True)
    print("Most influential concepts for this prediction:")
    for name, sens in sensitivities[:10]:
        print(f"  {name:25s}: sensitivity = {sens:.4f}")
```

---

## 3. Automated Concept-based Explanations (ACE)

### 3.1 Overview

```python
"""
ACE (Ghorbani et al. 2019): Automated Concept-based Explanations

Problem with TCAV: requires human-defined concept sets.
    "What if we don't know which concepts to test?"
    "What if the model uses concepts we haven't thought of?"

ACE solves this by AUTOMATICALLY discovering concepts from the data.

Algorithm:
1. MULTI-RESOLUTION SEGMENTATION: For each image in the target class,
   segment it at multiple resolutions (coarse → fine).
   This captures concepts at different scales:
   - Coarse segments: whole objects, large regions
   - Medium segments: parts (head, wing, wheel)
   - Fine segments: textures, patterns, details

2. ACTIVATION EXTRACTION: For each segment, resize it and extract
   the model's activations at a chosen layer.

3. CLUSTERING: Cluster the segment activations across all images.
   Each cluster represents a candidate concept: segments that
   activate the model similarly are grouped together.

4. OUTLIER REMOVAL: Remove small clusters and outliers.

5. TCAV TESTING: For each cluster (candidate concept), compute
   TCAV score to measure its importance to the target class.
"""

from sklearn.cluster import KMeans
from skimage.segmentation import slic
from scipy.ndimage import label as ndimage_label
import torch
import numpy as np


class ACEExplainer:
    """
    Automated Concept-based Explanations (ACE).

    Discovers concepts automatically from images of a target class,
    without requiring human-defined concept sets.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.target_layer = target_layer
        self.activations = None

        # Register hook
        layer = dict(self.model.named_modules())[self.target_layer]
        layer.register_forward_hook(
            lambda m, inp, out: setattr(self, 'activations', out.detach())
        )

    def multi_resolution_segmentation(
        self,
        image: np.ndarray,
        resolutions: list[int] = None
    ) -> list[np.ndarray]:
        """
        Segment an image at multiple resolutions using SLIC.

        Parameters:
            image: Input image, shape (H, W, 3), values in [0, 1]
            resolutions: Number of segments at each resolution.
                        Default: [15, 50, 80] (coarse, medium, fine)

        Returns:
            List of binary masks, one per segment across all resolutions.
            Each mask has shape (H, W).
        """
        if resolutions is None:
            resolutions = [15, 50, 80]

        all_segments = []

        for n_segments in resolutions:
            # SLIC superpixel segmentation
            segment_labels = slic(
                image,
                n_segments=n_segments,
                compactness=10,
                sigma=1,
                start_label=0
            )

            # Convert each segment label to a binary mask
            unique_labels = np.unique(segment_labels)
            for seg_label in unique_labels:
                mask = (segment_labels == seg_label).astype(np.float32)

                # Filter: skip very small segments (< 1% of image)
                # and very large segments (> 50% of image)
                area_ratio = mask.sum() / mask.size
                if 0.01 < area_ratio < 0.50:
                    all_segments.append(mask)

        return all_segments

    def extract_segment_activation(
        self,
        image: torch.Tensor,
        mask: np.ndarray,
        resize_to: int = 224
    ) -> np.ndarray:
        """
        Extract model activations for a specific image segment.

        The segment is isolated by masking the image (setting
        non-segment pixels to the mean value) and then extracting
        activations from the target layer.

        Why mask instead of crop? Masking preserves spatial relationships
        and context, giving more meaningful activations.
        """
        # Apply mask: keep segment pixels, set others to mean
        image_np = image.permute(1, 2, 0).numpy()  # (H, W, 3)

        # Resize mask to image dimensions
        from skimage.transform import resize as sk_resize
        mask_resized = sk_resize(mask, image_np.shape[:2], order=0)

        # Create masked image
        mean_pixel = image_np.mean(axis=(0, 1))
        masked_image = image_np * mask_resized[:, :, None] + \
                       mean_pixel * (1 - mask_resized[:, :, None])

        # Convert back to tensor and get activations
        masked_tensor = torch.tensor(
            masked_image, dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.model(masked_tensor)
            act = self.activations.cpu()
            if act.dim() == 4:
                # Global average pool spatial dimensions
                act = act.mean(dim=[2, 3])
            return act.squeeze().numpy()

    def discover_concepts(
        self,
        images: list[torch.Tensor],
        n_concepts: int = 25,
        min_cluster_size: int = 5,
        resolutions: list[int] = None
    ) -> dict:
        """
        Full ACE pipeline: discover concepts from a set of images.

        Parameters:
            images: List of images from the target class
            n_concepts: Number of concept clusters to find
            min_cluster_size: Minimum segments per cluster
            resolutions: Segmentation resolutions

        Returns:
            Dict with discovered concepts, their exemplar segments,
            and activation vectors.
        """
        print(f"Step 1: Multi-resolution segmentation of {len(images)} images")

        all_activations = []
        all_segment_info = []  # Track which image and mask each activation came from

        for img_idx, image in enumerate(images):
            if (img_idx + 1) % 10 == 0:
                print(f"  Processing image {img_idx + 1}/{len(images)}")

            # Convert to numpy for segmentation
            img_np = image.permute(1, 2, 0).numpy()
            if img_np.max() > 1.0:
                img_np = img_np / 255.0

            # Segment at multiple resolutions
            segments = self.multi_resolution_segmentation(img_np, resolutions)

            for seg_idx, mask in enumerate(segments):
                # Extract activation for this segment
                activation = self.extract_segment_activation(image, mask)
                all_activations.append(activation)
                all_segment_info.append({
                    "image_idx": img_idx,
                    "segment_idx": seg_idx,
                    "mask": mask
                })

        activations_matrix = np.array(all_activations)
        print(f"  Total segments: {len(activations_matrix)}")
        print(f"  Activation dim: {activations_matrix.shape[1]}")

        # Step 2: Cluster segments
        print(f"\nStep 2: Clustering into {n_concepts} concepts")

        # Normalize activations before clustering
        # This ensures that cluster assignments are based on direction,
        # not magnitude (similar to cosine similarity)
        norms = np.linalg.norm(activations_matrix, axis=1, keepdims=True)
        normalized = activations_matrix / (norms + 1e-10)

        kmeans = KMeans(n_clusters=n_concepts, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized)

        # Step 3: Filter small clusters and organize results
        print(f"\nStep 3: Filtering and organizing concepts")

        concepts = {}
        for cluster_id in range(n_concepts):
            member_indices = np.where(cluster_labels == cluster_id)[0]

            if len(member_indices) < min_cluster_size:
                continue  # Skip tiny clusters (likely noise)

            # Compute mean activation for this concept (the concept vector)
            concept_vector = activations_matrix[member_indices].mean(axis=0)
            concept_vector = concept_vector / (np.linalg.norm(concept_vector) + 1e-10)

            concepts[f"concept_{cluster_id}"] = {
                "vector": concept_vector,
                "n_members": len(member_indices),
                "member_indices": member_indices,
                "segments": [all_segment_info[i] for i in member_indices],
                "mean_activation_norm": norms[member_indices].mean(),
            }

        print(f"  Discovered {len(concepts)} valid concepts "
              f"(filtered from {n_concepts} clusters)")

        for name, info in sorted(concepts.items()):
            print(f"    {name}: {info['n_members']} segments, "
                  f"mean activation norm = {info['mean_activation_norm']:.3f}")

        return concepts

    def rank_concepts_by_importance(
        self,
        concepts: dict,
        test_images: torch.Tensor,
        class_idx: int,
        random_images: torch.Tensor
    ) -> list[tuple[str, float, float]]:
        """
        Rank discovered concepts by TCAV score.

        For each concept, train a CAV and compute TCAV score.
        Returns concepts sorted by importance.
        """
        tcav_obj = TCAV(self.model, self.target_layer, self.device)

        results = []

        for concept_name, concept_info in concepts.items():
            # The concept vector IS the CAV (it separates concept
            # segments from others in activation space)
            cav = concept_info["vector"]

            # Compute TCAV score
            score_result = tcav_obj.compute_tcav_score(
                test_images, class_idx, cav
            )

            results.append((
                concept_name,
                score_result["score"],
                score_result["p_value"]
            ))

        # Sort by TCAV score (descending)
        results.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)

        print("\nConcept Importance Ranking:")
        print(f"{'Concept':20s} {'TCAV Score':>12s} {'p-value':>10s} {'Significant?':>14s}")
        print("-" * 60)
        for name, score, p_val in results:
            sig = "YES" if p_val < 0.05 else "no"
            print(f"{name:20s} {score:12.3f} {p_val:10.6f} {sig:>14s}")

        return results
```

---

## 4. Net2Vec: Learning Concept Vectors from Few Examples

### 4.1 Core Approach

```python
"""
Net2Vec (Fong & Vedaldi 2018): Learning concept vectors from very few
labeled examples.

Key differences from TCAV:
- TCAV trains a linear SVM to find the concept direction
- Net2Vec trains a linear layer that predicts concept presence from
  individual neuron (channel) activations

This reveals WHICH neurons encode the concept, not just the direction
in activation space.

The approach: For a single convolutional channel c at layer l,
train a linear combination of the spatial activation map to predict
the concept's segmentation mask:

    concept_prediction(x, y) = sigmoid(w^T * activation_map_c(x, y) + b)

If this simple predictor works well for channel c, then that channel
encodes the concept.
"""


class Net2Vec:
    """
    Learn concept vectors from a small number of annotated examples.
    """

    def __init__(self, model: nn.Module, target_layer: str, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.activations = None

        layer = dict(self.model.named_modules())[target_layer]
        layer.register_forward_hook(
            lambda m, inp, out: setattr(self, 'activations', out.detach())
        )

    def learn_concept_vector(
        self,
        images: torch.Tensor,
        concept_masks: torch.Tensor,
        n_channels: int = None
    ) -> dict:
        """
        Learn which channels (neurons) encode a concept.

        Parameters:
            images: Example images, shape (N, 3, H, W)
            concept_masks: Binary masks for concept presence, shape (N, H', W')
            n_channels: Number of channels at target layer (auto-detected)

        Returns:
            Dict with per-channel concept alignment scores and the
            overall concept vector.
        """
        # Get activations for all images
        all_activations = []
        with torch.no_grad():
            for img in images:
                _ = self.model(img.unsqueeze(0).to(self.device))
                all_activations.append(self.activations.cpu())

        activations = torch.cat(all_activations, dim=0)  # (N, C, H_a, W_a)
        n_images, n_channels_actual, h_a, w_a = activations.shape

        if n_channels is None:
            n_channels = n_channels_actual

        # Resize concept masks to match activation spatial dimensions
        from torch.nn.functional import interpolate
        masks_resized = interpolate(
            concept_masks.unsqueeze(1).float(),
            size=(h_a, w_a),
            mode="nearest"
        ).squeeze(1)  # (N, H_a, W_a)

        # For each channel, compute IoU with the concept mask
        # (similar to Network Dissection, but from the concept's perspective)
        channel_scores = []

        for c in range(n_channels):
            channel_act = activations[:, c, :, :]  # (N, H_a, W_a)

            # Threshold at the 99.5th percentile (same as Network Dissection)
            threshold = torch.quantile(channel_act.flatten(), 0.995)
            binary_act = (channel_act > threshold).float()

            # IoU between channel activation and concept mask
            intersection = (binary_act * masks_resized).sum()
            union = ((binary_act + masks_resized) > 0).float().sum()
            iou = (intersection / (union + 1e-10)).item()

            channel_scores.append(iou)

        channel_scores = np.array(channel_scores)

        # The concept vector: weighted combination of channels
        # Channels with higher IoU get higher weight
        concept_vector = channel_scores / (channel_scores.sum() + 1e-10)

        # Find the top-k most concept-aligned channels
        top_channels = np.argsort(-channel_scores)[:10]

        print(f"Concept Vector Summary:")
        print(f"  Top concept-aligned channels:")
        for rank, ch in enumerate(top_channels):
            print(f"    Channel {ch}: IoU = {channel_scores[ch]:.4f}")

        return {
            "concept_vector": concept_vector,
            "channel_scores": channel_scores,
            "top_channels": top_channels
        }
```

---

## 5. Concept Completeness

### 5.1 Do Concepts Fully Explain the Model?

```python
"""
Concept Completeness (Yeh et al. 2020): A critical question for
concept-based explanations is whether the discovered concepts are
SUFFICIENT to explain the model's behavior.

If we can reconstruct the model's predictions from concept activations
alone, then the concepts are complete. If not, the model uses additional
information that our concepts don't capture.

Completeness Score:
    C = 1 - [error of concept-based predictor / error of trivial predictor]

Where:
    - concept-based predictor: predicts labels from concept activations
    - trivial predictor: always predicts the majority class

C = 1.0 → concepts perfectly explain the model
C = 0.0 → concepts are no better than random
C < 0.0 → concepts are misleading
"""


def compute_concept_completeness(
    model: nn.Module,
    concept_vectors: list[np.ndarray],
    X_test: torch.Tensor,
    y_test: np.ndarray,
    target_layer: str,
    device: str = "cpu"
) -> float:
    """
    Compute the completeness score for a set of concepts.

    Parameters:
        model: The model being explained
        concept_vectors: List of concept activation vectors (CAVs)
        X_test: Test images
        y_test: True labels
        target_layer: Layer where concepts are defined

    Returns:
        Completeness score in [0, 1]
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    model_to_eval = model.to(device)
    model_to_eval.eval()

    # Extract activations at target layer
    activations_list = []
    hook_ref = [None]

    def hook_fn(module, input, output):
        hook_ref[0] = output.detach()

    layer = dict(model_to_eval.named_modules())[target_layer]
    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for img in X_test:
            _ = model_to_eval(img.unsqueeze(0).to(device))
            act = hook_ref[0].cpu()
            if act.dim() == 4:
                act = act.mean(dim=[2, 3])
            activations_list.append(act.squeeze().numpy())

    handle.remove()
    activations = np.array(activations_list)

    # Project activations onto concept directions
    # Each concept vector defines a direction; the projection
    # gives a "concept activation level" for each instance
    concept_activations = np.zeros((len(X_test), len(concept_vectors)))
    for i, cav in enumerate(concept_vectors):
        concept_activations[:, i] = activations @ cav

    # Train a simple classifier on concept activations
    concept_clf = LogisticRegression(max_iter=1000)
    concept_clf.fit(concept_activations, y_test)
    concept_preds = concept_clf.predict(concept_activations)
    concept_acc = accuracy_score(y_test, concept_preds)

    # Trivial baseline: majority class
    from collections import Counter
    majority_class = Counter(y_test).most_common(1)[0][0]
    trivial_acc = (y_test == majority_class).mean()

    # Also get model's own accuracy for reference
    model_preds = []
    with torch.no_grad():
        for img in X_test:
            output = model_to_eval(img.unsqueeze(0).to(device))
            model_preds.append(output.argmax(1).cpu().item())
    model_acc = accuracy_score(y_test, model_preds)

    # Completeness score
    # How much of the model's accuracy is captured by concepts?
    if model_acc > trivial_acc:
        completeness = (concept_acc - trivial_acc) / (model_acc - trivial_acc)
    else:
        completeness = 0.0

    completeness = max(0.0, min(1.0, completeness))

    print(f"Concept Completeness Analysis:")
    print(f"  Model accuracy:         {model_acc:.4f}")
    print(f"  Concept-based accuracy: {concept_acc:.4f}")
    print(f"  Trivial baseline:       {trivial_acc:.4f}")
    print(f"  Completeness score:     {completeness:.4f}")
    print()

    if completeness > 0.9:
        print(f"  EXCELLENT: Concepts explain >90% of model behavior.")
    elif completeness > 0.7:
        print(f"  GOOD: Concepts capture most of model behavior, "
              f"but some information is missing.")
    elif completeness > 0.4:
        print(f"  MODERATE: Significant model behavior is NOT captured "
              f"by the current concepts. Consider discovering more concepts.")
    else:
        print(f"  POOR: Concepts fail to explain model behavior. "
              f"The model likely uses very different features.")

    return completeness
```

---

## 6. Practical: TCAV on InceptionV3 with Custom Concept Sets

### 6.1 Complete End-to-End Example

```python
def tcav_inception_experiment():
    """
    Complete practical example: Using TCAV to audit InceptionV3
    for concept-level biases and associations.

    Experiment: Test whether the concept "striped" is important
    for classifying "zebra" (it should be) and whether "dotted"
    is important (it shouldn't be).

    In a real bias audit, you might test:
    - Is "male face" important for "doctor"? (potential gender bias)
    - Is "white skin" important for "criminal"? (potential racial bias)
    - Is "hospital" important for "sick person"? (spurious correlation)
    """
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder

    # Load InceptionV3
    model = models.inception_v3(pretrained=True, transform_input=True)
    model.eval()

    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # --- Prepare concept datasets ---
    # In practice, collect ~30-50 images per concept.
    # Concept images should be DIVERSE (different objects that share
    # the concept, not just one type of object).
    #
    # Good "striped" concept set: zebras, tigers, striped shirts,
    #   barber poles, railroad tracks, venetian blinds
    # Bad "striped" concept set: only zebra images
    #   (this would conflate "striped" with "zebra")

    # For demonstration, we create synthetic concept images
    def create_synthetic_concept_images(pattern: str, n: int = 30):
        """Create synthetic images with specific patterns."""
        images = []
        for _ in range(n):
            img = torch.rand(3, 299, 299)

            if pattern == "striped":
                # Add horizontal stripes
                for row in range(0, 299, 10):
                    if (row // 10) % 2 == 0:
                        img[:, row:row+5, :] = 0
            elif pattern == "dotted":
                # Add dots
                for _ in range(50):
                    cx, cy = np.random.randint(10, 289, 2)
                    r = 5
                    for dx in range(-r, r+1):
                        for dy in range(-r, r+1):
                            if dx*dx + dy*dy <= r*r:
                                img[:, cx+dx, cy+dy] = 0
            elif pattern == "random":
                pass  # Already random noise

            images.append(img)
        return torch.stack(images)

    print("Creating concept image sets...")
    striped_images = create_synthetic_concept_images("striped", n=30)
    dotted_images = create_synthetic_concept_images("dotted", n=30)
    random_images_1 = create_synthetic_concept_images("random", n=30)
    random_images_2 = create_synthetic_concept_images("random", n=30)
    random_images_3 = create_synthetic_concept_images("random", n=30)

    # Test images: zebra class images (class index 340 in ImageNet)
    # In practice, use real zebra images from ImageNet
    test_images = create_synthetic_concept_images("striped", n=50)

    # --- Run TCAV ---
    # Test at Mixed_5d layer (middle of InceptionV3)
    # This is a good choice because:
    # - Early layers (Mixed_5a): too low-level (edges, colors)
    # - Late layers (Mixed_7c): too task-specific
    # - Middle layers: balance between feature richness and generality

    target_layer = "Mixed_5d"  # InceptionV3 layer name
    zebra_class_idx = 340     # ImageNet class index for zebra

    print("\n" + "=" * 60)
    print("Experiment 1: Is 'striped' important for 'zebra'?")
    print("=" * 60)

    tcav_striped = tcav_with_statistical_testing(
        model=model,
        target_layer=target_layer,
        concept_images=striped_images,
        random_image_sets=[random_images_1, random_images_2, random_images_3],
        test_images=test_images,
        class_idx=zebra_class_idx,
        n_runs=6
    )

    print("\n" + "=" * 60)
    print("Experiment 2: Is 'dotted' important for 'zebra'?")
    print("=" * 60)

    tcav_dotted = tcav_with_statistical_testing(
        model=model,
        target_layer=target_layer,
        concept_images=dotted_images,
        random_image_sets=[random_images_1, random_images_2, random_images_3],
        test_images=test_images,
        class_idx=zebra_class_idx,
        n_runs=6
    )

    # --- Compare results ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Striped → Zebra: TCAV = {tcav_striped['mean_score']:.3f}, "
          f"p = {tcav_striped['p_value']:.6f}, "
          f"significant = {tcav_striped['is_significant']}")
    print(f"  Dotted  → Zebra: TCAV = {tcav_dotted['mean_score']:.3f}, "
          f"p = {tcav_dotted['p_value']:.6f}, "
          f"significant = {tcav_dotted['is_significant']}")

    print("\nExpected results:")
    print("  - 'Striped' should be significant (zebras ARE striped)")
    print("  - 'Dotted' should NOT be significant (zebras are not dotted)")

    return tcav_striped, tcav_dotted


tcav_inception_experiment()
```

### 6.2 Concept-Based Model Auditing Pipeline

```python
def concept_audit_pipeline(
    model: nn.Module,
    target_layer: str,
    class_names: dict[int, str],
    concept_datasets: dict[str, torch.Tensor],
    random_datasets: list[torch.Tensor],
    test_datasets: dict[int, torch.Tensor],
    n_runs: int = 10
) -> dict:
    """
    Complete concept-based auditing pipeline.

    This pipeline tests ALL concepts against ALL classes,
    producing a comprehensive concept importance matrix.

    Parameters:
        model: Model to audit
        target_layer: Layer for concept extraction
        class_names: {class_idx: "class name"}
        concept_datasets: {"concept_name": concept_images}
        random_datasets: List of random image sets for statistical testing
        test_datasets: {class_idx: test_images_for_class}
        n_runs: Number of random runs per TCAV test

    Returns:
        Matrix of TCAV scores: concepts x classes
    """
    results = {}

    n_concepts = len(concept_datasets)
    n_classes = len(class_names)
    total_tests = n_concepts * n_classes

    print(f"Concept Audit: {n_concepts} concepts x {n_classes} classes "
          f"= {total_tests} tests")
    print("=" * 70)

    test_num = 0
    for concept_name, concept_images in concept_datasets.items():
        results[concept_name] = {}

        for class_idx, class_name in class_names.items():
            test_num += 1
            print(f"\n[{test_num}/{total_tests}] "
                  f"Testing '{concept_name}' → '{class_name}'")

            test_images = test_datasets[class_idx]

            tcav_result = tcav_with_statistical_testing(
                model=model,
                target_layer=target_layer,
                concept_images=concept_images,
                random_image_sets=random_datasets,
                test_images=test_images,
                class_idx=class_idx,
                n_runs=n_runs
            )

            results[concept_name][class_name] = {
                "tcav_score": tcav_result["mean_score"],
                "p_value": tcav_result["p_value"],
                "significant": tcav_result["is_significant"]
            }

    # --- Generate audit report ---
    print("\n" + "=" * 70)
    print("CONCEPT AUDIT REPORT")
    print("=" * 70)

    # Print matrix
    class_list = list(class_names.values())
    concept_list = list(concept_datasets.keys())

    header = f"{'Concept':20s}" + "".join(f"{c:>15s}" for c in class_list)
    print(header)
    print("-" * len(header))

    for concept in concept_list:
        row = f"{concept:20s}"
        for class_name in class_list:
            score = results[concept][class_name]["tcav_score"]
            sig = results[concept][class_name]["significant"]
            marker = "*" if sig else " "
            row += f"{score:14.3f}{marker}"
        print(row)

    print(f"\n* = statistically significant (p < 0.05)")

    # Flag potentially problematic associations
    print("\nPotentially Problematic Associations:")
    for concept in concept_list:
        for class_name in class_list:
            r = results[concept][class_name]
            if r["significant"] and r["tcav_score"] > 0.7:
                print(f"  ALERT: '{concept}' strongly associated with "
                      f"'{class_name}' (TCAV={r['tcav_score']:.3f})")

    return results
```

---

## Summary

- **TCAV** (Kim et al. 2018) defines concepts as directions in activation space using Concept Activation Vectors (CAVs). The TCAV score quantifies what fraction of test instances are positively influenced by the concept. Statistical testing across multiple random sets ensures reliability.
- **Concept Bottleneck Models** (Koh et al. 2020) embed concept prediction as an explicit architectural bottleneck, enabling concept-level intervention at test time. Three training strategies (independent, sequential, joint) trade off interpretability against task accuracy.
- **ACE** (Ghorbani et al. 2019) discovers concepts automatically through multi-resolution segmentation and clustering, eliminating the need for human-defined concept sets. Discovered concepts are ranked by TCAV importance.
- **Net2Vec** (Fong & Vedaldi 2018) identifies which individual neurons encode a concept, providing finer-grained insight than TCAV's global direction.
- **Concept completeness** measures whether a set of concepts is sufficient to explain the model's behavior. A completeness score near 1.0 means the concepts capture essentially all decision-relevant information.
- For practical model auditing, test all concepts against all classes to build a comprehensive concept importance matrix and flag potentially problematic associations (e.g., gender or race concepts linked to job classifications).

---

## Exercises

### Exercise 1: TCAV Concept Set Design (Beginner)

For a pre-trained InceptionV3 model, design concept sets for "furry", "metallic", and "natural landscape". Collect 30+ images per concept from the internet. Train CAVs at three different layers (Mixed_5b, Mixed_6a, Mixed_7a) and compare the CAV classification accuracies. Which layer best captures each concept? Why might different concepts be best represented at different layers?

### Exercise 2: Concept Bottleneck Bird Classifier (Intermediate)

Using the CUB-200-2011 dataset (which has 312 binary attribute annotations per image), build a Concept Bottleneck Model for bird species classification. Train with all three strategies (independent, sequential, joint) and compare: (a) concept prediction accuracy, (b) species classification accuracy, and (c) the interpretability of the label predictor's weight matrix. Which strategy gives the best overall tradeoff?

### Exercise 3: Automated Concept Discovery (Intermediate)

Apply ACE to discover the top 10 concepts used by a ResNet-50 to classify "airplane" images from ImageNet. Visualize the exemplar segments for each discovered concept. Can you name what each concept represents? Compute concept completeness: do these 10 concepts fully explain the model's airplane classification?

### Exercise 4: Bias Auditing with TCAV (Advanced)

Audit a face attribute classifier (e.g., CelebA model predicting "attractive") for demographic biases. Create concept sets for "young", "old", "male", "female", "light skin", "dark skin". Run the full concept audit pipeline and generate a report. Identify any statistically significant demographic associations that could indicate bias. Propose mitigation strategies for any bias found.

### Exercise 5: Concept Intervention Study (Advanced)

Build a medical image CBM that predicts pneumonia from chest X-rays using 15 radiological concepts (e.g., "consolidation", "pleural effusion", "cardiomegaly"). Simulate a clinical workflow: (a) the model makes a prediction, (b) a radiologist reviews the predicted concepts and corrects 3 of them, (c) measure how much the corrected prediction improves. Design experiments to determine the minimum number of concept corrections needed to achieve clinician-level accuracy.

---

[Previous: Advanced SHAP](./06_Advanced_SHAP.md) | [Overview](./00_Overview.md) | [Next: Counterfactual Explanations](./08_Counterfactual_Explanations.md)

---

**License**: CC BY-NC 4.0
