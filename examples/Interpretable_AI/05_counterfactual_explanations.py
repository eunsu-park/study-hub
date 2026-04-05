"""
05. Counterfactual Explanations

Generates counterfactual explanations for a tabular classifier: "What minimal
changes to the input would flip the model's decision?"  Two approaches are
implemented -- a gradient-based optimizer following Wachter et al. (2017) and
the DiCE library (Mothilal et al. 2020) -- together with quality metrics and
actionability constraints.

Covered topics:
    - Wachter et al. counterfactual generation via gradient descent
    - DiCE integration using dice-ml library
    - Quality metrics: proximity, sparsity, plausibility
    - Actionability constraint checking (immutable / directional features)
    - End-to-end loan-approval dataset example

Related to: L08 - Counterfactual Explanations

Requirements:
    pip install torch numpy scikit-learn dice-ml pandas
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ====== Section 1: Synthetic Loan-Approval Dataset ======

def create_loan_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic loan-approval dataset with realistic feature distributions.

    The approval decision follows a logistic boundary so that the dataset
    is non-trivially separable -- gradient-based counterfactual search has
    something meaningful to work with.
    """
    np.random.seed(seed)

    # Each feature has a real-world interpretation that matters for
    # actionability constraints later (e.g. age is immutable).
    age = np.random.normal(35, 10, n_samples).clip(18, 70)
    income = np.random.exponential(50, n_samples).clip(10, 300)
    debt_ratio = np.random.beta(2, 5, n_samples)
    credit_score = np.random.normal(650, 80, n_samples).clip(300, 850)
    employment = np.random.exponential(5, n_samples).clip(0, 40)

    # Deterministic label with a soft boundary (logistic function).
    # Coefficients are chosen so that ~40-60% of applicants are approved,
    # giving a balanced binary classification task.
    logit = (
        0.02 * (credit_score - 600)
        + 0.03 * income
        - 3.0 * debt_ratio
        + 0.1 * employment
        - 2.0
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    approved = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "debt_ratio": debt_ratio,
        "credit_score": credit_score,
        "employment": employment,
        "approved": approved,
    })
    return df


FEATURE_NAMES = ["age", "income", "debt_ratio", "credit_score", "employment"]


# ====== Section 2: Loan-Approval Classifier ======

class LoanClassifier(nn.Module):
    """Two-hidden-layer binary classifier for loan approval.

    Architecture is deliberately small -- the point of this example is
    the *explanation* method, not the model's predictive power.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    epochs: int = 80,
) -> LoanClassifier:
    """Train the loan classifier with BCE loss and Adam optimizer.

    Returns the model in eval mode, ready for counterfactual generation.
    """
    model = LoanClassifier(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ====== Section 3: Wachter et al. Counterfactual Generator ======

def generate_wachter_counterfactual(
    model: nn.Module,
    x_orig: np.ndarray,
    target_class: int,
    scaler: StandardScaler,
    lam: float = 0.1,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 0.01,
) -> dict:
    """Wachter et al. (2017) counterfactual via gradient descent.

    Minimises:  L_pred(f(x'), target) + lambda * ||x' - x||^2

    The prediction loss pushes x' toward the desired class; the distance
    penalty keeps it close to the original instance.  We work in scaled
    feature space so that gradients are well-conditioned across features
    with vastly different magnitudes (e.g. income vs debt_ratio).

    Returns a dict with the counterfactual (in original feature space),
    the number of iterations, and final loss components.
    """
    model.eval()

    # Work in scaled space for numerically stable optimization
    x_sc = scaler.transform(x_orig.reshape(1, -1)).flatten()
    x_cf = torch.tensor(x_sc, dtype=torch.float32, requires_grad=True)
    x_ref = torch.tensor(x_sc, dtype=torch.float32)
    target_t = torch.tensor(float(target_class))

    opt = torch.optim.Adam([x_cf], lr=lr)

    for i in range(max_iter):
        opt.zero_grad()

        # Prediction loss: BCE pushes the model output toward target_class
        logit = model(x_cf.unsqueeze(0))
        pred_loss = nn.BCEWithLogitsLoss()(logit.squeeze(), target_t)

        # Proximity loss: L2 penalty keeps the CF close to the original
        dist_loss = lam * torch.sum((x_cf - x_ref) ** 2)

        loss = pred_loss + dist_loss
        loss.backward()
        opt.step()

        # Early stopping once prediction has flipped and loss is small
        with torch.no_grad():
            prob = torch.sigmoid(model(x_cf.unsqueeze(0))).item()
            predicted = int(prob > 0.5)
            if predicted == target_class and pred_loss.item() < tol:
                break

    # Map back to original feature space for human-readable output
    cf_original = scaler.inverse_transform(
        x_cf.detach().numpy().reshape(1, -1)
    ).flatten()

    return {
        "counterfactual": cf_original,
        "iterations": i + 1,
        "pred_loss": pred_loss.item(),
        "dist_loss": dist_loss.item(),
        "final_prob": prob,
    }


# ====== Section 4: Quality Metrics ======

def compute_cf_metrics(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    X_train: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Evaluate a counterfactual on three quality axes.

    1. **Proximity** -- L1 and L2 distance in normalised space.
       Smaller is better; the CF should be close to the original.

    2. **Sparsity** -- fraction of features that changed.
       Fewer changes are easier for humans to understand and act on.

    3. **Plausibility** -- distance to the nearest training example.
       The CF should look like a *real* data point, not an alien one.
    """
    # Normalise by training range so all features contribute equally
    ranges = X_train.max(axis=0) - X_train.min(axis=0)
    ranges = np.where(ranges == 0, 1, ranges)
    normed_diff = np.abs(x_cf - x_orig) / ranges

    l1 = float(np.sum(normed_diff))
    l2 = float(np.sqrt(np.sum(normed_diff ** 2)))

    # Sparsity: count features that moved by more than 1% of their range
    changed = normed_diff > 0.01
    sparsity = float(changed.sum()) / len(feature_names)

    # Plausibility: Euclidean distance to nearest training point
    dists = np.sqrt(((X_train - x_cf) ** 2 / ranges ** 2).sum(axis=1))
    plausibility = float(dists.min())

    return {
        "proximity_l1": round(l1, 4),
        "proximity_l2": round(l2, 4),
        "sparsity": round(sparsity, 4),
        "plausibility_nn_dist": round(plausibility, 4),
    }


# ====== Section 5: Actionability Constraints ======

# Real-world constraints on loan features:
#   age          -- immutable  (applicant cannot change their age)
#   income       -- increase-only
#   debt_ratio   -- decrease-only
#   credit_score -- increase-only
#   employment   -- increase-only
ACTIONABILITY = {
    "age": "immutable",
    "income": "increase_only",
    "debt_ratio": "decrease_only",
    "credit_score": "increase_only",
    "employment": "increase_only",
}


def check_actionability(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    feature_names: list[str],
    constraints: dict[str, str],
    tolerance: float = 1e-3,
) -> dict:
    """Verify whether a counterfactual respects actionability constraints.

    Constraint types:
      - "immutable"      : feature must not change at all
      - "increase_only"  : feature may only increase (e.g. income)
      - "decrease_only"  : feature may only decrease (e.g. debt ratio)
      - "any"            : no restriction

    Returns a dict mapping each feature to a (passed, violation_msg) tuple.
    """
    results = {}
    for i, name in enumerate(feature_names):
        delta = x_cf[i] - x_orig[i]
        constraint = constraints.get(name, "any")

        if constraint == "immutable" and abs(delta) > tolerance:
            results[name] = (False, f"changed by {delta:+.2f} (immutable)")
        elif constraint == "increase_only" and delta < -tolerance:
            results[name] = (False, f"decreased by {delta:.2f} (increase-only)")
        elif constraint == "decrease_only" and delta > tolerance:
            results[name] = (False, f"increased by {delta:+.2f} (decrease-only)")
        else:
            results[name] = (True, "OK")

    return results


def generate_actionable_counterfactual(
    model: nn.Module,
    x_orig: np.ndarray,
    target_class: int,
    scaler: StandardScaler,
    feature_names: list[str],
    constraints: dict[str, str],
    lam: float = 0.1,
    lr: float = 0.01,
    max_iter: int = 1500,
) -> dict:
    """Extended Wachter search with projection to enforce actionability.

    After each gradient step, immutable features are clamped to their
    original value; directional features are clipped.  This guarantees
    the final counterfactual respects all constraints, at the cost of
    a potentially longer search (or no solution if constraints are too tight).
    """
    model.eval()

    x_sc = scaler.transform(x_orig.reshape(1, -1)).flatten()
    x_cf = torch.tensor(x_sc, dtype=torch.float32, requires_grad=True)
    x_ref = torch.tensor(x_sc, dtype=torch.float32)
    target_t = torch.tensor(float(target_class))

    # Precompute masks for efficient projection
    x_orig_sc = x_sc.copy()
    immutable_mask = [constraints.get(n, "any") == "immutable" for n in feature_names]
    increase_mask = [constraints.get(n, "any") == "increase_only" for n in feature_names]
    decrease_mask = [constraints.get(n, "any") == "decrease_only" for n in feature_names]

    opt = torch.optim.Adam([x_cf], lr=lr)

    for step in range(max_iter):
        opt.zero_grad()

        logit = model(x_cf.unsqueeze(0))
        pred_loss = nn.BCEWithLogitsLoss()(logit.squeeze(), target_t)
        dist_loss = lam * torch.sum((x_cf - x_ref) ** 2)
        loss = pred_loss + dist_loss
        loss.backward()
        opt.step()

        # Projection step: enforce constraints in scaled space
        with torch.no_grad():
            for i in range(len(feature_names)):
                if immutable_mask[i]:
                    x_cf[i] = x_orig_sc[i]
                elif increase_mask[i] and x_cf[i].item() < x_orig_sc[i]:
                    x_cf[i] = x_orig_sc[i]
                elif decrease_mask[i] and x_cf[i].item() > x_orig_sc[i]:
                    x_cf[i] = x_orig_sc[i]

        with torch.no_grad():
            prob = torch.sigmoid(model(x_cf.unsqueeze(0))).item()
            if int(prob > 0.5) == target_class and pred_loss.item() < 0.01:
                break

    cf_orig_space = scaler.inverse_transform(
        x_cf.detach().numpy().reshape(1, -1)
    ).flatten()

    return {
        "counterfactual": cf_orig_space,
        "iterations": step + 1,
        "final_prob": prob,
    }


# ====== Section 6: DiCE Integration ======

def run_dice_explanations(
    model: nn.Module,
    scaler: StandardScaler,
    df: pd.DataFrame,
    x_query: np.ndarray,
    X_train: np.ndarray,
) -> None:
    """Generate diverse counterfactuals using DiCE library.

    DiCE (Mothilal et al. 2020) produces *multiple* diverse CFs
    simultaneously, giving users several actionable options rather
    than a single point explanation.
    """
    try:
        import dice_ml
    except ImportError:
        print("[INFO] dice-ml not installed -- skipping DiCE demo.")
        print("       Install with: pip install dice-ml")
        return

    # DiCE needs a pandas-level data description
    dice_data = dice_ml.Data(
        dataframe=df,
        continuous_features=FEATURE_NAMES,
        outcome_name="approved",
    )

    # Wrap our PyTorch model with the sklearn-compatible interface DiCE expects
    class DiceModelWrapper:
        """Thin wrapper to make our PyTorch model compatible with DiCE."""

        def __init__(self, model, scaler, feature_names):
            self.model = model
            self.scaler = scaler
            self.feature_names = feature_names

        def predict(self, X):
            if isinstance(X, pd.DataFrame):
                X = X[self.feature_names].values.astype(np.float32)
            X_sc = self.scaler.transform(X)
            with torch.no_grad():
                logits = self.model(torch.tensor(X_sc))
                return (torch.sigmoid(logits) > 0.5).int().numpy()

        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                X = X[self.feature_names].values.astype(np.float32)
            X_sc = self.scaler.transform(X)
            with torch.no_grad():
                logits = self.model(torch.tensor(X_sc))
                p1 = torch.sigmoid(logits).numpy()
                return np.column_stack([1 - p1, p1])

    wrapper = DiceModelWrapper(model, scaler, FEATURE_NAMES)

    dice_model = dice_ml.Model(
        model=wrapper,
        backend="sklearn",
        model_type="classifier",
    )

    explainer = dice_ml.Dice(dice_data, dice_model, method="random")

    try:
        query_instance = pd.DataFrame([x_query], columns=FEATURE_NAMES)
        dice_result = explainer.generate_counterfactuals(
            query_instance,
            total_CFs=4,
            desired_class="opposite",
        )
        print("DiCE counterfactuals:")
        dice_result.visualize_as_dataframe(show_only_changes=True)

        # Compute quality metrics for each generated CF
        cf_list = dice_result.cf_examples_list[0].final_cfs_df
        print(f"\n{len(cf_list)} diverse counterfactuals generated.")
        for i, row in cf_list.iterrows():
            cf_vals = row[FEATURE_NAMES].values.astype(float)
            m = compute_cf_metrics(x_query, cf_vals, X_train, FEATURE_NAMES)
            print(f"  CF-{i}: proximity_l2={m['proximity_l2']:.3f}  "
                  f"sparsity={m['sparsity']:.2f}  "
                  f"plausibility={m['plausibility_nn_dist']:.3f}")
    except Exception as e:
        print(f"[INFO] DiCE demo encountered an issue: {e}")
        print("       This is expected in some environments.")


# ====== Section 7: Main Pipeline ======

def main() -> None:
    """Run the full counterfactual explanation pipeline."""
    print("=" * 65)
    print("  Counterfactual Explanations")
    print("  Wachter et al. | Actionability Constraints | DiCE")
    print("=" * 65)

    # --- Step 1: Generate dataset ---
    print("\n[1] Generating Synthetic Loan-Approval Dataset")
    print("-" * 50)
    df = create_loan_dataset(n_samples=2000)
    print(f"  Samples: {len(df)}  |  Approved: {df['approved'].sum()} "
          f"({df['approved'].mean():.1%})")
    print(df[FEATURE_NAMES].describe().round(2).to_string())

    # --- Step 2: Train classifier ---
    print("\n[2] Training Loan-Approval Classifier")
    print("-" * 50)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["approved"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = train_classifier(X_train_sc, y_train, n_features=len(FEATURE_NAMES))

    with torch.no_grad():
        test_logits = model(torch.tensor(X_test_sc))
        preds = (torch.sigmoid(test_logits) > 0.5).float()
        acc = (preds == torch.tensor(y_test)).float().mean().item()
    print(f"  Test accuracy: {acc:.2%}")

    # --- Step 3: Wachter counterfactual ---
    print("\n[3] Wachter et al. -- Gradient-Based Counterfactual Search")
    print("-" * 50)

    denied_indices = np.where(y_test == 0)[0]
    if len(denied_indices) == 0:
        print("  No denied applicants in test set -- skipping demo.")
        return

    idx = denied_indices[0]
    x_query = X_test[idx]

    print(f"  Original instance (denied):")
    for name, val in zip(FEATURE_NAMES, x_query):
        print(f"    {name:15s}: {val:.2f}")

    result = generate_wachter_counterfactual(
        model, x_query, target_class=1, scaler=scaler, lam=0.1,
    )
    cf = result["counterfactual"]

    print(f"\n  Counterfactual (approved):")
    for name, val in zip(FEATURE_NAMES, cf):
        print(f"    {name:15s}: {val:.2f}")
    print(f"  Iterations: {result['iterations']}  |  P(approved) = {result['final_prob']:.3f}")

    # Show which features changed the most
    deltas = cf - x_query
    print("\n  Feature changes:")
    for name, orig, delta in zip(FEATURE_NAMES, x_query, deltas):
        if abs(delta) > 1e-3:
            print(f"    {name:15s}: {orig:8.2f} -> {orig + delta:8.2f}  (delta={delta:+.2f})")

    # --- Step 4: Quality metrics ---
    print("\n[4] Counterfactual Quality Metrics")
    print("-" * 50)

    metrics = compute_cf_metrics(x_query, cf, X_train, FEATURE_NAMES)
    print("  Metrics for the Wachter counterfactual:")
    for k, v in metrics.items():
        print(f"    {k:25s}: {v}")

    # --- Step 5: Actionability constraints ---
    print("\n[5] Actionability Constraint Checking")
    print("-" * 50)

    print("  Unconstrained counterfactual -- actionability check:")
    checks = check_actionability(x_query, cf, FEATURE_NAMES, ACTIONABILITY)
    for name, (passed, msg) in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name:15s}: {msg}")

    # Generate a constrained (actionable) CF
    print("\n  Generating actionable counterfactual ...")
    act_result = generate_actionable_counterfactual(
        model, x_query, target_class=1, scaler=scaler,
        feature_names=FEATURE_NAMES, constraints=ACTIONABILITY,
    )
    act_cf = act_result["counterfactual"]
    print(f"  Actionable CF:")
    for name, val in zip(FEATURE_NAMES, act_cf):
        print(f"    {name:15s}: {val:.2f}")
    print(f"  P(approved) = {act_result['final_prob']:.3f}  |  "
          f"Iterations: {act_result['iterations']}")

    checks_act = check_actionability(x_query, act_cf, FEATURE_NAMES, ACTIONABILITY)
    all_pass = all(v[0] for v in checks_act.values())
    print(f"\n  All constraints satisfied: {all_pass}")
    for name, (passed, msg) in checks_act.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name:15s}: {msg}")

    # Compare quality metrics
    print("\n  Quality comparison (unconstrained vs actionable):")
    m_unc = compute_cf_metrics(x_query, cf, X_train, FEATURE_NAMES)
    m_act = compute_cf_metrics(x_query, act_cf, X_train, FEATURE_NAMES)
    print(f"    {'Metric':25s} {'Unconstrained':>15s} {'Actionable':>15s}")
    for key in m_unc:
        print(f"    {key:25s} {m_unc[key]:15.4f} {m_act[key]:15.4f}")

    # --- Step 6: DiCE ---
    print("\n[6] DiCE -- Diverse Counterfactual Explanations")
    print("-" * 50)
    run_dice_explanations(model, scaler, df, x_query, X_train)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print("""
  Counterfactual explanations answer: "What would need to change
  for the model to give a different prediction?"

  Key takeaways:
    1. Wachter et al. -- simple gradient-based approach: minimise
       prediction loss + distance penalty.
    2. Actionability constraints (immutable / directional) make
       counterfactuals realistic and actionable.
    3. Quality is measured on three axes: proximity, sparsity,
       plausibility.
    4. DiCE produces *diverse* sets of counterfactuals, giving
       users multiple actionable options.
    5. Trade-off: tighter constraints may make it harder (or
       impossible) to find a valid counterfactual.
    """)


if __name__ == "__main__":
    main()
