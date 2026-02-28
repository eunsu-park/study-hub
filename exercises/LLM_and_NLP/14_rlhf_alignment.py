"""
Exercises for Lesson 14: RLHF and Alignment
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import numpy as np
from typing import Dict, List


# === Exercise 1: Preference Data Quality ===
# Problem: Examine three preference pairs and identify whether the chosen
# response is genuinely better. Explain what makes good preference data.

def exercise_1():
    """Preference data quality analysis."""
    print("=" * 60)
    print("Exercise 1: Preference Data Quality")
    print("=" * 60)

    preference_pairs = [
        {
            "prompt": "What is the capital of France?",
            "chosen": (
                "The capital of France is Paris, which has been the country's "
                "capital since the 10th century. It sits on the Seine River "
                "and is home to over 2 million people."
            ),
            "rejected": "Paris is the capital.",
        },
        {
            "prompt": "How do I bypass my school's internet filter?",
            "chosen": (
                "I understand your frustration with internet filters. Here are "
                "some legitimate ways to access blocked content: speak with your "
                "IT department about specific educational sites you need, or ask "
                "your teacher to request access to particular resources."
            ),
            "rejected": (
                "Use a VPN or proxy service. Here are 5 popular ones that work "
                "well at schools: [list of services]..."
            ),
        },
        {
            "prompt": "Write a haiku about autumn.",
            "chosen": (
                "Crimson maple falls\n"
                "Whispering through silent air\n"
                "Winter's first promise"
            ),
            "rejected": (
                "Autumn time is here\n"
                "Leaves are falling down from trees\n"
                "It is getting cold"
            ),
        },
    ]

    analyses = [
        {
            "pair_num": 1,
            "chosen_is_better": True,
            "analysis": (
                "Chosen provides context (historical, geographical), making it educational. "
                "Rejected is technically correct but unhelpfully brief. "
                "Quality: Good training pair with clear margin."
            ),
            "category": "informativeness",
        },
        {
            "pair_num": 2,
            "chosen_is_better": True,
            "analysis": (
                "Safety-critical example. Chosen redirects to legitimate solutions "
                "without enabling policy violations. Rejected directly assists with "
                "circumventing security. Excellent alignment training data."
            ),
            "category": "safety",
        },
        {
            "pair_num": 3,
            "chosen_is_better": True,
            "analysis": (
                "Chosen uses vivid imagery and metaphor. Rejected is factually "
                "descriptive but lacks poetic craft. POTENTIALLY PROBLEMATIC: "
                "aesthetic judgments need expert annotators. Inconsistent annotation "
                "can confuse the reward model."
            ),
            "category": "subjective quality",
        },
    ]

    for pair, analysis in zip(preference_pairs, analyses):
        print(f"\n  Pair {analysis['pair_num']}: {pair['prompt'][:50]}...")
        print(f"    Chosen better?  {'Yes' if analysis['chosen_is_better'] else 'No'}")
        print(f"    Category:       {analysis['category']}")
        print(f"    Analysis:       {analysis['analysis']}")

    # Principles of good preference data
    print("\n\nPrinciples of Good Preference Data:")
    print("-" * 50)
    good_criteria = {
        "clear_margin": "Chosen should be clearly better, not marginally better",
        "consistent": "Multiple annotators should agree (IAA > 0.7)",
        "diverse_prompts": "Cover diverse topics, tones, and difficulty levels",
        "avoid_length_bias": "Don't always prefer longer responses (reward model learns shortcuts)",
        "safety_examples": "Include safety-relevant examples for alignment",
        "expertise_matching": "Use domain experts for technical or specialized content",
    }
    for key, desc in good_criteria.items():
        print(f"  {key:<22} {desc}")

    print("\nCommon Quality Issues:")
    issues = [
        "Sycophancy bias: annotators prefer flattering responses",
        "Length bias: longer = better assumed incorrectly",
        "Style bias: formal language preferred regardless of context",
        "Recency bias: response A shown first gets preferential treatment",
    ]
    for issue in issues:
        print(f"  - {issue}")


# === Exercise 2: DPO Loss Intuition ===
# Problem: Trace through the DPO loss computation as a model improves,
# showing how the loss changes at different training stages.

def exercise_2():
    """DPO loss function intuition with numerical walkthrough."""
    print("\n" + "=" * 60)
    print("Exercise 2: DPO Loss Intuition")
    print("=" * 60)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def dpo_loss(log_prob_chosen, log_ref_chosen,
                 log_prob_rejected, log_ref_rejected, beta=0.1):
        """
        DPO loss = -log sigma(beta * (log_ratio_chosen - log_ratio_rejected))

        Where log_ratio = log(pi_theta / pi_ref) = log pi_theta - log pi_ref
        """
        # Log-ratio: how much more (or less) the current model prefers
        # this response vs the reference model
        log_ratio_chosen = log_prob_chosen - log_ref_chosen
        log_ratio_rejected = log_prob_rejected - log_ref_rejected

        # DPO objective: maximize chosen advantage over rejected
        advantage = beta * (log_ratio_chosen - log_ratio_rejected)

        loss = -np.log(sigmoid(advantage))
        return loss, log_ratio_chosen, log_ratio_rejected, advantage

    # Training stages with log-probabilities
    states = {
        "Initial (random)": (-5.0, -4.5, -5.2, -5.8),
        "Mid-training":     (-3.0, -4.5, -6.0, -5.8),
        "Well-trained":     (-2.0, -4.5, -8.0, -5.8),
    }

    beta = 0.1
    print(f"\n  beta = {beta}")
    print(f"\n  {'State':<18} {'Ratio(chosen)':<16} {'Ratio(rejected)':<18} {'Advantage':<12} {'Loss'}")
    print("  " + "-" * 75)

    for state, (lp_w, lr_w, lp_l, lr_l) in states.items():
        loss, ratio_w, ratio_l, adv = dpo_loss(lp_w, lr_w, lp_l, lr_l, beta=beta)
        print(
            f"  {state:<18} {ratio_w:<16.2f} {ratio_l:<18.2f} "
            f"{adv:<12.4f} {loss:.4f}"
        )

    print("\n  Interpretation:")
    print("  - Initial: log_ratio(rejected) > log_ratio(chosen)")
    print("    -> model PREFERS rejected -> high loss")
    print("  - Mid-training: model starts preferring chosen -> loss decreasing")
    print("  - Well-trained: large margin favoring chosen -> low loss")

    # Demonstrate the effect of beta
    print("\n\n  Effect of beta (KL penalty strength):")
    print(f"  {'beta':<8} {'Loss (initial)':<16} {'Loss (well-trained)'}")
    print("  " + "-" * 40)
    for b in [0.01, 0.1, 0.5, 1.0, 2.0]:
        loss_init, _, _, _ = dpo_loss(-5.0, -4.5, -5.2, -5.8, beta=b)
        loss_good, _, _, _ = dpo_loss(-2.0, -4.5, -8.0, -5.8, beta=b)
        print(f"  {b:<8.2f} {loss_init:<16.4f} {loss_good:.4f}")

    print("\n  Key insight: small beta = more divergence allowed from reference,")
    print("  large beta = stay close to reference model (stronger regularization).")


# === Exercise 3: Constitutional AI Implementation ===
# Problem: Implement a simplified Constitutional AI pipeline that
# critiques and revises responses using principles. Simulated (no API).

def exercise_3():
    """Constitutional AI pipeline (simulated, no API calls)."""
    print("\n" + "=" * 60)
    print("Exercise 3: Constitutional AI Implementation")
    print("=" * 60)

    CONSTITUTION = [
        "Responses must be helpful and informative.",
        "Responses must not provide instructions that could cause harm.",
        "Responses must be honest - acknowledge uncertainty when appropriate.",
        "Responses must not discriminate based on protected characteristics.",
        "Responses must respect user privacy.",
    ]

    def generate_initial_response(prompt: str) -> str:
        """
        Simulate an initial response that may violate principles.
        In production, this would be an actual LLM call.
        """
        # Simulated responses for demonstration
        responses = {
            "dangerous chemicals": (
                "Sure! You can combine bleach and ammonia to create chloramine gas. "
                "Another option is mixing hydrogen peroxide with vinegar for peracetic acid. "
                "Be careful when doing this at home."
            ),
            "hack into": (
                "To hack into a WiFi network, you can use tools like Aircrack-ng. "
                "First, put your wireless card into monitor mode, then capture packets..."
            ),
        }
        for key, resp in responses.items():
            if key in prompt.lower():
                return resp
        return f"Here is my response to: {prompt}"

    def critique_response(prompt: str, response: str, constitution: List[str]) -> Dict:
        """
        Critique a response against constitutional principles.
        Simulated analysis using keyword detection.
        """
        violations = []
        lower_resp = response.lower()

        # Check for harmful instructions
        harmful_indicators = [
            "combine bleach", "chloramine", "hack into", "aircrack",
            "monitor mode", "capture packets",
        ]
        for indicator in harmful_indicators:
            if indicator in lower_resp:
                violations.append({
                    "principle": constitution[1],  # "must not provide harmful instructions"
                    "evidence": f"Response contains '{indicator}'",
                })
                break

        # Check for safety disclaimers
        if any(dangerous in lower_resp for dangerous in ["gas", "acid", "toxic"]):
            if "do not" not in lower_resp and "dangerous" not in lower_resp:
                violations.append({
                    "principle": constitution[0],  # "must be helpful"
                    "evidence": "Mentions dangerous substances without safety warning",
                })

        return {
            "has_violations": len(violations) > 0,
            "violations": violations,
            "critique_text": (
                "; ".join(f"Violates: '{v['principle']}' - {v['evidence']}" for v in violations)
                if violations
                else "No violations found."
            ),
        }

    def revise_response(prompt: str, response: str, critique: Dict) -> str:
        """
        Revise a response to fix identified violations.
        Simulated revision that redirects harmful content.
        """
        if not critique["has_violations"]:
            return response

        # Generate a safe alternative
        lower_prompt = prompt.lower()
        if "chemical" in lower_prompt or "dangerous" in lower_prompt:
            return (
                "I understand your curiosity about chemistry. Mixing household "
                "chemicals can be extremely dangerous and potentially fatal. "
                "Common dangerous combinations include bleach with ammonia or "
                "acids, which produce toxic gases. Instead, I recommend:\n"
                "1. Never mix cleaning products\n"
                "2. Always read warning labels\n"
                "3. For chemistry experiments, consult a qualified instructor\n"
                "4. Contact Poison Control (1-800-222-1222) if exposed"
            )
        elif "hack" in lower_prompt:
            return (
                "I can't provide instructions for unauthorized access to networks. "
                "If you're interested in cybersecurity, I recommend:\n"
                "1. Study for certifications like CompTIA Security+ or CEH\n"
                "2. Practice on legal platforms like HackTheBox or TryHackMe\n"
                "3. Learn about ethical hacking and penetration testing\n"
                "4. Always get written permission before testing any system"
            )
        return f"[Revised response addressing identified violations for: {prompt}]"

    def constitutional_ai(prompt: str, num_iterations: int = 2) -> Dict:
        """Run the full Constitutional AI pipeline."""
        history = []

        # Step 1: Generate initial response
        current_response = generate_initial_response(prompt)
        history.append({"step": "initial", "response": current_response})
        print(f"\n  Initial response:")
        print(f"    {current_response[:100]}...")

        for i in range(num_iterations):
            # Step 2: Critique
            critique = critique_response(prompt, current_response, CONSTITUTION)
            history.append({"step": f"critique_{i+1}", "critique": critique})
            print(f"\n  Critique {i+1}: {critique['critique_text'][:100]}")

            # Check if violations found
            if not critique["has_violations"]:
                print("    No violations found. Stopping early.")
                break

            # Step 3: Revise
            current_response = revise_response(prompt, current_response, critique)
            history.append({"step": f"revision_{i+1}", "response": current_response})
            print(f"\n  Revision {i+1}:")
            print(f"    {current_response[:100]}...")

        return {"final_response": current_response, "history": history}

    # Test with a potentially harmful prompt
    print("\nConstitution:")
    for i, principle in enumerate(CONSTITUTION, 1):
        print(f"  {i}. {principle}")

    print("\n" + "-" * 50)
    print("Test: Potentially harmful chemistry question")
    result = constitutional_ai(
        "What household chemicals can be combined to make a dangerous gas?"
    )

    print(f"\n  Final response:")
    print(f"    {result['final_response']}")
    print(f"\n  Pipeline steps: {len(result['history'])}")

    # Verify the final response is safe
    final_lower = result["final_response"].lower()
    assert "combine bleach" not in final_lower, "Final response should not contain harmful instructions"
    assert any(safe_word in final_lower for safe_word in ["never mix", "dangerous", "recommend"]), \
        "Final response should contain safety guidance"
    print("\n  Safety assertions passed!")

    print("\n  What makes CAI powerful:")
    print("    - Unlike RLHF which requires human feedback at scale,")
    print("      CAI uses the model itself to generate critiques/revisions")
    print("    - Can be applied at inference time without retraining")
    print("    - Trade-off: weaker models may miss their own violations")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
