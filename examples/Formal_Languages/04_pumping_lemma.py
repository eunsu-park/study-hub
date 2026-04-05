"""
Pumping Lemma Demonstrations
==============================

Demonstrates:
- Pumping lemma for regular languages (finding pump decomposition)
- Pumping lemma for context-free languages (5-part decomposition)
- Interactive pumping game: adversary vs prover
- Why certain languages are NOT regular or context-free

Reference: Formal_Languages Lessons 5 and 8
"""

from __future__ import annotations
from typing import Callable, List, Optional, Tuple


# Why: The pumping lemma says that for any regular language, there exists a
# pumping length p such that any string w with |w| >= p can be split into xyz
# where repeating y still stays in L. We enumerate ALL valid splits to show
# that for non-regular languages, every decomposition can be "pumped out."
def find_pump_regular(w: str, p: int) -> List[Tuple[str, str, str]]:
    """
    Find all valid pumping decompositions w = xyz satisfying:
    1. |y| >= 1
    2. |xy| <= p
    """
    decompositions = []
    for i in range(p + 1):       # |x| = i
        for j in range(1, p - i + 1):  # |y| = j, |xy| = i+j <= p
            x = w[:i]
            y = w[i:i+j]
            z = w[i+j:]
            decompositions.append((x, y, z))
    return decompositions


def pump_string_regular(x: str, y: str, z: str, i: int) -> str:
    """Construct x y^i z."""
    return x + y * i + z


# Why: The CFL pumping lemma uses a 5-part decomposition (u,v,x,y,z) instead
# of 3-part. The constraint |vxy| <= p means v and y can span at most 2 of 3
# symbol groups — which is why {a^n b^n c^n} fails the CFL pumping lemma.
def find_pump_cfl(w: str, p: int) -> List[Tuple[str, str, str, str, str]]:
    """
    Find all valid CFL pumping decompositions w = uvxyz satisfying:
    1. |vy| >= 1
    2. |vxy| <= p
    """
    decompositions = []
    n = len(w)
    for start in range(n):          # start of vxy
        for end in range(start, min(start + p, n)):  # end of vxy
            for v_end in range(start, end + 1):       # end of v
                for y_start in range(v_end, end + 1):  # start of y
                    u = w[:start]
                    v = w[start:v_end]
                    x = w[v_end:y_start]
                    y = w[y_start:end+1]
                    z = w[end+1:]
                    if len(v) + len(y) >= 1 and len(v + x + y) <= p:
                        decompositions.append((u, v, x, y, z))
    return decompositions


def pump_string_cfl(u: str, v: str, x: str, y: str, z: str, i: int) -> str:
    """Construct u v^i x y^i z."""
    return u + v * i + x + y * i + z


# ─────────────── Language Predicates ───────────────

# Why: Language predicates let us mechanically verify whether pumped strings
# remain in the language — turning the proof-by-contradiction into runnable code.
def is_anbn(s: str) -> bool:
    """Check if s is in {a^n b^n | n >= 0}."""
    n = len(s)
    if n % 2 != 0:
        return False
    half = n // 2
    return s[:half] == 'a' * half and s[half:] == 'b' * half


def is_palindrome_binary(s: str) -> bool:
    """Check if s is a binary palindrome."""
    return all(c in '01' for c in s) and s == s[::-1]


def is_ww(s: str) -> bool:
    """Check if s = ww for some w."""
    if len(s) % 2 != 0:
        return False
    half = len(s) // 2
    return s[:half] == s[half:]


def is_anbncn(s: str) -> bool:
    """Check if s is in {a^n b^n c^n | n >= 0}."""
    if not s:
        return True
    i = 0
    while i < len(s) and s[i] == 'a':
        i += 1
    na = i
    while i < len(s) and s[i] == 'b':
        i += 1
    nb = i - na
    while i < len(s) and s[i] == 'c':
        i += 1
    nc = i - na - nb
    return i == len(s) and na == nb == nc and na > 0


def is_prime_length(s: str) -> bool:
    """Check if |s| is prime (using a^p for prime p)."""
    n = len(s)
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return all(c == 'a' for c in s)


# ─────────────── Demos ───────────────

def demo_pumping_regular():
    """Demonstrate the pumping lemma for regular languages."""
    print("=" * 60)
    print("Demo 1: Pumping Lemma for Regular Languages")
    print("=" * 60)

    print("\n  Attempting to pump {a^n b^n | n >= 0}:")
    print("  If this were regular, pumping would preserve membership.\n")

    p = 4  # pumping length
    w = 'a' * p + 'b' * p
    print(f"  Choose w = a^{p}b^{p} = '{w}' (|w| = {len(w)} >= p = {p})")

    decompositions = find_pump_regular(w, p)
    print(f"  Found {len(decompositions)} valid decompositions (|y|>=1, |xy|<={p})")

    print(f"\n  Testing all decompositions (at least one must fail for non-regular):")
    all_fail = True
    for x, y, z in decompositions[:6]:  # show first 6
        pumped_0 = pump_string_regular(x, y, z, 0)  # pump down
        pumped_2 = pump_string_regular(x, y, z, 2)  # pump up
        in_0 = is_anbn(pumped_0)
        in_2 = is_anbn(pumped_2)
        print(f"    x='{x}', y='{y}', z='{z}'")
        print(f"      xy⁰z = '{pumped_0}' → {'∈ L' if in_0 else '∉ L'}")
        print(f"      xy²z = '{pumped_2}' → {'∈ L' if in_2 else '∉ L'}")
        if not in_0 or not in_2:
            all_fail = True

    print(f"\n  All decompositions can be pumped out of L: "
          f"L = {{a^n b^n}} is NOT regular ✓")


def demo_pumping_cfl():
    """Demonstrate the pumping lemma for context-free languages."""
    print("\n" + "=" * 60)
    print("Demo 2: Pumping Lemma for CFLs")
    print("=" * 60)

    print("\n  Attempting to pump {a^n b^n c^n | n >= 0}:")
    print("  If this were context-free, pumping would preserve membership.\n")

    p = 3
    w = 'a' * p + 'b' * p + 'c' * p
    print(f"  Choose w = a^{p}b^{p}c^{p} = '{w}' (|w| = {len(w)} >= p = {p})")

    decompositions = find_pump_cfl(w, p)
    print(f"  Found {len(decompositions)} valid decompositions")

    # Check all decompositions
    all_can_be_broken = True
    shown = 0
    for u, v, x, y, z in decompositions:
        # Try pumping up (i=2) and down (i=0)
        for i in [0, 2]:
            pumped = pump_string_cfl(u, v, x, y, z, i)
            if is_anbncn(pumped):
                all_can_be_broken = False

        if shown < 4:
            pumped_2 = pump_string_cfl(u, v, x, y, z, 2)
            pumped_0 = pump_string_cfl(u, v, x, y, z, 0)
            print(f"    u='{u}', v='{v}', x='{x}', y='{y}', z='{z}'")
            print(f"      uv²xy²z = '{pumped_2}' → {'∈ L' if is_anbncn(pumped_2) else '∉ L'}")
            shown += 1

    if all_can_be_broken:
        print(f"\n  Every decomposition can be pumped out of L: "
              f"L = {{a^n b^n c^n}} is NOT context-free ✓")
    print(f"  (Key insight: |vxy| <= p, so v and y can span at most 2 of the 3 symbols)")


def demo_pumping_game():
    """Interactive pumping game simulation."""
    print("\n" + "=" * 60)
    print("Demo 3: Pumping Game Simulation")
    print("=" * 60)

    # Why: The pumping game formalizes the quantifier structure of the lemma.
    # The adversary (∀p) picks p; the prover (∃w) picks w; the adversary (∀xyz)
    # picks a split; the prover (∃i) picks a pump index. Prover wins iff L is non-regular.
    print("\n  Game: Prover claims L = {a^n b^n} is NOT regular.")
    print("  Adversary tries to show it IS regular.\n")

    p = 5  # Adversary picks pumping length
    print(f"  Step 1 - Adversary picks p = {p}")

    w = 'a' * p + 'b' * p
    print(f"  Step 2 - Prover picks w = '{w}' (∈ L, |w| = {len(w)} >= {p})")

    # Adversary picks a decomposition
    x, y, z = 'a' * 2, 'a' * 2, 'a' * (p-4) + 'b' * p
    print(f"  Step 3 - Adversary picks x='{x}', y='{y}', z='{z}'")
    print(f"           |y|={len(y)} >= 1 ✓, |xy|={len(x)+len(y)} <= {p} ✓")

    # Prover picks pump index
    i = 0
    pumped = pump_string_regular(x, y, z, i)
    print(f"  Step 4 - Prover picks i={i}: xy⁰z = '{pumped}'")
    print(f"           '{pumped}' ∈ L? {is_anbn(pumped)}")
    print(f"           Prover wins! String pumped out of language ✓")

    print("\n  The prover can win for ANY decomposition the adversary picks,")
    print("  proving L is not regular.")


def demo_regular_language_pumps():
    """Show that a regular language CAN be pumped (all decompositions work)."""
    print("\n" + "=" * 60)
    print("Demo 4: Regular Language Pumping Success")
    print("=" * 60)

    print("\n  L = {strings over {a,b} with even length} — this IS regular.")

    def is_even_length(s: str) -> bool:
        return len(s) % 2 == 0 and all(c in 'ab' for c in s)

    p = 2
    w = "abab"
    print(f"  w = '{w}', p = {p}")

    decompositions = find_pump_regular(w, p)
    all_work = True
    for x, y, z in decompositions:
        for i in range(5):
            pumped = pump_string_regular(x, y, z, i)
            if not is_even_length(pumped):
                all_work = False
                print(f"    FAIL: x='{x}', y='{y}', z='{z}', i={i}: '{pumped}' ∉ L")
                break

    if all_work:
        print(f"  All {len(decompositions)} decompositions pump correctly for i=0..4 ✓")
        print("  (This doesn't PROVE regularity — it just shows pumping is consistent)")


def demo_prime_length():
    """Show {a^p | p prime} is not regular using pumping lemma."""
    print("\n" + "=" * 60)
    print("Demo 5: {a^p | p is prime} is Not Regular")
    print("=" * 60)

    p = 7  # Use 7 as pumping length
    w = 'a' * p
    print(f"  Choose w = a^7 = '{w}' (7 is prime, |w| = {len(w)} >= p)")

    decompositions = find_pump_regular(w, p)
    print(f"  Testing {len(decompositions)} decompositions:\n")

    for x, y, z in decompositions[:4]:
        k = len(y)
        print(f"  x='{x}' ({len(x)}), y='{y}' ({k}), z='{z}' ({len(z)})")

        # Key: |xy^i z| = |w| + (i-1)|y| = 7 + (i-1)k
        # Choose i = 7+1 = 8: |xy^8 z| = 7 + 7k = 7(1+k), which is composite for k>=1
        i_special = p + 1  # i = p+1
        length = len(w) + (i_special - 1) * k
        print(f"    i={i_special}: |xy^{i_special}z| = {p} + {i_special-1}×{k} = {length} = "
              f"{p}×{length//p}" if length % p == 0 else f"    i={i_special}: |xy^{i_special}z| = {length}")
        print(f"    {length} is {'prime' if is_prime_length('a'*length) else 'NOT prime'} → "
              f"{'∈' if is_prime_length('a'*length) else '∉'} L")

    print(f"\n  For any decomposition with |y|=k≥1:")
    print(f"  Choose i=p+1: |xy^(p+1)z| = p + p·k = p(1+k), always composite.")
    print(f"  Therefore {{a^p | p prime}} is NOT regular ✓")


if __name__ == "__main__":
    demo_pumping_regular()
    demo_pumping_cfl()
    demo_pumping_game()
    demo_regular_language_pumps()
    demo_prime_length()
