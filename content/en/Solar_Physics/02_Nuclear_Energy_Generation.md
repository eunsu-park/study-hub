# Nuclear Energy Generation

[← Previous: Solar Interior](01_Solar_Interior.md) | [Next: Helioseismology →](03_Helioseismology.md)

## Learning Objectives

1. Derive the thermonuclear reaction rate from the Coulomb barrier tunneling probability and the Maxwell-Boltzmann distribution
2. Calculate the Gamow peak energy and its width for the proton-proton reaction
3. Describe all three branches of the pp chain, their branching ratios, and the neutrinos produced
4. Explain the CNO cycle and its temperature dependence relative to the pp chain
5. Outline the solar neutrino problem and its resolution through neutrino oscillations
6. Calculate the energy generation rate and its dependence on temperature and density

---

## Why This Matters

The Sun shines because nuclear fusion converts hydrogen into helium in its core, releasing energy according to Einstein's $E = mc^2$. This process powers all life on Earth and has been operating steadily for 4.57 billion years. Understanding *how* fusion works in the Sun — the specific reaction sequences, their rates, and the resulting neutrino fluxes — is one of the great triumphs of twentieth-century physics, bridging nuclear physics, quantum mechanics, and astrophysics.

The story of solar energy generation is also the story of the **solar neutrino problem**: for three decades, experiments detected far fewer neutrinos from the Sun than theory predicted. The resolution — neutrino oscillations, requiring neutrinos to have mass — earned the 2002 and 2015 Nobel Prizes and fundamentally changed particle physics. Solar neutrinos remain our most direct probe of conditions in the solar core.

> **Analogy**: Nuclear fusion in the Sun is like a ball rolling over a hill in classical physics — except the ball is too slow to reach the top. Quantum tunneling is the miracle that lets the ball "pass through" the hill with a tiny probability. Because there are so many protons ($\sim 10^{57}$) colliding so frequently ($\sim 10^{37}$ collisions per second per proton), even a fantastically small tunneling probability per collision is enough to power the Sun.

---

## 1. Thermonuclear Reactions

### The Coulomb Barrier

Two protons repel each other via the Coulomb force. To fuse, they must approach closely enough ($r \sim 1$ fm $= 10^{-15}$ m) for the strong nuclear force to take over. The height of the Coulomb barrier for two nuclei with charges $Z_1 e$ and $Z_2 e$ is:

$$E_\text{Coulomb} = \frac{Z_1 Z_2 e^2}{4\pi\epsilon_0 r_0} \approx \frac{Z_1 Z_2 \times 1.44 \text{ MeV·fm}}{r_0 \text{ (fm)}}$$

For two protons ($Z_1 = Z_2 = 1$) approaching to $r_0 \sim 1$ fm:

$$E_\text{Coulomb} \sim 1.4 \text{ MeV}$$

But the average thermal energy of protons at the Sun's core temperature ($T_c = 1.57 \times 10^7$ K) is:

$$\langle E \rangle = \frac{3}{2} k_B T_c \approx \frac{3}{2} \times 8.617 \times 10^{-5} \text{ eV/K} \times 1.57 \times 10^7 \text{ K} \approx 2.0 \text{ keV}$$

This is nearly **1000 times too small** to overcome the Coulomb barrier classically! Fusion is possible only because of **quantum mechanical tunneling**.

### Tunneling Probability

The probability of tunneling through the Coulomb barrier is given by the Gamow factor:

$$P_\text{tunnel}(E) \propto \exp\left(-\sqrt{\frac{E_G}{E}}\right)$$

where $E_G$ is the **Gamow energy**:

$$\boxed{E_G = 2 m_r c^2 (\pi Z_1 Z_2 \alpha)^2}$$

Here $m_r$ is the reduced mass of the two nuclei, $c$ is the speed of light, and $\alpha \approx 1/137$ is the fine-structure constant.

For the p-p reaction ($Z_1 = Z_2 = 1$, $m_r = m_p/2$):

$$E_G = 2 \times \frac{m_p c^2}{2} \times (\pi \times 1 \times 1 \times \frac{1}{137})^2 = m_p c^2 \times \left(\frac{\pi}{137}\right)^2$$

$$E_G \approx 938.3 \text{ MeV} \times 5.25 \times 10^{-4} = 493 \text{ keV}$$

At $E = 2$ keV (typical thermal energy), $\sqrt{E_G/E} \approx \sqrt{493/2} \approx 15.7$, giving $P_\text{tunnel} \sim e^{-15.7} \sim 1.5 \times 10^{-7}$. The tunneling probability is extremely small but not zero.

### The Cross-Section

The nuclear reaction cross-section is conventionally written as:

$$\sigma(E) = \frac{S(E)}{E} \exp\left(-\sqrt{\frac{E_G}{E}}\right)$$

where:
- $1/E$ accounts for the quantum mechanical de Broglie wavelength ($\sigma \propto \lambda^2 \propto 1/E$)
- The exponential is the Gamow (tunneling) factor
- $S(E)$ is the **astrophysical S-factor**, which contains all the nuclear physics. For reactions governed by the strong force alone, $S(E)$ varies slowly with energy. The beauty of this parameterization is that the rapidly varying parts (barrier penetration and de Broglie scaling) are factored out, leaving $S(E)$ as a nearly constant function that can be measured in the laboratory and extrapolated to stellar energies.

For the p-p reaction, the S-factor is extraordinarily small: $S(0) \approx 4.01 \times 10^{-25}$ MeV·barn, because this reaction requires a simultaneous weak interaction (one proton must convert to a neutron via $p \to n + e^+ + \nu_e$).

---

## 2. The Gamow Peak

### Derivation

The thermonuclear reaction rate per unit volume is:

$$r = n_1 n_2 \langle \sigma v \rangle$$

where $\langle \sigma v \rangle$ is the thermally averaged product of cross-section and relative velocity:

$$\langle \sigma v \rangle = \left(\frac{8}{\pi m_r}\right)^{1/2} \frac{1}{(k_B T)^{3/2}} \int_0^\infty S(E) \exp\left(-\frac{E}{k_B T} - \sqrt{\frac{E_G}{E}}\right) dE$$

The integrand is a product of two competing exponentials:

- $\exp(-E/k_BT)$: the Maxwell-Boltzmann distribution, which **decreases** with increasing energy (fewer particles have high energy)
- $\exp(-\sqrt{E_G/E})$: the Gamow tunneling factor, which **increases** with increasing energy (higher energy particles tunnel more easily)

The product has a sharp peak — the **Gamow peak** — at the energy where these two trends optimally balance.

### Gamow Peak Energy

Differentiating the exponent $f(E) = -E/(k_BT) - \sqrt{E_G/E}$ and setting $f'(E_0) = 0$:

$$-\frac{1}{k_B T} + \frac{1}{2}\sqrt{\frac{E_G}{E_0^3}} = 0$$

Solving for $E_0$:

$$\boxed{E_0 = \left(\frac{E_G (k_B T)^2}{4}\right)^{1/3}}$$

This is the most probable energy at which reactions occur — neither the most energetic particles (too few) nor the most common particles (can't tunnel).

For the p-p reaction at $T = 1.57 \times 10^7$ K ($k_BT = 1.35$ keV):

$$E_0 = \left(\frac{493 \times 1.35^2}{4}\right)^{1/3} \text{ keV} = \left(\frac{493 \times 1.82}{4}\right)^{1/3} \text{ keV} = (224)^{1/3} \text{ keV} \approx 6.1 \text{ keV}$$

> **Key insight**: Reactions occur at $E_0 \approx 6$ keV, which is about $4.5 \, k_BT$ — well above the average thermal energy but far below the Coulomb barrier ($\sim 1.4$ MeV). The Gamow peak sits in the tail of the Maxwell-Boltzmann distribution where tunneling is efficient enough to matter.

### Width of the Gamow Peak

The effective width of the peak (treating the exponent as a Gaussian near $E_0$) is:

$$\Delta E_0 = \frac{4}{\sqrt{3}} (E_0 k_B T)^{1/2}$$

For the p-p reaction: $\Delta E_0 \approx \frac{4}{\sqrt{3}} (6.1 \times 1.35)^{1/2} \approx 6.6$ keV.

The Gamow peak is remarkably narrow: the effective energy "window" for nuclear reactions spans roughly $6.1 \pm 3.3$ keV, a tiny sliver of the overall energy distribution. This narrow window explains the extreme temperature sensitivity of nuclear reaction rates.

### Temperature Sensitivity

Near the Gamow peak, the reaction rate can be shown to scale as a power law in temperature:

$$\langle \sigma v \rangle \propto T^{n} \quad \text{where} \quad n = \frac{\tau - 2}{3}, \quad \tau = 3\frac{E_0}{k_B T}$$

For the p-p reaction: $\tau = 3 \times 6.1/1.35 \approx 13.6$, giving $n \approx 3.9$. So $\epsilon_{pp} \propto T^4$ approximately. A 10% increase in temperature increases the pp reaction rate by about 50%.

---

## 3. The pp Chain

### Overview

The net result of the pp chain is:

$$4p \to {}^4\text{He} + 2e^+ + 2\nu_e + \text{energy}$$

The mass deficit gives the energy release:

$$Q = [4m_p - m_\alpha - 2m_e]c^2 = 26.73 \text{ MeV}$$

Of this, about 2% ($\sim 0.59$ MeV on average) is carried away by neutrinos and lost from the star. The remaining $\sim 26.14$ MeV is deposited as thermal energy.

The pp chain has three branches, named pp-I, pp-II, and pp-III. All branches share the same first two steps.

### Common First Steps

**Step 1** — The pp reaction (rate-limiting):

$$p + p \to d + e^+ + \nu_e \qquad Q = 1.442 \text{ MeV}$$

This is extraordinarily slow because it requires a simultaneous weak interaction ($p \to n + e^+ + \nu_e$). The cross-section is $\sim 10^{20}$ times smaller than a typical strong-interaction cross-section. The average proton in the Sun waits about $\sim 10^{10}$ years before undergoing this reaction — most protons alive today will outlive the Sun.

The neutrino from this reaction is called a **pp neutrino** and has a continuous spectrum with maximum energy $E_\nu = 0.42$ MeV.

There is also a rare variant:

$$p + e^- + p \to d + \nu_e \qquad Q = 1.442 \text{ MeV}$$

This **pep reaction** produces a monoenergetic neutrino at $E_\nu = 1.44$ MeV. It occurs only 0.4% as often as the pp reaction.

**Step 2** — Deuterium burning (very fast):

$$d + p \to {}^3\text{He} + \gamma \qquad Q = 5.493 \text{ MeV}$$

This is a strong-interaction process and occurs within seconds of deuterium formation. Deuterium never accumulates.

### pp-I Branch (Dominant: ~86%)

$${}^3\text{He} + {}^3\text{He} \to {}^4\text{He} + 2p \qquad Q = 12.860 \text{ MeV}$$

This completes the chain. Two ${}^3$He nuclei (from two separate instances of steps 1-2) combine to form ${}^4$He plus two protons, which return to the pool.

**Energy bookkeeping for pp-I**:
- 2 × (pp reaction): $2 \times 1.442 = 2.884$ MeV
- 2 × (d+p reaction): $2 \times 5.493 = 10.986$ MeV
- 1 × (${}^3$He + ${}^3$He): 12.860 MeV
- Total: 26.73 MeV
- Neutrino losses: $2 \times 0.265 = 0.53$ MeV (average pp neutrino energy)

### pp-II Branch (~14%)

Instead of two ${}^3$He nuclei finding each other, one ${}^3$He captures a pre-existing ${}^4$He:

$${}^3\text{He} + {}^4\text{He} \to {}^7\text{Be} + \gamma \qquad Q = 1.586 \text{ MeV}$$

Then ${}^7$Be captures an electron:

$${}^7\text{Be} + e^- \to {}^7\text{Li} + \nu_e \qquad Q = 0.862 \text{ MeV}$$

The **${}^7$Be neutrino** is monoenergetic at either 0.862 MeV (90%) or 0.384 MeV (10%), depending on whether ${}^7$Li is left in its ground state or first excited state.

Finally, ${}^7$Li is destroyed:

$${}^7\text{Li} + p \to 2 \, {}^4\text{He} \qquad Q = 17.347 \text{ MeV}$$

### pp-III Branch (~0.02%)

In rare cases, ${}^7$Be captures a proton instead of an electron:

$${}^7\text{Be} + p \to {}^8\text{B} + \gamma \qquad Q = 0.137 \text{ MeV}$$

${}^8$B is unstable and undergoes beta-plus decay:

$${}^8\text{B} \to {}^8\text{Be}^* + e^+ + \nu_e \qquad Q = 17.98 \text{ MeV}$$

The **${}^8$B neutrino** has a continuous spectrum extending to 14.06 MeV — by far the highest-energy pp-chain neutrino. These are the neutrinos detected by the Homestake, Kamiokande, and SNO experiments.

${}^8$Be$^*$ immediately splits:

$${}^8\text{Be}^* \to 2 \, {}^4\text{He}$$

### Summary of pp Chain Neutrinos

| Source | Energy | Spectrum | Flux at Earth (cm$^{-2}$s$^{-1}$) |
|--------|--------|----------|----------------------------------|
| pp | $\leq 0.42$ MeV | Continuous | $5.98 \times 10^{10}$ |
| pep | 1.44 MeV | Line | $1.44 \times 10^{8}$ |
| ${}^7$Be | 0.862 / 0.384 MeV | Lines | $5.00 \times 10^{9}$ |
| ${}^8$B | $\leq 14.06$ MeV | Continuous | $5.58 \times 10^{6}$ |
| hep (${}^3$He+p) | $\leq 18.77$ MeV | Continuous | $7.98 \times 10^{3}$ |

> **Physical insight**: Although pp-III contributes only 0.02% of the solar luminosity, the high-energy ${}^8$B neutrinos it produces are the easiest to detect — they interact much more readily with matter. This is why early solar neutrino experiments (sensitive only to high-energy neutrinos) were so important despite probing such a minor branch.

---

## 4. The CNO Cycle

### Overview

The **CNO cycle** (also called the CNO bi-cycle) uses carbon, nitrogen, and oxygen as catalysts to convert hydrogen to helium. The dominant branch (CN cycle) is:

$${}^{12}\text{C} + p \to {}^{13}\text{N} + \gamma$$
$${}^{13}\text{N} \to {}^{13}\text{C} + e^+ + \nu_e \qquad (t_{1/2} = 10 \text{ min})$$
$${}^{13}\text{C} + p \to {}^{14}\text{N} + \gamma$$
$${}^{14}\text{N} + p \to {}^{15}\text{O} + \gamma \qquad \text{(slowest step)}$$
$${}^{15}\text{O} \to {}^{15}\text{N} + e^+ + \nu_e \qquad (t_{1/2} = 2 \text{ min})$$
$${}^{15}\text{N} + p \to {}^{12}\text{C} + {}^4\text{He}$$

The net reaction is again $4p \to {}^4\text{He} + 2e^+ + 2\nu_e + \text{energy}$, with $Q = 25.03$ MeV (slightly different neutrino energy loss than pp chain).

### Temperature Sensitivity

The rate-limiting step is ${}^{14}\text{N}(p,\gamma){}^{15}\text{O}$, which has a much higher Coulomb barrier ($Z_1 Z_2 = 7$) than the pp reaction ($Z_1 Z_2 = 1$). The Gamow energy scales as $(Z_1 Z_2)^2$, giving a much steeper temperature dependence:

$$\epsilon_\text{CNO} \propto T^{16-20}$$

compared to $\epsilon_\text{pp} \propto T^4$. This means:

- At $T < 1.7 \times 10^7$ K: pp chain dominates (including in the Sun)
- At $T > 1.7 \times 10^7$ K: CNO cycle dominates (in stars with $M \gtrsim 1.3 M_\odot$)

In the Sun, the CNO cycle contributes only about **1%** of the total luminosity. However, recent measurements by the Borexino experiment (2020) have directly detected CNO neutrinos from the Sun for the first time, confirming this contribution and providing a direct measurement of the Sun's core metallicity.

### Why CNO Matters for the Sun

Even though the CNO cycle is a minor energy source, it is important for several reasons:

1. **Core metallicity probe**: CNO neutrino flux depends on C and N abundances in the core, offering a way to test the solar abundance problem independently of helioseismology
2. **Composition evolution**: CNO processing converts most of the initial ${}^{12}$C, ${}^{13}$C, ${}^{14}$N, ${}^{15}$N, ${}^{16}$O into ${}^{14}$N (the slowest step creates a bottleneck), changing the relative abundances
3. **Other stars**: For more massive stars, the CNO cycle dominates and creates a convective core (due to the extreme temperature sensitivity concentrating energy generation in a very small region)

---

## 5. Energy Generation Rate

### Power-Law Approximation

The energy generation rate (power per unit mass) can be approximated as a power law:

$$\epsilon = \epsilon_0 \rho^{\lambda} T^{\nu}$$

For the pp chain:

$$\epsilon_\text{pp} \approx \epsilon_0 \rho T^4 \quad (\lambda = 1, \; \nu \approx 4)$$

For the CNO cycle:

$$\epsilon_\text{CNO} \approx \epsilon_0 \rho T^{16-20} \quad (\lambda = 1, \; \nu \approx 16\text{-}20)$$

The linear dependence on $\rho$ reflects the binary nature of the reactions (reaction rate $\propto n_1 n_2 \propto \rho^2$, and $\epsilon = \text{rate} \times E / \rho$, so $\epsilon \propto \rho$).

### Luminosity Integral

The total luminosity is:

$$L = \int_0^{R_\odot} 4\pi r^2 \rho(r) \, \epsilon\big(\rho(r), T(r)\big) \, dr$$

Because of the steep temperature dependence of $\epsilon$, energy generation is strongly concentrated in the core:

- **50%** of $L_\odot$ is generated within the inner $0.10 R_\odot$
- **90%** of $L_\odot$ is generated within the inner $0.24 R_\odot$
- Beyond $0.30 R_\odot$, energy generation is negligible

This extreme concentration means that the luminosity $L(r)$ is essentially constant throughout the radiative and convective zones — all the energy is produced in the core and simply flows outward.

### Energy Generation in Equilibrium

In thermal equilibrium (which the Sun is, to an excellent approximation), the total nuclear energy generation rate equals the luminosity:

$$L_\odot = \int_0^{R_\odot} 4\pi r^2 \rho \epsilon \, dr = 3.828 \times 10^{26} \text{ W}$$

The mass consumption rate is:

$$\dot{M} = \frac{L_\odot}{c^2} \approx \frac{3.828 \times 10^{26}}{(3 \times 10^8)^2} \approx 4.26 \times 10^9 \text{ kg/s}$$

The Sun converts about **4.26 million tonnes of matter into energy every second**. However, most of this mass is returned as helium — the actual mass loss is the binding energy difference. The hydrogen consumption rate is about $6.2 \times 10^{11}$ kg/s, or $\sim 6 \times 10^{17}$ kg per 30 million years.

---

## 6. The Solar Neutrino Problem

### The Prediction

John Bahcall and collaborators, using the Standard Solar Model, predicted the flux of solar neutrinos reaching Earth. The total electron neutrino flux from all pp-chain reactions is approximately $6.6 \times 10^{10}$ cm$^{-2}$s$^{-1}$.

For the Homestake chlorine experiment, which was sensitive primarily to ${}^8$B and ${}^7$Be neutrinos:

$${}^{37}\text{Cl} + \nu_e \to {}^{37}\text{Ar} + e^-$$

Bahcall predicted a capture rate of $\sim 8$ SNU (Solar Neutrino Units, where 1 SNU = 1 capture per $10^{36}$ target atoms per second).

### The Observation

Ray Davis's Homestake experiment (1968-1998), using 100,000 gallons of perchloroethylene ($\text{C}_2\text{Cl}_4$) deep in the Homestake gold mine, measured:

$$R_\text{obs} = 2.56 \pm 0.16 \pm 0.16 \text{ SNU}$$

This was roughly **one-third** of the predicted rate. The discrepancy persisted for three decades and was confirmed by multiple experiments:

- **Kamiokande / Super-Kamiokande** (Japan): Water Cherenkov detectors, sensitive to ${}^8$B neutrinos via $\nu_e + e^- \to \nu_e + e^-$. Measured $\sim 45\%$ of the SSM prediction.
- **SAGE / GALLEX / GNO**: Gallium experiments, sensitive to pp neutrinos (${}^{71}\text{Ga} + \nu_e \to {}^{71}\text{Ge} + e^-$). Measured $\sim 55\%$ of prediction.

The energy-dependent deficit (different reduction factors for different neutrino energies) was a crucial clue.

### The Resolution: Neutrino Oscillations

The solution came from the **Sudbury Neutrino Observatory (SNO)** in Canada (2001-2002). SNO used heavy water ($\text{D}_2\text{O}$) and could measure three different reactions:

1. **Charged Current (CC)**: $\nu_e + d \to p + p + e^-$ — sensitive only to electron neutrinos ($\nu_e$)
2. **Neutral Current (NC)**: $\nu_x + d \to p + n + \nu_x$ — sensitive to ALL neutrino flavors ($\nu_e$, $\nu_\mu$, $\nu_\tau$)
3. **Elastic Scattering (ES)**: $\nu_x + e^- \to \nu_x + e^-$ — sensitive to all flavors, but with reduced cross-section for $\nu_\mu$, $\nu_\tau$

The landmark result:

- **CC flux** (only $\nu_e$): $1.76 \times 10^6$ cm$^{-2}$s$^{-1}$ — about 1/3 of SSM
- **NC flux** (all flavors): $5.09 \times 10^6$ cm$^{-2}$s$^{-1}$ — **consistent with SSM**!

The total neutrino flux was correct — the "missing" electron neutrinos had **oscillated** into muon and tau neutrinos during their journey from the core to Earth.

### The MSW Effect

Neutrino oscillations in the Sun are enhanced by the **Mikheyev-Smirnov-Wolfenstein (MSW) effect**: electron neutrinos interact with the electrons in the solar interior via charged-current interactions, giving them an effective potential:

$$V = \sqrt{2} G_F n_e$$

where $G_F$ is the Fermi constant and $n_e$ is the electron number density. This potential modifies the oscillation probability, and for high-energy neutrinos (${}^8$B), it can cause **resonant conversion** from $\nu_e$ to $\nu_\mu$ or $\nu_\tau$ as the neutrino propagates through the density gradient of the Sun.

The survival probability for electron neutrinos depends on their energy:

- **Low energy** ($E \lesssim 1$ MeV, pp neutrinos): vacuum oscillations dominate. $P(\nu_e \to \nu_e) \approx 1 - \frac{1}{2}\sin^2(2\theta_{12}) \approx 0.55$
- **High energy** ($E \gtrsim 5$ MeV, ${}^8$B neutrinos): MSW effect dominates. $P(\nu_e \to \nu_e) \approx \sin^2\theta_{12} \approx 0.31$

The transition between these regimes occurs in the 1-5 MeV range. This energy-dependent survival probability precisely explains why different experiments (sensitive to different energy ranges) measured different deficit fractions.

### Borexino: The Complete Picture

The Borexino experiment (Gran Sasso, Italy) has detected neutrinos from nearly every branch of the pp chain:

| Source | First detection | Borexino measurement |
|--------|----------------|---------------------|
| ${}^7$Be | 2007 | $48.3 \pm 1.1$ counts/(day·100 ton) |
| ${}^8$B | 2008 | Low-energy threshold measurement |
| pep | 2012 | $3.1 \pm 0.6$ counts/(day·100 ton) |
| pp | 2014 | $144 \pm 13$ counts/(day·100 ton) |
| CNO | 2020 | $7.2^{+3.0}_{-1.7}$ counts/(day·100 ton) |

Borexino's detection of CNO neutrinos was a milestone — the first direct evidence of the CNO cycle operating in any star.

---

## 7. Nuclear Timescales

### Hydrogen Burning Lifetime

The total nuclear energy available from converting all core hydrogen to helium:

$$E_\text{nuc} = 0.007 \times f_\text{core} \times M_\odot c^2$$

where $0.007$ is the fraction of rest mass converted to energy ($4m_p - m_\alpha = 0.007 \times 4m_p$) and $f_\text{core} \approx 0.10$ is the fraction of the Sun's mass in the core that will eventually be processed. The hydrogen-burning (main-sequence) lifetime is:

$$t_\text{MS} = \frac{E_\text{nuc}}{L_\odot} = \frac{0.007 \times 0.10 \times M_\odot c^2}{L_\odot}$$

$$t_\text{MS} \approx \frac{0.007 \times 0.10 \times (1.989 \times 10^{30})(3 \times 10^8)^2}{3.828 \times 10^{26}} \approx 3.3 \times 10^{17} \text{ s} \approx 10^{10} \text{ years}$$

The Sun has been on the main sequence for 4.57 Gyr and has roughly 5 Gyr remaining.

### Why the pp Reaction is the Bottleneck

The pp reaction ($p + p \to d + e^+ + \nu_e$) is the slowest step because:

1. It requires a **weak interaction** (converting a proton to a neutron) simultaneously with the nuclear collision
2. The weak interaction cross-section is $\sim 10^{-44}$ cm$^2$, compared to $\sim 10^{-24}$ cm$^2$ for strong interactions
3. The average proton-proton collision rate is $\sim 10^{37}$ s$^{-1}$ per proton, but the probability of the weak process occurring during each collision is $\sim 10^{-20}$

The resulting characteristic timescale for a single proton to participate in the pp reaction is:

$$\tau_{pp} \sim 10^{10} \text{ years}$$

This remarkably long timescale — comparable to the age of the Sun — is what makes the Sun a stable, long-lived star rather than a bomb. If the pp reaction were governed by the strong force alone (no weak interaction required), the Sun would have exhausted its fuel in seconds.

---

## Summary

- Nuclear fusion in the Sun overcomes the Coulomb barrier through **quantum tunneling**, described by the Gamow factor $\exp(-\sqrt{E_G/E})$
- Reactions occur primarily at the **Gamow peak** energy $E_0 = (E_G k_B^2 T^2 / 4)^{1/3} \approx 6$ keV for the pp reaction — far above average thermal energy but far below the Coulomb barrier
- The **pp chain** converts 4 protons to ${}^4$He in three branches (pp-I: 86%, pp-II: 14%, pp-III: 0.02%), releasing 26.73 MeV per ${}^4$He and producing neutrinos at characteristic energies
- The **CNO cycle** is a minor contributor (~1%) in the Sun but dominates in more massive stars, with $\epsilon_\text{CNO} \propto T^{16-20}$ vs $\epsilon_\text{pp} \propto T^4$
- The **solar neutrino problem** (observed flux 1/3 of prediction) was resolved by **neutrino oscillations** (confirmed by SNO in 2001), requiring neutrinos to have mass
- The pp reaction is the bottleneck ($\tau \sim 10^{10}$ years per proton) because it requires a simultaneous weak interaction, making the Sun a stable, long-lived energy source

---

## Practice Problems

### Problem 1: Gamow Peak

Calculate the Gamow peak energy $E_0$ and width $\Delta E_0$ for the reaction ${}^{12}\text{C}(p,\gamma){}^{13}\text{N}$ at $T = 1.5 \times 10^7$ K. Use $Z_1 = 1$, $Z_2 = 6$, and $m_r = m_p \times 12/13$. Compare $E_0$ with the pp Gamow peak. Why is the CNO Gamow peak at higher energy?

### Problem 2: pp Chain Energy Bookkeeping

(a) Show that the net energy release for the pp-I chain is 26.73 MeV by adding up the Q-values of all reactions (remember that two pp reactions and two d+p reactions are needed for each ${}^3$He+${}^3$He reaction). (b) Of this 26.73 MeV, how much is carried away by neutrinos (average) and how much heats the Sun? (c) What fraction of the original proton rest mass energy ($4 m_p c^2$) is converted to energy?

### Problem 3: Solar Luminosity from First Principles

The pp reaction rate is approximately $r_{pp} = 3.09 \times 10^{-37} \rho^2 X^2 T_6^{-2/3} \exp(-33.80/T_6^{1/3})$ reactions cm$^{-3}$ s$^{-1}$, where $T_6 = T/(10^6$ K$)$. Assuming uniform core conditions ($T = 1.5 \times 10^7$ K, $\rho = 100$ g/cm$^3$, $X = 0.5$), estimate the luminosity produced within a sphere of radius $0.1 R_\odot$. Compare with $L_\odot$. (Each pp-I chain releases $\sim 26$ MeV and requires 2 pp reactions.)

### Problem 4: CNO vs pp Crossover Temperature

The ratio of CNO to pp energy generation rates is approximately:

$$\frac{\epsilon_\text{CNO}}{\epsilon_\text{pp}} \propto T^{13} \times \frac{Z_\text{CNO}}{X}$$

At solar center conditions ($T = 1.57 \times 10^7$ K), this ratio is about 0.01. At what temperature does $\epsilon_\text{CNO} = \epsilon_\text{pp}$? This crossover temperature determines whether a star has a convective or radiative core.

### Problem 5: Neutrino Oscillation Survival Probability

The electron neutrino survival probability in vacuum is $P(\nu_e \to \nu_e) = 1 - \sin^2(2\theta_{12}) \sin^2(\Delta m_{21}^2 L / 4E_\nu)$, where $\theta_{12} \approx 33.4°$ and $\Delta m_{21}^2 = 7.53 \times 10^{-5}$ eV$^2$. (a) For $L = 1$ AU and $E_\nu = 0.3$ MeV (pp neutrino), calculate $\Delta m_{21}^2 L / (4 E_\nu)$ in natural units. (b) Since $L \gg$ the oscillation length, the $\sin^2$ term averages to 1/2. Calculate the averaged survival probability. (c) Compare with the Borexino measurement. (d) Why does the MSW effect matter for ${}^8$B neutrinos but not for pp neutrinos?

---

[← Previous: Solar Interior](01_Solar_Interior.md) | [Next: Helioseismology →](03_Helioseismology.md)
