# Calculus and Differential Equations

## Introduction

Calculus is the mathematical language of change. While algebra describes static relationships, calculus provides the tools to analyze quantities that vary continuously -- the trajectory of a spacecraft, the growth of a population, the flow of heat through a material, or the fluctuation of stock prices. Differential equations extend this power by expressing relationships between functions and their rates of change, forming the backbone of mathematical modeling across nearly every branch of science and engineering.

This course builds from the foundational concepts of limits and derivatives through integration techniques, and then advances into ordinary and partial differential equations. Whether you are studying physics, data science, machine learning, or engineering, these ideas will appear repeatedly as the mathematical substrate beneath the surface.

## Learning Objectives

By the end of this course, you will be able to:

1. **Explain** the formal definition of limits and continuity using the epsilon-delta framework
2. **Compute** derivatives using differentiation rules and apply them to optimization problems
3. **Evaluate** definite and indefinite integrals using substitution, by-parts, and partial fractions
4. **Apply** integration to calculate areas, volumes, arc lengths, and physical quantities
5. **Analyze** the convergence of infinite sequences and series using standard tests
6. **Derive** Taylor and Maclaurin series for common functions and bound approximation error
7. **Solve** first-order ODEs using separation of variables, integrating factors, and exact equations
8. **Solve** second-order linear ODEs with constant coefficients (homogeneous and non-homogeneous)
9. **Model** real-world phenomena using differential equations (population dynamics, circuits, mechanics)
10. **Apply** Laplace transforms to solve initial value problems
11. **Classify** and solve basic partial differential equations (heat, wave, Laplace)
12. **Implement** numerical methods for integration and ODE solving using Python (NumPy, SciPy, SymPy)

## Prerequisites

- **High school algebra and trigonometry**: comfort with algebraic manipulation, functions, and basic trigonometric identities
- **Basic Python programming**: variables, loops, functions, and basic plotting (see [Programming](../Programming/00_Overview.md) topic)
- **Recommended**: Familiarity with NumPy arrays (see [Data Science](../Data_Science/00_Overview.md) L01-L03)

## Required Libraries

```bash
pip install numpy scipy sympy matplotlib
```

| Library | Purpose |
|---------|---------|
| **NumPy** | Numerical arrays, basic math operations |
| **SciPy** | Numerical integration (`scipy.integrate`), ODE solvers |
| **SymPy** | Symbolic differentiation, integration, equation solving |
| **Matplotlib** | Visualization of functions, convergence, phase portraits |

## Course Outline

| # | Filename | Title | Description |
|---|----------|-------|-------------|
| 00 | `00_Overview.md` | Course Overview | Introduction, prerequisites, study path |
| 01 | `01_Limits_and_Continuity.md` | Limits and Continuity | Epsilon-delta definition, limit laws, continuity, IVT |
| 02 | `02_Derivatives_Fundamentals.md` | Derivatives Fundamentals | Difference quotient, differentiation rules, chain rule |
| 03 | `03_Applications_of_Derivatives.md` | Applications of Derivatives | Optimization, related rates, L'Hopital, Taylor polynomials |
| 04 | `04_Integration_Fundamentals.md` | Integration Fundamentals | Riemann sums, FTC, antiderivatives |
| 05 | `05_Integration_Techniques.md` | Integration Techniques | Substitution, by-parts, partial fractions, improper integrals |
| 06 | `06_Applications_of_Integration.md` | Applications of Integration | Volumes, arc length, surface area, physical applications |
| 07 | `07_Sequences_and_Series.md` | Sequences and Series | Convergence tests, power series, Taylor series |
| 08 | `08_Parametric_and_Polar.md` | Parametric Curves and Polar Coordinates | Parametric equations, polar curves, area and arc length |
| 09 | `09_Multivariable_Functions.md` | Multivariable Functions | Partial derivatives, gradients, directional derivatives |
| 10 | `10_Multiple_Integrals.md` | Multiple Integrals | Double and triple integrals, change of variables |
| 11 | `11_Vector_Calculus.md` | Vector Calculus | Line integrals, surface integrals, Green/Stokes/Divergence theorems |
| 12 | `12_First_Order_ODE.md` | First-Order ODEs | Separable, linear, exact, Bernoulli equations |
| 13 | `13_Second_Order_ODE.md` | Second-Order ODEs | Homogeneous, non-homogeneous, undetermined coefficients, variation of parameters |
| 14 | `14_Systems_of_ODE.md` | Systems of ODEs | Matrix methods, phase portraits, stability analysis |
| 15 | `15_Laplace_Transform_for_ODE.md` | Laplace Transform for ODE | Transform pairs, inverse transforms, solving IVPs |
| 16 | `16_Power_Series_Solutions.md` | Power Series Solutions of ODE | Power series method, Frobenius method, Bessel/Legendre |
| 17 | `17_Introduction_to_PDE.md` | Introduction to PDEs | Classification, heat equation, wave equation, Laplace equation |
| 18 | `18_Fourier_Series_and_PDE.md` | Fourier Series and PDE | Fourier coefficients, Sturm-Liouville, separation of variables |
| 19 | `19_Numerical_Methods_for_DE.md` | Numerical Methods for DE | Euler, Runge-Kutta, adaptive step, stiff systems |
| 20 | `20_Applications_and_Modeling.md` | Applications and Modeling | Population dynamics, mechanical systems, electrical circuits, coupled models |

## Study Path

```
Phase 1: Foundations of Calculus (Lessons 01-04)
  Limits --> Derivatives --> Integration basics
       |
Phase 2: Techniques and Applications (Lessons 05-08)
  Integration techniques --> Applications --> Series
  --> Parametric & Polar
       |
Phase 3: Multivariable and Vector Calculus (Lessons 09-11)
  Multivariable functions --> Multiple integrals --> Vector Calculus
       |
Phase 4: Ordinary Differential Equations (Lessons 12-16)
  First-order --> Second-order --> Systems
  --> Laplace transforms --> Power series solutions
       |
Phase 5: PDEs and Numerical Methods (Lessons 17-19)
  Introduction to PDEs --> Fourier series --> Numerical methods
       |
Phase 6: Applications (Lesson 20)
  Population dynamics, circuits, mechanical systems, coupled models
```

**Recommended pace**: 1-2 lessons per week, with practice problems completed before moving on. Phase 1 should be mastered before advancing, as all later material builds on derivatives and integrals.

## Connections to Other Topics

This course has deep connections to several other topics in the study materials:

| Related Topic | Connection |
|---------------|------------|
| [Mathematical Methods](../Mathematical_Methods/00_Overview.md) | Extends into Fourier analysis (L06), ODEs/PDEs (L07-L08), special functions (L09), complex analysis (L11), Green's functions (L13), variational calculus (L14) |
| [Numerical Simulation](../Numerical_Simulation/00_Overview.md) | Applies numerical ODE/PDE solvers from this course to physical simulations |
| [Math for AI](../Math_for_AI/00_Overview.md) | Uses derivatives (backpropagation), gradients (optimization), and matrix calculus throughout |
| [Deep Learning](../Deep_Learning/00_Overview.md) | Gradient-based optimization, loss function landscapes, automatic differentiation |
| [Signal Processing](../Signal_Processing/00_Overview.md) | Fourier series/transforms, convolution, differential equation models for LTI systems |
| [Plasma Physics](../Plasma_Physics/00_Overview.md) | MHD equations, transport theory, and wave equations are all PDEs |
| [Control Theory](../Control_Theory/00_Overview.md) | Transfer functions, Laplace transforms, state-space representations |

## References

### Textbooks
- **Stewart, J.** *Calculus: Early Transcendentals*, 9th Ed. (Cengage, 2020) -- comprehensive undergraduate reference
- **Zill, D.G.** *Advanced Engineering Mathematics*, 7th Ed. (Jones & Bartlett, 2022) -- ODEs/PDEs with applications
- **Strang, G.** *Calculus* (Wellesley-Cambridge Press) -- freely available at [ocw.mit.edu](https://ocw.mit.edu/ans7870/resources/Strang/Edited/Calculus/Calculus.pdf)
- **Tenenbaum, M. & Pollard, H.** *Ordinary Differential Equations* (Dover) -- classic, affordable reference

### Online Resources
- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) -- outstanding visual intuition
- [MIT OCW 18.01 Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)
- [MIT OCW 18.03 Differential Equations](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/)
- [Paul's Online Math Notes](https://tutorial.math.lamar.edu/) -- excellent worked examples
- [Khan Academy Calculus](https://www.khanacademy.org/math/calculus-1)

---

[Next: Limits and Continuity](./01_Limits_and_Continuity.md)
