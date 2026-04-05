# Control Theory Examples

Control theory example code

## Files

| File | Lesson | Description |
|------|--------|-------------|
| `01_modeling.py` | L02 | Physical system modeling (mass-spring-damper, DC motor, linearization) |
| `02_transfer_functions.py` | L03 | Transfer functions, poles/zeros, block diagram algebra, Mason's formula |
| `03_time_response.py` | L04 | Time-domain response, specification calculation, steady-state error |
| `04_stability.py` | L05 | Routh-Hurwitz criterion, stability ranges |
| `05_root_locus.py` | L06 | Root locus, asymptotes, breakaway points, gain selection |
| `06_bode_nyquist.py` | L07-08 | Bode/Nyquist plots, gain/phase margins |
| `07_pid_control.py` | L09 | PID control, Ziegler-Nichols tuning, anti-windup |
| `08_state_space.py` | L11-12 | State-space, controllability, observability, PBH test |
| `09_state_feedback.py` | L13 | Pole placement, observer design, separation principle |
| `10_lqr_kalman.py` | L14 | LQR, Kalman filter, LQG |
| `11_digital_control.py` | L15 | ZOH discretization, digital PID, Jury test |

## Requirements

- Python 3.9+
- NumPy

## Usage

```bash
python examples/Control_Theory/01_modeling.py
python examples/Control_Theory/07_pid_control.py
# etc.
```

All examples are self-contained and run without external dependencies beyond NumPy.
For visualization, consider adding matplotlib to plot Bode diagrams, root loci, and step responses.
