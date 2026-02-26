# Control Theory Examples

제어 이론 예제 코드 / Control theory example code

## Files

| File | Lesson | Description |
|------|--------|-------------|
| `01_modeling.py` | L02 | 물리 시스템 모델링 (질량-스프링-댐퍼, DC 모터, 선형화) / Physical system modeling |
| `02_transfer_functions.py` | L03 | 전달 함수, 극점/영점, 블록 선도 대수, 메이슨 공식 / Transfer functions, block diagrams |
| `03_time_response.py` | L04 | 시간 영역 응답, 사양 계산, 정상 상태 오차 / Time-domain analysis, specifications |
| `04_stability.py` | L05 | 라우스-허위츠 판별법, 안정도 범위 / Routh-Hurwitz criterion, stability ranges |
| `05_root_locus.py` | L06 | 근궤적, 점근선, 이탈점, 이득 선정 / Root locus computation and gain design |
| `06_bode_nyquist.py` | L07-08 | 보드/나이퀴스트 선도, 이득/위상 여유 / Bode/Nyquist plots, stability margins |
| `07_pid_control.py` | L09 | PID 제어, 지글러-니콜스 동조, 안티와인드업 / PID tuning, anti-windup |
| `08_state_space.py` | L11-12 | 상태 공간, 가제어성, 가관측성, PBH 검정 / State-space, controllability, observability |
| `09_state_feedback.py` | L13 | 극배치, 관측기 설계, 분리 원리 / Pole placement, observer, separation principle |
| `10_lqr_kalman.py` | L14 | LQR, 칼만 필터, LQG / Optimal control, Kalman filter |
| `11_digital_control.py` | L15 | ZOH 이산화, 디지털 PID, 주리 검정 / Digital control, discretization |

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
