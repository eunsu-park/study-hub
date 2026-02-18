# 신호 처리 - 개요

## 소개

신호 처리(Signal Processing)는 신호를 분석하고, 변형하고, 합성하기 위한 수학적 체계입니다. 신호란 물리적 현상의 특성이나 행동에 관한 정보를 전달하는 양을 말합니다. 음성 인식과 오디오 공학에서부터 레이더 시스템과 의료 영상에 이르기까지, 신호 처리는 현대 기술을 구동하는 근본적인 도구를 제공합니다.

이 강좌는 신호와 시스템에 대한 연속시간 및 이산시간 기술로 시작하여, 주요 변환 영역 기법(푸리에(Fourier), Z-변환, 웨이블릿(wavelet))을 거치고, 디지털 필터의 설계와 구현을 다루며, 스펙트럼 추정, 적응 필터링(adaptive filtering), 영상 처리, 통신 등의 응용까지 체계적으로 학습합니다.

각 레슨은 엄밀한 수학적 유도와 NumPy, SciPy, Matplotlib을 사용한 Python 구현을 함께 제시하여, 개념을 직접 계산하고 시각화할 수 있도록 합니다.

---

## 학습 목표

이 강좌를 마치면 다음을 할 수 있습니다:

1. **신호 분류 및 특성화**: 신호(연속/이산, 결정론적/확률적, 에너지/전력)와 시스템(LTI, 인과적, 안정적) 분류
2. **변환 영역 분석 적용**: 푸리에 급수(Fourier series), 푸리에 변환(Fourier transform), DFT/FFT, Z-변환 활용
3. **디지털 필터 설계 및 구현**: 특정 주파수 응답 사양에 맞는 FIR 및 IIR 디지털 필터 설계
4. **표본화 이론 이해**: 나이퀴스트 정리(Nyquist theorem), 앨리어싱(aliasing), 복원 이해
5. **스펙트럼 추정 수행**: 고전적 방법 및 매개변수적 방법 활용
6. **적응 필터 구현**: 실시간 신호 처리 작업에 적용
7. **비정상 신호 분석**: 시간-주파수 기법(STFT, 웨이블릿) 활용
8. **신호 처리 응용**: 오디오, 영상, 통신, 레이더 등 실용적 영역 적용

---

## 선수 지식

| 주제 | 학습 위치 | 필요한 이유 |
|-------|---------------|------------|
| 미적분학 (단변수 및 다변수) | 대학 미적분학 강의 | 적분, 미분, 극한이 전반에 걸쳐 사용됨 |
| 선형대수(Linear algebra) | [Mathematical_Methods L03](../Mathematical_Methods/03_Linear_Algebra.md) | 벡터 공간, 고유값, 행렬 연산 |
| 복소수(Complex numbers) | [Mathematical_Methods L02](../Mathematical_Methods/02_Complex_Numbers.md) | 오일러 공식, 페이저 표현 |
| 푸리에 급수 및 변환 | [Mathematical_Methods L07-L08](../Mathematical_Methods/07_Fourier_Series.md) | 수학적 기반 (이 강좌에서 신호 처리에 적용) |
| 라플라스 변환(Laplace transform) | [Mathematical_Methods L15](../Mathematical_Methods/15_Laplace_Transform.md) | 전달 함수, 시스템 분석 |
| 기초 Python (NumPy, Matplotlib) | [Python 토픽](../Python/00_Overview.md) | 모든 코드 예제가 Python 사용 |

---

## 레슨 목록

| 번호 | 파일명 | 제목 | 핵심 주제 |
|-----|----------|-------|------------|
| 00 | 00_Overview.md | 개요 | 강좌 소개 및 학습 가이드 |
| 01 | [01_Signals_and_Systems.md](./01_Signals_and_Systems.md) | 신호와 시스템 | 연속/이산 신호, 에너지/전력, 기본 신호, 시스템 특성 |
| 02 | [02_LTI_Systems_and_Convolution.md](./02_LTI_Systems_and_Convolution.md) | LTI 시스템과 합성곱 | 선형성, 시불변성, 합성곱 적분/합, 임펄스 응답 |
| 03 | [03_Fourier_Series_and_Applications.md](./03_Fourier_Series_and_Applications.md) | 푸리에 급수와 응용 | 주기 신호 분해, 스펙트럼, 파르세발 정리 |
| 04 | [04_Continuous_Fourier_Transform.md](./04_Continuous_Fourier_Transform.md) | 연속 푸리에 변환 | CTFT 특성, 주파수 영역 분석, 필터링 |
| 05 | 05_Sampling_and_Reconstruction.md | 표본화와 복원 | 나이퀴스트 정리, 앨리어싱, 안티-앨리어싱, DAC 복원 |
| 06 | 06_Discrete_Fourier_Transform.md | 이산 푸리에 변환 | DFT 정의, FFT 알고리즘, 영 패딩, 누설 |
| 07 | 07_Z_Transform.md | Z-변환 | Z-변환, 수렴역(ROC), 역 Z-변환, 전달 함수 |
| 08 | 08_Digital_Filter_Fundamentals.md | 디지털 필터 기초 | FIR vs IIR, 주파수 응답, 선형/비선형 위상 |
| 09 | 09_FIR_Filter_Design.md | FIR 필터 설계 | 윈도우 방법, Parks-McClellan, 선형 위상 조건 |
| 10 | 10_IIR_Filter_Design.md | IIR 필터 설계 | 버터워스(Butterworth), 체비쇼프(Chebyshev), 타원형, 쌍선형 변환 |
| 11 | 11_Multirate_Processing.md | 다중률 신호 처리 | 데시메이션(decimation), 보간(interpolation), 다상 필터, 표본율 변환 |
| 12 | 12_Spectral_Analysis.md | 스펙트럼 분석 | 주기도표(periodogram), Welch 방법, AR/ARMA 매개변수 모델 |
| 13 | 13_Adaptive_Filters.md | 적응 필터 | LMS, NLMS, RLS, 에코 제거, 잡음 제거 |
| 14 | 14_Time_Frequency_Analysis.md | 시간-주파수 분석 | STFT, 스펙트로그램(spectrogram), 웨이블릿 변환, CWT, DWT |
| 15 | 15_Image_Signal_Processing.md | 영상 신호 처리 | 2D DFT, 공간 필터, 에지 검출, 영상 개선 |
| 16 | 16_Applications.md | 응용 | 오디오 처리, 통신, 레이더/소나, 생체의학 신호 |

---

## 필요 라이브러리

```bash
pip install numpy scipy matplotlib
```

- **NumPy**: 배열 연산, FFT, 선형대수
- **SciPy**: 신호 처리 (`scipy.signal`), 필터 설계, 스펙트럼 분석, 웨이블릿
- **Matplotlib**: 신호 시각화, 스펙트로그램, 주파수 응답 그래프

### 선택적 라이브러리

```bash
pip install soundfile PyWavelets librosa
```

- **soundfile**: 오디오 파일 읽기/쓰기
- **PyWavelets** (`pywt`): 웨이블릿 변환 (CWT, DWT)
- **librosa**: 오디오 특징 추출 및 분석

---

## 권장 학습 경로

### 1단계: 기초 (레슨 01-04) — 2-3주

```
01 신호와 시스템
        │
        ▼
02 LTI 시스템과 합성곱
        │
        ▼
03 푸리에 급수 ──▶ 04 연속 푸리에 변환
```

- 신호 분류 및 기본 시스템 특성
- LTI 시스템의 핵심 연산으로서의 합성곱
- 주기 및 비주기 신호의 주파수 영역 분해
- 변환 특성과 물리적 해석

**목표**: 견고한 시간 영역 및 주파수 영역 기초 구축

### 2단계: 이산시간 체계 (레슨 05-07) — 2-3주

```
04 CTFT ──▶ 05 표본화와 복원
                        │
                        ▼
              06 DFT와 FFT ──▶ 07 Z-변환
```

- 표본화 이론을 통한 연속시간에서 이산시간으로의 연결
- DFT와 FFT를 이용한 계산적 주파수 분석
- 라플라스 변환의 이산시간 대응으로서의 Z-변환

**목표**: 이산시간 신호 분석을 위한 수학적 도구 숙달

### 3단계: 디지털 필터 설계 (레슨 08-10) — 2-3주

```
07 Z-변환 ──▶ 08 디지털 필터 기초
                            │
                    ┌───────┴───────┐
                    ▼               ▼
           09 FIR 설계     10 IIR 설계
```

- FIR과 IIR 구조 간의 트레이드오프 이해
- 정밀한 주파수 응답 사양에 맞는 필터 설계
- `scipy.signal`을 사용한 Python 필터 구현

**목표**: 어떤 사양에도 대응하는 디지털 필터 설계 및 구현 능력

### 4단계: 고급 분석 (레슨 11-14) — 2-3주

```
11 다중률 처리    12 스펙트럼 분석
                                    │
                                    ▼
                           13 적응 필터
                                    │
                                    ▼
                        14 시간-주파수 분석
```

- 표본율 변환 및 효율적인 다중률 구조
- 고전적 및 매개변수적 스펙트럼 추정
- 실시간 추적을 위한 적응 알고리즘
- 결합 시간-주파수 표현 (STFT, 웨이블릿)

**목표**: 비정상(non-stationary) 및 실시간 신호를 포함한 고급 분석 시나리오 처리

### 5단계: 응용 (레슨 15-16) — 1-2주

```
15 영상 신호 처리 ──▶ 16 응용
```

- 1D 처리를 2D 영상으로 확장
- 오디오, 통신, 레이더, 생체의학 분야의 종합 응용

**목표**: 신호 처리 기법을 실세계 문제에 적용

---

## 다른 주제와의 연계

| 관련 주제 | 연계 내용 |
|---------------|-----------|
| [Mathematical Methods](../Mathematical_Methods/00_Overview.md) | 푸리에 분석 (L07-08), 라플라스 변환 (L15), 복소 해석 (L14) |
| [Numerical Simulation](../Numerical_Simulation/00_Overview.md) | PDE 이산화 (L07-08), 스펙트럼 방법 (L21), FDTD (L15-16) |
| [Deep Learning](../Deep_Learning/00_Overview.md) | CNN의 1D/2D 합성곱, 가중 필터링으로서의 어텐션 |
| [Computer Vision](../Computer_Vision/00_Overview.md) | 영상 필터링, 에지 검출, 주파수 영역 영상 처리 |
| [Data Science](../Data_Science/00_Overview.md) | 시계열 분석, 스펙트럼 밀도 추정 |
| [Plasma Physics](../Plasma_Physics/00_Overview.md) | 플라즈마 파동의 푸리에 분석, 시뮬레이션의 스펙트럼 방법 |

---

## 참고 문헌

### 교재

1. **Oppenheim, A. V. & Willsky, A. S.** *Signals and Systems* (2nd ed.), Prentice Hall, 1997 — 연속 및 이산 신호와 시스템에 관한 고전적 참고서
2. **Oppenheim, A. V. & Schafer, R. W.** *Discrete-Time Signal Processing* (3rd ed.), Pearson, 2010 — 이산시간 이론과 필터 설계의 표준 교재
3. **Haykin, S. & Van Veen, B.** *Signals and Systems* (2nd ed.), Wiley, 2003 — 이론과 응용의 균형이 훌륭한 교재
4. **Proakis, J. G. & Manolakis, D. G.** *Digital Signal Processing* (4th ed.), Pearson, 2006 — MATLAB/Python 예제를 포함한 포괄적인 DSP 참고서
5. **Mallat, S.** *A Wavelet Tour of Signal Processing* (3rd ed.), Academic Press, 2009 — 웨이블릿 이론의 권위 있는 참고서

### 온라인 자료

- [MIT OpenCourseWare 6.003 — Signals and Systems](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/)
- [MIT OpenCourseWare 6.341 — Discrete-Time Signal Processing](https://ocw.mit.edu/courses/6-341-discrete-time-signal-processing-fall-2005/)
- [SciPy Signal Processing Documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [Think DSP — Free Python DSP textbook](https://greenteapress.com/wp/think-dsp/)
