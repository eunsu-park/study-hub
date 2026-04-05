# 11. NCU와 NSYS로 프로파일링

**이전**: [루프라인 모델](./10_Roofline_Model.md) | **다음**: [스트림과 비동기](./12_Streams_and_Async.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Nsight Compute (`ncu`)로 개별 커널 병목 프로파일링
2. Nsight Systems (`nsys`)로 타임라인과 시스템 레벨 동작 분석
3. 메트릭 출력에서 최상위 성능 병목 식별
4. ncu 보고서의 최고속도(SOL) 섹션 해석
5. 프로파일링 우선 최적화 워크플로우 구축

---

## 1. 두 가지 도구, 두 가지 규모

| 도구 | 초점 | 세분성 |
|------|------|--------|
| **ncu** (Nsight Compute) | 개별 커널 내부 | 커널별 하드웨어 카운터 |
| **nsys** (Nsight Systems) | 전체 애플리케이션 타임라인 | CPU+GPU+PCIe 활동 |

**워크플로우**: 먼저 `nsys`로 어떤 커널이 느린지 찾은 다음, 해당 커널에 `ncu`를 사용하여 이유를 파악합니다.

---

## 2. Nsight Systems (`nsys`)

```bash
# 기본 타임라인 캡처
nsys profile --stats=true --output=myapp ./my_application

# GPU 메트릭 포함
nsys profile \
    --gpu-metrics-device=all \
    --cudabacktrace=all \
    --output=myapp \
    ./my_application

# 보고서 보기
nsys-ui myapp.nsys-rep
```

### nsys stats 출력 읽기

```
[6/7] Executing 'gpukernsum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Name
 --------  ---------------  ---------  --------  --------  ----
    74.3%       145,230,000        100   1452300  1449200   my_gemm_kernel
    15.2%        29,800,000       1000     29800    29500   my_elementwise_kernel
     6.8%        13,300,000        100   133000    132000   cudaMemcpy (HtoD)
```

이 출력의 핵심 관찰사항:
- `my_gemm_kernel`이 GPU 시간의 74% 소비 — 이것을 먼저 최적화
- `my_elementwise_kernel`은 1000번 실행 — 배치 처리하거나 퓨전 가능
- PCIe 전송은 상대적으로 작음 — 메모리 관리는 괜찮음

---

## 3. Nsight Compute (`ncu`) — 기본 사용법

```bash
# 단일 커널 프로파일링 (모든 메트릭 — 느림)
ncu --kernel-name myKernel ./my_app

# 특정 메트릭 세트로 프로파일링 (더 빠름)
ncu --set default ./my_app

# 파일로 출력하여 GUI에서 보기
ncu --output profile_report ./my_app
ncu-ui profile_report.ncu-rep

# 특정 메트릭 프로파일링
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
    sm__warps_active.avg.pct_of_peak_sustained_active \
    ./my_app
```

---

## 4. 최고속도(SOL) 섹션

ncu 보고서에서 가장 중요한 부분은 **Speed of Light** 섹션입니다:

```
Section: Speed Of Light Throughput
─────────────────────────────────────────────────────────
메트릭 이름                          메트릭 값    단위
─────────────────────────────────────────────────────────
DRAM 주파수                          1,593.0  MHz
SM 주파수                              765.0  MHz
경과 사이클                          2,048,0  사이클
메모리 [%]                              83.2    %   ← 메모리 활용률
DRAM 처리량 [%]                         80.1    %   ← 최대 BW의 %
경과 사이클                             2048  사이클
지속 시간                               2.68   ms
L1/TEX 캐시 처리량 [%]                  82.0    %
L2 캐시 처리량 [%]                      78.3    %
SM 활성 사이클                       1,945,0  사이클
연산 (SM) [%]                           12.3    %   ← 연산 활용률
─────────────────────────────────────────────────────────
```

**해석**:
- `DRAM 처리량 = 80%` + `연산 = 12%` → **메모리 병목 커널**
- 80%와 100% 사이의 격차는 메모리 접근 패턴 최적화 여지를 나타냄
- 두 값이 모두 >70%이면 커널이 잘 균형 잡혀 있음

**의사 결정 트리**:
```
DRAM 처리량 > 70% AND 연산 < 30% → 메모리 병목 → 합치기 수정, 로드 감소
연산 > 70% AND DRAM < 30%        → 연산 병목 → 벡터화, 텐서 코어, 언롤
둘 다 > 60%                       → 잘 균형 잡힘 → 루프라인 근처 성능
둘 다 < 40%                       → 다른 문제 (점유율, 실행 오버헤드, 동기화)
```

---

## 5. 핵심 메트릭 그룹

### 메모리 메트릭

```bash
ncu --metrics \
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\           # 전역 로드 바이트
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\           # 전역 저장 바이트
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\ # 섹터/요청 (합치기)
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,\         # L2 미스
    dram__bytes_read.sum,\                                   # DRAM 읽기
    dram__bytes_write.sum                                    # DRAM 쓰기
    ./my_kernel
```

목표값:
- `sectors_per_request` = 1.0 → 완벽한 합치기
- `l2_miss_rate` < 20% → 좋은 L2 캐시 활용

### 연산 메트릭

```bash
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\   # SM 활용률 %
    sm__warps_active.avg.pct_of_peak_sustained_active,\  # 점유율
    sm__inst_executed_pipe_fma.sum,\                     # FMA 명령
    sm__inst_executed_pipe_alu.sum                       # ALU 명령
    ./my_kernel
```

### 공유 메모리 메트릭

```bash
ncu --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\  # 로드 뱅크 충돌
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\  # 저장 뱅크 충돌
    l1tex__t_bytes_pipe_lsu_mem_shared_op_ld.sum                # 공유 로드 바이트
    ./my_kernel
```

---

## 6. 실용적인 프로파일링 워크플로우

```bash
# 1단계: 느린 커널 찾기
nsys profile --stats=true ./my_app 2>&1 | grep -A 20 gpukernsum

# 2단계: SOL 개요 얻기
ncu --set default --kernel-name "slow_kernel" ./my_app

# 3단계: 병목 진단
# 메모리 병목인 경우:
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    --kernel-name "slow_kernel" ./my_app

# 연산 병목인 경우:
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name "slow_kernel" ./my_app

# 4단계: 특정 수정 효과 측정
ncu --metrics dram__bytes_read.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./my_app_before ./my_app_after
```

---

## 7. Warp 상태 분포 해석

Nsight Compute는 warp가 시간을 어디에 쓰는지 보고합니다:

```
Warp 상태 통계
─────────────────────────────────────────────────
상태              평균 (# warp)   활성 사이클의 %
─────────────────────────────────────────────────
Stall MIO Throttle      18.4           52.3%   ← 메모리 문제
Issue                    8.2           23.3%   ← 실제 작업
Stall Wait                4.1           11.7%
Stall Long Scoreboard     3.3            9.4%   ← 긴 지연 (전역 메모리)
기타                      1.5            3.3%
─────────────────────────────────────────────────
```

**Stall MIO Throttle** (52%)는 warp가 메모리 명령이 발행될 때까지 블록됨을 의미합니다 — 메모리 병목, 진행 중인 메모리 요청이 너무 많거나 합치기가 나쁜 경우 가능.

**Stall Long Scoreboard** = 전역 메모리 로드가 반환될 때까지 기다림 — 메모리 병목 커널에서 예상됨.

---

## 8. 텐서 코어를 위한 NCU

텐서 코어를 사용하는 커널 (WMMA 또는 cuBLAS)의 경우:

```bash
ncu --metrics \
    sm__inst_executed_pipe_tensor_op_hmma.sum,\    # FP16 텐서 코어 연산
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active \
    ./tensor_core_kernel
```

텐서 코어 사용을 기대했는데 `hmma` 연산이 0이면, 커널이 CUDA 코어로 폴백하고 있습니다.

---

## 9. 일반적인 성능 문제와 ncu 시그니처

| 문제 | ncu 시그니처 |
|------|------------|
| 비합치 전역 메모리 | `sectors_per_request` >> 1 |
| 공유 메모리 뱅크 충돌 | `bank_conflicts` > 0 |
| 낮은 점유율 | `warps_active` < 50% |
| 레지스터 스필링 | `launch__registers_per_thread` 메트릭 + ptxas 출력 확인 |
| CPU-GPU 직렬화 | `nsys` 타임라인에서 커널 사이 GPU 유휴 표시 |
| 실행 오버헤드를 숨기기에 너무 짧은 커널 | nsys에서 `Duration` < 10 μs |
| 텐서 코어 미활용 | `tensor_op_hmma` 수가 0 |

---

## 핵심 요약

- **nsys** → 어떤 커널이 느린지 찾기; **ncu** → 이유 이해
- **최고속도** 섹션은 즉시 알려줍니다: 메모리 병목 (DRAM% 높음, 연산% 낮음) 또는 연산 병목 (반대)
- `sectors_per_request`는 단일 가장 실행 가능한 합치기 메트릭 — 목표 1.0
- Warp 상태 분포는 정지 이유 드러냄: `Long Scoreboard` = 전역 메모리, `MIO Throttle` = 명령 큐 꽉참
- 최적화하기 **전에** 프로파일링 — 병목이 어디 있는지에 대한 직감은 80%의 경우 틀림

---

**다음**: [12. 스트림과 비동기](./12_Streams_and_Async.md) — CUDA 스트림으로 계산과 데이터 전송을 오버랩하고, 더블 버퍼링 파이프라인을 구현하며, 이벤트로 시간을 정밀하게 측정합니다.
