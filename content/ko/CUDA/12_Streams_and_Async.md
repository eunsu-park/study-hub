# 12. 스트림과 비동기

**이전**: [NCU와 NSYS로 프로파일링](./11_Profiling_with_NCU_NSYS.md) | **다음**: [CUDA 그래프](./13_CUDA_Graphs.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. CUDA 스트림이 동시 커널 실행과 데이터 전송을 가능하게 하는 방법 설명
2. 비동기 전송을 위해 고정 메모리와 `cudaMemcpyAsync` 사용
3. 계산과 전송을 오버랩하는 더블 버퍼링 파이프라인 구현
4. CUDA 이벤트와 nsys로 오버랩 효율 측정
5. 실제 워크로드에 스트림 기반 동시성 적용

---

## 1. 기본 스트림 문제

**기본 스트림**의 모든 CUDA 연산(커널 + memcpy)은 **순차적으로** 실행됩니다:

```
기본 스트림 타임라인:
─────────────────────────────────────────────────────────
H→D 전송 │  커널 A  │  커널 B  │  D→H 전송
─────────────────────────────────────────────────────────
GPU 복사 엔진:  ████                        ████
GPU 연산:              ████████

총 시간 = 모든 연산의 합
```

GPU에는 두 개의 DMA 엔진 (H→D 복사와 D→H 복사)이 있어서 연산 엔진과 **동시에** 실행할 수 있습니다. 기본 스트림은 이 병렬성을 낭비합니다.

---

## 2. CUDA 스트림

스트림은 **순서가 있는 연산 큐**로, 스트림 내에서는 순서대로 실행되지만, **서로 다른 스트림은 오버랩될 수 있습니다**.

### 스트림 우선순위

스트림에 우선순위를 할당하면 GPU 스케줄러가 높은 우선순위의 작업을 우선 처리합니다:

```c
// 유효한 우선순위 범위 쿼리 (정수가 낮을수록 우선순위가 높음)
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
// 일반적인 범위: leastPriority=0, greatestPriority=-1

// 실시간 추론을 위한 고우선순위 스트림
cudaStream_t hi_stream;
cudaStreamCreateWithPriority(&hi_stream, cudaStreamNonBlocking, greatestPriority);

// 백그라운드 배치 처리를 위한 기본 우선순위 스트림
cudaStream_t lo_stream;
cudaStreamCreateWithPriority(&lo_stream, cudaStreamNonBlocking, leastPriority);

// 고우선순위 커널은 다음 스케줄링 포인트에서 저우선순위 커널을 선점
inference_kernel<<<grid, block, 0, hi_stream>>>(d_query);
batch_kernel    <<<grid, block, 0, lo_stream>>> (d_batch);
```

스트림 우선순위는 스케줄링 힌트이지 엄격한 보장이 아닙니다. 두 스트림이 같은 SM을 놓고 경쟁할 때 효과가 발휘되며, 리소스가 허용하는 경우 GPU는 두 스트림을 동시에 실행할 수도 있습니다.

```c
// 원래 스트림 생성 (비우선순위 방식):

// 스트림 생성
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// stream1의 연산이 stream2와 동시에 실행됨
kernel_A<<<grid, block, 0, stream1>>>(d_a);      // stream1에서 실행
cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream2);

// 특정 스트림 기다리기
cudaStreamSynchronize(stream1);  // stream1만 기다림

// 모든 스트림 기다리기
cudaDeviceSynchronize();

// 정리
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

---

## 3. 비동기 Memcpy는 고정 메모리가 필요

`cudaMemcpyAsync`는 **고정 (페이지-잠금) 호스트 메모리**를 사용할 때만 CPU에 즉시 반환합니다:

```c
// 고정 할당
float *h_pinned;
cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault);

// 비동기 복사 — CPU가 즉시 계속됨
cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, stream);

// CPU가 다른 작업을 할 수 있음
do_cpu_work();

// d_data를 사용하기 전에 복사가 완료되었는지 확인
cudaStreamSynchronize(stream);
```

**페이지 가능** 메모리에서는 `cudaMemcpyAsync`가 데이터를 임시 고정 버퍼에 스테이지할 때까지 블록 — 비동기 이점이 사라짐.

---

## 4. 더블 버퍼링 파이프라인

계산과 전송을 오버랩하는 전형적인 패턴:

```
파이프라이닝 없이:
  H→D [청크 0] │ 커널[0] │ H→D [청크 1] │ 커널[1] │ D→H [0] │ D→H [1]

더블 버퍼링 (2 스트림):
  스트림 0: H→D [청크 0] │              │ 커널[0] │              │ D→H [0]
  스트림 1:               │ H→D [청크 1] │          │ 커널[1]    │          │ D→H [1]
  타임라인:  ─────────────┬──────────────┬──────────┬──────────────┬──────────
                         오버랩!         오버랩!
```

구현:

```c
const int NUM_STREAMS = 2;
const int CHUNK = N / NUM_STREAMS;  // 청크당 원소 수

cudaStream_t streams[NUM_STREAMS];
float *d_in[NUM_STREAMS], *d_out[NUM_STREAMS];
float *h_in_pinned, *h_out_pinned;

// 설정
cudaHostAlloc(&h_in_pinned,  N * sizeof(float), cudaHostAllocDefault);
cudaHostAlloc(&h_out_pinned, N * sizeof(float), cudaHostAllocDefault);

for (int s = 0; s < NUM_STREAMS; s++) {
    cudaStreamCreate(&streams[s]);
    cudaMalloc(&d_in[s],  CHUNK * sizeof(float));
    cudaMalloc(&d_out[s], CHUNK * sizeof(float));
}

// h_in_pinned를 데이터로 초기화...

// 파이프라인: 각 청크에 대해 해당 스트림에 H→D, 커널, D→H 발행
for (int s = 0; s < NUM_STREAMS; s++) {
    int offset = s * CHUNK;

    // H→D: 비동기 복사 (즉시 반환)
    cudaMemcpyAsync(d_in[s], h_in_pinned + offset,
                    CHUNK * sizeof(float), cudaMemcpyHostToDevice, streams[s]);

    // 커널 (같은 스트림에서 H→D 후 실행)
    int gridSize = (CHUNK + 255) / 256;
    process_kernel<<<gridSize, 256, 0, streams[s]>>>(d_in[s], d_out[s], CHUNK);

    // D→H: 비동기 복사 (같은 스트림에서 커널 후 실행)
    cudaMemcpyAsync(h_out_pinned + offset, d_out[s],
                    CHUNK * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
}

// 모든 스트림 기다리기
cudaDeviceSynchronize();
```

---

## 5. 두 개 이상의 스트림

N개의 청크와 N개의 스트림으로, 파이프라인이 더 효율적입니다. 4개 이상의 스트림을 사용하면 파이프라인 시작/종료 비용을 숨길 수 있습니다:

```c
const int NSTREAMS = 4;
const int CHUNK = (N + NSTREAMS - 1) / NSTREAMS;

cudaStream_t streams[NSTREAMS];
float *d_buf[NSTREAMS];

for (int s = 0; s < NSTREAMS; s++) {
    cudaStreamCreate(&streams[s]);
    cudaMalloc(&d_buf[s], CHUNK * sizeof(float));
}

for (int s = 0; s < NSTREAMS; s++) {
    int sz = min(CHUNK, N - s * CHUNK);
    if (sz <= 0) continue;
    cudaMemcpyAsync(d_buf[s], h_pin + s * CHUNK, sz * sizeof(float),
                    cudaMemcpyHostToDevice, streams[s]);
    process<<<(sz+255)/256, 256, 0, streams[s]>>>(d_buf[s], sz);
    cudaMemcpyAsync(h_out + s * CHUNK, d_buf[s], sz * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[s]);
}
cudaDeviceSynchronize();
```

---

## 6. 동기화를 위한 스트림 이벤트

이벤트는 하나의 스트림이 다른 스트림의 특정 지점을 기다리게 합니다:

```c
cudaEvent_t event;
cudaEventCreate(&event);

// 스트림 1이 연산 완료 시 이벤트 기록
kernel_A<<<grid, block, 0, stream1>>>(d_a);
cudaEventRecord(event, stream1);  // stream1의 타임라인에 이벤트 기록

// 스트림 2가 stream1의 이벤트를 기다린 후 진행
cudaStreamWaitEvent(stream2, event, 0);  // stream2가 의존성 삽입
kernel_B<<<grid, block, 0, stream2>>>(d_b);  // stream1의 kernel_A 이후에 실행

cudaEventDestroy(event);
```

사용 사례:
- 스트림 간 생산자/소비자 패턴
- 팬아웃: 한 스트림의 결과가 여러 소비자 스트림에 공급
- 배리어: 여러 스트림이 최종 연산 전에 완료되어야 함

---

## 7. 이벤트로 시간 측정

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// 단일 커널 시간 측정
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);  // stop 이벤트 기다리기

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
printf("커널: %.3f ms\n", ms);

// 전체 파이프라인 시간 측정 (전송 포함)
cudaEventRecord(pipeline_start, 0);  // 스트림 0 = 기본 스트림
// ... 다른 스트림의 파이프라인 연산 ...
cudaDeviceSynchronize();
cudaEventRecord(pipeline_stop, 0);
cudaEventSynchronize(pipeline_stop);
cudaEventElapsedTime(&total_ms, pipeline_start, pipeline_stop);
```

---

## 8. 예상 속도 향상 분석

```
N=1GB 데이터 처리:
  커널 연산 시간:   500 ms
  H→D 전송 시간:    200 ms  (PCIe 4.0 ×16)
  D→H 전송 시간:    200 ms

스트림 없이 (순차적):
  총 시간 = 200 + 500 + 200 = 900 ms

2-스트림 더블 버퍼링:
  스트림 0: H→D(0) → 커널(0) → D→H(0)
  스트림 1:        H→D(1) → 커널(1) → D→H(1)

  타임라인: 100 + 500 + 100 = 700 ms  (전송이 연산과 오버랩)
  속도 향상: 900/700 = 1.29×

다중 스트림 파이프라이닝 (전송 << 연산):
  유효 시간: max(500, 200+200) = 500 ms  (거의 완벽한 오버랩)
  이론적 속도 향상: 900/500 = 1.8×

참고: 속도 향상은 전송_시간 / 연산_시간 비율에 달려 있습니다.
연산 >> 전송이면 파이프라이닝으로 약 2배 속도 향상 달성.
전송 >> 연산이면 파이프라이닝이 거의 도움이 안 됨.
```

---

## 9. nsys로 오버랩 확인

```bash
nsys profile --output=streams_test ./my_streamed_app
nsys-ui streams_test.nsys-rep
```

nsys 타임라인 GUI에서 확인할 사항:
- **CUDA HW** 행: 실제 GPU 커널 실행 표시
- **MemCpy HtoD** / **MemCpy DtoH** 행: DMA 전송 표시
- **오버랩**: 전송과 커널이 서로 다른 행에서 동시에 실행되어야 함

스트림을 사용함에도 순차적으로 보이는 경우 일반적인 원인:
1. 비고정 호스트 메모리 (`cudaHostAlloc` 사용)
2. 스트림 사이의 기본 스트림 연산 (모든 스트림 블록)
3. 블록 크기가 너무 큼 — 커널이 전체 GPU를 차지, 오버랩 여유 없음
4. 스트림 실행 사이의 호스트 블로킹 호출

---

## 핵심 요약

- CUDA 스트림은 커널과 데이터 전송의 동시 실행을 가능하게 함
- `cudaMemcpyAsync`는 진정한 비동기를 위해 **고정 메모리** (`cudaHostAlloc`) 필요
- **더블 버퍼링**은 청크 N+1의 H→D 전송을 청크 N의 커널과 오버랩
- `cudaStreamWaitEvent`는 호스트를 블록하지 않고 스트림 간 의존성 생성
- **nsys 타임라인**으로 오버랩 확인 — GPU에는 별도의 DMA 엔진과 연산 엔진이 있음
- 파이프라이닝의 속도 향상 = `총_시간 / max(연산_시간, 전송_시간)` (이상적인 경우)

---

**다음**: [13. CUDA 그래프](./13_CUDA_Graphs.md) — GPU 연산 시퀀스를 그래프로 캡처하고, CPU 실행 오버헤드 없이 재실행하며, 소형 배치 추론의 지연을 극적으로 줄입니다.
