# 13. CUDA 그래프

**이전**: [스트림과 비동기](./12_Streams_and_Async.md) | **다음**: [병렬 리덕션](./14_Parallel_Reduction.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 반복적인 커널 실행에 CPU 측 오버헤드가 있는 이유 설명
2. 스트림 캡처를 통한 CUDA 그래프 캡처
3. `cudaGraphLaunch`로 그래프 인스턴스화 및 재실행
4. 재캡처 없이 그래프 노드 매개변수 업데이트
5. 소형 배치 추론에 그래프 적용하여 지연 감소

---

## 1. CPU 실행 오버헤드 문제

모든 CUDA 커널 실행에는 CPU 측 오버헤드가 있습니다:

```
일반적인 커널 실행 오버헤드:
  cudaLaunchKernel() 호출:   5–20 μs  (소프트웨어 오버헤드)
  GPU 실행 지연:             ~5 μs

100개의 커널이 있는 신경망 추론 순방향 패스:
  실행 오버헤드만으로도: 100 × 15 μs = 1,500 μs = 1.5 ms

실제 GPU 연산이 0.5 ms (소형 배치)라면, 오버헤드가
연산 시간의 3배 — 실시간 추론에서 완전히 허용 불가.
```

이것이 CUDA 10.0에서 CUDA 그래프가 도입된 이유입니다.

---

## 2. CUDA 그래프 개념

**CUDA 그래프**는 GPU 연산의 DAG(방향 비순환 그래프)를 캡처합니다:

```
그래프 구조:
  MemcpyHtoD(A) ──→ kernel_1(A) ──→ kernel_2(A,B) ──→ MemcpyDtoH(result)
  MemcpyHtoD(B) ──↗
```

한 번 캡처되면, **전체 그래프**를 **단일 CPU 호출**로 GPU에 제출할 수 있습니다 — 연산별 실행 오버헤드가 제거됩니다. GPU는 의존성 순서로 그래프의 연산을 실행합니다.

---

## 3. 스트림 캡처

그래프를 만드는 가장 쉬운 방법 — 일반 스트림 연산 기록:

```c
// 1단계: 캡처 시작
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 2단계: 연산 정상적으로 발행 — 실행되지 않고 CAPTURED됨
cudaMemcpyAsync(d_A, h_A, bytes_A, cudaMemcpyHostToDevice, stream);
cudaMemcpyAsync(d_B, h_B, bytes_B, cudaMemcpyHostToDevice, stream);
kernel_1<<<grid, block, 0, stream>>>(d_A, d_tmp);
kernel_2<<<grid, block, 0, stream>>>(d_tmp, d_B, d_out);
cudaMemcpyAsync(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost, stream);

// 3단계: 캡처 종료 — 그래프 생성
cudaStreamEndCapture(stream, &graph);

// 4단계: 인스턴스화 (이 GPU를 위한 그래프 컴파일)
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 5단계: 그래프 준비됨. 스트림이 더 이상 캡처하지 않음.
```

**중요**: 캡처 중에는 GPU 작업이 실제로 실행되지 않습니다. 모든 연산이 그래프 구조에 기록됩니다.

---

## 4. 그래프 재실행

```c
// 각 실행: 단일 CPU 호출, 전체 캡처된 시퀀스 재실행
for (int step = 0; step < 1000; step++) {
    // 호스트에서 입력 데이터 업데이트
    prepare_next_batch(h_A, h_B, step);

    // 전체 그래프 실행
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    // 결과 읽기
    process_output(h_out);
}
```

**벤치마크 비교** (100개 커널 신경망 순방향 패스, 배치=1):

```
그래프 없이: 100 × 15 μs 오버헤드 + 0.5 ms 연산 = 추론당 2.0 ms
그래프 사용: ~10 μs 오버헤드 + 0.5 ms 연산       = 추론당 0.51 ms
속도 향상: 3.9×  (거의 전적으로 실행 오버헤드 제거에서 옴)
```

---

## 5. 그래프 매개변수 업데이트

실행 간에 **데이터 포인터나 값**만 변경되면 (그래프 구조가 아닌 경우), 재캡처 없이 그래프를 업데이트할 수 있습니다:

```c
// 커널의 매개변수 업데이트
cudaKernelNodeParams params;
cudaGraphKernelNodeGetParams(kernelNode, &params);

// 인수 변경
float new_alpha = 2.0f;
params.kernelParams[0] = &new_alpha;
cudaGraphKernelNodeSetParams(kernelNode, &params);

// 재인스턴스화 (재캡처보다 훨씬 빠름)
cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &params);
```

포인터 업데이트의 경우 (예: 다른 출력 버퍼):

```c
cudaGraphExecMemcpyNodeSetParams1D(graphExec, memcpyNode,
    d_new_out, h_out, bytes, cudaMemcpyDeviceToHost);
```

---

## 6. 다중 스트림 그래프 캡처

여러 스트림의 연산을 캡처하여 그래프에서 병렬성을 표현합니다:

```c
// 두 개의 병렬 브랜치
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 두 스트림 함께 캡처
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

// 브랜치 1 (stream1)
kernel_branch_A<<<grid, block, 0, stream1>>>(d_a);

// 브랜치 2 (stream2) — 전역 모드로 인해 캡처됨
kernel_branch_B<<<grid, block, 0, stream2>>>(d_b);

// 결합: stream2가 stream1을 기다림 (그래프에 의존성 생성)
cudaEvent_t joinEvent;
cudaEventCreate(&joinEvent);
cudaEventRecord(joinEvent, stream1);
cudaStreamWaitEvent(stream2, joinEvent, 0);

// 결합 후 stream2에서 계속
kernel_merge<<<grid, block, 0, stream2>>>(d_a, d_b, d_out);

// 캡처 종료 — stream1이 루트
cudaStreamEndCapture(stream1, &graph);
```

결과 그래프는 브랜치 A와 브랜치 B 사이의 병렬성을 캡처하여, GPU가 동시에 실행할 수 있게 합니다.

---

## 7. 조건부 경로를 가진 그래프 실행 (CUDA 12.4+)

CUDA 12.4는 조건부 그래프 노드를 도입했습니다 — CPU-GPU 왕복 없이 완전히 GPU에서 if-else와 while-loop 구조:

```c
// (단순화된 의사 코드 — 실제 API는 cudaGraphConditionalHandle 사용)
// 조건부 노드 생성
cudaGraphNode_t condNode;
cudaConditionalNodeParams condParams = { .type = cudaGraphCondTypeIf };
cudaGraphAddConditionalNode(&condNode, graph, deps, ndeps, &condParams);

// 조건부 내에서 "true"와 "false" 브랜치를 서브그래프로 캡처
// GPU가 CPU 관여 없이 런타임에 조건을 평가
```

이는 어떤 GPU 연산을 실행할지 결정하기 위해 CPU에 결과를 반환하는 지연을 제거합니다.

---

## 8. CUDA 그래프를 사용해야 할 때

**그래프를 사용해야 하는 경우:**
- 커널이 매우 짧고 (< 1 ms) 실행 오버헤드가 상당할 때
- 같은 연산 시퀀스가 여러 번 반복될 때 (추론 서빙)
- 퓨전할 수 없는 많은 소형 커널이 있을 때
- 실시간 제약 (로봇, 트레이딩, 인터랙티브 애플리케이션)

**그래프를 사용하지 말아야 하는 경우:**
- 그래프 구조가 데이터에 따라 변경될 때 (동적 제어 흐름)
- 연산이 충분히 길어서 실행 오버헤드가 무시 가능할 때 (> 10 ms)
- 반복 간에 메모리 할당 패턴이 변경될 때

```
경험 법칙: launch_overhead > compute_time의 10%이면 그래프 사용
  10개 커널로 compute < 100 μs: overhead = 150 μs → 150% → 반드시 그래프 사용
  10개 커널로 compute > 10 ms:  overhead = 150 μs → 1.5% → 그래프 건너뜀
```

---

## 9. 완전한 예시: 추론 서빙

```c
// 시작 시 한 번만 그래프 구성
void setup_inference_graph(Model *m, cudaGraphExec_t *exec) {
    cudaStream_t captureStream;
    cudaStreamCreate(&captureStream);

    cudaGraph_t graph;
    cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal);

    // 순방향 패스 커널
    embedding_lookup<<<grid, block, 0, captureStream>>>(
        d_input_ids, m->d_embed, m->d_hidden, m->vocab_size, m->d_model);
    for (int layer = 0; layer < m->n_layers; layer++) {
        attention_kernel<<<grid, block, shm, captureStream>>>(
            m->d_hidden, m->d_kv_cache + layer * kv_size,
            m->d_attn_out, m->seq_len, m->d_head);
        ffn_kernel<<<grid, block, 0, captureStream>>>(
            m->d_attn_out, m->d_hidden, m->d_ffn_w1[layer], m->d_ffn_w2[layer]);
    }
    lm_head<<<grid, block, 0, captureStream>>>(m->d_hidden, d_logits, m->vocab_size);

    cudaStreamEndCapture(captureStream, &graph);
    cudaGraphInstantiate(exec, graph, NULL, NULL, 0);

    cudaGraphDestroy(graph);
    cudaStreamDestroy(captureStream);
}

// 추론 루프 (호출당 그래프 오버헤드 없음)
void run_inference(cudaGraphExec_t exec, cudaStream_t stream,
                   const int *h_tokens, float *h_logits, int n_tokens) {
    cudaMemcpyAsync(d_input_ids, h_tokens, n_tokens * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaGraphLaunch(exec, stream);
    cudaMemcpyAsync(h_logits, d_logits, vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}
```

---

## 핵심 요약

- CPU 커널 실행 오버헤드 (커널당 5–20 μs)는 짧거나 반복적인 커널에서 병목이 됨
- **CUDA 그래프**는 GPU 연산의 DAG를 캡처하고 단일 CPU 호출로 재실행
- **스트림 캡처** (`cudaStreamBeginCapture/EndCapture`)가 그래프를 구성하는 가장 쉬운 방법
- `cudaGraphLaunch`는 모든 개별 커널 실행을 단일 호출로 대체 — 일반적으로 CPU 오버헤드 20–100배 감소
- 재캡처 없이 노드 매개변수 (포인터, 스칼라) 업데이트 — `cudaGraphExecKernelNodeSetParams` 사용
- 그래프는 추론 서빙, 실시간 시스템, 많은 짧은 커널이 있는 모든 워크로드에 가장 유익

---

**다음**: [14. 병렬 리덕션](./14_Parallel_Reduction.md) — 기본 GPU 기본 연산 구현: 트리 리덕션, warp 셔플 리덕션, 다단계 리덕션, CUB 장치 리덕션.
