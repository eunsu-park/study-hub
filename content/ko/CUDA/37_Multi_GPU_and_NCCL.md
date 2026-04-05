# 37. Multi-GPU Programming and NCCL

**이전**: [Fused Kernel Patterns](./36_Fused_Kernel_Patterns.md) | **다음**: [Capstone CUDA Application](./38_Capstone_CUDA_Application.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. `cudaMemcpyPeer`를 사용하여 GPU 간 데이터 전송하고 NVLink P2P 접근 활성화하기
2. NCCL 통신자를 초기화하고 그래디언트 동기화를 위해 `ncclAllReduce` 호출하기
3. AllReduce 그래디언트 평균화로 데이터 병렬 분산 학습 구현하기
4. Megatron-LM에서 사용되는 텐서 병렬성 (열/행 선형 분할) 이해하기
5. 파이프라인 병렬성 (스테이지 할당, 마이크로 배치) 및 그 트레이드오프 설명하기

---

## 1. Multi-GPU 하드웨어

```
인터커넥트 대역폭 비교:

NVLink (노드 내):
  NVLink 3.0 (A100):   양방향 600 GB/s (12 링크 × 50 GB/s)
  NVLink 4.0 (H100):   양방향 900 GB/s
  NVSwitch (8-GPU):   모든 GPU 간 전이분할 대역폭

PCIe (노드 내, NVLink 없음):
  PCIe 4.0 × 16:      ~32 GB/s (방향당)
  PCIe 5.0 × 16:      ~64 GB/s (방향당)

InfiniBand (노드 간):
  HDR 200 Gb/s:       GPU당 ~25 GB/s (RDMA / GPUDirect 사용)
  NDR 400 Gb/s:       GPU당 ~50 GB/s

규칙: 그래디언트 AllReduce 시간 ≈ 2 × 모델 크기 / 대역폭 × (nGPU-1)/nGPU
  8개 A100 (NVLink)에서 7B 모델 (FP16): 14GB × 7/8 / 600GB/s ≈ 반복당 20 ms
  vs PCIe: 14GB × 7/8 / 32GB/s ≈ 380 ms — 19× 느림!
```

---

## 2. Peer-to-Peer 메모리 접근

```c
// GPU 0과 GPU 1 사이의 P2P 접근 확인 및 활성화
void enable_p2p(int src, int dst) {
    int can_p2p;
    cudaDeviceCanAccessPeer(&can_p2p, src, dst);
    if (!can_p2p) {
        fprintf(stderr, "GPU %d가 GPU %d 메모리에 직접 접근할 수 없음\n", src, dst);
        return;
    }
    cudaSetDevice(src);
    cudaDeviceEnablePeerAccess(dst, 0);
    printf("P2P 활성화: GPU %d → GPU %d\n", src, dst);
}

// GPU 1에서 GPU 0으로 직접 복사
void p2p_copy(float *dst_gpu0, const float *src_gpu1, size_t bytes) {
    cudaMemcpyPeer(dst_gpu0, 0,     // 대상 device
                   src_gpu1, 1,     // 소스 device
                   bytes);
    // NVLink 사용 시: ~400 GB/s로 복사; P2P 없이: 호스트 메모리 경유 (~12 GB/s)
}

// 비동기 P2P 복사
void p2p_copy_async(float *dst, int dst_dev,
                    const float *src, int src_dev,
                    size_t bytes, cudaStream_t stream) {
    cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream);
}
```

---

## 3. NCCL 설정 및 통신자

NCCL (NVIDIA Collective Communications Library)은 GPU 인터커넥트에 최적화된 MPI 스타일 collective를 제공합니다:

```c
#include <nccl.h>

#define NCCL_CHECK(call) do {                                    \
    ncclResult_t r = call;                                       \
    if (r != ncclSuccess) {                                      \
        fprintf(stderr, "NCCL 오류 %s, %s:%d\n",                \
                ncclGetErrorString(r), __FILE__, __LINE__);      \
        exit(1);                                                 \
    }                                                            \
} while(0)

// 4개 GPU에 대한 통신자 초기화 (단일 프로세스)
void init_nccl(ncclComm_t *comms, int nGPU) {
    // 고유 ID 생성 (프로세스 0이 생성하여 모든 프로세스에 브로드캐스트)
    ncclUniqueId uid;
    ncclGetUniqueId(&uid);

    // 각 통신자 초기화 (GPU당 하나)
    for (int g = 0; g < nGPU; g++) {
        cudaSetDevice(g);
        NCCL_CHECK(ncclCommInitRank(&comms[g], nGPU, uid, g));
    }
}

// 멀티 프로세스 변형: 각 프로세스가 자신의 rank로 호출
void init_nccl_multiprocess(ncclComm_t *comm, int nranks, int rank,
                             ncclUniqueId uid) {
    NCCL_CHECK(ncclCommInitRank(comm, nranks, uid, rank));
}
```

---

## 4. NCCL AllReduce

AllReduce는 모든 GPU에 걸쳐 텐서를 합산 (또는 max/min/prod)하고 결과를 모든 GPU에 배포합니다:

```c
// AllReduce: 각 GPU가 d_grad를 기여하고 sum / nGPU (평균)를 받음
// 모든 nGPU device에서 동시에 호출됨
void allreduce_gradients(
    float **d_grads,      // d_grads[g] = GPU g의 그래디언트
    int param_count,
    int nGPU,
    ncclComm_t *comms,
    cudaStream_t *streams)
{
    // 모든 GPU에서 동시에 AllReduce 실행 (병렬 thread 또는 stream에서 수행해야 함)
    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < nGPU; g++) {
        cudaSetDevice(g);
        NCCL_CHECK(ncclAllReduce(
            (const void*)d_grads[g],  // 송신 버퍼
            (void*)d_grads[g],        // 수신 버퍼 (인플레이스)
            param_count,
            ncclFloat,                // 데이터 타입
            ncclSum,                  // reduction 연산
            comms[g],
            streams[g]));
    }
    NCCL_CHECK(ncclGroupEnd());

    // 1/nGPU를 곱해 평균 그래디언트 얻기 (GPU당 별도로 수행)
    for (int g = 0; g < nGPU; g++) {
        cudaSetDevice(g);
        scale_kernel<<<(param_count+255)/256, 256, 0, streams[g]>>>(
            d_grads[g], 1.f / nGPU, param_count);
    }
}

// 다른 NCCL collective:
// ncclBroadcast:   루트에서 모든 GPU로 전송
// ncclAllGather:   각 GPU가 청크를 기여; 모두 전체 연결본을 받음
// ncclReduceScatter: reduce + 분할 배포 (ZeRO 옵티마이저에서 사용)
// ncclSend/ncclRecv: point-to-point (ncclGroupStart/End과 쌍으로 사용)
```

---

## 5. 데이터 병렬성

각 GPU는 완전한 모델 복사본을 보유하고, 미니 배치의 일부를 처리하며, AllReduce를 통해 그래디언트를 동기화합니다:

```c
// 데이터 병렬 학습 루프 (4개 GPU, CUDA stream을 사용한 단일 프로세스)
void train_data_parallel(
    Model *models,    // models[g] = GPU g의 모델 (동일한 가중치)
    float **d_data,   // d_data[g] = GPU g의 데이터 샤드
    int nGPU, int steps)
{
    ncclComm_t    comms[4];
    cudaStream_t  streams[4];
    for (int g = 0; g < nGPU; g++) {
        cudaSetDevice(g);
        cudaStreamCreate(&streams[g]);
    }
    init_nccl(comms, nGPU);

    for (int step = 0; step < steps; step++) {
        // --- 각 GPU에서 순전파 + 역전파 ---
        for (int g = 0; g < nGPU; g++) {
            cudaSetDevice(g);
            // 각 GPU는 batch_size/nGPU 샘플 처리
            forward_backward(models[g], d_data[g], streams[g]);
        }

        // --- 모든 GPU에서 그래디언트 동기화 ---
        NCCL_CHECK(ncclGroupStart());
        for (int g = 0; g < nGPU; g++) {
            cudaSetDevice(g);
            for (int l = 0; l < models[g].n_layers; l++) {
                NCCL_CHECK(ncclAllReduce(
                    models[g].grads[l], models[g].grads[l],
                    models[g].layer_size[l],
                    ncclFloat, ncclSum,
                    comms[g], streams[g]));
            }
        }
        NCCL_CHECK(ncclGroupEnd());

        // --- 옵티마이저 단계: 각 GPU가 자체 가중치 업데이트 ---
        for (int g = 0; g < nGPU; g++) {
            cudaSetDevice(g);
            optimizer_step(models[g], 1.f/nGPU, streams[g]);  // 그래디언트를 1/nGPU로 스케일
        }

        // 가중치는 다음 이유로 동기화 상태 유지: 동일한 초기화 + 동일한 AllReduce 결과
    }
}
```

---

## 6. 텐서 병렬성 (Megatron-LM 스타일)

텐서 병렬성은 개별 가중치 행렬을 GPU에 분할합니다. 선형 레이어 Y = X·W에 대해:

```
열 병렬 선형 (W를 열 방향으로 분할):
  GPU 0: W_col0 [IC × OC/2]   Y0 = X · W_col0 계산
  GPU 1: W_col1 [IC × OC/2]   Y1 = X · W_col1 계산
  → AllGather로 전체 Y = [Y0, Y1] 얻기

행 병렬 선형 (W를 행 방향으로 분할, 열 병렬 이후):
  GPU 0: W_row0 [OC/2 × H]   입력 샤드 X0, 부분 Z0 = X0 · W_row0 계산
  GPU 1: W_row1 [OC/2 × H]   입력 샤드 X1, 부분 Z1 = X1 · W_row1 계산
  → AllReduce (합산) Z = Z0 + Z1

텐서 병렬성을 사용한 트랜스포머 어텐션 (Megatron):
  각 GPU가 H/nGPU 어텐션 헤드 처리
  어텐션 내에서 통신 불필요 (헤드는 독립적)
  출력 프로젝션에서만 AllReduce
  트랜스포머 block당 통신: AllReduce 호출 2번
```

```c
// 열 병렬 선형: 각 GPU가 W[:, my_col_start:my_col_end]를 가짐
void column_parallel_linear(
    cublasHandle_t handle,
    const float *d_X,       // [batch × IC] — 모든 GPU에서 동일
    const float *d_W_shard, // [IC × local_OC] — 각 GPU마다 다름
    float *d_Y_shard,       // [batch × local_OC] — 로컬 출력
    int batch, int IC, int local_OC)
{
    // 표준 GEMM (순전파에서 통신 불필요)
    sgemm_rowmajor(handle, d_X, d_W_shard, d_Y_shard, batch, local_OC, IC);
    // 샤드는 독립적으로 유용 (이후 행 병렬 또는 AllGather를 위해)
}

// 열 병렬 이후 AllGather: 모든 GPU 샤드 수집
void allgather_output(
    float *d_Y_shard, int local_OC,
    float *d_Y_full,  int total_OC,
    int batch, int rank, int nGPU,
    ncclComm_t comm, cudaStream_t stream)
{
    NCCL_CHECK(ncclAllGather(
        d_Y_shard,        // 송신: 내 샤드 [batch × local_OC]
        d_Y_full,         // 수신: 전체 [batch × total_OC]
        batch * local_OC, // 카운트: GPU당 원소 수
        ncclFloat,
        comm, stream));
}
```

---

## 7. 파이프라인 병렬성

파이프라인 병렬성은 모델 레이어를 GPU에 분할합니다 (GPU 0 = 레이어 0-5, GPU 1 = 레이어 6-11 등):

```
GPipe (나이브):
  스테이지 0: 마이크로 배치 0에 대해 레이어 0-5 순전파 → 스테이지 1에 활성화 전송
  스테이지 1: 마이크로 배치 0에 대해 레이어 6-11 순전파 → ...
  역순으로 역전파
  문제: 파이프라인 버블 = 총 시간의 (nStages-1)/nStages 낭비

1F1B 스케줄 (PipeDream):
  파이프라인 버블을 채우기 위해 순전파와 역전파 마이크로 배치 교차 수행
  각 스테이지 교대: 순전파 1단계, 역전파 1단계
  버블 비율을 1/m으로 줄임 (m = 배치당 마이크로 배치 수)
```

```c
// 파이프라인을 위한 point-to-point 통신 (다음 스테이지에 활성화 전송)
void pipeline_send_recv(
    const float *d_act_out, int n_act,  // 전송할 활성화
    float *d_act_in,                    // 수신된 활성화 버퍼
    int rank, int nStages,
    ncclComm_t comm, cudaStream_t stream)
{
    NCCL_CHECK(ncclGroupStart());

    // 다음 스테이지로 전송
    if (rank < nStages - 1)
        NCCL_CHECK(ncclSend(d_act_out, n_act, ncclFloat, rank+1, comm, stream));

    // 이전 스테이지에서 수신
    if (rank > 0)
        NCCL_CHECK(ncclRecv(d_act_in, n_act, ncclFloat, rank-1, comm, stream));

    NCCL_CHECK(ncclGroupEnd());
}
```

---

## 8. 병렬성 비교

```
전략           단계당 통신 비용        GPU당 메모리    적합한 경우
-----------------------------------------------------------------
데이터 병렬    AllReduce(파라미터)     전체 모델        표준 학습
텐서 병렬      레이어당 2×AllReduce   1/nGPU 모델      큰 레이어, 노드 내
파이프라인     활성화 send/recv       모델/nStages     매우 큰 모델, 노드 간

하이브리드 (대형 모델에서 일반적):
  데이터 병렬 (노드 간) × 텐서 병렬 (노드 내) × 파이프라인 (노드 그룹 간)
  예시: GPT-3 학습 = 64 데이터 병렬 × 8 텐서 병렬 × 8 파이프라인 병렬
         = 총 4096개 GPU

통신 볼륨 비교 (7B FP16 모델, 8개 GPU):
  데이터 병렬:   AllReduce 14GB ≈ 200ms (PCIe) / 20ms (NVLink)
  텐서 병렬:     레이어당 2 AllReduce × 32 레이어
                 각 ≈ 4MB → 64 × 4MB = 256MB 합계 ≈ 3ms (NVLink)
  → 텐서 병렬이 노드 내에서 우세 (짧고 빈번하며 NVLink)
  → 데이터 병렬이 노드 간에서 우세 (드물고 큰 배치)
```

---

## 핵심 요약

- **NVLink vs PCIe**: NVLink는 PCIe의 ~32 GB/s 대비 600 GB/s 제공; AllReduce 성능 차이는 ~19× — NVLink는 텐서/데이터 병렬성에 필수적
- **NCCL AllReduce** 인플레이스 연산: `sendbuff`와 `recvbuff` 모두에 동일한 포인터 전달; 단일 프로세스에서 여러 GPU에 실행할 때는 항상 `ncclGroupStart()`/`ncclGroupEnd()`로 래핑
- **데이터 병렬성**: 가장 단순한 접근 — 각 GPU가 데이터 샤드에서 전체 모델 실행; AllReduce로 그래디언트 동기화; 가중치는 GPU 간에 자동으로 동일하게 유지
- **텐서 병렬성**: 가중치 행렬을 열 방향 후 행 방향으로 GPU에 분할; 트랜스포머 레이어당 AllReduce 호출 2번만; 빠른 노드 내 인터커넥트 (NVLink) 필요
- **파이프라인 병렬성**: 다른 레이어를 다른 GPU에 할당; point-to-point send/recv 사용; 1F1B 스케줄은 파이프라인 버블 비율을 ~1/m으로 줄임 (m = 마이크로 배치)
- 실제 대형 모델 학습에서는 세 가지를 모두 결합: 노드 간 데이터 병렬, 노드 내 텐서 병렬, 노드 그룹 간 파이프라인 병렬

---

**다음**: [38. Capstone CUDA Application](./38_Capstone_CUDA_Application.md) — 완전한 2D 유체 시뮬레이션 또는 사용자 정의 CUDA kernel을 사용한 소형 LLM 추론 엔진을 구축하여 이 코스에서 배운 모든 것을 통합합니다.
