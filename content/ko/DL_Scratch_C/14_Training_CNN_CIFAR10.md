# 14. CIFAR-10에서 CNN 학습하기

**이전**: [LeNet과 AlexNet](./13_LeNet_and_AlexNet.md) | **다음**: [VGG와 딥 네트워크](./15_VGG_and_Deep_Networks.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 완전한 학습 루프 조립하기: 데이터 로더 → forward → loss → backward → 옵티마이저
2. 수치 안정성을 고려한 cross-entropy loss 구현하기
3. momentum과 weight decay를 포함한 SGD 구현하기
4. 에폭별 학습 loss와 테스트 정확도 추적하기
5. 단순한 CNN으로 CIFAR-10에서 약 80% 테스트 정확도 달성하기

---

## 1. Cross-Entropy Loss(교차 엔트로피 손실)

C개 클래스에 대한 분류:

```
Softmax:  p[i] = exp(logit[i]) / Σ_j exp(logit[j])
Loss:     L = -log(p[y])   여기서 y는 정답 클래스

수치적으로 안정적인 방법: exp 이전에 최댓값 빼기
  z[i] = logit[i] - max(logit)
  p[i] = exp(z[i]) / Σ_j exp(z[j])
```

softmax + cross-entropy forward와 backward 결합:

```c
// softmax_cross_entropy_forward:
//   logits: [N, C]
//   labels: [N] int in [0, C)
//   평균 loss 반환, softmax 확률을 probs[N*C]에 기록
float softmax_cross_entropy_forward(
    const float  *logits,  // [N, C]
    const uint8_t *labels, // [N]
    float        *probs,   // [N, C] softmax 출력
    int N, int C) {

    float total_loss = 0.0f;

    for (int n = 0; n < N; n++) {
        const float *row = logits + n * C;
        float       *p   = probs  + n * C;

        // 수치적으로 안정적인 softmax
        float max_val = row[0];
        for (int c = 1; c < C; c++)
            if (row[c] > max_val) max_val = row[c];

        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            p[c] = expf(row[c] - max_val);
            sum += p[c];
        }
        for (int c = 0; c < C; c++) p[c] /= sum;

        // Cross-entropy: -log(p[y])
        int y = labels[n];
        total_loss += -logf(p[y] + 1e-9f);
    }
    return total_loss / N;
}

// softmax_cross_entropy_backward:
//   dlogits[n][c] = (p[n][c] - 1{c == y[n]}) / N
void softmax_cross_entropy_backward(
    const float  *probs,   // [N, C]
    const uint8_t *labels, // [N]
    float        *dlogits, // [N, C]
    int N, int C) {

    memcpy(dlogits, probs, N * C * sizeof(float));
    for (int n = 0; n < N; n++)
        dlogits[n * C + labels[n]] -= 1.0f;

    // N으로 나누기 (평균 감소)
    float inv_N = 1.0f / N;
    for (int i = 0; i < N * C; i++)
        dlogits[i] *= inv_N;
}
```

---

## 2. Momentum과 Weight Decay를 포함한 SGD

```c
// 파라미터 텐서당 SGD 상태
typedef struct {
    float *velocity;  // momentum 버퍼, param과 동일한 shape
    int    n;         // 원소 수
} SGDState;

// sgd_update: 하나의 파라미터 텐서 업데이트
// v   = momentum * v - lr * (grad + weight_decay * param)
// p  += v
void sgd_update(
    float    *param,
    float    *grad,
    SGDState *state,
    float lr, float momentum, float weight_decay) {

    for (int i = 0; i < state->n; i++) {
        float g = grad[i] + weight_decay * param[i];
        state->velocity[i] = momentum * state->velocity[i] - lr * g;
        param[i] += state->velocity[i];
    }
}

// 각 forward-backward pass 전에 모든 그래디언트 초기화
void zero_grad(float **grads, int *sizes, int num_tensors) {
    for (int i = 0; i < num_tensors; i++)
        memset(grads[i], 0, sizes[i] * sizeof(float));
}
```

---

## 3. 학습률 스케줄

선형 warmup을 포함한 cosine decay:

```c
float get_lr(int step, int warmup_steps, int total_steps,
             float lr_max, float lr_min) {
    if (step < warmup_steps) {
        // 선형 warmup
        return lr_max * ((float)step / warmup_steps);
    }
    // Cosine decay
    float progress = (float)(step - warmup_steps) / (total_steps - warmup_steps);
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(M_PI * progress));
}
```

단계적 감소 (더 단순하며 CIFAR-10에서 일반적으로 사용):

```c
float lr_step_decay(float base_lr, int epoch, int *milestones, float gamma, int n_milestones) {
    float lr = base_lr;
    for (int i = 0; i < n_milestones; i++)
        if (epoch >= milestones[i]) lr *= gamma;
    return lr;
}
// 예시: base_lr=0.1, milestones={100,150}, gamma=0.1
// → 에폭 100까지 lr=0.1, 150까지 0.01, 이후 0.001
```

---

## 4. 정확도 측정

```c
// top1_accuracy: 올바른 예측의 비율
float top1_accuracy(const float *logits, const uint8_t *labels, int N, int C) {
    int correct = 0;
    for (int n = 0; n < N; n++) {
        const float *row = logits + n * C;
        int pred = 0;
        for (int c = 1; c < C; c++)
            if (row[c] > row[pred]) pred = c;
        if (pred == labels[n]) correct++;
    }
    return (float)correct / N;
}
```

---

## 5. CIFAR-10을 위한 단순 CNN

최소한이지만 효과적인 아키텍처:

```
Input:  [N, 3, 32, 32]
Block1: Conv(3→32,  3×3, p=1) → BN → ReLU → MaxPool(2×2) → [N, 32, 16, 16]
Block2: Conv(32→64, 3×3, p=1) → BN → ReLU → MaxPool(2×2) → [N, 64, 8, 8]
Block3: Conv(64→128,3×3, p=1) → BN → ReLU → MaxPool(2×2) → [N, 128, 4, 4]
GAP:   [N, 128]
FC:    128 → 10
```

파라미터 수: ~17만 — 빠른 학습, 약 80% 정확도 도달.

---

## 6. 완전한 학습 루프

```c
int main(void) {
    srand(42);

    // --- 데이터 ---
    CIFAR10Dataset *train_ds = cifar10_load("cifar-10/data_batch_1.bin", 1);
    CIFAR10Dataset *test_ds  = cifar10_load("cifar-10/test_batch.bin",   0);
    DataLoader *train_dl = dataloader_create(train_ds, /*batch=*/128, /*augment=*/1);
    DataLoader *test_dl  = dataloader_create(test_ds,  /*batch=*/128, /*augment=*/0);

    // --- 모델 및 옵티마이저 ---
    SimpleCNN *model = simple_cnn_create();
    simple_cnn_init_weights(model);

    float lr = 0.1f, momentum = 0.9f, weight_decay = 1e-4f;
    int milestones[] = {100, 150}, n_milestones = 2;
    float gamma = 0.1f;

    float *batch_X = malloc(128L * 3 * 32 * 32 * sizeof(float));
    uint8_t *batch_y = malloc(128);

    FILE *log = fopen("training_log.csv", "w");
    fprintf(log, "epoch,train_loss,test_acc\n");

    // --- 학습 ---
    for (int epoch = 0; epoch < 200; epoch++) {
        float cur_lr = lr_step_decay(lr, epoch, milestones, gamma, n_milestones);

        // 학습 단계
        dataloader_shuffle(train_dl);
        float train_loss = 0.0f;
        int n_batches = 0;

        while (dataloader_next(train_dl, batch_X, batch_y)) {
            // 그래디언트 초기화
            simple_cnn_zero_grad(model);

            // Forward pass
            float *logits = model->logit_buf;
            simple_cnn_forward(model, batch_X, logits, 128, /*training=*/1);

            // Loss
            float loss = softmax_cross_entropy_forward(
                logits, batch_y, model->probs, 128, 10);
            train_loss += loss;

            // Backward pass (모든 모델 그래디언트 설정)
            softmax_cross_entropy_backward(model->probs, batch_y, model->dlogits, 128, 10);
            simple_cnn_backward(model, batch_X, 128);

            // 파라미터 업데이트
            simple_cnn_update(model, cur_lr, momentum, weight_decay);

            n_batches++;
        }
        train_loss /= n_batches;

        // 평가 단계
        float total_acc = 0.0f;
        int n_test_batches = 0;

        while (dataloader_next(test_dl, batch_X, batch_y)) {
            float *logits = model->logit_buf;
            simple_cnn_forward(model, batch_X, logits, 128, /*training=*/0);
            total_acc += top1_accuracy(logits, batch_y, 128, 10);
            n_test_batches++;
        }
        float test_acc = total_acc / n_test_batches;

        printf("Epoch %3d | lr=%.4f | loss=%.4f | test_acc=%.2f%%\n",
               epoch + 1, cur_lr, train_loss, test_acc * 100.0f);
        fprintf(log, "%d,%.4f,%.4f\n", epoch + 1, train_loss, test_acc);
        fflush(log);
    }

    fclose(log);
    simple_cnn_free(model);
    cifar10_free(train_ds);
    cifar10_free(test_ds);
    return 0;
}
```

---

## 7. 예상 학습 곡선

```
단순 3-블록 CNN, SGD(lr=0.1→0.001), 200 에폭, batch=128:

에폭   1: loss=2.20  test=10.2%  (무작위 기준선 = 10%)
에폭  10: loss=1.65  test=42.3%
에폭  50: loss=0.89  test=72.1%
에폭 100: loss=0.63  test=78.8%
에폭 150: loss=0.47  test=80.2%  ← 100, 150에서 LR 감소
에폭 200: loss=0.41  test=80.9%

총 시간: 현대 CPU (M2/i7)에서 약 15분
```

흔한 실패 원인:

```
Loss가 2.3에 머뭄 (= -log(1/10)):   학습률이 너무 낮거나 가중치 초기화 오류
Loss 폭발 (1 에폭 후 NaN):          학습률이 너무 높음, 그래디언트 클리핑 누락
Test acc >> 에폭 1부터 train acc:   eval/train 모드 혼동 (BN/dropout)
Train acc >> test acc가 15% 이상:   과적합 — dropout, weight decay, 증강 추가
```

---

## 8. 루프 프로파일링

시간이 어디에 소비되는지 파악:

```c
#include <time.h>

clock_t t0 = clock();
simple_cnn_forward(model, batch_X, logits, 128, 1);
double forward_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000;

t0 = clock();
simple_cnn_backward(model, batch_X, 128);
double backward_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000;

printf("forward: %.1fms  backward: %.1fms  ratio: %.1fx\n",
       forward_ms, backward_ms, backward_ms / forward_ms);
// 일반적인 비율: backward ≈ forward의 2–3배
```

---

## 핵심 정리

- **Cross-entropy loss**: 수치 안정성을 위해 최댓값 빼기; backward는 단순히 `(softmax - one_hot) / N`
- **SGD + momentum**: `v = β*v - lr*(grad + wd*param)` — weight decay는 각 스텝마다 가중치를 줄여 정규화
- **LR 스케줄**: 고정 마일스톤(100, 150)에서의 단계적 감소는 CIFAR-10에서 단순하고 효과적
- **Eval 모드**: 테스트 중 BN 배치 통계와 dropout 비활성화 — 이를 잊으면 테스트 loss가 부풀려짐
- **그래디언트 체크**: 전체 학습 실행 전에 작은 배치로 forward → backward 검증
- Backward는 forward보다 약 2–3배 느림 — 이것은 정상 (세 가지 그래디언트 모두 계산: dX, dW, db)

---

**다음**: [15. VGG와 딥 네트워크](./15_VGG_and_Deep_Networks.md) — VGG-16/19 아키텍처, 네트워크 깊이의 효과, 깊은 네트워크에서의 그래디언트 소실, 대규모 파라미터 계산.
