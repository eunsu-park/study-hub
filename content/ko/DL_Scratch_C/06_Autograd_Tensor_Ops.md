# 06. Autograd 텐서 연산

**이전**: [Autograd 엔진](./05_Autograd_Engine.md) | **다음**: [메모리 관리자](./07_Memory_Manager.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 행렬 곱셈의 역방향 패스 유도 및 구현
2. Softmax의 역방향 패스 유도 및 구현
3. 퓨전된 softmax-cross-entropy 역방향 패스 구현
4. 유한 차분 테스트로 모든 그래디언트 검증
5. 여러 autograd 연산을 합성하여 전체 MLP forward+backward 계산

---

## 1. Matmul 역방향

### 순방향

```
C = A @ B      (A: [M,K], B: [K,N], C: [M,N])
```

### 역방향 유도

`dL/dC`(C와 같은 shape)를 받아 `dL/dA`와 `dL/dB`를 계산해야 합니다.

행렬 곱의 연쇄 법칙을 사용하면:

```
dL/dA = dL/dC @ B^T       [M,N] @ [N,K] = [M,K]  ✓
dL/dB = A^T @ dL/dC       [K,M] @ [M,N] = [K,N]  ✓
```

**유도 스케치**: 임의의 스칼라 손실 `L`에 대해,
```
dL/dA[i,k] = sum_j (dL/dC[i,j] * dC[i,j]/dA[i,k])
            = sum_j (dL/dC[i,j] * B[k,j])
            = (dL/dC @ B^T)[i,k]
```

### 구현

```c
typedef struct {
    AGNode *A;
    AGNode *B;
} MatmulCtx;

static void matmul_backward(AGNode *node) {
    MatmulCtx *ctx = (MatmulCtx *)node->ctx;
    AGNode *A = ctx->A, *B = ctx->B;
    Tensor *dC = node->grad;  // dL/dC

    // dL/dA = dL/dC @ B^T
    if (A->requires_grad) {
        Tensor *B_T = tensor_transpose(B->tensor, 0, 1);
        // beta=1.0으로 누적 (덮어쓰기 아님)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    dC->shape[0], B_T->shape[1], dC->shape[1],
                    1.0f, dC->data, dC->shape[1],
                    B_T->data, B_T->shape[1],
                    1.0f, A->grad->data, A->grad->shape[1]);
        tensor_free(B_T);
    }

    // dL/dB = A^T @ dL/dC
    if (B->requires_grad) {
        Tensor *A_T = tensor_transpose(A->tensor, 0, 1);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    A_T->shape[0], dC->shape[1], A_T->shape[1],
                    1.0f, A_T->data, A_T->shape[1],
                    dC->data, dC->shape[1],
                    1.0f, B->grad->data, B->grad->shape[1]);
        tensor_free(A_T);
    }
}
```

> **그래디언트 누적**: `+=` 사용 이유는 텐서가 그래프의 여러 브랜치에 나타날 수 있기 때문입니다. `cblas_sgemm`에서 `beta=1.0`으로 호출하여 누적합니다.

---

## 2. Softmax 순방향과 역방향

### 순방향

길이 `N`의 벡터 `x`에 대해:

```
softmax(x)[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))
```

`max(x)`를 빼는 것은 출력을 변경하지 않으면서 오버플로우를 방지합니다.

```c
void softmax_forward(float *out, const float *x, size_t N) {
    float m = x[0];
    for (size_t i = 1; i < N; i++) if (x[i] > m) m = x[i];

    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        out[i] = expf(x[i] - m);
        sum += out[i];
    }
    for (size_t i = 0; i < N; i++) out[i] /= sum;
}
```

### 역방향

`p = softmax(x)`이고 업스트림 그래디언트 `dL/dp`가 주어지면:

```
dL/dx[i] = p[i] * (dL/dp[i] - dot(dL/dp, p))
```

```c
void softmax_backward(float *dx, const float *dp, const float *p, size_t N) {
    // dot = sum_j dp[j] * p[j]
    float dot = 0.0f;
    for (size_t j = 0; j < N; j++) dot += dp[j] * p[j];

    for (size_t i = 0; i < N; i++)
        dx[i] += p[i] * (dp[i] - dot);
}
```

---

## 3. Cross-Entropy 손실

### 순방향

타겟 클래스 `t`와 로짓 `x`에 대한 단일 예시:

```
L = -log(softmax(x)[t])
  = -x[t] + log(sum_j exp(x[j] - max(x))) + max(x)
```

수치적으로 안정한 구현:

```c
float cross_entropy_forward(const float *logits, int target, size_t N) {
    float m = logits[0];
    for (size_t i = 1; i < N; i++) if (logits[i] > m) m = logits[i];

    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) sum += expf(logits[i] - m);

    return -(logits[target] - m) + logf(sum);
}
```

### 퓨전된 Softmax-Cross-Entropy 역방향

softmax backward와 cross-entropy backward를 분리하는 대신, 퓨전된 형태가 더 간단합니다:

```
dL/dx[i] = softmax(x)[i] - 1{i == t}
```

유도:
```
L = -log(p[t]), p = softmax(x)

dL/dx[i] = p[i] - 1{i==t}
```

```c
void softmax_crossentropy_backward(float *dx, const float *logits,
                                   int target, size_t N) {
    // dx에 softmax(logits) 계산
    float m = logits[0];
    for (size_t i = 1; i < N; i++) if (logits[i] > m) m = logits[i];
    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) { dx[i] = expf(logits[i] - m); sum += dx[i]; }
    for (size_t i = 0; i < N; i++) dx[i] /= sum;

    // 타겟에서 1 빼기
    dx[target] -= 1.0f;
}
```

---

## 4. 전체 MLP: 순방향 + 역방향 테스트

```c
static void test_mlp_gradients(void) {
    // 소형 네트워크: 입력 2 → 은닉 4 → 출력 2
    float W1_data[] = {0.1f, -0.2f, 0.3f, 0.4f,
                       0.5f,  0.1f,-0.3f, 0.2f};
    float W2_data[] = {0.3f, -0.1f, 0.4f, 0.2f,
                      -0.2f,  0.5f,-0.1f, 0.3f};
    float x_data[]  = {1.0f, 0.0f};
    int   target    = 1;

    AGNode *x  = ag_leaf(x_data,  2, (size_t[]){1,2}, false);
    AGNode *W1 = ag_leaf(W1_data, 2, (size_t[]){2,4}, true);
    AGNode *W2 = ag_leaf(W2_data, 2, (size_t[]){4,2}, true);

    // 순방향
    AGNode *h1    = ag_relu(ag_matmul(x, W1));   // [1,4]
    AGNode *logits = ag_matmul(h1, W2);           // [1,2]
    AGNode *loss   = ag_cross_entropy(logits, target);

    printf("loss = %.6f\n", loss->tensor->data[0]);

    // 역방향
    ag_backward(loss);

    // 그래디언트 검사
    gradient_check_node(loss, W1, 1e-4f, 1e-3f, "W1");
    gradient_check_node(loss, W2, 1e-4f, 1e-3f, "W2");
}
```

**예상 출력**:
```
loss = 0.693147   (log(2) — 무작위 초기화 근처 50/50)
그래디언트 검사 W1: 통과 (eps=1.0e-04, rtol=1.0e-03)
그래디언트 검사 W2: 통과
```

---

## 5. 그래디언트 누적 vs 그래디언트 초기화

핵심 정확도 문제: 그래디언트는 역방향 호출 간에 **누적됩니다**. 학습 스텝 사이에 0으로 설정해야 합니다.

```c
void ag_zero_grad(AGNode *node) {
    if (node == NULL || node->visited) return;
    node->visited = true;

    if (node->grad) {
        memset(node->grad->data, 0, node->grad->numel * sizeof(float));
    }
    for (int i = 0; i < AUTOGRAD_MAX_INPUTS; i++)
        ag_zero_grad(node->inputs[i]);
}
```

---

## 6. 역방향 공식 요약

| 연산 | 순방향 | dL/d(입력) |
|------|--------|-----------|
| `C = A @ B` | — | `dL/dA = dL/dC @ B^T`, `dL/dB = A^T @ dL/dC` |
| `y = x + b` | — | `dL/dx = dL/dy`, `dL/db = sum(dL/dy, axis=0)` |
| `y = relu(x)` | `x > 0 ? x : 0` | `dL/dx = dL/dy * (x > 0)` |
| `p = softmax(x)` | `exp(x-max)/sum` | `dL/dx[i] = p[i]*(dL/dp[i] - dot(dL/dp, p))` |
| `L = CE(x, t)` | `-log(p[t])` | `dL/dx[i] = p[i] - 1{i==t}` |
| `y = LayerNorm(x)` | `(x-μ)/σ*γ+β` | L24에서 유도 |
| `C = attention(Q,K,V)` | `softmax(QK^T/√d)V` | L37에서 유도 |

---

## 핵심 요약

- **Matmul 역방향**: `dA = dC @ B^T`, `dB = A^T @ dC` — 미분되지 않는 피연산자를 전치
- **Softmax 역방향**: `dx[i] = p[i] * (dp[i] - dot(dp, p))` — 순방향 출력 `p` 필요
- **퓨전 CE 역방향**: `dx[i] = softmax(x)[i] - 1{i==t}` — softmax+CE 합성보다 간단
- 항상 유한 차분으로 검증; 부호 오류나 누락된 전치만으로도 학습이 망가짐
- 각 역방향 패스 전에 그래디언트를 0으로 설정

---

**다음**: [07. 메모리 관리자](./07_Memory_Manager.md) — Arena allocator와 참조 카운팅된 텐서 풀을 구축하여 추론 중 `malloc`/`free` 오버헤드를 제거합니다.
