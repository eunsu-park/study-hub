# 05. Autograd 엔진

**이전**: [최적화된 행렬 곱셈](./04_Optimized_Matmul.md) | **다음**: [Autograd 텐서 연산](./06_Autograd_Tensor_Ops.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 연산 그래프가 무엇이며 자동 미분을 가능하게 하는 이유 설명
2. 연결 리스트와 함수 포인터를 사용하여 C로 동적 연산 그래프 구현
3. C에서 그래프에 대한 위상 정렬 수행
4. `backward()`를 호출하여 그래프를 통해 그래디언트 전파
5. 유한 차분법을 사용하여 그래디언트 검증

---

## 1. 자동 미분이란?

연산 시퀀스로 계산된 함수 `f(x)`가 주어지면, **autograd**는 프로그래머가 수동으로 미분을 작성하지 않아도 `df/dx`를 계산합니다.

두 가지 모드가 있습니다:
- **순방향 모드**: 미분을 앞으로 전파 (`f: R → R^n`에 효율적)
- **역방향 모드 (역전파)**: 출력에서 입력으로 전파 (`f: R^n → R`에 효율적 — 전형적인 DL 케이스)

우리는 **역방향 모드 AD**를 구현합니다.

### 연쇄 법칙

`z = f(y)`이고 `y = g(x)`이면:
```
dL/dx = dL/dz * dz/dy * dy/dx
```

전체 연산 그래프 `L = f_n(f_{n-1}(...f_1(x)...))`의 경우:
```
dL/dx = dL/df_n * df_n/df_{n-1} * ... * df_1/dx
```

Autograd는 이것을 오른쪽에서 왼쪽으로 평가합니다: `dL/dL = 1`에서 시작하여 각 연산을 통해 역방향으로 전파합니다.

---

## 2. 연산 그래프

그래프의 각 텐서는 **노드**입니다. 연산은 노드 간의 엣지를 만듭니다.

```
순방향 패스:
  x → [matmul] → y → [relu] → z → [sum] → L

역방향 패스 (역순):
  L → dL/dz=1 → [relu_backward] → dL/dy → [matmul_backward] → dL/dx
```

### 노드 구조

```c
// autograd.h
#pragma once
#include "tensor.h"
#include <stddef.h>

#define AUTOGRAD_MAX_INPUTS 4

typedef struct AGNode {
    Tensor *tensor;          // 이 연산의 출력 텐서
    Tensor *grad;            // 이 노드에 대한 그래디언트 (tensor와 같은 shape)

    // 이 노드를 생성한 연산
    void (*backward_fn)(struct AGNode *node);
    void *ctx;               // 역방향을 위한 저장된 컨텍스트 (입력, 기타 데이터)

    // 이 노드가 의존하는 입력들
    struct AGNode *inputs[AUTOGRAD_MAX_INPUTS];
    int n_inputs;

    // 위상 정렬 북키핑
    bool visited;
    bool requires_grad;
} AGNode;

// 텐서를 감싸는 노드 할당
AGNode *ag_node_new(Tensor *tensor, bool requires_grad);
void    ag_node_free(AGNode *node);

// 핵심 autograd 연산
void    ag_backward(AGNode *root);          // 전체 역방향 패스 실행
void    ag_zero_grad(AGNode *node);         // 그래디언트 재귀적으로 0으로 설정
```

---

## 3. 위상 정렬

`backward()`는 **역 위상 순서**로 노드를 방문해야 합니다: 각 노드의 그래디언트는 입력으로 전파되기 전에 완전히 계산되어야 합니다.

```c
// autograd.c
#include "autograd.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    AGNode **nodes;
    int      count;
    int      capacity;
} NodeList;

static void nodelist_push(NodeList *list, AGNode *node) {
    if (list->count >= list->capacity) {
        list->capacity = list->capacity * 2 + 8;
        list->nodes = realloc(list->nodes, list->capacity * sizeof(AGNode *));
    }
    list->nodes[list->count++] = node;
}

// DFS 후위 순회 → 위상 순서 (역순)
static void topo_dfs(AGNode *node, NodeList *order) {
    if (node == NULL || node->visited) return;
    node->visited = true;

    for (int i = 0; i < AUTOGRAD_MAX_INPUTS && node->inputs[i] != NULL; i++)
        topo_dfs(node->inputs[i], order);

    nodelist_push(order, node);  // 후위: 노드 다음 입력들
}
```

---

## 4. 역방향 패스

```c
void ag_backward(AGNode *root) {
    // 루트 그래디언트 초기화 = 1.0 (스칼라 손실)
    if (root->grad == NULL) {
        root->grad = tensor_zeros(root->tensor->ndim, root->tensor->shape);
        for (size_t i = 0; i < root->grad->numel; i++)
            root->grad->data[i] = 1.0f;
    }

    // 위상 순서 빌드
    NodeList order = {NULL, 0, 0};
    topo_dfs(root, &order);

    // 역 위상 순서로 방문 (루트 → 리프)
    for (int i = order.count - 1; i >= 0; i--) {
        AGNode *node = order.nodes[i];
        if (node->backward_fn != NULL && node->requires_grad) {
            // 모든 입력 노드에 grad 버퍼 확보
            for (int j = 0; j < AUTOGRAD_MAX_INPUTS; j++) {
                if (node->inputs[j] && node->inputs[j]->requires_grad &&
                    node->inputs[j]->grad == NULL) {
                    node->inputs[j]->grad = tensor_zeros(
                        node->inputs[j]->tensor->ndim,
                        node->inputs[j]->tensor->shape);
                }
            }
            node->backward_fn(node);
        }
    }

    free(order.nodes);
}
```

---

## 5. 예시: 스칼라 Autograd

`add`와 `mul`을 스칼라 값으로 구현하여 엔진을 검증해 봅시다:

```c
// add backward를 위한 저장된 컨텍스트
typedef struct { AGNode *a; AGNode *b; } AddCtx;

static void add_backward(AGNode *node) {
    AddCtx *ctx = (AddCtx *)node->ctx;
    for (size_t i = 0; i < node->grad->numel; i++) {
        // dL/da += dL/d(out) * 1
        if (ctx->a->requires_grad)
            ctx->a->grad->data[i] += node->grad->data[i];
        // dL/db += dL/d(out) * 1
        if (ctx->b->requires_grad)
            ctx->b->grad->data[i] += node->grad->data[i];
    }
}

AGNode *ag_add(AGNode *a, AGNode *b) {
    assert(a->tensor->numel == b->tensor->numel);

    // 순방향 계산
    Tensor *out_t = tensor_zeros(a->tensor->ndim, a->tensor->shape);
    for (size_t i = 0; i < a->tensor->numel; i++)
        out_t->data[i] = a->tensor->data[i] + b->tensor->data[i];

    AGNode *out    = ag_node_new(out_t, a->requires_grad || b->requires_grad);
    out->inputs[0] = a;
    out->inputs[1] = b;
    out->n_inputs  = 2;

    AddCtx *ctx = malloc(sizeof(AddCtx));
    ctx->a = a; ctx->b = b;
    out->ctx          = ctx;
    out->backward_fn  = add_backward;  // 명명된 함수 포인터

    return out;
}
```

---

## 6. 유한 차분 검증

**항상 수치적으로 그래디언트를 검증하세요** — autograd 구현을 믿기 전에.

```c
// 확인: df/dx[i] ≈ (f(x + ε*e_i) - f(x - ε*e_i)) / (2ε)
void gradient_check(AGNode *(*forward_fn)(AGNode *), AGNode *x,
                    float eps, float rtol) {
    // 순방향 + 역방향 실행
    AGNode *out = forward_fn(x);
    ag_backward(out);
    float *analytic = x->grad->data;

    // 수치 그래디언트 계산
    float *numeric = calloc(x->tensor->numel, sizeof(float));
    for (size_t i = 0; i < x->tensor->numel; i++) {
        float orig = x->tensor->data[i];

        x->tensor->data[i] = orig + eps;
        AGNode *out_plus = forward_fn(x);
        float f_plus = out_plus->tensor->data[0];

        x->tensor->data[i] = orig - eps;
        AGNode *out_minus = forward_fn(x);
        float f_minus = out_minus->tensor->data[0];

        numeric[i] = (f_plus - f_minus) / (2.0f * eps);
        x->tensor->data[i] = orig;  // 복원
    }

    // 비교
    bool passed = true;
    for (size_t i = 0; i < x->tensor->numel; i++) {
        float rel_err = fabsf(analytic[i] - numeric[i]) /
                        (fabsf(numeric[i]) + 1e-8f);
        if (rel_err > rtol) {
            printf("실패 인덱스 %zu: analytic=%.6f numeric=%.6f err=%.4f\n",
                   i, analytic[i], numeric[i], rel_err);
            passed = false;
        }
    }
    printf("그래디언트 검사: %s (eps=%.1e, rtol=%.1e)\n",
           passed ? "통과" : "실패", eps, rtol);
    free(numeric);
}
```

---

## 7. 전체 예시: MLP 순방향 + 역방향

```c
int main(void) {
    // 입력 x [2, 3], 가중치 W [3, 2]
    size_t x_shape[] = {2, 3};
    size_t W_shape[] = {3, 2};
    Tensor *x_t = tensor_zeros(2, x_shape);
    Tensor *W_t = tensor_zeros(2, W_shape);

    // 테스트 값으로 채우기
    float x_vals[] = {1,2,3, 4,5,6};
    float W_vals[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    memcpy(x_t->data, x_vals, sizeof(x_vals));
    memcpy(W_t->data, W_vals, sizeof(W_vals));

    AGNode *x = ag_node_new(x_t, true);
    AGNode *W = ag_node_new(W_t, true);

    // 순방향: h = x @ W  → [2, 2]
    AGNode *h    = ag_matmul(x, W);
    AGNode *loss = ag_sum(h);

    printf("loss = %.4f\n", loss->tensor->data[0]);

    // 역방향
    ag_backward(loss);

    printf("dL/dW:\n");
    tensor_print(W->grad, "W.grad");

    return 0;
}
```

---

## 핵심 요약

- 연산 그래프는 순방향 패스 중에 *어떤 연산이 어떤 텐서를 생성했는지*를 기록합니다
- `backward_fn`은 각 노드에 저장된 함수 포인터입니다; `grad`를 노드의 입력에 누적합니다
- 위상 정렬은 각 노드의 그래디언트가 입력으로 전파되기 전에 완전히 누적되도록 보장합니다
- **그래디언트 누적** (`+=`, `=` 아님): 그래프를 통한 여러 경로는 기여도를 합산해야 합니다
- 새로운 역방향 구현을 신뢰하기 전에 항상 그래디언트 검사를 실행하세요

---

**다음**: [06. Autograd 텐서 연산](./06_Autograd_Tensor_Ops.md) — matmul, softmax, cross-entropy의 역방향 패스를 구현하고 유한 차분으로 검증합니다.
