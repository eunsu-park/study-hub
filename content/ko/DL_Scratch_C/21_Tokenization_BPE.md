# 21. 토크나이제이션과 BPE

**이전**: [현대 CNN 벤치마크](./20_Modern_CNN_Benchmark.md) | **다음**: [임베딩 테이블](./22_Embedding_Table.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 언어 모델에 토크나이제이션이 필요한 이유 설명하기
2. BPE (Byte Pair Encoding, 바이트 쌍 인코딩) 병합 알고리즘을 처음부터 구현하기
3. GPT-2와 GPT-4에서 사용하는 바이트 수준 BPE 설명하기
4. C 언어로 사전 학습된 GPT-2 어휘 (encoder.json + merges.bpe)를 로드하고 적용하기
5. 로드한 어휘를 사용해 텍스트를 인코딩하고 디코딩하기

---

## 1. 토크나이제이션이 왜 필요한가?

신경망은 원시 텍스트가 아닌 고정 크기의 수치 벡터를 처리합니다:

```
"Hello, world!" → [15496, 11, 995, 0]  (GPT-2 토큰 ID)

텍스트를 숫자로 변환하는 방법:
  문자 단위:    'H','e','l','l','o' → [72,101,108,108,111]
    장점: 어휘 크기가 작음 (~100개 문자)
    단점: 시퀀스가 매우 길어지고, 장거리 의존성 모델링이 어려움

  단어 단위:    "Hello" → 어휘의 15000번째 단어
    장점: 의미 단위로 표현
    단점: 어휘 크기가 매우 크고, 미지 단어(OOV 문제) 발생

  서브워드 (BPE): "Hello" → [15496]  또는  "unbelievable" → ["un","believ","able"]
    장점: 모든 단어 처리 가능, 컴파일 어휘 (~50K 토큰), OOV 없음
    단점: 토크나이저 학습이 필요
```

---

## 2. 바이트 쌍 인코딩 알고리즘

BPE는 개별 바이트/문자에서 시작하여 가장 빈번한 쌍을 반복적으로 병합합니다:

```
학습 알고리즘:
  1. 시작: 텍스트를 개별 문자로 분리 (바이트 수준 BPE는 바이트 단위)
  2. 카운트: 가장 빈번한 인접 쌍 찾기
  3. 병합: 해당 쌍의 모든 출현을 새 토큰으로 교체
  4. 반복: 어휘 크기가 목표에 도달할 때까지 (예: GPT-2는 50,257)

예시:
  말뭉치: "aaabdaaabac"
  초기: a a a b d a a a b a c    freq(aa)=4, freq(ab)=2, freq(bd)=1
  병합(a,a)→Z: Z a b d Z a b a c  freq(Za)=2, freq(ab)=2, freq(bd)=1
  병합(Z,a)→Y: Y b d Y b a c      ...
  목표 어휘 크기에 도달할 때까지 계속
```

### 최소 BPE 학습 구현

```c
#include <string.h>
#include <stdlib.h>

// 간소화된 BPE — 실제 구현에서는 성능을 위해 해시 맵 사용
typedef struct {
    int  a, b;       // 병합할 쌍
    int  result;     // 새 토큰 ID
} BPEMerge;

// 토큰 배열에서 가장 빈번한 인접 쌍 찾기
void find_best_pair(const int *tokens, int n,
                    int *best_a, int *best_b, int *best_freq) {
    *best_freq = 0;
    for (int i = 0; i < n - 1; i++) {
        int a = tokens[i], b = tokens[i+1];
        // 출현 횟수 카운트 (단순 O(n²) — 실제 구현에서는 해시맵 사용)
        int freq = 0;
        for (int j = 0; j < n - 1; j++)
            if (tokens[j] == a && tokens[j+1] == b) freq++;
        if (freq > *best_freq) {
            *best_freq = freq; *best_a = a; *best_b = b;
        }
    }
}

// (a,b)의 모든 출현을 new_id로 교체
int apply_merge(int *tokens, int n, int a, int b, int new_id) {
    int out = 0;
    for (int i = 0; i < n; ) {
        if (i < n-1 && tokens[i] == a && tokens[i+1] == b) {
            tokens[out++] = new_id;
            i += 2;
        } else {
            tokens[out++] = tokens[i++];
        }
    }
    return out;  // 새 길이
}

// BPE 학습 (병합 목록 반환)
BPEMerge *bpe_train(const int *corpus, int corpus_len,
                    int base_vocab, int target_vocab,
                    int *n_merges_out) {
    int n_merges = target_vocab - base_vocab;
    BPEMerge *merges = malloc(n_merges * sizeof(BPEMerge));
    int *tokens = malloc(corpus_len * sizeof(int));
    memcpy(tokens, corpus, corpus_len * sizeof(int));
    int n = corpus_len;
    int next_id = base_vocab;

    for (int m = 0; m < n_merges; m++) {
        int a, b, freq;
        find_best_pair(tokens, n, &a, &b, &freq);
        if (freq < 2) { n_merges = m; break; }  // 더 이상 병합할 것 없음
        merges[m] = (BPEMerge){ a, b, next_id };
        n = apply_merge(tokens, n, a, b, next_id);
        next_id++;
    }
    *n_merges_out = n_merges;
    free(tokens);
    return merges;
}
```

---

## 3. 바이트 수준 BPE (GPT-2 방식)

GPT-2는 문자가 아닌 바이트를 기본 어휘로 사용합니다 — OOV 없이 모든 유니코드를 처리합니다:

```
기본 어휘: 256개 바이트 값 (0-255)
병합: 50,000회 병합 연산 → 전체 어휘 = 50,256 + 1 (<|endoftext|>) = 50,257

바이트 매핑: 원시 바이트 → "출력 가능한" 표현으로 변환
  'A' (65) → 'A'
  ' ' (32) → 'Ġ'  (유니코드 공백 표현)
  '\n' (10) → 'Ċ'

장점:
  - 미지 토큰이 절대 없음 (모든 바이트 시퀀스 인코딩 가능)
  - 코드, 다국어 텍스트, 이모지 처리 가능
  - 컴팩트: 일반 영어 단어는 단일 토큰이 됨
```

---

## 4. C 언어로 GPT-2 토크나이저 로드하기

GPT-2는 두 파일을 공개합니다: `encoder.json` (어휘)과 `merges.bpe` (병합 규칙):

```c
#include <stdio.h>
#include <string.h>

#define MAX_VOCAB   50257
#define MAX_MERGES  50000
#define MAX_TOKEN_LEN 256

typedef struct {
    char    str[MAX_TOKEN_LEN];  // 토큰 문자열
    int     id;
} VocabEntry;

typedef struct {
    char first[MAX_TOKEN_LEN];
    char second[MAX_TOKEN_LEN];
} MergeRule;

typedef struct {
    VocabEntry  vocab[MAX_VOCAB];
    int         vocab_size;
    MergeRule   merges[MAX_MERGES];
    int         n_merges;
    // 역방향 맵: token_id → 문자열 (vocab[]의 인덱스)
    int         id_to_idx[MAX_VOCAB];
} Tokenizer;

// merges.bpe 로드 (텍스트 형식: 줄마다 "first second", 헤더 줄 건너뜀)
void load_merges(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path, "r");
    char line[512];
    fgets(line, sizeof(line), f);  // "#version: ..." 헤더 건너뜀
    tok->n_merges = 0;
    while (fgets(line, sizeof(line), f) && tok->n_merges < MAX_MERGES) {
        char a[256], b[256];
        if (sscanf(line, "%255s %255s", a, b) == 2) {
            strncpy(tok->merges[tok->n_merges].first,  a, MAX_TOKEN_LEN-1);
            strncpy(tok->merges[tok->n_merges].second, b, MAX_TOKEN_LEN-1);
            tok->n_merges++;
        }
    }
    fclose(f);
}

// 문자열을 GPT-2 토큰 ID로 인코딩 (간소화 — 실제 구현에서는 트라이 사용)
void tokenize(const Tokenizer *tok, const char *text,
              int *out_ids, int *out_len, int max_len) {
    // 1. 텍스트를 바이트 시퀀스로 변환 → 초기 토큰 문자열
    // 2. 낮은 병합 인덱스(높은 우선순위) 순서로 병합 규칙을 반복 적용
    // ... (전체 구현은 쌍에 대한 우선순위 큐를 사용)
    // 실제 구현은 tiktoken 또는 llm.c 참고
    *out_len = 0;
    printf("(간소화 버전: 전체 BPE는 llm.c 토크나이저로 구현하세요)\n");
}
```

### tiktoken의 C 호환 출력 활용하기

실용적인 사용을 위해 Python의 tiktoken으로 사전 토크나이징 후 저장하는 방법:

```bash
# Python으로 데이터셋을 사전 토크나이징하고 바이너리 int32 파일로 저장
python3 -c "
import tiktoken
enc = tiktoken.get_encoding('gpt2')
text = open('input.txt').read()
tokens = enc.encode(text)
import numpy as np
np.array(tokens, dtype=np.int32).tofile('tokens.bin')
print(f'Encoded {len(tokens)} tokens')
"
```

그런 다음 C에서 로드합니다:

```c
int *load_tokens(const char *path, int *n_tokens) {
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    *n_tokens = (int)(sz / sizeof(int32_t));
    int *tokens = malloc(*n_tokens * sizeof(int));
    fread(tokens, sizeof(int32_t), *n_tokens, f);
    fclose(f);
    return tokens;
}
```

---

## 5. 토큰 통계

영어 텍스트에 대한 GPT-2 토크나이제이션 통계:

```
단어당 평균 토큰 수: ~1.3  (가장 흔한 단어 = 1 토큰)
문자당 토큰 수:      ~0.3  (평균적으로 문자 3~4개당 토큰 1개)

예시 (GPT-2):
  "hello"        → [31373]             (1 토큰)
  " world"       → [995]               (1 토큰, 앞 공백 포함 인코딩)
  "GPT"          → [38, 11571]         (2 토큰 — "G", "PT")
  "tokenization" → [30001, 1634]       (2 토큰)
  "supercalifragilisticexpialidocious" → 12 토큰

압축 비율: tokens_per_char ≈ 0.25
  → 토큰 1개 ≈ 평균 4바이트 텍스트
  → 1K 토큰 컨텍스트 ≈ 영어 텍스트 750단어
```

---

## 6. 특수 토큰

GPT-2와 현대 LLM은 특수 토큰을 예약합니다:

```c
#define GPT2_EOT_TOKEN 50256  // <|endoftext|> — 문서 구분자
// Llama 3 추가 토큰:
// <|begin_of_text|> = 128000
// <|end_of_text|>   = 128001
// <|start_header_id|> = 128006  (명령 형식용)

// EOT를 시퀀스 구분자로 사용:
int tokens[MAX_SEQ];
tokens[0] = GPT2_EOT_TOKEN;  // 각 문서 앞에 추가
// 이후 문서 텍스트 토크나이징...
tokens[doc_len + 1] = GPT2_EOT_TOKEN;  // 끝에 추가
```

---

## 핵심 요약

- **BPE**: 바이트/문자에서 시작하여 가장 빈번한 인접 쌍을 반복적으로 병합 — OOV 없이 모든 입력을 처리하는 서브워드 어휘 생성
- GPT-2는 256개 기본 바이트 + 50,000회 병합 = 50,257 토큰의 **바이트 수준 BPE** 사용
- 토크나이저는 모델과 독립적: 대규모 데이터셋을 한 번 사전 토크나이징하고 바이너리 int32 파일로 저장
- 실제 C 코드에서는 Python tiktoken으로 사전 토크나이징에 의존; C 측은 바이너리 토큰 파일만 로드하면 됨
- 토큰 1개 ≈ 영어 텍스트 4문자; 컨텍스트 길이는 문자가 아닌 토큰 단위로 측정

---

**다음**: [22. 임베딩 테이블](./22_Embedding_Table.md) — 토큰 임베딩 룩업 테이블, 가중치 묶기(weight tying), HuggingFace 바이너리 형식에서 GPT-2 가중치 로드하기.
