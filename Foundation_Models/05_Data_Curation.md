# 05. 데이터 큐레이션

## 개요

Foundation Model의 성능은 데이터 품질과 다양성에 크게 의존합니다. "Garbage in, garbage out"이 그 어느 때보다 중요합니다. 이 레슨에서는 대규모 사전학습 데이터셋의 구축, 정제, 관리 방법을 다룹니다.

---

## 1. 주요 Pre-training 데이터셋

### 1.1 데이터셋 개요

```
┌──────────────────────────────────────────────────────────────────┐
│                Pre-training 데이터셋 진화                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2018: BookCorpus + Wikipedia (3.3B 토큰) → BERT                │
│         │                                                        │
│  2019: WebText (40GB, Reddit 링크) → GPT-2                      │
│         │                                                        │
│  2020: C4 (750GB, Common Crawl 정제) → T5                       │
│         │                                                        │
│  2020: The Pile (825GB, 22개 소스) → GPT-Neo, Pythia            │
│         │                                                        │
│  2022: ROOTS (1.6TB, 59언어) → BLOOM                            │
│         │                                                        │
│  2023: RedPajama (1.2T 토큰) → RedPajama-INCITE                 │
│         │                                                        │
│  2024: FineWeb (15T 토큰) → 최신 오픈 모델들                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 주요 데이터셋 비교

| 데이터셋 | 크기 | 소스 | 특징 |
|----------|------|------|------|
| **The Pile** | 825GB | 22개 다양한 소스 | 코드, 학술, 책 포함 |
| **C4** | 750GB | Common Crawl | 영어만, 필터링됨 |
| **RedPajama** | 1.2T 토큰 | LLaMA 레시피 복제 | 오픈소스 |
| **ROOTS** | 1.6TB | 59개 언어 | 다국어, BigScience |
| **FineWeb** | 15T 토큰 | Common Crawl | HuggingFace, 최신 |
| **Dolma** | 3T 토큰 | 다양한 소스 | Allen AI, 투명성 강조 |

### 1.3 The Pile 구성

```python
# The Pile의 22개 하위 데이터셋
PILE_COMPONENTS = {
    # 웹 텍스트
    'Pile-CC': 227.12,      # Common Crawl 정제
    'OpenWebText2': 62.77,  # Reddit 링크 웹페이지

    # 책과 문학
    'Books3': 100.96,       # 도서
    'BookCorpus2': 6.30,    # 추가 도서
    'Gutenberg': 10.88,     # 공개 도서

    # 학술
    'PubMed Central': 90.27,   # 의학 논문
    'ArXiv': 56.21,            # 과학 논문
    'PubMed Abstracts': 19.26, # 논문 초록
    'PhilPapers': 2.38,        # 철학 논문
    'NIH ExPorter': 1.89,      # NIH 연구 정보

    # 코드
    'Github': 95.16,        # 깃허브 코드
    'StackExchange': 32.20, # Q&A

    # 기타
    'Wikipedia (en)': 16.11,
    'FreeLaw': 51.15,       # 법률 문서
    'USPTO': 22.90,         # 특허
    'DM Mathematics': 7.75, # 수학 문제
    'Ubuntu IRC': 5.52,     # IRC 로그
    'EuroParl': 4.59,       # EU 의회
    'HackerNews': 3.90,
    'YoutubeSubtitles': 3.73,
    'Enron Emails': 0.88,
}

# 비율 계산
total = sum(PILE_COMPONENTS.values())
for name, size in sorted(PILE_COMPONENTS.items(), key=lambda x: -x[1])[:5]:
    print(f"{name}: {size:.1f}GB ({size/total*100:.1f}%)")
```

---

## 2. 데이터 수집

### 2.1 Common Crawl 활용

```python
import gzip
import json
from warcio.archiveiterator import ArchiveIterator
import requests

class CommonCrawlExtractor:
    """Common Crawl에서 텍스트 추출"""

    CC_INDEX_URL = "https://index.commoncrawl.org/CC-MAIN-2024-10-index"

    def fetch_warc_paths(self, domain: str, limit: int = 100) -> list[str]:
        """특정 도메인의 WARC 파일 경로 조회"""
        params = {
            'url': f'*.{domain}/*',
            'output': 'json',
            'limit': limit
        }
        response = requests.get(self.CC_INDEX_URL, params=params)
        return [json.loads(line)['filename'] for line in response.text.strip().split('\n')]

    def extract_text_from_warc(self, warc_url: str) -> list[dict]:
        """WARC 파일에서 텍스트 추출"""
        results = []

        response = requests.get(
            f"https://data.commoncrawl.org/{warc_url}",
            stream=True
        )

        with gzip.open(response.raw, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type == 'response':
                    url = record.rec_headers.get_header('WARC-Target-URI')
                    content = record.content_stream().read().decode('utf-8', errors='ignore')

                    # HTML에서 텍스트 추출 (trafilatura 등 사용)
                    text = self.extract_text(content)

                    if text:
                        results.append({
                            'url': url,
                            'text': text,
                            'timestamp': record.rec_headers.get_header('WARC-Date')
                        })

        return results

    def extract_text(self, html: str) -> str:
        """HTML에서 본문 텍스트 추출"""
        try:
            import trafilatura
            return trafilatura.extract(html)
        except:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            # script, style 제거
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            return soup.get_text(separator=' ', strip=True)
```

### 2.2 GitHub 코드 수집

```python
import os
from github import Github
from typing import Generator

class GitHubCodeCollector:
    """GitHub에서 코드 수집"""

    # 수집할 언어와 확장자
    LANGUAGES = {
        'python': ['.py'],
        'javascript': ['.js', '.jsx', '.ts', '.tsx'],
        'java': ['.java'],
        'cpp': ['.cpp', '.hpp', '.c', '.h'],
        'go': ['.go'],
        'rust': ['.rs'],
    }

    def __init__(self, token: str):
        self.github = Github(token)

    def collect_repos(
        self,
        language: str,
        min_stars: int = 100,
        limit: int = 1000
    ) -> Generator[dict, None, None]:
        """인기 저장소 수집"""
        query = f"language:{language} stars:>{min_stars}"
        repos = self.github.search_repositories(query, sort='stars')

        for i, repo in enumerate(repos):
            if i >= limit:
                break

            yield {
                'name': repo.full_name,
                'stars': repo.stargazers_count,
                'language': repo.language,
                'license': repo.license.key if repo.license else None,
                'url': repo.html_url
            }

    def extract_code_files(
        self,
        repo_name: str,
        extensions: list[str]
    ) -> Generator[dict, None, None]:
        """저장소에서 코드 파일 추출"""
        repo = self.github.get_repo(repo_name)

        try:
            contents = repo.get_contents("")
            while contents:
                file_content = contents.pop(0)

                if file_content.type == "dir":
                    contents.extend(repo.get_contents(file_content.path))
                elif any(file_content.path.endswith(ext) for ext in extensions):
                    try:
                        content = file_content.decoded_content.decode('utf-8')
                        yield {
                            'path': file_content.path,
                            'content': content,
                            'size': file_content.size
                        }
                    except:
                        continue
        except Exception as e:
            print(f"Error processing {repo_name}: {e}")
```

---

## 3. 데이터 정제 파이프라인

### 3.1 품질 필터링

```python
import re
from typing import Optional
import fasttext
from collections import Counter

class QualityFilter:
    """텍스트 품질 필터링"""

    def __init__(self, lang_model_path: str = 'lid.176.bin'):
        # FastText 언어 감지 모델
        self.lang_detector = fasttext.load_model(lang_model_path)

    def filter_document(self, text: str, target_lang: str = 'en') -> Optional[str]:
        """
        문서 필터링

        Returns:
            정제된 텍스트 또는 None (필터링됨)
        """
        # 1. 기본 필터
        if not self._basic_filter(text):
            return None

        # 2. 언어 필터
        if not self._language_filter(text, target_lang):
            return None

        # 3. 품질 점수
        if not self._quality_score_filter(text):
            return None

        # 4. 텍스트 정제
        cleaned = self._clean_text(text)

        return cleaned if len(cleaned) > 100 else None

    def _basic_filter(self, text: str) -> bool:
        """기본 필터링 규칙"""
        # 최소/최대 길이
        if len(text) < 100 or len(text) > 100000:
            return False

        # 단어 수
        words = text.split()
        if len(words) < 20:
            return False

        # 평균 단어 길이 (너무 짧거나 긴 경우 스팸 가능성)
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3 or avg_word_len > 15:
            return False

        # 알파벳 비율
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars / len(text) < 0.6:
            return False

        return True

    def _language_filter(self, text: str, target_lang: str) -> bool:
        """언어 필터링"""
        # 첫 500자로 언어 감지
        sample = text[:500].replace('\n', ' ')
        predictions = self.lang_detector.predict(sample, k=1)

        lang = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]

        return lang == target_lang and confidence > 0.8

    def _quality_score_filter(self, text: str) -> bool:
        """품질 점수 기반 필터링"""
        lines = text.split('\n')

        # 줄 끝 구두점 비율
        end_punct = sum(1 for line in lines if line.strip() and line.strip()[-1] in '.!?')
        punct_ratio = end_punct / max(len(lines), 1)

        # 대문자로 시작하는 줄 비율
        cap_start = sum(1 for line in lines if line.strip() and line.strip()[0].isupper())
        cap_ratio = cap_start / max(len(lines), 1)

        # 불릿/번호 목록 비율 (너무 높으면 목록 페이지)
        bullet_lines = sum(1 for line in lines if re.match(r'^\s*[\-\*\•\d\.]\s', line))
        bullet_ratio = bullet_lines / max(len(lines), 1)

        # 품질 점수
        if punct_ratio < 0.3:  # 너무 적은 구두점
            return False
        if bullet_ratio > 0.5:  # 너무 많은 목록
            return False

        return True

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # URL 제거
        text = re.sub(r'https?://\S+', '', text)

        # 이메일 제거
        text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)

        # 과도한 공백 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # 제어 문자 제거
        text = ''.join(c for c in text if c.isprintable() or c in '\n\t')

        return text.strip()
```

### 3.2 중복 제거

```python
import hashlib
from datasketch import MinHash, MinHashLSH
from typing import Generator

class DeduplicationPipeline:
    """대규모 중복 제거 파이프라인"""

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        ngram_size: int = 5
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size

        # LSH 인덱스
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.seen_hashes = set()

    def get_minhash(self, text: str) -> MinHash:
        """텍스트의 MinHash 계산"""
        minhash = MinHash(num_perm=self.num_perm)

        # N-gram 생성
        words = text.lower().split()
        for i in range(len(words) - self.ngram_size + 1):
            ngram = ' '.join(words[i:i + self.ngram_size])
            minhash.update(ngram.encode('utf-8'))

        return minhash

    def exact_dedup(self, text: str) -> bool:
        """
        정확한 중복 제거 (해시 기반)

        Returns:
            True if unique, False if duplicate
        """
        # 정규화된 텍스트의 해시
        normalized = ' '.join(text.lower().split())
        text_hash = hashlib.md5(normalized.encode()).hexdigest()

        if text_hash in self.seen_hashes:
            return False

        self.seen_hashes.add(text_hash)
        return True

    def fuzzy_dedup(self, doc_id: str, text: str) -> bool:
        """
        퍼지 중복 제거 (MinHash LSH)

        Returns:
            True if unique, False if near-duplicate found
        """
        minhash = self.get_minhash(text)

        # 유사 문서 검색
        result = self.lsh.query(minhash)

        if result:
            return False

        # 새 문서 추가
        self.lsh.insert(doc_id, minhash)
        return True

    def deduplicate_stream(
        self,
        documents: Generator[dict, None, None]
    ) -> Generator[dict, None, None]:
        """
        스트리밍 중복 제거
        """
        for i, doc in enumerate(documents):
            text = doc['text']
            doc_id = doc.get('id', str(i))

            # 1단계: 정확한 중복
            if not self.exact_dedup(text):
                continue

            # 2단계: 유사 중복
            if not self.fuzzy_dedup(doc_id, text):
                continue

            yield doc


# 사용 예시
def deduplicate_dataset(input_path: str, output_path: str):
    """데이터셋 중복 제거"""
    pipeline = DeduplicationPipeline(threshold=0.85)

    def read_documents(path):
        with open(path, 'r') as f:
            for line in f:
                yield json.loads(line)

    unique_count = 0
    total_count = 0

    with open(output_path, 'w') as out:
        for doc in pipeline.deduplicate_stream(read_documents(input_path)):
            out.write(json.dumps(doc) + '\n')
            unique_count += 1
        total_count += 1

    print(f"Total: {total_count}, Unique: {unique_count}")
    print(f"Dedup ratio: {(1 - unique_count/total_count)*100:.1f}%")
```

---

## 4. 데이터 믹싱

### 4.1 도메인 믹싱 전략

```python
import numpy as np
from dataclasses import dataclass
from typing import Iterator

@dataclass
class DataSource:
    name: str
    path: str
    weight: float  # 샘플링 가중치
    quality_score: float  # 품질 점수 (0-1)

class DataMixer:
    """
    다양한 소스의 데이터 믹싱

    전략:
    1. 품질 기반: 고품질 소스 더 많이 샘플링
    2. 다양성 기반: 모든 도메인 균형있게
    3. Scaling law 기반: 최적 비율 탐색
    """

    # LLaMA 스타일 믹싱 비율
    LLAMA_MIX = {
        'CommonCrawl': 0.67,    # 웹
        'C4': 0.15,             # 정제된 웹
        'Github': 0.045,        # 코드
        'Wikipedia': 0.045,     # 백과사전
        'Books': 0.045,         # 도서
        'ArXiv': 0.025,         # 과학
        'StackExchange': 0.02,  # Q&A
    }

    def __init__(self, sources: list[DataSource]):
        self.sources = sources
        self.normalize_weights()

    def normalize_weights(self):
        """가중치 정규화"""
        total = sum(s.weight for s in self.sources)
        for source in self.sources:
            source.weight /= total

    def temperature_sampling(
        self,
        temperature: float = 1.0
    ) -> list[float]:
        """
        Temperature 기반 샘플링 확률 조정

        temperature < 1: 고빈도 소스에 집중
        temperature > 1: 균등하게 분산
        """
        weights = np.array([s.weight for s in self.sources])

        # Temperature 적용
        adjusted = np.power(weights, 1 / temperature)
        adjusted /= adjusted.sum()

        return adjusted.tolist()

    def sample_batch(
        self,
        batch_size: int,
        temperature: float = 1.0
    ) -> list[tuple[str, int]]:
        """
        배치 샘플링

        Returns:
            List of (source_name, num_samples)
        """
        probs = self.temperature_sampling(temperature)

        # 각 소스에서 샘플링할 문서 수
        samples = np.random.multinomial(batch_size, probs)

        return [
            (source.name, count)
            for source, count in zip(self.sources, samples)
        ]

    def iter_mixed_data(
        self,
        batch_size: int = 1000,
        temperature: float = 1.0
    ) -> Iterator[dict]:
        """혼합 데이터 이터레이터"""
        source_iters = {
            s.name: self._read_source(s.path)
            for s in self.sources
        }

        while True:
            batch_plan = self.sample_batch(batch_size, temperature)

            for source_name, count in batch_plan:
                for _ in range(count):
                    try:
                        yield next(source_iters[source_name])
                    except StopIteration:
                        # 소스 재시작 또는 종료
                        break

    @staticmethod
    def _read_source(path: str) -> Iterator[dict]:
        """데이터 소스 읽기"""
        with open(path, 'r') as f:
            for line in f:
                yield json.loads(line)


# 최적 믹싱 비율 탐색
def find_optimal_mix(
    sources: list[DataSource],
    validation_data: list,
    model_fn,
    n_trials: int = 20
) -> dict[str, float]:
    """
    Bayesian Optimization으로 최적 믹싱 비율 탐색
    """
    import optuna

    def objective(trial):
        # 각 소스의 가중치 샘플링
        weights = {}
        for source in sources:
            weights[source.name] = trial.suggest_float(
                source.name, 0.01, 1.0
            )

        # 정규화
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        # 모델 학습 및 검증
        # (실제로는 작은 모델로 프록시 실험)
        val_loss = model_fn(weights, validation_data)

        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
```

### 4.2 다국어 믹싱

```python
class MultilingualMixer:
    """
    다국어 데이터 믹싱

    전략:
    1. 영어 과대표집 방지
    2. 저자원 언어 업샘플링
    3. 언어 유사성 기반 그룹핑
    """

    # 언어별 기본 비율 (BLOOM 스타일)
    BLOOM_RATIOS = {
        'en': 0.30,  # 영어
        'zh': 0.15,  # 중국어
        'fr': 0.12,  # 프랑스어
        'es': 0.10,  # 스페인어
        'pt': 0.08,  # 포르투갈어
        'ar': 0.05,  # 아랍어
        # ... 기타 언어
    }

    def __init__(self, language_weights: dict[str, float]):
        self.language_weights = language_weights

    def exponential_smoothing(
        self,
        alpha: float = 0.3
    ) -> dict[str, float]:
        """
        지수 스무딩으로 저자원 언어 업샘플링

        P(lang) ∝ P_original(lang)^alpha

        alpha < 1: 저자원 언어 비율 증가
        alpha = 1: 원본 비율 유지
        """
        smoothed = {
            lang: weight ** alpha
            for lang, weight in self.language_weights.items()
        }

        total = sum(smoothed.values())
        return {lang: w/total for lang, w in smoothed.items()}

    def sample_by_language(
        self,
        documents: list[dict],
        target_ratio: dict[str, float]
    ) -> list[dict]:
        """언어별 목표 비율에 맞게 샘플링"""
        by_lang = {}
        for doc in documents:
            lang = doc.get('lang', 'en')
            by_lang.setdefault(lang, []).append(doc)

        sampled = []
        total_target = len(documents)

        for lang, ratio in target_ratio.items():
            if lang in by_lang:
                n_samples = int(total_target * ratio)
                lang_docs = by_lang[lang]

                if len(lang_docs) >= n_samples:
                    # 다운샘플링
                    sampled.extend(np.random.choice(lang_docs, n_samples, replace=False))
                else:
                    # 업샘플링
                    sampled.extend(np.random.choice(lang_docs, n_samples, replace=True))

        return sampled
```

---

## 5. 데이터 품질 평가

### 5.1 자동 품질 점수

```python
import kenlm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DataQualityScorer:
    """데이터 품질 자동 평가"""

    def __init__(
        self,
        perplexity_model_path: str = None,
        classifier_model_name: str = None
    ):
        # 1. Perplexity 기반 (KenLM)
        if perplexity_model_path:
            self.lm = kenlm.Model(perplexity_model_path)
        else:
            self.lm = None

        # 2. 분류기 기반 (예: 위키피디아 vs 웹)
        if classifier_model_name:
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                classifier_model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(classifier_model_name)
        else:
            self.classifier = None

    def perplexity_score(self, text: str) -> float:
        """
        KenLM perplexity 점수

        낮을수록 고품질 (언어 모델에 자연스러운 텍스트)
        """
        if self.lm is None:
            return 0.0

        # 문장 단위 perplexity
        score = self.lm.score(text, bos=True, eos=True)
        perplexity = 10 ** (-score / len(text.split()))

        return perplexity

    def classifier_score(self, text: str) -> float:
        """
        품질 분류기 점수 (0-1)

        높을수록 고품질
        """
        if self.classifier is None:
            return 0.5

        inputs = self.tokenizer(
            text[:512],
            return_tensors='pt',
            truncation=True
        )

        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # positive class 확률
        return probs[0, 1].item()

    def heuristic_score(self, text: str) -> dict[str, float]:
        """휴리스틱 기반 품질 점수"""
        lines = text.split('\n')
        words = text.split()

        scores = {
            # 1. 알파벳 비율
            'alpha_ratio': sum(c.isalpha() for c in text) / max(len(text), 1),

            # 2. 줄당 평균 단어 수
            'words_per_line': len(words) / max(len(lines), 1),

            # 3. 중복 줄 비율
            'unique_lines_ratio': len(set(lines)) / max(len(lines), 1),

            # 4. 구두점 비율
            'punct_ratio': sum(c in '.,!?;:' for c in text) / max(len(text), 1),

            # 5. 대문자 비율 (너무 높으면 스팸)
            'caps_ratio': sum(c.isupper() for c in text) / max(len(text), 1),

            # 6. 숫자 비율
            'digit_ratio': sum(c.isdigit() for c in text) / max(len(text), 1),
        }

        return scores

    def combined_score(self, text: str) -> float:
        """종합 품질 점수"""
        heuristics = self.heuristic_score(text)

        # 각 휴리스틱의 이상적 범위
        score = 1.0

        # 알파벳 비율: 0.7-0.9 이상적
        if heuristics['alpha_ratio'] < 0.6:
            score *= 0.8

        # 대문자 비율: 0.1 이하 이상적
        if heuristics['caps_ratio'] > 0.3:
            score *= 0.7

        # 중복 줄: 0.8 이상 이상적
        if heuristics['unique_lines_ratio'] < 0.5:
            score *= 0.6

        # Perplexity 점수 (낮을수록 좋음)
        ppl = self.perplexity_score(text)
        if ppl > 1000:
            score *= 0.5
        elif ppl > 500:
            score *= 0.8

        return score
```

---

## 6. 실습: FineWeb 스타일 파이프라인

```python
class FineWebPipeline:
    """
    FineWeb 스타일 데이터 파이프라인

    단계:
    1. URL 필터링
    2. 텍스트 추출
    3. 언어 감지
    4. 품질 필터링
    5. 중복 제거
    6. PII 제거
    """

    def __init__(self):
        self.quality_filter = QualityFilter()
        self.dedup = DeduplicationPipeline()
        self.quality_scorer = DataQualityScorer()

    def process_batch(
        self,
        warc_batch: list[dict]
    ) -> list[dict]:
        """배치 처리"""
        results = []

        for record in warc_batch:
            # 1. URL 필터링
            if not self._url_filter(record['url']):
                continue

            # 2. 텍스트 추출
            text = self._extract_text(record['html'])
            if not text:
                continue

            # 3. 품질 필터링
            text = self.quality_filter.filter_document(text)
            if not text:
                continue

            # 4. 품질 점수
            score = self.quality_scorer.combined_score(text)
            if score < 0.5:
                continue

            # 5. PII 마스킹
            text = self._mask_pii(text)

            results.append({
                'url': record['url'],
                'text': text,
                'quality_score': score
            })

        # 6. 중복 제거
        return list(self.dedup.deduplicate_stream(iter(results)))

    def _url_filter(self, url: str) -> bool:
        """URL 기반 필터링"""
        # 블랙리스트 도메인
        blacklist = ['porn', 'xxx', 'adult', 'gambling']
        if any(b in url.lower() for b in blacklist):
            return False

        # 허용 확장자
        if any(url.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif']):
            return False

        return True

    def _extract_text(self, html: str) -> str:
        """HTML에서 본문 추출"""
        import trafilatura
        return trafilatura.extract(html) or ''

    def _mask_pii(self, text: str) -> str:
        """개인정보 마스킹"""
        import re

        # 이메일
        text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)

        # 전화번호 (미국 형식)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

        # IP 주소
        text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)

        # 신용카드
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)

        return text


# 실행
if __name__ == "__main__":
    pipeline = FineWebPipeline()

    # Common Crawl 배치 처리
    warc_batch = [...]  # WARC 레코드

    cleaned_data = pipeline.process_batch(warc_batch)

    print(f"입력: {len(warc_batch)}, 출력: {len(cleaned_data)}")
    print(f"필터링 비율: {(1 - len(cleaned_data)/len(warc_batch))*100:.1f}%")
```

---

## 참고 자료

### 데이터셋
- [The Pile](https://pile.eleuther.ai/)
- [RedPajama](https://github.com/togethercomputer/RedPajama-Data)
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [Dolma](https://github.com/allenai/dolma)

### 논문
- Gao et al. (2020). "The Pile: An 800GB Dataset of Diverse Text"
- Penedo et al. (2023). "The RefinedWeb Dataset for Falcon LLM"
- Soldaini et al. (2024). "Dolma: An Open Corpus of 3T Tokens"

### 도구
- [trafilatura](https://github.com/adbar/trafilatura): HTML 텍스트 추출
- [datasketch](https://github.com/ekzhu/datasketch): MinHash LSH
- [fasttext](https://fasttext.cc/): 언어 감지
