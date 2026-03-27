# 05. Data Curation

## Learning Objectives

After completing this lesson, you will be able to:

1. Trace the evolution of major pre-training datasets (BookCorpus, C4, The Pile, RedPajama, FineWeb) and explain how scale and composition choices have influenced model capabilities.
2. Implement a data curation pipeline covering web crawl extraction, quality filtering (heuristics and model-based), deduplication (exact and near-duplicate), and domain mixing.
3. Apply quality signals such as perplexity filtering, language identification, and content safety classifiers to remove low-quality or harmful documents at scale.
4. Explain the role of data mixing ratios and domain upsampling strategies (e.g., data mixture weights used in LLaMA and Dolma) on downstream task performance.
5. Evaluate the ethical and legal considerations of using web-scraped data, including copyright, PII removal, and bias mitigation in large-scale corpora.
6. Design a reproducible data curation workflow with data versioning and documentation practices to enable transparency and reproducibility.

---

## Overview

The performance of Foundation Models heavily depends on data quality and diversity. "Garbage in, garbage out" is more critical than ever. This lesson covers the construction, refinement, and management of large-scale pre-training datasets.

---

## 1. Major Pre-training Datasets

### 1.1 Dataset Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                Pre-training Dataset Evolution                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2018: BookCorpus + Wikipedia (3.3B tokens) → BERT              │
│         │                                                        │
│  2019: WebText (40GB, Reddit links) → GPT-2                     │
│         │                                                        │
│  2020: C4 (750GB, Common Crawl filtered) → T5                   │
│         │                                                        │
│  2020: The Pile (825GB, 22 sources) → GPT-Neo, Pythia          │
│         │                                                        │
│  2022: ROOTS (1.6TB, 59 languages) → BLOOM                      │
│         │                                                        │
│  2023: RedPajama (1.2T tokens) → RedPajama-INCITE              │
│         │                                                        │
│  2024: FineWeb (15T tokens) → Latest open models               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Major Dataset Comparison

| Dataset | Size | Source | Features |
|---------|------|--------|----------|
| **The Pile** | 825GB | 22 diverse sources | Includes code, academic, books |
| **C4** | 750GB | Common Crawl | English only, filtered |
| **RedPajama** | 1.2T tokens | LLaMA recipe replication | Open source |
| **ROOTS** | 1.6TB | 59 languages | Multilingual, BigScience |
| **FineWeb** | 15T tokens | Common Crawl | HuggingFace, latest |
| **Dolma** | 3T tokens | Various sources | Allen AI, transparency focus |

### 1.3 The Pile Composition

```python
# The Pile's 22 subdatasets
PILE_COMPONENTS = {
    # Web text
    'Pile-CC': 227.12,      # Filtered Common Crawl
    'OpenWebText2': 62.77,  # Reddit-linked webpages

    # Books and literature
    'Books3': 100.96,       # Books
    'BookCorpus2': 6.30,    # Additional books
    'Gutenberg': 10.88,     # Public domain books

    # Academic
    'PubMed Central': 90.27,   # Medical papers
    'ArXiv': 56.21,            # Scientific papers
    'PubMed Abstracts': 19.26, # Paper abstracts
    'PhilPapers': 2.38,        # Philosophy papers
    'NIH ExPorter': 1.89,      # NIH research info

    # Code
    'Github': 95.16,        # GitHub code
    'StackExchange': 32.20, # Q&A

    # Other
    'Wikipedia (en)': 16.11,
    'FreeLaw': 51.15,       # Legal documents
    'USPTO': 22.90,         # Patents
    'DM Mathematics': 7.75, # Math problems
    'Ubuntu IRC': 5.52,     # IRC logs
    'EuroParl': 4.59,       # EU parliament
    'HackerNews': 3.90,
    'YoutubeSubtitles': 3.73,
    'Enron Emails': 0.88,
}

# Calculate ratios
total = sum(PILE_COMPONENTS.values())
for name, size in sorted(PILE_COMPONENTS.items(), key=lambda x: -x[1])[:5]:
    print(f"{name}: {size:.1f}GB ({size/total*100:.1f}%)")
```

---

## 2. Data Collection

### 2.1 Using Common Crawl

```python
import gzip
import json
from warcio.archiveiterator import ArchiveIterator
import requests

class CommonCrawlExtractor:
    """Extract text from Common Crawl"""

    CC_INDEX_URL = "https://index.commoncrawl.org/CC-MAIN-2024-10-index"

    def fetch_warc_paths(self, domain: str, limit: int = 100) -> list[str]:
        """Query WARC file paths for specific domain"""
        params = {
            'url': f'*.{domain}/*',
            'output': 'json',
            'limit': limit
        }
        response = requests.get(self.CC_INDEX_URL, params=params)
        return [json.loads(line)['filename'] for line in response.text.strip().split('\n')]

    def extract_text_from_warc(self, warc_url: str) -> list[dict]:
        """Extract text from WARC file"""
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

                    # Extract text from HTML (using trafilatura, etc.)
                    text = self.extract_text(content)

                    if text:
                        results.append({
                            'url': url,
                            'text': text,
                            'timestamp': record.rec_headers.get_header('WARC-Date')
                        })

        return results

    def extract_text(self, html: str) -> str:
        """Extract main text from HTML"""
        try:
            import trafilatura
            return trafilatura.extract(html)
        except:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            # Remove script, style
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            return soup.get_text(separator=' ', strip=True)
```

### 2.2 GitHub Code Collection

```python
import os
from github import Github
from typing import Generator

class GitHubCodeCollector:
    """Collect code from GitHub"""

    # Languages and extensions to collect
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
        """Collect popular repositories"""
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
        """Extract code files from repository"""
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

## 3. Data Cleaning Pipeline

### 3.1 Quality Filtering

```python
import re
from typing import Optional
import fasttext
from collections import Counter

class QualityFilter:
    """Text quality filtering"""

    def __init__(self, lang_model_path: str = 'lid.176.bin'):
        # FastText language detection model
        self.lang_detector = fasttext.load_model(lang_model_path)

    def filter_document(self, text: str, target_lang: str = 'en') -> Optional[str]:
        """
        Filter document

        Returns:
            Cleaned text or None (filtered out)
        """
        # 1. Basic filter
        if not self._basic_filter(text):
            return None

        # 2. Language filter
        if not self._language_filter(text, target_lang):
            return None

        # 3. Quality score
        if not self._quality_score_filter(text):
            return None

        # 4. Text cleaning
        cleaned = self._clean_text(text)

        return cleaned if len(cleaned) > 100 else None

    def _basic_filter(self, text: str) -> bool:
        """Basic filtering rules"""
        # Min/max length
        if len(text) < 100 or len(text) > 100000:
            return False

        # Word count
        words = text.split()
        if len(words) < 20:
            return False

        # Average word length (too short/long suggests spam)
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3 or avg_word_len > 15:
            return False

        # Alphabet ratio
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars / len(text) < 0.6:
            return False

        return True

    def _language_filter(self, text: str, target_lang: str) -> bool:
        """Language filtering"""
        # Detect language from first 500 chars
        sample = text[:500].replace('\n', ' ')
        predictions = self.lang_detector.predict(sample, k=1)

        lang = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]

        return lang == target_lang and confidence > 0.8

    def _quality_score_filter(self, text: str) -> bool:
        """Quality score-based filtering"""
        lines = text.split('\n')

        # Line-ending punctuation ratio
        end_punct = sum(1 for line in lines if line.strip() and line.strip()[-1] in '.!?')
        punct_ratio = end_punct / max(len(lines), 1)

        # Lines starting with capital letter ratio
        cap_start = sum(1 for line in lines if line.strip() and line.strip()[0].isupper())
        cap_ratio = cap_start / max(len(lines), 1)

        # Bullet/number list ratio (too high suggests list page)
        bullet_lines = sum(1 for line in lines if re.match(r'^\s*[\-\*\•\d\.]\s', line))
        bullet_ratio = bullet_lines / max(len(lines), 1)

        # Quality score
        if punct_ratio < 0.3:  # Too little punctuation
            return False
        if bullet_ratio > 0.5:  # Too many lists
            return False

        return True

    def _clean_text(self, text: str) -> str:
        """Clean text"""
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Remove emails
        text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)

        # Clean excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Remove control characters
        text = ''.join(c for c in text if c.isprintable() or c in '\n\t')

        return text.strip()
```

### 3.2 Deduplication

```python
import hashlib
from datasketch import MinHash, MinHashLSH
from typing import Generator

class DeduplicationPipeline:
    """Large-scale deduplication pipeline"""

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        ngram_size: int = 5
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size

        # LSH index
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.seen_hashes = set()

    def get_minhash(self, text: str) -> MinHash:
        """Calculate MinHash of text"""
        minhash = MinHash(num_perm=self.num_perm)

        # Generate N-grams
        words = text.lower().split()
        for i in range(len(words) - self.ngram_size + 1):
            ngram = ' '.join(words[i:i + self.ngram_size])
            minhash.update(ngram.encode('utf-8'))

        return minhash

    def exact_dedup(self, text: str) -> bool:
        """
        Exact deduplication (hash-based)

        Returns:
            True if unique, False if duplicate
        """
        # Hash of normalized text
        normalized = ' '.join(text.lower().split())
        text_hash = hashlib.md5(normalized.encode()).hexdigest()

        if text_hash in self.seen_hashes:
            return False

        self.seen_hashes.add(text_hash)
        return True

    def fuzzy_dedup(self, doc_id: str, text: str) -> bool:
        """
        Fuzzy deduplication (MinHash LSH)

        Returns:
            True if unique, False if near-duplicate found
        """
        minhash = self.get_minhash(text)

        # Search for similar documents
        result = self.lsh.query(minhash)

        if result:
            return False

        # Add new document
        self.lsh.insert(doc_id, minhash)
        return True

    def deduplicate_stream(
        self,
        documents: Generator[dict, None, None]
    ) -> Generator[dict, None, None]:
        """
        Streaming deduplication
        """
        for i, doc in enumerate(documents):
            text = doc['text']
            doc_id = doc.get('id', str(i))

            # Stage 1: Exact duplicates
            if not self.exact_dedup(text):
                continue

            # Stage 2: Near-duplicates
            if not self.fuzzy_dedup(doc_id, text):
                continue

            yield doc


# Usage example
def deduplicate_dataset(input_path: str, output_path: str):
    """Deduplicate dataset"""
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

## 4. Data Mixing

### 4.1 Domain Mixing Strategy

```python
import numpy as np
from dataclasses import dataclass
from typing import Iterator

@dataclass
class DataSource:
    name: str
    path: str
    weight: float  # Sampling weight
    quality_score: float  # Quality score (0-1)

class DataMixer:
    """
    Mix data from various sources

    Strategies:
    1. Quality-based: Sample more from high-quality sources
    2. Diversity-based: Balance all domains
    3. Scaling law-based: Search for optimal ratios
    """

    # LLaMA-style mixing ratios
    LLAMA_MIX = {
        'CommonCrawl': 0.67,    # Web
        'C4': 0.15,             # Filtered web
        'Github': 0.045,        # Code
        'Wikipedia': 0.045,     # Encyclopedia
        'Books': 0.045,         # Books
        'ArXiv': 0.025,         # Scientific
        'StackExchange': 0.02,  # Q&A
    }

    def __init__(self, sources: list[DataSource]):
        self.sources = sources
        self.normalize_weights()

    def normalize_weights(self):
        """Normalize weights"""
        total = sum(s.weight for s in self.sources)
        for source in self.sources:
            source.weight /= total

    def temperature_sampling(
        self,
        temperature: float = 1.0
    ) -> list[float]:
        """
        Adjust sampling probabilities with temperature

        temperature < 1: Focus on high-frequency sources
        temperature > 1: Distribute more evenly
        """
        weights = np.array([s.weight for s in self.sources])

        # Apply temperature
        adjusted = np.power(weights, 1 / temperature)
        adjusted /= adjusted.sum()

        return adjusted.tolist()

    def sample_batch(
        self,
        batch_size: int,
        temperature: float = 1.0
    ) -> list[tuple[str, int]]:
        """
        Sample batch

        Returns:
            List of (source_name, num_samples)
        """
        probs = self.temperature_sampling(temperature)

        # Number of documents to sample from each source
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
        """Mixed data iterator"""
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
                        # Restart source or terminate
                        break

    @staticmethod
    def _read_source(path: str) -> Iterator[dict]:
        """Read data source"""
        with open(path, 'r') as f:
            for line in f:
                yield json.loads(line)


# Search for optimal mixing ratios
def find_optimal_mix(
    sources: list[DataSource],
    validation_data: list,
    model_fn,
    n_trials: int = 20
) -> dict[str, float]:
    """
    Search for optimal mixing ratios with Bayesian Optimization
    """
    import optuna

    def objective(trial):
        # Sample weight for each source
        weights = {}
        for source in sources:
            weights[source.name] = trial.suggest_float(
                source.name, 0.01, 1.0
            )

        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        # Train model and validate
        # (In practice, use small proxy model)
        val_loss = model_fn(weights, validation_data)

        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
```

### 4.2 Multilingual Mixing

```python
class MultilingualMixer:
    """
    Multilingual data mixing

    Strategies:
    1. Prevent English over-representation
    2. Upsample low-resource languages
    3. Group by language similarity
    """

    # Default language ratios (BLOOM style)
    BLOOM_RATIOS = {
        'en': 0.30,  # English
        'zh': 0.15,  # Chinese
        'fr': 0.12,  # French
        'es': 0.10,  # Spanish
        'pt': 0.08,  # Portuguese
        'ar': 0.05,  # Arabic
        # ... other languages
    }

    def __init__(self, language_weights: dict[str, float]):
        self.language_weights = language_weights

    def exponential_smoothing(
        self,
        alpha: float = 0.3
    ) -> dict[str, float]:
        """
        Upsample low-resource languages with exponential smoothing

        P(lang) ∝ P_original(lang)^alpha

        alpha < 1: Increase low-resource language ratio
        alpha = 1: Keep original ratio
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
        """Sample to match target ratio per language"""
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
                    # Downsample
                    sampled.extend(np.random.choice(lang_docs, n_samples, replace=False))
                else:
                    # Upsample
                    sampled.extend(np.random.choice(lang_docs, n_samples, replace=True))

        return sampled
```

---

## 5. Data Quality Evaluation

### 5.1 Automatic Quality Scoring

```python
import kenlm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DataQualityScorer:
    """Automatic data quality evaluation"""

    def __init__(
        self,
        perplexity_model_path: str = None,
        classifier_model_name: str = None
    ):
        # 1. Perplexity-based (KenLM)
        if perplexity_model_path:
            self.lm = kenlm.Model(perplexity_model_path)
        else:
            self.lm = None

        # 2. Classifier-based (e.g., Wikipedia vs Web)
        if classifier_model_name:
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                classifier_model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(classifier_model_name)
        else:
            self.classifier = None

    def perplexity_score(self, text: str) -> float:
        """
        KenLM perplexity score

        Lower is better (more natural text for language model)
        """
        if self.lm is None:
            return 0.0

        # Sentence-level perplexity
        score = self.lm.score(text, bos=True, eos=True)
        perplexity = 10 ** (-score / len(text.split()))

        return perplexity

    def classifier_score(self, text: str) -> float:
        """
        Quality classifier score (0-1)

        Higher is better quality
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

        # Positive class probability
        return probs[0, 1].item()

    def heuristic_score(self, text: str) -> dict[str, float]:
        """Heuristic-based quality score"""
        lines = text.split('\n')
        words = text.split()

        scores = {
            # 1. Alphabet ratio
            'alpha_ratio': sum(c.isalpha() for c in text) / max(len(text), 1),

            # 2. Average words per line
            'words_per_line': len(words) / max(len(lines), 1),

            # 3. Unique lines ratio
            'unique_lines_ratio': len(set(lines)) / max(len(lines), 1),

            # 4. Punctuation ratio
            'punct_ratio': sum(c in '.,!?;:' for c in text) / max(len(text), 1),

            # 5. Uppercase ratio (too high suggests spam)
            'caps_ratio': sum(c.isupper() for c in text) / max(len(text), 1),

            # 6. Digit ratio
            'digit_ratio': sum(c.isdigit() for c in text) / max(len(text), 1),
        }

        return scores

    def combined_score(self, text: str) -> float:
        """Combined quality score"""
        heuristics = self.heuristic_score(text)

        # Ideal range for each heuristic
        score = 1.0

        # Alphabet ratio: 0.7-0.9 ideal
        if heuristics['alpha_ratio'] < 0.6:
            score *= 0.8

        # Uppercase ratio: < 0.1 ideal
        if heuristics['caps_ratio'] > 0.3:
            score *= 0.7

        # Unique lines: > 0.8 ideal
        if heuristics['unique_lines_ratio'] < 0.5:
            score *= 0.6

        # Perplexity score (lower is better)
        ppl = self.perplexity_score(text)
        if ppl > 1000:
            score *= 0.5
        elif ppl > 500:
            score *= 0.8

        return score
```

---

## 6. Practice: FineWeb-style Pipeline

```python
class FineWebPipeline:
    """
    FineWeb-style data pipeline

    Steps:
    1. URL filtering
    2. Text extraction
    3. Language detection
    4. Quality filtering
    5. Deduplication
    6. PII removal
    """

    def __init__(self):
        self.quality_filter = QualityFilter()
        self.dedup = DeduplicationPipeline()
        self.quality_scorer = DataQualityScorer()

    def process_batch(
        self,
        warc_batch: list[dict]
    ) -> list[dict]:
        """Process batch"""
        results = []

        for record in warc_batch:
            # 1. URL filtering
            if not self._url_filter(record['url']):
                continue

            # 2. Text extraction
            text = self._extract_text(record['html'])
            if not text:
                continue

            # 3. Quality filtering
            text = self.quality_filter.filter_document(text)
            if not text:
                continue

            # 4. Quality score
            score = self.quality_scorer.combined_score(text)
            if score < 0.5:
                continue

            # 5. PII masking
            text = self._mask_pii(text)

            results.append({
                'url': record['url'],
                'text': text,
                'quality_score': score
            })

        # 6. Deduplication
        return list(self.dedup.deduplicate_stream(iter(results)))

    def _url_filter(self, url: str) -> bool:
        """URL-based filtering"""
        # Blacklist domains
        blacklist = ['porn', 'xxx', 'adult', 'gambling']
        if any(b in url.lower() for b in blacklist):
            return False

        # Allowed extensions
        if any(url.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif']):
            return False

        return True

    def _extract_text(self, html: str) -> str:
        """Extract main text from HTML"""
        import trafilatura
        return trafilatura.extract(html) or ''

    def _mask_pii(self, text: str) -> str:
        """Mask personal information"""
        import re

        # Email
        text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)

        # Phone number (US format)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

        # IP address
        text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)

        # Credit card
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)

        return text


# Execute
if __name__ == "__main__":
    pipeline = FineWebPipeline()

    # Process Common Crawl batch
    warc_batch = [...]  # WARC records

    cleaned_data = pipeline.process_batch(warc_batch)

    print(f"Input: {len(warc_batch)}, Output: {len(cleaned_data)}")
    print(f"Filtering ratio: {(1 - len(cleaned_data)/len(warc_batch))*100:.1f}%")
```

---

## References

### Datasets
- [The Pile](https://pile.eleuther.ai/)
- [RedPajama](https://github.com/togethercomputer/RedPajama-Data)
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [Dolma](https://github.com/allenai/dolma)

### Papers
- Gao et al. (2020). "The Pile: An 800GB Dataset of Diverse Text"
- Penedo et al. (2023). "The RefinedWeb Dataset for Falcon LLM"
- Soldaini et al. (2024). "Dolma: An Open Corpus of 3T Tokens"

### Tools
- [trafilatura](https://github.com/adbar/trafilatura): HTML text extraction
- [datasketch](https://github.com/ekzhu/datasketch): MinHash LSH
- [fasttext](https://fasttext.cc/): Language detection

---

## Exercises

### Exercise 1: Quality Filter Design

You are building a quality filter for a web-scraped dataset. For each of the following documents, explain which heuristic filter(s) would flag it and why.

**Document A:**
```
BUY NOW!!! CLICK HERE!!! LIMITED TIME OFFER!!!
BEST PRICES GUARANTEED!!! ACT FAST!!!
BUY BUY BUY!!! DISCOUNT DISCOUNT!!!
```

**Document B:**
```
a a a a a a a a a a a a a a a a a a a a a a
b b b b b b b b b b b b b b b b b b b b b b
```

**Document C:**
```
def sort_array(arr): return sorted(arr)
x=1;y=2;z=x+y;print(z);a=[];for i in range(10):a.append(i)
```

<details>
<summary>Show Answer</summary>

**Document A: Spam/advertisement**
- **Uppercase ratio > 0.3**: Most characters are uppercase → `caps_ratio` flag
- **Punctuation ratio too high**: Excessive `!!!` sequences
- **Unique lines ratio too low**: Repetitive patterns ("BUY", "DISCOUNT") → low `unique_lines_ratio`
- **Perplexity**: KenLM would assign very high perplexity because these sequences don't resemble natural language distributions

**Document B: Repetitive/garbage**
- **Unique lines ratio near 0**: Every line is nearly identical → low `unique_lines_ratio`
- **Low alphabet-to-meaningful-content ratio**: Only single letters — `words_per_line` ≈ 1
- **High perplexity**: Highly repetitive text doesn't match natural language

**Document C: Code (context-dependent filter)**
- **Low alphabet ratio**: High density of punctuation (`;`, `=`, `(`, `)`, `:`)
- **Low words-per-line**: Code lines are dense and short
- **Whether to keep or discard** depends on the dataset's purpose: for a general language model this might be filtered, but for a code-focused dataset (The Pile's GitHub subset) it would be kept and assigned to the "code" domain with appropriate upsampling.

</details>

---

### Exercise 2: MinHash Deduplication

Explain the MinHash LSH (Locality-Sensitive Hashing) near-duplicate detection algorithm at a high level. Then answer:

1. What is the difference between exact deduplication and near-duplicate deduplication?
2. Why is exact hash matching insufficient for web-crawled data?
3. What is the Jaccard similarity, and how does MinHash estimate it efficiently?

<details>
<summary>Show Answer</summary>

**1. Exact vs Near-duplicate deduplication:**

- **Exact deduplication**: Remove documents with identical content (same MD5/SHA hash). Fast and lossless but misses variants.
- **Near-duplicate deduplication**: Remove documents that are highly similar but not identical (e.g., the same article reprinted on 50 different sites with minor changes). More comprehensive but computationally harder.

**2. Why exact hashing is insufficient:**

Web-crawled data contains many near-duplicates that are not exact copies:
- The same news article syndicated with minor edits (headline changes, added local bylines)
- Scraped content with slight URL or metadata differences
- Forum posts copied across sites with timestamps changed
These would all get different hash values but carry nearly identical information, bloating the dataset without adding knowledge diversity.

**3. Jaccard similarity and MinHash:**

**Jaccard similarity** between two documents A and B, when represented as sets of n-grams (shingles), is:
```
J(A, B) = |A ∩ B| / |A ∪ B|
```

Computing this directly for millions of document pairs is O(n²) — infeasible at scale.

**MinHash estimates Jaccard similarity** using the property that for a random hash function h:
```
P[min_h(A) = min_h(B)] = J(A, B)
```

By computing k independent min-hash values (a "signature"), the fraction of matching values approximates J(A, B) with variance that decreases as k increases. LSH then groups documents with similar signatures into buckets so only bucket members need pairwise comparison — reducing O(n²) to near-linear.

</details>

---

### Exercise 3: Data Mixing Strategy

LLaMA 2 uses the following approximate data mixture:
- English web (CommonCrawl): 67%
- Code (GitHub): 8%
- Wikipedia: 4%
- Books (Gutenberg/Books3): 4%
- ArXiv papers: 2%
- StackExchange: 2%
- Other: 13%

Answer the following:

1. Why is code data included despite the model being primarily a language model?
2. Why might Wikipedia receive 4% despite being a tiny fraction of the raw web data?
3. If you wanted to improve the model's mathematical reasoning, which data sources would you upsample and why?

<details>
<summary>Show Answer</summary>

**1. Why include code data:**
- Code improves **logical and structured reasoning**: programming requires precise step-by-step thinking, which transfers to math and formal reasoning.
- Code introduces **formal grammars and structure**: docstrings link natural language to formal specifications, helping the model understand instructions precisely.
- Empirically demonstrated: models trained without code perform worse on reasoning benchmarks even for non-coding tasks.

**2. Why Wikipedia is upsampled relative to its web fraction:**
- Wikipedia is **high-quality, factually verified, and encyclopedic**: it represents exactly the kind of clean, structured knowledge that benefits general-purpose language models.
- Raw web data is mostly low-quality ads, spam, and boilerplate. Wikipedia's actual fraction of the raw web is tiny, but its knowledge density is orders of magnitude higher.
- **Domain upsampling**: deliberately sampling high-quality domains at rates above their natural frequency is a standard curation technique.

**3. Improving mathematical reasoning — sources to upsample:**
- **ArXiv**: Contains proofs, derivations, and mathematical exposition. Currently only 2% despite containing extremely high-quality mathematical reasoning patterns.
- **StackExchange** (especially Math SE, Physics SE): Contains step-by-step problem-solving with explanations.
- **Code** (especially Python/Julia math libraries, Jupyter notebooks): Mathematical code forces precise reasoning.
- **Synthetic math data**: Generate math problems and solutions (e.g., via WebMath, MATH dataset, GSM8K-style augmentation) to directly target this capability.

</details>

---

### Exercise 4: Ethical Considerations in Data Curation

For each of the following data curation decisions, identify the ethical concern and propose a mitigation strategy.

1. Using all publicly available web text, including text from social media platforms.
2. Training on GitHub code repositories without filtering for license type.
3. Using perplexity-based filtering where the reference language model was trained primarily on English text.

<details>
<summary>Show Answer</summary>

**1. Social media text:**
- **Concern**: Personal information exposure (PII), privacy violation, amplification of harmful speech (hate speech, misinformation), and data from users who did not consent to AI training use.
- **Mitigation**: Apply PII removal (regex + NER-based detection for names, emails, phone numbers, locations), content safety classification to remove harmful content, respect robots.txt and platform Terms of Service, consider opt-out mechanisms for user-generated content.

**2. GitHub code without license filtering:**
- **Concern**: Copyright infringement. Code under GPL, AGPL, or non-commercial licenses may prohibit use in commercially deployed AI systems. The GitHub Copilot lawsuit raised exactly this issue.
- **Mitigation**: Filter to permissive licenses only (MIT, Apache 2.0, BSD), document license composition of the code subset, consider using a separate legal analysis for ambiguous cases.

**3. English-biased perplexity filtering:**
- **Concern**: Systematic disadvantage to non-English languages. If the reference model is English-trained, text in other languages will receive artificially high perplexity scores and be disproportionately filtered out, creating a linguistically biased dataset.
- **Mitigation**: Use per-language perplexity thresholds calibrated to reference models trained on each language, or use language-ID-first filtering followed by per-language quality filtering. The ROOTS/BLOOM approach used separate quality filters for each of 59 languages.

</details>
