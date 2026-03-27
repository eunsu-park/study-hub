"""
09. RAG (Retrieval-Augmented Generation) Example

Combining document retrieval with LLM generation
"""

import numpy as np

print("=" * 60)
print("RAG System")
print("=" * 60)


# ============================================
# 1. Simple RAG Implementation (NumPy only)
# ============================================
print("\n[1] Simple RAG (NumPy)")
print("-" * 40)

class SimpleVectorStore:
    """Simple vector store"""
    def __init__(self):
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents, embeddings):
        self.documents = documents
        self.embeddings = np.array(embeddings)

    def search(self, query_embedding, top_k=3):
        """Search by cosine similarity"""
        query = np.array(query_embedding)

        # Cosine similarity
        similarities = np.dot(self.embeddings, query) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query)
        )

        # Top k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]


# Example documents
documents = [
    "Python is a high-level programming language known for its readability.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing deals with text and speech.",
    "Computer vision enables machines to interpret images."
]

# Simulated embeddings (in practice, use a model)
np.random.seed(42)
embeddings = np.random.randn(len(documents), 128)

# Vector store
store = SimpleVectorStore()
store.add_documents(documents, embeddings)

# Search
query_embedding = np.random.randn(128)
results = store.search(query_embedding, top_k=2)

print("Search results:")
for doc, score in results:
    print(f"  [{score:.4f}] {doc[:50]}...")


# ============================================
# 2. Sentence Transformers + RAG
# ============================================
print("\n[2] Sentence Transformers RAG")
print("-" * 40)

try:
    from sentence_transformers import SentenceTransformer

    # Embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Document embeddings
    doc_embeddings = model.encode(documents)
    print(f"Document embeddings shape: {doc_embeddings.shape}")

    # Query
    query = "What is machine learning?"
    query_embedding = model.encode(query)

    # Search
    store = SimpleVectorStore()
    store.add_documents(documents, doc_embeddings)
    results = store.search(query_embedding, top_k=2)

    print(f"\nQuery: {query}")
    print("Search results:")
    for doc, score in results:
        print(f"  [{score:.4f}] {doc}")

except ImportError:
    print("sentence-transformers not installed")


# ============================================
# 3. Chunking
# ============================================
print("\n[3] Text Chunking")
print("-" * 40)

def chunk_text(text, chunk_size=100, overlap=20):
    """Chunking with overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

long_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines,
as opposed to natural intelligence displayed by animals including humans.
AI research has been defined as the field of study of intelligent agents,
which refers to any system that perceives its environment and takes actions
that maximize its chance of achieving its goals. The term "artificial intelligence"
had previously been used to describe machines that mimic and display "human"
cognitive skills that are associated with the human mind, such as "learning" and
"problem-solving". This definition has since been rejected by major AI researchers
who now describe AI in terms of rationality and acting rationally.
"""

chunks = chunk_text(long_text, chunk_size=150, overlap=30)
print(f"Original length: {len(long_text)} chars")
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"  Chunk {i+1}: {chunk[:50]}...")


# ============================================
# 4. Complete RAG Pipeline
# ============================================
print("\n[4] Complete RAG Pipeline")
print("-" * 40)

class RAGPipeline:
    """RAG Pipeline"""

    def __init__(self, embedding_model=None):
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.embedding_model = embedding_model

    def add_documents(self, documents, chunk_size=200, overlap=50):
        """Add documents and chunk them"""
        self.documents = documents

        # Chunking
        for doc in documents:
            doc_chunks = chunk_text(doc, chunk_size, overlap)
            self.chunks.extend(doc_chunks)

        # Embedding
        if self.embedding_model:
            self.embeddings = self.embedding_model.encode(self.chunks)
        else:
            # Simulated embeddings
            self.embeddings = np.random.randn(len(self.chunks), 128)

        print(f"{len(documents)} documents -> {len(self.chunks)} chunks")

    def retrieve(self, query, top_k=3):
        """Retrieve relevant chunks"""
        if self.embedding_model:
            query_emb = self.embedding_model.encode(query)
        else:
            query_emb = np.random.randn(128)

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]

    def generate(self, query, context):
        """Build prompt (in practice, call LLM)"""
        prompt = f"""Answer based on the context:

Context:
{context}

Question: {query}

Answer:"""
        return prompt

    def query(self, question, top_k=3):
        """RAG query"""
        # Retrieve
        relevant_chunks = self.retrieve(question, top_k)
        context = "\n\n".join(relevant_chunks)

        # Generate prompt
        prompt = self.generate(question, context)

        return {
            "question": question,
            "context": context,
            "prompt": prompt
        }


# RAG pipeline test
rag = RAGPipeline()
rag.add_documents([long_text])

result = rag.query("What is artificial intelligence?", top_k=2)
print(f"\nQuestion: {result['question']}")
print(f"Context length: {len(result['context'])} chars")
print(f"Prompt preview:\n{result['prompt'][:200]}...")


# ============================================
# 5. OpenAI RAG (API required)
# ============================================
print("\n[5] OpenAI RAG Example (code only)")
print("-" * 40)

openai_rag_code = '''
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class OpenAIRAG:
    def __init__(self):
        self.client = OpenAI()
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents):
        self.documents = documents
        self.embeddings = self.embed_model.encode(documents)

    def search(self, query, top_k=3):
        query_emb = self.embed_model.encode(query)
        similarities = cosine_similarity([query_emb], self.embeddings)[0]
        top_idx = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_idx]

    def query(self, question, top_k=3):
        # Retrieve
        relevant = self.search(question, top_k)
        context = "\\n\\n".join(relevant)

        # LLM call
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer based on the context."},
                {"role": "user", "content": f"Context:\\n{context}\\n\\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
'''
print(openai_rag_code)


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("RAG Summary")
print("=" * 60)

summary = """
RAG Pipeline:
    1. Documents -> Chunking -> Embedding -> Store in Vector DB
    2. Query -> Embedding -> Search similar documents
    3. Query + Documents -> LLM -> Answer

Key Code:
    # Embedding
    embeddings = model.encode(documents)

    # Search
    similarities = cosine_similarity([query_emb], embeddings)
    top_docs = documents[top_indices]

    # Generate
    prompt = f"Context: {context}\\nQuestion: {query}"
    response = llm.generate(prompt)
"""
print(summary)
