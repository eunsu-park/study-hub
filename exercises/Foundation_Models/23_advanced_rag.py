"""
Exercises for Lesson 23: Advanced RAG
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: HyDE vs Direct Query ===
# Problem: Analyze when HyDE helps vs when it does not.

def exercise_1():
    """Solution: HyDE vs direct query analysis"""
    print("  Core insight: Question embeddings and document embeddings occupy")
    print("  different regions of embedding space. HyDE generates a hypothetical")
    print("  answer document first, then embeds THAT as the search query.")
    print("  Document-to-document similarity works better than question-to-document.")
    print()

    print("  Task A: 'What are the side effects of aspirin?' (factual)")
    print("    Direct query: moderate similarity (different linguistic style)")
    print("    HyDE hypothetical: 'Common side effects include GI irritation...'")
    print("    HyDE embedding -> high similarity with target documents (same style)")
    print("    HyDE HELPS here -- factual questions benefit from style matching.")
    print()

    print("  Task B: 'Find formal legal tone document about contract termination'")
    print("    Query already describes a document style, not a factual question.")
    print("    HyDE risk: may hallucinate specific legal clauses not in corpus.")
    print("    Hybrid search (dense + BM25) may work better without hallucination risk.")
    print()

    print("  When HyDE does NOT help (or hurts):")
    print("    1. Specific factual lookups (LLM may hallucinate wrong numbers)")
    print("    2. Very short queries in well-matched corpora")
    print("    3. Time-sensitive queries (hypothetical reflects outdated knowledge)")


# === Exercise 2: Multi-hop Retrieval Design ===
# Problem: Design pipeline for a question requiring 2+ retrieval steps.

def exercise_2():
    """Solution: Multi-hop retrieval design"""
    print("  Question: 'Was the judge in Smith v. Johnson 2019 also involved")
    print("  in intellectual property disputes in 2020?'")
    print()

    print("  Why single retrieval fails:")
    print("    No single document contains all this information.")
    print("    Judge's name unknown upfront -> query too vague.")
    print()

    print("  Multi-hop pipeline:")
    print()
    print("    Step 1: Retrieve case details")
    print("      Query: 'Smith v. Johnson 2019 case'")
    print("      Retrieved: Case record with judge name")
    print("      Extracted: judge_name = 'Judge Robert Thompson'")
    print()
    print("    Step 2: Use extracted info for second retrieval")
    print("      Query: 'Judge Robert Thompson intellectual property 2020'")
    print("      Retrieved: IP cases from 2020 listing Judge Thompson")
    print("      Answer: 'Yes, Judge Thompson presided over DataTech v. InnoSoft'")
    print()

    # Implementation skeleton
    print("  Implementation pattern:")
    print("    class MultiHopRetriever:")
    print("      def retrieve(self, question, retriever, llm):")
    print("        # Step 1: Initial retrieval")
    print("        docs_1 = retriever.get_relevant_documents(question)")
    print("        # Extract key entities from step 1")
    print("        entity = llm.invoke(f'Extract entity from {docs_1}')")
    print("        # Step 2: Targeted retrieval")
    print("        docs_2 = retriever.get_relevant_documents(f'{entity} IP 2020')")
    print("        return docs_1 + docs_2")
    print()
    print("  Key principle: extract SPECIFIC, GROUNDED entities (name, ID, date)")
    print("  rather than vague summaries for precise step 2 queries.")


# === Exercise 3: Reranking vs Retrieval Quality ===
# Problem: Explain why two-stage (retrieve then rerank) outperforms either alone.

def exercise_3():
    """Solution: Two-stage retrieval analysis"""
    print("  Why retrieval alone (top-3 directly) can fail:")
    print("    Bi-encoders encode query and document INDEPENDENTLY.")
    print("    Efficient but coarse: compresses entire documents into fixed vectors.")
    print("    Optimized for recall (finding related docs) not precision (best doc).")
    print("    May miss best answer that scored 4th or 5th due to compression.")
    print()

    print("  Why reranking alone (no initial retrieval) is impossible:")
    print("    Cross-encoders jointly process query+document (token-level attention).")
    print("    Highly accurate but O(n) forward passes over entire corpus.")
    print("    Running on 1M documents per query would take hours.")
    print()

    print("  Why two-stage works best:")
    print("    Stage 1: Bi-encoder (fast recall)")
    print("      Retrieve top-100 from 1M documents in ~20ms")
    print("      High recall: correct answer almost certainly in top-100")
    print()
    print("    Stage 2: Cross-encoder (accurate reranking)")
    print("      Rerank top-100 -> select top-3 in ~200ms")
    print("      High precision: token-level attention identifies true relevance")
    print()
    print("    Result: Fast (bi-encoder speed) + Accurate (cross-encoder quality)")
    print("    Total latency: ~220ms vs hours for cross-encoder alone")
    print()
    print("  Two-stage improves MRR@3 by 10-20% over bi-encoder alone.")


# === Exercise 4: Self-RAG Reflection Token Design ===
# Problem: Design reflection tokens for a medical chatbot.

def exercise_4():
    """Solution: Self-RAG reflection token design"""
    tokens = [
        {
            "name": "[Retrieve]",
            "question": "Should the system retrieve external information?",
            "yes_example": "What is the maximum safe dose of acetaminophen?",
            "no_example": "How should I tell my family about my diagnosis?",
        },
        {
            "name": "[IsREL]",
            "question": "Is the retrieved document relevant to the query?",
            "yes_example": "Retrieved aspirin side effects doc for aspirin query -> RELEVANT",
            "no_example": "Retrieved aspirin doc for acetaminophen query -> NOT_RELEVANT -> re-retrieve",
        },
        {
            "name": "[IsSUP]",
            "question": "Does the response faithfully reflect retrieved documents?",
            "yes_example": "Retrieved: 'max dose 4g/day for healthy adults', Generated matches -> SUPPORTED",
            "no_example": "Retrieved: '4g/day', Generated: '3g/day' -> PARTIALLY_SUPPORTED -> revise",
        },
        {
            "name": "[IsUSE]",
            "question": "Is the overall response useful to the user?",
            "yes_example": "Relevant + supported + addresses question + safety caveat -> USEFUL",
            "no_example": "Accurate but missing 'consult pharmacist' caveat -> PARTIALLY",
        },
    ]

    for t in tokens:
        print(f"  Token: {t['name']} -- {t['question']}")
        print(f"    YES: {t['yes_example']}")
        print(f"    NO:  {t['no_example']}")
        print()

    print("  Example flow for: 'Can I take ibuprofen with blood pressure medication?'")
    print("    [Retrieve] = YES (drug interaction facts needed)")
    print("    [IsREL] = YES (retrieved NSAIDs + antihypertensives entry)")
    print("    [IsSUP] = YES (response matches 'NSAIDs reduce antihypertensive effectiveness')")
    print("    [IsUSE] = PARTIALLY (should add 'consult your pharmacist' caveat)")
    print()
    print("  This four-token chain ensures the medical chatbot doesn't hallucinate")
    print("  drug facts, doesn't include irrelevant docs, and remains safe.")


if __name__ == "__main__":
    print("=== Exercise 1: HyDE vs Direct Query ===")
    exercise_1()
    print("\n=== Exercise 2: Multi-hop Retrieval Design ===")
    exercise_2()
    print("\n=== Exercise 3: Reranking vs Retrieval Quality ===")
    exercise_3()
    print("\n=== Exercise 4: Self-RAG Reflection Tokens ===")
    exercise_4()
    print("\nAll exercises completed!")
