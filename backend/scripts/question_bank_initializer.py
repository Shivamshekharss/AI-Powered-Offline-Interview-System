"""
Question Bank Initializer - InterviewAce AI

PURPOSE:
    Populate the vector store with high-quality, curated interview questions
    
WHAT IT DOES:
    - Creates comprehensive question sets
    - Organizes by category (Python, LLMs, RAG, etc.)
    - Sets appropriate difficulty levels
    - Adds expected answers and follow-ups
    - Loads everything into vector store
    
WHY WE NEED IT:
    - Demonstrates GenAI knowledge (LLMs, RAG, LangChain)
    - Provides quality question base for RAG retrieval
    - Shows your understanding of the field
    - Makes the system immediately useful
    
CATEGORIES COVERED:
    - Python Programming
    - LLMs & Generative AI
    - RAG (Retrieval-Augmented Generation)
    - LangChain & LangGraph
    - Vector Databases
    - Prompt Engineering
    - AI Agents
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.rag.vector_store import VectorStore, Question
from typing import List


def create_python_questions() -> List[Question]:
    """
    Create Python programming questions
    
    Why these questions:
        - Fundamental for any developer role
        - Shows programming depth
        - Common interview topics
    """
    questions = [
        Question(
            id="py_001",
            question="What are Python decorators and how do they work?",
            category="Python",
            difficulty="Medium",
            topics=["decorators", "functions", "metaprogramming"],
            expected_answer="A decorator is a function that takes another function and extends its behavior without explicitly modifying it. Uses @ syntax and wraps functions.",
            follow_ups=[
                "Can you implement a timing decorator?",
                "What's the difference between @staticmethod and @classmethod?",
                "How do you pass arguments to decorators?"
            ]
        ),
        Question(
            id="py_002",
            question="Explain Python generators and their benefits",
            category="Python",
            difficulty="Medium",
            topics=["generators", "iterators", "memory-efficiency"],
            expected_answer="Generators are functions that yield values one at a time using the yield keyword. They're memory-efficient for large datasets and support lazy evaluation.",
            follow_ups=[
                "What's the difference between yield and return?",
                "How do you send values back into a generator?",
                "When would you use a generator over a list comprehension?"
            ]
        ),
        Question(
            id="py_003",
            question="What is the difference between list and tuple in Python?",
            category="Python",
            difficulty="Easy",
            topics=["data-structures", "mutability"],
            expected_answer="Lists are mutable (can be modified), tuples are immutable (cannot be modified). Tuples are faster and can be used as dictionary keys.",
            follow_ups=[
                "When would you use a tuple over a list?",
                "Can you have a list inside a tuple?",
                "What's the performance difference?"
            ]
        ),
        Question(
            id="py_004",
            question="Explain the Global Interpreter Lock (GIL) in Python",
            category="Python",
            difficulty="Hard",
            topics=["concurrency", "threading", "performance"],
            expected_answer="The GIL is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously. Impacts CPU-bound multi-threaded programs.",
            follow_ups=[
                "How does GIL affect multi-threaded programs?",
                "What are alternatives to bypass GIL?",
                "Does GIL affect multiprocessing?"
            ]
        ),
        Question(
            id="py_005",
            question="What are context managers and how do you implement one?",
            category="Python",
            difficulty="Medium",
            topics=["context-managers", "with-statement", "resource-management"],
            expected_answer="Context managers handle resource allocation and cleanup using __enter__ and __exit__ methods. Used with 'with' statement for safe resource handling.",
            follow_ups=[
                "Implement a custom context manager",
                "What's the @contextmanager decorator?",
                "How do you handle exceptions in __exit__?"
            ]
        ),
        Question(
            id="py_006",
            question="Explain list comprehensions vs generator expressions",
            category="Python",
            difficulty="Easy",
            topics=["comprehensions", "generators", "syntax"],
            expected_answer="List comprehensions create lists immediately [x for x in range(10)], generator expressions create generators (x for x in range(10)). Generators are more memory-efficient.",
            follow_ups=[
                "When would you choose one over the other?",
                "Can you nest comprehensions?",
                "What about dictionary comprehensions?"
            ]
        ),
    ]
    return questions


def create_llm_questions() -> List[Question]:
    """
    Create LLM & Generative AI questions
    
    Why these questions:
        - Core to GenAI engineer roles
        - Shows understanding of LLM fundamentals
        - Relevant to your target jobs
    """
    questions = [
        Question(
            id="llm_001",
            question="What is the difference between GPT and BERT?",
            category="LLMs",
            difficulty="Medium",
            topics=["transformers", "architecture", "pre-training"],
            expected_answer="GPT is autoregressive (predicts next token), decoder-only architecture. BERT is bidirectional, encoder-only, uses masked language modeling. GPT better for generation, BERT better for understanding.",
            follow_ups=[
                "Which would you use for text classification?",
                "Explain the attention mechanism",
                "What about encoder-decoder models like T5?"
            ]
        ),
        Question(
            id="llm_002",
            question="What are embeddings in the context of LLMs?",
            category="LLMs",
            difficulty="Easy",
            topics=["embeddings", "vector-representations", "semantic-search"],
            expected_answer="Embeddings are dense vector representations of text that capture semantic meaning. Similar concepts have similar vectors, enabling semantic search and comparison.",
            follow_ups=[
                "How are embeddings created?",
                "What's the typical dimension of embeddings?",
                "Difference between word2vec and transformer embeddings?"
            ]
        ),
        Question(
            id="llm_003",
            question="Explain the concept of temperature in LLM generation",
            category="LLMs",
            difficulty="Medium",
            topics=["sampling", "generation", "hyperparameters"],
            expected_answer="Temperature controls randomness in text generation. Lower temperature (0.1-0.5) = more deterministic/focused. Higher temperature (0.8-1.5) = more creative/random. Temperature 0 = greedy decoding.",
            follow_ups=[
                "When would you use high vs low temperature?",
                "What other sampling parameters exist?",
                "Explain top-k and nucleus sampling"
            ]
        ),
        Question(
            id="llm_004",
            question="What is fine-tuning and when would you use it?",
            category="LLMs",
            difficulty="Medium",
            topics=["fine-tuning", "transfer-learning", "training"],
            expected_answer="Fine-tuning adapts a pre-trained model to specific tasks by training on domain-specific data. Use when you need specialized behavior, have quality training data, and base model isn't sufficient.",
            follow_ups=[
                "What's the difference between fine-tuning and prompt engineering?",
                "Explain LoRA and PEFT",
                "How much data do you need for fine-tuning?"
            ]
        ),
        Question(
            id="llm_005",
            question="What is the attention mechanism in transformers?",
            category="LLMs",
            difficulty="Hard",
            topics=["attention", "transformers", "architecture"],
            expected_answer="Attention allows models to weigh importance of different parts of input. Self-attention computes relationships between all positions. Multi-head attention learns different representation subspaces.",
            follow_ups=[
                "Explain the Query, Key, Value matrices",
                "What's the computational complexity of attention?",
                "How does attention differ from RNNs?"
            ]
        ),
        Question(
            id="llm_006",
            question="What are the main challenges in deploying LLMs to production?",
            category="LLMs",
            difficulty="Hard",
            topics=["deployment", "production", "mlops"],
            expected_answer="Key challenges: latency (slow inference), cost (compute-intensive), scaling (handling load), reliability (consistency), safety (harmful outputs), and monitoring (tracking performance).",
            follow_ups=[
                "How do you optimize LLM inference?",
                "What's quantization and when to use it?",
                "How do you monitor LLM outputs?"
            ]
        ),
    ]
    return questions


def create_rag_questions() -> List[Question]:
    """
    Create RAG (Retrieval-Augmented Generation) questions
    
    Why these questions:
        - RAG is core to modern GenAI applications
        - Shows you understand practical AI implementation
        - Directly relevant to your project
    """
    questions = [
        Question(
            id="rag_001",
            question="What is RAG and why is it useful?",
            category="RAG",
            difficulty="Easy",
            topics=["rag", "retrieval", "grounding"],
            expected_answer="RAG combines retrieval of relevant documents with LLM generation. Useful for: grounding responses in facts, reducing hallucinations, accessing external knowledge, and updating information without retraining.",
            follow_ups=[
                "How does RAG reduce hallucinations?",
                "What are the main components of a RAG system?",
                "RAG vs fine-tuning - when to use each?"
            ]
        ),
        Question(
            id="rag_002",
            question="Explain the typical RAG pipeline architecture",
            category="RAG",
            difficulty="Medium",
            topics=["architecture", "pipeline", "components"],
            expected_answer="RAG pipeline: 1) Document ingestion & chunking, 2) Embedding generation, 3) Vector storage, 4) Query embedding, 5) Similarity search, 6) Context retrieval, 7) Prompt augmentation, 8) LLM generation.",
            follow_ups=[
                "Why do we chunk documents?",
                "What's the role of embeddings?",
                "How do you choose chunk size?"
            ]
        ),
        Question(
            id="rag_003",
            question="What are vector databases and why use them in RAG?",
            category="RAG",
            difficulty="Medium",
            topics=["vector-databases", "similarity-search", "storage"],
            expected_answer="Vector databases store and efficiently search high-dimensional embeddings. Enable fast similarity search, scale to millions of vectors, support filtering, and provide low-latency retrieval for RAG.",
            follow_ups=[
                "Compare FAISS, Pinecone, and ChromaDB",
                "What's cosine similarity vs L2 distance?",
                "How do you handle vector database updates?"
            ]
        ),
        Question(
            id="rag_004",
            question="How do you evaluate RAG system performance?",
            category="RAG",
            difficulty="Hard",
            topics=["evaluation", "metrics", "quality"],
            expected_answer="Evaluate retrieval (precision, recall, MRR) and generation (relevance, faithfulness, answer correctness). Use metrics like RAGAS, human evaluation, and A/B testing.",
            follow_ups=[
                "What's the difference between faithfulness and relevance?",
                "How do you test retrieval quality?",
                "What metrics matter most for production?"
            ]
        ),
        Question(
            id="rag_005",
            question="What strategies improve RAG retrieval quality?",
            category="RAG",
            difficulty="Hard",
            topics=["optimization", "retrieval", "techniques"],
            expected_answer="Strategies: hybrid search (keyword + semantic), query expansion, reranking, metadata filtering, multi-query retrieval, hierarchical retrieval, and contextual embeddings.",
            follow_ups=[
                "Explain hybrid search benefits",
                "What's a reranker and when to use it?",
                "How does query expansion work?"
            ]
        ),
        Question(
            id="rag_006",
            question="What are the main challenges in building production RAG systems?",
            category="RAG",
            difficulty="Hard",
            topics=["production", "challenges", "engineering"],
            expected_answer="Challenges: latency (multiple steps), cost (embeddings + LLM), data freshness, chunking strategy, retrieval precision, context window limits, and handling multi-document queries.",
            follow_ups=[
                "How do you optimize RAG latency?",
                "What's the cost breakdown of RAG?",
                "How do you keep vector store updated?"
            ]
        ),
    ]
    return questions


def create_langchain_questions() -> List[Question]:
    """
    Create LangChain & LangGraph questions
    
    Why these questions:
        - LangChain is a key framework for your role
        - Shows practical implementation knowledge
        - Relevant to your actual project
    """
    questions = [
        Question(
            id="lc_001",
            question="What is LangChain and what problems does it solve?",
            category="LangChain",
            difficulty="Easy",
            topics=["framework", "llm-orchestration"],
            expected_answer="LangChain is a framework for building LLM applications. Solves: chaining LLM calls, managing prompts, handling memory/state, integrating tools, building agents, and orchestrating complex workflows.",
            follow_ups=[
                "What are the main LangChain components?",
                "LangChain vs building from scratch?",
                "What's the difference between chains and agents?"
            ]
        ),
        Question(
            id="lc_002",
            question="Explain LangChain chains and when to use them",
            category="LangChain",
            difficulty="Medium",
            topics=["chains", "composition", "workflow"],
            expected_answer="Chains compose multiple components in sequence. Types: LLMChain (single LLM call), SequentialChain (multiple steps), RouterChain (conditional routing). Use for deterministic multi-step workflows.",
            follow_ups=[
                "What's the difference between SimpleSequentialChain and SequentialChain?",
                "How do you handle errors in chains?",
                "Can chains be nested?"
            ]
        ),
        Question(
            id="lc_003",
            question="What are LangChain agents and how do they differ from chains?",
            category="LangChain",
            difficulty="Medium",
            topics=["agents", "tools", "reasoning"],
            expected_answer="Agents use LLMs to decide which tools to call and in what order. Unlike chains (fixed sequence), agents reason about actions dynamically based on input and previous observations.",
            follow_ups=[
                "What agent types exist in LangChain?",
                "How do you create custom tools?",
                "What's ReAct prompting?"
            ]
        ),
        Question(
            id="lc_004",
            question="How does memory work in LangChain?",
            category="LangChain",
            difficulty="Medium",
            topics=["memory", "conversation", "state"],
            expected_answer="LangChain memory stores conversation history. Types: ConversationBufferMemory (stores all), ConversationSummaryMemory (summarizes), ConversationBufferWindowMemory (keeps last k). Enables context-aware conversations.",
            follow_ups=[
                "When would you use summary vs buffer memory?",
                "How do you persist memory across sessions?",
                "What about vector store memory?"
            ]
        ),
        Question(
            id="lc_005",
            question="What is LangGraph and when would you use it?",
            category="LangChain",
            difficulty="Hard",
            topics=["langgraph", "state-machines", "workflows"],
            expected_answer="LangGraph builds stateful, multi-agent workflows as graphs. Nodes are functions, edges define flow. Use for complex agentic systems with loops, conditional branching, human-in-the-loop, and parallel execution.",
            follow_ups=[
                "How is LangGraph different from chains?",
                "Explain state management in LangGraph",
                "What's a StateGraph vs MessageGraph?"
            ]
        ),
        Question(
            id="lc_006",
            question="How do you implement custom tools in LangChain?",
            category="LangChain",
            difficulty="Medium",
            topics=["tools", "agents", "integration"],
            expected_answer="Create tools using @tool decorator or Tool.from_function(). Requires: function name, description (for agent reasoning), and implementation. Agents use descriptions to decide when to call tools.",
            follow_ups=[
                "How do you handle tool errors?",
                "What makes a good tool description?",
                "Can tools call other tools?"
            ]
        ),
    ]
    return questions


def create_prompt_engineering_questions() -> List[Question]:
    """
    Create Prompt Engineering questions
    
    Why these questions:
        - Critical skill for working with LLMs
        - Shows understanding of practical AI usage
        - Low-cost way to improve performance
    """
    questions = [
        Question(
            id="pe_001",
            question="What are the key principles of effective prompt engineering?",
            category="Prompt Engineering",
            difficulty="Easy",
            topics=["prompting", "best-practices"],
            expected_answer="Key principles: be clear and specific, provide context, use examples (few-shot), specify format, break complex tasks into steps, and iterate based on results.",
            follow_ups=[
                "What's the difference between zero-shot and few-shot?",
                "How many examples for few-shot prompting?",
                "What's chain-of-thought prompting?"
            ]
        ),
        Question(
            id="pe_002",
            question="Explain chain-of-thought (CoT) prompting",
            category="Prompt Engineering",
            difficulty="Medium",
            topics=["cot", "reasoning", "techniques"],
            expected_answer="CoT prompting encourages LLMs to show step-by-step reasoning before answering. Improves performance on complex tasks by breaking down problems. Can be zero-shot ('Let's think step by step') or few-shot.",
            follow_ups=[
                "When is CoT most effective?",
                "What's self-consistency in CoT?",
                "Tree-of-thought vs chain-of-thought?"
            ]
        ),
        Question(
            id="pe_003",
            question="What is prompt injection and how do you prevent it?",
            category="Prompt Engineering",
            difficulty="Hard",
            topics=["security", "safety", "vulnerabilities"],
            expected_answer="Prompt injection is when users manipulate prompts to override instructions. Prevention: input validation, delimiter separation, privilege separation, output filtering, and using system messages properly.",
            follow_ups=[
                "Give examples of prompt injection attacks",
                "What's indirect prompt injection?",
                "How do you test for vulnerabilities?"
            ]
        ),
        Question(
            id="pe_004",
            question="How do you structure prompts for consistent JSON output?",
            category="Prompt Engineering",
            difficulty="Medium",
            topics=["structured-output", "json", "formatting"],
            expected_answer="Use clear format instructions, provide JSON schema/example, explicitly request JSON only, use system message for format rules, and consider function calling or JSON mode APIs for guaranteed structure.",
            follow_ups=[
                "What's OpenAI's JSON mode?",
                "How do you handle parsing errors?",
                "Function calling vs prompt engineering?"
            ]
        ),
    ]
    return questions


def create_agentic_ai_questions() -> List[Question]:
    """
    Create Agentic AI & Multi-Agent System questions
    
    Why these questions:
        - Shows understanding of advanced AI systems
        - Relevant to your agentic interview project
        - Demonstrates system design thinking
    """
    questions = [
        Question(
            id="agent_001",
            question="What defines an AI agent and how is it different from a simple LLM call?",
            category="Agentic AI",
            difficulty="Medium",
            topics=["agents", "autonomy", "tools"],
            expected_answer="AI agents use LLMs to reason about actions, access tools, and make decisions autonomously. Unlike simple calls, agents: plan multi-step tasks, use tools dynamically, handle feedback loops, and adapt based on observations.",
            follow_ups=[
                "What's the ReAct framework?",
                "How do agents decide which tool to use?",
                "What are the limitations of current agents?"
            ]
        ),
        Question(
            id="agent_002",
            question="Explain multi-agent systems and when to use them",
            category="Agentic AI",
            difficulty="Hard",
            topics=["multi-agent", "orchestration", "collaboration"],
            expected_answer="Multi-agent systems have specialized agents collaborating. Benefits: separation of concerns, parallel processing, expert specialization. Use when: tasks need different expertise, parallel work possible, or complex workflows required.",
            follow_ups=[
                "How do agents communicate?",
                "What's agent handoff?",
                "Centralized vs decentralized orchestration?"
            ]
        ),
        Question(
            id="agent_003",
            question="What are the key challenges in building production agent systems?",
            category="Agentic AI",
            difficulty="Hard",
            topics=["production", "reliability", "cost"],
            expected_answer="Challenges: reliability (agents can fail), cost (multiple LLM calls), latency (tool usage overhead), debugging (complex reasoning), safety (unintended actions), and determinism (hard to test).",
            follow_ups=[
                "How do you test agent behavior?",
                "What's agent observability?",
                "How do you control agent costs?"
            ]
        ),
    ]
    return questions


def create_vector_db_questions() -> List[Question]:
    """
    Create Vector Database questions
    
    Why these questions:
        - Core component of RAG systems
        - Shows understanding of data infrastructure
        - Practical skill for GenAI engineers
    """
    questions = [
        Question(
            id="vdb_001",
            question="What are the main differences between FAISS, Pinecone, and ChromaDB?",
            category="Vector Databases",
            difficulty="Medium",
            topics=["vector-db", "comparison", "tools"],
            expected_answer="FAISS: Facebook's open-source, in-memory, fast but needs management. Pinecone: managed cloud service, scalable, costs $. ChromaDB: open-source, persistent, simple API, good for prototypes. Choice depends on scale, budget, infrastructure.",
            follow_ups=[
                "When would you choose each?",
                "What about Weaviate or Qdrant?",
                "How do you migrate between vector DBs?"
            ]
        ),
        Question(
            id="vdb_002",
            question="Explain approximate nearest neighbor (ANN) search and why it matters",
            category="Vector Databases",
            difficulty="Hard",
            topics=["ann", "algorithms", "performance"],
            expected_answer="ANN finds approximate nearest vectors instead of exact ones. Trade accuracy for speed. Algorithms: HNSW (graph-based), IVF (inverted file), LSH (locality-sensitive hashing). Crucial for scaling to millions of vectors.",
            follow_ups=[
                "What's the accuracy vs speed tradeoff?",
                "Explain HNSW algorithm",
                "How do you tune ANN parameters?"
            ]
        ),
        Question(
            id="vdb_003",
            question="How do you handle vector database updates in production?",
            category="Vector Databases",
            difficulty="Hard",
            topics=["updates", "production", "data-management"],
            expected_answer="Strategies: incremental updates (add new vectors), batch reindexing (rebuild periodically), versioning (maintain multiple indexes), soft deletes (mark inactive), and monitoring (track index size/performance).",
            follow_ups=[
                "How do you handle document deletions?",
                "What about concurrent updates?",
                "How do you version embeddings?"
            ]
        ),
    ]
    return questions


def initialize_question_bank():
    """
    Main function to populate the vector store with all questions
    
    Process:
        1. Create question sets for each category
        2. Combine all questions
        3. Load into vector store (creates embeddings)
        4. Display statistics
    
    This runs once to set up your question bank!
    """
    print("=" * 80)
    print("INITIALIZING QUESTION BANK")
    print("=" * 80)
    
    # Initialize vector store
    print("\n📦 Initializing Vector Store...")
    store = VectorStore()
    
    # Create all question categories
    print("\n📝 Creating Question Sets...")
    
    all_questions = []
    
    print("  ✓ Python questions...")
    all_questions.extend(create_python_questions())
    
    print("  ✓ LLM questions...")
    all_questions.extend(create_llm_questions())
    
    print("  ✓ RAG questions...")
    all_questions.extend(create_rag_questions())
    
    print("  ✓ LangChain questions...")
    all_questions.extend(create_langchain_questions())
    
    print("  ✓ Prompt Engineering questions...")
    all_questions.extend(create_prompt_engineering_questions())
    
    print("  ✓ Agentic AI questions...")
    all_questions.extend(create_agentic_ai_questions())
    
    print("  ✓ Vector Database questions...")
    all_questions.extend(create_vector_db_questions())
    
    print(f"\n📊 Total questions created: {len(all_questions)}")
    
    # Add to vector store (this creates embeddings!)
    print("\n🔄 Adding questions to vector store...")
    print("   (This will create embeddings - may take 30-60 seconds)")
    
    success_count = store.add_questions_batch(all_questions)
    
    print(f"\n✅ Successfully added {success_count} questions!")
    
    # Display statistics
    print("\n" + "=" * 80)
    print("QUESTION BANK STATISTICS")
    print("=" * 80)
    
    stats = store.get_statistics()
    
    print(f"\n📈 Total Questions: {stats['total_questions']}")
    
    print(f"\n📚 By Category:")
    for category, count in sorted(stats['categories'].items()):
        print(f"   {category:.<30} {count}")
    
    print(f"\n⭐ By Difficulty:")
    for difficulty, count in sorted(stats['difficulties'].items()):
        print(f"   {difficulty:.<30} {count}")
    
    print(f"\n🏷️  Top Topics:")
    sorted_topics = sorted(stats['topics'].items(), key=lambda x: x[1], reverse=True)
    for i, (topic, count) in enumerate(sorted_topics[:10], 1):
        print(f"   {i}. {topic:.<28} {count}")
    
    # Test search
    print("\n" + "=" * 80)
    print("TESTING SEARCH FUNCTIONALITY")
    print("=" * 80)
    
    test_queries = [
        "How do RAG systems work?",
        "Explain Python decorators",
        "What is LangChain?",
        "How to build AI agents?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = store.search(query, top_k=3)
        print(f"   Found {len(results)} relevant questions")
    
    print("\n" + "=" * 80)
    print("✅ QUESTION BANK INITIALIZATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n💾 Data saved to: {store.storage_path}")
    print(f"📊 Vector index: {store.index_path}")
    print(f"📝 Metadata: {store.metadata_path}")
    
    print("\n🎉 Your Interview Ace AI now has a comprehensive question bank!")
    print("   - Ready for RAG-powered question retrieval")
    print("   - Demonstrates your GenAI knowledge")
    print("   - Production-ready for interviews")


if __name__ == "__main__":
    """
    Run this script to populate your question bank
    
    Usage:
        python backend/scripts/question_bank_initializer.py
    
    This will:
        - Create 40+ curated interview questions
        - Generate embeddings for all questions
        - Store in vector database
        - Make your RAG system functional
    """
    initialize_question_bank()