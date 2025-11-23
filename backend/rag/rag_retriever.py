"""
RAG Question Retriever - InterviewAce AI

PURPOSE:
    Intelligently retrieve relevant interview questions from the vector store
    
WHAT IT DOES:
    - Semantic search through question bank
    - Filter by category, difficulty, topics
    - Hybrid retrieval (semantic + keyword)
    - Rank and score results
    - Provide context for retrieved questions
    
WHY WE NEED IT:
    - Fast question retrieval (milliseconds)
    - Leverages curated question bank
    - Reduces LLM generation costs
    - Ensures question quality
    - Enables intelligent question selection
    
RETRIEVAL STRATEGIES:
    1. Semantic Search - Find by meaning
    2. Filtered Search - Category/difficulty constraints
    3. Hybrid Search - Combine semantic + filters
    4. Diverse Retrieval - Ensure variety
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from backend.rag.vector_store import VectorStore, Question


@dataclass
class RetrievalResult:
    """
    Represents a retrieved question with metadata
    
    Why this class:
        - Organizes retrieval results cleanly
        - Includes relevance score for ranking
        - Provides context about why it was retrieved
        - Makes it easy to pass to other components
    
    Attributes:
        question: The Question object
        relevance_score: How relevant (lower = more relevant in L2 distance)
        retrieval_method: How it was found (semantic/filtered/hybrid)
        context: Why this question was selected
    """
    question: Question
    relevance_score: float
    retrieval_method: str
    context: str
    
    def __repr__(self) -> str:
        """Pretty print for debugging"""
        return (
            f"RetrievalResult(\n"
            f"  question='{self.question.question[:60]}...'\n"
            f"  score={self.relevance_score:.3f}\n"
            f"  method={self.retrieval_method}\n"
            f"  category={self.question.category}\n"
            f"  difficulty={self.question.difficulty}\n"
            f")"
        )


class RAGRetriever:
    """
    Intelligent question retrieval system
    
    ARCHITECTURE:
    
        Job Description / Topic
              ↓
        [Query Analysis] - Extract key requirements
              ↓
        [Vector Store Search] - Semantic similarity
              ↓
        [Filtering] - Apply constraints
              ↓
        [Ranking] - Score and sort
              ↓
        [Diversity] - Ensure variety
              ↓
        Top N Questions + Context
    
    USAGE:
        retriever = RAGRetriever()
        
        # Simple semantic search
        results = retriever.retrieve(
            query="Python backend engineer",
            top_k=5
        )
        
        # Filtered search
        results = retriever.retrieve_filtered(
            query="Python concepts",
            category="Python",
            difficulty="Medium",
            top_k=3
        )
        
        # Diverse retrieval
        results = retriever.retrieve_diverse(
            query="Full-stack developer",
            categories=["Python", "JavaScript"],
            top_k=10
        )
    """
    
    def __init__(self):
        """
        Initialize the RAG retriever
        
        What happens:
            1. Load vector store (already populated)
            2. Set default parameters
            3. Ready to retrieve!
        """
        print("Initializing RAG Retriever...")
        
        # Load vector store
        self.vector_store = VectorStore()
        
        # Default retrieval parameters
        self.default_top_k = 5
        self.min_relevance_threshold = 2.0  # L2 distance threshold
        
        print(f"  ✓ Loaded vector store with {len(self.vector_store.questions)} questions")
        print(f"  ✓ Ready to retrieve!")
    
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Simple semantic retrieval
        
        Use when:
            - You want questions similar to a topic
            - No specific category/difficulty constraints
            - Quick, straightforward retrieval
        
        Process:
            1. Search vector store by semantic similarity
            2. Filter by minimum score if specified
            3. Wrap in RetrievalResult objects
        
        Args:
            query: Search query (e.g., "Python backend development")
            top_k: Number of results to return
            min_score: Optional minimum relevance score (lower = better)
        
        Returns:
            List of RetrievalResult objects
        
        Example:
            results = retriever.retrieve("machine learning", top_k=3)
            for result in results:
                print(result.question.question)
        """
        print(f"\n🔍 Simple Retrieval: '{query}'")
        
        # Search vector store
        raw_results = self.vector_store.search(query, top_k=top_k)
        
        # Apply score threshold if specified
        if min_score:
            raw_results = [(q, score) for q, score in raw_results if score <= min_score]
        
        # Wrap in RetrievalResult
        results = []
        for question, score in raw_results:
            result = RetrievalResult(
                question=question,
                relevance_score=score,
                retrieval_method="semantic",
                context=f"Retrieved via semantic search for: '{query}'"
            )
            results.append(result)
        
        print(f"  ✓ Retrieved {len(results)} questions")
        
        return results
    
    
    def retrieve_filtered(
        self,
        query: str,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        topics: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Filtered semantic retrieval
        
        Use when:
            - You need specific category questions (Python only)
            - Want specific difficulty level (Medium)
            - Need questions on specific topics
        
        Process:
            1. Search vector store semantically
            2. Apply category filter
            3. Apply difficulty filter
            4. Apply topic filter
            5. Return top K that match all filters
        
        Args:
            query: Search query
            category: Filter by category (e.g., "Python", "LLMs")
            difficulty: Filter by difficulty (Easy/Medium/Hard)
            topics: Filter by topics (e.g., ["decorators", "functions"])
            top_k: Number of results
        
        Returns:
            List of RetrievalResult objects matching all filters
        
        Example:
            results = retriever.retrieve_filtered(
                query="advanced concepts",
                category="Python",
                difficulty="Hard",
                top_k=3
            )
        """
        print(f"\n🔍 Filtered Retrieval: '{query}'")
        print(f"   Filters: category={category}, difficulty={difficulty}, topics={topics}")
        
        # Search with filters
        raw_results = self.vector_store.search(
            query=query,
            top_k=top_k * 2,  # Get more, then filter
            category_filter=category,
            difficulty_filter=difficulty
        )
        
        # Additional topic filtering if specified
        if topics:
            filtered_results = []
            for question, score in raw_results:
                # Check if any specified topic is in question's topics
                if any(topic.lower() in [t.lower() for t in question.topics] for topic in topics):
                    filtered_results.append((question, score))
            raw_results = filtered_results
        
        # Take top K
        raw_results = raw_results[:top_k]
        
        # Wrap in RetrievalResult
        results = []
        filter_desc = []
        if category:
            filter_desc.append(f"category={category}")
        if difficulty:
            filter_desc.append(f"difficulty={difficulty}")
        if topics:
            filter_desc.append(f"topics={topics}")
        
        context_msg = f"Retrieved via filtered search ({', '.join(filter_desc)}) for: '{query}'"
        
        for question, score in raw_results:
            result = RetrievalResult(
                question=question,
                relevance_score=score,
                retrieval_method="filtered_semantic",
                context=context_msg
            )
            results.append(result)
        
        print(f"  ✓ Retrieved {len(results)} filtered questions")
        
        return results
    
    
    def retrieve_diverse(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        questions_per_category: int = 2,
        total_max: int = 10
    ) -> List[RetrievalResult]:
        """
        Diverse retrieval across multiple categories/difficulties
        
        Use when:
            - You want variety in questions
            - Testing multiple skill areas
            - Creating comprehensive interview sets
        
        Process:
            1. For each category, retrieve top questions
            2. Ensure balanced representation
            3. Mix difficulty levels
            4. Limit total to max
        
        Args:
            query: Search query
            categories: List of categories to include
            difficulties: List of difficulties to include
            questions_per_category: How many per category
            total_max: Maximum total questions
        
        Returns:
            List of RetrievalResult objects with diverse coverage
        
        Example:
            results = retriever.retrieve_diverse(
                query="full-stack engineer",
                categories=["Python", "JavaScript", "RAG"],
                questions_per_category=2,
                total_max=6
            )
        """
        print(f"\n🔍 Diverse Retrieval: '{query}'")
        print(f"   Categories: {categories}")
        print(f"   Questions per category: {questions_per_category}")
        
        all_results = []
        
        # If no categories specified, use all available
        if not categories:
            stats = self.vector_store.get_statistics()
            categories = list(stats['categories'].keys())
        
        # Retrieve from each category
        for category in categories:
            print(f"   Retrieving from: {category}")
            
            category_results = self.retrieve_filtered(
                query=query,
                category=category,
                top_k=questions_per_category
            )
            
            # Update context to indicate diverse retrieval
            for result in category_results:
                result.retrieval_method = "diverse"
                result.context = f"Retrieved for diverse coverage - category: {category}"
            
            all_results.extend(category_results)
        
        # If difficulties specified, filter
        if difficulties:
            all_results = [
                r for r in all_results 
                if r.question.difficulty in difficulties
            ]
        
        # Limit to total_max
        all_results = all_results[:total_max]
        
        print(f"  ✓ Retrieved {len(all_results)} diverse questions")
        
        return all_results
    
    
    def retrieve_by_job_description(
        self,
        job_description: str,
        top_k: int = 10
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Intelligent retrieval based on job description
        
        Use when:
            - Starting an interview
            - Have full job description
            - Want comprehensive question set
        
        Process:
            1. Analyze job description for key skills
            2. Retrieve questions for each skill area
            3. Balance across categories
            4. Prioritize by relevance
        
        Args:
            job_description: Full job description text
            top_k: Total questions to retrieve
        
        Returns:
            Dictionary with categories as keys, results as values
        
        Example:
            results = retriever.retrieve_by_job_description(
                job_description="Senior Python Engineer with ML experience...",
                top_k=8
            )
            # Returns: {"Python": [results], "LLMs": [results], ...}
        """
        print(f"\n🔍 Job Description Analysis")
        print(f"   Analyzing: '{job_description[:100]}...'")
        
        # Simple keyword-based category detection
        # In production, you'd use more sophisticated NLP
        job_lower = job_description.lower()
        
        category_keywords = {
            "Python": ["python", "django", "flask", "fastapi"],
            "JavaScript": ["javascript", "react", "node", "typescript"],
            "LLMs": ["llm", "language model", "gpt", "ai", "machine learning"],
            "RAG": ["rag", "retrieval", "vector", "embeddings"],
            "LangChain": ["langchain", "langgraph"],
            "Agentic AI": ["agent", "agents", "autonomous"],
            "Vector Databases": ["vector database", "faiss", "pinecone", "chromadb"]
        }
        
        # Detect relevant categories
        relevant_categories = []
        for category, keywords in category_keywords.items():
            if any(keyword in job_lower for keyword in keywords):
                relevant_categories.append(category)
        
        print(f"   Detected categories: {relevant_categories}")
        
        # If no categories detected, use most common ones
        if not relevant_categories:
            relevant_categories = ["Python", "LLMs", "RAG"]
            print(f"   Using default categories: {relevant_categories}")
        
        # Retrieve diverse questions
        questions_per_cat = max(2, top_k // len(relevant_categories))
        
        results = self.retrieve_diverse(
            query=job_description,
            categories=relevant_categories,
            questions_per_category=questions_per_cat,
            total_max=top_k
        )
        
        # Group by category
        categorized = {}
        for result in results:
            category = result.question.category
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(result)
        
        print(f"\n  ✓ Retrieved {len(results)} questions across {len(categorized)} categories")
        for cat, cat_results in categorized.items():
            print(f"     {cat}: {len(cat_results)} questions")
        
        return categorized
    
    
    def get_follow_up_questions(
        self,
        original_question: Question,
        user_answer: str,
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """
        Get relevant follow-up questions
        
        Use when:
            - User answered a question
            - Want to probe deeper
            - Need contextual follow-ups
        
        Process:
            1. Use original question's follow_ups if available
            2. Search for related questions by topic
            3. Filter by same category but harder difficulty
        
        Args:
            original_question: The question that was asked
            user_answer: User's answer (for future: analyze depth)
            top_k: Number of follow-ups
        
        Returns:
            List of relevant follow-up questions
        
        Example:
            follow_ups = retriever.get_follow_up_questions(
                original_question=q,
                user_answer="A decorator wraps a function...",
                top_k=2
            )
        """
        print(f"\n🔍 Finding Follow-ups for: '{original_question.question[:50]}...'")
        
        results = []
        
        # Strategy 1: Use predefined follow-ups
        if original_question.follow_ups:
            print(f"   Using {len(original_question.follow_ups)} predefined follow-ups")
            # Note: These are strings, not Question objects
            # In a full system, you'd search for these as queries
            for i, follow_up in enumerate(original_question.follow_ups[:top_k]):
                # Search for similar questions
                search_results = self.retrieve(follow_up, top_k=1)
                if search_results:
                    result = search_results[0]
                    result.context = f"Follow-up to: '{original_question.question[:40]}...'"
                    results.append(result)
        
        # Strategy 2: Search by same topics, harder difficulty
        if len(results) < top_k:
            print(f"   Searching by topics: {original_question.topics}")
            
            # Get harder difficulty
            difficulty_order = ["Easy", "Medium", "Hard"]
            current_idx = difficulty_order.index(original_question.difficulty)
            harder_difficulties = difficulty_order[current_idx + 1:] if current_idx < 2 else [original_question.difficulty]
            
            # Search by topics
            topic_query = " ".join(original_question.topics)
            for difficulty in harder_difficulties:
                topic_results = self.retrieve_filtered(
                    query=topic_query,
                    category=original_question.category,
                    difficulty=difficulty,
                    top_k=top_k - len(results)
                )
                
                for result in topic_results:
                    result.context = f"Deeper dive into: {', '.join(original_question.topics)}"
                    results.append(result)
                
                if len(results) >= top_k:
                    break
        
        results = results[:top_k]
        print(f"  ✓ Found {len(results)} follow-up questions")
        
        return results
    
    
    def get_statistics(self) -> Dict:
        """
        Get retriever statistics
        
        Why we need this:
            - Monitor question bank coverage
            - Dashboard metrics
            - System health check
        
        Returns:
            Dictionary with stats about available questions
        """
        return self.vector_store.get_statistics()


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    """
    Test the RAG retriever with various scenarios
    
    This demonstrates:
        1. Simple semantic retrieval
        2. Filtered retrieval
        3. Diverse retrieval
        4. Job description analysis
        5. Follow-up questions
    """
    
    print("=" * 80)
    print("TESTING: RAG Question Retriever")
    print("=" * 80)
    
    # Initialize retriever
    retriever = RAGRetriever()
    
    # Test 1: Simple retrieval
    print("\n" + "=" * 80)
    print("TEST 1: Simple Semantic Retrieval")
    print("=" * 80)
    
    results = retriever.retrieve("Python decorators and functions", top_k=3)
    
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.question.question}")
        print(f"   Category: {result.question.category}")
        print(f"   Difficulty: {result.question.difficulty}")
        print(f"   Score: {result.relevance_score:.3f}")
    
    # Test 2: Filtered retrieval
    print("\n" + "=" * 80)
    print("TEST 2: Filtered Retrieval (Python, Medium)")
    print("=" * 80)
    
    results = retriever.retrieve_filtered(
        query="advanced Python concepts",
        category="Python",
        difficulty="Medium",
        top_k=3
    )
    
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.question.question}")
        print(f"   Topics: {', '.join(result.question.topics)}")
    
    # Test 3: Diverse retrieval
    print("\n" + "=" * 80)
    print("TEST 3: Diverse Retrieval")
    print("=" * 80)
    
    results = retriever.retrieve_diverse(
        query="AI and machine learning",
        categories=["Python", "LLMs", "RAG"],
        questions_per_category=2,
        total_max=6
    )
    
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result.question.category}] {result.question.question[:60]}...")
    
    # Test 4: Job description analysis
    print("\n" + "=" * 80)
    print("TEST 4: Job Description Analysis")
    print("=" * 80)
    
    job_desc = """
    Senior GenAI Engineer
    
    We're looking for an experienced engineer to build LLM-powered applications.
    
    Requirements:
    - Expert in Python and LangChain
    - Experience with RAG systems and vector databases
    - Understanding of prompt engineering
    - Built production AI agents
    """
    
    categorized_results = retriever.retrieve_by_job_description(job_desc, top_k=8)
    
    print("\nResults by Category:")
    for category, cat_results in categorized_results.items():
        print(f"\n{category}:")
        for result in cat_results:
            print(f"  • {result.question.question[:60]}...")
    
    # Test 5: Follow-up questions
    print("\n" + "=" * 80)
    print("TEST 5: Follow-up Questions")
    print("=" * 80)
    
    # Get a question first
    initial = retriever.retrieve("Python decorators", top_k=1)[0]
    print(f"\nOriginal Question: {initial.question.question}")
    
    follow_ups = retriever.get_follow_up_questions(
        original_question=initial.question,
        user_answer="Decorators wrap functions to extend behavior",
        top_k=3
    )
    
    print("\nFollow-up Questions:")
    for i, result in enumerate(follow_ups, 1):
        print(f"\n{i}. {result.question.question}")
        print(f"   Difficulty: {result.question.difficulty}")
        print(f"   Context: {result.context}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("Retriever Statistics")
    print("=" * 80)
    
    stats = retriever.get_statistics()
    print(f"\nTotal Questions: {stats['total_questions']}")
    print(f"Categories: {len(stats['categories'])}")
    print(f"Difficulties: {list(stats['difficulties'].keys())}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED ✓")
    print("=" * 80)