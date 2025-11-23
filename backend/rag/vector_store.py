"""
Vector Store - InterviewAce AI RAG System

PURPOSE:
    Store interview questions as vectors and search them by meaning.
    
WHAT IT DOES:
    - Stores questions with their embeddings (vectors)
    - Searches questions by semantic similarity
    - Manages a persistent knowledge base
    
WHY WE NEED IT:
    - Fast retrieval (milliseconds vs seconds)
    - Cost-effective (no GPT calls for existing questions)
    - Quality control (curated, proven questions)
    - Scalable (handles 1000s of questions)

TECH STACK:
    - FAISS: Vector database (Facebook AI Similarity Search)
    - OpenAI Embeddings: Convert text to vectors
    - JSON: Store metadata (categories, difficulty, etc.)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Optional, Tuple
import json
import faiss
import numpy as np
from datetime import datetime

from backend.rag.embeddings import EmbeddingManager


class Question:
    """
    Represents a single interview question with metadata
    
    Why this class exists:
        - Organizes question data in a structured way
        - Makes it easy to add/modify question attributes
        - Type hints for better code quality
    """
    
    def __init__(
        self,
        id: str,
        question: str,
        category: str,
        difficulty: str,
        topics: List[str],
        expected_answer: str,
        follow_ups: List[str],
        embedding: Optional[List[float]] = None
    ):
        """
        Initialize a question object
        
        Args:
            id: Unique identifier (e.g., "python_001")
            question: The actual question text
            category: Category (e.g., "Python", "JavaScript")
            difficulty: Easy/Medium/Hard
            topics: Related topics (e.g., ["decorators", "functions"])
            expected_answer: What we're looking for in an answer
            follow_ups: Deeper questions to ask based on answer
            embedding: Vector representation (added when stored)
        
        Why we store metadata:
            - Enables filtering (show only Python questions)
            - Tracks difficulty progression
            - Helps with follow-up questions
            - Improves searchability
        """
        self.id = id
        self.question = question
        self.category = category
        self.difficulty = difficulty
        self.topics = topics
        self.expected_answer = expected_answer
        self.follow_ups = follow_ups
        self.embedding = embedding
        self.created_at = datetime.now().isoformat()
    
    
    def to_dict(self) -> Dict:
        """
        Convert question to dictionary for JSON storage
        
        Why we need this:
            - FAISS only stores vectors, not metadata
            - We need separate JSON file for metadata
            - Makes it easy to save/load
        
        Returns:
            Dictionary with all question data
        """
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "difficulty": self.difficulty,
            "topics": self.topics,
            "expected_answer": self.expected_answer,
            "follow_ups": self.follow_ups,
            "embedding": self.embedding,  # Store for reference
            "created_at": self.created_at
        }
    
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Question':
        """
        Create question object from dictionary
        
        Why we need this:
            - Load questions from JSON storage
            - Reverse of to_dict()
        
        Args:
            data: Dictionary with question data
            
        Returns:
            Question object
        """
        return cls(
            id=data["id"],
            question=data["question"],
            category=data["category"],
            difficulty=data["difficulty"],
            topics=data["topics"],
            expected_answer=data["expected_answer"],
            follow_ups=data["follow_ups"],
            embedding=data.get("embedding")
        )


class VectorStore:
    """
    Manages the vector database for interview questions
    
    ARCHITECTURE:
    
        User Query: "Python decorators"
              ↓
        [Embedding Manager] → Convert to vector
              ↓
        [FAISS Index] → Find similar vectors
              ↓
        [Metadata Store] → Get full question details
              ↓
        Return: Top matching questions
    
    STORAGE:
        - FAISS index: Stores vectors (fast search)
        - JSON file: Stores metadata (full question details)
        - In-memory cache: For quick repeated access
    """
    
    def __init__(self, storage_path: str = "backend/data/vector_store"):
        """
        Initialize the vector store
        
        Args:
            storage_path: Where to save FAISS index and metadata
        
        Why we need persistent storage:
            - Don't re-embed questions every time
            - Save money on embedding API calls
            - Faster startup time
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist
        
        # File paths
        self.index_path = self.storage_path / "questions.index"
        self.metadata_path = self.storage_path / "questions.json"
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager()
        
        # Dimension of embeddings (text-embedding-3-large = 3072)
        self.dimension = 3072
        
        # FAISS index (will be created or loaded)
        self.index: Optional[faiss.Index] = None
        
        # Metadata storage (list of Question objects)
        self.questions: List[Question] = []
        
        # Load existing data if available
        self._load()
        
        print(f"Vector Store initialized:")
        print(f"  Storage: {self.storage_path}")
        print(f"  Questions: {len(self.questions)}")
        print(f"  Dimension: {self.dimension}")
    
    
    def _load(self):
        """
        Load existing vector store from disk
        
        Why private method (underscore):
            - Internal implementation detail
            - Called automatically in __init__
            - Users don't need to call this directly
        
        What it does:
            1. Load FAISS index (vectors)
            2. Load metadata (question details)
            3. Sync both in memory
        """
        try:
            # Load FAISS index
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                print(f"  ✓ Loaded FAISS index: {self.index.ntotal} vectors")
            else:
                # Create new index
                # L2 distance (Euclidean) - measures vector similarity
                self.index = faiss.IndexFlatL2(self.dimension)
                print(f"  ✓ Created new FAISS index")
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    self.questions = [Question.from_dict(q) for q in data]
                print(f"  ✓ Loaded metadata: {len(self.questions)} questions")
            
        except Exception as e:
            print(f"  ⚠ Error loading: {e}")
            print(f"  ✓ Starting with empty store")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.questions = []
    
    
    def _save(self):
        """
        Save vector store to disk
        
        What it saves:
            1. FAISS index (all vectors)
            2. Metadata JSON (all question details)
        
        When called:
            - After adding new questions
            - After bulk operations
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                data = [q.to_dict() for q in self.questions]
                json.dump(data, f, indent=2)
            
            print(f"  ✓ Saved to disk: {len(self.questions)} questions")
            
        except Exception as e:
            print(f"  ✗ Error saving: {e}")
            raise
    
    
    def add_question(self, question: Question) -> bool:
        """
        Add a single question to the vector store
        
        Process:
            1. Create embedding for question text
            2. Add vector to FAISS index
            3. Store question metadata
            4. Save to disk
        
        Args:
            question: Question object to add
            
        Returns:
            True if successful, False otherwise
        
        Why return bool:
            - Allows error handling by caller
            - Can check success before proceeding
        """
        try:
            print(f"\n📝 Adding question: {question.id}")
            
            # Step 1: Create embedding
            embedding = self.embedding_manager.create_embedding(question.question)
            question.embedding = embedding  # Store in question object
            
            # Step 2: Convert to numpy array (FAISS requirement)
            vector = np.array([embedding], dtype='float32')
            
            # Step 3: Add to FAISS index
            self.index.add(vector)
            
            # Step 4: Add to metadata
            self.questions.append(question)
            
            # Step 5: Save to disk
            self._save()
            
            print(f"  ✓ Added successfully (total: {len(self.questions)})")
            return True
            
        except Exception as e:
            print(f"  ✗ Error adding question: {e}")
            return False
    
    
    def add_questions_batch(self, questions: List[Question]) -> int:
        """
        Add multiple questions efficiently
        
        Why batch operation:
            - Faster than adding one by one
            - Single embedding API call for all
            - One disk write at the end
        
        Args:
            questions: List of Question objects
            
        Returns:
            Number of questions successfully added
        """
        print(f"\n📚 Adding {len(questions)} questions in batch...")
        
        try:
            # Step 1: Get all question texts
            texts = [q.question for q in questions]
            
            # Step 2: Create embeddings in batch (efficient!)
            embeddings = self.embedding_manager.create_embeddings_batch(texts)
            
            # Step 3: Add embeddings to questions
            for question, embedding in zip(questions, embeddings):
                question.embedding = embedding
            
            # Step 4: Convert to numpy array
            vectors = np.array(embeddings, dtype='float32')
            
            # Step 5: Add all to FAISS index
            self.index.add(vectors)
            
            # Step 6: Add all to metadata
            self.questions.extend(questions)
            
            # Step 7: Single save operation
            self._save()
            
            print(f"  ✓ Added {len(questions)} questions successfully")
            return len(questions)
            
        except Exception as e:
            print(f"  ✗ Error in batch add: {e}")
            return 0
    
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        category_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None
    ) -> List[Tuple[Question, float]]:
        """
        Search for similar questions using semantic search
        
        THE CORE OF RAG!
        
        Process:
            1. Convert query to embedding
            2. Search FAISS for similar vectors
            3. Get metadata for matched questions
            4. Apply filters if specified
            5. Return top results with scores
        
        Args:
            query: Search query (e.g., "Python decorators")
            top_k: Number of results to return
            category_filter: Only return this category (e.g., "Python")
            difficulty_filter: Only return this difficulty (e.g., "Medium")
        
        Returns:
            List of (Question, similarity_score) tuples
            
        Score meaning:
            - Lower is better (L2 distance)
            - Typically 0.5 to 2.0 for relevant results
        """
        print(f"\n🔍 Searching: '{query}'")
        print(f"  Filters: category={category_filter}, difficulty={difficulty_filter}")
        
        try:
            # Step 1: Create query embedding
            query_embedding = self.embedding_manager.create_embedding(query)
            query_vector = np.array([query_embedding], dtype='float32')
            
            # Step 2: Search FAISS
            # k = number of results to retrieve
            # We get more than needed for filtering
            search_k = min(top_k * 3, len(self.questions))
            distances, indices = self.index.search(query_vector, search_k)
            
            # Step 3: Get questions and apply filters
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                # Get question
                question = self.questions[idx]
                
                # Apply category filter
                if category_filter and question.category != category_filter:
                    continue
                
                # Apply difficulty filter
                if difficulty_filter and question.difficulty != difficulty_filter:
                    continue
                
                # Add to results
                results.append((question, float(distance)))
                
                # Stop if we have enough
                if len(results) >= top_k:
                    break
            
            # Step 4: Print results
            print(f"\n  Found {len(results)} results:")
            for i, (q, score) in enumerate(results, 1):
                print(f"    {i}. [{q.category}] {q.question[:60]}... (score: {score:.3f})")
            
            return results
            
        except Exception as e:
            print(f"  ✗ Search error: {e}")
            return []
    
    
    def get_by_category(self, category: str) -> List[Question]:
        """
        Get all questions in a category
        
        Why we need this:
            - Show all Python questions
            - Filter by topic
            - Category-specific review
        
        Args:
            category: Category name (e.g., "Python")
            
        Returns:
            List of questions in that category
        """
        return [q for q in self.questions if q.category == category]
    
    
    def get_by_difficulty(self, difficulty: str) -> List[Question]:
        """
        Get all questions of a difficulty level
        
        Why we need this:
            - Progressive difficulty
            - Skill-based filtering
            - Training progression
        
        Args:
            difficulty: Difficulty level (Easy/Medium/Hard)
            
        Returns:
            List of questions at that difficulty
        """
        return [q for q in self.questions if q.difficulty == difficulty]
    
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the question bank
        
        Why we need this:
            - Monitor question coverage
            - Identify gaps
            - Dashboard metrics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_questions": len(self.questions),
            "categories": {},
            "difficulties": {},
            "topics": {}
        }
        
        # Count by category
        for q in self.questions:
            stats["categories"][q.category] = stats["categories"].get(q.category, 0) + 1
            stats["difficulties"][q.difficulty] = stats["difficulties"].get(q.difficulty, 0) + 1
            
            # Count topics
            for topic in q.topics:
                stats["topics"][topic] = stats["topics"].get(topic, 0) + 1
        
        return stats


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    """
    Test the vector store with sample questions
    
    This demonstrates:
        1. Creating questions
        2. Adding to store
        3. Searching semantically
        4. Filtering results
    """
    
    print("=" * 80)
    print("TESTING: Vector Store")
    print("=" * 80)
    
    # Initialize store
    store = VectorStore()
    
    # Create sample questions
    print("\n" + "=" * 80)
    print("Creating Sample Questions")
    print("=" * 80)
    
    sample_questions = [
        Question(
            id="python_001",
            question="What are Python decorators and how do they work?",
            category="Python",
            difficulty="Medium",
            topics=["decorators", "functions", "metaprogramming"],
            expected_answer="A decorator is a function that wraps another function to extend its behavior...",
            follow_ups=[
                "Can you write a custom decorator?",
                "What's the difference between @staticmethod and @classmethod?"
            ]
        ),
        Question(
            id="python_002",
            question="Explain Python generators and their benefits",
            category="Python",
            difficulty="Medium",
            topics=["generators", "iterators", "memory"],
            expected_answer="Generators are functions that yield values one at a time using yield keyword...",
            follow_ups=[
                "What's the difference between yield and return?",
                "When would you use a generator over a list?"
            ]
        ),
        Question(
            id="js_001",
            question="What are JavaScript promises and async/await?",
            category="JavaScript",
            difficulty="Medium",
            topics=["promises", "async", "callbacks"],
            expected_answer="Promises represent eventual completion or failure of an async operation...",
            follow_ups=[
                "How do you handle errors in promises?",
                "What's the difference between Promise.all and Promise.race?"
            ]
        ),
        Question(
            id="python_003",
            question="What is the difference between list and tuple in Python?",
            category="Python",
            difficulty="Easy",
            topics=["data-structures", "lists", "tuples"],
            expected_answer="Lists are mutable, tuples are immutable...",
            follow_ups=[
                "When would you use a tuple over a list?",
                "Can you modify a tuple?"
            ]
        ),
        Question(
            id="python_004",
            question="Explain the Global Interpreter Lock (GIL) in Python",
            category="Python",
            difficulty="Hard",
            topics=["concurrency", "threading", "performance"],
            expected_answer="The GIL is a mutex that protects access to Python objects...",
            follow_ups=[
                "How does GIL affect multi-threaded programs?",
                "What are alternatives to GIL?"
            ]
        )
    ]
    
    # Add questions
    print("\n" + "=" * 80)
    print("Adding Questions to Store")
    print("=" * 80)
    
    store.add_questions_batch(sample_questions)
    
    # Test searches
    print("\n" + "=" * 80)
    print("TEST 1: Search for decorators")
    print("=" * 80)
    
    results = store.search("How do decorators work in Python?", top_k=3)
    
    print("\n" + "=" * 80)
    print("TEST 2: Search with category filter")
    print("=" * 80)
    
    results = store.search(
        "asynchronous programming",
        top_k=3,
        category_filter="JavaScript"
    )
    
    print("\n" + "=" * 80)
    print("TEST 3: Search with difficulty filter")
    print("=" * 80)
    
    results = store.search(
        "Python concepts",
        top_k=3,
        difficulty_filter="Easy"
    )
    
    # Get statistics
    print("\n" + "=" * 80)
    print("Question Bank Statistics")
    print("=" * 80)
    
    stats = store.get_statistics()
    print(f"\nTotal Questions: {stats['total_questions']}")
    print(f"\nBy Category:")
    for cat, count in stats['categories'].items():
        print(f"  {cat}: {count}")
    print(f"\nBy Difficulty:")
    for diff, count in stats['difficulties'].items():
        print(f"  {diff}: {count}")
    print(f"\nTop Topics:")
    sorted_topics = sorted(stats['topics'].items(), key=lambda x: x[1], reverse=True)
    for topic, count in sorted_topics[:5]:
        print(f"  {topic}: {count}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED ✓")
    print("=" * 80)