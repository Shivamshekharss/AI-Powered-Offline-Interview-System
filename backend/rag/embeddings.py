"""
Embeddings Manager - InterviewAce AI RAG System (HuggingFace Version)

Replaces Groq API embeddings with FREE local HuggingFace embedding model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config.settings import get_settings


class EmbeddingManager:
    """Manages text-to-vector conversion using HuggingFace embeddings (FREE)"""

    def __init__(self):
        self.settings = get_settings()

        # Model name from .env
        self.model_name = self.settings.hf_embedding_model  

        # Load model once
        print(f"🔄 Loading HuggingFace embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Cache
        self.cache: Dict[str, List[float]] = {}

        # Set embedding dimension (used by VectorStore)
        self.dimension = self.model.get_sentence_embedding_dimension()

        print(f"✅ HuggingFace Embedding Model Loaded!")
        print(f"📏 Embedding Dimension: {self.dimension}")

    # ------------------------------------------------------------------
    # Core embedding (single)
    # ------------------------------------------------------------------
    def create_embedding(self, text: str) -> List[float]:
        """Convert single text to vector using HF embeddings"""

        text = text.strip().replace("\n", " ")

        # Cache check
        if text in self.cache:
            print(f"  Cache hit for: {text[:50]}...")
            return self.cache[text]

        # Encode text
        try:
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()

            self.cache[text] = embedding
            print(f"  Created embedding for: {text[:50]}... ({len(embedding)} dims)")

            return embedding

        except Exception as e:
            print(f"❌ Error creating embedding: {e}")
            raise

    # ------------------------------------------------------------------
    # Batch Embeddings
    # ------------------------------------------------------------------
    def create_embeddings_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:

        if show_progress:
            print(f"\nCreating embeddings for {len(texts)} texts...")

        clean_texts = []
        indices = []
        embeddings = [None] * len(texts)

        # Cache lookup
        for i, text in enumerate(texts):
            text = text.strip().replace("\n", " ")

            if text in self.cache:
                embeddings[i] = self.cache[text]
                if show_progress:
                    print(f"  [{i+1}/{len(texts)}] Cache hit: {text[:50]}...")
            else:
                clean_texts.append(text)
                indices.append(i)

        # Encode remaining texts
        if clean_texts:
            if show_progress:
                print(f"  Encoding {len(clean_texts)} new texts using HF model...")

            try:
                batch_vectors = self.model.encode(clean_texts, convert_to_numpy=True).tolist()

                for n, idx in enumerate(indices):
                    text = clean_texts[n]
                    vector = batch_vectors[n]

                    self.cache[text] = vector
                    embeddings[idx] = vector

                    if show_progress:
                        print(f"  [{idx+1}/{len(texts)}] Embedded: {text[:50]}...")

            except Exception as e:
                print(f"❌ Batch embedding error: {e}")
                raise

        print("✅ Completed batch embedding!")
        return embeddings

    # ------------------------------------------------------------------
    # Similarity Calculations
    # ------------------------------------------------------------------
    def calculate_similarity(self, embedding1, embedding2) -> float:
        """Cosine similarity"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    def find_most_similar(self, query_embedding, candidate_embeddings, top_k=5):
        sims = []
        for i, emb in enumerate(candidate_embeddings):
            sims.append((i, self.calculate_similarity(query_embedding, emb)))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def get_cache_size(self):
        return len(self.cache)

    def clear_cache(self):
        self.cache.clear()
        print("Embedding cache cleared.")
