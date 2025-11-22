# AI-Powered-Offline-Interview-System
🔥 AI Interview Simulator (Offline, Multi-Agent, Llama-Powered)

A fully offline, privacy-preserving AI interview simulator built with local LLMs using Ollama, featuring a multi-agent architecture, real-time evaluation, dynamic question generation, and a clean Streamlit UI.

🚀 Features
🧠 Multi-Agent Intelligence

Interviewer Agent → Conducts the interview and asks adaptive questions

Question Generator → Creates role-specific, difficulty-scaled questions

Evaluator Agent → Scores candidate answers (technical depth, clarity, relevance)

Explanation Agent → Gives model-backed explanations and corrections

⚙️ Offline-First Architecture

No API keys

No internet required

Powered 100% by Ollama + Llama 3.1 + local embeddings

📚 Retrieval-Augmented Generation (RAG)

Uses nomic-embed-text for embeddings

Stores vectors in ChromaDB

Improves interview quality using contextual retrieval

🖥️ Clean Frontend

Streamlit interface

Real-time interaction

Persistent interview sessions
🏗️ Architecture Overview
           ┌──────────────────────────────┐
           │         Streamlit UI          │
           └───────────────┬──────────────┘
                           │
                 Interview Orchestrator
                           │
   ┌────────────────┬──────────────┬─────────────────┐
   │                │              │                 │
Question     Interviewer     Evaluator      Explanation
Generator         Agent          Agent            Agent
   │                │              │                 │
   └─────────────── Llama 3.1 via Ollama ────────────┘
                           │
                  Local Embeddings (nomic)
                           │
                   Chroma Vector Store
                   
  📦 Folder Structure

  project-root/
│── app.py
│── README.md
│── .env
│── backend/
│    ├── agents/
│    │    ├── interviewer_agent.py
│    │    ├── question_generator.py
│    │    ├── evaluator_agent.py
│    │    ├── simple_explanation.py
│    ├── orchestrator/
│    │    └── interview_orchestrator.py
│    ├── core/
│    │    ├── config.py
│    │    ├── embeddings.py
│    │    └── vector_store.py
│── data/
│── logs/
│── venv/



