#===================== Imports ===================================

from backend.groq_client import groq_chat
from backend.config.settings import get_settings, setup_langsmith

settings = get_settings()
setup_langsmith()

print("\n" + "="*60)
print("SIMPLE GROQ AGENT - TECH TERM EXPLAINER")
print("="*60)

print("\n Connecting to Groq LLM...")
print(f"Model: {settings.groq_model}")
print("Connected to Groq.\n")

# ==================== Prompt Template ============================

template = """
You are a helpful technical instructor explaining concepts to beginners.

Explain the following technical term in simple language that a beginner can understand:
Term: {term}

Provide:
1. A simple one-sentence definition
2. A real-world analogy
3. A practical example

Keep it concise and beginner-friendly.
"""

print("Prompt template created.")
print("\n Building Groq request function...\n")


def run_groq_chain(term: str) -> str:
    """Simulates LangChain style chain using Groq API only"""

    prompt_text = template.format(term=term)

    messages = [
        {"role": "system", "content": "You explain technical concepts simply."},
        {"role": "user", "content": prompt_text}
    ]

    return groq_chat(messages)


# ==================== Test Terms ================================

test_terms = ["n8n", "saas", "DSA"]

for term in test_terms:
    print(f"\n{'='*50}")
    print(f"Explaining: {term}")
    print("="*60)

    result = run_groq_chain(term)

    print(f"\n{result}")
    print("\n" + "="*60)
