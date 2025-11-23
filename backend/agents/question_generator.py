"""
Question Generator Agent - InterviewAce AI (Groq Version)

This is the fully updated and conflict-free version.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from backend.config.settings import get_settings, setup_langsmith


# =========================================================
# 1. PYDANTIC MODELS (Used ONLY for parsing AI output)
# =========================================================

# ...existing code...
class GeneratedQuestion(BaseModel):
    """Model representing a single generated question from the LLM"""

    question: str = Field(description="The interview question text")
    category: str = Field(description="technical, behavioral or system_design")
    difficulty: str = Field(description="easy, medium, or hard")
    expected_topics: Optional[List[str]] = Field(default=None, description="Topics the answer should cover")
    expected_answer: Optional[Union[str, List[str]]] = Field(default=None, description="Optional canonical answer or list of key points")
    follow_up: Optional[str] = Field(default=None, description="Optional follow-up question")
    follow_up_questions: Optional[List[str]] = Field(default=None, description="Optional list of follow-up questions (plural)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Explain the difference between REST and GraphQL APIs.",
                "category": "technical",
                "difficulty": "medium",
                "expected_topics": ["REST", "GraphQL", "trade-offs"],
                "expected_answer": "REST uses fixed endpoints; GraphQL allows clients to request only needed fields.",
                "follow_up": "When would you choose GraphQL instead of REST?",
                "follow_up_questions": ["How would you design pagination?", "What about caching?"]
            }
        }
    )
# ...existing code...

class QuestionSet(BaseModel):
    """Model representing an entire set of generated questions"""

    role: str = Field(description="Job role or topic")
    questions: List[GeneratedQuestion] = Field(description="Generated questions list")
    total_count: int = Field(description="Total number of questions returned")


# =========================================================
# 2. QUESTION GENERATOR AGENT
# =========================================================

class QuestionGeneratorAgent:
    def __init__(self):
        """Initialize the structured-output Groq-based generator"""

        self.settings = get_settings()

        # Use Groq instead of OpenAI
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=self.settings.groq_api_key,
            temperature=0.7,
            max_tokens=3000,
        )

        # Structured output parser
        self.output_parser = PydanticOutputParser(pydantic_object=QuestionSet)

        # Build prompt template
        self.prompt = self._build_prompt()

        # LLM chain: prompt → Groq LLM → Pydantic parser
        self.chain = self.prompt | self.llm | self.output_parser

        print("🚀 Question Generator Agent (using Groq) Initialized")

    # -----------------------------------------------------
    # Build prompt template
    # -----------------------------------------------------
    def _build_prompt(self) -> ChatPromptTemplate:
        format_instructions = self.output_parser.get_format_instructions()

        # Escape for LangChain
        format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

        return ChatPromptTemplate.from_messages([
            ("system", f"""
You are an expert technical interviewer with 15+ years of experience.

Your responsibilities:
- Generate structured interview questions
- Ensure topic variety
- Ensure difficulty balance
- Follow the schema exactly
- Output ONLY valid JSON

Follow this JSON schema:
{format_instructions}
"""),
            ("user", """
Generate interview questions for this job role:

JOB DESCRIPTION:
{job_description}

CONTEXT:
{context}

NUMBER OF QUESTIONS REQUESTED:
{num_questions}

Respond ONLY using the JSON structure described above.
""")
        ])

    # -----------------------------------------------------
    # Generate questions (standard)
    # -----------------------------------------------------
    def generate_questions(self, job_description: str, num_questions: int = 10, context: str = "") -> QuestionSet:
        try:
            print(f"\n🤖 Generating {num_questions} questions using Groq...")

            result = self.chain.invoke({
                "job_description": job_description,
                "num_questions": num_questions,
                "context": context or "No extra context."
            })

            print("✅ Questions generated successfully")
            return result

        except Exception as e:
            print(f"❌ Error generating questions: {e}")
            raise

    # -----------------------------------------------------
    # Resume-aware generation
    # -----------------------------------------------------
    def generating_questions_by_resume(self, job_description: str, resume_text: str, num_questions: int = 10) -> QuestionSet:

        context = f"""
CANDIDATE RESUME:
{resume_text}

GUIDELINES:
- Focus questions on the resume's skills and projects.
- Test claimed experience deeply.
- Compare resume skills with job requirements.
"""

        return self.generate_questions(
            job_description=job_description,
            num_questions=num_questions,
            context=context,
        )


# =========================================================
# DEMO (optional)
# =========================================================

if __name__ == "__main__":
    setup_langsmith()

    agent = QuestionGeneratorAgent()

    jd = """
Senior Python + GenAI Engineer.
Skills required: LLMs, LangChain, RAG, Vector DBs, FastAPI, AWS.
"""

    qs = agent.generate_questions(jd, num_questions=5)
    print(qs)
