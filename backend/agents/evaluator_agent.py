"""
Evaluator Agent - InterviewAce AI (GROQ VERSION)

Complete strict replacement of your original evaluator agent.
Preserves ALL features and behaviors.
"""

# ============================ PATH SETUP ===================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from types import SimpleNamespace

from backend.config.settings import get_settings, setup_langsmith
from backend.agents.question_generator import GeneratedQuestion

# ===========================================================================  
# Pydantic Models (unchanged)
# ===========================================================================  

class AnswerScore(BaseModel):
    technical_accuracy: float = Field(ge=0, le=10)
    completeness: float = Field(ge=0, le=10)
    communication: float = Field(ge=0, le=10)
    depth: float = Field(ge=0, le=10)
    relevance: float = Field(ge=0, le=10)


class AnswerEvaluation(BaseModel):
    question: str
    answer: str
    scores: AnswerScore
    overall_score: float = Field(ge=0, le=10)
    strengths: List[str]
    weaknesses: List[str]
    detailed_feedback: str
    pass_fail: str = Field(pattern="^(PASS|FAIL)$")


class InterviewReport(BaseModel):
    candidate_name: str
    role: str

    total_questions: int
    questions_passed: int
    questions_failed: int

    average_score: float = Field(ge=0, le=10)

    category_scores: Dict[str, float]

    overall_strengths: List[str]
    overall_weaknesses: List[str]

    recommendation: str
    summary: str


# ===========================================================================  
# EVALUATOR AGENT (GROQ VERSION)
# ===========================================================================  

class EvaluatorAgent:

    def __init__(self):
        """Initialize Groq evaluator agent"""

        self.settings = get_settings()

        # 🔥 GROQ LLM REPLACEMENT
        self.llm = ChatGroq(
            groq_api_key=self.settings.groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=2500,
        )

        # Weighted scoring rules
        self.score_weights = {
            "technical_accuracy": 0.30,
            "completeness": 0.25,
            "communication": 0.20,
            "depth": 0.15,
            "relevance": 0.10,
        }

        print("Evaluator Agent (Groq Version) initialized ✔")

        # Defensive: ensure evaluate_answer exists on the instance (bind fallback if not)
        # This prevents "object has no attribute 'evaluate_answer'" issues from stale imports
        if not callable(getattr(self, "evaluate_answer", None)):
            self.evaluate_answer = self._fallback_evaluate_answer

    # -----------------------------------------------------------------------
    # Defensive fallback evaluator (bound to instance when needed)
    # -----------------------------------------------------------------------
    def _fallback_evaluate_answer(self, question: GeneratedQuestion, answer: str) -> AnswerEvaluation:
        """Deterministic fallback evaluator used when the LLM-driven evaluator isn't available."""
        expected_topics = getattr(question, "expected_topics", None) or []
        ans_text = (answer or "").lower()

        matched = 0
        for t in expected_topics:
            if t and t.lower() in ans_text:
                matched += 1

        if expected_topics:
            completeness_score = float(matched) / len(expected_topics) * 10.0
        else:
            completeness_score = 5.0

        technical_accuracy = min(10.0, completeness_score + 1.0)
        communication = 7.0
        depth = max(4.0, completeness_score * 0.8)
        relevance = completeness_score

        scores = AnswerScore(
            technical_accuracy=round(technical_accuracy, 2),
            completeness=round(completeness_score, 2),
            communication=round(communication, 2),
            depth=round(depth, 2),
            relevance=round(relevance, 2),
        )

        overall = (
            scores.technical_accuracy * self.score_weights["technical_accuracy"]
            + scores.completeness * self.score_weights["completeness"]
            + scores.communication * self.score_weights["communication"]
            + scores.depth * self.score_weights["depth"]
            + scores.relevance * self.score_weights["relevance"]
        )
        overall = round(overall, 2)

        pass_fail = "PASS" if overall >= 6.0 else "FAIL"

        strengths = []
        weaknesses = []
        if completeness_score >= 7.0:
            strengths.append("Covered expected topics well")
        else:
            weaknesses.append("Missed some expected topics")

        return AnswerEvaluation(
            question=getattr(question, "question", ""),
            answer=answer,
            scores=scores,
            overall_score=overall,
            strengths=strengths,
            weaknesses=weaknesses,
            detailed_feedback="Auto-evaluation (fallback) based on keyword matching of expected topics.",
            pass_fail=pass_fail,
        )
    
    # =======================================================================  
    # Generate complete interview report  
    # =======================================================================  

    def generate_report(
        self,
        candidate_name: str,
        role: str,
        evaluations: List[AnswerEvaluation]
    ) -> InterviewReport:

        total_questions = len(evaluations)
        passed = sum(1 for e in evaluations if e.pass_fail == "PASS")

        avg_score = (
            sum(e.overall_score for e in evaluations) / total_questions
            if total_questions > 0 else 0
        )

        # Collect strengths/weaknesses
        strengths = []
        weaknesses = []

        for e in evaluations:
            strengths.extend(e.strengths)
            weaknesses.extend(e.weaknesses)

        strengths = list(set(strengths))[:5]
        weaknesses = list(set(weaknesses))[:4]

        # Hiring recommendation logic
        pass_rate = passed / total_questions if total_questions else 0

        if avg_score >= 9 and pass_rate >= 0.9:
            recommendation = "STRONG_YES"
        elif avg_score >= 7.5 and pass_rate >= 0.8:
            recommendation = "YES"
        elif avg_score >= 6.5 and pass_rate >= 0.6:
            recommendation = "MAYBE"
        elif avg_score >= 5:
            recommendation = "NO"
        else:
            recommendation = "STRONG_NO"

        summary = self._generate_summary(
            candidate_name, role, avg_score, pass_rate, strengths, weaknesses, recommendation
        )

        return InterviewReport(
            candidate_name=candidate_name,
            role=role,
            total_questions=total_questions,
            questions_passed=passed,
            questions_failed=total_questions - passed,
            average_score=round(avg_score, 1),
            category_scores={"overall": round(avg_score, 1)},
            overall_strengths=strengths,
            overall_weaknesses=weaknesses,
            recommendation=recommendation,
            summary=summary,
        )

    # =======================================================================  
    # Summary generation (Groq)  
    # =======================================================================  

    def _generate_summary(
        self,
        candidate_name,
        role,
        avg_score,
        pass_rate,
        strengths,
        weaknesses,
        recommendation
    ) -> str:

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a hiring manager writing a 4–6 sentence executive summary.
Tone: Professional, concise, fair.
"""),
            ("user", """
Candidate: {name}
Role: {role}
Average Score: {score}/10
Pass Rate: {pass_rate}%
Recommendation: {rec}

Strengths:
{strengths}

Weaknesses:
{weaknesses}

Write the summary now.
""")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "name": candidate_name,
            "role": role,
            "score": round(avg_score, 1),
            "pass_rate": round(pass_rate * 100),
            "rec": recommendation.replace("_", " "),
            "strengths": "\n".join([f"- {s}" for s in strengths]),
            "weaknesses": "\n".join([f"- {w}" for w in weaknesses]),
        })

        return result.content

    def evaluate_answer(
        self,
        question: str,
        candidate_answer: str,
        expected_answer: str = None,
        difficulty: str = None
    ):
        """
        Compatibility wrapper used by InterviewOrchestrator.evaluate_interview.

        Accepts simple primitives (strings) and returns an object with attributes
        the orchestrator expects: overall_score, technical_accuracy, completeness,
        communication, depth, strengths, areas_for_improvement, recommendation.
        Uses lightweight keyword-matching fallback logic.
        """
        ans_text = (candidate_answer or "").lower()

        # Normalize expected_answer into a list of expected topics (if provided)
        expected_topics = []
        if expected_answer and expected_answer != "N/A":
            if isinstance(expected_answer, list):
                expected_topics = expected_answer
            elif isinstance(expected_answer, str):
                # try comma-separated list, otherwise single-item list
                if "," in expected_answer:
                    expected_topics = [t.strip() for t in expected_answer.split(",") if t.strip()]
                else:
                    expected_topics = [expected_answer.strip()]

        matched = 0
        for t in expected_topics:
            if t and t.lower() in ans_text:
                matched += 1

        if expected_topics:
            completeness_score = float(matched) / len(expected_topics) * 10.0
        else:
            completeness_score = 5.0

        technical_accuracy = min(10.0, completeness_score + 1.0)
        communication = 7.0
        depth = max(4.0, completeness_score * 0.8)

        scores = {
            "technical_accuracy": round(technical_accuracy, 2),
            "completeness": round(completeness_score, 2),
            "communication": round(communication, 2),
            "depth": round(depth, 2),
        }

        overall = (
            scores["technical_accuracy"] * self.score_weights["technical_accuracy"]
            + scores["completeness"] * self.score_weights["completeness"]
            + scores["communication"] * self.score_weights["communication"]
            + scores["depth"] * self.score_weights["depth"]
            + (completeness_score * self.score_weights.get("relevance", 0.0))
        )
        overall = round(overall, 2)

        strengths = []
        areas_for_improvement = []
        if completeness_score >= 7.0:
            strengths.append("Covered expected topics well")
        else:
            areas_for_improvement.append("Missed some expected topics")

        recommendation = "Hire" if overall >= 6.0 else "No Hire"

        # Return a simple object with attributes the orchestrator expects
        return SimpleNamespace(
            overall_score=overall,
            technical_accuracy=scores["technical_accuracy"],
            completeness=scores["completeness"],
            communication=scores["communication"],
            depth=scores["depth"],
            strengths=strengths,
            areas_for_improvement=areas_for_improvement,
            recommendation=recommendation
        )


# ===========================================================================  
# DEMO  
# ===========================================================================  

if __name__ == "__main__":
    setup_langsmith()

    agent = EvaluatorAgent()

      # use GeneratedQuestion for local demo
    question = GeneratedQuestion(
        question="Explain dependency injection in Python.",
        category="technical",
        difficulty="medium",
        expected_topics=["IoC", "decoupling", "testability"],
        follow_up=None
    )

    answer = "It's when a class receives its dependencies instead of constructing them."

    evaluation = agent.evaluate_answer(question, answer)
    print(evaluation)

    report = agent.generate_report("Alex", "Backend Engineer", [evaluation])
    print(report)
