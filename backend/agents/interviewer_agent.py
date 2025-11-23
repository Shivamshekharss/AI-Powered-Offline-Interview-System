"""
Interviewer Agent - InterviewAce AI (GROQ Version)

Fully rewritten to use Groq API instead of OpenAI / LangChain.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from backend.config.settings import get_settings
from backend.groq_client import groq_chat
from backend.agents.question_generator import GeneratedQuestion as InterviewQuestion



# ==================================== DATA MODELS ==================================

class InterviewResponse(BaseModel):
    question: str
    answer: str
    follow_up_needed: bool


class InterviewSession(BaseModel):
    candidate_name: str
    role: str
    questions: List[InterviewQuestion]
    current_question_index: int = 0
    responses: List[InterviewResponse] = []
    is_complete: bool = False



# ================================ INTERVIEWER AGENT ================================

class InterviewerAgent:
    """
    Groq-powered interviewer:
    - Asks natural interview questions
    - Processes candidate responses
    - Generates follow-ups
    - Maintains conversation history
    """

    def __init__(self):
        self.settings = get_settings()
        self.chat_history: List[Dict] = []
        print("🤖 Interviewer Agent (Groq Version) Initialized")

    # ------------------------------------------------------------------------------

    def start_interview(
        self,
        candidate_name: str,
        role: str,
        questions: List[InterviewQuestion]
    ) -> InterviewSession:

        session = InterviewSession(
            candidate_name=candidate_name,
            role=role,
            questions=questions
        )

        # Reset memory
        self.chat_history = []

        # Generate welcome message
        welcome = self._generate_welcome_message(candidate_name, role)
        print(f"\n{welcome}\n")

        return session

    # ------------------------------------------------------------------------------

    def _generate_welcome_message(self, candidate_name: str, role: str) -> str:
        """ Generate friendly welcome message using Groq """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly, professional interviewer. "
                    "Generate a short (2-3 sentences) welcome message."
                )
            },
            {
                "role": "user",
                "content": f"Candidate name: {candidate_name}, Role: {role}"
            }
        ]

        return groq_chat(messages)

    # ------------------------------------------------------------------------------

    def ask_question(self, session: InterviewSession) -> str:
        """
        Get next question (natural phrasing).
        """

        if session.current_question_index >= len(session.questions):
            session.is_complete = True
            return "That completes all the questions. Thank you!"

        current_q = session.questions[session.current_question_index]

        question_text = self._phrase_question_naturally(
            current_q,
            session.current_question_index + 1,
            len(session.questions)
        )

        self.chat_history.append({"role": "assistant", "content": question_text})

        return question_text

    # ------------------------------------------------------------------------------

    def _phrase_question_naturally(
        self,
        question: InterviewQuestion,
        question_num: int,
        total_questions: int
    ) -> str:
        """ Rephrase question in friendly interviewer tone """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are conducting an interview. "
                    "Rephrase the provided question in a conversational, friendly, "
                    "but professional manner. Do NOT change the meaning."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question {question_num} of {total_questions}\n"
                    f"Category: {question.category}\n"
                    f"Difficulty: {question.difficulty}\n"
                    f"Question: {question.question}\n\n"
                    f"Rephrase it naturally."
                )
            }
        ]

        return groq_chat(messages)

    # ------------------------------------------------------------------------------

    def receive_answer(self, session: InterviewSession, answer: str) -> Optional[str]:
        """ Process candidate answer and decide follow-up """

        current_q = session.questions[session.current_question_index]

        self.chat_history.append({"role": "user", "content": answer})

        follow_up = self._generate_follow_up(current_q, answer)

        session.responses.append(
            InterviewResponse(
                question=current_q.question,
                answer=answer,
                follow_up_needed=(follow_up is not None)
            )
        )

        if follow_up:
            self.chat_history.append({"role": "assistant", "content": follow_up})
            return follow_up

        # Go to next question
        session.current_question_index += 1
        return None

    # ------------------------------------------------------------------------------

    def _generate_follow_up(
        self,
        question: InterviewQuestion,
        answer: str
    ) -> Optional[str]:

        if not question.follow_up:
            return None

        # Step 1: Ask Groq whether follow-up is needed
        decision_messages = [
            {
                "role": "system",
                "content": (
                    "You are an interviewer. Decide if a follow-up question is needed.\n"
                    "Reply ONLY with YES or NO.\n\n"
                    "Say YES if:\n"
                    "- Answer is vague or incomplete\n"
                    "- Key topics missing\n\n"
                    "Say NO if answer is complete."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question.question}\n"
                    f"Expected topics: {', '.join(question.expected_topics)}\n"
                    f"Candidate Answer: {answer}\n\n"
                    f"Should I ask the follow-up question?"
                )
            }
        ]

        decision = groq_chat(decision_messages).strip().upper()

        if decision == "YES":
            return f"{question.follow_up}"

        return None

    # ------------------------------------------------------------------------------

    def provide_feedback(self, session: InterviewSession) -> str:
        """ Final interview feedback """

        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in self.chat_history
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional interviewer. Provide a concise 3–4 sentence summary.\n"
                    "Include:\n"
                    "1. Strength\n"
                    "2. Area that stood out\n"
                    "3. A warm closing message"
                )
            },
            {"role": "user", "content": history_text}
        ]

        return groq_chat(messages)



# ============================== DEMO / TEST MODE ===================================

def conduct_interview_interactive(
    candidate_name: str,
    role: str,
    questions: List[InterviewQuestion]
):
    """
    CLI demo for testing
    """

    interviewer = InterviewerAgent()
    session = interviewer.start_interview(candidate_name, role, questions)

    print("\n" + "=" * 80)
    print("INTERVIEW SESSION STARTED")
    print("=" * 80)

    while not session.is_complete:
        q = interviewer.ask_question(session)
        print(f"\nInterviewer: {q}\n")

        if session.is_complete:
            break

        answer = input("Your answer: ").strip()

        if answer.lower() == "quit":
            print("Interview ended early.")
            break

        follow_up = interviewer.receive_answer(session, answer)

        if follow_up:
            print(f"\nInterviewer: {follow_up}\n")
            follow_up_answer = input("Your answer: ").strip()

            interviewer.chat_history.append(
                {"role": "user", "content": follow_up_answer}
            )

            session.current_question_index += 1

    print("\n" + "=" * 80)
    print("INTERVIEW COMPLETE")
    print("=" * 80)

    feedback = interviewer.provide_feedback(session)
    print(f"\nInterviewer: {feedback}\n")
