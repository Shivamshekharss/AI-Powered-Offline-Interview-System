"""
Interview Pipeline Orchestrator - InterviewAce AI

PURPOSE:
    Coordinate all components to run complete end-to-end interviews
    
WHAT IT DOES:
    - Orchestrates the entire interview workflow
    - Retrieves questions using RAG
    - Conducts interviews with AI agents
    - Evaluates answers automatically
    - Generates comprehensive reports
    
WHY WE NEED IT:
    - Single entry point for interviews
    - Automatic coordination of all agents
    - Production-ready pipeline
    - Demonstrates complete GenAI system
    
COMPONENTS ORCHESTRATED:
    1. RAG Retriever - Find relevant questions
    2. Question Generator - Create custom questions
    3. Interviewer Agent - Conduct conversation
    4. Evaluator Agent - Score answers
    5. Report Generator - Create final report

WORKFLOW:
    Job Description → RAG Retrieval → Interview → Evaluation → Report
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from backend.rag.rag_retriever import RAGRetriever, RetrievalResult
from backend.rag.vector_store import Question
from backend.agents.question_generator import QuestionGeneratorAgent
from backend.agents.interviewer_agent import InterviewerAgent
from backend.agents.evaluator_agent import EvaluatorAgent


@dataclass
class InterviewQuestion:
    """
    Represents a question in the interview
    
    Why this class:
        - Tracks question source (RAG or generated)
        - Stores user's answer
        - Links to evaluation
        - Complete audit trail
    """
    question: str
    category: str
    difficulty: str
    source: str
    topics: List[str] = field(default_factory=list)
    expected_answer: str = "N/A"
    follow_ups: List[str] = field(default_factory=list)


@dataclass
class InterviewSession:
    """
    Complete interview session data
    
    Why this class:
        - Maintains all interview state
        - Easy to serialize/save
        - Comprehensive record
        - Useful for analytics
    """
    session_id: str
    candidate_name: str
    job_title: str
    job_description: str
    start_time: str
    end_time: Optional[str] = None
    
    questions: List[InterviewQuestion] = field(default_factory=list)
    overall_score: Optional[float] = None
    category_scores: Dict[str, float] = field(default_factory=dict)
    recommendation: Optional[str] = None
    detailed_feedback: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving/API response"""
        return {
            "session_id": self.session_id,
            "candidate_name": self.candidate_name,
            "job_title": self.job_title,
            "job_description": self.job_description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "questions": [
                {
                    "question": q.question,
                    "category": q.category,
                    "difficulty": q.difficulty,
                    "source": q.source,
                    "topics": q.topics,
                    "user_answer": q.user_answer,
                    "evaluation": q.evaluation
                }
                for q in self.questions
            ],
            "overall_score": self.overall_score,
            "category_scores": self.category_scores,
            "recommendation": self.recommendation,
            "detailed_feedback": self.detailed_feedback
        }


class InterviewOrchestrator:
    """
    Master orchestrator for complete interview pipeline
    
    ARCHITECTURE:
    
        InterviewOrchestrator
              ↓
        ┌─────────────────────────────────┐
        │  Component Coordination         │
        ├─────────────────────────────────┤
        │ 1. RAG Retriever                │
        │    → Find relevant questions    │
        │                                 │
        │ 2. Question Generator           │
        │    → Create custom questions    │
        │                                 │
        │ 3. Interviewer Agent            │
        │    → Conduct conversation       │
        │                                 │
        │ 4. Evaluator Agent              │
        │    → Score answers              │
        │                                 │
        │ 5. Report Generator             │
        │    → Create final report        │
        └─────────────────────────────────┘
    
    USAGE:
        orchestrator = InterviewOrchestrator()
        
        # Run complete interview
        session = orchestrator.run_complete_interview(
            candidate_name="Alice",
            job_description="Senior Python Engineer...",
            num_questions=5
        )
        
        # Get report
        report = orchestrator.generate_report(session)
    """
    
    def __init__(self):
        """
        Initialize the orchestrator with all components
        
        What happens:
            1. Load RAG retriever (question bank)
            2. Initialize question generator
            3. Initialize interviewer agent
            4. Initialize evaluator agent
            5. Ready to orchestrate!
        """
        print("=" * 80)
        print("INITIALIZING INTERVIEW ORCHESTRATOR")
        print("=" * 80)
        
        print("\n📦 Loading components...")
        
        # Initialize RAG retriever
        print("  1. RAG Retriever...")
        self.rag_retriever = RAGRetriever()
        
        # Initialize question generator (fallback)
        print("  2. Question Generator...")
        self.question_generator = QuestionGeneratorAgent()
        
        # Initialize interviewer agent
        print("  3. Interviewer Agent...")
        self.interviewer = InterviewerAgent()
        
        # Initialize evaluator agent
        print("  4. Evaluator Agent...")
        self.evaluator = EvaluatorAgent()
        
        print("\n✅ All components loaded successfully!")
        print("=" * 80)
    
    
    def prepare_questions(
        self,
        job_description: str,
        num_questions: int = 5,
        use_rag: bool = True
    ) -> List[InterviewQuestion]:
        """
        Prepare interview questions using RAG + generation
        
        Strategy:
            1. Try RAG retrieval first (fast, curated)
            2. Fall back to generation if needed
            3. Mix retrieval + generation for variety
        
        Process:
            1. Analyze job description with RAG
            2. Retrieve relevant questions
            3. Generate additional if < num_questions
            4. Convert to InterviewQuestion format
        
        Args:
            job_description: Full job posting
            num_questions: Total questions needed
            use_rag: Whether to use RAG (vs pure generation)
        
        Returns:
            List of InterviewQuestion objects ready for interview
        
        Example:
            questions = orchestrator.prepare_questions(
                job_description="Python Engineer with ML exp...",
                num_questions=5,
                use_rag=True
            )
        """
        print("\n" + "=" * 80)
        print("PREPARING INTERVIEW QUESTIONS")
        print("=" * 80)
        
        questions = []
        
        # Strategy 1: RAG Retrieval
        if use_rag:
            print(f"\n📚 Retrieving questions from knowledge base...")
            
            try:
                # Get questions by job description
                categorized_results = self.rag_retriever.retrieve_by_job_description(
                    job_description=job_description,
                    top_k=num_questions
                )
                
                # Convert to InterviewQuestion format
                for category, results in categorized_results.items():
                    for result in results:
                        q = result.question
                        interview_q = InterviewQuestion(
                            question=gen_q.question,
                            category=gen_q.category or "General",
                            difficulty=gen_q.difficulty or "Medium",
                            source="generated",
                            topics=gen_q.expected_topics or [],       
                            expected_answer="N/A",                     #  generator does NOT return expected answer
                            follow_ups=[gen_q.follow_up] if gen_q.follow_up else []    # correct field
                        )

                        questions.append(interview_q)
                
                print(f"  ✓ Retrieved {len(questions)} questions from RAG")
                
            except Exception as e:
                print(f"  ⚠ RAG retrieval failed: {e}")
                print(f"  → Falling back to generation")
        
        # Strategy 2: Generate additional questions if needed
        remaining = num_questions - len(questions)
        
        if remaining > 0:
            print(f"\n🤖 Generating {remaining} additional questions...")
            
            try:
                # Use question generator
                generated = self.question_generator.generate_questions(
                    job_description=job_description,
                    num_questions=remaining
                )
                
                # Convert to InterviewQuestion format
                for gen_q in generated.questions:
                    interview_q = InterviewQuestion(
                        question=gen_q.question,
                        category=gen_q.category or "General",
                        difficulty=gen_q.difficulty or "Medium",
                        source="generated",
                        topics=gen_q.expected_topics or [],
                        expected_answer=gen_q.expected_answer or "N/A",
                        follow_ups=gen_q.follow_up_questions or []
                    )
                    questions.append(interview_q)
                
                print(f"  ✓ Generated {remaining} questions")
                
            except Exception as e:
                print(f"  ⚠ Generation failed: {e}")
        
        # Limit to requested number
        questions = questions[:num_questions]
        
        print(f"\n✅ Prepared {len(questions)} total questions")
        print(f"   RAG: {sum(1 for q in questions if q.source == 'rag')}")
        print(f"   Generated: {sum(1 for q in questions if q.source == 'generated')}")
        
        return questions
    
    
    def conduct_interview(
        self,
        session: InterviewSession,
        simulate: bool = False
    ) -> InterviewSession:
        """
        Conduct the interview using Interviewer Agent
        
        Process:
            1. For each question:
               a. Present question to candidate
               b. Collect answer (or simulate)
               c. Store in session
            2. Return updated session
        
        Args:
            session: InterviewSession with prepared questions
            simulate: If True, generate simulated answers (for testing)
        
        Returns:
            Updated session with answers
        
        Note:
            In production, this would be interactive.
            For demo, we simulate or use pre-defined answers.
        """
        print("\n" + "=" * 80)
        print("CONDUCTING INTERVIEW")
        print("=" * 80)
        
        print(f"\nCandidate: {session.candidate_name}")
        print(f"Position: {session.job_title}")
        print(f"Questions: {len(session.questions)}")
        
        if simulate:
            print("\n⚠ SIMULATION MODE - Generating mock answers")
        
        # Conduct interview
        for i, question in enumerate(session.questions, 1):
            print(f"\n{'=' * 60}")
            print(f"Question {i}/{len(session.questions)}")
            print(f"{'=' * 60}")
            print(f"\nCategory: {question.category}")
            print(f"Difficulty: {question.difficulty}")
            print(f"\nQ: {question.question}")
            
            if simulate:
                # Generate simulated answer
                # In production, this would be interactive
                simulated_answers = {
                    "Easy": "This is a basic answer that covers the fundamental concept.",
                    "Medium": "This is a detailed answer that demonstrates understanding of the concept with some examples and edge cases.",
                    "Hard": "This is a comprehensive answer that shows deep understanding, discusses tradeoffs, provides real-world examples, and mentions advanced considerations."
                }
                answer = simulated_answers.get(question.difficulty, "I understand the concept.")
                print(f"\n[Simulated] A: {answer}")
            else:
                # Interactive mode (for production)
                answer = input("\nYour answer: ")
            
            # Store answer
            question.user_answer = answer
        
        print(f"\n✅ Interview completed - {len(session.questions)} questions answered")
        
        return session
    
    
    def evaluate_interview(
        self,
        session: InterviewSession
    ) -> InterviewSession:
        """
        Evaluate all answers using Evaluator Agent
        
        Process:
            1. For each answered question:
               a. Call evaluator agent
               b. Get scores and feedback
               c. Store evaluation
            2. Calculate overall metrics
            3. Return updated session
        
        Args:
            session: InterviewSession with answers
        
        Returns:
            Session with evaluations added
        """
        print("\n" + "=" * 80)
        print("EVALUATING ANSWERS")
        print("=" * 80)
        
        print(f"\nEvaluating {len(session.questions)} answers...")
        
        category_scores = {}
        all_scores = []
        
        for i, question in enumerate(session.questions, 1):
            if not question.user_answer:
                continue
            
            print(f"\n  {i}. Evaluating: {question.question[:50]}...")
            
            try:
                # Call evaluator
                evaluation = self.evaluator.evaluate_answer(
                    question=question.question,
                    candidate_answer=question.user_answer,
                    expected_answer=question.expected_answer,
                    difficulty=question.difficulty
                )
                
                # Store evaluation
                question.evaluation = {
                    "overall_score": evaluation.overall_score,
                    "technical_accuracy": evaluation.technical_accuracy,
                    "completeness": evaluation.completeness,
                    "communication": evaluation.communication,
                    "depth": evaluation.depth,
                    "strengths": evaluation.strengths,
                    "areas_for_improvement": evaluation.areas_for_improvement,
                    "recommendation": evaluation.recommendation
                }
                
                # Track scores by category
                if question.category not in category_scores:
                    category_scores[question.category] = []
                category_scores[question.category].append(evaluation.overall_score)
                all_scores.append(evaluation.overall_score)
                
                print(f"     Score: {evaluation.overall_score}/10")
                
            except Exception as e:
                print(f"     ⚠ Evaluation failed: {e}")
        
        # Calculate overall metrics
        if all_scores:
            session.overall_score = sum(all_scores) / len(all_scores)
            
            # Calculate category averages
            for category, scores in category_scores.items():
                session.category_scores[category] = sum(scores) / len(scores)
        
        print(f"\n✅ Evaluation completed")
        if session.overall_score is not None:
            print(f"   Overall Score: {session.overall_score:.1f}/10")
            print(f"   Category Scores:")
            for cat, score in session.category_scores.items():
                print(f"      {cat}: {score:.1f}/10")
        else:
            print(f"   Overall Score: No evaluations completed")
        
        return session
    
    
    def generate_report(
        self,
        session: InterviewSession
    ) -> str:
        """
        Generate comprehensive interview report
        
        Process:
            1. Aggregate all evaluations
            2. Calculate overall metrics
            3. Make hiring recommendation
            4. Format professional report
        
        Args:
            session: Completed interview session
        
        Returns:
            Formatted report string
        """
        print("\n" + "=" * 80)
        print("GENERATING INTERVIEW REPORT")
        print("=" * 80)
        
        # Determine recommendation based on score
        if session.overall_score is None:
            recommendation = "Unable to determine - evaluation incomplete"
        elif session.overall_score >= 8.0:
            recommendation = "Strong Hire - Excellent performance across all areas"
        elif session.overall_score >= 6.5:
            recommendation = "Hire - Solid performance with minor gaps"
        elif session.overall_score >= 5.0:
            recommendation = "Maybe - Mixed performance, additional evaluation recommended"
        else:
            recommendation = "No Hire - Significant gaps in required knowledge"
        
        session.recommendation = recommendation
        
        # Build detailed feedback
        strengths = []
        improvements = []
        
        for question in session.questions:
            if question.evaluation:
                if question.evaluation.get("strengths"):
                    strengths.extend(question.evaluation["strengths"])
                if question.evaluation.get("areas_for_improvement"):
                    improvements.extend(question.evaluation["areas_for_improvement"])
        
        # Create report
        score_display = f"{session.overall_score:.1f}/10" if session.overall_score is not None else "N/A"
        
        report = f"""
{'=' * 80}
INTERVIEW REPORT
{'=' * 80}

CANDIDATE INFORMATION
---------------------
Name: {session.candidate_name}
Position: {session.job_title}
Interview Date: {session.start_time}
Session ID: {session.session_id}

OVERALL PERFORMANCE
-------------------
Overall Score: {score_display}
Recommendation: {recommendation}

CATEGORY BREAKDOWN
------------------
"""
        
        for category, score in sorted(session.category_scores.items()):
            report += f"{category:.<30} {score:.1f}/10\n"
        
        report += f"""

QUESTION-BY-QUESTION ANALYSIS
------------------------------
"""
        
        for i, question in enumerate(session.questions, 1):
            report += f"""
Question {i}: {question.question}
Category: {question.category} | Difficulty: {question.difficulty} | Source: {question.source}
"""
            if question.evaluation:
                eval_data = question.evaluation
                report += f"""Score: {eval_data['overall_score']:.1f}/10
  Technical Accuracy: {eval_data['technical_accuracy']}/10
  Completeness: {eval_data['completeness']}/10
  Communication: {eval_data['communication']}/10
  Depth: {eval_data['depth']}/10

"""
                if eval_data.get('strengths'):
                    report += "  Strengths:\n"
                    for strength in eval_data['strengths'][:2]:
                        report += f"    • {strength}\n"
                
                if eval_data.get('areas_for_improvement'):
                    report += "  Areas for Improvement:\n"
                    for area in eval_data['areas_for_improvement'][:2]:
                        report += f"    • {area}\n"
        
        report += f"""

OVERALL ASSESSMENT
------------------
"""
        
        # Summarize strengths
        if strengths:
            unique_strengths = list(set(strengths))[:5]
            report += "\nKey Strengths:\n"
            for strength in unique_strengths:
                report += f"  ✓ {strength}\n"
        
        # Summarize improvements
        if improvements:
            unique_improvements = list(set(improvements))[:5]
            report += "\nAreas for Development:\n"
            for improvement in unique_improvements:
                report += f"  → {improvement}\n"
        
        report += f"""

FINAL RECOMMENDATION
--------------------
{recommendation}

{'=' * 80}
END OF REPORT
{'=' * 80}
"""
        
        session.detailed_feedback = report
        session.end_time = datetime.now().isoformat()
        
        print("✅ Report generated successfully")
        
        return report
    
    
    def run_complete_interview(
        self,
        candidate_name: str,
        job_description: str,
        job_title: str = "Software Engineer",
        num_questions: int = 5,
        simulate: bool = True,
        use_rag: bool = True
    ) -> InterviewSession:
        """
        Run complete end-to-end interview
        
        THIS IS THE MAIN ENTRY POINT!
        
        Process:
            1. Create session
            2. Prepare questions (RAG + generation)
            3. Conduct interview
            4. Evaluate answers
            5. Generate report
        
        Args:
            candidate_name: Name of candidate
            job_description: Full job posting
            job_title: Position title
            num_questions: Number of questions to ask
            simulate: If True, generate mock answers
            use_rag: If True, use RAG retrieval
        
        Returns:
            Complete InterviewSession with all data
        
        Example:
            session = orchestrator.run_complete_interview(
                candidate_name="Alice Smith",
                job_description="Senior Python Engineer with ML...",
                num_questions=5,
                simulate=True
            )
            
            print(session.detailed_feedback)
        """
        print("\n" + "=" * 80)
        print("STARTING COMPLETE INTERVIEW PIPELINE")
        print("=" * 80)
        
        # Create session
        session = InterviewSession(
            session_id=f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            candidate_name=candidate_name,
            job_title=job_title,
            job_description=job_description,
            start_time=datetime.now().isoformat()
        )
        
        print(f"\n📋 Session created: {session.session_id}")
        print(f"   Candidate: {candidate_name}")
        print(f"   Position: {job_title}")
        
        try:
            # Step 1: Prepare questions
            questions = self.prepare_questions(
                job_description=job_description,
                num_questions=num_questions,
                use_rag=use_rag
            )
            session.questions = questions
            
            # Step 2: Conduct interview
            session = self.conduct_interview(
                session=session,
                simulate=simulate
            )
            
            # Step 3: Evaluate answers
            session = self.evaluate_interview(session)
            
            # Step 4: Generate report
            report = self.generate_report(session)
            
            print("\n" + "=" * 80)
            print("✅ INTERVIEW PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 80)
            
            print(f"\n📊 Final Results:")
            if session.overall_score is not None:
                print(f"   Overall Score: {session.overall_score:.1f}/10")
            else:
                print(f"   Overall Score: N/A")
            print(f"   Recommendation: {session.recommendation}")
            print(f"   Report Length: {len(report)} characters")
            
            return session
            
        except Exception as e:
            print(f"\n❌ Error in interview pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    
    def save_session(
        self,
        session: InterviewSession,
        output_dir: str = "backend/data/interviews"
    ):
        """
        Save interview session to disk
        
        Why we need this:
            - Persist interview data
            - Enable analysis later
            - Create audit trail
            - Build datasets
        
        Args:
            session: Completed interview session
            output_dir: Where to save
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_path / f"{session.session_id}.json"
        with open(json_path, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)
        
        # Save report as text
        txt_path = output_path / f"{session.session_id}_report.txt"
        with open(txt_path, 'w') as f:
            f.write(session.detailed_feedback)
        
        print(f"\n💾 Session saved:")
        print(f"   JSON: {json_path}")
        print(f"   Report: {txt_path}")


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    """
    Test the complete interview orchestrator
    
    This demonstrates the FULL SYSTEM working together:
        - RAG retrieval
        - Question generation
        - Interview conduction
        - Answer evaluation
        - Report generation
    """
    
    print("=" * 80)
    print("TESTING: COMPLETE INTERVIEW ORCHESTRATOR")
    print("=" * 80)
    
    # Initialize orchestrator
    orchestrator = InterviewOrchestrator()
    
    # Sample job description
    job_description = """
    Senior GenAI Engineer

    We're seeking an experienced engineer to build production LLM applications.
    
    Requirements:
    - 5+ years Python experience
    - Expert in LangChain and LangGraph
    - Experience building RAG systems
    - Understanding of vector databases
    - Prompt engineering skills
    - Built and deployed AI agents
    
    Responsibilities:
    - Design and implement GenAI solutions
    - Build RAG pipelines for knowledge retrieval
    - Develop multi-agent systems
    - Optimize LLM performance
    - Mentor junior engineers
    """
    
    # Run complete interview
    print("\n" + "=" * 80)
    print("RUNNING COMPLETE INTERVIEW")
    print("=" * 80)
    
    session = orchestrator.run_complete_interview(
        candidate_name="Alex Chen",
        job_description=job_description,
        job_title="Senior GenAI Engineer",
        num_questions=5,
        simulate=True,  # Generate mock answers
        use_rag=True    # Use RAG retrieval
    )
    
    # Display report
    print("\n" + "=" * 80)
    print("INTERVIEW REPORT")
    print("=" * 80)
    print(session.detailed_feedback)
    
    # Save session
    orchestrator.save_session(session)
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)
    
    print("\n🎉 CONGRATULATIONS!")
    print("=" * 80)
    print("You've built a complete, production-ready GenAI interview system!")
    print("\nSystem Components:")
    print("  ✅ RAG-powered question retrieval")
    print("  ✅ GPT-4 question generation")
    print("  ✅ AI interviewer agent")
    print("  ✅ AI evaluator agent")
    print("  ✅ Comprehensive reporting")
    print("  ✅ End-to-end orchestration")
    print("\nThis demonstrates:")
    print("  • LangChain expertise")
    print("  • RAG implementation")
    print("  • Vector database usage")
    print("  • Multi-agent coordination")
    print("  • Production system design")
    print("  • LLM orchestration")
    print("\n🚀 Ready for your portfolio!")
    print("=" * 80)