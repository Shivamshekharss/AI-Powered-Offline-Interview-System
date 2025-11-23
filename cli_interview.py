"""
Interactive CLI Interview Application - InterviewAce AI

A command-line interface for conducting real AI-powered technical interviews.
Users can take interviews, get evaluated, and receive detailed reports.

Usage:
    python cli_interview.py
    
    Then follow the prompts to:
    - Enter your name
    - Paste job description
    - Answer interview questions
    - Receive evaluation and report
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.orchestrator.interview_orchestrator import InterviewOrchestrator
from datetime import datetime
import os


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def print_section(text):
    """Print formatted section"""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80 + "\n")


def get_multiline_input(prompt):
    """Get multi-line input from user"""
    print(prompt)
    print("(Press Enter twice when done, or type 'DONE' on a new line)")
    print("-" * 80)
    
    lines = []
    empty_count = 0
    
    while True:
        line = input()
        
        if line.strip().upper() == 'DONE':
            break
            
        if line.strip() == '':
            empty_count += 1
            if empty_count >= 2:
                break
        else:
            empty_count = 0
            
        lines.append(line)
    
    return '\n'.join(lines).strip()


def conduct_interactive_interview():
    """Main interactive interview flow"""
    
    clear_screen()
    
    # Welcome
    print_header("🎯 INTERVIEW ACE AI - Interactive Technical Interview")
    
    print("""
Welcome to InterviewAce AI!

This AI-powered system will:
  ✓ Analyze the job description
  ✓ Retrieve or generate relevant technical questions
  ✓ Conduct a comprehensive interview
  ✓ Evaluate your answers with detailed feedback
  ✓ Provide a complete interview report
  
Let's get started!
""")
    
    input("Press Enter to continue...")
    clear_screen()
    
    # Step 1: Get candidate information
    print_header("📋 CANDIDATE INFORMATION")
    
    candidate_name = input("Enter your name: ").strip()
    
    while not candidate_name:
        print("❌ Name cannot be empty")
        candidate_name = input("Enter your name: ").strip()
    
    print(f"\nHello, {candidate_name}! 👋\n")
    
    # Step 2: Get job information
    print_section("📄 JOB DESCRIPTION")
    
    print("Please paste the job description you're preparing for:")
    job_description = get_multiline_input("")
    
    while not job_description or len(job_description) < 50:
        print("❌ Job description seems too short. Please provide more details.")
        job_description = get_multiline_input("")
    
    print("\n✅ Job description received!")
    
    job_title = input("\nJob Title (e.g., 'Senior Python Engineer'): ").strip()
    if not job_title:
        job_title = "Software Engineer"
    
    # Step 3: Interview settings
    print_section("⚙️ INTERVIEW SETTINGS")
    
    num_questions_input = input("How many questions would you like? (default: 5): ").strip()
    num_questions = 5
    
    if num_questions_input.isdigit():
        num_questions = max(1, min(int(num_questions_input), 10))
    
    use_rag_input = input("Use RAG (retrieve from knowledge base)? (Y/n): ").strip().lower()
    use_rag = use_rag_input != 'n'
    
    # Confirm
    clear_screen()
    print_header("📊 INTERVIEW CONFIGURATION")
    
    print(f"""
Candidate: {candidate_name}
Position: {job_title}
Questions: {num_questions}
RAG Enabled: {'Yes' if use_rag else 'No'}

Job Description Preview:
{job_description[:200]}{'...' if len(job_description) > 200 else ''}
""")
    
    confirm = input("\nProceed with interview? (Y/n): ").strip().lower()
    
    if confirm == 'n':
        print("\n❌ Interview cancelled.")
        return
    
    # Step 4: Initialize system
    clear_screen()
    print_header("🚀 INITIALIZING INTERVIEW SYSTEM")
    
    print("Loading AI agents and knowledge base...")
    print("(This may take a few moments on first run)\n")
    
    try:
        orchestrator = InterviewOrchestrator()
    except Exception as e:
        print(f"\n❌ Error initializing system: {e}")
        print("\nPlease check:")
        print("  1. OpenAI API key is set in .env file")
        print("  2. All dependencies are installed")
        print("  3. Vector store is initialized")
        return
    
    # Step 5: Prepare questions
    print_section("📚 PREPARING QUESTIONS")
    
    print("Analyzing job description and selecting questions...")
    
    try:
        questions = orchestrator.prepare_questions(
            job_description=job_description,
            num_questions=num_questions,
            use_rag=use_rag
        )
    except Exception as e:
        print(f"\n❌ Error preparing questions: {e}")
        return
    
    print(f"\n✅ {len(questions)} questions prepared!")
    input("\nPress Enter to start the interview...")
    
    # Step 6: Conduct interview
    clear_screen()
    print_header("🎤 INTERVIEW IN PROGRESS")
    
    print(f"""
Instructions:
  • Read each question carefully
  • Take your time to formulate thoughtful answers
  • Be as detailed or concise as you feel appropriate
  • Type your answer and press Enter twice when done
  
Total Questions: {len(questions)}

Let's begin!
""")
    
    input("Press Enter to see the first question...")
    
    # Ask questions
    for i, question in enumerate(questions, 1):
        clear_screen()
        
        print_header(f"QUESTION {i} of {len(questions)}")
        
        print(f"Category: {question.category}")
        print(f"Difficulty: {question.difficulty}")
        print(f"Source: {'📚 Knowledge Base (RAG)' if question.source == 'rag' else '🤖 AI Generated'}")
        
        print(f"\n{'=' * 80}")
        print(f"\n{question.question}\n")
        print("=" * 80)
        
        # Get answer
        answer = get_multiline_input("\nYour answer:")
        
        while not answer or len(answer) < 10:
            print("\n❌ Answer seems too short. Please provide more detail.")
            answer = get_multiline_input("\nYour answer:")
        
        question.user_answer = answer
        
        print(f"\n✅ Answer recorded ({len(answer)} characters)")
        
        if i < len(questions):
            input("\nPress Enter for next question...")
    
    # Step 7: Evaluate
    clear_screen()
    print_header("🔍 EVALUATING ANSWERS")
    
    print("AI is analyzing your answers...")
    print("This may take 30-60 seconds...\n")
    
    from backend.orchestrator.interview_orchestrator import InterviewSession
    
    session = InterviewSession(
        session_id=f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        candidate_name=candidate_name,
        job_title=job_title,
        job_description=job_description,
        start_time=datetime.now().isoformat(),
        questions=questions
    )
    
    try:
        session = orchestrator.evaluate_interview(session)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 8: Generate report
    print("\n📝 Generating comprehensive report...")
    
    try:
        report = orchestrator.generate_report(session)
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        return
    
    # Step 9: Display results
    clear_screen()
    print(report)
    
    # Step 10: Save option
    print("\n" + "=" * 80)
    save_option = input("Save interview report to file? (Y/n): ").strip().lower()
    
    if save_option != 'n':
        try:
            orchestrator.save_session(session)
            print(f"\n✅ Interview saved successfully!")
            print(f"   Session ID: {session.session_id}")
            print(f"   Location: backend/data/interviews/")
        except Exception as e:
            print(f"\n⚠️  Error saving: {e}")
    
    # Final message
    print("\n" + "=" * 80)
    print("INTERVIEW COMPLETE".center(80))
    print("=" * 80)
    
    score_display = f"{session.overall_score:.1f}/10" if session.overall_score is not None else "N/A"
    
    print(f"""
Thank you for using InterviewAce AI, {candidate_name}!

Your Results:
  • Overall Score: {score_display}
  • Recommendation: {session.recommendation}
  • Questions Answered: {len(questions)}

Next Steps:
  ✓ Review the detailed feedback above
  ✓ Focus on areas for improvement
  ✓ Practice and come back for another interview!

Good luck with your job search! 🚀
""")


def main():
    """Main entry point"""
    try:
        conduct_interactive_interview()
    except KeyboardInterrupt:
        print("\n\n❌ Interview cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()