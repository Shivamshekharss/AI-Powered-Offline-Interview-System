"""
Streamlit Web UI - InterviewAce AI

A beautiful, interactive web interface for AI-powered technical interviews.

Usage:
    streamlit run app.py
    
Features:
    - Interactive job description input
    - Real-time interview questions
    - Live answer collection
    - Detailed evaluation reports
    - Download results as PDF
"""

import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.orchestrator.interview_orchestrator import InterviewOrchestrator, InterviewSession
from datetime import datetime
import json


# Page configuration
st.set_page_config(
    page_title="AcePrep AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .question-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .score-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'session' not in st.session_state:
        st.session_state.session = None
    if 'interview_complete' not in st.session_state:
        st.session_state.interview_complete = False
    if 'stage' not in st.session_state:
        st.session_state.stage = 'setup'  # setup, interview, results


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 AcePrep AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Technical Interview Assistant</p>', unsafe_allow_html=True)
# ...existing code...)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        
        if st.session_state.stage == 'setup':
            st.info("📝 Step 1: Setup Interview")
        elif st.session_state.stage == 'interview':
            st.success(f"🎤 Step 2: Answering Question {st.session_state.current_question + 1}/{len(st.session_state.questions)}")
        elif st.session_state.stage == 'results':
            st.success("✅ Step 3: View Results")
        
        st.markdown("---")
       # ...existing code...
        st.markdown("""
        ### About
        
        InterviewAce AI helps you practice technical interviews with:
        - 📚 RAG-powered question retrieval
        - 🤖 AI-generated custom questions
        - 🔍 Detailed answer evaluation
        - 📊 Comprehensive feedback
        
         ### Tech Stack
        - Groq Llama 3.3-70B
        - LangChain (RAG + Evaluation)
        - HuggingFace SentenceTransformers
        - FAISS Vector Store
        - Streamlit UI
        - Python 3.10
        """)

        if st.button("🔄 Start New Interview"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content based on stage
    if st.session_state.stage == 'setup':
        show_setup_page()
    elif st.session_state.stage == 'interview':
        show_interview_page()
    elif st.session_state.stage == 'results':
        show_results_page()


def show_setup_page():
    """Show interview setup page"""
    st.header("📝 Interview Setup")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Candidate Information")
        candidate_name = st.text_input("Your Name", placeholder="e.g., John Doe")
        
        job_title = st.text_input("Job Title", placeholder="e.g., Senior Python Engineer")
        
        job_description = st.text_area(
            "Job Description",
            placeholder="Paste the complete job description here...",
            height=250,
            help="Provide the full job posting including requirements and responsibilities"
        )
    
    with col2:
        st.subheader("Interview Settings")
        
        num_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=10,
            value=5,
            help="How many questions would you like to answer?"
        )
        
        use_rag = st.checkbox(
            "Use RAG Retrieval",
            value=True,
            help="Retrieve questions from the knowledge base (40+ curated questions)"
        )
        
        st.info(f"""
        **Interview Duration**
        
        Approximately {num_questions * 3}-{num_questions * 5} minutes
        
        **Question Sources:**
        - {'✅' if use_rag else '❌'} Knowledge Base (RAG)
        - ✅ AI Generation
        """)
    
    # Validation and start button
    st.markdown("---")
    
    if not candidate_name:
        st.warning("⚠️ Please enter your name")
    elif not job_description or len(job_description) < 50:
        st.warning("⚠️ Please provide a detailed job description (at least 50 characters)")
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Start Interview", use_container_width=True):
                with st.spinner("Initializing AI interview system..."):
                    try:
                        # Initialize orchestrator
                        st.session_state.orchestrator = InterviewOrchestrator()
                        
                        # Prepare questions
                        questions = st.session_state.orchestrator.prepare_questions(
                            job_description=job_description,
                            num_questions=num_questions,
                            use_rag=use_rag
                        )
                        
                        # Create session
                        st.session_state.session = InterviewSession(
                            session_id=f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            candidate_name=candidate_name,
                            job_title=job_title or "Software Engineer",
                            job_description=job_description,
                            start_time=datetime.now().isoformat(),
                            questions=questions
                        )
                        
                        st.session_state.questions = questions
                        st.session_state.current_question = 0
                        st.session_state.answers = []
                        st.session_state.stage = 'interview'
                        
                        st.success("✅ Interview ready! Starting now...")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error initializing interview: {e}")
                        st.error("Please check your .env file and ensure all dependencies are installed.")


def show_interview_page():
    """Show interview question and answer page"""
    current_idx = st.session_state.current_question
    if "questions" not in st.session_state or len(st.session_state.questions) == 0:
        st.error("❌ No questions generated yet. Please start the interview again.")
        return

    if current_idx >= len(st.session_state.questions):
        st.error("❌ Invalid question index. Restarting interview.")
        st.session_state.current_question = 0
        return

    question = st.session_state.questions[current_idx]

    
    # Progress
    progress = (current_idx) / len(st.session_state.questions)
    st.progress(progress, text=f"Question {current_idx + 1} of {len(st.session_state.questions)}")
    
    st.markdown("---")
    
    # Question display
    st.markdown(f"""
    <div class="question-box">
        <h3>Question {current_idx + 1}</h3>
        <p><strong>Category:</strong> {question.category} | <strong>Difficulty:</strong> {question.difficulty}</p>
        <p><strong>Source:</strong> {'📚 Knowledge Base (RAG)' if question.source == 'rag' else '🤖 AI Generated'}</p>
        <hr>
        <h4>{question.question}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Answer input
    st.subheader("Your Answer")
    
    answer_key = f"answer_{current_idx}"
    answer = st.text_area(
        "Type your answer here",
        height=250,
        key=answer_key,
        placeholder="Take your time to provide a thoughtful, detailed answer...",
        help="Aim for 3-5 sentences or more. Be specific and provide examples where relevant."
    )
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_idx > 0:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.current_question -= 1
                st.rerun()
    
    with col3:
        if not answer or len(answer) < 20:
            st.warning("⚠️ Answer too short (min 20 characters)")
        else:
            if current_idx < len(st.session_state.questions) - 1:
                if st.button("Next ➡️", use_container_width=True):
                    # Save answer
                    st.session_state.questions[current_idx].user_answer = answer
                    st.session_state.current_question += 1
                    st.rerun()
            else:
                if st.button("✅ Submit Interview", use_container_width=True):
                    # Save last answer
                    st.session_state.questions[current_idx].user_answer = answer
                    
                    # Evaluate
                    with st.spinner("🔍 AI is evaluating your answers... This may take 30-60 seconds..."):
                        try:
                            session = st.session_state.session
                            orchestrator = st.session_state.orchestrator
                            
                            # Evaluate
                            session = orchestrator.evaluate_interview(session)
                            
                            # Generate report
                            report = orchestrator.generate_report(session)
                            
                            st.session_state.session = session
                            st.session_state.stage = 'results'
                            
                            st.success("✅ Evaluation complete!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error during evaluation: {e}")


def show_results_page():
    """Show interview results and report"""
    session = st.session_state.session
    
    st.header("📊 Interview Results")
    
    # Overall score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="score-box">
            <h3>Overall Score</h3>
            <h1 style="color: #4caf50;">{session.overall_score:.1f}/10</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="score-box">
            <h3>Questions</h3>
            <h1 style="color: #2196f3;">{len(session.questions)}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        recommendation_color = "#4caf50" if "Hire" in session.recommendation else "#ff9800"
        st.markdown(f"""
        <div class="score-box">
            <h3>Recommendation</h3>
            <p style="color: {recommendation_color}; font-size: 1.2rem; font-weight: bold;">{session.recommendation.split('-')[0].strip()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Category scores
    st.subheader("📈 Category Breakdown")
    
    if session.category_scores:
        cols = st.columns(len(session.category_scores))
        for idx, (category, score) in enumerate(session.category_scores.items()):
            with cols[idx]:
                st.metric(category, f"{score:.1f}/10")
    
    st.markdown("---")
    
    # Detailed feedback
    st.subheader("📝 Detailed Feedback")
    
    tabs = st.tabs([f"Q{i+1}" for i in range(len(session.questions))] + ["Full Report"])
    
    # Question tabs
    for i, question in enumerate(session.questions):
        with tabs[i]:
            st.markdown(f"**Question:** {question.question}")
            st.markdown(f"**Category:** {question.category} | **Difficulty:** {question.difficulty}")
            
            if question.evaluation:
                eval_data = question.evaluation
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Your Answer:**")
                    st.info(question.user_answer)
                
                with col2:
                    st.markdown("**Score Breakdown:**")
                    st.write(f"Overall: {eval_data['overall_score']}/10")
                    st.write(f"Technical: {eval_data['technical_accuracy']}/10")
                    st.write(f"Completeness: {eval_data['completeness']}/10")
                    st.write(f"Communication: {eval_data['communication']}/10")
                    st.write(f"Depth: {eval_data['depth']}/10")
                
                if eval_data.get('strengths'):
                    st.markdown("**✅ Strengths:**")
                    for strength in eval_data['strengths']:
                        st.write(f"- {strength}")
                
                if eval_data.get('areas_for_improvement'):
                    st.markdown("**📈 Areas for Improvement:**")
                    for area in eval_data['areas_for_improvement']:
                        st.write(f"- {area}")
    
    # Full report tab
    with tabs[-1]:
        st.text_area("Complete Report", session.detailed_feedback, height=600)
    
    # Download options
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download JSON
        json_data = json.dumps(session.to_dict(), indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"{session.session_id}.json",
            mime="application/json"
        )
    
    with col2:
        # Download Report
        st.download_button(
            label="📄 Download Report",
            data=session.detailed_feedback,
            file_name=f"{session.session_id}_report.txt",
            mime="text/plain"
        )
    
    with col3:
        # Save to server
        if st.button("💾 Save to Server"):
            try:
                st.session_state.orchestrator.save_session(session)
                st.success(f"✅ Saved to backend/data/interviews/")
            except Exception as e:
                st.error(f"❌ Error saving: {e}")


if __name__ == "__main__":
    main() 