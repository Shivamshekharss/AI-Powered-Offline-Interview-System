"""
Setup Script for InterviewAce AI

Automates the setup process for new users.

Usage:
    python setup.py
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"⏳ {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}")
        print(f"   {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.9+"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"❌ Python 3.9+ required. You have {version.major}.{version.minor}.{version.micro}")
        return False


def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path(".env")
    example_path = Path(".env.example")
    
    if env_path.exists():
        print("ℹ️  .env file already exists")
        overwrite = input("   Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("   Keeping existing .env file")
            return True
    
    if example_path.exists():
        print("📝 Creating .env file from template...")
        try:
            with open(example_path, 'r') as src, open(env_path, 'w') as dst:
                dst.write(src.read())
            print("✅ .env file created")
            print("\n⚠️  IMPORTANT: Edit .env file and add your OpenAI API key!")
            return True
        except Exception as e:
            print(f"❌ Error creating .env: {e}")
            return False
    else:
        print("❌ .env.example not found")
        return False


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "backend/data/vector_store",
        "backend/data/interviews"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return True


def main():
    """Main setup process"""
    print_header("🎯 InterviewAce AI - Setup")
    
    print("""
Welcome to InterviewAce AI setup!

This script will:
  1. Check Python version
  2. Install dependencies
  3. Create configuration files
  4. Initialize the question bank
  5. Verify installation

Let's get started!
""")
    
    input("Press Enter to continue...")
    
    # Step 1: Check Python
    print_header("Step 1: Python Version Check")
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        print("   Please install Python 3.9 or higher")
        return
    
    # Step 2: Install dependencies
    print_header("Step 2: Install Dependencies")
    
    install = input("Install dependencies from requirements.txt? (Y/n): ").strip().lower()
    if install != 'n':
        if not run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing dependencies"
        ):
            print("\n⚠️  Installation had errors, but continuing...")
    
    # Step 3: Create directories
    print_header("Step 3: Create Directories")
    if not create_directories():
        print("\n❌ Setup failed: Could not create directories")
        return
    
    # Step 4: Create .env file
    print_header("Step 4: Create Configuration")
    if not create_env_file():
        print("\n⚠️  Warning: .env file not created")
        print("   You'll need to create it manually")
    
    # Step 5: Initialize question bank
    print_header("Step 5: Initialize Question Bank")
    
    initialize = input("Initialize question bank with 40+ curated questions? (Y/n): ").strip().lower()
    if initialize != 'n':
        print("\n⏳ Initializing question bank (this may take 1-2 minutes)...")
        if run_command(
            f"{sys.executable} backend/scripts/question_bank_initializer.py",
            "Question bank initialization"
        ):
            print("✅ Question bank ready!")
        else:
            print("\n⚠️  Warning: Question bank initialization failed")
            print("   You can run it manually later:")
            print(f"   {sys.executable} backend/scripts/question_bank_initializer.py")
    
    # Step 6: Final checks
    print_header("Step 6: Verification")
    
    # Check if .env has API key
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            content = f.read()
            if "your_openai_api_key_here" in content:
                print("⚠️  Warning: OpenAI API key not set in .env file")
                print("   Please edit .env and add your API key")
            else:
                print("✅ .env file configured")
    
    # Check if question bank exists
    vector_store_path = Path("backend/data/vector_store/questions.index")
    if vector_store_path.exists():
        print("✅ Question bank initialized")
    else:
        print("⚠️  Question bank not found")
    
    # Complete
    print_header("🎉 Setup Complete!")
    
    print("""
InterviewAce AI is ready to use!

Next Steps:

1. Edit .env file and add your OpenAI API key:
   OPENAI_API_KEY=your_actual_key_here

2. Run the application:
   
   Interactive CLI:
   python cli_interview.py
   
   Web UI:
   streamlit run app.py
   
   Test the system:
   python backend/orchestrator/interview_orchestrator.py

3. Read the README.md for detailed usage instructions

Happy interviewing! 🚀
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()