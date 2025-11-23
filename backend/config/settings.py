import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = BASE_DIR / ".env"

# Load environment variables (override=True ensures replacement)
load_dotenv(dotenv_path=ENV_FILE, override=True)

class Settings(BaseModel):
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL"))
    groq_temperature: float = Field(default_factory=lambda: float(os.getenv("GROQ_TEMPERATURE", "0.5")))
    groq_api_url: str = Field(default_factory=lambda: os.getenv("GROQ_API_URL"))
    
     # HuggingFace Embeddings
    hf_api_key: str = Field(default_factory=lambda: os.getenv("HF_API_KEY"))
    hf_embedding_model: str = Field(
        default_factory=lambda: os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    @validator("groq_api_url")
    def validate_api_url(cls, v):
        if not v:
            raise ValueError("❌ GROQ_API_URL missing in .env")
        return v

def setup_langsmith():
    """
    Enabling Langsmith Tracing

    What it does:
    - Tracks every LLM call we make
    - Shows us prompts, response, timing, cost
    - super helpful for debugging !

    USAGE:
        setup_langsmith()  # Call this once at the app startup
    """ 

    settings = get_settings()

    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        # Set environment variable so LangChain knows to use LangSmith
        os.environ["LANGCHAIN_TRACING_V2"]= "true"
        os.environ["LANGCHAIN_API_KEY"] = str(settings.langchain_api_key)
        os.environ["LANGCHAIN_PROJECT"] = str(settings.langchain_project)

        print("LangSmith monitoring enabled!")
        print(f"View traces at: https://smith.langchain.com/")
        
    else:
        print("LangSmith disabled (no API key found)")
        print("Get free key at : https://smith.langchain.com/")


_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
