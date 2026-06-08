import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("NVIDIA_API_KEY")
    or os.getenv("API_KEY")
)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
