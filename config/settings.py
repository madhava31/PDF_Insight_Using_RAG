"""Configuration Settings Module

This module loads and exposes environment variables for:
- OpenAI/LLM API credentials and endpoints
- Model selection
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Load API key from environment (tries multiple common variable names)
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("NVIDIA_API_KEY")
    or os.getenv("API_KEY")
)

# API endpoint base URL (default: NVIDIA's endpoint)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")

# LLM model name (default: NVIDIA Gemma 2 2b)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
