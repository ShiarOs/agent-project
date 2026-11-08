import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_PATH = ROOT / "reports" / "evaluation_report.md"


@dataclass(frozen=True)
class Settings:
    top_k: int = 3
    min_score: float = 0.18
    max_agent_steps: int = 8
    max_tool_calls: int = 6


def load_settings() -> Settings:
    # Exported environment variables win over values in .env.
    load_dotenv(ROOT / ".env")
    return Settings(
        top_k=int(os.getenv("TOP_K", "3")),
        min_score=float(os.getenv("MIN_SCORE", "0.18")),
        max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "8")),
        max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "6")),
    )
