import pytest

from src.config import Settings
from src.data_validation import load_dataset


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture
def settings():
    return Settings(top_k=3, min_score=0.18, max_agent_steps=8, max_tool_calls=6)
