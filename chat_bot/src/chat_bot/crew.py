import os
from pathlib import Path
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from typing import List

# ── Store ChromaDB inside the project so it's visible in VS Code ──────────────
_PROJECT_ROOT = Path(__file__).parents[2]  # → chat_bot/
_CHROMA_PATH  = _PROJECT_ROOT / "chroma_db"
_CHROMA_PATH.mkdir(exist_ok=True)
os.environ["CREWAI_STORAGE_DIR"] = str(_CHROMA_PATH)


@CrewBase
class ChatBot():
    """ChatBot crew with PDF + ChromaDB knowledge retrieval"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/agents.yaml'
    tasks_config  = 'config/tasks.yaml'

    # ── Knowledge Sources ─────────────────────────────────────────────────────

    # 1. User profile stored as a string
    user_preference = StringKnowledgeSource(
        content=(
            "User name is Manish Porwal. He is 39 years old and lives in Bengaluru. "
            "He is a software engineer and has 15 years of experience in the IT industry."
        ),
        metadata={"source": "user_profile"},
    )

    # 2. PDF document — path is relative to the project root (where knowledge/ lives)
    #    CrewAI looks for PDFs inside the knowledge/ directory by default
    pdf_source = PDFKnowledgeSource(
        file_paths=["story_book.pdf"],      # file inside knowledge/
        metadata={"source": "story_book"},
    )

    # ── ChromaDB embedder config ──────────────────────────────────────────────
    # CrewAI uses ChromaDB internally. Setting 'storage_path' makes it
    # PERSISTENT on disk so vectors survive between runs (no re-embedding).
    # provider="openai" uses text-embedding-3-small by default (free tier friendly).
    CHROMA_EMBEDDER = {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            # api_key is picked up automatically from OPENAI_API_KEY env var
        },
    }

    # ── Agent ─────────────────────────────────────────────────────────────────
    @agent
    def helpful_assistant(self) -> Agent:
        return Agent(
            config=self.agents_config['helpful_assistant'],  # type: ignore[index]
            verbose=True,
        )

    # ── Task ──────────────────────────────────────────────────────────────────
    @task
    def helpful_assistant_task(self) -> Task:
        return Task(
            config=self.tasks_config['helpful_assistant_task'],  # type: ignore[index]
        )

    # ── Crew ──────────────────────────────────────────────────────────────────
    @crew
    def crew(self) -> Crew:
        """Creates the ChatBot crew with PDF + ChromaDB knowledge"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            tracing=True,   # ← enables detailed execution traces
            # Both sources are indexed in ChromaDB on first run,
            # then retrieved semantically on every subsequent query.
            knowledge_sources=[self.user_preference, self.pdf_source],
            embedder=self.CHROMA_EMBEDDER,
        )
