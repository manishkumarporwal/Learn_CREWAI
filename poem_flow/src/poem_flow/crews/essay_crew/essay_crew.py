from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class EssayCrew:
    """Essay Writing Crew — Researcher → Writer → Editor"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    # ── Agents ────────────────────────────────────────────────────────────────
    @agent
    def essay_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["essay_researcher"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def essay_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["essay_writer"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def essay_editor(self) -> Agent:
        return Agent(
            config=self.agents_config["essay_editor"],  # type: ignore[index]
            verbose=True,
        )

    # ── Tasks ─────────────────────────────────────────────────────────────────
    @task
    def research_essay_topic(self) -> Task:
        return Task(
            config=self.tasks_config["research_essay_topic"],  # type: ignore[index]
        )

    @task
    def write_essay(self) -> Task:
        return Task(
            config=self.tasks_config["write_essay"],  # type: ignore[index]
        )

    @task
    def edit_essay(self) -> Task:
        return Task(
            config=self.tasks_config["edit_essay"],  # type: ignore[index]
        )

    # ── Crew ──────────────────────────────────────────────────────────────────
    @crew
    def crew(self) -> Crew:
        """Creates the Essay Writing Crew"""
        return Crew(
            agents=self.agents,   # auto-populated by @agent decorators
            tasks=self.tasks,     # auto-populated by @task  decorators
            process=Process.sequential,
            verbose=True,
        )
