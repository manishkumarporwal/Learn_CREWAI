from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class StoryCrew:
    """Story Writing Crew — Planner → Writer → Critic/Editor"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    # ── Agents ────────────────────────────────────────────────────────────────
    @agent
    def story_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["story_planner"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def story_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["story_writer"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def story_critic(self) -> Agent:
        return Agent(
            config=self.agents_config["story_critic"],  # type: ignore[index]
            verbose=True,
        )

    # ── Tasks ─────────────────────────────────────────────────────────────────
    @task
    def plan_story(self) -> Task:
        return Task(
            config=self.tasks_config["plan_story"],  # type: ignore[index]
        )

    @task
    def write_story(self) -> Task:
        return Task(
            config=self.tasks_config["write_story"],  # type: ignore[index]
        )

    @task
    def critique_and_polish_story(self) -> Task:
        return Task(
            config=self.tasks_config["critique_and_polish_story"],  # type: ignore[index]
        )

    # ── Crew ──────────────────────────────────────────────────────────────────
    @crew
    def crew(self) -> Crew:
        """Creates the Story Writing Crew"""
        return Crew(
            agents=self.agents,   # auto-populated by @agent decorators
            tasks=self.tasks,     # auto-populated by @task  decorators
            process=Process.sequential,
            verbose=True,
        )
