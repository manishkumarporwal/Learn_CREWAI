"""
RLHF-Enhanced CrewAI - Reinforcement Learning from Human Feedback
=================================================================
This module wraps your MarketingCrew with a human-in-the-loop feedback system.

How it works:
  1. Crew runs and produces output.
  2. Human rates the output (1-5) and provides optional textual corrections.
  3. Feedback is stored in a JSON log.
  4. On the next run, past feedback is injected into agent backstories /
     task descriptions as "lessons learned", steering agents toward better output.
  5. Optionally: if score < threshold, the crew re-runs automatically with
     richer context (like a "negative reward" signal).
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ── env ──────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[3] / ".env")

from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool

# ── Paths ─────────────────────────────────────────────────────────────────────
FEEDBACK_LOG = Path(__file__).parent / "rlhf_feedback.json"
MAX_RETRIES   = 3          # max automatic re-runs if score < threshold
SCORE_THRESHOLD = 3        # scores below this trigger an automatic retry


# ── Feedback store ────────────────────────────────────────────────────────────
def load_feedback() -> list[dict]:
    """Load all past feedback entries."""
    if FEEDBACK_LOG.exists():
        with open(FEEDBACK_LOG, "r") as f:
            return json.load(f)
    return []


def save_feedback(entry: dict) -> None:
    """Append a new feedback entry and persist."""
    history = load_feedback()
    history.append(entry)
    with open(FEEDBACK_LOG, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✅  Feedback saved to {FEEDBACK_LOG}")


def summarise_feedback(topic: str, last_n: int = 5) -> str:
    """
    Build a concise 'lessons-learned' string from the most recent N
    feedback entries for the given topic.  This is injected back into
    agents so they learn from past mistakes.
    """
    history = load_feedback()
    # filter by topic (partial match) and low score
    relevant = [
        h for h in history
        if topic.lower() in h.get("topic", "").lower()
    ][-last_n:]

    if not relevant:
        return ""

    lines = ["\n\n### Lessons learned from previous runs (RLHF):"]
    for h in relevant:
        score   = h.get("score", "?")
        comment = h.get("comment", "").strip()
        liked   = h.get("liked",   "").strip()
        ts      = h.get("timestamp", "")
        lines.append(
            f"- Run {ts}: score={score}/5 | "
            f"What worked: {liked or 'N/A'} | "
            f"What to improve: {comment or 'N/A'}"
        )
    return "\n".join(lines)


# ── Crew with RLHF backstory injection ───────────────────────────────────────
@CrewBase
class RLHFMarketingCrew:
    """Marketing crew that adapts its agents' backstories from human feedback."""

    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    def __init__(self, topic: str, lessons: str = ""):
        self.topic   = topic
        self.lessons = lessons   # injected from past RLHF feedback

    # ── Agents ────────────────────────────────────────────────────────────────
    @agent
    def research_agent(self) -> Agent:
        base_backstory = (
            "You're a seasoned marketing content researcher with a knack for uncovering "
            "the latest developments in {topic}. Known for your ability to find the most "
            "relevant information and present it in a clear and concise manner."
        )
        return Agent(
            role=f"{self.topic} Marketing Content Researcher",
            goal=f"Uncover latest trends and insights in {self.topic}",
            backstory=base_backstory.format(topic=self.topic) + self.lessons,
            tools=[SerperDevTool()],
            verbose=True,
        )

    @agent
    def writer_agent(self) -> Agent:
        base_backstory = (
            "You're a meticulous content writer with a keen eye for detail. "
            "You're known for your ability to turn complex data into engaging and "
            "informative content that is easy to understand and act on."
        )
        return Agent(
            role=f"{self.topic} Marketing Content Writer",
            goal=f"Create engaging and informative content based on {self.topic} research",
            backstory=base_backstory + self.lessons,
            verbose=True,
        )

    @agent
    def reviewer_agent(self) -> Agent:
        base_backstory = (
            "You're a meticulous content reviewer known for making reports "
            "detailed, accurate, and highly actionable."
        )
        return Agent(
            role=f"{self.topic} Marketing Content Reviewer",
            goal=f"Review and refine {self.topic} content for clarity, accuracy, and engagement",
            backstory=base_backstory + self.lessons,
            verbose=True,
        )

    # ── Tasks ─────────────────────────────────────────────────────────────────
    @task
    def research_task(self) -> Task:
        extra = (
            "\n\nIMPORTANT based on past feedback: focus on concrete data points, "
            "statistics, and recent (current year) developments." + self.lessons
        )
        return Task(
            description=(
                f"Conduct a thorough research about {self.topic}. "
                f"The current year is {datetime.now().year}." + extra
            ),
            expected_output=f"A list with 10 bullet points of the most relevant "
                            f"information about {self.topic}",
            agent=self.research_agent(),
            tools=[SerperDevTool()],
        )

    @task
    def writing_task(self) -> Task:
        extra = (
            "\n\nIMPORTANT based on past feedback: use clear headings, "
            "include a TL;DR section, and keep sentences concise." + self.lessons
        )
        return Task(
            description=(
                f"Write a blog post about {self.topic} based on the research. "
                f"Current year is {datetime.now().year}." + extra
            ),
            expected_output=f"A well-structured blog post about {self.topic}",
            agent=self.writer_agent(),
            output_file="output.md",
        )

    @task
    def reviewing_task(self) -> Task:
        extra = (
            "\n\nIMPORTANT based on past feedback: highlight actionable "
            "takeaways and cite sources where possible." + self.lessons
        )
        return Task(
            description=(
                "Review the content and expand each topic into a full report section." + extra
            ),
            expected_output=(
                "A fully fledged report with main topics, each with a full section. "
                "Formatted as markdown without '```'."
            ),
            agent=self.reviewer_agent(),
            output_file="improvements.md",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
        )


# ── Human-feedback collection ─────────────────────────────────────────────────
def collect_human_feedback(topic: str, result: str) -> dict:
    """Prompt the human evaluator and return a structured feedback dict."""
    print("\n" + "═" * 60)
    print("🧑  HUMAN FEEDBACK  — Please rate the output")
    print("═" * 60)
    print("\n[CREW OUTPUT PREVIEW — first 800 chars]\n")
    print(str(result)[:800] + ("…" if len(str(result)) > 800 else ""))
    print("\n" + "─" * 60)

    while True:
        try:
            score = int(input("\n⭐  Score (1=terrible … 5=excellent): ").strip())
            if 1 <= score <= 5:
                break
            print("   Please enter a number between 1 and 5.")
        except ValueError:
            print("   Invalid input, try again.")

    liked   = input("👍 What did the agents do well? (Enter to skip): ").strip()
    comment = input("✏️  What should be improved next time?  (Enter to skip): ").strip()

    return {
        "timestamp": datetime.now().isoformat(),
        "topic":     topic,
        "score":     score,
        "liked":     liked,
        "comment":   comment,
    }


# ── Main RLHF loop ────────────────────────────────────────────────────────────
def run_with_rlhf(topic: str) -> None:
    """
    Run the crew, collect human feedback, optionally retry if score is low.
    This loop mimics the RLHF reward-model / policy-improvement cycle.
    """
    current_year = str(datetime.now().year)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n🚀  Attempt {attempt}/{MAX_RETRIES} for topic: '{topic}'")

        # ── Load lessons from past feedback ──────────────────────────────────
        lessons = summarise_feedback(topic)
        if lessons:
            print("\n📚  Injecting past RLHF lessons into agent prompts…")
            print(lessons[:400] + ("…" if len(lessons) > 400 else ""))

        # ── Kick off the crew ─────────────────────────────────────────────────
        crew_instance = RLHFMarketingCrew(topic=topic, lessons=lessons)
        result = crew_instance.crew().kickoff(
            inputs={"topic": topic, "current_year": current_year}
        )

        print("\n\n" + "═" * 60)
        print("✅  CREW FINISHED — collecting human feedback…")
        print("═" * 60)

        # ── Collect feedback ──────────────────────────────────────────────────
        feedback = collect_human_feedback(topic, result)
        save_feedback(feedback)

        # ── Decide whether to retry ───────────────────────────────────────────
        if feedback["score"] >= SCORE_THRESHOLD:
            print(f"\n🎉  Score {feedback['score']}/5 meets threshold. Done!")
            break
        elif attempt < MAX_RETRIES:
            print(
                f"\n⚠️  Score {feedback['score']}/5 is below threshold ({SCORE_THRESHOLD}). "
                f"Re-running with enriched context… (attempt {attempt+1})"
            )
            time.sleep(2)
        else:
            print(
                f"\n🔴  Reached max retries ({MAX_RETRIES}). "
                f"Feedback saved for future improvement."
            )

    print("\n\n========= ALL DONE =========")
    print(f"Full feedback log: {FEEDBACK_LOG}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    topic = input("Enter the marketing topic to research: ").strip()
    run_with_rlhf(topic)
