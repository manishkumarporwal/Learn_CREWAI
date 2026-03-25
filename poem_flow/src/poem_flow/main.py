#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from poem_flow.crews.poem_crew.poem_crew   import PoemCrew
from poem_flow.crews.essay_crew.essay_crew import EssayCrew
from poem_flow.crews.story_crew.story_crew import StoryCrew


# ── Shared flow state ─────────────────────────────────────────────────────────
class ContentState(BaseModel):
    # Poem
    sentence_count: int = 1
    poem: str = ""

    # Essay
    essay_topic: str = "Artificial Intelligence and the Future of Work"
    paragraph_count: int = 3
    essay: str = ""

    # Story
    story_topic: str = "a lone astronaut who discovers an abandoned space station"
    story_genre: str = "science fiction"
    story: str = ""


# ── Flow ──────────────────────────────────────────────────────────────────────
class ContentFlow(Flow[ContentState]):

    # ── Step 1: Poem ──────────────────────────────────────────────────────────
    @start()
    def generate_sentence_count(self, crewai_trigger_payload: dict = None):
        print("\n🎭  [1/3] Starting POEM generation...")

        if crewai_trigger_payload:
            self.state.sentence_count = crewai_trigger_payload.get(
                "sentence_count", randint(1, 5)
            )
        else:
            self.state.sentence_count = randint(1, 5)

    @listen(generate_sentence_count)
    def generate_poem(self):
        result = (
            PoemCrew()
            .crew()
            .kickoff(inputs={"sentence_count": self.state.sentence_count})
        )
        print("\n✅  Poem generated!\n", result.raw)
        self.state.poem = result.raw

        # Save poem
        with open("poem.txt", "w") as f:
            f.write(self.state.poem)

    # ── Step 2: Essay ─────────────────────────────────────────────────────────
    @listen(generate_poem)
    def generate_essay(self):
        print("\n📝  [2/3] Starting ESSAY generation...")
        result = (
            EssayCrew()
            .crew()
            .kickoff(inputs={
                "topic":           self.state.essay_topic,
                "paragraph_count": self.state.paragraph_count,
            })
        )
        print("\n✅  Essay generated!")
        self.state.essay = result.raw

    # ── Step 3: Story ─────────────────────────────────────────────────────────
    @listen(generate_essay)
    def generate_story(self):
        print("\n📖  [3/3] Starting STORY generation...")
        result = (
            StoryCrew()
            .crew()
            .kickoff(inputs={
                "topic": self.state.story_topic,
                "genre": self.state.story_genre,
            })
        )
        print("\n✅  Story generated!")
        self.state.story = result.raw

    @listen(generate_story)
    def summarise(self):
        print("\n\n" + "═" * 60)
        print("🎉  ALL CONTENT GENERATED SUCCESSFULLY!")
        print("═" * 60)
        print(f"\n🎭  Poem      → poem.txt")
        print(f"📝  Essay     → essay_output.md")
        print(f"📖  Story     → story_output.md")
        print("═" * 60 + "\n")


# ── Entry points ──────────────────────────────────────────────────────────────
def kickoff():
    ContentFlow().kickoff()


def plot():
    ContentFlow().plot()


def run_with_trigger():
    import json, sys
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Pass JSON as argument.")
    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument.")

    flow = ContentFlow()
    result = flow.kickoff({"crewai_trigger_payload": trigger_payload})
    return result


if __name__ == "__main__":
    kickoff()
