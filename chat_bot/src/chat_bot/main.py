#!/usr/bin/env python
import os
import sys
import warnings
from datetime import datetime

from chat_bot.crew import ChatBot

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew interactively.
    Type your question when prompted, then press Enter.
    """
    print("\n" + "═" * 50)
    print("🤖  ChatBot — Powered by CrewAI + ChromaDB")
    print("   Knowledge: User Profile + Story Book PDF")
    print("═" * 50)
    question = input("\n❓ Ask a question: ").strip()
    if not question:
        question = "What stories are in the story book PDF?"
        print(f"   (using default: {question})")

    inputs = {'question': question}

    try:
        result = ChatBot().crew().kickoff(inputs=inputs)
        print("\n\n========== FINAL ANSWER ==========")
        print(result)
        print("═" * 50 + "\n")
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """Train the crew for a given number of iterations."""
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        ChatBot().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """Replay the crew execution from a specific task."""
    try:
        ChatBot().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """Test the crew execution and return the results."""
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    try:
        ChatBot().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    run()
