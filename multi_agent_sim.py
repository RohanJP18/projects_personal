import random
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

AGENT_NAMES = ["Alpha", "Beta", "Gamma"]
INITIAL_GOALS = {
    "Alpha": "Collect information about the weather.",
    "Beta": "Find resources to build a shelter.",
    "Gamma": "Trade knowledge with other agents."
}

class LLM_Agent:
    def __init__(self, name, goal):
        self.name = name
        self.goal = goal
        self.memory = []

    def generate_message(self, context):
        messages = [
            {"role": "system", "content": f"You are agent {self.name}. Your objective is: {self.goal}"},
            {"role": "user", "content": f"Here is the context so far: {context}\nRespond with your next action or message."}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        message = response["choices"][0]["message"]["content"]
        self.memory.append(message)
        return message

def run_simulation(rounds=3):
    agents = {name: LLM_Agent(name, INITIAL_GOALS[name]) for name in AGENT_NAMES}
    context = ""

    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1} ---")
        for name, agent in agents.items():
            print(f"[{name} acting...]")
            response = agent.generate_message(context)
            print(response)
            context += f"\n{name}: {response}"

    print("\nFinal context:")
    print(context)

if __name__ == "__main__":
    run_simulation()
