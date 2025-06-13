import os
import openai
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()  # Make sure .env file has OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")


def web_search(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=max_results):
            results.append(f"{r['title']}: {r['body']} ({r['href']})")
    return results


def summarize_results(results, query):
    joined_results = "\n".join(results)
    system_prompt = "You are an intelligent research agent that summarizes findings from web data."
    user_prompt = f"Here are some results I found for: {query}\n\n{joined_results}\n\nCan you summarize the most important insights?"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message["content"]


def agent(query):
    print(f"[Agent] Searching for: {query}")
    search_results = web_search(query)
    if not search_results:
        return "No relevant results found."

    print("[Agent] Summarizing...")
    summary = summarize_results(search_results, query)
    return summary


if __name__ == "__main__":
    user_query = input("What would you like to know about?\n> ")
    final_response = agent(user_query)
    print("\n[Final Answer]\n")
    print(final_response)
