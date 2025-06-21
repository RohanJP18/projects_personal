#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(system, user):
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content": system},
            {"role":"user",   "content": user},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content

def write_file(path, content):
    Path(path).write_text(content, encoding="utf-8")
    print(f"Wrote {path}")

def run_tests(test_file):
    try:
        # invoke pytest as a module under the current Python interpreter
        res = subprocess.run(
            [sys.executable, "-m", "pytest", test_file],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout
    except subprocess.CalledProcessError as e:
        return e.stdout + "\n" + e.stderr

def main(task):
    base = Path("auto_dev_output")
    base.mkdir(exist_ok=True)

    # 1) PLAN
    plan = chat(
        "You are a senior software architect. Break down the user request into numbered implementation steps.",
        task
    )
    write_file(base/"PLAN.txt", plan)

    # 2) CODE (one module per step)
    plan_steps = [line for line in plan.splitlines() if line.strip() and line[0].isdigit()]
    code_modules = []
    for idx, step in enumerate(plan_steps, start=1):
        prompt = (
            f"You are a senior Python engineer.\n"
            f"Implement step {idx} as a standalone Python module.\n"
            f"Step description:\n{step}\n"
            "Include any imports and write clean, test-ready code."
        )
        code = chat("Generate the Python code only.", prompt)
        filename = base/f"step_{idx}.py"
        write_file(filename, code)
        code_modules.append(filename.name)

    # 3) TESTS
    tests = []
    for mod in code_modules:
        prompt = (
            f"You have a module `auto_dev_output/{mod}`.\n"
            "Write pytest unit tests covering its main functionality.\n"
            "Import the module and test edge cases."
        )
        test_code = chat("Generate pytest code only.", prompt)
        test_file = base/f"test_{mod}"
        write_file(test_file, test_code)
        tests.append(test_file.name)

    # 4) RUN
    print("\n🚀 Running tests with pytest...\n")
    for t in tests:
        result = run_tests(str(base/t))
        print(f"=== Results for {t} ===\n{result}\n")

    # 5) SUMMARY
    summary = chat(
        "You are a project manager. Summarize the planning, coding, and test outcomes succinctly.",
        (
            f"Plan:\n{plan}\n\n"
            f"Generated modules: {code_modules}\n\n"
            f"Test files: {tests}\n\n"
            "Test outputs above."
        )
    )
    write_file(base/"SUMMARY.txt", summary)
    print("\nDone! See auto_dev_output/ for PLAN, code, tests & SUMMARY.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Autonomous Dev CLI")
    p.add_argument("task", help="Describe what you want built")
    args = p.parse_args()
    main(args.task)
