import asyncio
import os
import sys

# Change to backend directory
os.chdir(r"C:\Users\manki\OneDrive\Desktop\NEXUS\backend")
sys.path.append(os.getcwd())

from scripts.benchmark.llm_eval import cmd_evaluate_llm

class DummyArgs:
    dataset = "scifact"
    preflight = False
    smoke_test = False

async def main():
    print("Running evaluate-llm...")
    await cmd_evaluate_llm(DummyArgs())
    print("ALL DONE")

asyncio.run(main())
