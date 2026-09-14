import asyncio
import os
from pathlib import Path
import sys

# Change to backend directory portably
backend_dir = Path(__file__).resolve().parent / "backend"
os.chdir(backend_dir)
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from scripts.benchmark.llm_eval import cmd_evaluate_llm


class DummyArgs:
    dataset = "scifact"
    preflight = False
    smoke_test = False


async def main():
    print("Running evaluate-llm...")
    await cmd_evaluate_llm(DummyArgs())
    print("ALL DONE")


if __name__ == "__main__":
    asyncio.run(main())
