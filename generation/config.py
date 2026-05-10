from pathlib import Path

import torch

MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
MAX_NEW_TOKENS = 512

TEST_MODE = True
TEST_SAMPLE_SIZE = 10

YEARS = [2021, 2024]
DAYS = range(1, 25)

CODE_VARIANTS = [
    "Write a highly optimized solution focusing on performance.",
    "Use a functional programming style where possible.",
    "Avoid using advanced libraries; rely on basic Python constructs.",
    "Use concise code, minimizing line count.",
]

STYLE_VARIANTS = [
    "Prefer short variable names.",
    "Prefer descriptive variable names.",
    "Use helper functions.",
    "Avoid helper functions.",
    "Favor list comprehensions.",
    "Avoid comprehensions.",
    "Use recursion where reasonable.",
    "Prefer iterative solutions.",
]

SYSTEM_PROMPT = (
    "You are a Python code generator.\n"
    "Output raw Python source code only.\n"
    "Do NOT use markdown.\n"
    "Do NOT wrap the code in triple backticks.\n"
    "Do NOT include explanations.\n"
    "Do NOT include comments outside the code.\n"
    "The first character of your response must be valid Python code."
    "DO NOT include any text before or after the code. ONLY output valid Python code that adheres to the problem requirements."
    "DO NOT include sample data or test cases in your response. ONLY output the code that solves the problem as specified in the prompt."
)

OUTPUT_PATH = Path("output.jsonl")
INPUT_DIR = Path("generation") / "data"
