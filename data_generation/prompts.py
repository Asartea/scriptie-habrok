from transformers import AutoTokenizer


class NormalConfig:
    system_prompt: str = """
You are a code synthesis engine.

Output constraints:
- Output ONLY Python source code.
- The output must define functions/classes required to solve the problem.
- Do not include any executable code that demonstrates usage.
- Do not include sample inputs, sample outputs, or test data.
- Do not include a main function or __main__ guard.
- Do not include hardcoded example arrays or placeholder datasets.
- Do not include comments of any kind.

Violation examples (forbidden):
- input =
- example =
- sample =
- test =
"""
    code_variants: list[str] = [
        "Write a highly optimized solution focusing on performance.",
        "Use a functional programming style where possible.",
        "Avoid using advanced libraries; rely on basic Python constructs.",
        "Use concise code, minimizing line count.",
    ]
    style_variants: list[str] = [
        "Prefer short variable names.",
        "Prefer descriptive variable names.",
        "Use helper functions.",
        "Avoid helper functions.",
        "Favor list comprehensions.",
        "Avoid comprehensions.",
        "Use recursion where reasonable.",
        "Prefer iterative solutions.",
    ]


class CompProgrammingConfig:
    system_prompt: str = """
You are a code synthesis engine specialized in competitive programming problems.

Output constraints:
- Output ONLY Python source code.
- The output must define functions/classes required to solve the problem.
- Do not include any executable code that demonstrates usage.
- Do not include sample inputs, sample outputs, or test data.
- Do not include a main function or __main__ guard.
- Do not include hardcoded example arrays or placeholder datasets.
- Do not include comments of any kind.

Violation examples (forbidden):
- input =
- example =
- sample =
- test =
""".strip()

    code_variants: list[str] = [
        "Write a highly optimized solution focusing on performance.",
        "Use a functional programming style where possible.",
        "Avoid using advanced libraries; rely on basic Python constructs.",
        "Use concise code, minimizing line count.",
        "Write code in the style of an Advent of Code leaderboard submission.",
        "Favor clever shortcuts and compact logic over readability.",
        "Assume the code is written under time pressure by an experienced competitor.",
        "Prefer dense, idiomatic competitive programming code.",
        "Optimize for fast parsing and low overhead.",
    ]

    style_variants: list[str] = [
        "Prefer short variable names.",
        "Prefer descriptive variable names.",
        "Use helper functions.",
        "Avoid helper functions.",
        "Favor list comprehensions.",
        "Avoid comprehensions.",
        "Use recursion where reasonable.",
        "Prefer iterative solutions.",
        "Use compact control flow.",
        "Minimize vertical whitespace.",
        "Favor in-place mutation where practical.",
        "Use standard library utilities aggressively.",
        "Prefer flat code structure over abstraction.",
    ]


type Config = NormalConfig | CompProgrammingConfig


def build_prompt(problem: str, code_variant: str, style_variant: str) -> str:
    return f"""
    You are solving a programming problem.

    OUTPUT FORMAT:
    - Output ONLY Python code
    - No markdown
    - No explanations

    BEGIN PROBLEM
    {problem}
    END PROBLEM

    BEGIN INSTRUCTIONS
    {code_variant}
    {style_variant}
    END INSTRUCTIONS

    Now write the solution.
    """.strip()
