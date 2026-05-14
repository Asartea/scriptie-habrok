from itertools import product
from random import shuffle


class BaseConfig:
    system_prompt: str
    code_variants: list[str]
    style_variants: list[str]
    max_variants_pairs: int = 25

    def generate_variant_pairs(self) -> list[tuple[str, str]]:
        pairs = list(product(self.code_variants, self.style_variants))
        shuffle(pairs)
        return pairs[: self.max_variants_pairs]


class NormalConfig(BaseConfig):
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
        "Write readable code.",
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


class CompProgrammingConfig(BaseConfig):
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
    ]

    style_variants: list[str] = [
        "Prefer short variable names.",
        "Prefer descriptive variable names.",
        "Use helper functions.",
        "Avoid helper functions.",
        "Favor list comprehensions.",
        "Prefer iterative solutions.",
        "Use compact control flow.",
        "Minimize vertical whitespace.",
        "Favor in-place mutation where practical.",
        "Use standard library utilities aggressively.",
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
