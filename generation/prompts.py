from transformers import AutoTokenizer


def build_prompt(problem: str, code_variant: str, style_variant: str) -> str:
    return f"""
You are solving an Advent of Code problem using Python.
Return ONLY valid Python code. You MUST ensure the code is syntactically correct and adheres to the problem requirements.
You MUST NOT include any explanations, comments, or text outside of the code. The code should be self-contained and executable.

Additional constraints:
{code_variant}
{style_variant}

Problem:
{problem}
""".strip()


def render_chat_prompt(
    tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str
) -> str:
    """Render a chat prompt for a given system and user prompt."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    rendered_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return rendered_prompt
