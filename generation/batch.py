import sys

from generation.config import SYSTEM_PROMPT
from generation.generator import generate_batch, tokenizer
from generation.prompts import render_chat_prompt
from generation.models import Job
from generation.validation import CodeValidationError, validate_code


def run_batch(
    jobs: list[Job], batch_size: int, seed: int | None = None
) -> list[tuple[Job, str]]:
    results: list[tuple[Job, str]] = []

    for i in range(0, len(jobs), batch_size):
        batch = jobs[i : i + batch_size]
        prompts = [
            render_chat_prompt(tokenizer, SYSTEM_PROMPT, job.prompt) for job in batch
        ]

        codes = generate_batch(prompts, seed=seed)
        results.extend((job, code) for job, code in zip(batch, codes, strict=True))

    return results


def validate_batch(results: list[tuple[Job, str]]):
    valid: list[tuple[Job, str]] = []
    failed: list[Job] = []

    for job, text in results:
        try:
            code = validate_code(text)
            valid.append((job, code))
        except CodeValidationError as e:
            print(f"Validation failed for {job.id}: {e}", file=sys.stderr)
            print(f"Code:\n{text}\n", file=sys.stderr, flush=True)
            failed.append(job)

    return valid, failed
