import sys

from data_generation.local.generator import GenerationModel, GenerationTokenizer
from data_generation.models import Job
from validation.validation import CodeValidationError, validate_code, extract_code


def render_chat_prompt(
    tokenizer: GenerationTokenizer, system_prompt: str, user_prompt: str
) -> str:
    """Render a chat prompt for a given system and user prompt."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    rendered_prompt = tokenizer.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return rendered_prompt


def generate_prompt_lens(
    prompts: list[str], tokenizer: GenerationTokenizer
) -> list[int]:
    return [
        len(tokenizer.tokenizer(prompt, add_special_tokens=False).input_ids)
        for prompt in prompts
    ]


def run_batch(
    jobs: list[Job],
    model: GenerationModel,
    tokenizer: GenerationTokenizer,
    system_prompt: str,
    batch_size: int,
    seed: int | None = None,
    max_new_tokens: int = 512,
) -> list[tuple[Job, str]]:
    results: list[tuple[Job, str]] = []

    rendered_prompts = [
        render_chat_prompt(tokenizer, system_prompt, job.prompt) for job in jobs
    ]
    prompt_lengths = generate_prompt_lens(rendered_prompts, tokenizer)

    sorted_jobs = sorted(
        zip(
            jobs,
            rendered_prompts,
            prompt_lengths,
            strict=True,
        ),
        key=lambda x: x[2],
    )

    for i in range(0, len(sorted_jobs), batch_size):
        batch = sorted_jobs[i : i + batch_size]
        prompts = [item[1] for item in batch]

        codes = model.generate(prompts, max_new_tokens, seed)
        for item, code in zip(batch, codes, strict=True):
            job = item[0]
            results.append((job, code))

    return results


def validate_batch(results: list[tuple[Job, str]]):
    valid: list[tuple[Job, str]] = []
    failed: list[Job] = []

    for job, text in results:
        try:
            code = extract_code(text)
            validate_code(code)
            valid.append((job, code))
        except CodeValidationError as e:
            print(f"Validation failed for {job.id}: {e}", file=sys.stderr)
            print(f"Code:\n{text}\n", file=sys.stderr, flush=True)
            failed.append(job)

    return valid, failed
