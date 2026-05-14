from data_generation.local.batch import run_batch, validate_batch
from data_generation.models import Job
from data_generation.local.generator import GenerationModel, GenerationTokenizer


def run_jobs(
    jobs: list[Job],
    model: GenerationModel,
    tokenizer: GenerationTokenizer,
    system_prompt: str,
    max_retries: int = 3,
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> list[tuple[Job, str]]:
    results: list[tuple[Job, str]] = run_batch(
        jobs, model, tokenizer, system_prompt, batch_size, max_new_tokens=max_new_tokens
    )

    valid, failed = validate_batch(results)
    all_valid = valid

    for i in range(max_retries):
        if not failed:
            break

        print(f"Retry {i + 1}: {len(failed)} failed")

        retry_results = run_batch(
            failed,
            model,
            tokenizer,
            system_prompt,
            batch_size,
            max_new_tokens=max_new_tokens,
        )

        valid_retry, failed_retry = validate_batch(retry_results)

        failed = failed_retry

        all_valid.extend(valid_retry)

    return all_valid
