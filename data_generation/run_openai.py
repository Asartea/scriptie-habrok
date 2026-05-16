from argparse import ArgumentParser, Namespace
from openai import OpenAI
from pathlib import Path

from data_generation.files import (
    load_completed_samples,
)
from utils import write_to_jsonl
from data_generation.jobs import build_jobs
from data_generation.prompts import (
    CompProgrammingConfig,
    NormalConfig,
)

from data_generation.openai.generate_batch_file import generate_batch_file


def run_openai_model(
    years: list[int],
    days: list[int],
    model: str,
    output_dir: Path,
    *,
    comp_programming: bool = False,
    test_mode: bool = False,
    test_sample_size: int = 10,
    max_new_tokens: int = 512,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{'comp' if comp_programming else 'normal'}_{'test' if test_mode else 'full'}.jsonl"
    )
    completed = load_completed_samples(output_path, model)
    batch_file_path = output_dir / "batch_requests.jsonl"

    client = OpenAI()
    prompt_config = CompProgrammingConfig() if comp_programming else NormalConfig()

    system_prompt = prompt_config.system_prompt

    data_dir = Path("data_generation") / "data" / "aoc-problems"

    jobs = build_jobs(
        years,
        days,
        data_dir,
        model,
        completed,
        comp_programming=comp_programming,
        test_mode=test_mode,
        test_sample_size=test_sample_size,
    )

    print(f"Pending jobs: {len(jobs)}")

    batch_lines = generate_batch_file(jobs, system_prompt, max_new_tokens)
    write_to_jsonl(batch_lines, batch_file_path)

    batch_file = client.files.create(
        file=batch_file_path.open("rb"),
        purpose="batch",
    )
    print(f"Batch file created with ID: {batch_file.id}")
    batch_job = client.batches.create(
        completion_window="24h",
        endpoint="/v1/responses",
        input_file_id=batch_file.id,
    )
    print(f"Batch job created with ID: {batch_job.id}")
    id_file = output_dir / "batch_job_id.txt"
    with open(id_file, "w", encoding="utf-8") as f:
        f.write(batch_job.id)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run OpenAI generation pipeline")

    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--days", type=int, nargs="+", required=True)

    parser.add_argument("--comp-programming", action="store_true")
    parser.add_argument("--test-mode", action="store_true", default=False)

    parser.add_argument("--test-sample-size", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path("data_generation") / "data" / f"{args.model}"

    run_openai_model(
        years=args.years,
        days=args.days,
        output_dir=output_dir,
        model=args.model,
        comp_programming=args.comp_programming,
        test_mode=args.test_mode,
        test_sample_size=args.test_sample_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
