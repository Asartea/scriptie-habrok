import random

from pathlib import Path
from itertools import product

from data_generation.models import Job
from data_generation.prompts import (
    build_prompt,
    NormalConfig,
    CompProgrammingConfig,
)
from data_generation.files import read_file


def build_jobs(
    years: list[int],
    days: list[int],
    data_dir: Path,
    model: str,
    completed_ids: set[str],
    *,
    comp_programming: bool = False,
    test_mode: bool = False,
    test_sample_size: int = 10,
) -> list[Job]:
    config = CompProgrammingConfig() if comp_programming else NormalConfig()

    jobs: list[Job] = []

    for year, day in product(years, days):
        base_dir = data_dir / str(year) / str(day)

        problem = "\n\n".join(
            filter(
                None,
                [
                    read_file(base_dir / "part1.txt"),
                    read_file(base_dir / "part2.txt"),
                ],
            )
        )

        if not problem:
            print(f"Warning: no problem statement for {year=} {day=}, skipping.")
            continue

        print(f"Building jobs for {year=} {day=}")

        jobs.extend(
            Job(
                year=year,
                day=day,
                model=model,
                use_comp_programming=comp_programming,
                code_variant=code_variant,
                style_variant=style_variant,
                prompt=build_prompt(problem, code_variant, style_variant),
            )
            for code_variant, style_variant in config.generate_variant_pairs()
        )

    jobs = [job for job in jobs if job.id not in completed_ids]

    if test_mode:
        return random.sample(jobs, min(test_sample_size, len(jobs)))

    return jobs
