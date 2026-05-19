from openai import OpenAI
import json
from pathlib import Path

from data_generation.files import (
    create_sample,
    write_all_samples,
)
from models.models import LLMSample
from data_generation.models import Job
from validation.validation import validate_code, extract_code


def retrieve_batch_file(
    client: OpenAI, batch_file_input_id: str, batch_file_output_id: str
) -> tuple[list[LLMSample], list[Job]]:
    batch_input_file_content = client.files.content(batch_file_input_id)
    batch_output_file_content = client.files.content(batch_file_output_id)

    valid: list[LLMSample] = []
    failed: list[Job] = []

    sorted_input_output = zip(
        sorted(
            batch_input_file_content.text.strip().splitlines(),
            key=lambda line: json.loads(line)["custom_id"],
        ),
        sorted(
            batch_output_file_content.text.strip().splitlines(),
            key=lambda line: json.loads(line)["custom_id"],
        ),
    )

    for input_line, output_line in sorted_input_output:
        input_data = json.loads(input_line)
        output_data = json.loads(output_line)

        if input_data["custom_id"] != output_data["custom_id"]:
            print(
                f"Warning: Mismatched custom_id in input and output files: {input_data['custom_id']} vs {output_data['custom_id']}"
            )
            continue

        custom_id: str = input_data["custom_id"]

        # FIXME: This is a really stupid hack for a really stupid problem:
        # Its been fixed in the batch job creation code, but there is still an OpenAI run with the old stuff and I don't want to have to rerun that
        # This replaces known stuff with hyphens with a underscore version (which is what origjnally caused the problem), changes all seperators to "@", and then changes back the rest of the underscore stuff to hyphens so that the rest of the code can work as normal
        if "in-place" in custom_id or "gpt-5-mini-2025-08-07" in custom_id:
            custom_id = custom_id.replace(
                "in-place",
                "in_place",
            )
            custom_id = custom_id.replace(
                "gpt-5-mini-2025-08-07", "gpt_5_mini_2025_08_07"
            )
        custom_id = custom_id.replace("-", "@")
        if "in_place" in custom_id or "gpt_5_mini_2025_08_07" in custom_id:
            custom_id = custom_id.replace("in_place", "in-place")
            custom_id = custom_id.replace(
                "gpt_5_mini_2025_08_07", "gpt-5-mini-2025-08-07"
            )

        model, year, day, code_variant, style_variant, mode = custom_id.rsplit("@", 5)
        prompt = input_data["body"]["input"][1]["content"]

        try:
            job = Job(
                model=model,
                year=int(year),
                day=int(day),
                prompt=prompt,
                code_variant=code_variant,
                style_variant=style_variant,
                use_comp_programming=mode == "comp",
            )
        except ValueError:
            print(f"Warning: Invalid custom_id format: {custom_id}")
            continue

        try:
            output_text: str = next(
                content["text"]
                for item in output_data["response"]["body"]["output"]
                if item["type"] == "message"
                for content in item["content"]
                if content["type"] == "output_text"
            )
        except StopIteration:
            print(f"Output text not found for job {job.id}")
            failed.append(job)
            continue

        code = extract_code(output_text)
        try:
            validate_code(code)
            sample = create_sample((job, code))
            valid.append(sample)
        except Exception as e:
            print(f"Validation failed for job {job.id}: {e}")
            failed.append(job)

    return valid, failed


def main():
    client = OpenAI()
    output_dir = Path("data_generation") / "data" / "gpt-5-mini-2025-08-07"
    output_path = output_dir / "samples.jsonl"
    batch_job_id_file: Path = output_dir / "batch_job_id.txt"
    if not batch_job_id_file.exists():
        print(f"Batch job ID file not found at {batch_job_id_file}")
        return

    batch_file_input_ids = []
    batch_file_output_ids = []

    with batch_job_id_file.open("r", encoding="utf-8") as f:
        for section in f.read().split("\n\n"):
            batch_file_id, batch_job_id = [
                line.split(": ")[1].strip()
                for line in section.splitlines()
                if line.startswith("Batch job created with ID:")
                or line.startswith("Batch file created with ID:")
            ]
            batch_file_output_id = client.batches.retrieve(batch_job_id).output_file_id
            batch_file_input_ids.append(batch_file_id)
            batch_file_output_ids.append(batch_file_output_id)

    for batch_file_input_id, batch_file_output_id in zip(
        batch_file_input_ids, batch_file_output_ids
    ):
        valid_samples, failed_jobs = retrieve_batch_file(
            client, batch_file_input_id, batch_file_output_id
        )
        if valid_samples[0].get("use_comp_programming", False):
            output_path = output_dir / "comp_samples.jsonl"
        else:
            output_path = output_dir / "normal_samples.jsonl"
        write_all_samples(output_path, valid_samples)
        print(
            f"Retrieved {len(valid_samples)} valid samples and {len(failed_jobs)} failed jobs from batch file {batch_file_input_id}"
        )


if __name__ == "__main__":
    main()
