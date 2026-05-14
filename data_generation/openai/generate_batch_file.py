from data_generation.models import Job
from typing import TypedDict, Literal


class Input(TypedDict):
    role: Literal["system", "user"]
    content: str


class BatchRequestBody(TypedDict):
    model: str
    input: list[Input]
    max_output_tokens: int


class BatchLine(TypedDict):
    custom_id: str
    method: str
    url: str
    body: BatchRequestBody


def generate_batch_line(job: Job, system_prompt: str, max_tokens: int) -> BatchLine:
    return {
        "custom_id": job.id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": job.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": job.prompt},
            ],
            "max_output_tokens": max_tokens,
        },
    }


def generate_batch_file(
    jobs: list[Job], system_prompt: str, max_tokens: int
) -> list[BatchLine]:
    return [generate_batch_line(job, system_prompt, max_tokens) for job in jobs]
