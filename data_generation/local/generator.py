# type: ignore
# HuggingFace transformers is very dynamically typed, and Pylance struggles to understand it.

import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class GenerationTokenizer:
    def __init__(self, model_name: str) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


class GenerationModel:
    def __init__(
        self,
        model_name: str,
        tokenizer: GenerationTokenizer,
        system_prompt: str,
    ) -> None:
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.tokenizer: PreTrainedTokenizerBase = tokenizer.tokenizer
        self.system_prompt = system_prompt

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        seed: int | None = None,
    ) -> list[str]:
        if seed is not None:
            torch.manual_seed(seed)

        formatted_prompts = [f"{self.system_prompt}\n\n{prompt}" for prompt in prompts]

        inputs: BatchEncoding = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True,
        )

        model_inputs: dict[str, Tensor] = {
            k: v.to(self.model.device) for k, v in inputs.items()
        }

        outputs: Tensor = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        attention_mask = model_inputs["attention_mask"]
        prompt_lengths: Tensor = attention_mask.sum(dim=1)

        texts: list[str] = [
            self.tokenizer.decode(
                output[int(prompt_length) :],
                skip_special_tokens=True,
            )
            for output, prompt_length in zip(
                outputs,
                prompt_lengths,
                strict=True,
            )
        ]

        del outputs
        del model_inputs
        del inputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return texts
