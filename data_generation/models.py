from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    year: int
    day: int
    model: str
    prompt: str
    code_variant: str
    style_variant: str
    use_comp_programming: bool = False

    @property
    def id(self) -> str:
        return f"{self.model}-{self.year}-{self.day}-{self.code_variant}-{self.style_variant}-{self.use_comp_programming}"
