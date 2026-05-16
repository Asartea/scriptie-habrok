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
        comp_type = "comp" if self.use_comp_programming else "normal"
        return f"{self.model}@{self.year}@{self.day}@{self.code_variant}@{self.style_variant}@{comp_type}"
