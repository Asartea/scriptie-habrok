class Job:
    year: int
    day: int
    model: str
    prompt: str
    code_variant: str
    style_variant: str

    def __init__(
        self,
        year: int,
        day: int,
        model: str,
        prompt: str,
        code_variant: str,
        style_variant: str,
    ):
        self.year = year
        self.day = day
        self.model = model
        self.prompt = prompt
        self.code_variant = code_variant
        self.style_variant = style_variant

    @property
    def id(self) -> str:
        return f"{self.model}-{self.year}-{self.day}-{self.code_variant}-{self.style_variant}"
