import ast
import re


class CodeValidationError(Exception):
    """Custom exception for code validation errors."""


class EmptyCodeError(CodeValidationError):
    """Raised when the generated code is empty."""

    def __init__(self, message: str = "Generated code is empty."):
        super().__init__(message)


class SyntaxErrorInCode(CodeValidationError):
    """Raised when the generated code contains syntax errors."""

    def __init__(self, error: SyntaxError):
        super().__init__(f"Syntax error in generated code: {error}")


class NoMeaningfulCodeError(CodeValidationError):
    """Raised when the generated code does not contain meaningful statements."""

    def __init__(
        self, message: str = "Generated code does not contain meaningful statements."
    ):
        super().__init__(message)


def extract_code(text: str) -> str:
    pattern = re.compile(r"```(?:python|py)\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

    match = pattern.search(text)
    return match.group(1).strip() if match else text.strip()


def is_meaningful_code(code: str) -> bool:

    return any(
        isinstance(node, ast.stmt)
        and not isinstance(node, (ast.Import, ast.ImportFrom))
        for node in ast.walk(ast.parse(code))
    )


def validate_code(code: str):
    """
    Validates that the given code is syntactically correct and contains at least one function definition.

    Args:
        code (str): The code to validate.
    Raises:
        EmptyCodeError: If the code is empty or only whitespace.
        SyntaxErrorInCode: If the code contains syntax errors.
        NoMeaningfulCodeError: If the code does not contain meaningful statements.
    """
    if not code or not code.strip():
        raise EmptyCodeError("Generated code is empty or whitespace.")

    try:
        ast.parse(code)
    except SyntaxError as e:
        raise SyntaxErrorInCode(e) from e

    if not is_meaningful_code(code):
        raise NoMeaningfulCodeError(
            "Generated code does not contain meaningful statements."
        )
