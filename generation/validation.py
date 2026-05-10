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


class MissingFunctionError(CodeValidationError):
    """Raised when the generated code is missing any functions."""

    def __init__(self, function_name: str):
        super().__init__(
            f"Generated code is missing required function: {function_name}"
        )


def extract_python(text: str) -> str:
    """Extracts Python code from the given text, handling common formatting patterns.

    Args:
        text (str): The input text containing the code.

    Returns:
        str: The extracted Python code.
    """
    text = text.strip()

    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    patterns = (
        "import ",
        "from ",
        "def ",
        "class ",
        "for ",
        "while ",
        "if ",
        "with ",
        "try:",
    )

    lines = text.splitlines()

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        if stripped.startswith(patterns) or re.match(
            r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped
        ):
            return "\n".join(lines[i:]).strip()

    return text


def validate_code(text: str):
    """
    Validates that the given code is syntactically correct and contains at least one function definition.

    Args:
        code (str): The code to validate.

    Returns:
        str: The validated code if it is valid.
    Raises:
        EmptyCodeError: If the code is empty or only whitespace.
        SyntaxErrorInCode: If the code contains syntax errors.
        MissingFunctionError: If the code does not contain any function definitions.
    """

    code = extract_python(text)

    if not code or not code.strip():
        raise EmptyCodeError("Generated code is empty or whitespace.")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SyntaxErrorInCode(e) from e

    if not any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)):
        raise MissingFunctionError("At least one function definition is required.")

    return code
