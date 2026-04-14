MAX_JD_LENGTH = 8000
MAX_JOB_TITLE_LENGTH = 200
MAX_EXTRA_INFO_LENGTH = 1000

def clean_input(input_str: str, max_char: int, required: bool = True) -> str:
    """
    Cleans the input string by stripping leading/trailing whitespace and removing extra spaces.
    """
    text = input_str.strip()
    if not text:
        if required:
            raise ValueError("Input cannot be empty")
        return text

    if len(text) > max_char:
        text = text[:max_char]

    return text