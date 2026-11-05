"""Everything the toddler has ever heard: the whole training data."""

CORPUS = [
    "I am hungry . I want milk .",
    "I am sad . I want mommy .",
    "I am tired . I want bed .",
]

MAX_LINES = 10  # the playgrounds keep the toddler's world small enough to follow


def check_corpus(lines):
    """A playground corpus: a list of 1 to MAX_LINES lines, each with at least 2 words separated by spaces."""
    if isinstance(lines, str) or not 1 <= len(lines) <= MAX_LINES:
        raise ValueError(f"MY_CORPUS must be a list of 1 to {MAX_LINES} lines, like CORPUS.")
    for line in lines:
        if not isinstance(line, str) or len(line.split()) < 2:
            raise ValueError(f"Each line needs at least 2 words, separated by spaces: {line!r}")
    return list(lines)
