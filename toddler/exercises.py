"""Checks for the "Your turn" cells: check() gives a hint, solution() shows the worked answer."""
from functools import lru_cache

import numpy as np

from .data import CORPUS

_EXERCISES = {}


def check(exercise, answer):
    """Say whether `answer` is right, with a hint if it is not."""
    validate, hint, _ = _EXERCISES[exercise]
    try:
        ok = None if _is_blank(answer) else validate(answer)
    except Exception as error:  # a wrong type or shape is just a wrong answer
        print(f"[{exercise}] Your answer caused an error: {error!r}")
        ok = False
    if ok is None:
        print(f"[{exercise}] Not answered yet: replace None with your answer.")
    elif ok:
        print(f"[{exercise}] Correct!")
    else:
        print(f"[{exercise}] Not yet. Hint: {hint}")
    return bool(ok)


def solution(exercise):
    """Print the worked answer."""
    print(f"[{exercise}] {_EXERCISES[exercise][2]}")


def _add(exercise, validate, hint, solution_text):
    _EXERCISES[exercise] = (validate, hint, solution_text)


def _is_blank(answer):
    if answer is None:
        return True
    if isinstance(answer, dict):
        return any(value is None for value in answer.values())
    if isinstance(answer, (list, tuple)):
        return any(value is None for value in answer)
    return False


def _close(a, b, tolerance=0.01):
    return abs(float(a) - float(b)) <= tolerance


# Step 0
_add(
    "0.1",
    lambda a: set(a) == {"am", "want"} and _close(a["am"], 0.5) and _close(a["want"], 0.5),
    "Count what followed 'I' (am 3 times, want 3 times), then divide each count by the total.",
    "After 'I': am 3/6 = 0.5, want 3/6 = 0.5. "
    "The counting model only sees 'I', so it can't tell the first 'I' from the second.",
)
_add(
    "0.2",
    lambda a: _close(a["counting model"], 1 / 3) and _close(a["perfect model"], 1),
    "The counting model only sees the last word, 'want'. A perfect model also sees 'hungry'.",
    "Counting model: 1/3 (want→milk, want→mommy and want→bed were each heard once). "
    "Perfect model: 1.0, because 'hungry' means it must be milk.",
)
_add(
    "0.3",
    lambda a: _close(a, 1 / 12),
    "Multiply the probability of each guess along the sentence. After 'am' any feeling is fine, "
    "as long as the thing after 'want' matches it.",
    "P(am | I) × P(want | I) × P(matching thing | want) = 1/2 × 1/2 × 1/3 = 1/12 ≈ 0.083. "
    "Every other guess is 100% sure.",
)
