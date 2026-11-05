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


@lru_cache(maxsize=None)
def _tokenizer(max_merges=None):
    from .bpe import BPETokenizer

    return BPETokenizer.train(CORPUS, max_merges)


# Step 1
_add(
    "1.1",
    lambda a: list(a) == _tokenizer().encode("I want bed ."),
    "Each word is one token here: look up _I, _want, _bed and _. in the vocabulary.",
    "[21, 28, 52, 22] = _I _want _bed _.",
)
_add(
    "1.2",
    lambda a: list(a) == _tokenizer().encode_word("_wand"),
    "Go down the merge list in order. Merge #8 needs `_wan t`: is there a t?",
    "_ w a n d → #5 → _w a n d → #6 → _wa n d → #7 → _wan d. Merge #8 needs `_wan t`: not found. "
    "Result: ['_wan', 'd'] (IDs 27, 17).",
)
_add(
    "1.3",
    lambda a: list(a) == _tokenizer().encode_word("_tamed"),
    "The merge list glues `_ a` (an a at a word start) and then `_a m`, never a plain `a m`. "
    "Which merges can fire on `_ t a m e d`?",
    "_ t a m e d → #10 (e d) → _ t a m ed → #27 (_ t) → _t a m ed. `a m` never glues: the list only has `_a` + `m`. "
    "Result: ['_t', 'a', 'm', 'ed'] (IDs 47, 2, 3, 30).",
)
_add(
    "1.4",
    lambda a: a.strip() == _tokenizer().decode([21, 24, 44, 22]),
    "Look up each ID, glue the pieces, turn _ into spaces, drop the first space.",
    "21 24 44 22 → _I _am _mom _. → 'I am mom .'",
)
_add(
    "1.5",
    lambda a: int(a) == len(_tokenizer(10).tokenize("I am hungry .")),
    "With 10 merges, _I, _am and _. are whole. Are the merges that build _hungry among the first 10?",
    "10 tokens: _I _am _ h u n g r y _. The merges that build _hungry (#11 to #16) are missing, "
    "so 'hungry' stays in 7 pieces.",
)


def _clean_row(token):
    from .embeddings import CLEAN_ROWS

    return np.array(CLEAN_ROWS[token], dtype=float)


def _check_dot(my_dot):
    rng = np.random.default_rng(1)
    for _ in range(5):
        a, b = rng.normal(size=8), rng.normal(size=8)
        result = my_dot(a, b)
        if result is None:
            return None
        if np.ndim(result) != 0 or not np.isclose(result, a @ b):
            return False
    return True


def _check_thing_column(table9):
    from .embeddings import CLEAN_ROWS

    table9 = np.asarray(table9, dtype=float)
    tok = _tokenizer()
    if table9.shape != (len(tok.vocab), 9):
        return False
    if not table9[:, 8].any():
        return None
    for token, values in CLEAN_ROWS.items():
        thing = 1 if token in ("_milk", "_mommy", "_bed") else 0
        if not np.allclose(table9[tok.token_to_id[token]], values + [thing]):
            return False
    return True


# Step 2
_add(
    "2.1",
    lambda a: np.ndim(a) == 0 and _close(a, _clean_row("_tired") @ _clean_row("_bed")),
    "Multiply column by column, then ADD the results: the answer is one number, not a row.",
    "_tired = 0 0 0 0 1 0 0 1 and _bed = 0 0 0 0 0 0 0 1 → multiply: 0 0 0 0 0 0 0 1 → sum = 1 "
    "(they share `sleep`).",
)
_add(
    "2.2",
    lambda a: a.strip().lstrip("_") == "milk",
    "Go column by column: feel 0 - 1 + 1, food 0 - 0 + 1, sleep 1 - 1 + 0.",
    "_bed - _tired + _hungry = 0 0 0 0 0 1 0 0 = _milk. Bed is to tired what milk is to hungry.",
)
_add(
    "2.3",
    _check_dot,
    "Start with total = 0, loop over the columns with i, add a[i] * b[i], and return total.",
    "\ndef my_dot(a, b):\n    total = 0\n    for i in range(len(a)):\n        total += a[i] * b[i]\n    return total",
)
_add(
    "2.4",
    _check_thing_column,
    "Set column 8 to 1 in the rows of _milk, _mommy and _bed only; every other used row keeps 0 there.",
    "\nfor token in ['_milk', '_mommy', '_bed']:\n    table9[tok.token_to_id[token], 8] = 1\n"
    "Now milk·mommy = milk·bed = mommy·bed = 1, and every other dot product stays the same.",
)
