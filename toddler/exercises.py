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


def _position(position):
    from .positions import position_vector

    return position_vector(position)


def _same_vector(answer, expected, tolerance=0.02):
    answer = np.asarray(answer, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    return answer.shape == expected.shape and bool(np.allclose(answer, expected, atol=tolerance))


def _check_position_function(my_position_vector):
    for position in range(1, 9):
        result = my_position_vector(position)
        if result is None:
            return None
        if not _same_vector(result, _position(position), tolerance=0.01):
            return False
    return True


# Step 3
_add(
    "3.1",
    lambda a: _same_vector(a, _position(7)),
    "Position 7 turns the hand by (7 - 1) × 45° = 270°. Where does a hand point at 270°?",
    "270° points straight down: (px, py) = (0, -1).",
)
_add(
    "3.2",
    lambda a: _same_vector(a, np.append(_clean_row("_want"), _position(6))),
    "Take the 8 meaning numbers of _want, then the 2 numbers of position 6 (225°).",
    "_want = 0 0 1 0 0 0 0 0 and position 6 (225°) = (-0.71, -0.71). "
    "Together: 0 0 1 0 0 0 0 0 -0.71 -0.71.",
)
_add(
    "3.3",
    lambda a: np.ndim(a) == 0 and _close(a, _position(2) @ _position(4), tolerance=0.02),
    "px × px + py × py, with position 2 = (0.71, 0.71) and position 4 = (-0.71, 0.71).",
    "0.71 × (-0.71) + 0.71 × 0.71 = -0.5 + 0.5 = 0. They are 2 steps (90°) apart, like positions 1 and 3.",
)
_add(
    "3.4",
    _check_position_function,
    "angle = math.radians((position - 1) * 45), then return [math.cos(angle), math.sin(angle)].",
    "\nimport math\n\ndef my_position_vector(position):\n    angle = math.radians((position - 1) * 45)\n"
    "    return [math.cos(angle), math.sin(angle)]",
)
_add(
    "3.5",
    lambda a: int(a) == 9,
    "After how many steps of 45° does the hand come back to where it started?",
    "Position 9: (9 - 1) × 45° = 360°, a full turn, so the hand points exactly like position 1.",
)


def _softmax(scores):
    scores = np.asarray(scores, dtype=float)
    exps = np.exp(scores - scores.max())
    return exps / exps.sum()


def _check_softmax(my_softmax):
    rng = np.random.default_rng(2)
    for _ in range(5):
        scores = rng.normal(size=6)
        result = my_softmax(scores)
        if result is None:
            return None
        if not _same_vector(result, _softmax(scores), tolerance=1e-6):
            return False
    return True


# Step 4
_add(
    "4.1",
    lambda a: np.ndim(a) == 0 and _close(a, np.exp(2) / (np.exp(2) + 5), tolerance=0.01),
    "The score of _hungry becomes 2 × 2 ÷ 2 = 2, the 5 others stay 0. Then softmax: e² ÷ (e² + 5 × e⁰).",
    "score 2 → e² = 7.39; total 7.39 + 5 = 12.39; share 7.39 ÷ 12.39 = 0.60.",
)
_add(
    "4.2",
    lambda a: str(a).strip().lower().lstrip("_") == "hug",
    "_want finds the feeling _sad. Which slot is _sad's value in, and where does Wo put that slot?",
    "_sad's value is (0, 1, 0, 0): slot s2, which Wo puts in the hug column. _want gets +0.99 hug.",
)
_add(
    "4.3",
    lambda a: np.ndim(a) == 0 and _close(a, 1 / 3),
    "A query of 0 gives every score 0, so every e^score is 1. How many tokens can _hungry see?",
    "3 tokens (#1, #2, #3), all with e^0 = 1: each gets 1 ÷ 3 = 1/3.",
)
_add(
    "4.4",
    _check_softmax,
    "exps = np.exp(scores), then return exps / exps.sum().",
    "\ndef my_softmax(scores):\n    exps = np.exp(scores)\n    return exps / exps.sum()",
)
