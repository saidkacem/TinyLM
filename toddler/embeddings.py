"""Step 2: embeddings, one row of numbers (a vector) per token."""
import numpy as np

COLUMNS = ["me", "be", "want", "stop", "feel", "food", "hug", "sleep"]

# Cleaned-up values a trained toddler could end up with; real learned values look like random decimals.
CLEAN_ROWS = {
    "_I":      [1, 0, 0, 0, 0, 0, 0, 0],
    "_am":     [0, 1, 0, 0, 0, 0, 0, 0],
    "_want":   [0, 0, 1, 0, 0, 0, 0, 0],
    "_.":      [0, 0, 0, 1, 0, 0, 0, 0],
    "_hungry": [0, 0, 0, 0, 1, 1, 0, 0],
    "_sad":    [0, 0, 0, 0, 1, 0, 1, 0],
    "_tired":  [0, 0, 0, 0, 1, 0, 0, 1],
    "_milk":   [0, 0, 0, 0, 0, 1, 0, 0],
    "_mommy":  [0, 0, 0, 0, 0, 0, 1, 0],
    "_bed":    [0, 0, 0, 0, 0, 0, 0, 1],
}


def random_table(vocab_size, dim=len(COLUMNS), seed=0):
    """The table at 'birth'."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.5, size=(vocab_size, dim)).round(2) + 0.0  # + 0.0 turns -0.0 into 0.0


def toddler_table(tokenizer, seed=0):
    """Random table whose 10 used rows hold the cleaned-up trained values."""
    table = random_table(len(tokenizer.vocab), seed=seed)
    for token, values in CLEAN_ROWS.items():
        table[tokenizer.token_to_id[token]] = values
    return table
