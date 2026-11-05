"""Step 2: embeddings, one row of numbers (a vector) per token."""
import numpy as np

COLUMNS = ["me", "be", "want", "stop", "feel", "food", "hug", "sleep"]

# Our hint: rows written by hand, easy to read. Training (Step 3) finds its own numbers, which look like random decimals.
HINT_ROWS = {
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


def hint_table(tokenizer, seed=0):
    """Random table whose used rows hold our hint (hint tokens missing from the vocabulary are skipped)."""
    table = random_table(len(tokenizer.vocab), seed=seed)
    for token, values in HINT_ROWS.items():
        if token in tokenizer.token_to_id:
            table[tokenizer.token_to_id[token]] = values
    return table
