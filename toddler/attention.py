"""Step 4: attention. Each token asks a question (query), shows a label (key) and offers a content (value)."""
import numpy as np

from .embeddings import COLUMNS
from .positions import POSITION_COLUMNS

STREAM = COLUMNS + POSITION_COLUMNS  # the 10 numbers of every vector: 8 for meaning, 2 for position
HEAD_SIZE = 4


def column(name):
    return STREAM.index(name)


def hint_attention():
    """Our hand-set head: _want asks "who is a feeling?", and takes that feeling's topic."""
    query, key, value = (np.zeros((len(STREAM), HEAD_SIZE)) for _ in range(3))
    out = np.zeros((HEAD_SIZE, len(STREAM)))
    query[column("want"), 0] = 6
    key[column("feel"), 0] = 2
    for slot, topic in enumerate(["food", "hug", "sleep"]):
        value[column(topic), slot] = 1
        out[slot, column(topic)] = 1
    return {"Wq": query, "Wk": key, "Wv": value, "Wo": out}


def softmax_rows(scores):
    exps = np.exp(scores - scores.max(axis=1, keepdims=True))  # subtracting the max avoids huge numbers, changes nothing
    return exps / exps.sum(axis=1, keepdims=True)


def attention(inputs, Wq, Wk, Wv, Wo):
    """One head of attention. Returns what it adds to each token's vector (mix · Wo), and the weights."""
    queries, keys, values = inputs @ Wq, inputs @ Wk, inputs @ Wv  # Q, K, V: one row per token
    scores = queries @ keys.T / np.sqrt(Wq.shape[1])  # Q · Kᵀ ÷ √(key size): one row per token asking
    scores[np.triu(np.ones(scores.shape, dtype=bool), k=1)] = -np.inf  # a token never sees the tokens after it
    weights = softmax_rows(scores)
    return weights @ values @ Wo, weights
