"""Step 5: the transformer block. Attention, then a feed-forward layer, each one adding its result to the stream."""
import numpy as np

from .attention import STREAM, attention, column

HIDDEN = 40  # neurons of the feed-forward layer: 4 x the vector size, like in real models

# Our hand-set neurons: (name, what it reads, its bias, its write row).
# A neuron reads with its column of W1. Its write row (its links to the 10 output units) is its row of W2.
NEURONS = [
    ("I, at the start", {"me": 1, "px": 1}, -1, {"me": -1, "be": 1}),
    ("I, later", {"me": 1, "px": -1}, -1, {"me": -1, "want": 1}),
    ("am", {"be": 1}, 0, {"be": -1, "feel": 1, "food": 1 / 3, "hug": 1 / 3, "sleep": 1 / 3}),
    ("a feeling", {"feel": 1}, 0, {"feel": -1, "food": -1, "hug": -1, "sleep": -1, "stop": 1}),
    ("the stop", {"stop": 1}, 0, {"stop": -1, "me": 1}),
    ("want", {"want": 1}, 0, {"want": -1, "feel": -1}),
    ("a thing", {"food": 1, "hug": 1, "sleep": 1, "me": -2, "be": -2, "want": -2, "stop": -2, "feel": -2}, 0,
     {"food": -1, "hug": -1, "sleep": -1, "stop": 1}),
]


def hint_feed_forward():
    """The 7 neurons above; the other 33 of the 40 stay at 0 in our hint."""
    W1, b1 = np.zeros((len(STREAM), HIDDEN)), np.zeros(HIDDEN)
    W2, b2 = np.zeros((HIDDEN, len(STREAM))), np.zeros(len(STREAM))
    for neuron, (_, reads, bias, writes) in enumerate(NEURONS):
        for name, weight in reads.items():
            W1[column(name), neuron] = weight
        b1[neuron] = bias
        for name, weight in writes.items():
            W2[neuron, column(name)] = weight
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def relu(x):
    return np.maximum(x, 0)


def feed_forward(stream, W1, b1, W2, b2):
    return relu(stream @ W1 + b1) @ W2 + b2  # F = ReLU(S · W1 + b1) · W2 + b2


def block(inputs, params):
    """Attention, then feed-forward; each one adds its result to the stream (the residual connection)."""
    added, weights = attention(inputs, params["Wq"], params["Wk"], params["Wv"], params["Wo"])
    stream = inputs + added
    return stream + feed_forward(stream, params["W1"], params["b1"], params["W2"], params["b2"]), weights
