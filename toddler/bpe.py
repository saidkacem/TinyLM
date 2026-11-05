"""Step 1: a BPE (byte pair encoding) tokenizer. `_` marks the start of every word."""


def to_words(text):
    return ["_" + word for word in text.split()]


def count_words(corpus):
    counts = {}  # insertion order = order of first appearance in the text
    for text in corpus:
        for word in to_words(text):
            counts[word] = counts.get(word, 0) + 1
    return counts


def count_pairs(splits, word_counts):
    pairs = {}
    for word, pieces in splits.items():
        for pair in zip(pieces, pieces[1:]):
            pairs[pair] = pairs.get(pair, 0) + word_counts[word]
    return pairs


def merge_pair(pair, pieces):
    left, right = pair
    out, i = [], 0
    while i < len(pieces):
        if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == pair:
            out.append(left + right)
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return out


def train_bpe(corpus, max_merges=None, verbose=False):
    """Learn the merge list. Returns (merges, vocab); a token's position in vocab is its ID."""
    word_counts = count_words(corpus)
    splits = {word: list(word) for word in word_counts}
    vocab = []
    for pieces in splits.values():
        for symbol in pieces:
            if symbol not in vocab:
                vocab.append(symbol)
    merges = []
    while max_merges is None or len(merges) < max_merges:
        pairs = count_pairs(splits, word_counts)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)  # on a tie, max() keeps the first pair met in the text
        new_splits = {word: merge_pair(best, pieces) for word, pieces in splits.items()}
        if verbose:
            _print_round(len(merges) + 1, pairs, best, splits, new_splits)
        splits = new_splits
        merges.append(best)
        vocab.append(best[0] + best[1])
    return merges, vocab


def _print_round(number, pairs, best, before, after):
    top = [f"{a} {b}" for (a, b), count in pairs.items() if count == pairs[best]]
    top = ", ".join(top[:4]) + (", ..." if len(top) > 4 else "")
    winner = f"{best[0]} {best[1]}"
    changes = ";  ".join(
        f"{' '.join(before[w])} → {' '.join(after[w])}" for w in before if after[w] != before[w]
    )
    print(f"#{number:<3} top ({pairs[best]}): {top:<28} winner: {winner:<12} | {changes}")


class BPETokenizer:
    def __init__(self, merges, vocab):
        self.merges = merges
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}

    @classmethod
    def train(cls, corpus, max_merges=None, verbose=False):
        return cls(*train_bpe(corpus, max_merges, verbose))

    def encode_word(self, word, verbose=False, show_misses=False):
        """Go down the merge list in order; glue each pair wherever it appears."""
        pieces = list(word)
        if verbose:
            print(f"{'start':<16}{' '.join(pieces)}")
        for number, pair in enumerate(self.merges, start=1):
            merged = merge_pair(pair, pieces)
            found = merged != pieces
            if verbose and (found or show_misses):
                label = f"#{number} {pair[0]} + {pair[1]}"
                print(f"{label:<16}" + ("yes → " + " ".join(merged) if found else "no"))
            pieces = merged
        return pieces

    def tokenize(self, text):
        return [piece for word in to_words(text) for piece in self.encode_word(word)]

    def encode(self, text):
        ids = []
        for piece in self.tokenize(text):
            if piece not in self.token_to_id:
                raise ValueError(f"{piece!r} is not in the vocabulary (real tokenizers fall back to bytes)")
            ids.append(self.token_to_id[piece])
        return ids

    def decode(self, ids):
        text = "".join(self.vocab[i] for i in ids).replace("_", " ")
        return text[1:] if text.startswith(" ") else text
