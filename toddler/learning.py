"""Step 3: learning. Guess, measure the surprise, push every number a little, repeat."""
import numpy as np

from .positions import position_vector


def softmax(scores):
    exps = np.exp(scores - scores.max())  # subtracting the max avoids huge numbers and changes nothing
    return exps / exps.sum()


def make_guesses(tokenizer, corpus):
    """Every guess of the game: (token seen, its position, right answer), as IDs."""
    guesses = []
    for line in corpus:
        ids = tokenizer.encode(line)
        for position in range(1, len(ids)):
            guesses.append((ids[position - 1], position, ids[position]))
    return guesses


def input_vector(input_table, token, position, use_positions=False):
    """The token's row; with positions (Step 4), followed by the clock hand of its position."""
    if use_positions:
        return np.append(input_table[token], position_vector(position))
    return input_table[token]


def average_loss(input_table, output_table, guesses, use_positions=False):
    total = 0.0
    for token, position, answer in guesses:
        probs = softmax(output_table @ input_vector(input_table, token, position, use_positions))
        total += -np.log(probs[answer])
    return total / len(guesses)


def train_round(input_table, output_table, guesses, learning_rate=0.5, use_positions=False):
    """One round: every guess, then one step with the average push. Returns the average loss before the step."""
    push_input = np.zeros_like(input_table)
    push_output = np.zeros_like(output_table)
    total_loss = 0.0
    for token, position, answer in guesses:
        x = input_vector(input_table, token, position, use_positions)
        probs = softmax(output_table @ x)
        total_loss += -np.log(probs[answer])
        push_scores = probs.copy()
        push_scores[answer] -= 1
        push_output += np.outer(push_scores, x)
        push_input[token] += (push_scores @ output_table)[: input_table.shape[1]]  # the clock part is not trained
    input_table -= learning_rate * push_input / len(guesses)
    output_table -= learning_rate * push_output / len(guesses)
    return total_loss / len(guesses)


def train(input_table, output_table, guesses, rounds=300, learning_rate=0.5, use_positions=False):
    """Train copies of the two tables. losses[k] is the average loss after k rounds (losses[0]: at birth)."""
    input_table, output_table = input_table.copy(), output_table.copy()
    losses = [train_round(input_table, output_table, guesses, learning_rate, use_positions) for _ in range(rounds)]
    losses.append(average_loss(input_table, output_table, guesses, use_positions))
    return input_table, output_table, losses
