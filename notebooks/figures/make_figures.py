"""Draw the figures used in the notebooks: .venv/bin/python notebooks/figures/make_figures.py"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from toddler.attention import STREAM, attention, hint_attention  # noqa: E402
from toddler.block import NEURONS, feed_forward, hint_feed_forward  # noqa: E402
from toddler.bpe import BPETokenizer  # noqa: E402
from toddler.data import CORPUS  # noqa: E402
from toddler.embeddings import COLUMNS, toddler_table  # noqa: E402
from toddler.positions import POSITION_COLUMNS, position_vector, with_positions  # noqa: E402

MONO = "DejaVu Sans Mono"
INK = "#1f2937"
MUTED = "#6b7280"
COLORS = {  # name: (fill, border)
    "gray": ("#f3f4f6", "#9ca3af"),
    "blue": ("#dbeafe", "#2563eb"),
    "green": ("#dcfce7", "#16a34a"),
    "violet": ("#ede9fe", "#7c3aed"),
    "orange": ("#ffedd5", "#ea580c"),
    "red": ("#fee2e2", "#dc2626"),
}
TOK = BPETokenizer.train(CORPUS)
TABLE = toddler_table(TOK)


def canvas(width, height, scale=0.8, bottom=0.0):
    """One unit = `scale` inches, so sizes in points stay predictable."""
    fig = plt.figure(figsize=(width * scale, (height - bottom) * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(bottom, height)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, color, lw=1.6, radius=0.15):
    fill, border = COLORS[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=fill, edgecolor=border, linewidth=lw))


def text(ax, x, y, s, size=11, color=INK, mono=False, bold=False, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, family=MONO if mono else "DejaVu Sans",
            weight="bold" if bold else "normal", ha=ha, va=va, **kw)


def arrow(ax, start, end, color=INK, lw=1.6, rad=0.0, head=14):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(
        arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=head,
        connectionstyle=f"arc3,rad={rad}"))


def chip(ax, cx, cy, s, w, h=0.42, color="gray", size=10, bold=False, mono=True):
    box(ax, cx - w / 2, cy - h / 2, w, h, color, lw=1.2, radius=0.08)
    text(ax, cx, cy, s, size=size, mono=mono, bold=bold)


def cell_colors(value):
    """(fill, text color) for one embedding number."""
    if value == 1:
        return "#2563eb", "white"
    if value == 0:
        return "#eff6ff", "#93c5fd"
    return "#f3f4f6", MUTED


def signed_colors(value):
    """(fill, text color): blue for positive, orange for negative, pale for 0."""
    if abs(value) < 1e-9:
        return "#f8fafc", "#cbd5e1"
    base = (37, 99, 235) if value > 0 else (234, 88, 12)
    strength = 0.3 + 0.7 * min(abs(value), 1)
    fill = tuple((255 - (255 - c) * strength) / 255 for c in base)
    return fill, "white" if abs(value) > 0.8 else INK


def number(value):
    if abs(value) < 1e-9:
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".").replace("-", "−")


def percent(share):
    """Like the notebook text: 98.8%, 33.3%, 50%, and 2 decimals below 1% (0.24%)."""
    return f"{share:.2%}" if share < 0.01 else f"{share:.1%}".replace(".0%", "%")


def save(fig, name):
    fig.savefig(HERE / name, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("wrote", name)


def pipeline():
    fig, ax = canvas(13.6, 14.0, bottom=0.95)
    left, width = 1.3, 4.0
    middle = left + width / 2
    tokens = ["_I", "_am", "_hungry", "_.", "_I", "_want"]
    xs = [6.85 + 1.2 * k for k in range(len(tokens))]

    box(ax, left, 13.05, width, 0.7, "gray")
    text(ax, middle, 13.4, "I am hungry . I want", size=12, mono=True)
    text(ax, 6.3, 13.4, "the text so far", size=10, color=MUTED, ha="left")

    rows = [  # (y center, height, color, title, step)
        (12.1, 1.0, "blue", "1. Tokenizer", "Step 1"),
        (10.3, 1.2, "green", "2. Embedding", "Step 2"),
        (8.7, 0.9, "green", "3. + Position", "Step 3"),
        (6.9, 1.3, "violet", "4. Transformer blocks × N", "Steps 4 and 5"),
        (5.1, 1.0, "orange", "5. Output layer", "Step 6"),
        (3.4, 1.2, "orange", "6. Softmax", "Step 6"),
        (1.6, 0.9, "red", "7. Pick", "Step 8"),
    ]
    previous_bottom = 13.05
    for y, h, color, title, step in rows:
        box(ax, left, y - h / 2, width, h, color)
        text(ax, middle, y + 0.14, title, size=12.5, bold=True)
        text(ax, middle, y - 0.24, step, size=9.5, color=MUTED)
        arrow(ax, (middle, previous_bottom), (middle, y + h / 2))
        arrow(ax, (5.4, y), (6.15, y), color=MUTED, lw=1.2, head=10)
        previous_bottom = y - h / 2

    # 1. token pieces and their IDs
    for x, token in zip(xs, tokens):
        box(ax, x - 0.55, 11.675, 1.1, 0.85, "blue", lw=1.2, radius=0.08)
        text(ax, x, 12.27, token, size=10, mono=True)
        text(ax, x, 11.9, str(TOK.token_to_id[token]), size=10, bold=True)
        arrow(ax, (x, 11.65), (x, 10.86), color=MUTED, lw=1, head=8)

    # 2. one vector (8 numbers) per token
    for x, token in zip(xs, tokens):
        for i, value in enumerate(TABLE[TOK.token_to_id[token]]):
            fill = "#16a34a" if value == 1 else "#f0fdf4"
            ax.add_patch(Rectangle((x - 0.21, 10.82 - 0.13 * (i + 1)), 0.42, 0.13,
                                   facecolor=fill, edgecolor="#bbf7d0", linewidth=0.8))
    text(ax, 10.05, 9.5, "each ID → a vector of 8 numbers", size=9.5, color=MUTED)

    # 3. positions
    for k, x in enumerate(xs, start=1):
        chip(ax, x, 8.7, f"#{k}", 0.6, h=0.4, color="green", size=9.5)

    # 4. attention: _want looks back at every earlier token, mostly at _hungry
    for x, token in zip(xs, tokens):
        ax.add_patch(Circle((x, 6.85), 0.12, facecolor="#7c3aed", edgecolor="none"))
        text(ax, x, 6.52, token, size=8.5, color=MUTED, mono=True)
    for k in range(len(tokens) - 1):
        strong = tokens[k] == "_hungry"
        arrow(ax, (xs[-1], 7.0), (xs[k], 7.0), color="#7c3aed" if strong else "#c4b5fd",
              lw=2.2 if strong else 0.9, rad=0.3, head=12 if strong else 7)
    text(ax, 10.05, 8.05, "attention: _want looks back, mostly at _hungry", size=9.5, color="#7c3aed")
    text(ax, 9.85, 6.15, "then feed-forward: each token thinks on its own", size=9.5, color=MUTED)

    # 5. one score per vocabulary token
    rng = np.random.default_rng(3)
    heights = rng.uniform(0.04, 0.2, len(TOK.vocab))
    tall = {"_milk": 0.85, "_mommy": 0.32, "_bed": 0.25}
    for token, height in tall.items():
        heights[TOK.token_to_id[token]] = height
    for i, height in enumerate(heights):
        special = TOK.vocab[i] in tall
        ax.add_patch(Rectangle((6.3 + 0.134 * i, 4.72), 0.1, height,
                               facecolor="#ea580c" if special else "#d1d5db", edgecolor="none"))
    text(ax, 6.3 + 0.134 * TOK.token_to_id["_milk"] + 0.05, 5.72, "_milk", size=8.5, mono=True)
    text(ax, 9.85, 4.45, "one score for each of the 53 tokens", size=9.5, color=MUTED)

    # 6. probabilities
    for row, (label, share) in enumerate([("_milk", 0.95), ("_mommy", 0.03), ("_bed", 0.02), ("others", 0.0)]):
        y = 3.85 - 0.3 * row
        text(ax, 6.3, y, label, size=9.5, mono=True, ha="left")
        ax.add_patch(Rectangle((7.5, y - 0.1), max(5.0 * share, 0.03), 0.2, facecolor="#ea580c", edgecolor="none"))
        text(ax, 7.6 + 5.0 * share, y, "≈0%" if share == 0 else f"{share:.0%}", size=9.5, ha="left")

    # 7. pick, then loop back
    chip(ax, 6.85, 1.6, "_milk", 1.1, h=0.5, color="red", size=10.5, bold=True)
    text(ax, 7.6, 1.6, "→ text becomes  I am hungry . I want milk", size=9.5, color=MUTED, ha="left")
    ax.plot([left, 0.75, 0.75], [1.6, 1.6, 13.4], color=INK, lw=1.6)
    arrow(ax, (0.75, 13.4), (left, 13.4))
    text(ax, 0.42, 7.5, "add the new token to the text, then run everything again",
         size=10, color=MUTED, rotation=90)
    save(fig, "00_pipeline.png")


def cycle_box(ax, x, y, w, h, color, title, lines):
    box(ax, x, y, w, h, color)
    text(ax, x + w / 2, y + h - 0.38, title, size=12, bold=True)
    for i, (line, mono) in enumerate(lines):
        text(ax, x + w / 2, y + h - 0.95 - 0.42 * i, line, size=11, mono=mono)


def training_loop():
    fig, ax = canvas(12.4, 6.6)
    w, h = 5.2, 2.3
    cycle_box(ax, 0.3, 3.9, w, h, "gray", "1. Show a text, hide the next token",
              [("I am hungry . I want  ___", True), ("(the right answer is: milk)", False)])
    cycle_box(ax, 6.9, 3.9, w, h, "blue", "2. The model guesses",
              [("_milk   30%", True), ("_mommy  40%", True), ("_bed    30%", True)])
    cycle_box(ax, 6.9, 0.4, w, h, "orange", "3. Measure the surprise (the loss)",
              [("the right answer got only 30%", False), ("→ big surprise = big loss", False)])
    cycle_box(ax, 0.3, 0.4, w, h, "green", "4. Nudge every weight a tiny bit",
              [("so that milk gets a bit more likely", False), ("(backpropagation finds the direction)", False)])
    arrow(ax, (5.55, 5.05), (6.85, 5.05))
    arrow(ax, (9.5, 3.85), (9.5, 2.75))
    arrow(ax, (6.85, 1.55), (5.55, 1.55))
    arrow(ax, (2.9, 2.75), (2.9, 3.85))
    text(ax, 6.2, 3.3, "repeat\nthousands\nof times", size=10, color=MUTED, bold=True)
    save(fig, "00_training_loop.png")


def bpe_loop():
    fig, ax = canvas(14.2, 5.4)
    w, h, y = 4.2, 2.5, 2.6
    cycle_box(ax, 0.3, y, w, h, "blue", "1. Count",
              [("every side-by-side pair", False), ("_ I  6    _ .  6", True), ("_ a  3    a m  3  ...", True)])
    cycle_box(ax, 5.0, y, w, h, "orange", "2. Pick",
              [("the most frequent pair", False), ("(tie: the first one in the text)", False), ("→ _ I", True)])
    cycle_box(ax, 9.7, y, w, h, "green", "3. Glue",
              [("it everywhere, add it to the", False), ("vocabulary and the merge list", False),
               ("_ I → _I   (merge #1)", True)])
    arrow(ax, (4.55, 3.85), (4.95, 3.85))
    arrow(ax, (9.25, 3.85), (9.65, 3.85))
    arrow(ax, (11.8, 2.55), (2.4, 2.55), rad=-0.22)
    text(ax, 7.1, 1.3, "next round: count again", size=10.5, color=MUTED, bold=True)
    text(ax, 7.1, 0.8, "stop when no pair is left (real tokenizers stop when the vocabulary is big enough)",
         size=9.5, color=MUTED, style="italic")
    save(fig, "01_bpe_loop.png")


def bpe_merge_tree():
    fig, ax = canvas(9.0, 6.4)
    leaves = {"_": 1.0, "w": 2.0, "a": 3.0, "n": 4.0, "t": 5.0, "e": 6.6, "d": 7.6}
    nodes = [  # (token, x, y, merge number, left child (x, y), right child (x, y))
        ("_w", 1.5, 2.0, 5, (1.0, 1.0), (2.0, 1.0)),
        ("_wa", 2.25, 3.0, 6, (1.5, 2.0), (3.0, 1.0)),
        ("_wan", 3.125, 4.0, 7, (2.25, 3.0), (4.0, 1.0)),
        ("_want", 4.06, 5.0, 8, (3.125, 4.0), (5.0, 1.0)),
        ("ed", 7.1, 2.0, 10, (6.6, 1.0), (7.6, 1.0)),
    ]
    h = 0.48
    for token, x, y, number, *children in nodes:
        for cx, cy in children:
            ax.plot([cx, x], [cy + h / 2, y - h / 2], color=MUTED, lw=1.4, zorder=0)
    for symbol, x in leaves.items():
        chip(ax, x, 1.0, symbol, 0.5, h=h, size=12)
    for token, x, y, number, *_ in nodes:
        final = token in ("_want", "ed")
        w = 0.3 + 0.13 * len(token)
        chip(ax, x, y, token, w, h=h, color="red" if final else "blue", size=12, bold=final)
        text(ax, x + w / 2 + 0.1, y, f"#{number}", size=10, color=MUTED, bold=True, ha="left")
        if final:
            text(ax, x, y + 0.48, f"ID {TOK.token_to_id[token]}", size=10, bold=True, color="#dc2626")
    text(ax, 0.3, 6.0, '"wanted" → _want + ed', size=12.5, bold=True, ha="left")
    text(ax, 4.3, 0.35, 'start: "wanted" cut into single symbols; each #number is a merge from the list',
         size=9.5, color=MUTED)
    save(fig, "01_bpe_merge_tree.png")


def embedding_lookup():
    fig, ax = canvas(15.6, 7.6)
    ids = [21, 22, 24, 28, 33, 36, 39, 42, 46, 50, 52]
    target = TOK.token_to_id["_hungry"]
    x0, cw, ch = 5.2, 0.72, 0.44
    for i, name in enumerate(COLUMNS):
        text(ax, x0 + cw * (i + 0.5), 7.0, name, size=9.5, color=MUTED)
    text(ax, 5.05, 7.0, "ID  token", size=9.5, color=MUTED, ha="right", mono=True)
    for r, token_id in enumerate(ids):
        y = 6.45 - 0.5 * r
        is_target = token_id == target
        text(ax, 5.05, y, f"{token_id} {TOK.vocab[token_id]:<7}", size=10, mono=True, ha="right", bold=is_target)
        for i, value in enumerate(TABLE[token_id]):
            fill, ink = cell_colors(value)
            ax.add_patch(Rectangle((x0 + cw * i, y - ch / 2), cw, ch, facecolor=fill, edgecolor="white", lw=1.5))
            text(ax, x0 + cw * (i + 0.5), y, f"{value:g}", size=8.5 if value not in (0, 1) else 10, color=ink)
        if is_target:
            target_y = y
            ax.add_patch(Rectangle((x0 - 0.06, y - 0.29), cw * 8 + 0.12, 0.58, fill=False,
                                   edgecolor="#ea580c", lw=2.5))
    text(ax, 8.1, 0.85, "11 of the 53 rows; the other rows (single letters, half-words) stay random",
         size=9, color=MUTED)

    chip(ax, 1.4, target_y + 0.8, "_hungry", 1.3, h=0.48, color="blue", size=11)
    arrow(ax, (1.4, target_y + 0.55), (1.4, target_y + 0.26), lw=1.4, head=10)
    chip(ax, 1.4, target_y, "ID 36", 1.3, h=0.48, color="blue", size=11, bold=True)
    arrow(ax, (2.1, target_y), (3.6, target_y), color="#ea580c", lw=2)
    text(ax, 2.85, target_y + 0.25, "read row 36", size=9.5, color="#ea580c")

    arrow(ax, (11.05, target_y), (11.75, target_y), color="#ea580c", lw=2)
    for i, value in enumerate(TABLE[target]):
        fill, ink = cell_colors(value)
        ax.add_patch(Rectangle((11.85 + 0.45 * i, target_y - 0.21), 0.42, 0.42, facecolor=fill,
                               edgecolor="#ea580c", lw=1))
        text(ax, 11.85 + 0.45 * i + 0.21, target_y, f"{value:g}", size=10, color=ink)
    text(ax, 13.62, target_y + 0.5, "the vector of _hungry", size=10, color=MUTED)
    save(fig, "02_embedding_lookup.png")


def similarity():
    content = ["_hungry", "_sad", "_tired", "_milk", "_mommy", "_bed"]
    vectors = np.array([TABLE[TOK.token_to_id[t]] for t in content])
    sim = vectors @ vectors.T
    pairs = {(0, 3), (1, 4), (2, 5), (3, 0), (4, 1), (5, 2)}
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.imshow(sim, cmap="Blues", vmin=0, vmax=2.6)
    ax.set_xticks(range(len(content)), content, rotation=40, ha="right", family=MONO)
    ax.set_yticks(range(len(content)), content, family=MONO)
    for i in range(len(content)):
        for j in range(len(content)):
            ax.text(j, i, f"{sim[i, j]:g}", ha="center", va="center", fontsize=12,
                    color="white" if sim[i, j] >= 2 else INK, weight="bold" if (i, j) in pairs else "normal")
    ax.set_title("dot products: who is close to whom", fontsize=12)
    save(fig, "02_similarity.png")


def position_clock():
    tokens = [TOK.vocab[i] for i in TOK.encode(CORPUS[0])]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.6, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([-1.3, 1.3], [0, 0], color="#e5e7eb", lw=1, zorder=0)
    ax.plot([0, 0], [-1.3, 1.3], color="#e5e7eb", lw=1, zorder=0)
    text(ax, 1.0, -0.17, "px", size=10, color=MUTED)
    text(ax, -0.18, 1.0, "py", size=10, color=MUTED)
    ax.add_patch(Circle((0, 0), 1.3, fill=False, edgecolor="#d1d5db", lw=2))
    for position, token in enumerate(tokens, start=1):
        x, y = position_vector(position)
        is_i = token == "_I"
        if is_i:
            arrow(ax, (0, 0), (1.22 * x, 1.22 * y), color="#dc2626", lw=2.6, head=18)
        ax.add_patch(Circle((1.3 * x, 1.3 * y), 0.07, color="#dc2626" if is_i else "#7c3aed", zorder=3))
        label = f"#{position} {token}"
        chip(ax, 1.85 * x, 1.85 * y, label, 0.2 + 0.075 * len(label), h=0.34,
             color="red" if is_i else "violet", size=10.5, bold=is_i)
    ax.add_patch(Circle((0, 0), 0.06, color=INK, zorder=4))
    text(ax, 0, -2.45, "the hand turns 45° per position: the two _I point in opposite directions",
         size=10, color="#dc2626")
    save(fig, "03_position_clock.png")


def adding_positions():
    columns = COLUMNS + POSITION_COLUMNS
    fig, ax = canvas(13.4, 6.9, bottom=0.7)
    x0, width, height = 3.3, 0.8, 0.5
    xs = [x0 + width * i + (0.3 if i >= 8 else 0) for i in range(10)]
    for x, name in zip(xs, columns):
        text(ax, x + width / 2, 6.0, name, size=10, color=MUTED)
    text(ax, xs[0] + 4 * width, 6.5, "meaning (8 numbers)", size=10.5, bold=True, color="#2563eb")
    text(ax, xs[8] + width, 6.5, "position (2)", size=10.5, bold=True, color="#ea580c")
    meaning = TABLE[TOK.token_to_id["_I"]]
    for block, position in enumerate([1, 5]):
        top = 5.3 - 2.7 * block
        hand = position_vector(position)
        rows = [("_I", np.append(meaning, [0, 0])), (f"+ position #{position}", np.append(np.zeros(8), hand)),
                ("= input", np.append(meaning, hand))]
        for r, (label, values) in enumerate(rows):
            y = top - 0.62 * r
            text(ax, x0 - 0.25, y, label, size=11.5, mono=True, ha="right", bold=r == 2)
            for x, value in zip(xs, values):
                fill, ink = signed_colors(value)
                ax.add_patch(Rectangle((x + 0.03, y - height / 2), width - 0.06, height, facecolor=fill, edgecolor="none"))
                text(ax, x + width / 2, y, number(value), size=10.5, color=ink)
        y = top - 1.24
        ax.add_patch(Rectangle((xs[0] - 0.06, y - 0.33), xs[-1] + width - xs[0] + 0.12, 0.66, fill=False,
                               edgecolor="#dc2626", lw=2))
        center = (xs[-1] + width + 1.0, top - 0.62)
        ax.add_patch(Circle(center, 0.5, fill=False, edgecolor="#d1d5db", lw=1.5))
        arrow(ax, center, (center[0] + 0.42 * hand[0], center[1] + 0.42 * hand[1]), color="#dc2626", lw=2.2)
        text(ax, center[0], center[1] - 0.75, f"hand #{position}", size=10, color=MUTED)
    save(fig, "03_adding_positions.png")


def two_hands():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6))
    clocks = [("fast hand: 45° per step", 45, "#9 points exactly like #1"),
              ("slow hand: 5.6° per step", 45 / 8, "#9 is far from #1, but #2 is almost #1")]
    for ax, (title, step, note) in zip(axes, clocks):
        ax.set_xlim(-1.9, 1.9)
        ax.set_ylim(-1.9, 1.9)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Circle((0, 0), 1.2, fill=False, edgecolor="#d1d5db", lw=2))
        text(ax, 0, 1.72, title, size=13, bold=True)
        text(ax, 0, -1.68, note, size=10.5, color=MUTED)
        for position, color in [(9, "#dc2626"), (2, "#16a34a"), (1, "#2563eb")]:
            x, y = position_vector(position, step)
            arrow(ax, (0, 0), (1.1 * x, 1.1 * y), color=color, lw=2.6, head=16)
        tips = {}
        for position in (1, 2, 9):
            tips.setdefault(tuple(position_vector(position, step).round(3)), []).append(f"#{position}")
        for (x, y), labels in tips.items():
            text(ax, 1.42 * x, 1.42 * y + (0.12 if y >= 0 else -0.12), " = ".join(labels), size=11, bold=True)
        ax.add_patch(Circle((0, 0), 0.05, color=INK, zorder=4))
    save(fig, "03_two_hands.png")


def training_together():
    fig, ax = canvas(16.4, 5.1, bottom=0.5)
    parts = [("Embedding table", "trained", "green"), ("+ Position clock", "fixed: not trained", "gray"),
             ("Attention", "trained", "violet"), ("Feed-forward", "trained", "violet"),
             ("Output layer", "trained", "orange"), ("Guess vs answer", "milk 30%: surprise!", "red")]
    width, gap, y, height = 2.3, 0.42, 3.4, 1.3
    centers = [0.3 + width / 2 + i * (width + gap) for i in range(len(parts))]
    for x, (name, note, color) in zip(centers, parts):
        box(ax, x - width / 2, y - height / 2, width, height, color)
        text(ax, x, y + 0.2, name, size=11.5, bold=True)
        text(ax, x, y - 0.25, note, size=9.5, color=MUTED)
    for left, right in zip(centers, centers[1:]):
        arrow(ax, (left + width / 2, y), (right - width / 2, y))
    back = 1.75
    ax.plot([centers[-1], centers[-1]], [y - height / 2, back], color="#dc2626", lw=1.8)
    for right, left in zip(centers[::-1], centers[-2::-1]):
        arrow(ax, (right, back), (left, back), color="#dc2626", lw=1.8)
    for x, (_, _, color) in zip(centers[:-1], parts[:-1]):
        if color != "gray":
            arrow(ax, (x, back), (x, y - height / 2), color="#dc2626", lw=1.4, head=10)
            text(ax, x + 0.12, (back + y - height / 2) / 2, "push", size=9.5, color="#dc2626", ha="left")
    text(ax, 0.3, 4.6, "1. forward: one guess with the current numbers, positions included", size=11.5, bold=True,
         ha="left")
    text(ax, 0.3, 0.95, "2. backward: the surprise flows back, and every trained table gets its push in the same round",
         size=11.5, bold=True, color="#dc2626", ha="left")
    save(fig, "03_training_together.png")


def sentence_attention():
    ids = TOK.encode(CORPUS[0])
    added, weights = attention(with_positions(TABLE[ids]), **hint_attention())
    return ids, added, weights


def number_grid(ax, x, top, matrix, row_labels, col_labels, title, red_row=None, cell=0.62, height=0.5):
    """A matrix as colored cells with their numbers: column names above, row names on the left."""
    text(ax, x + matrix.shape[1] * cell / 2, top + 0.95, title, size=12.5, bold=True)
    for j, name in enumerate(col_labels):
        text(ax, x + (j + 0.5) * cell, top + 0.3, name, size=9, color=MUTED, mono=True)
    for i, values in enumerate(matrix):
        y = top - (i + 0.5) * height
        if row_labels:
            text(ax, x - 0.15, y, row_labels[i], size=9.5, mono=True, ha="right", bold=i == red_row,
                 color="#dc2626" if i == red_row else INK)
        for j, value in enumerate(values):
            fill, ink = signed_colors(value)
            ax.add_patch(Rectangle((x + j * cell + 0.03, y - height / 2 + 0.03), cell - 0.06, height - 0.06,
                                   facecolor=fill, edgecolor="none"))
            text(ax, x + (j + 0.5) * cell, y, number(value), size=9.5, color=ink)


def matrix_product(name, left, right, result, row, column, caption, cell=0.62, height=0.5):
    """left · right = result, as grids. left and right are (title, matrix, row names, column names), result is
    (title, matrix, column names). Row `row` of left and column `column` of right make the framed number of result."""
    (a_title, a, a_rows, a_cols), (b_title, b, b_rows, b_cols), (c_title, c, c_cols) = left, right, result
    top = 1.1 + max(len(a), len(b)) * height
    x_a = 1.7
    x_dot = x_a + a.shape[1] * cell + 0.5
    x_b = x_dot + 1.4
    x_eq = x_b + b.shape[1] * cell + 0.5
    x_c = x_eq + 0.5
    fig, ax = canvas(x_c + c.shape[1] * cell + 0.3, top + 1.4)
    number_grid(ax, x_a, top, a, a_rows, a_cols, a_title, red_row=row, cell=cell, height=height)
    number_grid(ax, x_b, top, b, b_rows, b_cols, b_title, cell=cell, height=height)
    number_grid(ax, x_c, top, c, None, c_cols, c_title, cell=cell, height=height)
    middle = top - len(a) * height / 2
    ax.add_patch(Circle((x_dot, middle), 0.08, color=INK))
    text(ax, x_eq, middle, "=", size=22, bold=True)
    frames = [(x_a, top - (row + 1) * height, a.shape[1] * cell, height, 2.2),
              (x_b + column * cell, top - len(b) * height, cell, len(b) * height, 2.2),
              (x_c + column * cell, top - (row + 1) * height, cell, height, 3.2)]
    for x, y, w, h, lw in frames:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#dc2626", lw=lw))
    text(ax, 0.2, 0.5, caption, size=11.5, bold=True, color="#dc2626", ha="left")
    save(fig, name)


def attention_inputs():
    ids = TOK.encode(CORPUS[0])
    return with_positions(TABLE[ids]), [f"#{p} {TOK.vocab[i]}" for p, i in enumerate(ids, start=1)]


def x_times_wq():
    inputs, labels = attention_inputs()
    stream, Wq = COLUMNS + POSITION_COLUMNS, hint_attention()["Wq"]
    slots = [f"s{s}" for s in range(1, Wq.shape[1] + 1)]
    matrix_product("04_x_times_wq.png", ("X  (8 × 10)", inputs, labels, stream), ("Wq  (10 × 4)", Wq, stream, slots),
                   ("Q  (8 × 4)", inputs @ Wq, slots), row=5, column=0,
                   caption="row 6 of X · column s1 of Wq = 1 × 6 = 6: slot s1 of the query of _want")


def q_times_kt():
    inputs, labels = attention_inputs()
    head = hint_attention()
    queries, keys = inputs @ head["Wq"], inputs @ head["Wk"]
    slots, numbers = [f"s{s}" for s in range(1, queries.shape[1] + 1)], [f"#{p}" for p in range(1, len(labels) + 1)]
    matrix_product("04_q_times_kt.png", ("Q  (8 × 4)", queries, labels, slots), ("Kᵀ  (4 × 8)", keys.T, slots, numbers),
                   ("Q · Kᵀ  (8 × 8)", queries @ keys.T, numbers), row=5, column=2,
                   caption="row 6 of Q · column 3 of Kᵀ = 6 × 2 = 12: the question of _want meets the label of _hungry")


def want_looks_back():
    ids, _, weights = sentence_attention()
    inputs, head = with_positions(TABLE[ids]), hint_attention()
    queries, keys, values = (inputs @ head[name] for name in ("Wq", "Wk", "Wv"))
    scores = queries @ keys.T / np.sqrt(queries.shape[1])
    tokens = [TOK.vocab[i] for i in ids]
    asker = tokens.index("_want")

    def vector(v):
        return "(" + ",".join(number(x) for x in v) + ")"

    fig, ax = canvas(16.9, 7.0, bottom=0.4)
    xs = [1.15 + 2.08 * k for k in range(len(tokens))]
    for k, (x, token) in enumerate(zip(xs, tokens)):
        hidden, strong, is_asker = k > asker, weights[asker, k] > 0.5, k == asker
        chip(ax, x, 3.6, f"#{k + 1} {token}", 1.8, h=0.55, color="red" if is_asker else "violet" if strong else "gray",
             size=11.5, bold=strong or is_asker)
        ink = INK if strong else MUTED
        text(ax, x, 2.9, f"key {vector(keys[k])}", size=9.5, mono=True, color=ink)
        text(ax, x, 2.5, f"value {vector(values[k])}", size=9.5, mono=True, color=ink)
        score = "score −∞" if hidden else f"score {number(scores[asker, k])}"
        text(ax, x, 2.0, score, size=10.5, color=ink, bold=strong)
        text(ax, x, 1.4, "0%: hidden" if hidden else percent(weights[asker, k]), size=13 if strong else 10.5,
             bold=strong, color="#7c3aed" if strong else MUTED)
        if k < asker:
            arrow(ax, (xs[asker] - 0.2, 3.95), (x, 3.95), color="#7c3aed" if strong else "#c4b5fd",
                  lw=3 if strong else 1, rad=0.35, head=16 if strong else 8)
    text(ax, xs[asker] - 0.35, 4.35, f"query {vector(queries[asker])}", size=10.5, mono=True, color="#dc2626",
         ha="left")
    text(ax, 0.3, 6.5, '_want asks "who is a feeling?"  Only _hungry has a matching key, so it gets almost all the '
         'attention', size=11.5, bold=True, ha="left")
    text(ax, 0.3, 0.75, f"mix = {weights[asker, 2]:.4f} × the value of _hungry = ({weights[asker, 2]:.4f},0,0,0)  →  "
         "Wo puts slot s1 in the food column: _want gets +0.99 food", size=11.5, color="#7c3aed", ha="left", bold=True)
    save(fig, "04_want_looks_back.png")


def attention_pattern():
    ids, _, weights = sentence_attention()
    labels = [f"#{p} {TOK.vocab[i]}" for p, i in enumerate(ids, start=1)]
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    shown = np.where(np.tril(np.ones_like(weights)) > 0, weights, np.nan)
    cmap = plt.get_cmap("Purples").copy()
    cmap.set_bad("#f3f4f6")
    ax.imshow(shown, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", family=MONO, fontsize=10)
    ax.set_yticks(range(len(labels)), labels, family=MONO, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if j > i:
                ax.text(j, i, "✗", ha="center", va="center", fontsize=10, color="#cbd5e1")
            else:
                value = weights[i, j]
                ax.text(j, i, percent(value), ha="center", va="center", fontsize=9,
                        color="white" if value > 0.6 else INK, weight="bold" if value > 0.6 else "normal")
    ax.set_xlabel("the token looked at (its key)", fontsize=11)
    ax.set_ylabel("the token asking (its query)", fontsize=11)
    ax.set_title("the weights: softmax of each row (✗ = the future, hidden)", fontsize=12)
    save(fig, "04_attention_pattern.png")


def block_diagram():
    fig, ax = canvas(14.4, 5.6, bottom=-0.3)
    y = 1.5
    ax.add_patch(FancyBboxPatch((0.4, y - 0.18), 13.3, 0.36, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor="#e5e7eb", edgecolor="none"))
    arrow(ax, (13.5, y), (14.15, y), color=MUTED, lw=2.5)
    text(ax, 0.4, 0.05, "each token's vector (10 numbers): the residual stream", size=10.5, color=MUTED, ha="left")
    text(ax, 14.1, 0.05, "to the output (Step 6)", size=10.5, color=MUTED, ha="right")
    for (left, title, note), plus in zip([(2.0, "Attention", "looks back at earlier tokens"),
                                          (7.9, "Feed-forward", "40 neurons, each token alone")], (5.6, 11.5)):
        box(ax, left, 2.7, 3.2, 1.3, "violet")
        text(ax, left + 1.6, 3.55, title, size=12.5, bold=True)
        text(ax, left + 1.6, 3.1, note, size=9.5, color=MUTED)
        arrow(ax, (left + 0.4, y + 0.18), (left + 0.4, 2.7), color="#7c3aed", lw=1.6)
        text(ax, left + 0.5, 2.15, "reads", size=9.5, color="#7c3aed", ha="left")
        ax.plot([left + 3.2, plus, plus], [3.35, 3.35, y + 0.42], color="#7c3aed", lw=1.6)
        arrow(ax, (plus, y + 0.6), (plus, y + 0.34), color="#7c3aed", lw=1.6)
        ax.add_patch(Circle((plus, y), 0.3, facecolor="white", edgecolor="#7c3aed", lw=2, zorder=3))
        text(ax, plus, y, "+", size=16, bold=True, color="#7c3aed", zorder=4)
        text(ax, plus + 0.4, 2.2, "adds its result", size=9.5, color="#7c3aed", ha="left")
    ax.add_patch(FancyBboxPatch((1.5, 0.45), 11.0, 4.0, boxstyle="round,pad=0,rounding_size=0.25", fill=False,
                                edgecolor="#9ca3af", lw=1.4, ls=(0, (5, 4))))
    text(ax, 7.0, 4.95, "one transformer block: each part reads the stream and adds its result to it", size=12,
         bold=True)
    save(fig, "05_block.png")


def want_through_block():
    ids = TOK.encode(CORPUS[0])
    inputs = with_positions(TABLE[ids])
    added, _ = attention(inputs, **hint_attention())
    stream = inputs + added
    thinking = feed_forward(stream, **hint_feed_forward())
    rows = [("_want", inputs[5]), ("+ attention", added[5]), ("+ feed-forward", thinking[5]),
            ("= final vector", stream[5] + thinking[5])]
    fig, ax = canvas(13.4, 4.4, bottom=0.3)
    x0, width, height = 3.4, 0.8, 0.52
    xs = [x0 + width * i + (0.3 if i >= 8 else 0) for i in range(10)]
    for x, name in zip(xs, STREAM):
        text(ax, x + width / 2, 3.95, name, size=10, color=MUTED)
    for r, (label, values) in enumerate(rows):
        y = 3.3 - 0.7 * r
        text(ax, x0 - 0.25, y, label, size=11.5, mono=True, ha="right", bold=r == 3)
        for x, value in zip(xs, values):
            fill, ink = signed_colors(value)
            ax.add_patch(Rectangle((x + 0.03, y - height / 2), width - 0.06, height, facecolor=fill, edgecolor="none"))
            text(ax, x + width / 2, y, number(value), size=10.5, color=ink)
    ax.add_patch(Rectangle((xs[0] - 0.06, 3.3 - 2.1 - 0.36), xs[-1] + width - xs[0] + 0.12, 0.72, fill=False,
                           edgecolor="#dc2626", lw=2))
    text(ax, xs[-1] + width + 0.25, 1.2, "food, and surely\nnot a feeling:\nlooks like _milk", size=10.5,
         color="#dc2626", ha="left", bold=True)
    save(fig, "05_want_through_block.png")


POSITIVE, NEGATIVE, FAINT = "#2563eb", "#ea580c", "#e5e7eb"


def stream_after_attention():
    """The 8 vectors of our sentence after attention (the stream S of Step 5), and the feed-forward hint."""
    inputs = with_positions(TABLE[TOK.encode(CORPUS[0])])
    added, _ = attention(inputs, **hint_attention())
    return inputs + added, hint_feed_forward()


def relu_plot():
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.axvspan(-3, 0, color="#f3f4f6", zorder=0)
    ax.axvspan(0, 3, color="#f5f3ff", zorder=0)
    ax.axhline(0, color="#9ca3af", lw=1, zorder=1)
    ax.axvline(0, color="#9ca3af", lw=1, zorder=1)
    xs = np.linspace(-3, 3, 301)
    ax.plot(xs, np.maximum(xs, 0), color="#7c3aed", lw=3, zorder=2)
    for x, (dx, dy, ha) in {-2: (0, -0.38, "center"), -0.29: (-0.12, 0.3, "right"), 0.5: (0.14, -0.22, "left"),
                            1: (-0.14, 0.24, "right"), 2: (-0.14, 0.24, "right")}.items():
        y = max(x, 0)
        ax.plot(x, y, "o", color=INK, ms=6, zorder=3)
        ax.text(x + dx, y + dy, f"{number(x)} → {number(y)}", ha=ha, va="center", fontsize=10, family=MONO)
    ax.text(-1.5, 2.6, "silent: ReLU gives 0", ha="center", fontsize=11, color=MUTED, weight="bold")
    ax.text(1.5, 2.6, "fires: ReLU keeps the number", ha="center", fontsize=11, color="#7c3aed", weight="bold")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.7, 2.95)
    ax.set_xlabel("z: what the neuron reads (bias included)", fontsize=10.5)
    ax.set_ylabel("ReLU(z)", fontsize=10.5)
    ax.set_title("ReLU(z) = the bigger of z and 0", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "05_relu.png")


def sum_badge(ax, x, y, radius=0.2, size=10):
    """The Σ of an output unit (layer 2 of the feed-forward layer): where its links arrive and are added up."""
    ax.add_patch(Circle((x, y), radius, facecolor="white", edgecolor="#7c3aed", lw=1.5, zorder=5))
    text(ax, x, y, "Σ", size=size, bold=True, color="#7c3aed", zorder=6)


def one_neuron():
    """Neuron 1 ("I, at the start") at the first _I: the neuron (layer 1), the 10 output units (layer 2), the add."""
    stream, ff = stream_after_attention()
    values, reads, bias, writes = stream[0], ff["W1"][:, 0], ff["b1"][0], ff["W2"][0]
    total = values @ reads + bias
    y_out = max(total, 0.0)
    outputs = y_out * writes
    fig, ax = canvas(18.4, 10.0, bottom=0.25)
    ys = [7.05 - 0.56 * i for i in range(len(STREAM))]
    cy = float(np.mean(ys))
    cell_w, cell_h, x_in, x_new = 0.95, 0.44, 1.2, 16.35
    top, bottom = 8.45, 1.05
    n_left, n_right, cx, radius, relu_left, relu_right = 3.9, 9.7, 5.3, 0.72, 6.95, 8.6
    o_left, o_right, x_unit, unit_w = 11.65, 15.25, 12.4, 2.05

    def cells(x, numbers, labels_left):
        for y, name, value in zip(ys, STREAM, numbers):
            text(ax, x - 0.15 if labels_left else x + cell_w + 0.15, y, name, size=11.5, mono=True, color=MUTED,
                 ha="right" if labels_left else "left")
            fill, ink = signed_colors(value)
            ax.add_patch(Rectangle((x, y - cell_h / 2), cell_w, cell_h, facecolor=fill, edgecolor="none", zorder=3))
            text(ax, x + cell_w / 2, y, number(value), size=12, color=ink, zorder=4)

    def link(start, end, weight):
        strong = abs(weight) > 1e-9
        ax.plot(*zip(start, end), color=(POSITIVE if weight > 0 else NEGATIVE) if strong else FAINT,
                lw=2.6 if strong else 1, zorder=2 if strong else 1)

    def header(x, title, subtitle, mono=False):
        text(ax, x, 8.05, title, size=12.5, bold=True)
        text(ax, x, 7.65, subtitle, size=11, color=MUTED, mono=mono)

    for left, right in [(n_left, n_right), (o_left, o_right)]:
        ax.add_patch(FancyBboxPatch((left, bottom), right - left, top - bottom, fill=False, edgecolor="#9ca3af",
                                    boxstyle="round,pad=0,rounding_size=0.25", lw=1.4, ls=(0, (5, 4))))
    header(x_in + cell_w / 2, "_I (#1) in the stream", "x: its 10 numbers")
    header((n_left + n_right) / 2, "layer 1: one neuron (1 of 40)", '"I, at the start": y = ReLU(read · x + bias)',
           mono=True)
    header((o_left + o_right) / 2, "3. layer 2: output units", "one per column: Σ, no ReLU")
    header(x_new + cell_w / 2, "4. add", "x + outputs")
    cells(x_in, values, labels_left=True)

    # layer 1: the neuron reads x with its read column, adds its bias, and ReLU gives its output y
    for y, weight in zip(ys, reads):
        link((x_in + cell_w, y), (cx, cy), weight)
        if abs(weight) > 1e-9:
            t = 0.36
            text(ax, x_in + cell_w + t * (cx - x_in - cell_w), y + t * (cy - y) + 0.27, f"× {number(weight)}",
                 size=13, bold=True, color=POSITIVE)
    text(ax, 3.02, 7.3, "read column", size=11.5, bold=True, color="#7c3aed")
    text(ax, 3.0, cy + 0.05, "× 0: the other\n8 columns are\nignored", size=10.5, color=MUTED, zorder=3,
         bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.add_patch(Circle((cx, cy), radius, facecolor="#ede9fe", edgecolor="#7c3aed", lw=2.2, zorder=5))
    text(ax, cx, cy, "Σ", size=26, bold=True, color="#7c3aed", zorder=6)
    text(ax, cx, cy + 1.1, "1. read", size=13, bold=True, color="#7c3aed")
    chip(ax, cx, cy - 1.6, f"bias {number(bias)}", 1.7, h=0.52, color="orange", size=12, bold=True)
    arrow(ax, (cx, cy - 1.34), (cx, cy - radius), color=NEGATIVE, lw=1.8)
    arrow(ax, (cx + radius, cy), (relu_left, cy), color=INK, lw=1.8)
    text(ax, (cx + radius + relu_left) / 2, cy + 0.32, f"z = {number(total)}", size=13.5, bold=True)
    box(ax, relu_left, cy - 0.7, relu_right - relu_left, 1.4, "violet")
    knee = (relu_left + relu_right) / 2
    ax.plot([relu_left + 0.25, knee, relu_right - 0.25], [cy - 0.38, cy - 0.38, cy + 0.46], color="#7c3aed", lw=2.4,
            zorder=4)
    text(ax, knee, cy + 1.1, "2. ReLU", size=13, bold=True, color="#7c3aed")
    text(ax, knee, cy - 1.05, f"z = {number(total)} > 0: it fires", size=12, bold=True, color="#7c3aed")
    arrow(ax, (relu_right, cy), (n_right - 0.15, cy), color=INK, lw=1.8)
    ax.add_patch(Circle((n_right, cy), 0.15, color="#7c3aed", zorder=6))
    text(ax, (relu_right + n_right) / 2, cy + 0.36, f"y = {number(y_out)}", size=13.5, bold=True, color="#7c3aed")
    text(ax, (n_left + n_right) / 2, 1.5, f"y = ReLU(1·1 (me) + 1·1 (px) − 1) = {number(y_out)}", size=12, mono=True)

    # layer 2: y travels along the 10 links of the write row, and each output unit adds up what arrives
    for y, weight in zip(ys, writes):
        link((n_right, cy), (x_unit, y), weight)
    for rank, j in enumerate(j for j, weight in enumerate(writes) if abs(weight) > 1e-9):
        side, lx = (1 if rank == 0 else -1), 10.9
        ly = cy + (lx - n_right) / (x_unit - n_right) * (ys[j] - cy) + 0.3 * side
        text(ax, lx, ly, f"× {'+' if writes[j] > 0 else ''}{number(writes[j])}", size=13, bold=True, zorder=4,
             color=POSITIVE if writes[j] > 0 else NEGATIVE, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    text(ax, (n_right + o_left) / 2, 7.65, "write row", size=11.5, bold=True, color="#7c3aed")
    text(ax, (n_right + o_left) / 2, 7.3, "its 10 links", size=10.5, color="#7c3aed")
    for y, name, value in zip(ys, STREAM, outputs):
        color = "gray" if abs(value) < 1e-9 else "blue" if value > 0 else "orange"
        box(ax, x_unit, y - cell_h / 2, unit_w, cell_h, color, lw=1.3, radius=0.08)
        sum_badge(ax, x_unit, y, radius=0.19, size=9)
        text(ax, x_unit + unit_w / 2 + 0.12, y, f"{name} {number(value)}", size=11.5, mono=True,
             color=MUTED if abs(value) < 1e-9 else INK, zorder=4)
    text(ax, (o_left + o_right) / 2, 1.5, "+ links from the 39 other\nneurons, silent here (y = 0)", size=10.5,
         color=MUTED)

    # the add: x itself goes around both layers (the residual stream), and the outputs are added to it
    for y in ys:
        arrow(ax, (x_unit + unit_w + 0.05, y), (x_new - 0.06, y), color="#cbd5e1", lw=1.2, head=8)
    cells(x_new, values + outputs, labels_left=False)
    bypass = 9.35
    ax.plot([x_in + cell_w / 2] * 2 + [x_new + cell_w / 2], [8.35, bypass, bypass], color=MUTED, lw=1.6)
    arrow(ax, (x_new + cell_w / 2, bypass), (x_new + cell_w / 2, 8.35), color=MUTED, lw=1.6)
    text(ax, (x_in + x_new + cell_w) / 2, bypass, "x itself goes around both layers, and the outputs are added to it: "
         "the residual stream (5.1)", size=11, color=MUTED, bbox=dict(facecolor="white", edgecolor="none", pad=2))
    text(ax, 0.3, 0.6, "the new _I: me 1 − 1 = 0 and be 0 + 1 = 1, the rest unchanged: it looks like _am, the token "
         "that comes next", size=13, bold=True, color="#dc2626", ha="left")
    save(fig, "05_one_neuron.png")


def layer_diagram(name, token=None, caption=()):
    """The feed-forward layer as a network: the 10 numbers of a token, layer 1 (the 7 hint neurons) and layer 2 (the
    10 output units, one per column). Without a token: the wiring (W1 and W2). With a token: the numbers that flow."""
    stream, ff = stream_after_attention()
    W1, b1, W2 = ff["W1"], ff["b1"], ff["W2"]
    used = len(NEURONS)
    if token is not None:
        values = stream[token]
        reads = values @ W1 + b1
        fired = np.maximum(reads, 0)
        written = fired @ W2
    fig, ax = canvas(17.4, 11.4, bottom=-0.1 - 0.5 * len(caption))
    ys_io = [9.0 - 0.8 * i for i in range(len(STREAM))]
    ys_n = [8.85 - 1.15 * k for k in range(used)]
    io_w, io_h, x_in, x_out = 1.85, 0.56, 0.7, 14.85
    n_x, n_w, n_h = 8.7, 3.6, 0.88
    in_right, n_left, n_right = x_in + io_w, n_x - n_w / 2, n_x + n_w / 2

    def line(start, end, weight, signal):
        live = abs(signal) > 1e-9
        color = (POSITIVE if signal > 0 else NEGATIVE) if live else FAINT
        ax.plot(*zip(start, end), color=color, lw=0.9 + 1.1 * min(abs(weight), 2), alpha=0.9 if live else 1,
                zorder=2 if live else 1, solid_capstyle="round")

    for i in range(len(STREAM)):
        for k in range(used):
            if abs(W1[i, k]) > 1e-9:
                line((in_right, ys_io[i]), (n_left, ys_n[k]), W1[i, k],
                     W1[i, k] if token is None else values[i] * W1[i, k])
    for k in range(used):
        for j in range(len(STREAM)):
            if abs(W2[k, j]) > 1e-9:
                signal = W2[k, j] if token is None else fired[k] * W2[k, j]
                line((n_right, ys_n[k]), (x_out, ys_io[j]), W2[k, j], signal)

    def node(x, y, label, value, unit=False):
        color = "gray" if value is None or abs(value) < 1e-9 else "blue" if value > 0 else "orange"
        box(ax, x, y - io_h / 2, io_w, io_h, color, lw=1.3, radius=0.1)
        if unit:
            sum_badge(ax, x, y)
        shown = label if value is None else f"{label} {number(value)}"
        faded = value is not None and abs(value) < 1e-9
        text(ax, x + io_w / 2 + (0.12 if unit else 0), y, shown, size=12.5, mono=True, color=MUTED if faded else INK,
             zorder=4)

    for i, column_name in enumerate(STREAM):
        node(x_in, ys_io[i], column_name, None if token is None else values[i])
        node(x_out, ys_io[i], column_name, None if token is None else written[i], unit=True)
    for k, (neuron_name, *_) in enumerate(NEURONS):
        if token is None:
            color, detail = "violet", f"bias {number(b1[k])}"
        else:
            color = "violet" if fired[k] > 0 else "gray"
            detail = f"reads {number(reads[k])} → y = {number(fired[k])}, " + ("fires" if fired[k] > 0 else "silent")
        box(ax, n_left, ys_n[k] - n_h / 2, n_w, n_h, color, lw=2 if color == "violet" and token is not None else 1.3,
            radius=0.12)
        text(ax, n_x, ys_n[k] + 0.18, f"{k + 1} · {neuron_name}", size=12.5, bold=True, zorder=4)
        text(ax, n_x, ys_n[k] - 0.21, detail, size=11, mono=True, color=INK if color == "violet" else MUTED, zorder=4)
    ax.add_patch(FancyBboxPatch((n_left, 0.5), n_w, 0.68, boxstyle="round,pad=0,rounding_size=0.12", fill=False,
                                edgecolor="#9ca3af", lw=1.3, ls=(0, (4, 3))))
    text(ax, n_x, 0.84, "8 … 40: all zeros, always silent", size=11, color=MUTED)

    first = "the token" if token is None else "_want in the stream"
    for x, title, subtitle in [(x_in + io_w / 2, first, "x: its 10 numbers"),
                               (n_x, "layer 1: the neurons", "7 of 40 used: y = ReLU(read · x + bias)"),
                               (x_out + io_w / 2, "layer 2: output units", "one per column: Σ, no ReLU")]:
        text(ax, x, 10.5, title, size=13.5, bold=True)
        text(ax, x, 9.98, subtitle, size=11, color=MUTED)
    text(ax, (in_right + n_left) / 2, 9.72, "W1: read columns", size=12.5, bold=True, color="#7c3aed")
    text(ax, (n_right + x_out) / 2, 9.72, "W2: write rows", size=12.5, bold=True, color="#7c3aed")
    text(ax, x_out + io_w / 2, 0.84, "then added to the token (5.1)", size=11, color=MUTED)
    third = ("gray line: carries 0", FAINT) if token is not None else ("thicker: bigger (⅓, 1 or 2)", INK)
    for k, (label, color) in enumerate([("blue line: a + number", POSITIVE), ("orange line: a − number", NEGATIVE),
                                         third]):
        y = 0.95 - 0.42 * k
        ax.plot([0.7, 1.4], [y, y], color=color, lw=2.6 if k < 2 else 1.4 if token is not None else 3.1)
        text(ax, 1.55, y, label, size=11, color=MUTED, ha="left")
    for k, line_text in enumerate(caption):
        text(ax, 0.7, -0.45 - 0.5 * k, line_text, size=12.5, bold=True, color="#dc2626", ha="left")
    save(fig, name)


def feed_forward_layer():
    layer_diagram("05_feed_forward_layer.png")


def want_through_layer():
    layer_diagram("05_want_through_layer.png", token=5, caption=(
        "only neuron 6 fires (y = 1). Along its links (its write row), the output units want and feel get −1 each.",
        "added to _want (5.1): want 1 − 1 = 0 and feel 0 − 1 = −1, while food 0.99 stays: food, and surely not a "
        "feeling."))


if __name__ == "__main__":
    pipeline()
    training_loop()
    bpe_loop()
    bpe_merge_tree()
    embedding_lookup()
    similarity()
    position_clock()
    adding_positions()
    two_hands()
    training_together()
    x_times_wq()
    q_times_kt()
    want_looks_back()
    attention_pattern()
    block_diagram()
    want_through_block()
    relu_plot()
    one_neuron()
    feed_forward_layer()
    want_through_layer()
