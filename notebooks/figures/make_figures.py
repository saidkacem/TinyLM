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

from toddler.bpe import BPETokenizer  # noqa: E402
from toddler.data import CORPUS  # noqa: E402
from toddler.embeddings import COLUMNS, toddler_table  # noqa: E402
from toddler.positions import POSITION_COLUMNS, position_vector  # noqa: E402

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
