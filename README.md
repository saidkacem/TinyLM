# LLM from scratch: the toddler model

Build a tiny language model by hand to see how real LLMs work: tokenization, embeddings, attention, transformers, training.
The toddler's whole world is three sentences:

    I am hungry . I want milk .
    I am sad . I want mommy .
    I am tired . I want bed .

## Setup

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/nbstripout --install    # keeps notebook outputs out of git commits

Open this folder in VS Code and open a notebook from `notebooks/`. At the top right, the kernel must be **.venv**
(click the kernel name, then Select Another Kernel > Python Environments > .venv).
Any other Python fails with a "... missing: this notebook runs on the wrong Python" message.

## How to practice

- Run the cells from top to bottom (Shift+Enter).
- In **Your turn** cells, replace `None` with your answer and run the cell: `check(...)` says if it is right, and gives a hint if not.
- Stuck? `solution("1.2")` shows the worked answer.
- Change things and re-run: every notebook is meant to be played with.

## Reading website

The same lessons as a website, without code but with interactive questions (progress is saved in the browser):

    .venv/bin/python website/build.py --serve

Then open http://localhost:8000. Without `--serve`, it only builds the site into `_site/`: open `_site/index.html`
directly, or copy that folder to any web server. The notebooks feed the site through a few invisible markers,
described at the top of `website/build.py`.

## Layout

- `notebooks/`: one notebook per step (`00_big_picture.ipynb`, `01_tokenization_bpe.ipynb`, ...)
- `notebooks/figures/`: the figures used in the notebooks, drawn by `make_figures.py`
- `toddler/`: the model's code, growing one step at a time; later notebooks import it
- `toddler/exercises.py`: the checks behind `check()` and `solution()`
- `website/`: the website builder, its page templates and assets

## Steps

0. Big picture: next-token guessing and a counting model
1. Tokenization (BPE)
2. Embeddings
3. Learning: scores, softmax, loss, backpropagation, training from random numbers
4. Positions
5. Attention
6. Transformer block
7. Generation
8. From toddler to ChatGPT

From Step 3 on, every step trains the toddler from random numbers (or from hand-written hint rows) and shows
the numbers converge. Steps 1, 3 and 4 end with a playground: write your own toddler world (up to 10 lines) and
run it again.
