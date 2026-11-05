# LLM from scratch: the toddler model

Build a tiny language model by hand to see how real LLMs work: tokenization, embeddings, attention, transformers, training.
The toddler's whole world is three sentences:

    I am hungry . I want milk .
    I am sad . I want mommy .
    I am tired . I want bed .

## Setup

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

Open this folder in VS Code, open a notebook from `notebooks/`, and select the `.venv` kernel (top right).

## How to practice

- Run the cells from top to bottom (Shift+Enter).
- In **Your turn** cells, replace `None` with your answer and run the cell: `check(...)` says if it is right, and gives a hint if not.
- Stuck? `solution("1.2")` shows the worked answer.
- Change things and re-run: every notebook is meant to be played with.

## Layout

- `notebooks/`: one notebook per step (`00_big_picture.ipynb`, `01_tokenization_bpe.ipynb`, ...)
- `toddler/`: the model's code, growing one step at a time; later notebooks import it
- `toddler/exercises.py`: the checks behind `check()` and `solution()`

## Steps

0. Big picture: next-token guessing and a counting model
1. Tokenization (BPE)
2. Embeddings
3. Positions
4. Attention
5. Transformer block
6. Output and softmax
7. Training
8. Generation
9. From toddler to ChatGPT
