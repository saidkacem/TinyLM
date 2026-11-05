"""Build the reading website from the notebooks.

    .venv/bin/python website/build.py            # build into _site/
    .venv/bin/python website/build.py --serve    # build, then preview at http://localhost:8000

How notebooks feed the site (the markers are HTML comments, so notebooks don't show them):
- only markdown cells are used; code cells are for practice
- text between <!-- notebook-only --> and <!-- /notebook-only --> is left out
- each <details><summary>...</summary>...</details> becomes a question card; inside it,
  <!-- answer: a | b --> adds a typed-answer check (<!-- format: ... --> is the input placeholder)
  and <!-- choices: *right | wrong | ... --> adds a multiple choice
"""
import argparse
import html
import json
import re
import shutil
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template

from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(ROOT))

from toddler.bpe import BPETokenizer  # noqa: E402
from toddler.data import CORPUS  # noqa: E402

ROADMAP = [  # (title, question) per step; steps without a notebook yet show as "coming soon"
    ("The big picture", "What does an LLM actually do?"),
    ("Tokenization", "How does text become numbers?"),
    ("Embeddings", "How can a number carry meaning?"),
    ("Learning", "How do random numbers become knowledge?"),
    ("Positions", "How does it know word order?"),
    ("Attention", "How does “want” find “hungry”?"),
    ("Transformer block", "How are the pieces stacked into a brain?"),
    ("Generation", "How does it write, one token at a time?"),
    ("Toddler → ChatGPT", "What changes at full scale?"),
]
FEELINGS, THINGS = ("hungry", "sad", "tired"), ("milk", "mommy", "bed")
BUILD_MARK = ".toddler-site"

MD = MarkdownIt("commonmark", {"html": True}).enable("table")
NOTEBOOK_ONLY = re.compile(r"<!--\s*notebook-only\s*-->.*?(?:<!--\s*/notebook-only\s*-->|\Z)", re.S)
DETAILS = re.compile(r"<details>\s*<summary>.*?</summary>(.*?)</details>", re.S)
QUIZ_META = re.compile(r"<!--\s*(answer|choices|format)\s*:(.*?)-->", re.S)
SUBHEADING = re.compile(r"^#{3,6} ", re.M)
TEMPLATES = {name: Template((HERE / "templates" / f"{name}.html").read_text(encoding="utf-8"))
             for name in ("index", "lesson")}

LOGO = ('<svg viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="logo" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#7c3aed"/><stop offset="1" stop-color="#2563eb"/></linearGradient></defs>'
        '<rect x="2" y="3" width="18" height="8" rx="4" fill="url(#logo)"/>'
        '<rect x="9" y="12.5" width="21" height="8" rx="4" fill="url(#logo)" opacity=".7"/>'
        '<rect x="2" y="22" width="13" height="8" rx="4" fill="#f97316"/></svg>')
THEME_ICONS = ('<svg class="moon" viewBox="0 0 24 24"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'
               '<svg class="sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4.5"/><path d="M12 1.5v3M12 19.5v3'
               'M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M1.5 12h3M19.5 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/></svg>')
FOOTER = ('<footer class="site-footer"><div class="wrap">Toddler LLM · built from the course notebooks '
          'with <code>website/build.py</code></div></footer>')


class Lesson:
    def __init__(self, path):
        self.step = int(path.stem.split("_")[0])
        self.url = f"step-{self.step}.html"
        self.question = ROADMAP[self.step][1] if self.step < len(ROADMAP) else ""
        self.quizzes = 0
        cells = [NOTEBOOK_ONLY.sub("", source).strip() for source in markdown_cells(path)]
        title, _, cells[0] = cells[0].partition("\n")
        self.title = title.lstrip("#").strip().split(":", 1)[-1].strip()
        self.body, self.toc = finish("".join(self.render(cell) for cell in cells if cell))
        self.minutes = max(1, round(len(re.sub(r"<[^>]+>", " ", self.body).split()) / 200))

    def render(self, source):
        parts, start = [], 0
        for match in DETAILS.finditer(source):
            before = source[start:match.start()]
            headings = [heading.start() for heading in SUBHEADING.finditer(before)]
            split = headings[-1] if headings else before.rstrip().rfind("\n\n") + 1
            parts += [MD.render(before[:split]), self.quiz(before[split:], match.group(1))]
            start = match.end()
        parts.append(MD.render(source[start:]))
        return "".join(parts)

    def quiz(self, question, body):
        self.quizzes += 1
        quiz_id = f"s{self.step}-q{self.quizzes}"
        meta = {key: value.strip() for key, value in QUIZ_META.findall(body)}
        controls = ""
        if "choices" in meta:
            options = [option.strip() for option in meta["choices"].split("|")]
            controls = '<div class="choices">' + "".join(
                f'<button type="button" class="choice" data-correct="{str(option.startswith("*")).lower()}">'
                f'{MD.renderInline(option.lstrip("*").strip())}</button>' for option in options) + "</div>"
        elif "answer" in meta:
            accepted = html.escape(json.dumps([answer.strip() for answer in meta["answer"].split("|")]))
            placeholder = html.escape(meta.get("format", "Type your answer"))
            controls = (f'<form class="try" data-accept="{accepted}"><input type="text" autocomplete="off" '
                        f'spellcheck="false" placeholder="{placeholder}" aria-label="Your answer">'
                        '<button type="submit">Check</button></form>')
        return (f'<div class="quiz" id="{quiz_id}" data-quiz="{quiz_id}"><span class="quiz-tag">Question</span>'
                f'{MD.render(question)}{controls}<p class="feedback" aria-live="polite"></p>'
                '<button type="button" class="show-answer">Show the answer</button>'
                f'<div class="answer" hidden>{MD.render(QUIZ_META.sub("", body))}</div></div>')


def markdown_cells(path):
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "markdown":
            source = cell["source"]
            yield "".join(source) if isinstance(source, list) else source


def finish(body):
    """Anchor every h2 for the table of contents, wrap figures and tables."""
    toc, used = [], set()

    def anchor(match):
        text = re.sub(r"<[^>]+>", "", match.group(1))
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"
        while slug in used:
            slug += "-2"
        used.add(slug)
        toc.append(f'<li><a href="#{slug}">{text}</a></li>')
        return f'<h2 id="{slug}">{match.group(1)}</h2>'

    body = re.sub(r"<h2>(.*?)</h2>", anchor, body)
    body = re.sub(r'<p><img src="([^"]+)" alt="([^"]*)"\s*/?></p>',
                  r'<figure><img src="\1" alt="\2"><figcaption>\2</figcaption></figure>', body)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    return body, "".join(toc)


def header(lessons, current=None):
    links = "".join(
        f'<a href="{lesson.url}"{" class=current" if lesson is current else ""} '
        f'title="Step {lesson.step}: {html.escape(lesson.title)}">{lesson.step}</a>' for lesson in lessons)
    return (f'<header class="site-header"><div class="wrap header-inner">'
            f'<a class="brand" href="index.html">{LOGO}<span>Toddler LLM</span></a>'
            f'<nav class="step-nav" aria-label="Steps">{links}</nav>'
            f'<button class="theme-toggle" type="button" aria-label="Dark or light mode">{THEME_ICONS}</button>'
            '</div></header>')


def pager(lesson, lessons):
    position = lessons.index(lesson)
    previous = lessons[position - 1] if position > 0 else None
    following = lessons[position + 1] if position + 1 < len(lessons) else None
    if previous:
        left = (f'<a class="prev" href="{previous.url}"><small>← Step {previous.step}</small>'
                f'<strong>{html.escape(previous.title)}</strong></a>')
    else:
        left = '<a class="prev" href="index.html"><small>← Back to</small><strong>All steps</strong></a>'
    if following:
        right = (f'<a class="next" href="{following.url}"><small>Step {following.step} →</small>'
                 f'<strong>{html.escape(following.title)}</strong></a>')
    elif lesson.step + 1 < len(ROADMAP):
        right = (f'<div class="soon"><small>Step {lesson.step + 1} · coming soon</small>'
                 f'<strong>{html.escape(ROADMAP[lesson.step + 1][0])}</strong></div>')
    else:
        right = ""
    return left + right


def demo_data(tokenizer):
    examples = []
    for sentence in CORPUS:
        words = sentence.split()
        prompt, answer = " ".join(words[:6]), words[6]
        others = [thing for thing in THINGS if thing != answer]
        examples.append({
            "text": prompt,
            "next": answer,
            "tokens": [[tokenizer.vocab[i], i] for i in tokenizer.encode(prompt)],
            "probs": [[f"_{answer}", 95], [f"_{others[0]}", 3], [f"_{others[1]}", 2]],
        })
    return json.dumps(examples)


def sentences_html():
    rows = []
    for sentence in CORPUS:
        chips = "".join(
            f'<span class="word{" feel" if word in FEELINGS else " thing" if word in THINGS else ""}">'
            f'{html.escape(word)}</span>' for word in sentence.split())
        rows.append(f'<div class="sentence">{chips}</div>')
    return "".join(rows)


def cards_html(lessons):
    available = {lesson.step: lesson for lesson in lessons}
    cards = []
    for step, (title, question) in enumerate(ROADMAP):
        lesson = available.get(step)
        if lesson:
            cards.append(
                f'<a class="card" href="{lesson.url}" data-step="{step}" data-total="{lesson.quizzes}">'
                f'<span class="num">Step {step}</span><h3>{html.escape(lesson.title)}</h3><p>{html.escape(question)}</p>'
                f'<div class="meter"><div><span></span></div>'
                f'<small>{lesson.minutes} min · {lesson.quizzes} questions</small></div></a>')
        else:
            cards.append(f'<div class="card soon"><span class="num">Step {step}</span>'
                         f'<h3>{html.escape(title)}</h3><p>{html.escape(question)}</p></div>')
    return "".join(cards)


def build(out):
    if out.exists():
        if not (out / BUILD_MARK).exists() and any(out.iterdir()):
            sys.exit(f"{out} is not empty and was not made by this script: pick another --out folder.")
        shutil.rmtree(out)
    shutil.copytree(HERE / "assets", out / "assets")
    shutil.copytree(NOTEBOOKS / "figures", out / "figures", ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    (out / BUILD_MARK).touch()

    lessons = [Lesson(path) for path in sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))]
    for lesson in lessons:
        page = TEMPLATES["lesson"].substitute(
            header=header(lessons, lesson), footer=FOOTER, step=lesson.step, title=html.escape(lesson.title),
            question=html.escape(lesson.question), minutes=lesson.minutes, count=lesson.quizzes,
            toc=lesson.toc, body=lesson.body, pager=pager(lesson, lessons))
        (out / lesson.url).write_text(page, encoding="utf-8")
    index = TEMPLATES["index"].substitute(
        header=header(lessons), footer=FOOTER, start=lessons[0].url if lessons else "#",
        demo=demo_data(BPETokenizer.train(CORPUS)), sentences=sentences_html(), cards=cards_html(lessons))
    (out / "index.html").write_text(index, encoding="utf-8")
    print(f"Built {len(lessons)} lessons into {out}")


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(out, port):
    with ThreadingHTTPServer(("127.0.0.1", port), partial(QuietHandler, directory=str(out))) as server:
        print(f"Open http://localhost:{port}   (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="Build the reading website from the notebooks.")
    parser.add_argument("--out", type=Path, default=ROOT / "_site", help="output folder (default: _site)")
    parser.add_argument("--serve", action="store_true", help="preview the site after building")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    out = args.out.resolve()
    build(out)
    if args.serve:
        serve(out, args.port)


if __name__ == "__main__":
    main()
