(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
  const KEY = "toddler-llm-progress";
  const load = () => {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
  };
  const remember = (id, state) => {
    const progress = load();
    if (progress[id] !== "correct") progress[id] = state;
    localStorage.setItem(KEY, JSON.stringify(progress));
  };
  const progress = load();

  $(".theme-toggle")?.addEventListener("click", () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  });

  // Reading progress bar
  const bar = $(".progress-bar span");
  if (bar) {
    const update = () => {
      const page = document.documentElement;
      const max = page.scrollHeight - page.clientHeight;
      bar.style.width = `${max > 0 ? (page.scrollTop / max) * 100 : 0}%`;
    };
    addEventListener("scroll", update, { passive: true });
    update();
  }

  // Blocks fade in as they scroll into view
  if ("IntersectionObserver" in window && !calm) {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("in");
      observer.unobserve(entry.target);
    }), { rootMargin: "0px 0px -6% 0px" });
    new Set($$(".content > *, .card, .sentence, .pager > *")).forEach(element => {
      element.classList.add("reveal");
      observer.observe(element);
    });
  }

  // Table of contents follows the section being read
  const tocLinks = $$(".toc a");
  if (tocLinks.length && "IntersectionObserver" in window) {
    const linkFor = new Map(tocLinks.map(link => [decodeURIComponent(link.hash.slice(1)), link]));
    const spy = new IntersectionObserver(entries => entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      tocLinks.forEach(link => link.classList.remove("active"));
      linkFor.get(entry.target.id)?.classList.add("active");
    }), { rootMargin: "-15% 0px -75% 0px" });
    $$(".content h2[id]").forEach(heading => spy.observe(heading));
  }

  // Questions
  const normalize = text => text.toLowerCase()
    .replace(/[[\](){}'"`]/g, " ")
    .split(/[\s,;+]+/)
    .filter(Boolean)
    .map(part => (/^-?\d*\.?\d+$/.test(part) ? String(parseFloat(part)) : part))
    .join(" ");

  $$(".quiz").forEach(quiz => {
    const id = quiz.dataset.quiz;
    const feedback = $(".feedback", quiz);
    const answer = $(".answer", quiz);
    const showButton = $(".show-answer", quiz);
    const say = (text, kind) => {
      feedback.textContent = text;
      feedback.className = `feedback ${kind}`;
    };
    const reveal = () => {
      if (!answer.hidden) return;
      answer.hidden = false;
      answer.classList.add("opening");
      showButton.hidden = true;
    };
    const solved = () => {
      quiz.classList.add("solved");
      say("Correct!", "good");
      remember(id, "correct");
      reveal();
    };
    const missed = element => {
      say("Not quite. Try again, or show the answer.", "bad");
      element.classList.remove("shake");
      void element.offsetWidth;
      element.classList.add("shake");
    };
    showButton.addEventListener("click", () => {
      reveal();
      remember(id, "seen");
    });

    const choices = $(".choices", quiz);
    if (choices) {
      const options = $$(".choice", choices);
      for (let i = options.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [options[i], options[j]] = [options[j], options[i]];
      }
      options.forEach(option => choices.append(option));
      choices.addEventListener("click", event => {
        const option = event.target.closest(".choice");
        if (!option || option.disabled) return;
        if (option.dataset.correct === "true") {
          option.classList.add("right");
          options.forEach(other => { other.disabled = true; });
          solved();
        } else {
          option.classList.add("wrong");
          option.disabled = true;
          missed(option);
        }
      });
    }

    const form = $(".try", quiz);
    if (form) {
      const accepted = JSON.parse(form.dataset.accept).map(normalize);
      const input = $("input", form);
      form.addEventListener("submit", event => {
        event.preventDefault();
        if (!input.value.trim()) return;
        if (accepted.includes(normalize(input.value))) {
          input.classList.add("right");
          input.readOnly = true;
          solved();
        } else {
          missed(input);
        }
      });
    }

    if (progress[id] === "correct") {
      quiz.classList.add("solved");
      say("Solved before. Try it again if you like.", "good");
    }
  });

  // Figures open full size on click
  const lightbox = $(".lightbox");
  if (lightbox) {
    const big = $("img", lightbox);
    const close = () => {
      lightbox.classList.remove("open");
      setTimeout(() => { lightbox.hidden = true; }, 250);
    };
    $$(".content figure img").forEach(image => image.addEventListener("click", () => {
      big.src = image.src;
      big.alt = image.alt;
      lightbox.hidden = false;
      requestAnimationFrame(() => requestAnimationFrame(() => lightbox.classList.add("open")));
    }));
    lightbox.addEventListener("click", close);
    addEventListener("keydown", event => {
      if (event.key === "Escape" && !lightbox.hidden) close();
    });
  }

  // Home: progress per step, "continue" button
  if (document.body.classList.contains("lesson-page")) {
    localStorage.setItem("toddler-llm-last", location.pathname.split("/").pop());
  }
  $$(".card[data-step]").forEach(card => {
    const total = Number(card.dataset.total);
    const done = Object.keys(progress).filter(key => key.startsWith(`s${card.dataset.step}-`)).length;
    if (!total || !done) return;
    requestAnimationFrame(() => { $(".meter span", card).style.width = `${Math.min(100, (done / total) * 100)}%`; });
    $(".meter small", card).textContent = `${done} of ${total} questions done`;
  });
  const resume = $(".resume");
  const last = localStorage.getItem("toddler-llm-last");
  if (resume && last) {
    resume.href = last;
    resume.textContent = `Continue: Step ${(last.match(/\d+/) || [""])[0]}`;
    resume.hidden = false;
  }

  // Home: the toddler reads a sentence and guesses the next word
  const demo = $(".demo");
  const demoData = $("#demo-data");
  if (demo && demoData) {
    const examples = JSON.parse(demoData.textContent);
    const text = $(".demo-text", demo);
    const tokens = $(".demo-tokens", demo);
    const bars = $(".demo-bars", demo);
    const make = (tag, className = "", content = "") => {
      const node = document.createElement(tag);
      node.className = className;
      node.textContent = content;
      return node;
    };
    const play = async (example, speed) => {
      text.textContent = "";
      tokens.replaceChildren();
      bars.replaceChildren();
      for (const char of example.text) {
        text.textContent += char;
        await wait(45 * speed);
      }
      await wait(350 * speed);
      for (const [token, id] of example.tokens) {
        const chip = make("span", "tok");
        chip.append(make("b", "", token), make("small", "", String(id)));
        tokens.append(chip);
        await wait(150 * speed);
      }
      await wait(400 * speed);
      for (const [label, percent] of example.probs) {
        const fill = make("i");
        const track = make("div");
        track.append(fill);
        const row = make("div", "bar");
        row.append(make("span", "", label), track, make("em", "", `${percent}%`));
        bars.append(row);
        await wait(60 * speed);
        fill.style.width = `${percent}%`;
        await wait(160 * speed);
      }
      await wait(900 * speed);
      bars.firstElementChild?.classList.add("win");
      text.append(make("mark", "", ` ${example.next}`));
    };
    if (calm) {
      play(examples[0], 0);
    } else {
      (async () => {
        for (let i = 0; ; i = (i + 1) % examples.length) {
          await play(examples[i], 1);
          await wait(2600);
        }
      })();
    }
  }
})();
