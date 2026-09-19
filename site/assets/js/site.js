(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function setupTabs() {
    const list = document.querySelector(".rail[role=tablist]");
    if (!list) return;
    const tabs = Array.from(list.querySelectorAll("[role=tab]"));
    const notes = Array.from(document.querySelectorAll(".sc-note"));
    const narrow = window.matchMedia("(max-width: 720px)");

    const syncOrientation = () => {
      list.setAttribute("aria-orientation", narrow.matches ? "horizontal" : "vertical");
    };
    syncOrientation();
    narrow.addEventListener("change", syncOrientation);

    const select = (tab, focus) => {
      tabs.forEach((other) => {
        const selected = other === tab;
        other.setAttribute("aria-selected", String(selected));
        other.tabIndex = selected ? 0 : -1;
        const panel = document.getElementById(other.getAttribute("aria-controls"));
        if (panel) panel.hidden = !selected;
      });
      const key = tab.id.replace("tab-", "");
      notes.forEach((note) => {
        note.hidden = note.dataset.note !== key;
      });
      if (focus) tab.focus();
      if (narrow.matches) tab.scrollIntoView({ block: "nearest", inline: "nearest" });
    };

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => select(tab, false));
      tab.addEventListener("keydown", (event) => {
        const index = tabs.indexOf(tab);
        let next = null;
        if (event.key === "ArrowDown" || event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
        if (event.key === "ArrowUp" || event.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
        if (event.key === "Home") next = tabs[0];
        if (event.key === "End") next = tabs[tabs.length - 1];
        if (next) {
          event.preventDefault();
          select(next, true);
        }
      });
    });
  }

  const BAR_COUNT = 15;
  const FRAME_MS = 16;
  const ATTACK = 0.35;
  const DECAY = 0.1;
  const BAR_SMOOTHING = 0.45;
  const BAR_MIN = 3;
  const BAR_MAX = 20;
  const STATIC_POSE = [4, 7, 11, 9, 14, 18, 13, 20, 15, 17, 10, 12, 6, 8, 4];

  function makeSmoother() {
    const shape = [];
    const seeds = [];
    let seed = 7;
    const random = () => {
      seed = (seed * 16807) % 2147483647;
      return (seed - 1) / 2147483646;
    };
    for (let i = 0; i < BAR_COUNT; i += 1) {
      shape.push(0.42 + 0.58 * Math.sin((Math.PI * (i + 0.5)) / BAR_COUNT));
      seeds.push(random() * 7);
    }
    const bars = new Array(BAR_COUNT).fill(0);
    let envelope = 0;
    let frame = 0;
    return {
      step(target) {
        const t = Math.min(1, Math.max(0, target));
        envelope += (t - envelope) * (t > envelope ? ATTACK : DECAY);
        frame += 1;
        const ms = frame * FRAME_MS;
        for (let i = 0; i < BAR_COUNT; i += 1) {
          const s = seeds[i];
          const noise = 0.62 + 0.38 * Math.sin(ms / (150 + s * 40) + s * 3.1);
          const value = Math.min(1, Math.max(0, envelope * shape[i] * noise));
          bars[i] += (value - bars[i]) * BAR_SMOOTHING;
        }
        return bars;
      },
      reset() {
        envelope = 0;
        bars.fill(0);
      },
    };
  }

  const SCENES = [
    {
      key: "en",
      title: "Chat with Sam",
      glyph: "chat",
      label: "English, in a chat app.",
      result: "The filler and the self-correction are gone.",
      speakMs: 4700,
      processMs: 1100,
      chunks: [
        { at: 1700, words: ["um", "can", "we", "move"] },
        { at: 2900, words: ["the", "call", "to", "thursday"] },
        { at: 4100, words: ["no", "wait"] },
      ],
      drop: ["um", "thursday", "no", "wait"],
      fix: { can: "Can" },
      append: ["Friday?"],
    },
    {
      key: "de",
      title: "Neue Nachricht",
      glyph: "mail",
      label: "German, in an email.",
      result: "Fillers out, capitals and a full stop in.",
      speakMs: 4600,
      processMs: 1200,
      chunks: [
        { at: 1700, words: ["ähm", "ich", "schicke"] },
        { at: 2900, words: ["dir", "die", "unterlagen"] },
        { at: 4100, words: ["also", "morgen"] },
      ],
      drop: ["ähm", "also"],
      fix: { ich: "Ich", unterlagen: "Unterlagen" },
      append: ["früh."],
    },
    {
      key: "sq",
      title: "Shënime",
      glyph: "doc",
      label: "Albanian, in a document.",
      result: "On the processor there is no draft for Albanian: the text appears when you let go.",
      speakMs: 3400,
      processMs: 2600,
      chunks: [],
      drop: [],
      fix: {},
      append: ["Faleminderit", "për", "ndihmën,", "shihemi", "nesër."],
    },
  ];

  function setupDemo() {
    const stage = document.querySelector("[data-demo]");
    const dock = document.querySelector(".pill-dock");
    const hero = document.querySelector(".hero");
    if (!stage || !dock || !hero) return;
    const pill = dock.querySelector("[data-pill]");
    const pillText = dock.querySelector("[data-pill-text]");
    const bars = Array.from(dock.querySelectorAll("[data-meter] i"));
    const title = stage.querySelector("[data-demo-title]");
    const glyph = stage.querySelector("[data-demo-glyph]");
    const keys = stage.querySelector("[data-demo-keys]");
    const note = stage.querySelector("[data-demo-note]");
    const toggle = stage.querySelector("[data-demo-toggle]");
    const scenes = Array.from(stage.querySelectorAll("[data-scene]"));
    const staticNote = note.innerHTML;
    const staticField = scenes[0].querySelector("[data-field]").innerHTML;

    const smoother = makeSmoother();
    let clock = 0;
    let last = null;
    let userPaused = false;
    let onScreen = true;
    let speaking = false;
    let meterOn = false;
    let timers = [];
    let generation = 0;
    let rafId = 0;
    let running = false;
    let stepCarry = 0;

    const active = () => !userPaused && onScreen && !document.hidden;

    const wait = (ms, gen) =>
      new Promise((resolve, reject) => {
        timers.push({ at: clock + ms, resolve, reject, gen });
      });

    const setBars = (values) => {
      for (let i = 0; i < BAR_COUNT; i += 1) {
        const h = BAR_MIN + (BAR_MAX - BAR_MIN) * values[i];
        bars[i].style.setProperty("--h", h.toFixed(2));
      }
    };

    const setStaticBars = () => {
      bars.forEach((bar, i) => bar.style.setProperty("--h", String(STATIC_POSE[i])));
    };

    const speechLevel = (t) => {
      const syllable = Math.abs(Math.sin(t / 105));
      const phrase = 0.72 + 0.28 * Math.sin(t / 530 + 1.3);
      const gap = Math.sin(t / 900) > 0.93 ? 0.25 : 1;
      return (0.34 + 0.62 * syllable) * phrase * gap;
    };

    const frame = (ts) => {
      rafId = requestAnimationFrame(frame);
      if (last === null) last = ts;
      const dt = Math.min(64, ts - last);
      last = ts;
      if (!active()) return;
      clock += dt;
      if (meterOn) {
        stepCarry += dt;
        let values = null;
        while (stepCarry >= FRAME_MS) {
          stepCarry -= FRAME_MS;
          values = smoother.step(speaking ? speechLevel(clock) : 0.03);
        }
        if (values) setBars(values);
      }
      const due = timers.filter((timer) => timer.at <= clock);
      if (due.length) {
        timers = timers.filter((timer) => timer.at > clock);
        due.forEach((timer) => timer.resolve());
      }
    };

    const showScene = (scene) => {
      scenes.forEach((el) => el.classList.toggle("is-active", el.dataset.scene === scene.key));
      title.textContent = scene.title;
      glyph.dataset.demoGlyph = scene.glyph;
      const field = stage.querySelector(`[data-scene="${scene.key}"] [data-field]`);
      field.textContent = "";
      const caret = document.createElement("span");
      caret.className = "caret";
      field.append(caret);
      return { field, caret };
    };

    const setNote = (scene, withResult) => {
      note.textContent = "";
      const strong = document.createElement("strong");
      strong.textContent = scene.label;
      note.append(strong);
      if (withResult) note.append(" " + scene.result);
    };

    const typeWords = async (field, caret, words, gen, draft) => {
      const made = [];
      for (const word of words) {
        const tok = document.createElement("span");
        tok.className = draft ? "tok is-draft" : "tok";
        tok.dataset.word = word;
        field.insertBefore(tok, caret);
        const needsSpace = field.querySelectorAll(".tok").length > 1;
        const text = (needsSpace ? " " : "") + word;
        for (let i = 1; i <= text.length; i += 1) {
          tok.textContent = text.slice(0, i);
          await wait(14, gen);
        }
        made.push(tok);
        await wait(22, gen);
      }
      return made;
    };

    const setPill = (state, text) => {
      pill.dataset.state = state;
      pillText.textContent = text || "";
    };

    const runScene = async (scene, gen) => {
      const { field, caret } = showScene(scene);
      setNote(scene, false);
      await wait(900, gen);

      keys.classList.add("is-down");
      smoother.reset();
      setBars(new Array(BAR_COUNT).fill(0));
      setPill("listening");
      meterOn = true;
      pill.classList.add("is-shown");
      await wait(160, gen);
      speaking = true;
      const start = clock;

      for (const chunk of scene.chunks) {
        const delay = chunk.at - (clock - start);
        if (delay > 0) await wait(delay, gen);
        await typeWords(field, caret, chunk.words, gen, true);
      }
      const rest = scene.speakMs - (clock - start);
      if (rest > 0) await wait(rest, gen);

      speaking = false;
      keys.classList.remove("is-down");
      await wait(90, gen);
      meterOn = false;
      setPill("processing", "Processing");
      await wait(scene.processMs, gen);

      const tokens = Array.from(field.querySelectorAll(".tok"));
      tokens.forEach((tok) => {
        const word = tok.dataset.word;
        if (scene.drop.includes(word)) {
          tok.classList.add("is-drop");
        } else if (scene.fix[word]) {
          tok.textContent = tok.textContent.replace(word, scene.fix[word]);
          tok.classList.add("is-fix");
        }
      });
      tokens.forEach((tok) => tok.classList.remove("is-draft"));
      pill.classList.remove("is-shown");
      await typeWords(field, caret, scene.append, gen, false);
      const kept = tokens.filter((tok) => !tok.classList.contains("is-drop"));
      const first = kept[0] || field.querySelector(".tok");
      if (first && first.textContent.startsWith(" ")) first.textContent = first.textContent.slice(1);

      const spark = document.createElement("span");
      spark.className = "spark";
      field.insertBefore(spark, caret);
      setNote(scene, true);
      await wait(700, gen);
      field.querySelectorAll(".is-fix").forEach((tok) => tok.classList.remove("is-fix"));
      await wait(500, gen);
      tokens.filter((tok) => tok.classList.contains("is-drop")).forEach((tok) => tok.remove());
      spark.remove();
      await wait(2600, gen);
    };

    const loop = async (gen) => {
      let index = 0;
      await wait(1400, gen);
      while (gen === generation) {
        await runScene(SCENES[index], gen);
        if (gen !== generation) return;
        index = (index + 1) % SCENES.length;
      }
    };

    const restoreStatic = () => {
      scenes.forEach((el) => el.classList.toggle("is-active", el.dataset.scene === "en"));
      title.textContent = SCENES[0].title;
      glyph.dataset.demoGlyph = SCENES[0].glyph;
      scenes[0].querySelector("[data-field]").innerHTML = staticField;
      note.innerHTML = staticNote;
      keys.classList.remove("is-down");
      setPill("listening");
      pill.classList.add("is-shown");
      setStaticBars();
    };

    const start = () => {
      if (running) return;
      running = true;
      generation += 1;
      timers = [];
      toggle.hidden = false;
      pill.classList.remove("is-shown");
      last = null;
      rafId = requestAnimationFrame(frame);
      loop(generation).catch(() => {});
    };

    const stop = () => {
      if (!running) return;
      running = false;
      generation += 1;
      timers = [];
      cancelAnimationFrame(rafId);
      toggle.hidden = true;
      userPaused = false;
      hero.classList.remove("is-paused");
      toggle.setAttribute("aria-pressed", "false");
      toggle.textContent = "Pause";
      restoreStatic();
    };

    toggle.addEventListener("click", () => {
      userPaused = !userPaused;
      toggle.setAttribute("aria-pressed", String(userPaused));
      toggle.textContent = userPaused ? "Play" : "Pause";
      hero.classList.toggle("is-paused", userPaused);
    });

    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            onScreen = entry.isIntersecting;
          });
        },
        { threshold: 0.05 }
      );
      observer.observe(stage);
    }

    const decide = () => {
      if (reduceMotion.matches) stop();
      else start();
    };
    reduceMotion.addEventListener("change", decide);
    decide();
  }

  setupTabs();
  setupDemo();
})();
