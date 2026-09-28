// Lockstep shim, installed in the page before any game script runs.
//
// A browser game keeps its own time: p5 (and three.js) redraw from
// requestAnimationFrame and read performance.now() / Date.now() for deltaTime,
// and the games schedule "3 s later" transitions with setTimeout. To make the
// page a gym env -- one step() advances the game by exactly N frames, whether
// the caller takes 1 ms or 30 s to decide -- all of those are replaced here by
// a clock that only moves when the host says so. Math.random is replaced by a
// seeded generator (the seed is the page's `?seed=` query parameter), so a
// reload with the same seed and the same key sequence replays the same game.
//
// Host API (window.__aigs):
//   boot(maxFrames)  -> tick until the game has a canvas and a state, or throw
//   step(names, n)   -> hold exactly the named keys, tick n frames, return
//                       {png: dataURL of the canvas, state: scalar game state}
(() => {
  const FRAME_MS = 1000 / 60;   // p5's frameRate(60) and rAF's nominal rate
  let now = 0;                  // fake performance.now(), ms
  let timers = [];              // {id, at, fn, args, every}
  let rafs = [];                // {id, fn}
  let nextId = 1;
  const epoch = 1_700_000_000_000; // a fixed wall clock, so Date.now() replays too

  // ---- clock -------------------------------------------------------------
  performance.now = () => now;
  const RealDate = Date;
  window.Date = class extends RealDate {
    constructor(...args) { args.length ? super(...args) : super(epoch + Math.floor(now)); }
    static now() { return epoch + Math.floor(now); }
  };
  window.setTimeout = (fn, ms = 0, ...args) => {
    const id = nextId++;
    timers.push({ id, at: now + Math.max(0, +ms || 0), fn, args, every: null });
    return id;
  };
  window.setInterval = (fn, ms = 0, ...args) => {
    const id = nextId++, every = Math.max(1, +ms || 0);
    timers.push({ id, at: now + every, fn, args, every });
    return id;
  };
  window.clearTimeout = window.clearInterval = (id) => { timers = timers.filter(t => t.id !== id); };
  window.requestAnimationFrame = (fn) => { const id = nextId++; rafs.push({ id, fn }); return id; };
  window.cancelAnimationFrame = (id) => { rafs = rafs.filter(r => r.id !== id); };

  function tick() {
    now += FRAME_MS;
    // Due timers fire in order of their deadline; one scheduled by another
    // that is also due fires in this same frame, as a browser would.
    for (;;) {
      let i = -1;
      timers.forEach((t, j) => { if (t.at <= now && (i < 0 || t.at < timers[i].at)) i = j; });
      if (i < 0) break;
      const [t] = timers.splice(i, 1);
      if (t.every) { t.at += t.every; timers.push(t); }
      t.fn(...t.args);
    }
    // Callbacks a frame requests run in the next frame, not this one.
    const due = rafs;
    rafs = [];
    for (const r of due) r.fn(now);
  }

  // ---- randomness --------------------------------------------------------
  // mulberry32: small, fast, good enough to replay a game.
  let state = (Number(new URLSearchParams(location.search).get("seed")) || 1) >>> 0;
  Math.random = () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  // ---- keyboard ----------------------------------------------------------
  // The keys the ten games listen for (they switch on e.keyCode), by the
  // upper-case pygame-style names the host uses: [keyCode, key, code].
  const KEYS = {
    LEFT: [37, "ArrowLeft", "ArrowLeft"], RIGHT: [39, "ArrowRight", "ArrowRight"],
    UP: [38, "ArrowUp", "ArrowUp"], DOWN: [40, "ArrowDown", "ArrowDown"],
    SPACE: [32, " ", "Space"], RETURN: [13, "Enter", "Enter"], ESCAPE: [27, "Escape", "Escape"],
    LSHIFT: [16, "Shift", "ShiftLeft"],
    W: [87, "w", "KeyW"], A: [65, "a", "KeyA"], S: [83, "s", "KeyS"], D: [68, "d", "KeyD"],
    Z: [90, "z", "KeyZ"], X: [88, "x", "KeyX"], R: [82, "r", "KeyR"],
    1: [49, "1", "Digit1"], 2: [50, "2", "Digit2"], 3: [51, "3", "Digit3"],
    4: [52, "4", "Digit4"], 5: [53, "5", "Digit5"],
  };
  let held = new Set();

  function sendKey(type, name) {
    const [keyCode, key, code] = KEYS[name];
    const e = new KeyboardEvent(type, { key, code, keyCode, which: keyCode, bubbles: true, cancelable: true });
    // p5 switches on which/keyCode, which the constructor may not honour.
    Object.defineProperty(e, "keyCode", { get: () => keyCode });
    Object.defineProperty(e, "which", { get: () => keyCode });
    window.dispatchEvent(e);      // p5 and the three.js game both listen on window
  }

  // ---- host API ----------------------------------------------------------
  function scalars(obj) {
    const out = {};
    for (const [k, v] of Object.entries(obj || {})) {
      if (["number", "string", "boolean"].includes(typeof v)) out[k] = v;
    }
    return out;
  }

  function ready() {
    if (!document.querySelector("canvas") || typeof window.getGameState !== "function") return false;
    const s = window.getGameState();
    return !!(s && typeof s === "object" && "gamePhase" in s);
  }

  window.__aigs = {
    keys: Object.keys(KEYS),
    boot(maxFrames) {
      for (let i = 0; i < maxFrames && !ready(); i++) tick();
      if (!ready()) throw new Error(`game did not come up within ${maxFrames} frames: `
        + `no <canvas>, or no window.getGameState() with a gamePhase`);
    },
    step(names, frames) {
      const want = new Set(names);
      for (const n of held) if (!want.has(n)) sendKey("keyup", n);
      for (const n of want) if (!held.has(n)) sendKey("keydown", n);
      held = want;
      for (let i = 0; i < frames; i++) tick();
      // Read the pixels in the same task as the draw: a WebGL canvas is
      // cleared once the browser composites it.
      const canvas = document.querySelector("canvas");
      return { png: canvas.toDataURL("image/png"), state: scalars(window.getGameState()) };
    },
  };
})();
