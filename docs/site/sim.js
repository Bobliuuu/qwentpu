/* qwenTPU array semantics in plain JS (the datapath, not the transformer).
   Mirrors qwentpu/circuit.py: W heads hold weight bits, ACT(i,v) ANDs activation bits with row i of every
   W head and writes row i of the accumulator heads, READ(c) is a multi-hot sum of rows (bit-plane counts),
   then a Horner clamp chain (adder tree) turns counts into the requantised output. */
(function (root) {
  "use strict";
  function cfg(kr, nc, ab, wb) { return { kr, nc, ab, wb, amax: (1 << ab) - 1, wlo: -(1 << (wb - 1)), whi: (1 << (wb - 1)) - 1 }; }
  function sBounds(c) { return { smin: -(1 << (c.wb - 1)) * c.kr, smax: ((1 << (c.wb - 1)) - 1) * c.kr }; }
  function ulo(c, p) { const { smax } = sBounds(c); return Math.floor(-(smax * ((1 << p) - 1)) / (1 << p)) + 0; }
  function uhi(c, p) { const { smin } = sBounds(c); return Math.ceil((c.amax - smin * ((1 << p) - 1)) / (1 << p)) + 0; }
  const wt = (c, b) => (b === c.wb - 1 ? -(1 << b) : (1 << b));

  /* Horner plan, a line-for-line mirror of circuit.horner_plan() / fits_code():
     top-down, q digits per step (largest q whose clamp windows fit 256); the last step is merged into the
     output-code FFN when the code thresholds can be read straight off the unclamped Z. */
  const WINDOW = 256;
  function fitsCode(c, zlo, zhi) {
    for (let t = 1; t <= c.amax; t++) if (!((zhi - t + 1 <= WINDOW) || (t - zlo <= WINDOW))) return false;
    return true;
  }
  function plan(c) {
    const { smin, smax } = sBounds(c), stages = [];
    let p = c.ab, lo = 0, hi = 0, maxWin = 0, error = null;
    while (p > 0) {
      let found = null;
      for (let q = p; q >= 1; q--) {
        const zlo = lo * (1 << q) + smin * ((1 << q) - 1), zhi = hi * (1 << q) + smax * ((1 << q) - 1);
        const L = ulo(c, p - q), H = uhi(c, p - q);
        const merged = p - q === 0 && fitsCode(c, zlo, zhi);
        if (merged || (Math.max(0, L - zlo) <= WINDOW && Math.max(0, zhi - H) <= WINDOW)) { found = { q, zlo, zhi, L, H, merged }; break; }
      }
      if (!found) { error = `adder tree does not fit the bf16 windows at p=${p}`; break; }
      const { q, zlo, zhi, L, H, merged } = found;
      const win = merged ? 0 : Math.max(0, L - zlo, zhi - H);
      stages.push({ p, q, p2: p - q, zlo, zhi, lo: L, hi: H, merged, clampLo: !merged && zlo < L, clampHi: !merged && zhi > H, win });
      maxWin = Math.max(maxWin, win); p -= q; lo = Math.max(zlo, L); hi = Math.min(zhi, H);
    }
    return { stages, maxWin, ok: !error, error };
  }
  /* n_q: Qwen l2-normalises q and scales by 1/sqrt(128): exact power of two iff n_q*128 is a power of 4 */
  const qrows = (c) => ((Math.log2(c.kr) % 2) === 1 ? c.kr : 2 * c.kr);

  const relu = (x) => (x > 0 ? x : 0);
  const bits2c = (w, wb) => Array.from({ length: wb }, (_, b) => ((w & ((1 << wb) - 1)) >> b) & 1);

  function refMatvec(c, W, x) {   // W[r][col], independent of the bit-plane pipeline
    return Array.from({ length: c.nc }, (_, col) => {
      let y = 0; for (let r = 0; r < c.kr; r++) y += x[r] * W[r][col];
      return Math.min(Math.max(y, 0), c.amax);
    });
  }

  function newState(c) {
    return {
      c, wbits: Array.from({ length: c.nc }, () => Array(c.kr).fill(null)),
      wval: Array.from({ length: c.kr }, () => Array(c.nc).fill(null)),
      acc: Array.from({ length: c.nc }, () => Array(c.kr).fill(null)),
      row: -1, readCol: -1, cnt: null, s: null, u: null, stagesDone: [], out: Array(c.nc).fill(null), ok: Array(c.nc).fill(null),
      x: null, ref: null, layer: 0, cell: null, note: "", phase: "idle",
    };
  }

  /* execute one micro-step on the state: the actual array semantics */
  function exec(st, s, P) {
    const c = st.c; st.note = s.note || ""; st.phase = s.kind;
    switch (s.kind) {
      case "load": st.wbits[s.col][s.r] = bits2c(s.w, c.wb); st.wval[s.r][s.col] = s.w; st.cell = [s.r, s.col]; st.row = -1; break;
      case "begin": st.layer = s.layer; st.x = s.x; st.ref = s.ref; st.out = Array(c.nc).fill(null); st.ok = Array(c.nc).fill(null); st.cell = null; st.row = -1; st.readCol = -1; st.cnt = null; st.stagesDone = []; break;
            case "act":
        for (let k = 0; k < c.nc; k++) {
          const wbk = st.wbits[k][s.i];
          st.acc[k][s.i] = Array.from({ length: c.ab * c.wb }, (_, n) => ((s.v >> Math.floor(n / c.wb)) & 1) & wbk[n % c.wb]);  // AND products
        }
        st.row = s.i; st.cell = null; break;
      case "read": {                         // multi-hot query over all rows of accumulator head col = exact sum of rows
        const cnt = Array.from({ length: c.ab }, () => Array(c.wb).fill(0));
        for (let i = 0; i < c.kr; i++) { const r = st.acc[s.col][i]; if (r) for (let n = 0; n < r.length; n++) cnt[Math.floor(n / c.wb)][n % c.wb] += r[n]; }
        st.readCol = s.col; st.cnt = cnt; st.row = -1;
        st.s = cnt.map((rowc) => rowc.reduce((a, v, b) => a + v * wt(c, b), 0));   // s_j = sum_b wt_b cnt[j][b]  (a free linear form)
        st.u = 0; st.stagesDone = []; break; }
      case "horner": {                       // Z = 2^q u + sum 2^(i-p2) s_i ; u' = Z + relu(lo-Z) - relu(Z-hi)
        const g = P.stages[s.k]; let Z = (1 << g.q) * st.u;
        for (let i = g.p2; i < g.p; i++) Z += (1 << (i - g.p2)) * st.s[i];
        // merged last step: the code FFN thresholds read relu+clamp(Z) straight off Z; otherwise Z + relu(lo-Z) - relu(Z-hi)
        const u2 = g.merged ? Math.min(Math.max(Z, 0), c.amax) : Z + (g.clampLo ? relu(g.lo - Z) : 0) - (g.clampHi ? relu(Z - g.hi) : 0);
        st.stagesDone.push({ ...g, Z, u: u2, uPrev: st.u, clamped: u2 !== Z }); st.u = u2; break; }
      case "out": st.out[s.col] = st.u; st.ok[s.col] = st.u === st.ref[s.col]; break;
      default: break;
    }
  }

  /* build a token/step list for a chain of layers (1 or 2). layers = [{W, x?}] ; x of layer>0 = previous outputs */
  function build(c, Ws, x0) {
    const P = plan(c), steps = [];
    if (!P.ok) throw new Error(P.error);
    let x = x0.slice();
    const probe = newState(c);
    Ws.forEach((W, L) => {
      const ref = refMatvec(c, W, x);
      const mk = (o) => steps.push(o);
      mk({ kind: "begin", tok: null, layer: L, x: x.slice(), ref, note: L ? "OUT(v) tokens become ACT(i,v) of the next layer: host only relabels, no arithmetic" : "start" });
      for (let r = 0; r < c.kr; r++) for (let col = 0; col < c.nc; col++)
        mk({ kind: "load", tok: `CELL(${r},${col},${W[r][col]})`, r, col, w: W[r][col], note: `ONE token writes w=${W[r][col]} into row ${r} of W head ${col}: the key AND(row ${r}, col ${col}) is formed free by conv+silu, the value is the weight bits. Weights then stay put.` });
      for (let i = 0; i < c.kr; i++)
        mk({ kind: "act", tok: `ACT(${i},${x[i]})`, i, v: x[i], note: `one-hot query reads row ${i} of every W head; bit-AND with the ${c.ab} activation bits; the products overwrite row ${i} of every accumulator head. Every vector rewrites all ${c.kr} rows, so there is nothing to flush.` });
      for (let col = 0; col < c.nc; col++) {
        mk({ kind: "read", tok: `READ(${col})`, col, note: `multi-hot query over the ${c.kr} written rows of accumulator ${col} (padded to n_q=${qrows(c)} so Qwen's q-scale is a power of two): one read returns the exact sum of rows = bit-plane counts` });
        P.stages.forEach((g, k) => mk({ kind: "horner", tok: null, k, note: g.merged ? `code FFN (last Horner step merged): absorb digit${g.q > 1 ? "s" : ""} ${g.p2}..${g.p - 1}; Z in [${g.zlo}, ${g.zhi}] fits the window, so thresholds on Z give the output code directly` : `adder-tree FFN ${k + 1}/${P.stages.length}: absorb ${g.q} digit${g.q > 1 ? "s" : ""} (${g.p2}..${g.p - 1}); Z in [${g.zlo}, ${g.zhi}], ${g.clampLo || g.clampHi ? "clamp to [" + g.lo + ", " + g.hi + "] (" + (g.clampLo && g.clampHi ? "two" : "one") + " SwiGLU unit" + (g.clampLo && g.clampHi ? "s" : "") + ")" : "range already fits: no clamp"}` }));
        mk({ kind: "out", tok: "OUT(?)", col, note: "model samples OUT(v): code FFN → lm_head argmax" });
      }
      // run this layer to get its outputs
      steps.slice(steps.findIndex((s, i) => s.kind === "begin" && s.layer === L)).forEach((s) => { exec(probe, s, P); if (s.kind === "out") s.tok = `OUT(${probe.out[s.col]})`; });
      x = probe.out.slice(0, c.kr);
      if (L + 1 < Ws.length && c.nc < c.kr) throw new Error("chain needs nc >= kr");
    });
    // fill OUT labels after the fact (display only; semantics come from exec)
    return { steps, plan: P };
  }

  /* run whole thing without display */
  function run(c, Ws, x0) {
    const { steps, plan: P } = build(c, Ws, x0), st = newState(c); let fails = 0, outs = [];
    for (const s of steps) { exec(st, s, P); if (s.kind === "out" && st.ok[s.col] === false) fails++; if (s.kind === "out") outs.push(st.out[s.col]); }
    return { fails, outs, steps: steps.length, plan: P };
  }

  function randCase(c, rnd) {
    const r = rnd || Math.random;
    const ri = (lo, hi) => lo + Math.floor(r() * (hi - lo + 1));
    for (let t = 0; t < 200; t++) {
      const W = Array.from({ length: c.kr }, () => Array.from({ length: c.nc }, () => (r() < 0.35 ? 0 : ri(Math.max(c.wlo, -1), Math.min(c.whi, 2)))));
      const x = Array.from({ length: c.kr }, () => ri(0, Math.min(c.amax, 3)));
      const y = refMatvec(c, W, x), mid = y.filter((v) => v > 0 && v < c.amax).length;
      if (mid >= c.nc / 2) return { W, x };
      if (t === 199) return { W, x };
    }
  }

  const api = { cfg, plan, fitsCode, qrows, ulo, uhi, sBounds, wt, refMatvec, newState, exec, build, run, randCase, bits2c };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.QSim = api;
})(typeof window !== "undefined" ? window : globalThis);
