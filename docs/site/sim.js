/* qwenTPU array semantics in plain JS (the datapath, not the transformer).
   Mirrors qwentpu/circuit.py: W heads hold weight bits, ACT(i,v) ANDs activation bits with row i of every
   W head and writes row i of the accumulator heads, READ(c) is a multi-hot sum of rows (bit-plane counts),
   then a Horner clamp chain (adder tree) turns counts into the requantised output. */
(function (root) {
  "use strict";
  /* g activations per ACT token, h outputs per READ/OUT token (token packing); g=h=1 is the plain array */
  function cfg(kr, nc, ab, wb, g, h) {
    g = g || 1; h = h || 1;
    const nout = 1 << ab, ks = kr / g, nread = nc / h, nw = 1 << wb;
    const nop = Math.pow(nout, h), cell0 = nop + 1, act0 = cell0 + kr * nc * nw, nact = ks * Math.pow(nout, g), read0 = act0 + nact;
    return { kr, nc, ab, wb, g, h, ks, nread, nout, nw, nop, cell0, act0, nact, read0, vocab: read0 + nread,
      amax: (1 << ab) - 1, wlo: -(1 << (wb - 1)), whi: (1 << (wb - 1)) - 1,
      tokensPerMatvec: ks + nread, macsPerToken: kr * nc / (ks + nread) };
  }
  const tokOut = (c, vs) => vs.reduce((a, v, t) => a + v * Math.pow(c.nout, t), 0);
  const tokAct = (c, s, vs) => c.act0 + s * Math.pow(c.nout, c.g) + vs.reduce((a, v, j) => a + v * Math.pow(c.nout, j), 0);
  const tokRead = (c, k) => c.read0 + k;
  const tokCell = (c, r, col, w) => c.cell0 + (r * c.nc + col) * c.nw + (w & (c.nw - 1));
  function sBounds(c) { return { smin: -(1 << (c.wb - 1)) * c.kr, smax: ((1 << (c.wb - 1)) - 1) * c.kr }; }
  function ulo(c, p) { const { smax } = sBounds(c); return Math.floor(-(smax * ((1 << p) - 1)) / (1 << p)) + 0; }
  function uhi(c, p) { const { smin } = sBounds(c); return Math.ceil((c.amax - smin * ((1 << p) - 1)) / (1 << p)) + 0; }
  const wt = (c, b) => (b === c.wb - 1 ? -(1 << b) : (1 << b));

  /* Horner plan: a line-for-line mirror of circuit.clamp_passes() / horner_plan() / fits_code().
     Top-down, q digits per step: the largest q whose clamp needs at most one 256-wide FFN pass, else q=1 with
     several lossless passes (each clamps to a partial bound <= 256 beyond the current range edge). The last step
     stops early once the code thresholds can be read straight off the remaining range (fits_code). */
  const WINDOW = 256;
  function fitsCode(c, zlo, zhi) {
    for (let t = 1; t <= c.amax; t++) if (!((zhi - t + 1 <= WINDOW) || (t - zlo <= WINDOW))) return false;
    return true;
  }
  function clampPasses(zlo, zhi, L, H, stop) {
    const out = []; let guard = 0;
    while ((zlo < L || zhi > H) && !(stop && stop(zlo, zhi)) && guard++ < 1000) {
      let lo = L - zlo <= WINDOW ? L : zlo + WINDOW, hi = zhi - H <= WINDOW ? H : zhi - WINDOW;
      lo = zlo < L ? lo : zlo; hi = zhi > H ? hi : zhi;
      out.push([lo, hi]); zlo = Math.max(zlo, lo); zhi = Math.min(zhi, hi);
    }
    return { passes: out, lo: zlo, hi: zhi };
  }
  function plan(c) {
    const { smin, smax } = sBounds(c), stages = [], fc = (a, b) => fitsCode(c, a, b);
    let p = c.ab, lo = 0, hi = 0, maxWin = 0, total = 0;
    while (p > 0) {
      let q, zlo, zhi, L, H, found = false;
      for (q = p; q >= 1; q--) {
        zlo = lo * (1 << q) + smin * ((1 << q) - 1); zhi = hi * (1 << q) + smax * ((1 << q) - 1);
        L = ulo(c, p - q); H = uhi(c, p - q);
        if ((p - q === 0 && fc(zlo, zhi)) || clampPasses(zlo, zhi, L, H).passes.length <= 1) { found = true; break; }
      }
      if (!found) { q = 1; zlo = lo * 2 + smin; zhi = hi * 2 + smax; }
      L = ulo(c, p - q); H = uhi(c, p - q);
      const merged = p - q === 0 && fc(zlo, zhi);
      const r = merged ? { passes: [], lo: zlo, hi: zhi } : clampPasses(zlo, zhi, L, H, p - q === 0 ? fc : null);
      let cl = zlo, ch = zhi, win = 0;
      for (const [a, b] of r.passes) { win = Math.max(win, a - cl, ch - b, 0); cl = Math.max(cl, a); ch = Math.min(ch, b); }
      stages.push({ p, q, p2: p - q, zlo, zhi, lo: L, hi: H, merged, passes: r.passes, npass: r.passes.length, endLo: r.lo, endHi: r.hi, win });
      maxWin = Math.max(maxWin, win); total += r.passes.length; p -= q; lo = r.lo; hi = r.hi;
    }
    return { stages, maxWin, ok: true, error: null, totalPasses: total };
  }
  /* n_q: Qwen l2-normalises q and scales by 1/sqrt(128): exact power of two iff n_q*128 is a power of 4 */
  const qrows = (c) => ((Math.log2(c.ks) % 2) === 1 ? c.ks : 2 * c.ks);

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
      c, wbits: Array.from({ length: c.nc }, () => Array.from({ length: c.g }, () => Array(c.ks).fill(null))),   // [col][copy j][state row s]
      wval: Array.from({ length: c.kr }, () => Array(c.nc).fill(null)),
      acc: Array.from({ length: c.nc }, () => Array.from({ length: c.g }, () => Array(c.ks).fill(null))),
      row: -1, readK: -1, slots: null, u: null, stagesDone: [], out: Array(c.nc).fill(null), ok: Array(c.nc).fill(null),
      x: null, ref: null, layer: 0, cell: null, note: "", phase: "idle", spent: 0, actVals: null, lastTok: null,
    };
  }

  /* execute one micro-step on the state: the actual array semantics */
  function exec(st, s, P) {
    const c = st.c; st.note = s.note || ""; st.phase = s.kind; if (s.tok) st.lastTok = s.tok;
    switch (s.kind) {
      case "load": st.wbits[s.col][s.r % c.g][Math.floor(s.r / c.g)] = bits2c(s.w, c.wb); st.wval[s.r][s.col] = s.w; st.cell = [s.r, s.col]; st.row = -1; break;
      case "begin": st.layer = s.layer; st.x = s.x; st.ref = s.ref; st.out = Array(c.nc).fill(null); st.ok = Array(c.nc).fill(null); st.cell = null; st.row = -1; st.readK = -1; st.slots = null; st.stagesDone = []; st.spent = 0; st.actVals = null; break;
      case "act":   // ONE token = g activations: copy j of every W head reads state row s (= weight row s*g+j), ANDs act bits of v_j, writes row s of acc copy j
        for (let k = 0; k < c.nc; k++) for (let j = 0; j < c.g; j++) {
          const wbk = st.wbits[k][j][s.s];
          st.acc[k][j][s.s] = Array.from({ length: c.ab * c.wb }, (_, n) => ((s.vs[j] >> Math.floor(n / c.wb)) & 1) & wbk[n % c.wb]);
        }
        st.row = s.s; st.cell = null; st.actVals = s.vs; st.spent++; break;
      case "read": {   // ONE token = h columns: multi-hot query over every written row of the g acc copies of each column = exact sums
        st.slots = Array.from({ length: c.h }, (_, t) => {
          const col = s.k * c.h + t, cnt = Array.from({ length: c.ab }, () => Array(c.wb).fill(0));
          for (let j = 0; j < c.g; j++) for (let i = 0; i < c.ks; i++) { const r = st.acc[col][j][i]; if (r) for (let n = 0; n < r.length; n++) cnt[Math.floor(n / c.wb)][n % c.wb] += r[n]; }
          return { col, cnt, s: cnt.map((rowc) => rowc.reduce((a, v, b) => a + v * wt(c, b), 0)) };   // s_a = sum_b wt_b cnt[a][b]
        });
        st.readK = s.k; st.row = -1; st.u = Array(c.h).fill(0); st.stagesDone = []; st.spent++; break; }
      case "horner": {   // every slot t has its own chain: Z = 2^q u + digits ; passes ; (last step ends in the code FFN thresholds)
        const g = P.stages[s.k], Zs = [], us = [];
        for (let t = 0; t < c.h; t++) {
          let Z = (1 << g.q) * st.u[t]; for (let i = g.p2; i < g.p; i++) Z += (1 << (i - g.p2)) * st.slots[t].s[i];
          let u = Z, cl = g.zlo, ch = g.zhi;
          for (const [a, b] of g.passes) { u = u + (a > cl ? relu(a - u) : 0) - (b < ch ? relu(u - b) : 0); cl = Math.max(cl, a); ch = Math.min(ch, b); }
          Zs.push(Z); us.push(g.p2 === 0 ? Math.min(Math.max(u, 0), c.amax) : u);
        }
        st.stagesDone.push({ ...g, Z: Zs, u: us, clamped: Zs.some((z, t) => z !== us[t]) }); st.u = us; break; }
      case "out": for (let t = 0; t < c.h; t++) { const col = s.k * c.h + t; st.out[col] = st.u[t]; st.ok[col] = st.u[t] === st.ref[col]; } break;
      default: break;
    }
  }

  /* build a token/step list for a chain of layers. layer L>0 consumes the outputs of layer L-1 (host relabel). */
  function build(c, Ws, x0) {
    const P = plan(c), steps = [];
    if (!P.ok) throw new Error(P.error);
    let x = x0.slice();
    const probe = newState(c), G = c.g > 1, Hh = c.h > 1;
    Ws.forEach((W, L) => {
      const ref = refMatvec(c, W, x), mk = (o) => steps.push(o), first = steps.length;
      mk({ kind: "begin", tok: null, layer: L, x: x.slice(), ref, note: L ? "the OUT tokens become the ACT tokens of the next layer: the host only relabels, no arithmetic" : "start" });
      for (let r = 0; r < c.kr; r++) for (let col = 0; col < c.nc; col++)
        mk({ kind: "load", tok: `CELL(${r},${col},${W[r][col]})`, id: tokCell(c, r, col, W[r][col]), r, col, w: W[r][col],
          note: `ONE token writes w=${W[r][col]} into state row ${Math.floor(r / c.g)} of weight-head copy ${r % c.g} of column ${col}${G ? ` (weight row ${r} = ${Math.floor(r / c.g)}·g + ${r % c.g})` : ""}: the key AND(row ${r}, col ${col}) is formed free by conv+silu, the value is the weight bits. Weights then stay put.` });
      for (let s = 0; s < c.ks; s++) {
        const vs = x.slice(s * c.g, s * c.g + c.g);
        mk({ kind: "act", tok: `ACT(${s}; ${vs.join(", ")})`, id: tokAct(c, s, vs), s, vs,
          note: `${G ? `ONE token carries ${c.g} activations (rows ${s * c.g}..${s * c.g + c.g - 1}). ` : ""}Query reads state row ${s} of ${G ? "all " + c.g + " weight-head copies of every column" : "every W head"}; bit-AND with the ${c.ab} activation bits${G ? " of its own value" : ""}; the products overwrite row ${s} of ${G ? "all " + c.g + " accumulator copies" : "every accumulator head"}. Every vector rewrites all ${c.ks} state rows, so there is nothing to flush.` });
      }
      for (let k = 0; k < c.nread; k++) {
        const cols = Array.from({ length: c.h }, (_, t) => k * c.h + t);
        mk({ kind: "read", tok: `READ(${k})`, id: tokRead(c, k), k,
          note: `${Hh ? `ONE token reads ${c.h} columns (${cols.join(", ")}) at once. ` : ""}Multi-hot query over the ${c.ks} written rows of ${G ? "the " + c.g + " accumulator copies of " : ""}${Hh ? "each column" : "column " + cols[0]} (padded to n_q=${qrows(c)} so Qwen's q-scale is a power of two): the exact sum of rows = bit-plane counts.${Hh ? " Each column has its own Horner chain (slot t = c mod h)." : ""}` });
        P.stages.forEach((g, i) => mk({ kind: "horner", tok: null, k: i,
          note: g.merged ? `code FFN (last Horner step merged): absorb digit${g.q > 1 ? "s" : ""} ${g.p2}..${g.p - 1}; Z in [${g.zlo}, ${g.zhi}] fits the window, so thresholds on Z give the output code directly`
            : `adder-tree step ${i + 1}/${P.stages.length}${Hh ? " (" + c.h + " slots in parallel)" : ""}: absorb ${g.q} digit${g.q > 1 ? "s" : ""} (${g.p2}..${g.p - 1}); Z in [${g.zlo}, ${g.zhi}], ${g.npass ? g.npass + " clamp pass" + (g.npass > 1 ? "es" : "") + " (one FFN each) to [" + g.endLo + ", " + g.endHi + "]" : "range already fits: no clamp"}${g.p2 === 0 ? "; code FFN thresholds finish the job" : ""}` }));
        mk({ kind: "out", tok: "OUT(?)", k, note: Hh ? `the model samples ONE OUT token carrying ${c.h} outputs (id = Σ v_t·${c.nout}^t)` : "the model samples OUT(v): code FFN → lm_head argmax" });
      }
      steps.slice(first).forEach((s) => { exec(probe, s, P); if (s.kind === "out") { const vs = Array.from({ length: c.h }, (_, t) => probe.out[s.k * c.h + t]); s.tok = `OUT(${vs.join(", ")})`; s.id = tokOut(c, vs); } });
      x = probe.out.slice(0, c.kr);
      if (L + 1 < Ws.length && c.nc < c.kr) throw new Error("chain needs nc >= kr");
    });
    return { steps, plan: P };
  }

  /* run whole thing without display */
  function run(c, Ws, x0) {
    const { steps, plan: P } = build(c, Ws, x0), st = newState(c); let fails = 0, outs = [];
    for (const s of steps) { exec(st, s, P); if (s.kind === "out") for (let t = 0; t < c.h; t++) { const col = s.k * c.h + t; if (st.ok[col] === false) fails++; outs.push(st.out[col]); } }
    return { fails, outs, steps, plan: P };
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

  const api = { cfg, tokOut, tokAct, tokRead, tokCell, plan, fitsCode, clampPasses, qrows, ulo, uhi, sBounds, wt, refMatvec, newState, exec, build, run, randCase, bits2c };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.QSim = api;
})(typeof window !== "undefined" ? window : globalThis);
