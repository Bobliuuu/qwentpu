/* qwenTPU site: stepper (uses sim.js) + data-driven charts (results.json) */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const fmt = (n) => (n == null ? "n/m" : n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 1 : 2).replace(/\.?0+$/, "") + "k" : String(n));
  const sci = (v) => (v == null ? "n/m" : v < 0.01 ? v.toExponential(1).replace("e-", "e−") : String(v));
  const NS = "http://www.w3.org/2000/svg";

  /* ---------- results ---------- */
  function loadResults() {
    if (window.RESULTS) return Promise.resolve(window.RESULTS);
    return fetch("data/results.json").then((r) => r.json());
  }
  function stats(R) {
    const h = R.headline || {}, f = (v, d) => (v == null ? "n/m" : v);
    const items = [
      [h.outputs_replayed != null ? h.outputs_replayed.toLocaleString("en-US") : "n/m", `outputs replayed, ${f(h.mismatches)} mismatches vs NumPy`],
      [h.peak_macs != null ? h.peak_macs.toLocaleString("en-US") + " MACs" : "n/m", `per matvec on a 32×32 array, ${f(h.peak_macs_per_token)} MACs/token`],
      [h.a100_peak_tok_s != null ? "up to " + Math.round(h.a100_peak_tok_s / 1000) + "k tok/s" : "n/m", "on one A100, batched replay"],
      [h.text_kl_after_chip != null ? "KL ≤ " + (Math.ceil(h.text_kl_after_chip * 1000) / 1000) : "n/m", "text after a TPU run vs no run at all"],
    ];
    $("stats").innerHTML = items.map(([n, l]) => `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
  }

  /* tiny SVG chart helpers */
  function el(tag, attrs, txt) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (txt != null) e.textContent = txt; return e; }
  function niceMax(v) { const p = Math.pow(10, Math.floor(Math.log10(v))); for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p; return 10 * p; }
  function barChart(opts) {  // groups: [{label, bars:[{v, color, pend, name}]}]
    const W = 360, H = 230, L = 38, R = 8, T = 10, B = 44, pw = W - L - R, ph = H - T - B;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.aria });
    const vals = opts.groups.flatMap((g) => g.bars.map((b) => b.v)).filter((v) => v != null && v > 0);
    if (!vals.length) { svg.appendChild(el("text", { x: 10, y: 20 }, "no data")); return svg; }
    const log = !!opts.log, lo = log ? Math.pow(10, Math.floor(Math.log10(Math.min(...vals)))) : 0;
    const hi = log ? Math.pow(10, Math.ceil(Math.log10(Math.max(...vals)))) : niceMax(Math.max(...vals));
    const Y = (v) => T + ph - (log ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : v / hi) * ph;
    const ticks = log ? Array.from({ length: Math.round(Math.log10(hi / lo)) + 1 }, (_, i) => lo * Math.pow(10, i)) : [0, 0.25, 0.5, 0.75, 1].map((f) => f * hi);
    ticks.forEach((t) => { svg.appendChild(el("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), class: "grid" })); svg.appendChild(el("text", { x: L - 5, y: Y(t) + 3, "text-anchor": "end" }, log ? sci(t) : fmt(t))); });
    const gw = pw / opts.groups.length;
    const defs = el("defs", {}); const pat = el("pattern", { id: "hatch" + opts.id, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    pat.appendChild(el("rect", { width: 6, height: 6, fill: "none" })); pat.appendChild(el("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: "var(--mute)", "stroke-width": 2 })); defs.appendChild(pat); svg.appendChild(defs);
    opts.groups.forEach((g, gi) => {
      const n = g.bars.length, bw = Math.min(26, (gw - 10) / n), x0 = L + gi * gw + (gw - bw * n - (n - 1) * 3) / 2;
      g.bars.forEach((b, bi) => {
        const x = x0 + bi * (bw + 3);
        if (b.v == null || b.v <= 0) { svg.appendChild(el("text", { x: x + bw / 2, y: T + ph - 4, "text-anchor": "middle" }, "n/m")); return; }
        const y = Y(b.v), h = T + ph - y;
        svg.appendChild(el("rect", { x, y, width: bw, height: Math.max(h, 1), rx: 3, fill: b.pend ? `url(#hatch${opts.id})` : b.color, stroke: b.pend ? "var(--mute)" : "none" }));
        svg.appendChild(el("text", { x: x + bw / 2, y: y - 3, "text-anchor": "middle", class: "v", "font-size": 9.5 }, opts.vfmt ? opts.vfmt(b.v) : fmt(b.v)));
      });
      const lab = el("text", { x: L + gi * gw + gw / 2, y: H - B + 14, "text-anchor": "middle" }); g.label.split("\n").forEach((ln, i) => { const t = el("tspan", { x: L + gi * gw + gw / 2, dy: i ? 12 : 0 }, ln); lab.appendChild(t); }); svg.appendChild(lab);
    });
    svg.appendChild(el("line", { x1: L, x2: W - R, y1: T + ph, y2: T + ph, stroke: "var(--mute)" }));
    return svg;
  }
  function legend(items) { const d = document.createElement("div"); d.className = "legend"; d.innerHTML = items.map(([c, t]) => `<span style="--c:${c}">${t}</span>`).join(""); return d; }
  function chartBox(title, sub, svg, leg) { const d = document.createElement("div"); d.className = "chart"; d.innerHTML = `<h3>${title}</h3><p>${sub}</p>`; d.appendChild(svg); if (leg) d.appendChild(leg); return d; }

  const ARR_COL = ["var(--teal)", "var(--violet)", "var(--amber)", "var(--rose)"];
  function speedChart(R, key, title, sub, id) {
    const sp = R.speed || [], arrays = [...new Set(sp.map((s) => s.array))], models = [...new Set(sp.map((s) => s.model))];
    const groups = models.map((m) => ({ label: m, bars: arrays.map((a, i) => { const e = sp.find((s) => s.model === m && s.array === a); return { v: e ? e[key] : null, color: ARR_COL[i % 4] }; }) }));
    return chartBox(title, sub, barChart({ id, aria: title, groups }), legend(arrays.map((a, i) => [ARR_COL[i % 4], a])));
  }
  function charts(R) {
    const C = $("charts"); C.innerHTML = "";
    const arr = R.arrays || [];
    C.appendChild(chartBox("MACs per token vs array size", "Tokens per matvec are KR + NC; MACs are KR × NC. Hatched = not yet measured.",
      barChart({ id: "a", aria: "MACs per token by array size", vfmt: (v) => v.toFixed(v < 10 ? 1 : 0),
        groups: arr.map((a) => ({ label: a.label.replace(" ", "\n"), bars: [{ v: a.macs_per_token, color: "var(--violet)", pend: a.pending }] })) })));
    C.appendChild(speedChart(R, "a100_b16", "A100 tokens/s, batch 16", "Replayed teacher-forced traces, one A100, bf16 + flash-linear-attention.", "s16"));
    C.appendChild(speedChart(R, "a100_b1", "A100 tokens/s, batch 1", "Larger arrays amortise the per-token overhead.", "s1"));
    const rs = R.restore || [];
    C.appendChild(chartBox("Text after a TPU run: KL vs no run", "Log scale, lower is better. Muting alone vs muting + cache restore.",
      barChart({ id: "k", aria: "KL after chip run", log: true, vfmt: sci,
        groups: rs.map((r) => ({ label: r.model, bars: [{ v: r.muting_first != null ? r.muting_first : r.muting_worst, color: "var(--rose)" }, { v: r.restore_worst, color: "var(--teal)" }] })) }),
      legend([["var(--rose)", "muting only, first text token"], ["var(--teal)", "muting + restore, worst token"]])));
    const m = R.models || [];
    $("models").innerHTML = `<tr><th>Model</th><th>Layers (Qwen + TPU)</th><th>Hidden</th><th>Value / key heads</th><th>Text KL vs base</th><th>Mismatches vs NumPy (replayed outputs)</th></tr>` +
      m.map((x) => `<tr><td><b>${x.model}</b></td><td>${x.layers}</td><td>${x.hidden}</td><td>${x.heads}</td><td>${x.text_kl}</td><td>${x.mismatches}</td></tr>`).join("");
  }

  /* ---------- stepper ---------- */
  const PRESETS = [
    { id: "4x4", label: "4×4 · u3 × s3", c: [4, 4, 3, 3] },
    { id: "8x8", label: "8×8 · u3 × s3 (default, 2 chip layers)", c: [8, 8, 3, 3] },
    { id: "8x16", label: "8×16 · u3 × s3", c: [8, 16, 3, 3] },
    { id: "16x16", label: "16×16 · u3 × s2", c: [16, 16, 3, 2] },
    { id: "8x8w", label: "8×8 · u4 × s4 (adder tree)", c: [8, 8, 4, 4] },
    { id: "16x16w", label: "16×16 · u4 × s4 (adder tree)", c: [16, 16, 4, 4] },
    { id: "8x8x", label: "8×8 · u5 × s5 (adder tree)", c: [8, 8, 5, 5] },
    { id: "32x32", label: "32×32 · u3 × s3 (1,024 MACs)", c: [32, 32, 3, 3] },
  ];
  let cfg, Ws, x0, built, st, idx = -1, timer = null, chained = false;
  const pre = $("preset"); PRESETS.forEach((p, i) => { const o = document.createElement("option"); o.value = i; o.textContent = p.label; pre.appendChild(o); }); pre.value = 1;

  function reset(keep) {
    stop();
    const p = PRESETS[+pre.value]; cfg = QSim.cfg(...p.c);
    chained = $("chain").checked;
    if (!keep) { const a = QSim.randCase(cfg); Ws = [a.W]; x0 = a.x; if (chained) Ws.push(QSim.randCase(cfg).W); }
    if (chained && Ws.length < 2) Ws.push(QSim.randCase(cfg).W);
    if (!chained) Ws = Ws.slice(0, 1);
    build(); buildInputs();
  }
  function build() {
    built = QSim.build(cfg, Ws, x0);
    $("tokens").innerHTML = built.steps.map((s, i) => s.tok ? `<span class="tok k-${s.kind === "load" ? "load" : s.kind}" data-i="${i}">${s.tok}</span>` : "").join("");
    idx = -1; st = QSim.newState(cfg); render();
  }
  function goto(i) {
    i = Math.max(-1, Math.min(built.steps.length - 1, i));
    if (i < idx) { st = QSim.newState(cfg); idx = -1; }
    while (idx < i) { idx++; QSim.exec(st, built.steps[idx], built.plan); }
    render();
  }
  function tokTarget() { return idx; }
  function next() { if (idx >= built.steps.length - 1) { stop(); return; } let n = idx + 1; goto(n); }
  function play() { if (timer) return stop(); if (idx >= built.steps.length - 1) goto(-1); $("play").textContent = "Pause"; const tick = () => { next(); if (timer) timer = setTimeout(tick, 1000 / +$("speed").value); }; timer = setTimeout(tick, 0); }
  function stop() { if (timer) clearTimeout(timer); timer = null; $("play").textContent = "Play"; }

  function bitsGrid(rows, cols, get, cls, rowHot) {
    let h = `<div class="bits ${cls || ""}" style="grid-template-columns:repeat(${cols},7px)">`;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { const v = get(r, c); h += `<i class="${v ? "on" : ""}${r === rowHot ? " act" : ""}"></i>`; }
    return h + "</div>";
  }
  function render() {
    const c = cfg, s = built.steps[idx] || {};
    $("bar").style.width = ((idx + 1) / built.steps.length * 100) + "%";
    const cur = built.steps[idx];
    document.querySelectorAll("#tokens .tok").forEach((t) => t.classList.toggle("cur", +t.dataset.i === idx));
    const curEl = document.querySelector(`#tokens .tok[data-i="${idx}"]`);
    if (curEl) { const box = $("tokens"); box.scrollLeft = curEl.offsetLeft - box.clientWidth / 2; }
    $("note").textContent = cur ? (st.layer ? "Layer 2 · " : (chained ? "Layer 1 · " : "")) + (cur.note || "") : `Press Play or step. Weights load once (${c.kr * c.nc} CELL tokens), then each vector costs ${c.kr + c.nc} tokens (${c.kr} ACT + ${c.nc} READ).`;
    // weights
    let t = `<table class="mat"><tr><th></th>${Array.from({ length: c.nc }, (_, j) => `<th>${j}</th>`).join("")}</tr>`;
    for (let r = 0; r < c.kr; r++) { t += `<tr><th>${r}</th>`; for (let j = 0; j < c.nc; j++) { const v = st.wval[r][j]; const hot = st.cell && st.cell[0] === r && st.cell[1] === j; t += `<td class="${v == null ? "empty" : ""}${hot ? " hot" : ""}${st.row === r ? " row" : ""}">${v == null ? "·" : v}</td>`; } t += "</tr>"; }
    $("wmat").innerHTML = t + "</table>";
    $("wlabel").textContent = `(${c.wb}-bit two's complement, ${c.wlo}..${c.whi})`;
    // x and outs
    const x = st.x || x0;
    $("xvec").innerHTML = x.map((v, i) => `<span class="o${st.row === i ? " ok" : ""}">${v}</span>`).join("") + `<span style="color:var(--mute);font-size:12px;align-self:center">${c.ab}-bit unsigned, 0..${c.amax}</span>`;
    $("outs").innerHTML = st.out.map((v, j) => `<span class="o${v == null ? "" : st.ok[j] ? " ok" : " bad"}" title="column ${j}">${v == null ? "·" : v}</span>`).join("");
    const done = st.out.filter((v) => v != null).length, bad = st.ok.filter((v) => v === false).length;
    $("verdict").innerHTML = bad ? `<b class="bad">✗ ${bad} mismatch</b>` : done === c.nc ? `<b class="ok">✓ all ${c.nc} outputs match the integer reference</b>` + (st.layer === 0 && chained ? ' <span style="color:var(--mute)">· OUT tokens now feed layer 2</span>' : "") : `<span style="color:var(--mute)">${done}/${c.nc} outputs read · the reference is clamp(relu(xᵀW), 0, ${c.amax})</span>`;
    // tiles
    $("tiles").innerHTML = Array.from({ length: c.nc }, (_, j) => {
      const wh = bitsGrid(c.kr, c.wb, (r, b) => st.wbits[j][r] && st.wbits[j][r][b], "", st.row);
      const ah = bitsGrid(c.kr, c.ab * c.wb, (r, n) => st.acc[j][r] && st.acc[j][r][n], "wide", st.row);
      return `<div class="tile${st.readCol === j ? " sel" : ""}"><h5><span>col ${j}</span><span>W | acc</span></h5><div style="display:flex;gap:6px">${wh}${ah}</div></div>`;
    }).join("");
    // reader
    if (st.cnt) {
      $("rcol").textContent = `column ${st.readCol}`;
      const P = built.plan;
      let h = `<table><tr><th>act bit j</th>${Array.from({ length: c.wb }, (_, b) => `<th>cnt·b${b}</th>`).join("")}<th>s<sub>j</sub></th></tr>`;
      st.cnt.forEach((row, j) => { h += `<tr><td>j=${j}</td>${row.map((v) => `<td>${v}</td>`).join("")}<td>${st.s[j]}</td></tr>`; });
      h += `</table><div style="margin:6px 0;color:var(--mute);font-size:12px">weights: ${Array.from({ length: c.wb }, (_, b) => "b" + b + "=" + QSim.wt(c, b)).join(" · ")}; y = Σ 2<sup>j</sup> s<sub>j</sub></div>`;
      h += `<table><tr><th>step</th><th>q</th><th>Z range</th><th>clamp</th><th>Z</th><th>u</th></tr>`;
      P.stages.forEach((g, k) => {
        const d = st.stagesDone[k], nm = g.merged ? "code FFN" : `FFN ${k + 1}`;
        const cl = g.merged ? "merged: thresholds on Z" : (g.clampLo || g.clampHi) ? `[${g.clampLo ? g.lo : "−∞"}, ${g.clampHi ? g.hi : "∞"}]` : "none";
        h += d ? `<tr><td>${nm}</td><td>${g.q}</td><td>[${g.zlo}, ${g.zhi}]</td><td>${cl}${d.clamped ? ' <span class="sat">hit</span>' : ""}</td><td>${d.Z}</td><td><b>${d.u}</b></td></tr>`
          : `<tr style="color:var(--mute)"><td>${nm}</td><td>${g.q}</td><td>[${g.zlo}, ${g.zhi}]</td><td>${cl}</td><td>·</td><td>·</td></tr>`;
      });
      h += `</table><div style="margin-top:6px;color:var(--mute);font-size:12px">chip layers for the tree: ${P.stages.filter((g) => !g.merged).length} clamp FFN${P.stages.filter((g) => !g.merged).length === 1 ? "" : "s"} + the code FFN · window ≤ 256 (max used ${P.maxWin})</div>`;
      if (st.stagesDone.length === P.stages.length) h += `<div style="margin-top:6px">result <b>${st.u}</b> = relu + clamp(y) in 0..${c.amax}. The code FFN turns it into the OUT(${st.u}) token.</div>`;
      $("reader").innerHTML = h;
    } else { $("rcol").textContent = ""; $("reader").innerHTML = '<span style="color:var(--mute)">waiting for a READ token…</span>'; }
  }
  function buildInputs() {
    const c = cfg; let h = `<b style="font-size:12px;color:var(--mute)">W</b><table>`;
    for (let r = 0; r < c.kr; r++) { h += "<tr>"; for (let j = 0; j < c.nc; j++) h += `<td><input data-w="${r},${j}" value="${Ws[0][r][j]}" inputmode="numeric"></td>`; h += "</tr>"; }
    h += `</table><b style="font-size:12px;color:var(--mute)">x</b><table><tr>`;
    for (let i = 0; i < c.kr; i++) h += `<td><input data-x="${i}" value="${x0[i]}" inputmode="numeric"></td>`;
    h += "</tr></table>"; $("inputs").innerHTML = h;
  }
  $("inputs").addEventListener("change", (e) => {
    const d = e.target.dataset, c = cfg; let v = parseInt(e.target.value, 10); if (isNaN(v)) v = 0;
    if (d.w) { const [r, j] = d.w.split(",").map(Number); v = Math.max(c.wlo, Math.min(c.whi, v)); Ws[0][r][j] = v; }
    else { v = Math.max(0, Math.min(c.amax, v)); x0[+d.x] = v; }
    e.target.value = v; stop(); build();
  });
  $("preset").onchange = () => reset(false);
  $("rand").onclick = () => reset(false);
  $("chain").onchange = () => reset(true);
  $("play").onclick = play;
  $("step").onclick = () => { stop(); next(); };
  $("back").onclick = () => { stop(); goto(idx - 1); };
  $("skip").onclick = () => { stop(); const i = built.steps.findIndex((s, k) => k > idx && s.kind === "act"); if (i > 0) goto(i - 1); };
  $("tokens").addEventListener("click", (e) => { const i = e.target.dataset && e.target.dataset.i; if (i != null) { stop(); goto(+i); } });
  document.addEventListener("keydown", (e) => { if (/input|select|textarea/i.test(e.target.tagName)) return; if (e.key === "ArrowRight") { stop(); next(); } else if (e.key === "ArrowLeft") { stop(); goto(idx - 1); } else if (e.key === " " && document.activeElement === document.body) { e.preventDefault(); play(); } });

  reset(false);
  loadResults().then((R) => { stats(R); charts(R); }).catch((e) => { $("charts").textContent = "Serve this folder over http (python3 -m http.server) or run docs/site/sync.sh: " + e; });
})();
