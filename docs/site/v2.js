/* qwenTPU site: v2 section. Token-by-token player driven by data/trace_v2.js (social/anim/trace_v2.json) + v2 numbers from results.json */
(function () {
  "use strict";
  const T = window.TRACE2; if (!T) return;
  const $ = (id) => document.getElementById(id);
  const digits = T.digits, N = 16;
  let di = 0, step = 0, timer = null;           // step = number of emitted tokens revealed (0 .. 64 READ/Y tokens + HALT = 65)
  const tokName = (tk, isY) => isY ? (tk.kind === "ACT" ? `ACT(${16 + tk.k};${tk.val})` : `OUTK(${tk.k};${tk.val})`) : `READ(${tk.k})`;
  function stream(d) {
    const out = [];
    d.toks.forEach((tk) => { out.push({ n: tokName(tk, false), cls: "k-read", tk, y: false }); out.push({ n: tokName(tk, true), cls: tk.kind === "ACT" ? "k-actv" : "k-out", tk, y: true }); });
    out.push({ n: "HALT", cls: "k-halt", tk: null, y: false });
    return out;
  }
  function stats() {
    const R = window.RESULTS && window.RESULTS.v2;
    const items = [];
    if (R) {
      const p = R.programs, tot = p.reduce((a, b) => a + b.runs, 0), ok = p.reduce((a, b) => a + b.exact, 0);
      items.push([`${ok} / ${tot}`, "prompt-chosen programs exact on Qwen3.5-0.8B (A100)"]);
      items.push([`${R.golden_mlp.depth2_exact} / ${R.golden_mlp.depth2_of}`, "digits: depth-2 golden MLP logits match"]);
      items.push([`${R.feedback.exact} / ${R.feedback.tokens}`, "ReLU feedback tokens exact on the real model"]);
    }
    items.push([`${(T.int_acc * 100).toFixed(1)}%`, `integer network, 540 held-out digits (float: ${(T.float_acc * 100).toFixed(1)}%)`]);
    $("v2stats").innerHTML = items.map(([n, l]) => `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
    if (R) {
      $("v2res").innerHTML = `<tr><th>compiled model</th><th>chip layers</th><th>programs run</th><th>exact</th><th>emitted tokens per program</th></tr>` +
        R.programs.map((r) => `<tr><td>${r.label}</td><td>${r.chip_layers}</td><td>${r.runs}</td><td><b>${r.exact} / ${r.runs}</b></td><td>${r.emitted_lens.join(", ")}</td></tr>`).join("") +
        `<tr><td>golden digit MLP, depth-2 program (16-wide u5s5)</td><td>20</td><td>${R.golden_mlp.depth2_of}</td><td><b>${R.golden_mlp.depth2_exact} / ${R.golden_mlp.depth2_of}</b></td><td>64</td></tr>` +
        `<tr><td>same model, depth-1 program (hidden layer alone)</td><td>20</td><td>${R.golden_mlp.depth1_of}</td><td><b>${R.golden_mlp.depth1_exact} / ${R.golden_mlp.depth1_of}</b></td><td>32</td></tr>` +
        `<tr><td>ReLU feedback, ${R.feedback.array}</td><td>·</td><td>${R.feedback.tokens} tokens</td><td><b>${R.feedback.exact} / ${R.feedback.tokens}</b></td><td>${R.feedback.distinct_y} distinct y values seen</td></tr>`;
    }
    $("v2acc").innerHTML = `<b>${(T.int_acc * 100).toFixed(1)}%</b> (float network ${(T.float_acc * 100).toFixed(1)}%)`;
    const dd = T.depth_demo;
    $("v2depth").innerHTML = `<tr><th>depth</th><th>entry block</th><th>prompt tokens</th><th>emitted</th><th>input x</th><th>output</th></tr>` +
      dd.progs.map((p) => `<tr><td><b>${p.depth}</b></td><td>${p.entry}</td><td>${p.prompt_len}</td><td>${p.emitted}</td><td class="mono">${p.x.join(" ")}</td><td class="mono">${p.out.join(" ")}</td></tr>`).join("");
  }
  function build() {
    const d = digits[di];
    $("v2img").innerHTML = d.img.flat().map((v) => `<i style="opacity:${(0.08 + 0.92 * v / 16).toFixed(2)}" title="${v}"></i>`).join("");
    $("v2x").innerHTML = d.x.map((v) => `<span class="o">${v}</span>`).join("");
    $("v2prompt").innerHTML = `<span class="mono" style="font-size:12.5px">CELL × 1024 (W1, W2) · ACT(r;0) × 16 · ACT(0;x0) … ACT(15;1)</span><div style="color:var(--mute);font-size:13px;margin-top:4px">1056 tokens, all data: no READ, no OUT, no op. The last ACT sets where the model starts.</div>`;
    const S = stream(d);
    $("v2tokens").innerHTML = S.map((s, i) => `<span class="tok ${s.cls}" data-i="${i}">${s.n}</span>`).join("");
    render();
  }
  function bars(el, vals, shown, curIdx, winIdx, padFrom) {
    el.innerHTML = vals.map((v, i) => {
      const on = i < shown, h = on ? Math.max(2, Math.round(100 * v / 31)) : 2;
      return `<div class="${i === curIdx ? "cur" : ""} ${i === winIdx ? "win" : ""} ${i >= padFrom ? "pad" : ""}" title="${i}"><span>${on ? v : ""}</span><b style="height:${h}px"></b><span>${i}</span></div>`;
    }).join("");
  }
  function render() {
    const d = digits[di], S = stream(d);
    document.querySelectorAll("#v2tokens .tok").forEach((e, i) => { e.classList.toggle("cur", i === step - 1); e.classList.toggle("fut", i >= step); });
    const cur = $("v2tokens").querySelector(".cur"); if (cur) { const c = $("v2tokens"); c.scrollLeft = cur.offsetLeft - c.clientWidth / 2; } else $("v2tokens").scrollLeft = 0;
    $("v2bar").style.width = (100 * step / S.length) + "%";
    const pairs = Math.floor(step / 2), l1 = Math.min(pairs, N), l2 = Math.max(0, Math.min(pairs - N, N));
    bars($("v2h"), d.y1, l1, step > 0 && step <= 2 * N ? Math.floor((step - 1) / 2) : -1, -1, 99);
    const winIdx = step >= 2 * N * 2 ? d.logits.indexOf(Math.max(...d.logits)) : -1;
    bars($("v2l"), d.logits.concat([0, 0, 0, 0, 0, 0]), l2, step > 2 * N && step <= 4 * N ? Math.floor((step - 1) / 2) - N : -1, winIdx, 10);
    const done = step >= S.length;
    $("v2counter").innerHTML = [["emitted", `${Math.min(step, 64)} / 64`], ["host work", "argmax; append"], ["layer", step === 0 ? "·" : done ? "2 (done)" : step <= 2 * N ? "1" : "2"], ["state", done ? "HALT" : step === 0 ? "prompt fed" : "running"]]
      .map(([a, b]) => `<div class="cnt"><span style="color:var(--mute);font-size:12px">${a}</span><b>${b}</b></div>`).join("");
    let note;
    if (step === 0) note = "The host fed the prompt (weights, input, entry point). Everything from here on is the model's own output. Press Play or step with the arrow keys.";
    else if (done) note = "The model emitted HALT. The host stops. Logits are the OUTK values of layer 2.";
    else {
      const s = S[step - 1], tk = s.tk;
      if (!s.y) note = `<b>${s.n}</b>: the model chose this token itself. It names column ${tk.k} of layer ${tk.layer + 1}; the next position will sum that column's weights against the activations written so far.`;
      else { const raw = tk.layer === 0 ? d.raw1[tk.k] : d.raw2[tk.k]; note = `<b>${s.n}</b>: exact sum ${raw} → relu → clamp to 0..31 → <b>${tk.val}</b>${tk.kind === "ACT" ? `, emitted as layer 2's input row ${16 + tk.k}` : `, a logit for digit ${tk.k}${tk.k >= 10 ? " (padding column, always 0)" : ""}`}.`; }
    }
    $("v2note").innerHTML = note;
    if (done) { const ok = d.pred === d.label; $("v2verdict").innerHTML = `argmax of the emitted logits: <b class="${ok ? "ok" : "bad"}">${d.pred}</b> · label <b>${d.label}</b> · ${ok ? "correct" : "wrong"} · logits equal the integer golden model`; }
    else $("v2verdict").innerHTML = `<span style="color:var(--mute)">label ${d.label}; prediction appears at HALT</span>`;
  }
  const len = () => stream(digits[di]).length;
  function next() { if (step >= len()) { stop(); return; } step++; render(); }
  function stop() { if (timer) clearTimeout(timer); timer = null; $("v2play").textContent = "Play"; }
  function play() { if (timer) return stop(); if (step >= len()) { step = 0; render(); } $("v2play").textContent = "Pause"; const tick = () => { next(); if (timer) timer = setTimeout(tick, 1000 / +$("v2speed").value); }; timer = setTimeout(tick, 0); }
  digits.forEach((d, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `held-out digit ${i} (label ${d.label}, predicted ${d.pred})`; $("v2digit").appendChild(o); });
  $("v2digit").onchange = () => { stop(); di = +$("v2digit").value; step = 0; build(); };
  $("v2play").onclick = play;
  $("v2step").onclick = () => { stop(); next(); };
  $("v2back").onclick = () => { stop(); step = Math.max(0, step - 1); render(); };
  $("v2reset").onclick = () => { stop(); step = 0; render(); };
  $("v2tokens").addEventListener("click", (e) => { const i = e.target.dataset && e.target.dataset.i; if (i != null) { stop(); step = +i + 1; render(); } });
  stats(); build();
  if (!window.RESULTS) fetch("data/results.json").then((r) => r.json()).then((R) => { window.RESULTS = R; stats(); });
})();
