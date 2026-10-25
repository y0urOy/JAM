'use strict';
/* JAM site app. Every paper number and label comes from D (data.js); media facts come from M
   (inlined by src/build.py from src/media.json). Nothing here stores state outside the page. */

// =============================================================================================
// helpers
// =============================================================================================
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const NS = 'http://www.w3.org/2000/svg';
function S(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  if (attrs) for (const k in attrs) {
    const v = attrs[k];
    if (v == null) continue;
    if (k === 'text') e.textContent = v;
    else if (k === 'cls') e.setAttribute('class', v);
    else e.setAttribute(k, v);
  }
  if (parent) parent.appendChild(e);
  return e;
}
function H(tag, attrs, parent) {
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    const v = attrs[k];
    if (v == null) continue;
    if (k === 'text') e.textContent = v;
    else if (k === 'cls') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  if (parent) parent.appendChild(e);
  return e;
}
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const saveData = !!(navigator.connection && navigator.connection.saveData);
const autoMotion = !reduced && !saveData;
function mediaCanRun(v) { const dialog = document.querySelector('dialog[open]'); return !document.hidden && (!dialog || dialog.contains(v)); }
function videoInView(v) { const r = v.getBoundingClientRect(); return r.width > 0 && r.bottom > 54 && r.top < innerHeight && r.right > 0 && r.left < innerWidth; }
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const MINUS = '−';
const pc = x => x.p + '%';                         // printed value + percent sign
const where = x => String(x.w).replace(/ (text|caption)$/, '');
// "Table 1", "Table 19" -> "Tables 1, 19"
function srcs(...vals) {
  const tabs = [], other = [];
  for (const x of vals.flat()) {
    if (!x) continue;
    const w = typeof x === 'string' ? x : where(x);
    const m = /^Tables? ([\d, ]+)$/.exec(w);
    if (m) m[1].split(/,\s*/).forEach(n => { if (!tabs.includes(n)) tabs.push(n); });
    else if (!other.includes(w)) other.push(w);
  }
  tabs.sort((a, b) => a - b);
  const t = tabs.length ? (tabs.length > 1 ? 'Tables ' : 'Table ') + tabs.join(', ') : '';
  return [t, ...other].filter(Boolean).join(' · ');
}
const signed = (v, d) => (v > 0 ? '+' : v < 0 ? MINUS : '') + Math.abs(v).toFixed(d);
const reductionText = r => MINUS + (Number.isInteger(r.v) ? r.v : r.p.replace('%', '')) + '%';
// math in HTML: segments [text, style] with style i (italic), u (upright), si/su (subscript), pi/pu (superscript)
function mh(segs) {
  return '<span class="m">' + segs.map(([t, s]) => {
    const inner = s.endsWith('i') ? `<i>${esc(t)}</i>` : esc(t);
    return s[0] === 's' && s.length === 2 ? `<sub>${inner}</sub>` : s[0] === 'p' ? `<sup>${inner}</sup>` : inner;
  }).join('') + '</span>';
}
// math in SVG <text>
function mt(parent, x, y, segs, size, o = {}) {
  const t = S('text', { x, y, cls: 'mt ' + (o.cls || ''), 'font-size': size, 'text-anchor': o.anchor || 'start' }, parent);
  if (o.fill) t.style.fill = o.fill;
  let base = 0;
  for (const [txt, s] of segs) {
    const lvl = s.length === 2 ? (s[0] === 's' ? .3 : -.42) : 0;
    const ts = S('tspan', { text: txt }, t);
    if (s.endsWith('u')) ts.setAttribute('class', 'up');
    if (lvl) ts.setAttribute('font-size', (size * .7).toFixed(1));
    const dy = (lvl - base) * size;
    if (Math.abs(dy) > .01) ts.setAttribute('dy', dy.toFixed(2));
    base = lvl;
  }
  return t;
}
const MATH = {
  LF: [['ℒ', 'u'], ['F', 'su']], LG: [['ℒ', 'u'], ['G', 'su']], Lact: [['ℒ', 'u'], ['act', 'su']],
  g: [['g', 'i'], ['k', 'si']], gh: [['ĝ', 'i'], ['k', 'si']], f0: [['f', 'i'], ['0', 'su']],
  f1T: [['f', 'i'], ['1:', 'su'], ['T', 'si']], fh1T: [['f̂', 'i'], ['1:', 'su'], ['T', 'si']], fT: [['f', 'i'], ['T', 'si']], fhT: [['f̂', 'i'], ['T', 'si']],
  a1H: [['â', 'i'], ['1:', 'su'], ['H', 'si']], h: [['h', 'i'], ['(ℓ)', 'pu']],
};
// chart SVG with role=img, named by its aria-label (a <title> would pop the browser's own tooltip over the marks')
function chartSVG(attrs, parent) {
  return S('svg', Object.assign({ role: 'img' }, attrs), parent);
}
function marker(svg, id, cls) {
  const defs = svg.querySelector('defs') || S('defs', null, svg);
  const m = S('marker', { id, viewBox: '0 0 10 10', refX: 8.6, refY: 5, markerWidth: 6.5, markerHeight: 6.5, orient: 'auto-start-reverse' }, defs);
  S('path', { d: 'M0,0.6 L10,5 L0,9.4 L2.2,5 z', cls }, m);
}
function seg(el, items, onPick, cur) {
  const fv = el.contains(document.activeElement) ? document.activeElement.dataset.v : null; // a rebuild keeps keyboard focus
  el.textContent = '';
  for (const it of items) {
    const b = H('button', { type: 'button', 'data-v': it.v, 'aria-pressed': String(it.v === cur) }, el);
    if (it.k) b.classList.add('k-' + it.k);
    if (it.html) b.innerHTML = it.html; else b.append(document.createTextNode(it.label));
    if (it.sub) H('small', { text: it.sub }, b);
    if (it.title) b.title = it.title;
    if (it.dis) b.setAttribute('aria-disabled', 'true');
    b.addEventListener('click', () => {
      if (b.getAttribute('aria-disabled') === 'true') { if (it.why) flashTip(b, it.why()); return; }
      onPick(it.v);
    });
  }
  if (fv != null) { const n = [...el.children].find(b => b.dataset.v === fv); if (n) n.focus({ preventScroll: true }); }
}
function press(el, v) { el.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === v))); }
// narrow layouts for SVG instruments: re-render when the host crosses a width breakpoint
const NARROW = [];
function watchNarrow(host, bp, render) {
  const st = { host, bp, render, n: host.clientWidth < bp };
  NARROW.push(st);
  return () => st.n;
}
let rzT = 0;
addEventListener('resize', () => {
  clearTimeout(rzT);
  rzT = setTimeout(() => { NARROW.forEach(st => { const n = st.host.clientWidth < st.bp; if (n !== st.n) { st.n = n; st.render(); } }); refreshCharts(); }, 120);
});

// ---------------------------------------------------------------- chart finishing: legible text, focus rings, one Tab stop per chart
/* SVG charts scale with their box. Text that would render below MIN_PX CSS px is enlarged in place (user units), so
   labels stay readable at tablet and phone widths without a separate layout for every width. */
const MIN_PX = 11;
function legible(svg) {
  const vb = svg.viewBox && svg.viewBox.baseVal, w = svg.getBoundingClientRect().width;
  if (!vb || !vb.width || !w) return;
  const sc = w / vb.width;
  svg.querySelectorAll('text').forEach(t => {
    if (t._fs0 == null) { t.style.fontSize = ''; t._fs0 = parseFloat(getComputedStyle(t).fontSize) || 12; }
    const f = t._fs0 * sc < MIN_PX ? MIN_PX / (t._fs0 * sc) : 1;
    t.style.fontSize = f > 1 ? (t._fs0 * f).toFixed(2) + 'px' : '';
    t.querySelectorAll('tspan[font-size]').forEach(ts => {
      const a = ts.getAttribute('font-size');
      if (!/%$/.test(a)) ts.style.fontSize = f > 1 ? (parseFloat(a) * f).toFixed(2) + 'px' : '';
    });
  });
}
// every focusable mark gets a transparent .hit shape behind it; the focus ring is drawn on that shape
function ensureHits(svg) {
  svg.querySelectorAll('.hit.auto').forEach(n => n.remove());
  svg.querySelectorAll('[tabindex]').forEach(el => {
    if ([...el.children].some(c => c.classList && c.classList.contains('hit'))) return;
    let b;
    try { b = el.getBBox(); } catch (e) { return; }
    if (!b || (!b.width && !b.height)) return;
    el.insertBefore(S('rect', { x: b.x - 3, y: b.y - 3, width: b.width + 6, height: b.height + 6, rx: 4, cls: 'hit auto' }), el.firstChild);
  });
}
// marks inside one chart share a single Tab stop; the arrow keys, Home and End move between them
function rove(svg) {
  const marks = [...svg.querySelectorAll('[data-mark]')];
  if (!marks.length) return;
  const keep = marks.find(m => m === document.activeElement) || marks.find(m => m.getAttribute('aria-pressed') === 'true') || marks[0];
  marks.forEach(m => m.setAttribute('tabindex', m === keep ? '0' : '-1'));
  if (svg._rove) return;
  svg._rove = 1;
  svg.addEventListener('keydown', e => {
    const ms = [...svg.querySelectorAll('[data-mark]')], i = ms.indexOf(document.activeElement);
    const k = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, Home: -1e4, End: 1e4 }[e.key];
    if (i < 0 || k == null) return;
    e.preventDefault();
    const j = clamp(i + k, 0, ms.length - 1);
    ms.forEach((m, n) => m.setAttribute('tabindex', n === j ? '0' : '-1'));
    ms[j].focus();
  });
}
function finish(svg) { if (!svg) return; legible(svg); ensureHits(svg); rove(svg); }
function refreshCharts() { document.querySelectorAll('main svg[role]').forEach(sv => { legible(sv); ensureHits(sv); }); placeArrowLabels(); }

// ---------------------------------------------------------------- tooltip (pointer, touch and keyboard focus)
const tip = $('#tip');
let tipOwner = null, tipFocus = false, tipKey = false;
// focus from the keyboard makes a focus tooltip; after a mouse press it stays a pointer tooltip, hidden on leave or scroll
addEventListener('keydown', () => { tipKey = true; }, true);
addEventListener('pointerdown', () => { tipKey = false; }, true);
function placeTip(x, y, yTop) {
  const r = tip.getBoundingClientRect();
  let tx = x + 14, ty = y + 16;
  if (tx + r.width > innerWidth - 8) tx = x - r.width - 14;
  if (ty + r.height > innerHeight - 8) ty = (yTop != null ? yTop : y) - r.height - 10;
  tx = clamp(tx, 8, Math.max(8, innerWidth - r.width - 8));
  ty = clamp(ty, 8, Math.max(8, innerHeight - r.height - 8));
  tip.style.transform = `translate(${Math.round(tx)}px, ${Math.round(ty)}px)`;
}
// the box takes the colour of the mark it describes (accent), cyan by default
function showTip(html, x, y, yTop, accent) { clearTimeout(flashTip.t); tip.innerHTML = html; tip.style.setProperty('--tip-c', accent || 'var(--cyan)'); tip.classList.add('on'); placeTip(x, y, yTop); }
function hideTip() { clearTimeout(flashTip.t); tip.classList.remove('on'); tipOwner = null; tipFocus = false; }
// a keyboard tooltip shows only while its mark is on screen below the sticky bar
const offScreen = el => { const r = el.getBoundingClientRect(), top = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0; return r.bottom < top || r.top > innerHeight; };
const atFocus = el => { const r = el.getBoundingClientRect(); return [r.left + Math.min(r.width / 2, 120), r.bottom - 4, r.top]; };
function bindTip(el, fn, label, accent) {
  el._tipFn = fn; el._tipAcc = accent;
  const show = (x, y, yTop) => showTip(fn(), x, y, yTop, el._tipAcc);
  el.addEventListener('pointermove', e => { if (e.pointerType === 'mouse') { tipOwner = el; tipFocus = false; show(e.clientX, e.clientY); } });
  el.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse' && tipOwner === el && !(tipFocus && document.activeElement === el)) hideTip(); });
  el.addEventListener('pointerdown', e => {
    if (e.pointerType !== 'mouse') { tipOwner = el; tipFocus = false; show(e.clientX, e.clientY); clearTimeout(bindTip.t); bindTip.t = setTimeout(hideTip, 2800); }
  });
  el.addEventListener('focus', () => {
    if (!tipKey && tipOwner === el && tip.classList.contains('on')) return; // a click keeps the pointer tooltip at the cursor
    tipOwner = el; tipFocus = tipKey; show(...atFocus(el));
    if (tipFocus && offScreen(el)) tip.classList.remove('on'); // shown by the scroll rule once the focusin rule brings it in
  });
  el.addEventListener('blur', hideTip);
  if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '0');
  if (label) el.setAttribute('aria-label', label);
  if (el instanceof SVGElement) el.setAttribute('data-mark', '');
}
function flashTip(el, msg) { const r = el.getBoundingClientRect(); showTip(esc(msg), r.left + 10, r.bottom, r.top); clearTimeout(flashTip.t); flashTip.t = setTimeout(hideTip, 2600); }
/* No browser title tooltips (a grey box): a title attribute becomes data-hint and the accessible description as soon
   as it is set, and a mouse resting on the element shows the hint in this tooltip instead. */
function toHint(el) {
  const t = el.getAttribute('title');
  el.removeAttribute('title');
  if (!t) { delete el.dataset.hint; el.removeAttribute('aria-description'); return; }
  el.dataset.hint = t;
  if (t.toLowerCase() !== (el.getAttribute('aria-label') || '').toLowerCase()) el.setAttribute('aria-description', t);
}
document.querySelectorAll('[title]').forEach(toHint);
new MutationObserver(ms => ms.forEach(m => (m.type === 'attributes' ? [m.target] : [...m.addedNodes]).forEach(n => {
  if (n.nodeType === 1) [n, ...n.querySelectorAll('[title]')].forEach(e => { if (e.hasAttribute('title')) toHint(e); });
}))).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ['title'] });
let hintEl = null, hintT = 0, hintAt = [0, 0];
const endHint = () => { clearTimeout(hintT); if (hintEl && tipOwner === hintEl) hideTip(); };
document.addEventListener('pointerover', e => {
  const el = e.pointerType === 'mouse' && e.target.closest ? e.target.closest('[data-hint]') : null;
  if (el === hintEl) return;
  endHint(); hintEl = el; hintAt = [e.clientX, e.clientY];
  if (el) hintT = setTimeout(() => { if (el.dataset.hint && (!tipOwner || !tip.classList.contains('on'))) { tipOwner = el; tipFocus = false; showTip(esc(el.dataset.hint), ...hintAt); } }, 350);
});
document.addEventListener('pointermove', e => { if (hintEl && e.pointerType === 'mouse') { hintAt = [e.clientX, e.clientY]; if (tipOwner === hintEl) placeTip(...hintAt); } }, { passive: true });
document.addEventListener('pointerout', e => { if (!e.relatedTarget) { endHint(); hintEl = null; } }); // the pointer left the window
document.addEventListener('pointerdown', endHint, true);
addEventListener('scroll', () => {
  if (!tipFocus && !tip.classList.contains('on')) return;
  // pointer tooltips, hints and flash messages hide on scroll
  if (!tipOwner || !tipFocus || document.activeElement !== tipOwner) { hideTip(); return; }
  if (offScreen(tipOwner)) tip.classList.remove('on'); // a keyboard tooltip follows its mark while the mark is on screen
  else { tip.classList.add('on'); placeTip(...atFocus(tipOwner)); }
}, { passive: true });
addEventListener('keydown', e => { if (e.key === 'Escape') hideTip(); });
/* Keyboard focus is always fully on screen: Firefox does not scroll focused SVG marks into view and leaves partly
   visible elements at the viewport edge; scroll-padding-top keeps the target clear of the sticky bar. */
addEventListener('focusin', e => {
  const el = e.target;
  if (!el || !el.getBoundingClientRect || el.closest('.topbar, dialog')) return;
  let kb = true;
  try { kb = el.matches(':focus-visible') || (tipKey && el.hasAttribute('data-mark')); } catch (x) {}
  if (!kb) return;
  const r = el.getBoundingClientRect(), top = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
  if (r.top >= top && r.bottom <= innerHeight) return;
  if (el instanceof SVGElement) { // Firefox's scrollIntoView is a no-op for elements inside an <svg>: scroll the page itself
    const d = r.height > innerHeight - top || r.top < top ? r.top - top : r.bottom - innerHeight + 8;
    window.scrollBy({ top: d });
  } else { try { el.scrollIntoView({ block: r.height > innerHeight - top ? 'start' : 'nearest', inline: 'nearest' }); } catch (x) {} }
});
// source line in a tooltip: reader-facing locators only (no data ids, no LaTeX file paths)
const tipSrc = s => {
  const out = String(s).replace(/,?\s*[\w./-]+\.tex(\s+L\d+)?/g, '').split(' · ').map(p => p.trim())
    .filter(p => p && !/^[\w-]+(\.[\w-]+){2,}$/.test(p)).join(' · ');
  return out ? `<span class="s">${esc(out)}</span>` : '';
};
// big readout: number + mono label, tooltip with exact values
function big(id, html, label, tipHtml, cls) {
  const el = $(id); el.textContent = '';
  const b = H('b', { html }, el); if (cls) b.className = cls;
  H('span', { html: label }, el); // trusted page strings; numeric runs are wrapped in .nw by the caller
  if (tipHtml) { el._tip = tipHtml; if (!el._bound) { bindTip(el, () => el._tip()); el._bound = 1; } el._tipAcc = cls === 'a' ? 'var(--amber)' : null; }
  return el;
}

// ---------------------------------------------------------------- media: format choice, Blob loading with fallbacks
const IS_SAFARI = /^((?!chrome|android|crios|fxios|edg).)*safari/i.test(navigator.userAgent);
const FMT = (() => { try { const v = document.createElement('video'); return !IS_SAFARI && v.canPlayType('video/webm; codecs="vp9"') === 'probably' ? 'webm' : 'mp4'; } catch (e) { return 'mp4'; } })();
const OTHER = FMT === 'webm' ? 'mp4' : 'webm';
const BLOB_MAX = 7e6;
const aborted = sig => !!(sig && sig.aborted);
async function fetchBlob(url, onProgress, signal) {
  const r = await fetch(url, signal ? { signal } : undefined);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  const total = +(r.headers.get('content-length') || 0);
  const type = r.headers.get('content-type') || (url.endsWith('.webm') ? 'video/webm' : 'video/mp4');
  if (!r.body || !r.body.getReader) { const b = await r.blob(); if (onProgress) onProgress(1); return b; }
  const rd = r.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    parts.push(value); got += value.length;
    if (onProgress && total) onProgress(got / total);
  }
  if (onProgress) onProgress(1);
  return new Blob(parts, { type });
}
/* Attach a source to <video>: Blob of the preferred format (small files) -> direct URL -> other format.
   Resolves on loadedmetadata, rejects when every option failed or `signal` aborted (a later attach on the same
   element silently supersedes an earlier one). */
function attachVideo(v, base, bytes, onProgress, forceBlob, signal) {
  if (v._objURL) { URL.revokeObjectURL(v._objURL); v._objURL = null; }
  const steps = [];
  const small = (bytes && bytes[FMT]) ? bytes[FMT] <= BLOB_MAX : false;
  if (small || forceBlob) steps.push({ fmt: FMT, blob: true });
  steps.push({ fmt: FMT, blob: false }, { fmt: OTHER, blob: false });
  const token = (v._token = (v._token || 0) + 1);
  return new Promise((resolve, reject) => {
    let i = -1;
    const cleanup = () => { v.removeEventListener('error', onErr); v.removeEventListener('loadedmetadata', onMeta); if (signal) signal.removeEventListener('abort', onAbort); };
    const onMeta = () => { if (token !== v._token) return cleanup(); cleanup(); v.dataset.fmt = steps[i].fmt; resolve(); };
    const onErr = () => { if (token !== v._token) return cleanup(); next(); };
    const onAbort = () => { cleanup(); reject(new Error('aborted')); };
    v.addEventListener('loadedmetadata', onMeta);
    v.addEventListener('error', onErr);
    if (signal) signal.addEventListener('abort', onAbort);
    const next = async () => {
      i++;
      if (token !== v._token) return cleanup();
      if (aborted(signal)) return onAbort();
      if (i >= steps.length) { cleanup(); reject(new Error('no playable source')); return; }
      const st = steps[i], url = `${base}.${st.fmt}`;
      if (st.blob) {
        try {
          const b = await fetchBlob(url, onProgress, signal);
          if (token !== v._token) return cleanup();
          if (aborted(signal)) return onAbort();
          v._objURL = URL.createObjectURL(b);
          v.src = v._objURL;
        } catch (e) { if (aborted(signal)) return onAbort(); next(); return; }
      } else {
        if (v._objURL) { URL.revokeObjectURL(v._objURL); v._objURL = null; }
        v.src = url;
      }
      v.load();
    };
    next();
  });
}
const PLAY_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>';
const PAUSE_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>';
const clockStr = t => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, '0')}`;
const mmss = t => `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, '0')}`;
// range inputs: one arrow-key press moves `stepS` seconds of the shown clock (default step is 0.1 % of the clip)
function scrubKeys(input, stepS, seek) {
  input.addEventListener('keydown', e => {
    const d = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 }[e.key];
    if (!d) return;
    e.preventDefault();
    seek(d * stepS);
  });
}

// =============================================================================================
// hero: authors, readout tiles, trailer (cinema overlay)
// =============================================================================================
function initHero() {
  const au = $('#authors');
  const sup = ['', '¹', '²'];
  // author names, homepage links and the corresponding-author mark
  const URL = { 'Yonghao Yu': 'https://y0uroy.github.io/', 'Lang Huang': 'https://layneh.github.io/', 'Runyi Li': 'https://lirunyi2001.github.io/',
    'Toshihiko Yamasaki': 'https://www.ee.t.u-tokyo.ac.jp/en/staff/yamasaki-toshihiko/' };
  const nm = a => URL[a.n.p] ? `<a href="${URL[a.n.p]}" target="_blank" rel="noopener">${esc(a.n.p)}</a>` : esc(a.n.p);
  const names = D.meta.authors.map(a => `<span class="nw"><b>${nm(a)}</b><sup>${a.aff}${a.corr ? '<span title="Corresponding author" aria-label="corresponding author">✉</span>' : ''}</sup></span>`);
  au.innerHTML = names.join(' · ') + '<br>' + D.meta.affs.map((a, i) => `<span class="nw">${sup[i + 1]}${esc(a.p)}</span>`).join(' · ');

  const host = $('#readouts');
  // reference layout: one big number and one label line per tile; exact values and sources live in the tooltip
  const tile = (num, unit, label, tipHtml, cls) => {
    const d = H('div', { cls: 'ro' + (cls ? ' ' + cls : ''), tabindex: '0' }, host);
    H('strong', { html: esc(num) + (unit ? `<small>${unit}</small>` : '') }, d);
    H('span', { cls: 'ro-l', text: label }, d);
    bindTip(d, () => tipHtml, null, { a: 'var(--amber)', v: 'var(--violet)' }[cls]); // the tile's edge colour
    return d;
  };
  const o = D.hero.ovis, w = D.hero.wan, f = D.hero.flops;
  const gainTip = x => `<b>${x.delta.p} pts</b> LIBERO-Plus, ${esc(x.jam)} vs ${esc(x.base)}; ${esc(x.text.p)} in ${esc(where(x.text))}.` +
    `<br>${esc(x.base)} ${pc(x.from)} → ${esc(x.jam)} ${pc(x.to)} (${esc(where(x.from))}). JAM: ${esc(x.cfg)}.` +
    tipSrc(`Δ from unrounded rates · ${where(x.delta)}`);
  tile(signed(o.delta.v, 1), 'PTS', 'LIBERO-Plus vs ImageWAM', gainTip(o));
  tile(signed(w.delta.v, 1), 'PTS', 'LIBERO-Plus vs FastWAM', gainTip(w));
  tile(reductionText(f.red), '', 'Training FLOPs vs FastWAM',
    `<b>${MINUS}${f.red.p}</b> estimated online training FLOPs per sample per update, JAM (Wan) with cached targets vs FastWAM: ${f.from.p} → ${f.to.p} TFLOPs (${esc(where(f.red))}; ${esc(f.abs.p)} in the abstract).` +
    `<br>Online targets: ${MINUS}${esc(f.online.p)}. Cached costs exclude one-time target extraction. A FLOP estimate, not wall-clock time.`, 'a');
  const zero = esc(D.hero.frames.zero.p);
  tile(D.hero.frames.zero.p, '', 'Frames generated, ever',
    `<b>${zero} frames</b> in training and deployment. “${esc(D.hero.frames.q.t)}”${tipSrc(D.hero.frames.q.w)}` +
    `“${esc(D.hero.frames.q3.t)}”${tipSrc(D.hero.frames.q3.w)}`, 'v');
  // the grid floor drifts only while the hero is on screen
  if ('IntersectionObserver' in window) new IntersectionObserver(es => es.forEach(e => $('.hero').classList.toggle('off', !e.isIntersecting))).observe($('.hero'));
}

/* Trailer: the hero shows only the poster. Play opens a modal cinema overlay (a <dialog>): focus moves into it and
   returns to the opener, the page behind is locked, Esc or Close shuts it and pauses the film. The film is fetched
   whole as a Blob (progress in the overlay), with the plain URL as fallback. */
const TR = { loading: false, loaded: false, opener: null };
const COARSE = matchMedia('(max-width: 900px) and (pointer: coarse)');
function cinemaOpen(opener) {
  const dlg = $('#cinema');
  if (!M.trailer) { flashTip(opener || $('#film-frame'), 'The trailer is not published yet.'); return; }
  if (!dlg.open) {
    pauseForOverlay($('#trailer'));
    TR.opener = opener || document.activeElement;
    document.documentElement.classList.add('locked');
    try { dlg.showModal(); } catch (e) { dlg.setAttribute('open', ''); }
    $('#cinema-close').focus();
    if (COARSE.matches) { // phones: the 16:9 film needs the whole screen (inside the click, so the request is allowed)
      const fr = $('#cinema-frame');
      try {
        const p = fr.requestFullscreen ? fr.requestFullscreen() : null;
        if (p && p.then) p.then(() => { try { screen.orientation.lock('landscape').catch(() => {}); } catch (e) {} }).catch(() => {});
      } catch (e) {}
    }
  }
  playTrailer();
}
function cinemaClose() { const dlg = $('#cinema'); if (dlg.close) dlg.close(); else { dlg.removeAttribute('open'); cinemaClosed(); } }
function cinemaClosed() {
  $('#trailer').pause();
  document.documentElement.classList.remove('locked');
  try { if (document.fullscreenElement) document.exitFullscreen().catch(() => {}); } catch (e) {}
  try { if (screen.orientation && screen.orientation.unlock) screen.orientation.unlock(); } catch (e) {}
  const o = TR.opener; TR.opener = null;
  if (o && o.focus) { try { o.focus({ preventScroll: true }); } catch (e) { o.focus(); } }
  resumeAfterOverlay();
}
async function playTrailer() {
  const v = $('#trailer'), dlg = $('#cinema'), ld = $('#cinema-load');
  if (TR.loaded) { if (dlg.open && mediaCanRun(v)) { if (v.ended) v.currentTime = 0; v.play().catch(() => {}); } return; }
  if (TR.loading) return;
  TR.loading = true;
  $('#cinema-retry').hidden = true;
  const prog = f => { ld.textContent = `Loading ${Math.round(100 * f)}%`; };
  ld.hidden = false; prog(0);
  try { await attachVideo(v, M.trailer.base, M.trailer.bytes, prog, true); }
  catch (e) { TR.loading = false; ld.textContent = 'Video could not load.'; $('#cinema-retry').hidden = false; return; }
  TR.loading = false; TR.loaded = true; ld.hidden = true;
  v.controls = true; v.removeAttribute('tabindex');
  if (dlg.open && mediaCanRun(v)) v.play().catch(() => {}); // loading may finish after the overlay closed or the tab was hidden
}
function initTrailer() {
  const dlg = $('#cinema');
  $('#film-play').addEventListener('click', e => { e.stopPropagation(); cinemaOpen(e.currentTarget); });
  $('#film-frame').addEventListener('click', () => { if (!$('#film-play').hidden) cinemaOpen($('#film-play')); });
  $('#cinema-close').addEventListener('click', cinemaClose);
  $('#cinema-retry').addEventListener('click', playTrailer);
  dlg.addEventListener('click', e => { if (e.target === dlg) cinemaClose(); });
  dlg.addEventListener('close', cinemaClosed);
  if (!M.trailer) $('#film-play').setAttribute('aria-disabled', 'true');
}

/* Code / Model / arXiv buttons. A URL makes the button a link; without one (arXiv, until the preprint is out) a click
   shows a short "coming soon" note. */
const LINKS = { code: 'https://github.com/y0urOy/JAM', model: 'https://huggingface.co/y0urOy/JAM', arxiv: null };
function initResources() {
  document.querySelectorAll('.res-btn').forEach(b => {
    const url = LINKS[b.dataset.res];
    if (url) {
      const a = H('a', { cls: b.className, href: url, target: '_blank', rel: 'noopener' });
      a.innerHTML = b.innerHTML;
      b.replaceWith(a);
    } else b.addEventListener('click', () => flashTip(b, b.dataset.soon || 'Coming soon'));
  });
  // the JAM mark in the top bar opens the code repository
  Object.assign($('.brand'), { href: LINKS.code, target: '_blank', rel: 'noopener' });
  $('.brand').setAttribute('aria-label', 'JAM, Just Action Models: code on GitHub');
}

// =============================================================================================
// navigation: active section + scroll progress + edge arrows when the strip overflows
// =============================================================================================
function initNav() {
  const links = $$('#nav a'), nav = $('#nav'), bar = $('#progress'), aL = $('#nav-l'), aR = $('#nav-r');
  const toggle = $('#nav-toggle'), box = $('#nav-box'), compact = matchMedia('(max-width: 1100px)');
  const closeNav = (focus = false) => { box.classList.remove('is-open'); toggle.setAttribute('aria-expanded', 'false'); if (focus) toggle.focus(); };
  toggle.addEventListener('click', () => { const open = box.classList.toggle('is-open'); toggle.setAttribute('aria-expanded', String(open)); });
  document.addEventListener('click', e => { if (!box.contains(e.target)) closeNav(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && box.classList.contains('is-open')) { closeNav(true); e.preventDefault(); } });
  compact.addEventListener('change', () => closeNav());
  links.forEach(a => a.addEventListener('click', () => {
    if (!compact.matches) return;
    closeNav();
    const section = document.querySelector(a.getAttribute('href')), h = section.querySelector('h2');
    if (h) {
      h.tabIndex = -1;
      requestAnimationFrame(() => {
        h.focus({ preventScroll: true });
        section.scrollIntoView({ block: 'start', behavior: reduced ? 'instant' : 'smooth' });
      });
    }
  }));
  const secs = links.map(a => document.getElementById(a.getAttribute('href').slice(1)));
  let cur = null, raf = 0;
  const arrows = () => {
    const max = nav.scrollWidth - nav.clientWidth;
    aL.hidden = !(max > 2 && nav.scrollLeft > 2);
    aR.hidden = !(max > 2 && nav.scrollLeft < max - 2);
  };
  const update = () => {
    raf = 0;
    const de = document.documentElement, max = de.scrollHeight - innerHeight;
    bar.style.transform = `scaleX(${max > 0 ? clamp(scrollY / max, 0, 1) : 0})`;
    const line = innerHeight * .38;
    let idx = -1;
    secs.forEach((s, i) => { if (s && s.getBoundingClientRect().top <= line) idx = i; });
    if (max > 0 && scrollY >= max - 4) idx = secs.length - 1;
    if (idx !== cur) {
      cur = idx;
      links.forEach((a, i) => { if (i === idx) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); });
      const a = links[idx];
      $('#nav-current').textContent = a ? `${a.querySelector('b').textContent} ${a.lastChild.textContent.trim()}` : 'Contents';
      toggle.setAttribute('aria-label', a ? `Browse sections. Current: ${a.lastChild.textContent.trim()}` : 'Browse all sections');
      if (a) {
        const l = a.offsetLeft - nav.offsetLeft, r = l + a.offsetWidth;
        if (l < nav.scrollLeft + 8 || r > nav.scrollLeft + nav.clientWidth - 8) nav.scrollTo({ left: Math.max(0, l - (nav.clientWidth - a.offsetWidth) / 2), behavior: 'auto' });
      }
    }
  };
  const step = d => nav.scrollBy({ left: d * Math.max(120, nav.clientWidth * .7), behavior: reduced ? 'auto' : 'smooth' });
  aL.addEventListener('click', () => step(-1));
  aR.addEventListener('click', () => step(1));
  nav.addEventListener('scroll', arrows, { passive: true });
  addEventListener('scroll', () => { if (!raf) raf = requestAnimationFrame(update); }, { passive: true });
  addEventListener('resize', () => { if (!raf) raf = requestAnimationFrame(update); arrows(); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(arrows);
  update(); arrows();
}

// =============================================================================================
// 01 · recipe mixer
// =============================================================================================
const BB = Object.fromEntries(D.idea.bb.map(b => [b.k, b]));
// Generate first (method only; the JAM paper reports no results for it, and its Figure 1 draws it as the dashed loop
// of "some WAMs"). For Wan this is Fast-WAM-IDM (Yuan et al., 2026, Sec. 3.3 and 4.1): future video latents are
// denoised first and the action expert reads them; in training it reads the recorded future, noise-augmented with
// p = 0.5. The image generators are drawn with the same structure around a generated future image (the ImageWAM
// paper itself has no generate-first variant).
const mix = { bb: 'wan', obj: 'fore', inf: 'act', view: 'train' };
const G = {};
const LAYOUT = {
  wide: {
    vb: [1000, 432], img: 'xMidYMid slice',
    frame: { x: 14, y: 30, w: 698, wGen: 646, wGen1: 628, h: 216, lx: 698, ly: 22 },  // 30 units clear of the content on every side: head box (682), generated stack (630) or single image (612)
    rec: { x0: 44, y0: 80, w: 96, h: 96, d: 9, lx: 34, ly: 198, lfs: 20, ltx: 78, sx: 101, sy: 202, sdy: 26, sa: 'middle' }, fs: { sub: 13.5, lbl: 13, tag: 12.5, hex: 17, m: 19 }, // square thumbnails
    unused: { x: 47, y: 106, w: 108, h: 36, cx: 101, cy: 124, fs: 20 },
    anchor: { x: 206, y: 98, w: 196, h: 78 }, eRecAnchor: 'M162,137 H200', eRecAnchor1: 'M144,137 H200', // from the stack's outer edge (158), or the endpoint frame's (140)
    head: { x: 446, y: 98, w: 236, h: 78 }, eBbHead: 'M564,286 V182', hLbl: [580, 226], // backbone top -> centre of the head (or generated image); label at the middle of the in-frame part
    eAnchorHex: 'M304,96 V72 H404', eHeadHex: 'M564,96 V72 H460', hexF: [432, 72], // clear of the frame's top edge
    gLbl: [354, 62, 'middle'], ghLbl: [512, 62, 'middle'], // centred over their segments (304-404, 460-564)
    gen: { x0: 516, y0: 80, w: 96, h: 96, d: 9, lx: 576, ly: 198, ltx: 616, sx: 580, sy: 224, sa: 'start' }, genLbl: [580, 274, 'start'], // right of the arrow: f-hat mid in-frame part, 'denoise / decode' mid below the frame
    eRecHexG: 'M162,128 H311', eGenHexG: 'M510,128 H363', hexG: [337, 128], // L_G midway between the stacks (158 | 516): equal arrows
    eRecHexG1: 'M144,128 H302', eGenHexG1: 'M510,128 H354', hexG1: 328, //   single endpoint frames (140 | 516): L_G moves to their midpoint
    genX: [547, 196, 581, 230], genXLbl: [590, 218],
    loop: 'M634,128 H780 V280', loop1: 'M616,128 H780 V280', loopLbl: [790, 140, 'start'], // from the stack's (or the one image's) right edge, outside the frame; label by the corner
    loopT: 'M110,60 V44 H780 V280', loopT1: 'M110,78 V44 H780 V280', loopTLbl: [300, 63, 'middle'],  // training: over the top, from the recorded future (the stack, or its one endpoint frame)
    f0: { x: 32, y: 282, w: 92, h: 92, lx: 28, ly: 400, ltx: 52, sx: 78, sy: 398, sdy: 26, sa: 'middle' }, eF0Bb: 'M128,328 H190', // square
    bb: { x: 196, y: 286, w: 484, h: 84 }, eBbEx: 'M680,328 H754', kv: [717, 318],
    ex: { x: 760, y: 286, w: 200, h: 84 }, eExAct: 'M860,286 V196', // up to the middle dot of the action arc
    act: { x0: 800, y0: 214, w: 120, amp: 26, lx: 872, ly: 250, ltx: 918, sx: 872, sy: 238, sdy: 26, sa: 'start' }, // arc of 9 dots, apex (860, 188)
    lact: { x: 860, y: 120, line: [860, 144, 860, 183.8] }, deployLbl: [28, 270], hex: 24, tfs: 20, // down to the apex dot
  },
  tall: {
    vb: [420, 720], img: 'xMidYMid slice',
    frame: { x: 8, y: 30, w: 404, h: 360, lx: 410, ly: 22 },
    rec: { x0: 22, y0: 62, w: 82, h: 72, d: 6, lx: 16, ly: 160, lfs: 18, ltx: 54 }, fs: { sub: 13, lbl: 14, tag: 13, hex: 16, m: 18 },
    unused: { x: 18, y: 80, w: 94, h: 32, cx: 65, cy: 96, fs: 16 },
    anchor: { x: 126, y: 58, w: 180, h: 70 }, eRecAnchor: 'M110,93 H122',
    head: { x: 126, y: 206, w: 180, h: 70 }, eBbHead: 'M225,434 V282', hLbl: [233, 372],
    eAnchorHex: 'M306,93 H364 V144', eHeadHex: 'M306,241 H364 V190', hexF: [364, 167],
    gLbl: [312, 84], ghLbl: [312, 232],
    gen: { x0: 168, y0: 206, w: 82, h: 72, d: 6, lx: 16, ly: 304, ltx: 52 }, genLbl: [233, 372],
    eRecHexG: 'M62,168 V213', eGenHexG: 'M164,236 H83', hexG: [62, 236], // starts under the 'recorded future' label
    genX: [210, 318, 240, 348], genXLbl: [248, 338],
    loop: 'M266,236 H404 V596 H398', loop1: 'M254,236 H404 V596 H398', loopLbl: [398, 320, 'end'], // above 'denoise / decode' (no overlap on phones)
    loopT: 'M118,93 H404 V596 H398', loopT1: 'M106,93 H404 V596 H398', loopTLbl: [398, 84, 'end'],     // training: from the recorded future (stack or endpoint)
    f0: { x: 16, y: 438, w: 92, h: 82, lx: 16, ly: 540, ltx: 38 }, eF0Bb: 'M112,479 H132',
    bb: { x: 138, y: 436, w: 254, h: 86 }, eBbEx: 'M265,522 V554', kv: [275, 543],
    ex: { x: 138, y: 560, w: 254, h: 72 }, eExAct: 'M265,632 V660',
    act: { x0: 180, y0: 680, w: 170, amp: 12, lx: 16, ly: 706, ltx: 50 },
    lact: { x: 376, y: 680, line: [354.2, 680, 359.5, 680] }, deployLbl: [16, 424], hex: 19, tfs: 16, // last dot -> hexagon
  },
};
let mixTall = () => false;
function hexNode(parent, cx, cy, r, kind, segs, fs) {
  const pts = [];
  for (let k = 0; k < 6; k++) { const a = Math.PI / 6 + k * Math.PI / 3; pts.push(`${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`); }
  S('polygon', { points: pts.join(' '), cls: 'hexa ' + kind }, parent);
  mt(parent, cx, cy + fs * .36, segs, fs, { anchor: 'middle', cls: 'hex-lbl' });
}
function buildMixer() {
  const svg = $('#mix-svg');
  const L = mixTall() ? LAYOUT.tall : LAYOUT.wide;
  svg.querySelectorAll(':scope > *').forEach(n => n.remove());
  svg.setAttribute('viewBox', `0 0 ${L.vb[0]} ${L.vb[1]}`);
  marker(svg, 'ah-ink', 'ah-ink'); marker(svg, 'ah-act', 'ah-act'); marker(svg, 'ah-fore', 'ah-fore'); marker(svg, 'ah-gen', 'ah-gen');
  const defs = svg.querySelector('defs');
  const cp = (id, x, y, w, h) => { const c = S('clipPath', { id }, defs); S('rect', { x, y, width: w, height: h, rx: 7 }, c); };
  const stackXY = (s, k) => [s.x0 + s.d * (2 - k), s.y0 - s.d * (2 - k)];
  cp('cp-f0', L.f0.x, L.f0.y, L.f0.w, L.f0.h);
  for (let k = 0; k < 3; k++) { cp('cp-r' + k, ...stackXY(L.rec, k), L.rec.w, L.rec.h); cp('cp-g' + k, ...stackXY(L.gen, k), L.gen.w, L.gen.h); }
  const f = S('filter', { id: 'noisy', x: 0, y: 0, width: '100%', height: '100%' }, defs);
  S('feTurbulence', { type: 'fractalNoise', baseFrequency: .9, numOctaves: 2, seed: 4, result: 'n' }, f);
  S('feColorMatrix', { in: 'n', type: 'saturate', values: .4, result: 'nc' }, f);
  S('feBlend', { in: 'SourceGraphic', in2: 'nc', mode: 'multiply', result: 'b' }, f);
  S('feComposite', { in: 'b', in2: 'SourceGraphic', operator: 'in' }, f);
  const tsz = L.tfs, hr = L.hex;
  const title = (g, x, y, t, sm) => S('text', { x, y, 'text-anchor': 'middle', cls: 't-title' + (sm ? ' sm' : ''), text: t, 'font-size': sm ? tsz - 2 : tsz }, g);
  const lbl = (g, x, y, t, fill, anchor) => S('text', { x, y, cls: 't-lbl', text: t, style: fill ? `fill:${fill}` : null, 'text-anchor': anchor || null, 'font-size': L.fs.lbl }, g);
  const sub = (g, x, y, t, style) => S('text', { x, y, 'text-anchor': 'middle', cls: 't-sub', text: t, style: style || null, 'font-size': L.fs.sub }, g);
  const pairMath = (g, P, segs, fill) => P.sx != null ? mt(g, P.sx, P.sy, segs, L.rec.lfs, { anchor: P.sa, fill }) : mt(g, P.lx, P.ly, segs, L.rec.lfs, { fill });
  const pairText = (g, P, t, fill) => P.sx != null ? lbl(g, P.sx, P.sy + P.sdy, t, fill, P.sa) : lbl(g, P.ltx, P.ly - 1, t, fill);
  // training frame
  G.frame = S('g', { cls: 'tl' }, svg);
  G.frameRect = S('rect', { x: L.frame.x, y: L.frame.y, width: L.frame.w, height: L.frame.h, rx: 14, cls: 'frame-dash' }, G.frame);
  G.frameLbl = S('text', { x: L.frame.lx, y: L.frame.ly, cls: 't-tag', 'text-anchor': 'end', style: 'fill:var(--c-fore)', text: 'TRAINING ONLY', 'font-size': L.fs.tag }, G.frame);
  // recorded future stack (real LIBERO-Long demonstration frames: +48, +32, +16 steps from back to front)
  G.rec = S('g', { cls: 'tl' }, svg);
  const recImgs = ['assets/img/demo-moka-88.jpg', 'assets/img/demo-moka-72.jpg', 'assets/img/demo-moka-56.jpg'];
  G.recBack = [];
  for (let k = 0; k < 3; k++) {
    const [x, y] = stackXY(L.rec, k);
    const im = S('image', { href: recImgs[k], x, y, width: L.rec.w, height: L.rec.h, preserveAspectRatio: L.img, 'clip-path': `url(#cp-r${k})` }, G.rec);
    const ed = S('rect', { x, y, width: L.rec.w, height: L.rec.h, rx: 7, cls: 'thumb-edge', style: 'stroke:var(--c-fore)' }, G.rec);
    if (k < 2) G.recBack.push(im, ed);
  }
  // the FastWAM recipe (Wan) uses the recorded window f1:T; the ImageWAM recipe (every other backbone) only its
  // endpoint fT, the frame 16 steps ahead (the front thumbnail)
  G.recM = [pairMath(G.rec, L.rec, MATH.f1T), pairMath(G.rec, L.rec, MATH.fT)];
  G.recT = pairText(G.rec, L.rec, 'recorded future');
  // captions centre under the stack, or under the single endpoint frame (the front thumbnail)
  G.recX = L.rec.sx != null ? [L.rec.sx, L.rec.x0 + L.rec.w / 2] : null;
  G.unused = S('g', { cls: 'tl', transform: `rotate(-7 ${L.unused.cx} ${L.unused.cy})` }, svg);
  S('rect', { x: L.unused.x, y: L.unused.y, width: L.unused.w, height: L.unused.h, rx: 6, cls: 'stampbox' }, G.unused);
  S('text', { x: L.unused.x + L.unused.w / 2, y: L.unused.y + L.unused.h / 2 + L.unused.fs * .36, 'text-anchor': 'middle', cls: 'stamp-t', text: 'UNUSED', 'font-size': L.unused.fs }, G.unused);
  // foresight branch
  G.fore = S('g', { cls: 'tl' }, svg);
  const A = L.anchor, Hd = L.head;
  S('rect', { x: A.x, y: A.y, width: A.w, height: A.h, rx: 14, cls: 'box fore' }, G.fore);
  title(G.fore, A.x + A.w / 2, A.y + A.h * .45, 'Future anchor', 1);
  sub(G.fore, A.x + A.w / 2, A.y + A.h * .77, 'frozen · cached', 'fill:var(--amber)');
  G.recAnchorPaths = [S('path', { d: L.eRecAnchor, cls: 'edge fore', 'marker-end': 'url(#ah-fore)' }, G.fore), S('path', { d: L.eRecAnchor, cls: 'flow fore' }, G.fore)];
  S('rect', { x: Hd.x, y: Hd.y, width: Hd.w, height: Hd.h, rx: 14, cls: 'box fore' }, G.fore);
  title(G.fore, Hd.x + Hd.w / 2, Hd.y + Hd.h * .45, 'Prediction head', 1);
  sub(G.fore, Hd.x + Hd.w / 2, Hd.y + Hd.h * .77, 'small transformer'); // Table 8: a two-layer transformer trunk
  S('path', { d: L.eBbHead, cls: 'edge act', 'marker-end': 'url(#ah-act)' }, G.fore);
  S('path', { d: L.eBbHead, cls: 'flow act' }, G.fore);
  G.hLblEl = mt(G.fore, L.hLbl[0], L.hLbl[1], MATH.h, L.fs.m, { fill: 'var(--cyan)' });
  S('path', { d: L.eAnchorHex, cls: 'edge fore', 'marker-end': 'url(#ah-fore)' }, G.fore);
  S('path', { d: L.eHeadHex, cls: 'edge act', 'marker-end': 'url(#ah-act)' }, G.fore);
  mt(G.fore, L.gLbl[0], L.gLbl[1], MATH.g, L.fs.m - 1, { fill: 'var(--amber)', anchor: L.gLbl[2] });
  mt(G.fore, L.ghLbl[0], L.ghLbl[1], MATH.gh, L.fs.m - 1, { fill: 'var(--cyan)', anchor: L.ghLbl[2] });
  hexNode(G.fore, L.hexF[0], L.hexF[1], hr, 'fore', MATH.LF, L.fs.hex);
  // generation branch: a video generator denoises future frames, an image generator one future image
  G.gen = S('g', { cls: 'tl' }, svg);
  G.genBack = [];
  for (let k = 0; k < 3; k++) {
    const [x, y] = stackXY(L.gen, k);
    const im = S('image', { href: recImgs[k], x, y, width: L.gen.w, height: L.gen.h, preserveAspectRatio: L.img, 'clip-path': `url(#cp-g${k})`, filter: 'url(#noisy)' }, G.gen);
    const ed = S('rect', { x, y, width: L.gen.w, height: L.gen.h, rx: 7, cls: 'thumb-edge', style: 'stroke:var(--c-gen)' }, G.gen);
    if (k < 2) G.genBack.push(im, ed);
  }
  G.genM = [pairMath(G.gen, L.gen, MATH.fh1T, 'var(--violet)'), pairMath(G.gen, L.gen, MATH.fhT, 'var(--violet)')];
  S('path', { d: L.eBbHead, cls: 'edge gen', 'marker-end': 'url(#ah-gen)' }, G.gen);
  S('path', { d: L.eBbHead, cls: 'flow gen' }, G.gen);
  G.genLblEl = lbl(G.gen, L.genLbl[0], L.genLbl[1], 'denoise / decode', 'var(--violet)', L.genLbl[2]);
  G.genLoss = S('g', { cls: 'tl' }, svg);
  G.recHexPath = S('path', { d: L.eRecHexG, cls: 'edge fore', 'marker-end': 'url(#ah-fore)' }, G.genLoss);
  G.genHexPath = S('path', { d: L.eGenHexG, cls: 'edge gen', 'marker-end': 'url(#ah-gen)' }, G.genLoss);
  G.hexGNodes = [0, 1].map(i => { const g = S('g', null, G.genLoss); hexNode(g, i && L.hexG1 != null ? L.hexG1 : L.hexG[0], L.hexG[1], hr, 'gen', MATH.LG, L.fs.hex); return g; });
  // one recorded endpoint frame (every backbone but Wan): the arrows start at its edge and L_G recentres
  G.endD = { recAnchor: [L.eRecAnchor, L.eRecAnchor1 || L.eRecAnchor], recHex: [L.eRecHexG, L.eRecHexG1 || L.eRecHexG],
    genHex: [L.eGenHexG, L.eGenHexG1 || L.eGenHexG] };
  G.genX = S('g', { cls: 'tl' }, svg);
  const [x1, y1, x2, y2] = L.genX;
  S('line', { x1, y1, x2, y2, cls: 'xmark' }, G.genX); S('line', { x1: x2, y1, x2: x1, y2, cls: 'xmark' }, G.genX);
  lbl(G.genX, L.genXLbl[0], L.genXLbl[1], 'no decoder', 'var(--magenta)');
  // generate-first (Fast-WAM-IDM): the action expert reads futures. In training these are the recorded futures
  // (noise-augmented); at inference, the generated ones
  G.loopTrain = S('g', { cls: 'tl' }, svg);
  G.loopTPaths = [S('path', { d: L.loopT, cls: 'edge fore', 'stroke-dasharray': '7 6', 'marker-end': 'url(#ah-fore)' }, G.loopTrain), S('path', { d: L.loopT, cls: 'flow fore' }, G.loopTrain)];
  G.loopTD = [L.loopT, L.loopT1];
  G.loopTLblEl = lbl(G.loopTrain, L.loopTLbl[0], L.loopTLbl[1], 'recorded futures → actions', 'var(--amber)', L.loopTLbl[2]);
  G.loop = S('g', { cls: 'tl' }, svg);
  G.loopPaths = [S('path', { d: L.loop, cls: 'edge gen', 'stroke-dasharray': '7 6', 'marker-end': 'url(#ah-gen)' }, G.loop), S('path', { d: L.loop, cls: 'flow gen' }, G.loop)];
  G.loopD = [L.loop, L.loop1];
  G.loopLblEl = lbl(G.loop, L.loopLbl[0], L.loopLbl[1], 'futures → actions', 'var(--violet)', L.loopLbl[2]);
  // policy path (always deployed)
  G.pol = S('g', null, svg);
  const live = S('g', { cls: 'live-path' }, G.pol);
  S('image', { href: 'assets/img/demo-moka-40.jpg', x: L.f0.x, y: L.f0.y, width: L.f0.w, height: L.f0.h, preserveAspectRatio: L.img, 'clip-path': 'url(#cp-f0)' }, G.pol);
  S('rect', { x: L.f0.x, y: L.f0.y, width: L.f0.w, height: L.f0.h, rx: 7, cls: 'thumb-edge', style: 'stroke:var(--c-act)' }, G.pol);
  pairMath(G.pol, L.f0, MATH.f0);
  pairText(G.pol, L.f0, 'current frame');
  S('path', { d: L.eF0Bb, cls: 'edge act', 'marker-end': 'url(#ah-act)' }, live);
  S('path', { d: L.eF0Bb, cls: 'flow act' }, live);
  const B = L.bb;
  G.bbBox = S('rect', { x: B.x, y: B.y, width: B.w, height: B.h, rx: 14, cls: 'box gen' }, live);
  title(G.pol, B.x + B.w / 2, B.y + B.h * .35, 'Backbone');
  G.bbName = sub(G.pol, B.x + B.w / 2, B.y + B.h * .62, '', 'font-weight:600'); // spaced for the enlarged labels of narrow layouts
  G.bbFam = sub(G.pol, B.x + B.w / 2, B.y + B.h * .87, '');
  S('path', { d: L.eBbEx, cls: 'edge act', 'marker-end': 'url(#ah-act)' }, live);
  S('path', { d: L.eBbEx, cls: 'flow act' }, live);
  lbl(G.pol, L.kv[0], L.kv[1], 'KV', null, mixTall() ? 'start' : 'middle');
  const E = L.ex;
  S('rect', { x: E.x, y: E.y, width: E.w, height: E.h, rx: 14, cls: 'box act' }, live);
  title(G.pol, E.x + E.w / 2, E.y + E.h / 2 + tsz * .34, 'Action expert');
  S('path', { d: L.eExAct, cls: 'edge act', 'marker-end': 'url(#ah-act)' }, live);
  // the action chunk: a symmetric arc with one dot per predicted step, its middle dot on the expert's arrow
  const arc = u => [L.act.x0 + u * L.act.w, L.act.y0 - Math.sin(u * Math.PI) * L.act.amp];
  const curve = Array.from({ length: 49 }, (_, k) => arc(k / 48)), dots = Array.from({ length: 9 }, (_, k) => arc(k / 8));
  S('polyline', { points: curve.map(p => p.map(n => n.toFixed(1)).join(',')).join(' '), fill: 'none', style: 'stroke:var(--c-act)', 'stroke-width': 2, opacity: .7 }, live);
  for (const p of dots) S('circle', { cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: 4.2, style: 'fill:var(--cyan)' }, live);
  pairMath(G.pol, L.act, MATH.a1H);
  pairText(G.pol, L.act, 'actions');
  G.lact = S('g', { cls: 'tl' }, svg);
  const [lx1, ly1, lx2, ly2] = L.lact.line;
  S('line', { x1: lx1, y1: ly1, x2: lx2, y2: ly2, style: 'stroke:var(--c-act)', 'stroke-width': 1.5, opacity: .7 }, G.lact);
  hexNode(G.lact, L.lact.x, L.lact.y, hr, 'act', MATH.Lact, L.fs.hex - 1);
  G.deployLbl = S('text', { x: L.deployLbl[0], y: L.deployLbl[1], cls: 't-tag tl', style: 'fill:var(--cyan)', text: 'RUNS ON THE ROBOT', 'font-size': L.fs.tag }, svg);
}
// labels on the rising arrow sit at the middle of its part inside (h, f-hat) or below ('denoise / decode') the
// training frame. Their size follows the page width (legible), so each baseline is set from the current font size;
// k is the label's ink centre above its baseline, in em (measured on renders).
function placeArrowLabels() {
  if (!G.genLblEl || mixTall()) return;
  const L = LAYOUT.wide, fb = L.frame.y + L.frame.h, tip = +L.eBbHead.match(/V([\d.]+)/)[1] - 1.8;
  const put = (el, y0, y1, k) => { if (el) el.setAttribute('y', ((y0 + y1) / 2 + k * (parseFloat(getComputedStyle(el).fontSize) || 16)).toFixed(2)); };
  put(G.hLblEl, tip, fb, .47); G.genM.forEach(e => put(e, tip, fb, .29)); put(G.genLblEl, fb, L.bb.y, .37);
}
const show = (g, on, dim) => { g.style.opacity = on ? (dim ? .13 : 1) : 0; g.style.visibility = on ? '' : 'hidden'; };

function renderMixer() {
  // the dials show methods, not results: every recipe that exists as a method can be picked. Only a visual
  // generator has a generation loss (switching to another backbone falls back to foresight), and generating first
  // needs that loss
  const avail = { none: true, gen: BB[mix.bb].kind === 'gen', fore: true };
  if (!avail[mix.obj]) mix.obj = 'fore';
  const genFirstOk = avail.gen && mix.obj === 'gen';
  if (!genFirstOk) mix.inf = 'act';
  const m = BB[mix.bb], obj = mix.obj;
  const canGen = m.kind === 'gen';
  $('#mix-loss').classList.toggle('on', obj === 'fore'); // JAM's own loss is selected
  press($('#dial-bb'), mix.bb); press($('#dial-obj'), obj); press($('#dial-inf'), mix.inf); press($('#mix-view'), mix.view);
  $('#dial-obj').querySelectorAll('button').forEach(b => b.setAttribute('aria-disabled', String(!avail[b.dataset.v])));
  $('#dial-inf').querySelector('[data-v="gen"]').setAttribute('aria-disabled', String(!genFirstOk));
  const dep = mix.view === 'deploy';
  $('#mix-svg').dataset.view = mix.view;
  G.bbName.textContent = m.name;
  G.bbFam.textContent = m.fam;
  G.bbName.style.fill = 'var(--violet)';
  const invalid = obj === 'gen' && !canGen;
  const genAtInf = obj === 'gen' && mix.inf === 'gen' && canGen;
  show(G.rec, true, dep || obj === 'none');
  show(G.unused, obj === 'none' && !dep);
  show(G.fore, obj === 'fore', dep);
  show(G.gen, obj === 'gen', (dep && !genAtInf) || invalid);
  show(G.genLoss, obj === 'gen', dep || invalid);
  show(G.genX, invalid);
  G.genLblEl.style.visibility = invalid ? 'hidden' : '';
  const oneImage = m.fam === 'image generator'; // ImageWAM-style generators produce a single future image
  const endpoint = m.fam !== 'video generator'; // ImageWAM recipe: targets from the recorded endpoint only
  G.genBack.forEach(e => { e.style.display = endpoint ? 'none' : ''; });
  G.genM[0].style.display = endpoint ? 'none' : ''; G.genM[1].style.display = endpoint ? '' : 'none';
  const ei = endpoint ? 1 : 0;
  G.recAnchorPaths.forEach(p => p.setAttribute('d', G.endD.recAnchor[ei]));
  G.recHexPath.setAttribute('d', G.endD.recHex[ei]); G.genHexPath.setAttribute('d', G.endD.genHex[ei]);
  G.hexGNodes.forEach((g, i) => { g.style.display = i === ei ? '' : 'none'; });
  G.recBack.forEach(e => { e.style.display = endpoint ? 'none' : ''; });
  G.recM[0].style.display = endpoint ? 'none' : ''; G.recM[1].style.display = endpoint ? '' : 'none';
  if (G.recX) [...G.recM, G.recT].forEach(e => e.setAttribute('x', G.recX[endpoint ? 1 : 0]));
  G.loopTPaths.forEach(p => p.setAttribute('d', G.loopTD[endpoint ? 1 : 0]));
  G.loopTLblEl.textContent = endpoint ? 'recorded future → actions' : 'recorded futures → actions';
  G.loopLblEl.textContent = oneImage ? 'future → actions' : 'futures → actions';
  G.loopPaths.forEach(p => p.setAttribute('d', G.loopD[oneImage ? 1 : 0]));
  show(G.loop, genAtInf && dep);
  show(G.loopTrain, genAtInf && !dep);
  show(G.frame, obj !== 'none', dep && !genAtInf);
  show(G.lact, !dep, false);
  show(G.deployLbl, dep);
  G.frameLbl.textContent = genAtInf ? 'GENERATION ALSO RUNS AT INFERENCE' : 'TRAINING ONLY';
  G.frameLbl.style.fill = genAtInf ? 'var(--violet)' : 'var(--c-fore)';
  G.frameRect.style.stroke = genAtInf ? 'var(--c-gen)' : '';
  const F = (mixTall() ? LAYOUT.tall : LAYOUT.wide).frame, fw = obj === 'gen' ? (endpoint && F.wGen1 ? F.wGen1 : F.wGen || F.w) : F.w; // hugs the content
  G.frameRect.setAttribute('width', fw); G.frameLbl.setAttribute('x', F.lx + fw - F.w);
  renderRecipe(invalid, genAtInf);
  legible($('#mix-svg'));
  placeArrowLabels();
}

function renderRecipe(invalid, genAtInf) {
  const m = BB[mix.bb], obj = mix.obj;
  const R = D.idea.res[mix.bb] || {}, r = R[obj];
  let tag, tcls, name, desc;
  // one pattern for every recipe: what the backbone learns from the recorded future, then what the action expert reads
  // (FastWAM masks the future tokens from the expert; ImageWAM skips generation at inference; Fast-WAM-IDM reads the
  // noise-augmented recorded future in training and the generated one at deployment; JAM's head and anchor are dropped)
  const curOnly = 'The action expert reads only the current frame.';
  if (invalid) {
    tag = 'N/A'; tcls = 't-no'; name = 'Needs a visual generator';
    desc = `${m.name} has no decoder to generate the future. The paper trains it with the foresight loss instead.`;
  } else if (obj === 'none') {
    if (m.kind === 'vlm') { tag = 'VLA'; tcls = 't-vla'; name = 'Vision-language-action'; }
    else { tag = 'ACTION ONLY'; tcls = 't-ao'; name = 'Action-only policy'; }
    desc = `The backbone learns nothing from the recorded future. ${curOnly}`;
  } else if (obj === 'gen') {
    tag = 'WAM'; tcls = 't-wam';
    name = genAtInf ? 'WAM, generating at test time' : 'World-action model';
    desc = `The backbone also learns to denoise the recorded future ${m.fam === 'video generator' ? 'video' : 'image'}. ` +
      (genAtInf ? 'The action expert reads that future: recorded in training, generated at deployment.' : curOnly);
  } else {
    tag = 'JAM'; tcls = ''; name = 'Just Action Model';
    desc = `The backbone, through a small head, learns to predict frozen features of the recorded future. ${curOnly}`;
  }
  const tg = $('#rc-tag'); tg.textContent = tag; tg.className = 'tag ' + (tcls || '');
  $('#rc-model').textContent = `${m.name} · ${m.fam}`;
  $('#rc-name').textContent = name;
  $('#rc-desc').textContent = desc;
  $('#rc-desc').title = genAtInf && mix.bb === 'wan' ? `“${D.idea.speed.q.t}” (${D.idea.speed.q.w})`
    : obj === 'gen' && !invalid ? `“${D.idea.objq.t}” (${D.idea.objq.w})` : '';
  // method only: the training objective (paper Eqs. 1-3) and what runs on the robot; results live in section 02
  const Lm = x => `<span class="m">ℒ<sub>${x}</sub></span>`, lm = x => `<span class="m">λ<sub>${x}</sub></span>`;
  $('#rc-obj').innerHTML = invalid ? '—'
    : obj === 'none' ? Lm('act') : obj === 'gen' ? `${Lm('act')} + ${lm('G')}${Lm('G')}` : `${Lm('act')} + ${lm('F')}${Lm('F')}`;
  // what each recipe does with the future in training and what runs on the robot (method facts, Sec. 3): JAM predicts frozen
  // features and deploys exactly the action policy; a WAM denoises video or generates an image at every update, and a
  // generate-first WAM also does so before each action chunk. Tone: cyan = JAM's advantage, violet = generation cost
  const img = m.fam === 'image generator';
  const put = (id, text, tone) => { const e = $(id); e.textContent = text; e.className = tone || ''; };
  if (invalid) { put('#rc-train', '—'); put('#rc-dep', '—'); }
  else if (obj === 'fore') { put('#rc-train', 'Frozen future features, no generation', 'pro'); put('#rc-dep', 'Backbone + action expert, no extra cost', 'pro'); }
  else if (obj === 'gen') {
    put('#rc-train', img ? 'Generates a future image every update' : 'Denoises future video every update', 'con');
    put('#rc-dep', genAtInf ? 'Generates the future before each action chunk' : 'Backbone + action expert', genAtInf ? 'con' : '');
  } else { put('#rc-train', 'Action loss only, the future goes unused'); put('#rc-dep', 'Backbone + action expert'); }
}

function initMixer() {
  // the foresight loss (paper Eq. 3) under the diagram, whose g and ĝ arrows it defines
  const L = mh(MATH.LF), gh = [['ĝ', 'i'], ['k,n', 'si']], g = [['g', 'i'], ['k,n', 'si']];
  $('#mix-eq').innerHTML = `${L} <span class="m">=</span> <span class="m frac"><span>1</span><span>|<i>K</i>|</span></span>` +
    `<span class="m">∑<sub><i>k</i>∈<i>K</i></sub> 𝔼<sub><i>n</i></sub>[1 − cos(</span>${mh(gh)}<span class="m">, sg[</span>${mh(g)}<span class="m">])]</span>`;
  // three keys across the width: the prediction (cyan, as the diagram's ĝ arrow), the target (amber, as its g arrow), the distance
  const key = (sym, cls, text) => `<div class="ml-key"><dt class="${cls}">${sym}</dt><dd>${text}</dd></div>`;
  $('#mix-loss-keys').innerHTML =
    key(mh(gh), 'pred', 'Prediction: the head reads the current frame’s tokens and predicts the feature at readout offset <i class="m">k</i>, token <i class="m">n</i>.') +
    key(`<span class="m">sg[</span>${mh(g)}<span class="m">]</span>`, 'tgt', 'Target: the frozen anchor’s feature of the recorded future at that offset; sg stops its gradient.') +
    key('<span class="m">1 − cos</span>', '', 'Cosine distance: only direction counts, averaged over the offsets in <i class="m">K</i> and the tokens <i class="m">n</i>.');
  mixTall = watchNarrow($('#mix-scroll'), 600, () => { buildMixer(); renderMixer(); });
  buildMixer();
  seg($('#dial-bb'), D.idea.bb.map(b => ({ v: b.k, label: b.name, sub: b.fam })), v => { mix.bb = v; renderMixer(); }, mix.bb);
  seg($('#dial-obj'), [
    { v: 'none', label: 'None', sub: 'action loss only', k: 'none' },
    { v: 'gen', html: 'Generate it <span class="m">ℒ<sub>G</sub></span>', sub: 'visual generation', k: 'gen', title: 'Generation loss (ℒG): denoise future video latents or generate a future image',
      why: () => `${BB[mix.bb].name} is not a visual generator, so it has no generation loss.` },
    { v: 'fore', html: 'Predict its features <span class="m">ℒ<sub>F</sub></span>', sub: 'foresight (JAM)', k: 'fore', title: 'Foresight loss (ℒF): predict frozen features of the recorded future' },
  ], v => { mix.obj = v; renderMixer(); }, mix.obj);
  seg($('#dial-inf'), [
    { v: 'act', label: 'Actions only', sub: 'backbone + expert' },
    { v: 'gen', label: 'Generate first', sub: 'futures → actions', k: 'gen', why: () => BB[mix.bb].kind === 'gen' ? 'Pick “Generate it” first: generating at inference needs the generation loss.' : `${BB[mix.bb].name} is not a visual generator, so there is nothing to generate first.` },
  ], v => { mix.inf = v; renderMixer(); }, mix.inf);
  $('#mix-view').querySelectorAll('button').forEach(b => b.addEventListener('click', () => { mix.view = b.dataset.v; renderMixer(); }));
  const n = D.idea.n, fam = D.idea.fam;
  big('#idea-big', esc(String(fam.v)), 'Backbone families · one objective',
    () => `<b>${esc(String(fam.v))} backbone families</b>: “${esc(fam.p)}” (${esc(where(fam))}), with one foresight objective.` +
      `<br>${n.p} backbones: ${D.idea.bb.map(b => esc(b.name)).join(', ')}, one JAM policy each (${esc(n.w)} lists their prediction heads).`);
  renderMixer();
  // the flowing dashes on the diagram's edges run only while the mixer is on screen
  if ('IntersectionObserver' in window) new IntersectionObserver(es => es.forEach(e => $('.mixer').classList.toggle('off', !e.isIntersecting))).observe($('.mixer'));
}

// =============================================================================================
// 03 · per backbone: success by loss on the future (Tables 1, 3, 14, 19)
// =============================================================================================
const PB_TABS = { plus: ['LIBERO-Plus', 'plus'], libero: ['LIBERO', 'lib'], robotwin: ['RoboTwin 2.0', 'rt'] };
const PB_OBJ = [['none', 'No future loss', 'var(--c-base)'], ['gen', 'Generation', 'var(--violet)'], ['fore', 'Foresight (JAM)', 'var(--cyan)']];
let pbTab = 'plus', pbNarrow = () => false;
function pbRows() { // backbone groups with the values the paper prints for this benchmark
  const key = PB_TABS[pbTab][1], out = [];
  for (const b of D.idea.bb) {
    const R = D.idea.res[b.k] || {}, rows = [];
    for (const [k, label, col] of PB_OBJ) {
      const rr = R[k], x = rr && rr[key]; if (x) rows.push({ k, label, col, x, rr });
    }
    if (rows.length) out.push({ b, rows });
  }
  return out;
}
function renderPerBackbone() {
  const [bench, key] = PB_TABS[pbTab];
  press($('#pb-tab'), pbTab);
  const groups = pbRows(), vals = groups.flatMap(g => g.rows.map(r => r.x.v));
  const span = Math.max(...vals) - Math.min(...vals), step = span > 20 ? 10 : span > 6 ? 2 : 1;
  const lo = Math.floor((Math.min(...vals) - step) / step) * step, hi = Math.min(100, Math.ceil((Math.max(...vals) + step / 2) / step) * step);
  const ticks = []; for (let v = lo; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(2));
  const sv = $('#pb-svg');
  sv.querySelectorAll(':scope > *').forEach(n => n.remove());
  const nar = pbNarrow();
  const VW = nar ? 420 : 1000, rh = nar ? 30 : 27, x0 = nar ? 168 : 250, x1 = nar ? 372 : 920, nx = nar ? 0 : 14, fs = nar ? 14.5 : 13.5;
  const X = v => x0 + (x1 - x0) * (v - lo) / (hi - lo);
  const GAP = 12; // extra space above every group after the first
  const nRows = groups.reduce((a, g) => a + 1 + g.rows.length, 0), Hh = 24 + nRows * rh + 30 + GAP * (groups.length - 1);
  sv.setAttribute('viewBox', `0 0 ${VW} ${Hh}`);
  $('#pb-svg').setAttribute('aria-label', `${bench}: success by loss on the future for each backbone`);
  // every tick keeps its grid line; labels thin out so they never touch on a phone
  const every = Math.max(1, Math.ceil(46 / ((x1 - x0) / Math.max(1, ticks.length - 1))));
  ticks.forEach((v, i) => {
    S('line', { x1: X(v), y1: 14, x2: X(v), y2: Hh - 30, cls: 'grid-l' }, sv);
    if (i % every === 0) S('text', { x: X(v), y: Hh - 10, 'text-anchor': 'middle', cls: 'ax', text: v + '%' }, sv);
  });
  const notes = { // evaluation protocol, shown in each row's tooltip
    plus: 'Fixed 1,506-task subset, one rollout each',
    libero: '40 tasks × 50 rollouts',
    robotwin: 'Clean/Randomized mean (Table 19); FastWAM and ImageWAM (FLUX.2) use released weights, ImageWAM (Ovis-U1) was trained locally',
  };
  let y = 26;
  groups.forEach((g, gi) => {
    if (gi) { y += GAP; S('line', { x1: 0, y1: y - rh / 2 - GAP / 2, x2: VW, y2: y - rh / 2 - GAP / 2, cls: 'grp-sep' }, sv); }
    // group header: accent bar, backbone name, family chip
    S('rect', { x: 0, y: y - 7, width: 3, height: 15, cls: 'grp-bar' }, sv);
    const nm = S('text', { x: 10, y: y + 5, cls: 'grp-name', text: g.b.name }, sv);
    const fx = 10 + nm.getComputedTextLength() + 10;
    const chip = S('rect', { x: fx, y: y - 8, height: 17, rx: 3, cls: 'grp-chip' }, sv);
    const ft = S('text', { x: fx + 7, y: y + 4, cls: 'grp-fam', text: g.b.fam.toUpperCase() }, sv);
    chip.setAttribute('width', (ft.getComputedTextLength() + 14).toFixed(1));
    y += rh;
    for (const r of g.rows) {
      const jam = r.k === 'fore', gg = S('g', null, sv);
      S('rect', { x: 0, y: y - rh / 2, width: VW, height: rh, cls: 'hit' }, gg);
      S('text', { x: nx, y: y + 5, cls: jam ? 'cat jam' : 'cat', 'font-size': fs, text: r.label }, gg);
      S('line', { x1: x0, y1: y, x2: X(r.x.v), y2: y, style: `stroke:${jam ? 'var(--cyan)' : 'var(--line-2)'}`, 'stroke-width': jam ? 2.5 : 1.5, opacity: jam ? .85 : 1 }, gg);
      const dot = jam ? 'fill:var(--panel);stroke:var(--cyan);stroke-width:3;filter:drop-shadow(0 0 5px var(--cyan))' : `fill:${r.col}`;
      S('circle', { cx: X(r.x.v), cy: y, r: jam ? 6 : 5, style: dot }, gg);
      S('text', { x: X(r.x.v) + 11, y: y + 4.5, cls: 'v', style: `fill:${jam ? 'var(--ink)' : 'var(--ink-2)'};font-weight:${jam ? 650 : 400}`, text: r.x.p }, gg);
      bindTip(gg, () => `<b>${pc(r.x)}</b> ${esc(bench)} · ${esc(r.rr.label)}${r.rr.cfg ? ', ' + esc(r.rr.cfg) : ''}${key === 'rt' && r.rr.rtNote ? ' · ' + esc(r.rr.rtNote) : ''}${tipSrc(where(r.x))}<span class="s">${esc(notes[pbTab])}</span>`,
        `${g.b.name}, ${r.label}: ${r.x.p}%`, jam ? 'var(--cyan)' : r.col);
      y += rh;
    }
  });
  finish(sv);
}
function initPerBackbone() {
  pbNarrow = watchNarrow($('#pb-scroll'), 620, renderPerBackbone);
  seg($('#pb-tab'), Object.entries(PB_TABS).map(([v, t]) => ({ v, label: t[0] })), v => { pbTab = v; renderPerBackbone(); }, pbTab);
  const lg = $('#pb-legend');
  for (const [l, st] of [['No future loss', 'background:var(--c-base)'], ['Generation', 'background:var(--violet)'], ['Foresight (JAM)', 'background:var(--panel);border:2px solid var(--cyan);box-shadow:0 0 6px var(--cyan)']]) {
    const sp = H('span', null, lg); H('i', { style: st }, sp); sp.append(l);
  }
  renderPerBackbone();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(renderPerBackbone);
}

// =============================================================================================
// 09 · ablations: robustness card (Figure 6 counts, Table 6 rates, Table 20 statistics, Table 21 runs)
// =============================================================================================
const RB = D.robust;
let rbPair = 'ovis';
let rbNarrow = () => false;
const baseShort = k => (k === 'ovis' ? 'ImageWAM' : 'FastWAM');
function renderRobust() {
  const P = RB.pairs[rbPair], bs = baseShort(rbPair);
  press($('#rb-pair'), rbPair);
  big('#robust-big', `${esc(P.jTot.p)}<span class="sep"> vs </span>${esc(P.bTot.p)}`, `Tasks only JAM / only ${bs} solved`,
    () => `<b>${P.jTot.p} vs ${P.bTot.p}</b> of ${RB.tasks.p} shared LIBERO-Plus tasks were solved by only ${esc(P.jam)} or only ${esc(P.base)} (one rollout each). ${esc(P.jam)}: ${esc(P.cfg)} targets.${tipSrc(where(P.jTot))}<br>“${esc(RB.q.t)} ${esc(RB.qx.t)}” (${esc(RB.q.w)}; ${esc(RB.six.p)})`);
  // the wide plot labels its two sides; the narrow (phone) layout needs the colour key
  const lg = $('#rb-legend'); lg.textContent = '';
  for (const [l, c] of [[`only ${bs} solved`, 'var(--c-gen)'], ['only JAM solved', 'var(--cyan)']]) { const s = H('span', null, lg); H('i', { style: `background:${c}` }, s); s.append(l); }
  lg.hidden = !rbNarrow();
  const sv = $('#rb-svg');
  sv.querySelectorAll(':scope > *').forEach(n => n.remove());
  const rowTip = i => () => `<b>${esc(RB.cats[i])}</b><br>Only JAM: ${P.jo[i].p} tasks · only ${esc(bs)}: ${P.bo[i].p}<br>Success ${pc(P.rb[i])} → ${pc(P.rj[i])}${tipSrc(`${RB.catsW} · Table 6 · ${P.jam} vs ${P.base}`)}`;
  if (rbNarrow()) {
    let y = 6; const per = 25, pitch = 11.8, W = 360;
    RB.cats.forEach((c, i) => {
      const g = S('g', { cls: 'row' }, sv), y0 = y;
      S('rect', { x: 0, y: y0 - 4, width: W, height: 10, cls: 'hit' }, g);
      S('text', { x: 0, y: y + 13, cls: 'cat', text: c }, g);
      S('text', { x: W, y: y + 13, 'text-anchor': 'end', cls: 'rate' + (P.rj[i].v < P.rb[i].v ? ' down' : ''), text: `${P.rb[i].p} → ${P.rj[i].p}` }, g);
      y += 26;
      for (const [n, col, tcol] of [[P.jo[i].v, 'var(--cyan)', 'var(--cyan)'], [P.bo[i].v, 'var(--c-gen)', 'var(--violet)']]) {
        S('text', { x: 0, y: y + 5, cls: 'cnt', text: n, style: `fill:${tcol}` }, g);
        for (let k = 0; k < n; k++) S('circle', { cx: 34 + (k % per) * pitch, cy: y + Math.floor(k / per) * 11.5, r: 3.7, style: `fill:${col}` }, g);
        y += Math.max(1, Math.ceil(n / per)) * 11.5 + 8;
      }
      g.firstChild.setAttribute('height', y - y0 + 4);
      bindTip(g, rowTip(i), `${c}: only JAM ${P.jo[i].p}, only ${bs} ${P.bo[i].p}`);
      y += 12;
    });
    sv.setAttribute('viewBox', `0 0 ${W} ${y}`);
  } else {
    const cx = 520, pitch = 8.6, rows = 3, rH = 52, y0 = 54;
    sv.setAttribute('viewBox', `0 0 1000 ${y0 + 7 * rH - 4}`);
    S('text', { x: cx - 16, y: 24, 'text-anchor': 'end', cls: 'hdr', text: `← ONLY ${bs.toUpperCase()}` }, sv);
    S('text', { x: cx + 16, y: 24, cls: 'hdr', text: 'ONLY JAM →' }, sv);
    S('text', { x: 996, y: 24, 'text-anchor': 'end', cls: 'hdr', text: 'SUCCESS %' }, sv);
    S('line', { x1: cx, y1: 34, x2: cx, y2: y0 + 7 * rH - 14, cls: 'base-l' }, sv);
    RB.cats.forEach((c, i) => {
      const y = y0 + i * rH, g = S('g', { cls: 'row' }, sv);
      S('rect', { x: 0, y: y - 12, width: 1000, height: rH - 6, cls: 'hit' }, g);
      S('text', { x: 0, y: y + 13, cls: 'cat', text: c }, g);
      const dots = (n, dir, col) => { for (let k = 0; k < n; k++) S('circle', { cx: cx + dir * (12 + Math.floor(k / rows) * pitch), cy: y + 1 + (k % rows) * pitch, r: 3.2, style: `fill:${col}` }, g); };
      dots(P.bo[i].v, -1, 'var(--c-gen)'); dots(P.jo[i].v, 1, 'var(--cyan)');
      const lw = 12 + Math.ceil(P.bo[i].v / rows) * pitch, rw = 12 + Math.ceil(P.jo[i].v / rows) * pitch;
      S('text', { x: cx - lw - 6, y: y + 14, 'text-anchor': 'end', cls: 'cnt', text: P.bo[i].p, style: 'fill:var(--violet)' }, g);
      S('text', { x: cx + rw + 6, y: y + 14, cls: 'cnt', text: P.jo[i].p, style: 'fill:var(--cyan)' }, g);
      S('text', { x: 996, y: y + 14, 'text-anchor': 'end', cls: 'rate' + (P.rj[i].v < P.rb[i].v ? ' down' : ''), text: `${P.rb[i].p} → ${P.rj[i].p}` }, g);
      bindTip(g, rowTip(i), `${c}: only JAM ${P.jo[i].p}, only ${bs} ${P.bo[i].p}`);
    });
  }
  // stat tiles
  const st = $('#rb-stats'); st.textContent = '';
  const add = (k, v, s, tipFn, c) => { const d = H('div', null, st); H('dt', { text: k }, d); const dd = H('dd', { text: v, cls: c || null }, d); if (s) H('small', { text: s }, dd); if (tipFn) bindTip(d, tipFn); };
  // Table 20's Δ comes from unrounded rates (79.95 − 65.27 = 14.68 vs printed +14.67); the tooltip's source line says so
  add('Average gain', `${P.delta.p} pts`, `${P.from.p} → ${P.to.p}`,
    () => `<b>${P.delta.p} pts</b> task-weighted LIBERO-Plus success.<br>${esc(P.base)} ${pc(P.from)} → ${esc(P.jam)} ${pc(P.to)} (${esc(where(P.from))}).${tipSrc('Δ from unrounded rates · ' + where(P.delta))}`, 'c');
  add(`${RB.ciLevel.p} CI`, `[${P.ciLo.p}, ${P.ciHi.p}]`, null,
    () => `<b>[${P.ciLo.p}, ${P.ciHi.p}]</b> ${RB.ciLevel.p} percentile interval from ${RB.resamples.p} task-bootstrap resamples.${tipSrc(where(P.ciLo) + ' · ' + where(RB.resamples))}`);
  add('McNemar p', P.p.p, null, () => `<b>${esc(P.p.p)}</b> exact two-sided McNemar test on matched tasks.${tipSrc(where(P.p))}`);
  const sd = D.abl.seeds.rows[rbPair === 'ovis' ? 1 : 0];
  add('3 runs', `${sd[2].p} ± ${sd[3].p}`, null,
    () => `<b>${sd[2].p} ± ${sd[3].p}</b> mean ± sample SD of ${esc(sd[0])} over three training runs: ${sd[1].map(x => x.p).join(', ')}. “${esc(D.abl.seeds.q2.t)}” “${esc(D.abl.seeds.q.t)}”${tipSrc(where(sd[2]) + ' · ' + D.abl.seeds.q.w)}`);
  finish(sv);
  // table
  const tb = $('#rb-table'); tb.textContent = '';
  const t = H('table', { cls: 'data' }, tb), hr = H('tr', null, H('thead', null, t));
  ['Method', ...RB.cols].forEach(h => H('th', { text: head(h), scope: 'col' }, hr));
  const body = H('tbody', null, t);
  for (const [name, vals, jam] of [[P.base, P.rb, 0], [P.jam, P.rj, 1]]) {
    const tr = H('tr', jam ? { cls: 'jam' } : null, body);
    H('td', { text: name + '†' }, tr);
    vals.forEach((v, i) => H('td', { text: v.p, cls: (jam ? v.v >= P.rb[i].v : v.v >= P.rj[i].v) ? 'b' : null }, tr));
  }
  $('#rb-cap').textContent = DAGGER; // the caption only defines †
}
function initRobust() {
  rbNarrow = watchNarrow($('#rb-scroll'), 640, renderRobust);
  seg($('#rb-pair'), [{ v: 'ovis', label: 'Ovis-U1', title: 'ImageWAM (Ovis-U1) vs JAM (Ovis-U1)' }, { v: 'wan', label: 'Wan', title: 'FastWAM (Wan) vs JAM (Wan)' }], v => { rbPair = v; renderRobust(); }, rbPair);
  renderRobust();
}

// =============================================================================================
// 04 · rollout theater: one clock (the longer clip's playhead) drives both screens
// =============================================================================================
/* Time base (shared with the trailer): hz = clip frames per second of the clock. The LIBERO clips store one frame
   per control step, written at 24 fps for 20 Hz control, so 1× is real simulated time (playbackRate = speed × 20/24)
   and the clocks show step / 20 seconds. RoboTwin control steps have no fixed duration (each move is TOPP-timed), so
   both RoboTwin pairs run on RoboTwin's own video clock: 1× = 10 control steps per second, as its evaluator records. */
// RoboTwin 2.0 pairs, head camera, one frame per control step at 10 fps. Each pair starts from the same scene (matching
// first frames); in Blocks both policies also receive the same instruction.
const ROBOTWIN_PAIRS = [
  { slug: 'can', bench: 'RoboTwin 2.0', setting: 'Clean', task: 'Put the can in the basket', short: 'Can', hz: 10, ar: '714 / 464', arn: 714 / 464,
    rec: '1× = 10 control steps per second, as RoboTwin records: one frame per control step',
    note: 'FastWAM knocks over the can. Wan (distilled) places it in the basket.',
    base: { method: 'FastWAM (released)', outcome: 'fail', v: 'Knocks over the can', dur: 70, fps: 10 },
    ours: { method: 'JAM (Wan)', outcome: 'success', v: 'Can in the basket', dur: 22.3, fps: 10 } },
  { slug: 'blocks', bench: 'RoboTwin 2.0', setting: 'Randomized', task: 'Stack the red, green, and blue blocks', short: 'Blocks', hz: 10, ar: '714 / 464', arn: 714 / 464,
    rec: '1× = 10 control steps per second, as RoboTwin records: one frame per control step',
    note: 'FastWAM never completes the stack in 1,200 steps. JAM (Wan) stacks all three blocks.',
    base: { method: 'FastWAM (released)', outcome: 'fail', v: 'Never completes the stack', dur: 120, fps: 10 },
    ours: { method: 'JAM (Wan)', outcome: 'success', v: 'Stack complete', dur: 43.4, fps: 10 } },
];
const PAIRS = [...D.rollouts.pairs, ...ROBOTWIN_PAIRS];
const CTRL_HZ = D.rollouts.ctrlHz;
const hzOf = P => P.hz || CTRL_HZ, rateOf = P => hzOf(P) / P.base.fps;
const simT = (t, d, P) => Math.max(0, Math.min(Math.floor(t * P.base.fps + 1e-6), Math.round(d * P.base.fps) - 1)) / hzOf(P);
const OK_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
const FAIL_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
const TH = { i: 0, speed: 2, playing: false, t: 0, raf: 0, loaded: -1, load: null, pendingSeek: null, userPaused: false, autoPaused: false, visible: false };
const vb = $('#v-base'), vo = $('#v-ours');
const thDurs = () => [vb.duration || PAIRS[TH.i].base.dur, vo.duration || PAIRS[TH.i].ours.dur];
const thLen = () => Math.max(...thDurs());
const thMaster = () => { const [db, dO] = thDurs(); return dO > db ? [vo, vb] : [vb, vo]; }; // [clock, follower]
const sideV = (P, side) => P[side].v || (side === 'base' ? P.fail : P.ok);
function thUI() {
  const P = PAIRS[TH.i], [db, dO] = thDurs(), dm = Math.max(db, dO), t = TH.t;
  $('#clk-base').textContent = clockStr(simT(Math.min(t, db), db, P));
  $('#clk-ours').textContent = clockStr(simT(Math.min(t, dO), dO, P));
  const sc = $('#th-scrub');
  sc.value = Math.round(1000 * t / dm);
  sc.setAttribute('aria-valuetext', `${simT(t, dm, P).toFixed(1)} of ${simT(dm, dm, P).toFixed(1)} ${P.hz ? 'seconds' : 'simulated seconds'}`);
  if (!(TH.load && TH.loaded !== TH.i)) $('#th-time').textContent = clockStr(simT(t, dm, P));
  for (const [side, d] of [['ours', dO], ['base', db]]) {
    const on = TH.loaded === TH.i && t >= d - 0.06, el = $('#vd-' + side);
    if (on !== el.classList.contains('on')) {
      el.classList.toggle('on', on);
      if (on) $('#th-live').textContent = `${P[side].method} ${P[side].outcome === 'success' ? 'succeeded' : 'failed'}: ${sideV(P, side)}.`;
    }
  }
}
function thIcon() { const b = $('#th-play'); b.innerHTML = TH.playing ? PAUSE_ICON : PLAY_ICON; b.setAttribute('aria-label', TH.playing ? 'Pause both rollouts' : 'Play both rollouts'); }
function thSeek(t) {
  const [db, dO] = thDurs(), half = .5 / PAIRS[TH.i].base.fps;
  TH.t = clamp(t, 0, Math.max(db, dO));
  if (TH.loaded === TH.i) {
    try { vb.currentTime = Math.min(TH.t, db - half); } catch (e) {}
    try { vo.currentTime = Math.min(TH.t, dO - half); } catch (e) {}
  }
  thUI();
}
function thTick() {
  if (!TH.playing) return;
  const [vm, vs] = thMaster(), dm = vm.duration || 0, ds = vs.duration || 0;
  TH.t = vm.currentTime;
  if (ds && TH.t < ds - 0.04) {
    if (vs.paused && !vm.paused) vs.play().catch(() => {});
    if (Math.abs(vs.currentTime - TH.t) > 0.1) { try { vs.currentTime = TH.t; } catch (e) {} }
  } else if (ds && !vs.paused) vs.pause();
  thUI();
  if (vm.ended || (dm && TH.t >= dm - 0.03)) { TH.t = dm; thPause(); thUI(); return; }
  TH.raf = requestAnimationFrame(thTick);
}
function thPlay() {
  if (!mediaCanRun(vb)) return;
  if (TH.loaded !== TH.i) { thLoad(TH.i).then(() => { if (!TH.userPaused) thPlay(); }).catch(() => {}); return; }
  const [vm, vs] = thMaster(), dm = vm.duration || 0;
  if (dm && TH.t >= dm - 0.05) thSeek(0);
  TH.playing = true; TH.autoPaused = false;
  vb.playbackRate = vo.playbackRate = TH.speed * rateOf(PAIRS[TH.i]);
  vm.play().catch(() => { TH.playing = false; thIcon(); });
  if (vs.duration && TH.t < vs.duration - 0.04) vs.play().catch(() => {});
  thIcon();
  cancelAnimationFrame(TH.raf); TH.raf = requestAnimationFrame(thTick);
}
function thPause() { TH.playing = false; vb.pause(); vo.pause(); cancelAnimationFrame(TH.raf); thIcon(); }
function thSelect(i) {
  TH.i = i;
  const P = PAIRS[i];
  $('#pairs').querySelectorAll('button').forEach((b, k) => b.setAttribute('aria-pressed', String(k === i)));
  $('#task-line').textContent = `“${P.task}”`;
  $('#task-line').title = `${P.bench} · ${P.setting}`;
  for (const side of ['base', 'ours']) {
    const ok = P[side].outcome === 'success', el = $('#vd-' + side);
    el.classList.toggle('ok', ok); el.classList.toggle('fail', !ok);
    el.querySelector('.ring').innerHTML = ok ? OK_ICON : FAIL_ICON;
    el.querySelector('span').textContent = sideV(P, side);
    $('#lbl-' + side).innerHTML = `${esc(P[side].method.replace(/ \(.*\)$/, ''))} <small>${esc(P[side].sub || (side === 'base' ? 'generation loss' : 'foresight loss'))}</small>`;
  }
  vb.poster = `assets/rollouts/${P.slug}-base.jpg`; vo.poster = `assets/rollouts/${P.slug}-ours.jpg`;
  // LIBERO clips are 510 x 464 after the wrist-fringe crop ('seam'); the RoboTwin head camera is wider
  const scr = $('#screens');
  scr.style.setProperty('--clip-ar', P.ar || '510 / 464'); scr.style.setProperty('--clip-arn', String(P.arn || 510 / 464));
  scr.querySelectorAll('.vid-frame').forEach(f => f.classList.toggle('seam', !P.ar));
  $('#th-speed').title = P.rec || `1× = real time: one frame per ${CTRL_HZ} Hz control step`;
}
/* One load per pair: repeated Play / Restart / scrub requests await the same promise; selecting another pair
   aborts the superseded downloads. Progress shows in the time readout and on the busy play button. */
function thLoad(i) {
  if (TH.load && TH.load.i === i) return TH.load.p;
  if (TH.load) TH.load.ctrl.abort();
  thPause(); thSelect(i);
  // drop the previous pair's video so the new posters (frame 0) show while the new clips load
  for (const v of [vb, vo]) { if (v._objURL) { URL.revokeObjectURL(v._objURL); v._objURL = null; } v.removeAttribute('src'); v.load(); }
  TH.loaded = -1; TH.t = 0;
  const P = PAIRS[i], mm = (M.rollouts || {})[P.slug] || {}, btn = $('#th-play');
  const ctrl = typeof AbortController === 'function' ? new AbortController() : { abort() {}, signal: undefined };
  const fr = [0, 0], sz = [(mm.base || {})[FMT] || 1, (mm.ours || {})[FMT] || 1];
  const L = { i, ctrl };
  const prog = () => {
    if (TH.load !== L || TH.loaded === i) return;
    const f = (fr[0] * sz[0] + fr[1] * sz[1]) / (sz[0] + sz[1]);
    $('#th-time').textContent = `Loading ${Math.round(100 * f)}%`;
    btn.style.setProperty('--load', (100 * f).toFixed(1) + '%');
  };
  TH.load = L;
  btn.setAttribute('aria-busy', 'true');
  thUI(); prog();
  L.p = Promise.all([
    attachVideo(vb, `assets/rollouts/${P.slug}-base`, mm.base, x => { fr[0] = x; prog(); }, false, ctrl.signal),
    attachVideo(vo, `assets/rollouts/${P.slug}-ours`, mm.ours, x => { fr[1] = x; prog(); }, false, ctrl.signal),
  ]).then(() => {
    if (TH.load !== L) throw new Error('superseded');
    btn.removeAttribute('aria-busy'); btn.style.removeProperty('--load');
    TH.loaded = i;
    const f = TH.pendingSeek; TH.pendingSeek = null;
    thSeek(f != null ? thLen() * f : 0);
  }, e => {
    if (TH.load === L) {
      TH.load = null; TH.pendingSeek = null;
      btn.removeAttribute('aria-busy'); btn.style.removeProperty('--load');
      $('#th-live').textContent = 'The rollout clips could not be loaded.';
      thUI(); $('#th-time').textContent = 'Load failed';
    }
    throw e;
  });
  return L.p;
}
function initTheater() {
  const pb = $('#pairs');
  PAIRS.forEach((P, i) => {
    const b = H('button', { type: 'button', 'aria-pressed': 'false', title: `${P.task} · ${P.bench} · ${P.setting}` }, pb);
    const kind = /^Task/.test(P.setting) ? P.bench : `${P.bench} · ${P.setting.split(' · ')[0].split(' ')[0].toLowerCase()}`;
    H('small', { text: kind }, b); b.append(P.short);
    b.addEventListener('click', () => {
      TH.userPaused = false;
      if (i === TH.i && TH.loaded === i) { thSeek(0); if (autoMotion) thPlay(); return; } // already loaded: restart
      thLoad(i).then(() => { if (autoMotion && !TH.userPaused) thPlay(); }).catch(() => {});
    });
  });
  $('#th-play').addEventListener('click', () => { if (TH.playing) { TH.userPaused = true; thPause(); } else { TH.userPaused = false; thPlay(); } });
  $('#th-restart').addEventListener('click', () => {
    TH.userPaused = false;
    if (TH.loaded === TH.i) { thSeek(0); thPlay(); } else thLoad(TH.i).then(() => thPlay()).catch(() => {});
  });
  const scrub = $('#th-scrub');
  const seekFrac = f => {
    if (TH.playing) { TH.userPaused = true; thPause(); }
    if (TH.loaded !== TH.i) { TH.pendingSeek = f; thLoad(TH.i).catch(() => {}); scrub.value = Math.round(1000 * f); return; }
    thSeek(thLen() * f);
  };
  scrub.addEventListener('input', () => seekFrac(scrub.value / 1000));
  // arrow keys: one second of the pair's time base per press (20 control steps on LIBERO)
  scrubKeys(scrub, 1, d => seekFrac(clamp((TH.t + d * rateOf(PAIRS[TH.i])) / thLen(), 0, 1)));
  seg($('#th-speed'), [{ v: '1', label: '1×' }, { v: '2', label: '2×' }, { v: '4', label: '4×' }], v => {
    TH.speed = +v; press($('#th-speed'), v); vb.playbackRate = vo.playbackRate = TH.speed * rateOf(PAIRS[TH.i]);
  }, '2');
  $('#th-speed').querySelectorAll('button').forEach(b => b.setAttribute('aria-label', `Playback speed ${b.textContent}`));
  thSelect(0); thIcon(); thUI();
  const io = new IntersectionObserver(es => es.forEach(e => {
    TH.visible = e.isIntersecting && e.intersectionRatio >= .35;
    if (TH.visible) {
      if (!saveData && TH.loaded !== TH.i && !TH.load) thLoad(TH.i).then(() => { if (TH.visible && autoMotion && !TH.userPaused) thPlay(); }).catch(() => {});
      else if (TH.autoPaused && !TH.userPaused && autoMotion) thPlay();
    } else if (TH.playing) { TH.autoPaused = true; thPause(); }
  }), { threshold: .35 });
  io.observe($('#screens'));
}

// =============================================================================================
// 05 · benchmarks: lollipop board (Tables 4, 5, 6) + full printed tables (Tables 18, 19, 6)
// =============================================================================================
let bdTab = 'plus';
let bdNarrow = () => false;
const SCALE = { plus: [0, 100, [0, 25, 50, 75, 100]], libero: [70, 100, [70, 80, 90, 100]], robotwin: [50, 100, [50, 60, 70, 80, 90, 100]] };
function piSVG(t, n) { // π0, π0.5, π0-Fast with a subscript
  const m = n.match(/^π(0\.5|0)(.*)$/);
  if (!m) { t.textContent = n; return; }
  S('tspan', { text: 'π' }, t); S('tspan', { text: m[1], 'font-size': '72%', dy: 4 }, t); S('tspan', { text: m[2], dy: -4 }, t);
}
const piHTML = n => { const m = n.match(/^π(0\.5|0)(.*)$/); return m ? `π<sub>${m[1]}</sub>${esc(m[2])}` : esc(n); };
function splitName(n) {
  let m = n.match(/^(.*?) \(([^,()]+), ([^()]+)\)(†?)$/);
  if (m) return [`${m[1]} (${m[2]})${m[4]}`, m[3]];
  m = n.match(/^(.*?) \((released weights|local training)\)(†?)$/);
  if (m) return [m[1] + m[3], m[2]];
  return [n, ''];
}
const DAGGER = '† Evaluated by the authors.';
// table heads spelled out (the paper prints P.T., Rand. and AVG); the CSV export takes the same heads
const HEAD = { 'P.T.': 'Embodied pretraining', 'Rand.': 'Randomized', AVG: 'Average' }, head = c => HEAD[c] || c;
function dataTable(host, T, cap) {
  host.textContent = '';
  const t = H('table', { cls: 'data' }, host), hr = H('tr', null, H('thead', null, t));
  T.cols.forEach(c => H('th', { text: head(c), scope: 'col', cls: c === 'P.T.' ? 'pt' : null, title: c === 'P.T.' ? D.bench.qpt.t.replace(/^The P\.T\. column marks/, '✓ marks') : null }, hr));
  const body = H('tbody', null, t);
  const numCols = T.cols.map((c, i) => i).filter(i => i >= 1 && T.cols[i] !== 'P.T.');
  for (const g of T.groups) {
    const gr = H('tr', { cls: 'grp' }, body); H('span', { text: g.g }, H('td', { colspan: T.cols.length }, gr));
    const max = {};
    for (const ci of numCols) max[ci] = Math.max(...g.rows.map(r => r.cells[ci - 1].v));
    for (const r of g.rows) {
      const tr = H('tr', r.jam ? { cls: 'jam' } : null, body);
      const [nm, q] = splitName(r.name);
      const td = H('td', { 'data-name': r.name }, tr); // the CSV takes the row name, the screen a second line
      td.innerHTML = piHTML(nm.replace('†', '')) + (nm.includes('†') ? '<sup>†</sup>' : '') + (q ? `<span class="q">${esc(q)}</span>` : '');
      r.cells.forEach((c, k) => {
        const ci = k + 1;
        if (T.cols[ci] === 'P.T.') H('td', { cls: 'pt ' + (c.v ? 'y' : 'n'), text: c.p }, tr);
        else H('td', { text: c.p, cls: c.v === max[ci] ? 'b' : null }, tr);
      });
    }
  }
  cap.textContent = T.groups.some(g => g.rows.some(r => r.name.includes('†'))) ? DAGGER : ''; // the caption only defines †
}
function renderBoards() {
  const T = D.bench.tabs[bdTab], B = T.board, [lo, hi, ticks] = SCALE[bdTab];
  press($('#bd-tab'), bdTab);
  const best = D.bench.best[bdTab];
  const lab = { plus: ['LIBERO-Plus · JAM (Qwen3.5-2B) · highest listed', D.bench.qplus], libero: [`LIBERO · JAM (Wan) · ${D.bench.gap.p} below LingBot-VA`, D.bench.qlib], robotwin: ['RoboTwin 2.0 · JAM (FLUX.2) · highest listed', D.bench.qrt] }[bdTab];
  big('#bench-big', esc(best.p) + '<small>%</small>', lab[0], () => `<b>${pc(best)}</b> ${esc(T.label)} (${esc(where(best))}).<br>“${esc(lab[1].t)}”${tipSrc(lab[1].w)}`);
  const sv = $('#bd-svg');
  sv.querySelectorAll(':scope > *').forEach(n => n.remove());
  const nar = bdNarrow();
  const VW = nar ? 420 : 1000, rh = nar ? 30 : 27, x0 = nar ? 168 : 250, x1 = nar ? 372 : 920, nx = nar ? 0 : 14, fs = nar ? 14.5 : 13.5;
  const X = v => x0 + (x1 - x0) * (v - lo) / (hi - lo);
  const nRows = B.groups.reduce((a, g) => a + 1 + g.rows.length, 0);
  const Hh = 24 + nRows * rh + 30;
  sv.setAttribute('viewBox', `0 0 ${VW} ${Hh}`);
  $('#bd-svg').setAttribute('aria-label', `${T.label}: average success by method (${T.board.t})`);
  // every tick keeps its grid line; labels thin out so they never touch on a phone
  const every = Math.max(1, Math.ceil(46 / ((x1 - x0) / Math.max(1, ticks.length - 1))));
  ticks.forEach((v, i) => {
    S('line', { x1: X(v), y1: 14, x2: X(v), y2: Hh - 30, cls: 'grid-l' }, sv);
    if (i % every === 0) S('text', { x: X(v), y: Hh - 10, 'text-anchor': 'middle', cls: 'ax', text: v + '%' }, sv);
  });
  const ours = T.proto.t.replace(/^†\s*Ours:\s*/, '').replace(/\s*Others:.*$/, '').replace(/\.$/, ''); // our protocol, for the tooltips
  let y = 26;
  for (const g of B.groups) {
    const genG = /generator/i.test(g.g); // generation baselines are violet
    S('text', { x: 0, y: y + 4, cls: 'grp', text: g.g.toUpperCase(), 'font-size': nar ? 11.5 : 11 }, sv);
    y += rh;
    for (const r of g.rows) {
      const avg = r.cells[r.cells.length - 1], pt = r.cells.find((c, k) => B.cols[k + 1] === 'P.T.');
      const gg = S('g', null, sv);
      S('rect', { x: 0, y: y - rh / 2, width: VW, height: rh, cls: 'hit' }, gg);
      const nt = S('text', { x: nx, y: y + 5, cls: r.jam ? 'cat jam' : 'cat', 'font-size': fs }, gg); // row name = category label
      piSVG(nt, r.name.replace(/†/g, ''));
      S('line', { x1: x0, y1: y, x2: X(avg.v), y2: y, style: `stroke:${r.jam ? 'var(--cyan)' : 'var(--line-2)'}`, 'stroke-width': r.jam ? 2.5 : 1.5, opacity: r.jam ? .85 : 1 }, gg);
      const mk = genG ? 'var(--violet)' : 'var(--muted)';
      S('circle', { cx: X(avg.v), cy: y, r: r.jam ? 6 : 5, style: r.jam ? 'fill:var(--panel);stroke:var(--cyan);stroke-width:3;filter:drop-shadow(0 0 5px var(--cyan))' : pt && pt.v ? `fill:${mk}` : `fill:var(--panel);stroke:${mk};stroke-width:2` }, gg);
      S('text', { x: X(avg.v) + 11, y: y + 4.5, cls: 'v', style: `fill:${r.jam ? 'var(--ink)' : 'var(--ink-2)'};font-weight:${r.jam ? 650 : 400}`, text: avg.p }, gg);
      bindTip(gg, () => `<b>${pc(avg)}</b> · ${esc(splitName(r.name).join(', ').replace(/, $/, '').replace(/†/g, ''))}<br>${pt && pt.v ? 'Uses embodied pretraining on robot data' : 'No embodied pretraining on robot data'}${r.dagger ? ` · evaluated by the authors: ${esc(ours)}` : ' · published value, original protocol'}${tipSrc(where(avg))}`, `${r.name.replace(/†/g, '')}: ${avg.p}%`, r.jam ? 'var(--cyan)' : mk);
      y += rh;
    }
  }
  dataTable($('#bd-table'), T.full, $('#bd-cap'));
  finish(sv);
}
function initBoards() {
  bdNarrow = watchNarrow($('#bd-scroll'), 620, renderBoards);
  seg($('#bd-tab'), Object.entries(D.bench.tabs).map(([v, t]) => ({ v, label: t.label })), v => { bdTab = v; renderBoards(); }, bdTab);
  const lg = $('#bd-legend');
  for (const [l, st] of [['JAM', 'background:var(--panel);border:2px solid var(--cyan);box-shadow:0 0 6px var(--cyan)'], ['Generative', 'background:transparent;border:2px solid var(--violet)'], ['No robot pretraining', 'background:transparent;border:2px solid var(--muted)'], ['Robot-data pretraining', 'background:var(--muted)']]) {
    const s = H('span', null, lg); H('i', { style: st }, s); s.append(l);
  }
  renderBoards();
}

// =============================================================================================
// 02 · foresight target designer (Table 1) + cosine toy
// =============================================================================================
const TGT = D.target;
// opens on JAM (Wan) with the pretrained Wan anchor: the two printed values behind the +10.96 readout
// each architecture opens on its best target (Wan: the frozen VAE tokenizer, [1,2]; Ovis-U1: pretrained Yak, [1])
const tgBest = rec => TGT.archs[rec].best.split('|');
const tg = { rec: 'wan', K: tgBest('wan')[0], anchor: tgBest('wan')[1] };
const TGI = { on: false }; // timeline thumbnails load once the module is near the viewport
// timeline frames: LIBERO-Long demo episode 124, steps TL_T0 ... TL_T0 + 32 (the first pot lifted and carried to the stove,
// so every 4-step thumbnail visibly differs)
const TL_T0 = 106;
let tgNarrow = () => false, stripNarrow = () => false;
const offs = key => key.replace(/[[\]]/g, '').split(',').map(Number);
function renderTarget() {
  const A = TGT.archs[tg.rec];
  press($('#tg-rec'), tg.rec);
  // combinations the paper did not run are disabled, so every choice leads to a printed result
  const ran = (K, an) => !!A.cells[`${K}|${an}`], NR = 'Not run in the paper';
  seg($('#tg-k'), A.offsets.map(o => ({ v: o, label: o.replace(/,/g, ', '), k: 'fore', dis: !ran(o, tg.anchor), title: ran(o, tg.anchor) ? null : NR,
    why: () => `${NR}: offsets ${o} with ${tg.anchor}.` })), v => { tg.K = v; renderTarget(); }, tg.K);
  seg($('#tg-anchor'), A.anchors.map(a => ({ v: a, label: a, k: 'fore', dis: !ran(tg.K, a), title: ran(tg.K, a) ? null : NR,
    why: () => `${NR}: ${a} with offsets ${tg.K}.` })), v => { tg.anchor = v; renderTarget(); }, tg.anchor);
  const key = `${tg.K}|${tg.anchor}`, cell = A.cells[key];
  // timeline
  const sv = $('#tg-svg'), nar = tgNarrow();
  sv.querySelectorAll(':scope > *').forEach(n => n.remove());
  const steps = A.frames[A.frames.length - 1];
  const VW = nar ? 420 : 1000, x0 = nar ? 34 : 70, x1 = nar ? 392 : 930, pad = nar ? 15 : 30;
  const fw = nar ? (tg.rec === 'wan' ? 26 : 44) : 48, fh = nar ? (tg.rec === 'wan' ? 28 : 40) : 52;
  const gy0 = nar ? 32 : 30, gy1 = nar ? 100 : 122, fy = nar ? (tg.rec === 'wan' ? 50 : 44) : 50, ay = nar ? 136 : 168;
  const X = s => x0 + (x1 - x0) * s / steps;
  sv.setAttribute('viewBox', `0 0 ${VW} ${nar ? 182 : 214}`);
  S('line', { x1: x0, y1: ay, x2: x1, y2: ay, cls: 'base-l' }, sv);
  const tickStep = tg.rec === 'wan' ? 4 : 2, labStep = nar ? tickStep * 2 : tickStep;
  for (let s = 0; s <= steps; s += tickStep) {
    S('line', { x1: X(s), y1: ay - 4, x2: X(s), y2: ay + 4, cls: 'base-l' }, sv);
    if (s % labStep === 0) S('text', { x: X(s), y: ay + (nar ? 19 : 24), 'text-anchor': 'middle', cls: 'ax', text: s === 0 ? '0 now' : '+' + s, 'font-size': nar ? 13.5 : null }, sv);
  }
  S('text', { x: x1, y: nar ? 176 : 210, 'text-anchor': 'end', cls: 'ax', text: 'environment steps', 'font-size': nar ? 13.5 : null }, sv);
  const sel = new Set(offs(tg.K));
  Object.entries(A.ranges).forEach(([k, [from, to]]) => {
    const kk = +k.replace(/\D/g, ''), on = sel.has(kk);
    const a = X(from) - pad, b = X(to) + pad;
    S('rect', { x: a, y: gy0, width: b - a, height: gy1 - gy0, rx: 10, style: `fill:${on ? 'var(--fore-wash)' : 'transparent'};stroke:${on ? 'var(--c-fore)' : 'var(--line-2)'};stroke-width:${on ? 2 : 1}`, 'stroke-dasharray': on ? null : '4 4' }, sv);
    S('text', { x: (a + b) / 2, y: gy1 + (nar ? 15 : 22), 'text-anchor': 'middle', cls: 'ax', text: k, style: `fill:${on ? 'var(--amber)' : 'var(--muted)'};font-weight:${on ? 700 : 400};font-size:${nar ? 12 : 13}px` }, sv);
    if (on) S('text', { x: (a + b) / 2, y: gy0 - 8, 'text-anchor': 'middle', cls: 'ax', text: 'target', style: `fill:var(--amber);font-size:${nar ? 11 : 11.5}px;letter-spacing:.1em` }, sv);
  });
  // real frames of the LIBERO-Long demonstration (episode frame 40 + step), as in 01 IDEA
  const cpth = S('clipPath', { id: 'tg-clip', clipPathUnits: 'objectBoundingBox' }, S('defs', null, sv));
  S('rect', { width: 1, height: 1, rx: .13, ry: .12 }, cpth);
  const inSel = s => Object.entries(A.ranges).some(([k, [from, to]]) => sel.has(+k.replace(/\D/g, '')) && s >= from && s <= to);
  for (const s of A.frames) {
    const cx = X(s), cur = s === 0, lit = cur || inSel(s);
    if (TGI.on) S('image', { href: `assets/img/tl-${String(TL_T0 + s).padStart(3, '0')}.jpg`, x: cx - fw / 2, y: fy, width: fw, height: fh, preserveAspectRatio: 'xMidYMid slice', 'clip-path': 'url(#tg-clip)', opacity: lit ? 1 : .4 }, sv);
    S('rect', { x: cx - fw / 2, y: fy, width: fw, height: fh, rx: 6, style: `fill:${TGI.on ? 'none' : 'var(--panel-2)'};stroke:${cur ? 'var(--c-act)' : lit ? 'var(--c-fore)' : 'var(--line-2)'};stroke-width:${lit ? 2 : 1}` }, sv);
  }
  S('text', { x: X(0), y: fy + fh + (nar ? 13 : 16), 'text-anchor': 'middle', cls: 'ax', text: 'input', style: `fill:var(--cyan);font-size:${nar ? 11 : 11.5}px` }, sv);
  // numbers
  const bigEl = $('#tg-big'); bigEl.textContent = '';
  const add = (k, x) => { const d = H('div', null, bigEl); H('dt', { text: k }, d); const dd = H('dd', null, d); if (!x) { dd.className = 'na'; dd.textContent = 'not reported'; } else { dd.textContent = pc(x); d.tabIndex = 0; bindTip(d, () => `<b>${pc(x)}</b> ${esc(k)} · ${esc(A.label)} target ${esc(tg.K)}, ${esc(tg.anchor)}${tipSrc(where(x))}`); } };
  add('LIBERO', cell && cell.lib); add('LIBERO-Plus', cell && cell.plus); // unreported cells read 'not reported'
  legible(sv);
  renderStrip(A, key);
}
// strip axis: 5-point ticks around each architecture's own values (no empty half), one span and one row layout for all
// architectures, so a switch keeps the scale and the height
const tgVals = a => [...Object.values(a.cells), ...Object.values(a.base).filter(Boolean)].map(c => c.plus.v);
const tgFloor = a => Math.floor((Math.min(...tgVals(a)) - 2) / 5) * 5;
const tgSpan = () => Math.max(...Object.values(TGT.archs).map(a => Math.ceil((Math.max(...tgVals(a)) + 2) / 5) * 5 - tgFloor(a)));
// evaluated targets in value order with their stack level: near-equal values (closer than 14 units) stack upward
function tgLifts(a, X) {
  let lastX = -1e9, lift = 0;
  return Object.entries(a.cells).sort((p, q) => p[1].plus.v - q[1].plus.v)
    .map(([kk, c]) => { const x = X(c.plus.v); lift = x - lastX < 14 ? lift + 1 : 0; lastX = x; return { kk, c, x, lift }; });
}
function renderStrip(A, key) {
  const st = $('#tg-strip'), nar = stripNarrow();
  st.querySelectorAll(':scope > *').forEach(n => n.remove());
  const VW = nar ? 420 : 640, xa = nar ? 18 : 20, xb = nar ? 402 : 620, fs = nar ? 13 : 11, span = tgSpan();
  const scale = lo => v => xa + (xb - xa) * (v - lo) / span, lo = tgFloor(A), X = scale(lo);
  const stack = Math.max(...Object.values(TGT.archs).map(a => Math.max(...tgLifts(a, scale(tgFloor(a))).map(d => d.lift))));
  // rows from the top: the unit (top-right corner), the gain bracket, the target dots, the baselines, the ticks
  const yC = (nar ? 42 : 38) + 18 * stack, yD = yC + (nar ? 30 : 26);
  // baselines (diamonds), left to right. Labels sit beside them, pointing outward, when both fit at the largest legibility
  // scaling of a desktop strip; otherwise under them, staggered when close
  const bl = [['none', 'var(--c-base)', 'action only'], ['gen', 'var(--c-gen)', 'generation']]
    .map(([k, c, l]) => ({ b: A.base[k], c, l, x: X(A.base[k].plus.v) })).sort((p, q) => p.x - q.x);
  const wMax = l => l.length * fs * .62 * Math.max(1, MIN_PX / (fs * 520 / VW));
  const beside = !nar && bl[0].x - 13 - wMax(bl[0].l) >= 0 && bl[1].x + 13 + wMax(bl[1].l) <= VW;
  const close = !beside && bl[1].x - bl[0].x < (nar ? 110 : 90);
  const yT = yD + (beside ? 24 : (nar ? 60 : 54) - (close ? 0 : fs + 3));
  st.setAttribute('viewBox', `0 0 ${VW} ${yT + (nar ? 5 : 4)}`);
  S('text', { x: xb, y: nar ? 13 : 11, 'text-anchor': 'end', cls: 'ax unit', text: 'LIBERO-Plus %', 'font-size': fs }, st);
  for (let v = lo; v <= lo + span; v += nar ? 10 : 5) {
    S('line', { x1: X(v), y1: nar ? 20 : 18, x2: X(v), y2: yT - (nar ? 16 : 13), cls: 'grid-l' }, st);
    S('text', { x: X(v), y: yT, 'text-anchor': 'middle', cls: 'ax', text: v, 'font-size': fs }, st);
  }
  bl.forEach(({ b, c, l, x }, i) => {
    const g = S('g', { cls: 'bl' }, st);
    S('path', { d: `M${x},${yD - 8} l7,8 l-7,8 l-7,-8 z`, style: `fill:${c}` }, g);
    if (beside) S('text', { x: i ? x + 13 : x - 13, y: yD + fs * .36, 'text-anchor': i ? 'start' : 'end', cls: 'ax', text: l, 'font-size': fs }, g);
    else {
      const half = l.length * fs * .31, lx = clamp(x, xa + half - 6, xb - half);
      S('text', { x: lx, y: yD + 21 + (close && i ? fs + 3 : 0), 'text-anchor': 'middle', cls: 'ax', text: l, 'font-size': fs }, g);
    }
    bindTip(g, () => `<b>${pc(b.plus)}</b> LIBERO-Plus · ${esc(b.label)}<br>LIBERO ${pc(b.lib)}${tipSrc(where(b.plus))}`, `${b.label}: ${b.plus.p}%`, c);
  });
  // evaluated targets (circles)
  let sel = null;
  const dots = tgLifts(A, X).map(({ kk, c, x, lift }) => ({ kk, c, x, y: yC - lift * 18, r: kk === key ? 8.5 : 6, on: kk === key }));
  for (const d of dots) {
    const { kk, c, x, y, r, on } = d;
    // the hit circle stops short of every other dot, so stacked near-equal values never cover one another
    const hr = Math.max(r + 1, Math.min(nar ? 18 : 12, ...dots.filter(o => o !== d).map(o => Math.hypot(o.x - x, o.y - y) - o.r - 1)));
    if (on) sel = { x, y, r, p: c.plus.p, v: c.plus.v, id: c.plus.id, label: kk };
    const g = S('g', { cls: 'pick', role: 'button', 'aria-pressed': String(on) }, st);
    S('circle', { cx: x, cy: y, r: hr, cls: 'hit' }, g);
    S('circle', { cx: x, cy: y, r, style: `fill:${on ? 'var(--cyan)' : 'var(--panel)'};stroke:var(--c-act);stroke-width:2` }, g);
    const [K, An] = kk.split('|');
    bindTip(g, () => `<b>${pc(c.plus)}</b> LIBERO-Plus · LIBERO ${pc(c.lib)}<br>Offsets ${esc(K)} · ${esc(An)}${tipSrc(where(c.plus))}`, `Select offsets ${K} with ${An}: ${c.plus.p}% LIBERO-Plus`, on ? 'var(--cyan)' : 'var(--c-act)');
    const pick = () => {
      const had = !!(document.activeElement && document.activeElement.closest && document.activeElement.closest('#tg-strip'));
      tg.K = K; tg.anchor = An; renderTarget();
      const n = $('#tg-strip .pick[aria-pressed="true"]'); if (had && n) n.focus();
    };
    g.addEventListener('click', pick);
    g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    g.style.cursor = 'pointer';
  }
  // the selected target against the generation baseline (FastWAM or ImageWAM); the bracket follows every selection,
  // runs above all dots it spans, and drops a dashed leg onto the generation diamond. Below the baseline: magenta
  const gen = A.base.gen;
  if (sel && gen) {
    const xg = X(gen.plus.v), x1 = Math.min(xg, sel.x), x2 = Math.max(xg, sel.x);
    const top = Math.min(sel.y - sel.r, ...dots.filter(o => o.x >= x1 - 9 && o.x <= x2 + 9).map(o => o.y - o.r));
    // a pair the paper reports (Table 20) keeps its delta from unrounded rates: 79.95 − 65.27 is printed as +14.67
    const pr = Object.values(D.robust.pairs).find(q => q.to.id === sel.id && q.from.id === gen.plus.id);
    const yb = top - (nar ? 14 : 12), dv = pr ? pr.delta.v : Math.round((sel.v - gen.plus.v) * 100) / 100, up = dv >= 0;
    const col = up ? 'var(--amber)' : 'var(--magenta)', txt = (up ? '+' : MINUS) + Math.abs(dv).toFixed(2);
    // a near-equal dot stacked right above the selection (Ovis-U1: 72.58 over 72.51) would sit on a straight riser, so the
    // riser passes beside it on the generation side, starting off the selected dot's shoulder
    const dir = xg < sel.x ? -1 : 1, over = dots.filter(o => !o.on && o.y < sel.y && Math.abs(o.x - sel.x) < o.r + 5);
    const rx = over.length ? sel.x + dir * Math.max(...over.map(o => dir * (o.x - sel.x) + o.r + 5)) : sel.x;
    const ry = sel.y - Math.sqrt(Math.max(0, (sel.r + 3) ** 2 - (rx - sel.x) ** 2));
    const g = S('g', { cls: 'gain-br' }, st);
    S('path', { d: `M${rx},${ry} V${yb}`, cls: 'br', style: `stroke:${col}` }, g); // separate segments, so the
    S('path', { d: `M${rx},${yb} H${xg}`, cls: 'br', style: `stroke:${col}` }, g);  // label avoids lines, not their box
    S('path', { d: `M${xg},${yb} V${yD - 10}`, cls: 'br', style: `stroke:${col}`, 'stroke-dasharray': '3 3' }, g);
    // only the label takes the pointer and the tooltip: a hit box around the whole bracket would cover the dots it spans
    const lg = S('g', { cls: 'br-lbl' }, g);
    S('text', { x: (xg + sel.x) / 2, y: yb - 5, 'text-anchor': 'middle', cls: 'br-t', text: txt, 'font-size': nar ? 13 : 11.5, style: `fill:${col}` }, lg);
    const [K, An] = sel.label.split('|');
    bindTip(lg, () => `<b>${txt} pts</b> LIBERO-Plus vs generation: offsets ${esc(K)} with ${esc(An)}, ${sel.p}%, against ${esc(gen.label)}, ${gen.plus.p}%${tipSrc(pr ? 'Δ from unrounded rates · ' + where(pr.delta) : where(gen.plus))}`,
      `${txt} points versus ${gen.label}`, col);
  }
  finish(st);
  // the selected value goes on top of every mark: above, right, left or below its dot, whichever touches no other dot,
  // the gain bracket, a baseline or the unit (placed after finish(), so the measured width includes any legibility scaling)
  if (sel) {
    const t = S('text', { cls: 'v sel-v', text: sel.p, style: `fill:var(--ink);font-family:var(--font-display);font-weight:700;font-size:${nar ? 15 : 13}px` }, st);
    legible(st);
    const tw = t.getComputedTextLength(), th = parseFloat(getComputedStyle(t).fontSize) || 13;
    const obs = [...st.querySelectorAll('.gain-br path, .gain-br text, .bl path, .bl text, .unit')].map(e => e.getBBox()); // bracket, baselines, unit
    const cands = [[sel.x, sel.y - sel.r - 6, 'middle'], [sel.x + sel.r + 6, sel.y + th * .35, 'start'],
      [sel.x - sel.r - 6, sel.y + th * .35, 'end'], [sel.x, sel.y + sel.r + th + 2, 'middle']];
    const box = ([cx, cy, an]) => { const x0 = an === 'middle' ? cx - tw / 2 : an === 'end' ? cx - tw : cx; return [x0, cy - th * .78, x0 + tw, cy + th * .12]; };
    const hit = b => dots.some(o => !o.on && b[0] < o.x + o.r + 2 && b[2] > o.x - o.r - 2 && b[1] < o.y + o.r + 2 && b[3] > o.y - o.r - 2)
      || obs.some(q => b[0] < q.x + q.width + 2 && b[2] > q.x - 2 && b[1] < q.y + q.height + 2 && b[3] > q.y - 2) || b[0] < 0 || b[2] > VW || b[1] < 0;
    const c = cands.find(k => !hit(box(k))), bt = st.querySelector('.gain-br text');
    if (c) { t.setAttribute('x', c[0].toFixed(1)); t.setAttribute('y', c[1].toFixed(1)); t.setAttribute('text-anchor', c[2]); }
    else if (bt) { // boxed in (e.g. right next to the baseline): the bracket label carries the value, "55.58  +0.27"
      const d = bt.textContent; bt.textContent = '';
      S('tspan', { text: sel.p, style: 'fill:var(--ink)' }, bt); S('tspan', { text: '\u2002' + d }, bt);
      t.remove(); ensureHits(st); // the label's hit box grows with it
    } else { t.setAttribute('x', cands[0][0].toFixed(1)); t.setAttribute('y', cands[0][1].toFixed(1)); t.setAttribute('text-anchor', 'middle'); }
  }
}
function initTarget() {
  tgNarrow = watchNarrow($('#tg-fig'), 620, renderTarget);
  stripNarrow = watchNarrow($('#tg-strip-box'), 520, renderTarget);
  seg($('#tg-rec'), Object.entries(TGT.archs).map(([v, a]) => ({ v, label: a.label, sub: a.sub, title: a.sub + ' architecture' })), v => {
    tg.rec = v; [tg.K, tg.anchor] = tgBest(v); renderTarget();
  }, tg.rec);
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => { if (!TGI.on && es.some(e => e.isIntersecting)) { TGI.on = true; io.disconnect(); renderTarget(); } }, { rootMargin: '600px 0px' });
    io.observe($('#tg-fig'));
  } else TGI.on = true;
  renderTarget();
}


// =============================================================================================
// 06/07 · analysis videos (custom controls; poster + source attached near the viewport)
// =============================================================================================
/* o.autoplay: play muted when >= 35 % visible (probe). Without it (attention) the video rests on its poster, the
   paper's frame, and downloads only when the viewer plays or scrubs. Both pause offscreen and keep a user pause. */
function videoInstrument(v, btn, scrub, time, src, o = {}) {
  let loading = null, visible = false;
  const auto = o.autoplay !== false;
  const status = $('#' + v.id.replace('-v', '-status'));
  const rate = () => { if (o.rate) v.defaultPlaybackRate = v.playbackRate = o.rate; }; // load() resets playbackRate
  rate();
  const ensure = () => {
    if (loading) return loading;
    status.hidden = false; status.textContent = 'Loading video…';
    btn.setAttribute('aria-busy', 'true');
    let lastProgress = -1;
    loading = attachVideo(v, src.base, src.bytes, f => {
      const percent = Math.floor(f * 10) * 10;
      if (percent !== lastProgress) { status.textContent = `Loading video · ${percent}%`; lastProgress = percent; }
    }).then(() => {
      status.hidden = true; btn.removeAttribute('aria-busy'); rate(); return true;
    }, () => {
      loading = null; btn.removeAttribute('aria-busy');
      status.hidden = false; status.textContent = 'Video unavailable. Press play to retry.'; return false;
    });
    return loading;
  };
  const poster = () => { if (v.dataset.poster && !v.getAttribute('poster')) v.setAttribute('poster', v.dataset.poster); };
  const sync = () => {
    if (v.duration) { scrub.value = Math.round(1000 * v.currentTime / v.duration); time.textContent = clockStr(v.currentTime); scrub.setAttribute('aria-valuetext', `${v.currentTime.toFixed(1)} of ${v.duration.toFixed(1)} seconds`); }
    btn.innerHTML = v.paused ? PLAY_ICON : PAUSE_ICON;
    btn.setAttribute('aria-label', (v.paused ? 'Play ' : 'Pause ') + btn.dataset.what);
    if (o.onSync) o.onSync();
  };
  const seekTo = async t => {
    v.dataset.user = 'paused'; v.pause();
    if (!await ensure()) return;
    if (v.duration) { try { v.currentTime = clamp(t, 0, v.duration - 0.01); } catch (e) {} }
  };
  btn.dataset.what = btn.getAttribute('aria-label').replace(/^Play /, '');
  let started = false;
  const fromPoster = () => { // the first play, by click or autoplay, continues from the poster frame
    if (!started && o.startAt != null && v.duration && v.currentTime < .01) { try { v.currentTime = o.startAt; } catch (e) {} }
    started = true;
  };
  btn.addEventListener('click', async () => {
    if (v.paused) {
      v.dataset.user = ''; if (!await ensure()) return;
      fromPoster();
      if (mediaCanRun(v)) v.play().catch(() => {});
    } else { v.dataset.user = 'paused'; v.pause(); }
  });
  scrub.addEventListener('input', async () => {
    const f = scrub.value / 1000;
    v.dataset.user = 'paused'; v.pause();
    if (!await ensure()) return;
    if (v.duration) { try { v.currentTime = v.duration * f; } catch (e) {} }
  });
  scrubKeys(scrub, 1, d => seekTo((v.currentTime || 0) + d)); // one second per arrow-key press
  ['timeupdate', 'play', 'pause', 'seeked', 'loadedmetadata'].forEach(ev => v.addEventListener(ev, sync));
  sync();
  if (o.startAt != null && src.dur) { // before loading, the controls describe the poster frame
    scrub.value = Math.round(1000 * o.startAt / src.dur); time.textContent = clockStr(o.startAt);
    scrub.setAttribute('aria-valuetext', `${o.startAt.toFixed(1)} of ${src.dur.toFixed(1)} seconds`);
  }
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { poster(); if (auto && !saveData && !document.hidden) ensure(); } }), { rootMargin: '300px 0px' }).observe(v);
    new IntersectionObserver(es => es.forEach(e => {
      visible = e.isIntersecting && e.intersectionRatio >= .35;
      if (visible) { if (auto && autoMotion && mediaCanRun(v) && v.dataset.user !== 'paused') ensure().then(ok => { if (ok && visible && mediaCanRun(v) && v.dataset.user !== 'paused') { fromPoster(); v.play().catch(() => {}); } }); }
      else if (!v.paused && !v.closest('.media-viewer[open]')) { v.pause(); }
    }), { threshold: .35 }).observe(v);
  } else poster();
  return { seekTo };
}
function initProbe() {
  const P = D.probe;
  videoInstrument($('#probe-v'), $('#probe-play'), $('#probe-scrub'), $('#probe-time'), M.probe, { rate: 1.5 }); // plays at 1.5×
  big('#probe-big', '+' + esc(P.horizon.v), 'Steps ahead · analysis probe',
    () => `<b>+${P.horizon.v} steps</b> “${esc(P.qfit.t)}” “${esc(P.qonly.t)}”${tipSrc(P.qfit.w)}`, 'a');
  // how to read: the six panels left to right, the colour scale, and Figure 9's mean future-frame PSNR
  const ps = P.psnr.map(r => `${esc(r.k === 'cur' ? 'current frame' : r.label)} ${r.v.p}`).join(' · ');
  $('#probe-how').innerHTML = `<ul class="how-list">` +
    `<li><b>Panels</b><span>Left to right: the current frame, the recorded frame ${P.horizon.v} steps later, the recorded change, then that change decoded from frozen ImageWAM, JAM [0] and JAM [1] features (all Ovis-U1). The paper’s example is step ${M.probe.exampleStep}.</span></li>` +
    `<li><b>Colours</b><span>Red: brighter later. Blue: darker. These are analysis probes, not frames any policy generates.</span></li>` +
    `<li><b>PSNR</b><span>Mean future-frame PSNR, dB: ${ps}</span></li></ul>` +
    `<p class="q">${esc(P.qsug.t)}</p>`; // no figure or section numbers, no quotation marks
}
function initAttention() {
  const A = D.attn, v = $('#att-v');
  // autoplays the episode loop from its first frame (the poster); the paper's example frame is a step later in it
  videoInstrument(v, $('#att-play'), $('#att-scrub'), $('#att-time'), M.att);
  // how to read: the six columns, the two kinds of map, and Figure 5's 100-frame shares (text / image / action)
  const sh = A.rows.map(r => `${esc(r.label)} ${r.v.map(x => x.p).join('/')}`).join(' · ');
  // the key set each JAM policy gives its largest share, read from the same shares
  const top = A.rows.filter(r => /^JAM/.test(r.label)).map(r => `${A.keys[r.v.reduce((b, x, k) => (x.v > r.v[b].v ? k : b), 0)]} for ${esc(r.label)}`).join(', ');
  $('#att-how').innerHTML = `<ul class="how-list">` +
    `<li><b>Panels</b><span>Left to right: the camera image; ImageWAM and JAM, both on Ovis-U1; JAM minus ImageWAM; JAM on DINOv2-L and on Qwen3.5-2B. Agent view on top, wrist view below. The paper’s example is step ${M.att.exampleStep}.</span></li>` +
    `<li><b>Maps</b><span>Where the action expert attends more than its own average over ${esc(A.frames.p)}: red far above it, dark blue at or below it. In the difference column, red marks more attention from JAM, blue more from ImageWAM, grey little difference.</span></li>` +
    `<li><b>Shares</b><span>The bar beside each map splits this frame’s attention between text, image and action keys. Over all ${esc(A.frames.p.replace('reference frames', 'frames'))}, ${A.keys.join(' / ')} (%): ${sh}. The largest share is ${top}.</span></li></ul>` +
    `<p class="q">${esc(A.q.t)}</p>`;
}

// =============================================================================================

// =============================================================================================
// 08 · compute ledger (Table 9)
// =============================================================================================
const CMP = { rec: 'wan', mode: 'cached' };
let cmpNarrow = () => false;
// Visual is slate, not violet: the JAM bar's visual branch is not generation
const COMP = [['Visual', 'tflops_visual', 'var(--c-slate)'], ['Action', 'tflops_action', 'var(--c-act)'], ['Head', 'tflops_head', 'var(--c-fore)'], ['Target', 'tflops_target', 'var(--c-gold)'], ['VAE', 'tflops_vae', 'var(--c-vae)']];
const INK_ON = { 'var(--c-act)': 'var(--bg)', 'var(--c-fore)': 'var(--bg)', 'var(--c-gold)': 'var(--bg)', 'var(--c-gen)': 'var(--bg)', 'var(--c-vae)': 'var(--bg)' }; // in-bar text contrast
function compTip(k, row, isBase) {
  const C = D.compute, T = C.tok;
  if (k === 'Visual') return CMP.rec === 'wan' ? `Trainable visual branch. Wan processes ${T.gen.p} under generation and ${T.fore.v} under foresight, plus key and value projections of ${T.ctx.p} per block.` : 'Trainable visual branch.';
  if (k === 'Action') return 'Action expert, same under both objectives.';
  if (k === 'Head') return isBase ? `No prediction head (printed ${row.params_head.p}).` : `Training-only prediction head, ${row.params_head.p}M parameters, dropped after training.`;
  if (k === 'Target') return isBase ? `No foresight target (printed ${row.params_anchor.p}).` : CMP.rec === 'wan' ? `Online VAE-direct targets are counted in the VAE column; extra anchor parameters: ${row.params_anchor.p}M.` : CMP.mode === 'online' ? `Frozen Yak18 anchor (${row.params_anchor.p}M parameters) run to ${T.ovisDepth.p} for online targets.` : `Targets read from the cache; anchor parameters at training time: ${row.params_anchor.p}M.`;
  if (k === 'VAE') return CMP.rec === 'wan' ? (isBase ? `Generation encodes the ${C.window.p}.` : CMP.mode === 'online' ? C.qvae.t : 'Cached foresight encodes only the current frame.') : 'VAE encoding.';
  return '';
}
function renderCompute() {
  const C = D.compute[CMP.rec], base = C.base, jam = C[CMP.mode];
  press($('#cmp-rec'), CMP.rec); press($('#cmp-mode'), CMP.mode);
  const bn = CMP.rec === 'wan' ? 'FastWAM' : 'ImageWAM';
  big('#compute-big', reductionText(jam.tflops_reduction), `Est. online FLOPs vs ${bn} · ${CMP.mode} targets`,
    () => `<b>${MINUS}${jam.tflops_reduction.p}</b> estimated online TFLOPs per sample per update: ${esc(C.baseName)} ${base.tflops_total.p} → ${esc(C.jamName)} ${jam.tflops_total.p} (${CMP.mode} targets; ${esc(C.cfg)}).${tipSrc(where(jam.tflops_reduction) + ' · not wall-clock time')}`);
  const sv = $('#cmp-svg'), nar = cmpNarrow();
  sv.querySelectorAll(':scope > *').forEach(n => n.remove());
  const VW = nar ? 420 : 1100, x0 = nar ? 8 : 196, W = nar ? 352 : 830, X = v => W * v / 16; // full panel width
  const rowsY = nar ? [64, 148] : [46, 118], axY = nar ? 204 : 186, Hh = axY + 6;
  sv.setAttribute('viewBox', `0 0 ${VW} ${Hh}`);
  // the unit sits in the top-right corner, above the grid and the end of the axis
  S('text', { x: x0 + W, y: 12, 'text-anchor': 'end', cls: 'ax', text: 'TFLOPs per sample per update', 'font-size': nar ? 12 : null }, sv);
  for (const v of [0, 4, 8, 12, 16]) { S('line', { x1: x0 + X(v), y1: 22, x2: x0 + X(v), y2: axY - 18, cls: v ? 'grid-l' : 'base-l' }, sv); S('text', { x: x0 + X(v), y: axY, 'text-anchor': 'middle', cls: 'ax', text: v, 'font-size': nar ? 12 : null }, sv); }
  [[C.baseName, base, 'generation', true], [C.jamName, jam, `foresight, ${CMP.mode} targets`, false]].forEach(([name, row, sub, isBase], i) => {
    const y = rowsY[i];
    const ny = nar ? y - 26 : y + 3, sy = nar ? y - 26 : y + 21;
    S('text', { x: 0, y: ny, cls: 'ax', text: name, style: `font-family:var(--font-display);font-weight:700;font-size:${nar ? 15 : 14}px;fill:${isBase ? 'var(--violet)' : 'var(--cyan)'}` }, sv);
    S('text', { x: nar ? VW : 0, y: sy, 'text-anchor': nar ? 'end' : null, cls: 'ax', text: sub }, sv);
    let x = x0, outR = -1e9; // right end of the last value written under this bar
    COMP.forEach(([k, f, col]) => {
      const c = row[f];
      if (c.v == null || c.v <= 0) return;
      const w = X(c.v), g = S('g', null, sv);
      S('rect', { x: x + (x > x0 ? 1 : 0), y: y - 12, width: Math.max(1.5, w - (x > x0 ? 2 : 0)), height: 30, rx: 3, style: `fill:${col}` }, g);
      // every segment shows its value: inside when it fits, otherwise under the bar in the segment's colour, left to right
      const t = S('text', { x: x + w / 2, y: y + 7.5, 'text-anchor': 'middle', text: c.p, style: `fill:${INK_ON[col] || 'var(--white)'};font-weight:650;font-size:12px;font-family:var(--font-display)` }, g);
      const tw = t.getComputedTextLength() || 7 * c.p.length;
      if (tw + 8 > w - 2) {
        const lx = Math.max(x + w / 2 - tw / 2, outR + 7);
        t.setAttribute('x', lx.toFixed(1)); t.setAttribute('y', y + 33); t.setAttribute('text-anchor', 'start'); t.style.fill = col;
        outR = lx + tw;
      }
      S('rect', { x, y: y - 16, width: Math.max(12, w), height: 38, cls: 'hit' }, g);
      bindTip(g, () => `<b>${c.p} TFLOPs</b> · ${k} · ${esc(name)}<br>${esc(compTip(k, row, isBase))}${!isBase && k !== 'Head' && k !== 'Target' ? `<br>Training-only parameters: head ${row.params_head.p}M, anchor ${row.params_anchor.p}M.` : ''}${tipSrc(where(c))}`, `${name}, ${k}: ${c.p} TFLOPs`, col);
      x += w;
    });
    S('text', { x: x + 10, y: y + 8, cls: 'v', text: row.tflops_total.p, style: 'font-family:var(--font-display);font-weight:700;font-size:14px;fill:var(--ink)' }, sv);
  });
  finish(sv);
}
function initCompute() {
  cmpNarrow = watchNarrow($('#cmp-box'), 520, renderCompute);
  seg($('#cmp-rec'), [{ v: 'wan', label: 'Wan', title: 'JAM (Wan) vs FastWAM' }, { v: 'ovis', label: 'Ovis-U1', title: 'JAM (Ovis-U1) vs ImageWAM (Ovis-U1)' }], v => { CMP.rec = v; renderCompute(); }, CMP.rec);
  seg($('#cmp-mode'), [{ v: 'cached', label: 'Cached' }, { v: 'online', label: 'Online' }], v => { CMP.mode = v; renderCompute(); }, CMP.mode);
  const lg = $('#cmp-legend');
  COMP.forEach(([l, , c]) => { const s = H('span', null, lg); H('i', { cls: 'sq', style: `background:${c}` }, s); s.append(l); });
  renderCompute();
}

// =============================================================================================
// 09 · ablation small multiples
// =============================================================================================
// title states the finding; the take is optional; table references live in the mark tooltips
function card(title, take, src, bench, quote) {
  const c = H('article', { cls: 'hud card' }, $('#abl'));
  H('h3', { html: title }, c);
  if (take) { const tk = H('p', { cls: 'take', text: take }, c); if (quote) tk.title = `“${quote.t}” (${quote.w})`; }
  else if (quote) c.querySelector('h3').title = `“${quote.t}” (${quote.w})`;
  const sv = chartSVG({ viewBox: '0 0 320 150', 'aria-label': title.replace(/<[^>]+>/g, '') + ' (' + bench + ')' }, c);
  sv.dataset.bench = bench;
  if (src) H('p', { cls: 'src', text: src }, c);
  return sv;
}
function lam(t, label) { // "λF = 0.4" with a subscript F
  const m = label.match(/^λF(.*)$/);
  if (!m) { t.textContent = label; return; }
  S('tspan', { text: 'λ' }, t); S('tspan', { text: 'F', 'font-size': '72%', dy: 3 }, t); S('tspan', { text: m[1], dy: -3 }, t);
}
function dotPlot(sv, items, lo, hi, o = {}) {
  // items: [label, value, highlight, tooltip extra]
  const x0 = o.x0 ?? 104, x1 = 296, X = v => x0 + (x1 - x0) * (v - lo) / (hi - lo), n = items.length, rh = Math.min(25, 108 / n);
  const top = o.ref ? 18 : 2;
  sv.setAttribute('viewBox', `0 0 320 ${top + 26 + n * rh}`);
  for (const v of o.ticks) { S('line', { x1: X(v), y1: top + 2, x2: X(v), y2: top + 6 + n * rh, cls: 'grid-l' }, sv); S('text', { x: X(v), y: top + 20 + n * rh, 'text-anchor': 'middle', cls: 'ax', text: v }, sv); }
  S('text', { x: 0, y: top + 20 + n * rh, cls: 'ax', text: (sv.dataset.bench || '') + ' %' }, sv);
  if (o.ref) {
    S('line', { x1: X(o.ref[0].v), y1: 13, x2: X(o.ref[0].v), y2: top + 8 + n * rh, style: 'stroke:var(--c-gen)', 'stroke-width': 1.5, 'stroke-dasharray': '3 3' }, sv);
    S('text', { x: X(o.ref[0].v), y: 9, 'text-anchor': 'middle', cls: 'ax', style: 'fill:var(--violet)', text: `${o.ref[1]} ${o.ref[0].p}` }, sv);
  }
  items.forEach(([l, x, hl, extra], i) => {
    const y = top + 10 + i * rh, g = S('g', null, sv);
    S('rect', { x: 0, y: y - rh / 2, width: 320, height: rh, cls: 'hit' }, g);
    const t = S('text', { x: 0, y: y + 4, cls: 'ax', style: `font-family:var(--font-body);font-size:12px;fill:${hl ? 'var(--ink)' : 'var(--ink-2)'};font-weight:${hl ? 600 : 400}` }, g);
    lam(t, l);
    S('line', { x1: x0, y1: y, x2: X(x.v), y2: y, style: `stroke:${hl ? 'var(--c-act)' : 'var(--line-2)'}`, 'stroke-width': 2 }, g);
    S('circle', { cx: X(x.v), cy: y, r: hl ? 5.5 : 4.5, style: `fill:${hl ? 'var(--cyan)' : 'var(--panel)'};stroke:${hl ? 'var(--cyan)' : 'var(--muted)'};stroke-width:2` }, g);
    const right = X(x.v) > 262;
    S('text', { x: X(x.v) + (right ? -9 : 9), y: y + 4, cls: 'v', 'text-anchor': right ? 'end' : 'start', text: x.p }, g);
    bindTip(g, () => `<b>${pc(x)}</b> · ${esc(l)}${extra ? '<br>' + esc(extra) : ''}${tipSrc(where(x))}`, `${l}: ${x.p}%`, hl ? 'var(--cyan)' : 'var(--muted)');
  });
}
function initAblations() {
  const A = D.abl;
  const cards = [];
  // 1 · backbone x target initialization (Table 2)
  let sv = card('Pretrain the <em>backbone</em>', null, null, 'LIBERO', A.factorial.q);
  {
    // one row per future anchor; filled = pretrained backbone, hollow = random backbone. The long bars are what the
    // backbone's weights are worth; the right ends line up because a random anchor works as well as a pretrained one
    const c = A.factorial.cells, x0 = 100, x1 = 288, X = v => x0 + (x1 - x0) * (v - 67) / 33;
    sv.setAttribute('viewBox', '0 0 320 146'); // as tall as the action-only card: both rows of cards match
    const body = 'font-family:var(--font-body);font-size:12px;fill:var(--ink-2)';
    const dot = (cx, cy, pre, g) => S('circle', { cx, cy, r: pre ? 5.5 : 4.5, style: pre ? 'fill:var(--cyan);stroke:var(--cyan);stroke-width:2' : 'fill:var(--panel);stroke:var(--muted);stroke-width:2' }, g);
    dot(7, 10, true, sv); S('text', { x: 17, y: 14, text: 'pretrained backbone', style: body }, sv);
    dot(143, 10, false, sv); S('text', { x: 153, y: 14, text: 'random backbone', style: body }, sv);
    for (const v of [70, 80, 90, 100]) { S('line', { x1: X(v), y1: 28, x2: X(v), y2: 124, cls: 'grid-l' }, sv); S('text', { x: X(v), y: 140, 'text-anchor': 'middle', cls: 'ax', text: v }, sv); }
    S('text', { x: 0, y: 140, cls: 'ax', text: 'LIBERO %' }, sv);
    [['Pretrained', 'pp', 'rp', 54], ['Random', 'pr', 'rr', 100]].forEach(([anchor, kp, kr, y]) => {
      S('text', { x: 0, y: y - 6, text: anchor, style: body.replace('ink-2', 'ink') }, sv);
      S('text', { x: 0, y: y + 13, text: 'future anchor', style: body.replace('12px', '11px').replace('ink-2', 'muted') }, sv);
      S('line', { x1: X(c[kr].v), y1: y, x2: X(c[kp].v), y2: y, style: 'stroke:var(--c-act)', 'stroke-width': 2, opacity: .55 }, sv);
      [[c[kr], false], [c[kp], true]].forEach(([x, pre]) => {
        const g = S('g', null, sv);
        S('circle', { cx: X(x.v), cy: y, r: 11, cls: 'hit' }, g);
        dot(X(x.v), y, pre, g);
        S('text', { x: X(x.v), y: y - 10, 'text-anchor': 'middle', cls: 'v', text: x.p }, g); // above its dot, clear of the axis
        bindTip(g, () => `<b>${pc(x)}</b> LIBERO · ${pre ? 'pretrained' : 'random'} backbone, ${anchor.toLowerCase()} future anchor (JAM (Wan), Appendix C Wan settings: block 18, offsets [1,2], λF = 0.6, 20k updates)<br>${esc(A.factorial.qkeep.t)}${tipSrc(where(x))}`,
          `${pre ? 'Pretrained' : 'Random'} backbone, ${anchor.toLowerCase()} anchor: ${x.p}%`, pre ? 'var(--cyan)' : 'var(--muted)');
      });
    });
  }
  cards.push(sv);
  // 2 · lambda_F (Table 12)
  const lbest = A.lambda.pts.reduce((b, p) => (p[1].v > b[1].v ? p : b));
  sv = card(`<span class="m"><i>λ</i><sub>F</sub></span>${esc(lbest[0].slice(2))} <em>works best</em>`, null, null, 'LIBERO', A.lambda.q);
  dotPlot(sv, A.lambda.pts.map(([l, x]) => [l, x, l === lbest[0], l === lbest[0] ? A.lambda.q.t : null]), 95, 98, { ref: [A.lambda.ref, 'FastWAM'], ticks: [95, 96, 97, 98] });
  cards.push(sv);
  // 3 · supervised Wan block (Table 11)
  const dbest = A.depth.pts.reduce((b, p) => (p[1].v > b[1].v ? p : b));
  sv = card(`Supervise <em>${esc(dbest[0].toLowerCase())} of ${A.depth.of}</em>`, null, null, 'LIBERO', A.depth.q2);
  dotPlot(sv, A.depth.pts.map(([l, x]) => [l, x, l === dbest[0], l === dbest[0] ? A.depth.q.t : null]), 95, 98, { ref: [A.depth.ref, 'FastWAM'], ticks: [95, 96, 97, 98] });
  cards.push(sv);
  // 4 · head (Table 13)
  const hbest = A.head.pts.reduce((b, p) => (p[1].v > b[1].v ? p : b));
  sv = card(`A <em>${esc(hbest[2].p)}M</em> transformer head`, null, null, 'LIBERO', A.head.q2); // the chart shows larger heads are not better
  dotPlot(sv, A.head.pts.map(([l, x, p]) => [`${l.split(' · ')[0]} ${p.p}M`, x, p === hbest[2], `Width ${l.split(' · ')[1]}, ${p.p}M parameters. ${p === hbest[2] ? A.head.q.t : A.head.q3.t}`]), 94.5, 98, { x0: 124, ticks: [95, 96, 97, 98] });
  cards.push(sv);
  // 5 · action-only controls (Table 14)
  sv = card('Also lifts <em>Qwen3.5-2B and DINOv2-L</em>', null, null, 'LIBERO-Plus', A.action.q);
  {
    const x0 = 112, X = v => x0 + 186 * (v - 60) / 25;
    sv.setAttribute('viewBox', '0 0 320 146');
    // legend under the title (as in the backbone card); the unit starts the tick row, as in the other cards
    const body = 'font-family:var(--font-body);font-size:12px;fill:var(--ink-2)';
    S('circle', { cx: 7, cy: 10, r: 4.5, style: 'fill:var(--panel);stroke:var(--muted);stroke-width:2' }, sv); S('text', { x: 17, y: 14, text: 'action only', style: body }, sv);
    S('circle', { cx: 105, cy: 10, r: 5.5, style: 'fill:var(--cyan)' }, sv); S('text', { x: 115, y: 14, text: 'JAM', style: body }, sv);
    for (const v of [60, 65, 70, 75, 80, 85]) { S('line', { x1: X(v), y1: 26, x2: X(v), y2: 124, cls: 'grid-l' }, sv); S('text', { x: X(v), y: 140, 'text-anchor': 'middle', cls: 'ax', text: v }, sv); }
    S('text', { x: 0, y: 140, cls: 'ax', text: sv.dataset.bench + ' %' }, sv);
    A.action.rows.forEach(([n, a, b], i) => {
      const y = 54 + i * 46, g = S('g', null, sv);
      S('rect', { x: 0, y: y - 20, width: 320, height: 40, cls: 'hit' }, g);
      S('text', { x: 0, y: y + 4, cls: 'ax', text: n, style: 'font-family:var(--font-body);font-size:12px;fill:var(--ink-2)' }, g);
      S('line', { x1: X(a.v), y1: y, x2: X(b.v) - 8, y2: y, style: 'stroke:var(--c-act)', 'stroke-width': 2 }, g);
      S('path', { d: `M${X(b.v) - 9},${y - 4} l7,4 l-7,4 z`, style: 'fill:var(--c-act)' }, g);
      S('circle', { cx: X(a.v), cy: y, r: 4.5, style: 'fill:var(--panel);stroke:var(--muted);stroke-width:2' }, g);
      S('circle', { cx: X(b.v), cy: y, r: 5.5, style: 'fill:var(--cyan)' }, g);
      S('text', { x: X(a.v), y: y + 19, 'text-anchor': 'middle', cls: 'v', text: a.p }, g);
      S('text', { x: X(b.v), y: y - 10, 'text-anchor': 'middle', cls: 'v', text: b.p }, g);
      const q = i ? A.action.q2 : A.action.q;
      bindTip(g, () => `<b>${esc(n)}</b> action-only ${pc(a)} → with foresight ${pc(b)}<br>“${esc(q.t)}”${tipSrc(where(a) + ' · ' + q.w)}`, `${n}: ${a.p}% to ${b.p}%`);
    });
  }
  cards.push(sv);
  // 6 · expert vs shared weights (Table 15)
  sv = card('Works <em>without an expert</em>', 'Similar observed success to shared 4B, fewer parameters.', null, 'LIBERO-Plus', A.shared.q2);
  dotPlot(sv, A.shared.pts.map(([l, x, p]) => [`${l} · ${p.p}`, x, l === 'Expert', `${l === 'Expert' ? 'Separate action expert' : 'Shared backbone weights'}; ${p.p} deployed parameters. ${A.shared.q2.t}`]), 78, 82, { x0: 118, ticks: [78, 80, 82] });
  cards.push(sv);
  cards.forEach(finish);
}

// =============================================================================================
// 10 · cite + footer
// =============================================================================================
function initCite() {
  const bib = D.meta.bibtex.p;
  $('#bib').textContent = bib;
  const b = $('#copy-bib');
  b.addEventListener('click', () => {
    const done = t => { b.textContent = t; clearTimeout(b._t); b._t = setTimeout(() => { b.textContent = 'Copy'; }, 2200); };
    const fallback = () => { const r = document.createRange(); r.selectNodeContents($('#bib')); const s = getSelection(); s.removeAllRanges(); s.addRange(r); done('Selected · copy it'); };
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(bib).then(() => done('Copied'), fallback);
      else fallback();
    } catch (e) { fallback(); }
  });
  // the copyright holder as in the repository's LICENSE
  $('#foot').innerHTML = `<span>© 2026 The JAM Authors · Code under the <a href="${LINKS.code}/blob/main/LICENSE">MIT License</a></span>`;
}

// =============================================================================================
// Viewers reuse the existing video and controls: no duplicate download, playhead or decoding stream.
const OVERLAY_MEDIA = new Set(), HIDDEN_MEDIA = new Set();
function pauseMediaInto(saved, except) {
  if (TH.playing) { TH.autoPaused = true; thPause(); }
  $$('video').forEach(v => { if (v !== except && !v.paused) { saved.add(v); v.pause(); } });
}
function resumeMediaFrom(saved) {
  saved.forEach(v => { if (mediaCanRun(v) && videoInView(v) && !v.ended && v.dataset.user !== 'paused') v.play().catch(() => {}); });
  saved.clear();
  if (TH.autoPaused && TH.visible && !TH.userPaused && mediaCanRun(vb)) thPlay();
}
function pauseForOverlay(v) { pauseMediaInto(OVERLAY_MEDIA, v); }
function resumeAfterOverlay() { resumeMediaFrom(OVERLAY_MEDIA); }
function initMediaViewer() {
  const dialog = $('#media-viewer'), content = $('#viewer-content');
  let current = null;
  $$('[data-expand]').forEach(button => button.addEventListener('click', () => {
    const v = document.getElementById(button.dataset.expand), col = v.closest('.vid-col');
    const source = v.id === 'probe-v' ? M.probe : M.att;
    const wasPlaying = !v.paused, anchor = document.createComment('expanded-video');
    current = { v, col, anchor, button };
    col.before(anchor); pauseForOverlay(v);
    col.style.setProperty('--video-ratio', source.w / source.h);
    content.append(col);
    $('#viewer-title').textContent = v.id === 'probe-v' ? 'Future-feature readout' : 'Attention analysis';
    document.documentElement.classList.add('locked');
    dialog.showModal(); $('#viewer-close').focus();
    if (wasPlaying && mediaCanRun(v)) v.play().catch(() => {});
  }));
  $('#viewer-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    if (!current) return;
    const { v, col, anchor, button } = current, wasPlaying = !v.paused;
    anchor.replaceWith(col); current = null;
    document.documentElement.classList.remove('locked');
    button.focus({ preventScroll: true });
    if (wasPlaying && mediaCanRun(v) && videoInView(v)) v.play().catch(() => {});
    else v.pause();
    resumeAfterOverlay();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pauseMediaInto(HIDDEN_MEDIA);
    else resumeMediaFrom(HIDDEN_MEDIA);
    const hero = $('.hero'), r = hero.getBoundingClientRect();
    hero.classList.toggle('off', document.hidden || r.bottom <= 54 || r.top >= innerHeight);
  });
}

function tableCSV(table) {
  const rows = [...table.rows], width = Math.max(...rows.map(r => [...r.cells].reduce((n, c) => n + c.colSpan, 0)));
  return rows.map(row => {
    const values = [];
    [...row.cells].forEach(cell => { values.push((cell.dataset.name ?? cell.textContent).replace(/\s+/g, ' ').trim()); for (let i = 1; i < cell.colSpan; i++) values.push(''); });
    while (values.length < width) values.push('');
    return values.map(value => '"' + value.replace(/"/g, '""') + '"').join(',');
  }).join('\r\n') + '\r\n';
}
function initTableExports() {
  $$('[data-csv]').forEach(button => button.addEventListener('click', () => {
    const table = document.querySelector('#' + button.dataset.csv + ' table');
    if (!table) return;
    const blob = new Blob(['\ufeff' + tableCSV(table)], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = button.dataset.csv === 'bd-table' ? `jam-${bdTab}-results.csv` : `jam-libero-plus-${rbPair}-paired.csv`;
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }));
  const hints = () => $$('.table-tools').forEach(tools => { const table = tools.nextElementSibling; tools.querySelector('span').hidden = table.scrollWidth <= table.clientWidth + 1; });
  $$('details.tv').forEach(d => d.addEventListener('toggle', hints));
  $('#bd-tab').addEventListener('click', () => requestAnimationFrame(hints));
  $('#rb-pair').addEventListener('click', () => requestAnimationFrame(hints));
  addEventListener('resize', hints); hints();
}

// boot
// =============================================================================================
function boot() {
  const steps = [initHero, initTrailer, initResources, initNav, initMixer, initPerBackbone, initRobust, initTheater, initBoards, initTarget, initProbe, initAttention, initCompute, initAblations, initCite, initMediaViewer, initTableExports];
  for (const f of steps) { try { f(); } catch (e) { console.error('[jam] ' + f.name + ' failed:', e); } }
  // text sizes depend on the laid-out width; fonts change mark extents
  requestAnimationFrame(refreshCharts);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(refreshCharts);
}
boot();
