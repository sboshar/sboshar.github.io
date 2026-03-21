/**
 * About-page background: Kaplan H7/H8 hat continuum.
 *
 * Ported from Craig Kaplan’s h7h8.js (same logic as https://cs.uwaterloo.ca/~csk/hat/h7h8.html ),
 * associated with Smith–Myers–Kaplan–Goodman-Strauss (2023). Use aligns with the
 * BSD-3-Clause hatviz family (https://github.com/isohedral/hatviz).
 */

const PI = Math.PI;
const SQRT3 = Math.sqrt(3);
/** α = 1 + √3; slider w∈[0,1] ⇒ a = αw, b = α(1−w) — h7h8.js */
const SHAPE_ALPHA = 1 + SQRT3;
/** Classic hat: default knob on h7h8, w = 1/(1+√3). */
export const HAT_CLASSIC_SHAPE_V = 1 / (1 + SQRT3);

function pt(x: number, y: number) {
  return { x, y };
}

/* ---------- Symbolic (a,b) linear forms — h7h8.js ---------- */

type AB = { a: number; b: number };
type SymPt = { x: AB; y: AB };

const mAB = (ca: number, cb: number): AB => ({ a: ca, b: cb });
const addAB = (u: AB, v: AB): AB => ({ a: u.a + v.a, b: u.b + v.b });
const subAB = (u: AB, v: AB): AB => ({ a: u.a - v.a, b: u.b - v.b });
const scaleAB = (u: AB, alpha: number): AB => ({ a: u.a * alpha, b: u.b * alpha });

function evalABcoord(u: AB, aa: number, bb: number) {
  return aa * u.a + bb * u.b;
}

function evalSym(p: SymPt, aa: number, bb: number) {
  return pt(evalABcoord(p.x, aa, bb), evalABcoord(p.y, aa, bb));
}

function paddSym(P: SymPt, Q: SymPt): SymPt {
  return { x: addAB(P.x, Q.x), y: addAB(P.y, Q.y) };
}

function psubSym(P: SymPt, Q: SymPt): SymPt {
  return { x: subAB(P.x, Q.x), y: subAB(P.y, Q.y) };
}

function trot2(ang: number) {
  const c = Math.cos(ang);
  const s = Math.sin(ang);
  return [c, -s, s, c];
}

function transAB(M: readonly number[], P: SymPt): SymPt {
  return {
    x: addAB(scaleAB(P.x, M[0]), scaleAB(P.y, M[1])),
    y: addAB(scaleAB(P.x, M[2]), scaleAB(P.y, M[3])),
  };
}

/* ---------- Geometry tree ---------- */

type TileLabel = 'single' | 'unflipped' | 'flipped';

class HatShape {
  pts: SymPt[];
  quad: SymPt[];
  label: TileLabel;

  constructor(pts: SymPt[], quad: SymPt[], label: TileLabel) {
    this.pts = pts;
    this.quad = quad;
    this.label = label;
  }

  translateInPlace(dp: SymPt) {
    for (let i = 0; i < this.pts.length; i++) {
      this.pts[i] = paddSym(this.pts[i], dp);
    }
    for (let i = 0; i < 4; i++) {
      this.quad[i] = paddSym(this.quad[i], dp);
    }
  }

  rotateAndMatch(T: readonly number[], qidx: number, P?: SymPt): HatShape {
    const pts = this.pts.map((p) => transAB(T, p));
    const quad = this.quad.map((p) => transAB(T, p));
    const ret = new HatShape(pts, quad, this.label);
    if (qidx >= 0 && P !== undefined) {
      ret.translateInPlace(psubSym(P, quad[qidx]));
    }
    return ret;
  }
}

class HatMeta {
  geoms: HatGeom[];
  quad: SymPt[];

  constructor() {
    this.geoms = [];
    this.quad = [];
  }

  addChild(g: HatGeom) {
    this.geoms.push(g);
  }

  translateInPlace(dp: SymPt) {
    for (const g of this.geoms) {
      g.translateInPlace(dp);
    }
    for (let i = 0; i < 4; i++) {
      this.quad[i] = paddSym(this.quad[i], dp);
    }
  }

  rotateAndMatch(T: readonly number[], qidx: number, P?: SymPt): HatMeta {
    const ret = new HatMeta();
    ret.geoms = this.geoms.map((g) => g.rotateAndMatch(T, -1));
    ret.quad = this.quad.map((p) => transAB(T, p));
    if (qidx >= 0 && P !== undefined) {
      ret.translateInPlace(psubSym(P, ret.quad[qidx]));
    }
    return ret;
  }
}

type HatGeom = HatShape | HatMeta;

function isShape(g: HatGeom): g is HatShape {
  return g instanceof HatShape;
}

/* ---------- buildBaseTiles / buildSupertiles — h7h8.js ---------- */

function buildBaseTiles(): { H8: HatShape; H7: HatMeta } {
  const edges = [
    ['a', 0],
    ['a', 2],
    ['b', 11],
    ['b', 1],
    ['a', 4],
    ['a', 2],
    ['b', 5],
    ['b', 3],
    ['a', 6],
    ['a', 8],
    ['a', 8],
    ['a', 10],
    ['b', 7],
  ] as const;

  const hr3 = 0.5 * SQRT3;
  const dirs = [
    pt(1, 0),
    pt(hr3, 0.5),
    pt(0.5, hr3),
    pt(0, 1),
    pt(-0.5, hr3),
    pt(-hr3, 0.5),
    pt(-1, 0),
    pt(-hr3, -0.5),
    pt(-0.5, -hr3),
    pt(0, -1),
    pt(0.5, -hr3),
    pt(hr3, -0.5),
  ];

  let prev: SymPt = { x: mAB(0, 0), y: mAB(0, 0) };
  const pts: SymPt[] = [prev];

  for (const e of edges) {
    const d = dirs[e[1]];
    if (e[0] === 'a') {
      prev = {
        x: addAB(prev.x, mAB(d.x, 0)),
        y: addAB(prev.y, mAB(d.y, 0)),
      };
    } else {
      prev = {
        x: addAB(prev.x, mAB(0, d.x)),
        y: addAB(prev.y, mAB(0, d.y)),
      };
    }
    pts.push(prev);
  }

  const quad: SymPt[] = [pts[1], pts[3], pts[9], pts[13]];
  const ret: { H8: HatShape; H7: HatMeta } = {} as { H8: HatShape; H7: HatMeta };

  ret.H8 = new HatShape(pts, quad, 'single');

  const fpts: SymPt[] = [];
  const len = pts.length;
  for (let idx = 0; idx < len; ++idx) {
    const p = pts[len - 1 - idx];
    fpts.push({ x: p.x, y: scaleAB(p.y, -1) });
  }
  const dp = psubSym(pts[0], fpts[5]);
  for (let idx = 0; idx < len; ++idx) {
    fpts[idx] = paddSym(fpts[idx], dp);
  }

  const comp = new HatMeta();
  comp.addChild(new HatShape(pts, quad, 'unflipped'));
  comp.addChild(new HatShape(fpts, quad, 'flipped'));
  comp.quad = quad;
  ret.H7 = comp;

  return ret;
}

function buildSupertiles(sys: { H8: HatShape | HatMeta; H7: HatMeta }) {
  const sing = sys.H8;
  const comp = sys.H7;

  const smeta = new HatMeta();
  const rules = [
    [PI / 3, 2, 0, false],
    [(2 * PI) / 3, 2, 0, false],
    [0, 1, 1, true],
    [(-2 * PI) / 3, 2, 2, false],
    [-PI / 3, 2, 0, false],
    [0, 2, 0, false],
  ] as const;

  smeta.addChild(sing);
  for (const r of rules) {
    const last = smeta.geoms[smeta.geoms.length - 1];
    const q = last instanceof HatShape ? last.quad : last.quad;
    if (r[3]) {
      smeta.addChild(comp.rotateAndMatch(trot2(r[0]), r[1], q[r[2]]));
    } else {
      smeta.addChild(sing.rotateAndMatch(trot2(r[0]), r[1], q[r[2]]));
    }
  }

  const g1 = smeta.geoms[1];
  const g2 = smeta.geoms[2];
  const g4 = smeta.geoms[4];
  const g6 = smeta.geoms[6];
  smeta.quad = [
    g1 instanceof HatShape ? g1.quad[3] : g1.quad[3],
    g2 instanceof HatShape ? g2.quad[0] : g2.quad[0],
    g4 instanceof HatShape ? g4.quad[3] : g4.quad[3],
    g6 instanceof HatShape ? g6.quad[0] : g6.quad[0],
  ];

  const cmeta = new HatMeta();
  cmeta.geoms = smeta.geoms.slice(0, smeta.geoms.length - 1);
  cmeta.quad = smeta.quad;

  return { H8: smeta, H7: cmeta };
}

function buildH8Patch(substeps: number): HatMeta {
  let sys: { H8: HatShape | HatMeta; H7: HatMeta } = buildBaseTiles();
  for (let i = 0; i < substeps; i++) {
    sys = buildSupertiles(sys);
  }
  return sys.H8 as HatMeta;
}

/* ---------- Flatten, bbox, edges ---------- */

function flattenShapes(g: HatGeom): HatShape[] {
  if (isShape(g)) return [g];
  const out: HatShape[] = [];
  for (const c of g.geoms) {
    out.push(...flattenShapes(c));
  }
  return out;
}

/** Centroid + principal-axis angle so we can cancel global drift/spin during Tile(a,b) morph. */
type PatchMoments = { cx: number; cy: number; ang: number };

function computePatchMoments(
  shapes: HatShape[],
  aa: number,
  bb: number,
  orientToRef: number | null
): PatchMoments {
  let sx = 0,
    sy = 0,
    n = 0;
  const pts: { x: number; y: number }[] = [];
  for (const sh of shapes) {
    for (const p of sh.pts) {
      const o = evalSym(p, aa, bb);
      pts.push(o);
      sx += o.x;
      sy += o.y;
      n++;
    }
  }
  const cx = sx / n;
  const cy = sy / n;
  let cxx = 0,
    cyy = 0,
    cxy = 0;
  for (const o of pts) {
    const dx = o.x - cx;
    const dy = o.y - cy;
    cxx += dx * dx;
    cyy += dy * dy;
    cxy += dx * dy;
  }
  let theta = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
  if (orientToRef !== null) {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const cr = Math.cos(orientToRef);
    const sr = Math.sin(orientToRef);
    if (c * cr + s * sr < 0) theta += PI;
  }
  return { cx, cy, ang: theta };
}

/** Map evaluated point into the reference frame (classic hat): no net translation/rotation vs ref. */
function stabilizePoint(x: number, y: number, cur: PatchMoments, ref: PatchMoments) {
  const dx = x - cur.cx;
  const dy = y - cur.cy;
  const da = ref.ang - cur.ang;
  const co = Math.cos(da);
  const si = Math.sin(da);
  const rx = co * dx - si * dy;
  const ry = si * dx + co * dy;
  return { x: rx + ref.cx, y: ry + ref.cy };
}

/** Max side of axis-aligned bbox of the stabilized patch (constant on-screen size vs morph). */
function stabilizedPatchSpan(
  shapes: HatShape[],
  aa: number,
  bb: number,
  cur: PatchMoments,
  ref: PatchMoments
): number {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const sh of shapes) {
    for (const p of sh.pts) {
      const o = evalSym(p, aa, bb);
      const st = stabilizePoint(o.x, o.y, cur, ref);
      minX = Math.min(minX, st.x);
      minY = Math.min(minY, st.y);
      maxX = Math.max(maxX, st.x);
      maxY = Math.max(maxY, st.y);
    }
  }
  return Math.max(maxX - minX, maxY - minY, 1e-6);
}

/** Evaluated 13-gon boundary of the primitive H8 tile at slider w (for API compat). */
export function hatOutlineFromShapeParam(w: number): { x: number; y: number }[] {
  const ww = Math.min(1, Math.max(0, w));
  const aa = SHAPE_ALPHA * ww;
  const bb = SHAPE_ALPHA * (1 - ww);
  const { H8 } = buildBaseTiles();
  return H8.pts.map((p) => evalSym(p, aa, bb));
}

/* ---------- Canvas: wireframe (glow = CSS filter on canvas — avoids per-path shadowBlur) ---------- */

const EDGE_STROKE_LIGHT = 'rgba(45, 48, 58, 0.32)';
const EDGE_STROKE_DARK = 'rgba(210, 214, 225, 0.26)';
const EDGE_LINE_WIDTH_LIGHT = 0.7;
const EDGE_LINE_WIDTH_DARK = 0.65;

/** Pointer near an edge → light runs the tile outline and fades. */
const PULSE_SPEED = 1.35;
const PULSE_BRIGHT_DECAY = 0.38;
const PULSE_HIT_MAX_PX = 34;
const PULSE_MAX_COUNT = 20;
/** Higher = fewer new pulses while the cursor moves. */
const HOVER_SPAWN_INTERVAL_MS = 220;

type LightPulse = {
  shapeIdx: number;
  edgeIdx: number;
  /** 0..1 along edge edgeIdx → edgeIdx+1 */
  t: number;
  /** 1 → 0 over time */
  bright: number;
};

function distPointToSeg(
  px: number,
  py: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): { dist: number; t: number } {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len2 = dx * dx + dy * dy;
  if (len2 < 1e-14) return { dist: Math.hypot(px - x1, py - y1), t: 0 };
  let t = ((px - x1) * dx + (py - y1) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  const qx = x1 + t * dx;
  const qy = y1 + t * dy;
  return { dist: Math.hypot(px - qx, py - qy), t };
}

export function initEinsteinHatBg(canvasId: string) {
  const el = document.getElementById(canvasId) as HTMLCanvasElement | null;
  const ctx0 = el?.getContext('2d');
  if (!el || !ctx0) return () => {};

  const canvas = el;
  const ctx = ctx0;

  const SUBST_STEPS = 4;
  /** Higher = larger tiles on screen (scale is fixed from classic-hat bbox, not current morph). */
  const TILE_ZOOM = 6.8;
  const SHAPE_CYCLE_MS = 52000;

  let root: HatMeta;
  try {
    root = buildH8Patch(SUBST_STEPS);
  } catch (e) {
    console.error('[einstein-hat-bg] H8 patch build failed', e);
    return () => {};
  }

  const flatShapes = flattenShapes(root);

  const aaRef = SHAPE_ALPHA * HAT_CLASSIC_SHAPE_V;
  const bbRef = SHAPE_ALPHA * (1 - HAT_CLASSIC_SHAPE_V);
  const refMoments = computePatchMoments(flatShapes, aaRef, bbRef, null);

  let animId = 0;
  let reducedMotion = false;
  const pulses: LightPulse[] = [];
  let lastFrameMs = 0;

  function isDark() {
    return document.documentElement.classList.contains('dark');
  }

  type View = {
    w: number;
    h: number;
    s: number;
    cx: number;
    cy: number;
    cw: number;
    ch: number;
  };

  let lastView: View | null = null;
  let lastAa = aaRef;
  let lastBb = bbRef;
  let lastCurMom: PatchMoments = refMoments;

  let lastHoverSpawnMs = 0;

  /** `stableSpan` = bbox of stabilized geometry this frame (not fixed ref), so zoom tracks morph extent. */
  function getView(stableSpan: number): View {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const pad = 1.08;
    const baseS = (Math.min(w, h) / stableSpan) * pad * 0.92;
    const s = baseS * TILE_ZOOM;
    return {
      w,
      h,
      s,
      cx: refMoments.cx,
      cy: refMoments.cy,
      cw: stableSpan,
      ch: stableSpan,
    };
  }

  function toScreen(x: number, y: number, v: View) {
    return {
      x: (x - v.cx) * v.s + v.w / 2,
      y: -(y - v.cy) * v.s + v.h / 2,
    };
  }

  function findNearestEdge(
    px: number,
    py: number,
    v: View,
    aa: number,
    bb: number,
    curMom: PatchMoments
  ): { shapeIdx: number; edgeIdx: number; t: number } | null {
    let bestD = PULSE_HIT_MAX_PX;
    let best: { shapeIdx: number; edgeIdx: number; t: number } | null = null;
    for (let si = 0; si < flatShapes.length; si++) {
      const sh = flatShapes[si]!;
      const n = sh.pts.length;
      for (let ei = 0; ei < n; ei++) {
        const oa = evalSym(sh.pts[ei]!, aa, bb);
        const ob = evalSym(sh.pts[(ei + 1) % n]!, aa, bb);
        const sa = stabilizePoint(oa.x, oa.y, curMom, refMoments);
        const sb = stabilizePoint(ob.x, ob.y, curMom, refMoments);
        const p0 = toScreen(sa.x, sa.y, v);
        const p1 = toScreen(sb.x, sb.y, v);
        const { dist, t } = distPointToSeg(px, py, p0.x, p0.y, p1.x, p1.y);
        if (dist < bestD) {
          bestD = dist;
          best = { shapeIdx: si, edgeIdx: ei, t };
        }
      }
    }
    return best;
  }

  function updatePulses(dt: number) {
    for (let i = pulses.length - 1; i >= 0; i--) {
      const p = pulses[i]!;
      p.bright *= Math.exp(-PULSE_BRIGHT_DECAY * dt);
      p.t += PULSE_SPEED * dt;
      const sh = flatShapes[p.shapeIdx];
      if (!sh) {
        pulses.splice(i, 1);
        continue;
      }
      const n = sh.pts.length;
      while (p.t >= 1) {
        p.t -= 1;
        p.edgeIdx = (p.edgeIdx + 1) % n;
      }
      if (p.bright < 0.012) pulses.splice(i, 1);
    }
  }

  function drawPulses(v: View, aa: number, bb: number, curMom: PatchMoments) {
    const dark = isDark();
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const p of pulses) {
      const sh = flatShapes[p.shapeIdx];
      if (!sh) continue;
      const n = sh.pts.length;
      const ei = p.edgeIdx;
      const oa = evalSym(sh.pts[ei]!, aa, bb);
      const ob = evalSym(sh.pts[(ei + 1) % n]!, aa, bb);
      const sa = stabilizePoint(oa.x, oa.y, curMom, refMoments);
      const sb = stabilizePoint(ob.x, ob.y, curMom, refMoments);
      const p0 = toScreen(sa.x, sa.y, v);
      const p1 = toScreen(sb.x, sb.y, v);
      const x = p0.x + p.t * (p1.x - p0.x);
      const y = p0.y + p.t * (p1.y - p0.y);
      const tTrail = Math.max(0, p.t - 0.14);
      const x0 = p0.x + tTrail * (p1.x - p0.x);
      const y0 = p0.y + tTrail * (p1.y - p0.y);
      ctx.strokeStyle = dark
        ? `rgba(210, 225, 255,${0.4 * p.bright})`
        : `rgba(95, 120, 230,${0.35 * p.bright})`;
      ctx.lineWidth = 2.4 * p.bright;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x, y);
      ctx.stroke();

      const r = 8 + 24 * Math.sqrt(p.bright);
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      if (dark) {
        g.addColorStop(0, `rgba(240, 248, 255,${0.6 * p.bright})`);
        g.addColorStop(0.4, `rgba(170, 195, 255,${0.2 * p.bright})`);
        g.addColorStop(1, 'rgba(170, 195, 255,0)');
      } else {
        g.addColorStop(0, `rgba(100, 125, 255,${0.45 * p.bright})`);
        g.addColorStop(0.4, `rgba(130, 155, 255,${0.18 * p.bright})`);
        g.addColorStop(1, 'rgba(130, 155, 255,0)');
      }
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * PI);
      ctx.fill();
    }
    ctx.restore();
  }

  function resizeCanvas() {
    /** Cap DPR: fewer pixels to fill each frame (big win vs jitter at high DPR). */
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = window.innerWidth;
    const h = window.innerHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function layout() {
    resizeCanvas();
  }

  function drawFrame(v: View, aa: number, bb: number, curMom: PatchMoments) {
    const dark = isDark();
    const stroke = dark ? EDGE_STROKE_DARK : EDGE_STROKE_LIGHT;
    const lw = dark ? EDGE_LINE_WIDTH_DARK : EDGE_LINE_WIDTH_LIGHT;

    ctx.clearRect(0, 0, v.w, v.h);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lw;

    for (const sh of flatShapes) {
      ctx.beginPath();
      for (let i = 0; i < sh.pts.length; i++) {
        const o = evalSym(sh.pts[i], aa, bb);
        const st = stabilizePoint(o.x, o.y, curMom, refMoments);
        const p = toScreen(st.x, st.y, v);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.stroke();
    }
  }

  function tick(timeMs: number) {
    const dt = lastFrameMs ? Math.min(0.05, (timeMs - lastFrameMs) / 1000) : 0;
    lastFrameMs = timeMs;

    const wSlider = reducedMotion
      ? HAT_CLASSIC_SHAPE_V
      : 0.5 + 0.5 * Math.sin((timeMs / SHAPE_CYCLE_MS) * 2 * PI);
    const wClamped = Math.min(0.99, Math.max(0.01, wSlider));
    const aa = SHAPE_ALPHA * wClamped;
    const bb = SHAPE_ALPHA * (1 - wClamped);

    const curMom = computePatchMoments(flatShapes, aa, bb, refMoments.ang);
    const stableSpan = stabilizedPatchSpan(flatShapes, aa, bb, curMom, refMoments);

    const v = getView(stableSpan);
    lastView = v;
    lastAa = aa;
    lastBb = bb;
    lastCurMom = curMom;

    if (!reducedMotion && pulses.length > 0) {
      updatePulses(dt);
    }

    drawFrame(v, aa, bb, curMom);
    if (!reducedMotion && pulses.length > 0) {
      drawPulses(v, aa, bb, curMom);
    }

    animId = requestAnimationFrame(tick);
  }

  /** Use window-level move so pulses work over main content too (canvas stays behind text). */
  function onGlobalPointerMove(e: PointerEvent) {
    if (reducedMotion || !lastView) return;
    const now = performance.now();
    if (now - lastHoverSpawnMs < HOVER_SPAWN_INTERVAL_MS) return;

    const rect = canvas.getBoundingClientRect();
    const xCss = e.clientX - rect.left;
    const yCss = e.clientY - rect.top;
    const sx = rect.width > 1 ? lastView.w / rect.width : 1;
    const sy = rect.height > 1 ? lastView.h / rect.height : 1;
    const x = xCss * sx;
    const y = yCss * sy;
    if (x < 0 || y < 0 || x > lastView.w || y > lastView.h) return;

    const hit = findNearestEdge(x, y, lastView, lastAa, lastBb, lastCurMom);
    if (!hit || pulses.length >= PULSE_MAX_COUNT) return;

    lastHoverSpawnMs = now;
    pulses.push({
      shapeIdx: hit.shapeIdx,
      edgeIdx: hit.edgeIdx,
      t: hit.t,
      bright: 1,
    });
  }

  function onResize() {
    layout();
  }

  reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  layout();
  window.addEventListener('pointermove', onGlobalPointerMove, { passive: true });
  animId = requestAnimationFrame(tick);
  window.addEventListener('resize', onResize);

  return () => {
    cancelAnimationFrame(animId);
    window.removeEventListener('resize', onResize);
    window.removeEventListener('pointermove', onGlobalPointerMove);
  };
}
