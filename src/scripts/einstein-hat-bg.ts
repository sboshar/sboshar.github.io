/**
 * About-page background: Kaplan H7/H8 hat continuum + edge “rain”.
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

/* ---------- Canvas drawing ---------- */

/** Four gray fills for light background (neutral, semi-transparent). */
const GRAYS_LIGHT = [
  'rgba(236, 236, 236, 0.52)',
  'rgba(210, 210, 210, 0.5)',
  'rgba(186, 186, 186, 0.48)',
  'rgba(162, 162, 162, 0.46)',
] as const;

/** Four gray fills for dark mode. */
const GRAYS_DARK = [
  'rgba(58, 58, 58, 0.5)',
  'rgba(72, 72, 72, 0.48)',
  'rgba(88, 88, 88, 0.46)',
  'rgba(104, 104, 104, 0.44)',
] as const;

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
  /** Stable 0..3 gray band per tile (does not depend on morph). */
  const tileGrayIx = flatShapes.map((_, i) => i % 4);

  const aaRef = SHAPE_ALPHA * HAT_CLASSIC_SHAPE_V;
  const bbRef = SHAPE_ALPHA * (1 - HAT_CLASSIC_SHAPE_V);
  const refMoments = computePatchMoments(flatShapes, aaRef, bbRef, null);

  let animId = 0;
  let reducedMotion = false;

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

  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
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
    const grays = dark ? GRAYS_DARK : GRAYS_LIGHT;
    const stroke = dark ? 'rgba(200, 200, 200, 0.14)' : 'rgba(48, 48, 48, 0.15)';
    const lw = dark ? 0.35 : 0.45;

    ctx.clearRect(0, 0, v.w, v.h);

    for (let ti = 0; ti < flatShapes.length; ti++) {
      const sh = flatShapes[ti];
      ctx.beginPath();
      for (let i = 0; i < sh.pts.length; i++) {
        const o = evalSym(sh.pts[i], aa, bb);
        const st = stabilizePoint(o.x, o.y, curMom, refMoments);
        const p = toScreen(st.x, st.y, v);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.fillStyle = grays[tileGrayIx[ti] ?? 0];
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lw;
      ctx.stroke();
    }
  }

  function tick(timeMs: number) {
    const wSlider = reducedMotion
      ? HAT_CLASSIC_SHAPE_V
      : 0.5 + 0.5 * Math.sin((timeMs / SHAPE_CYCLE_MS) * 2 * PI);
    const wClamped = Math.min(0.99, Math.max(0.01, wSlider));
    const aa = SHAPE_ALPHA * wClamped;
    const bb = SHAPE_ALPHA * (1 - wClamped);

    const curMom = computePatchMoments(flatShapes, aa, bb, refMoments.ang);
    const stableSpan = stabilizedPatchSpan(flatShapes, aa, bb, curMom, refMoments);

    const v = getView(stableSpan);
    drawFrame(v, aa, bb, curMom);

    animId = requestAnimationFrame(tick);
  }

  function onResize() {
    layout();
  }

  reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  layout();
  animId = requestAnimationFrame(tick);
  window.addEventListener('resize', onResize);

  return () => {
    cancelAnimationFrame(animId);
    window.removeEventListener('resize', onResize);
  };
}
