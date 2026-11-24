/* chucha core: file parsers, bonds, VESTA polyhedra and convex hulls. No DOM; runs in the page and in node.
   A frame is {symbols, xyz: [[x,y,z]...], cell: [[a],[b],[c]] | null, periodic, data: {name: [per-atom]}, labels, name}.
   A volume is {n: [n0,n1,n2], values: Float32Array (C order, periodic grid over cell), cell, name}. */
(function (root) {
"use strict";
let TABLES = {colors: {}, rules: {}, covalent: {}, symbols: []};  // filled by setTables (from Python's tables)
function setTables(t) { TABLES = t; }
const BOHR = 0.529177210903;

// ------------------------------------------------------------------ vectors and cells
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const add = (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
const scale = (a, s) => [a[0]*s, a[1]*s, a[2]*s];
const norm = a => Math.hypot(a[0], a[1], a[2]);
const det3 = A => dot(A[0], cross(A[1], A[2]));
function inv3(A) {  // M with frac = xyz · M (rows of A are the cell vectors)
  const d = det3(A), [a, b, c] = A;
  return [[b[1]*c[2]-b[2]*c[1], a[2]*c[1]-a[1]*c[2], a[1]*b[2]-a[2]*b[1]],
          [b[2]*c[0]-b[0]*c[2], a[0]*c[2]-a[2]*c[0], a[2]*b[0]-a[0]*b[2]],
          [b[0]*c[1]-b[1]*c[0], a[1]*c[0]-a[0]*c[1], a[0]*b[1]-a[1]*b[0]]].map(r => r.map(x => x / d));
}
const toFrac = (r, M) => [0, 1, 2].map(j => r[0]*M[0][j] + r[1]*M[1][j] + r[2]*M[2][j]);
const toCart = (f, A) => [0, 1, 2].map(j => f[0]*A[0][j] + f[1]*A[1][j] + f[2]*A[2][j]);
function cellFromPar(a, b, c, al, be, ga) {  // ASE convention: a along x, b in the xy plane
  const r = Math.PI / 180, cs = x => { const v = Math.cos(x * r); return Math.abs(v) < 1e-10 ? 0 : v; };
  const ca = cs(al), cb = cs(be), cg = cs(ga), sg = Math.sin(ga * r);
  const cy = (ca - cb * cg) / sg;
  return [[a, 0, 0], [b * cg, b * sg, 0], [c * cb, c * cy, c * Math.sqrt(Math.max(0, 1 - cb * cb - cy * cy))]];
}
const isPeriodic = f => !!(f.periodic && f.cell && Math.abs(det3(f.cell)) > 1e-6);

// ------------------------------------------------------------------ parsers
const nums = s => s.trim().split(/\s+/).map(Number);
function cleanSymbol(s) {  // "Fe_pv" -> "Fe", "FE1" -> "Fe", "O2-" -> "O"
  const m = String(s).match(/^([A-Za-z])([a-z]?)/);
  if (!m) return "X";
  let e = m[1].toUpperCase() + m[2];
  if (m[2] && !TABLES.colors[e] && TABLES.colors[m[1].toUpperCase()]) e = m[1].toUpperCase();
  return e;
}
function poscarHeader(L, k) {  // -> {cell, species, counts, k (next line)}
  let s = nums(L[k + 1]);
  const lat = [nums(L[k + 2]), nums(L[k + 3]), nums(L[k + 4])].map(r => r.slice(0, 3));
  let sc = s[0];
  if (sc < 0) sc = Math.cbrt(-sc / Math.abs(det3(lat)));          // negative scale = cell volume
  const cell = lat.map(r => scale(r, sc));
  let t = L[k + 5].trim().split(/\s+/), species, counts, next;
  if (t.every(x => /^\d+$/.test(x))) {                              // VASP 4: no species line
    counts = t.map(Number); species = L[k].trim().split(/\s+/); next = k + 6;
    if (species.length < counts.length) species = counts.map(() => "X");
  } else { species = t; counts = nums(L[k + 6]); next = k + 7; }
  species = species.slice(0, counts.length).map(cleanSymbol);
  const symbols = [];
  counts.forEach((c, i) => { for (let j = 0; j < c; j++) symbols.push(species[i]); });
  return {cell, symbols, k: next, scale: sc};
}
function parsePoscar(text, name) {
  const L = text.split(/\r?\n/), h = poscarHeader(L, 0);
  let k = h.k;
  if (/^\s*s/i.test(L[k])) k++;                                     // Selective dynamics
  const cart = /^\s*[ck]/i.test(L[k]); k++;
  const xyz = h.symbols.map((_, i) => {
    const v = nums(L[k + i]).slice(0, 3);
    return cart ? scale(v, h.scale) : toCart(v, h.cell);
  });
  return {frames: [{symbols: h.symbols, xyz, cell: h.cell, periodic: true, data: {}, labels: null, name}]};
}
function parseXdatcar(text, name) {
  const L = text.split(/\r?\n/), frames = [];
  let k = 0, h = null;
  while (k < L.length) {
    if (!L[k].trim()) { k++; continue; }
    if (!/configuration/i.test(L[k]) && !/^\s*direct\s*$/i.test(L[k])) { h = poscarHeader(L, k); k = h.k; continue; }
    if (!h) throw new Error("XDATCAR: no header");
    k++;
    const xyz = h.symbols.map((_, i) => toCart(nums(L[k + i]).slice(0, 3), h.cell));
    k += h.symbols.length;
    frames.push({symbols: h.symbols, xyz, cell: h.cell, periodic: true, data: {}, labels: null, name: `${name} #${frames.length}`});
  }
  if (!frames.length) throw new Error("XDATCAR: no configurations");
  return {frames};
}
function parseXyz(text, name) {  // plain and extended xyz, several frames
  const L = text.split(/\r?\n/), frames = [];
  let k = 0;
  while (k < L.length) {
    if (!L[k].trim()) { k++; continue; }
    const n = parseInt(L[k]); if (!(n >= 0)) throw new Error(`xyz: expected an atom count on line ${k + 1}`);
    const info = {}, re = /(\w+)\s*=\s*("([^"]*)"|'([^']*)'|(\S+))/g;
    let m; while ((m = re.exec(L[k + 1] || ""))) info[m[1].toLowerCase()] = m[3] ?? m[4] ?? m[5];
    let cols = [["species", "S", 1], ["pos", "R", 3]];
    if (info.properties) {
      const p = info.properties.split(":"); cols = [];
      for (let i = 0; i + 2 < p.length; i += 3) cols.push([p[i], p[i + 1], +p[i + 2]]);
    }
    const symbols = [], xyz = [], data = {};
    for (let a = 0; a < n; a++) {
      const t = L[k + 2 + a].trim().split(/\s+/);
      let c = 0;
      cols.forEach(([key, type, w]) => {
        const v = t.slice(c, c + w); c += w;
        if (key === "species") symbols.push(cleanSymbol(v[0]));
        else if (key === "pos") xyz.push(v.map(Number));
        else if (type === "R" || type === "I") (data[key] = data[key] || []).push(w === 1 ? +v[0] : v.map(Number));
      });
    }
    let cell = null;
    if (info.lattice) { const v = nums(info.lattice); cell = [v.slice(0, 3), v.slice(3, 6), v.slice(6, 9)]; }
    const pbc = info.pbc ? /t/i.test(info.pbc) : !!cell;
    frames.push({symbols, xyz, cell, periodic: pbc && !!cell, data, labels: null, name: frames.length ? `${name} #${frames.length}` : name});
    k += n + 2;
  }
  if (frames.length > 1) frames[0].name = `${name} #0`;
  return {frames};
}
function parseCube(text, name) {
  const L = text.split(/\r?\n/);
  const h = nums(L[2]); let na = h[0];
  const origin = h.slice(1, 4);
  const ax = [3, 4, 5].map(i => nums(L[i]));
  const bohr = ax[0][0] > 0, u = bohr ? BOHR : 1;
  const n = ax.map(r => Math.abs(r[0]));
  const cell = ax.map((r, i) => scale(r.slice(1, 4), n[i] * u));
  let k = 6; const extra = na < 0; na = Math.abs(na);
  const symbols = [], xyz = [];
  for (let a = 0; a < na; a++, k++) {
    const t = nums(L[k]);
    symbols.push(TABLES.symbols[t[0]] || "X");
    xyz.push(scale(sub(t.slice(2, 5), origin), u));
  }
  if (extra) k++;
  const total = n[0] * n[1] * n[2], values = new Float32Array(total);
  let p = 0;
  for (; k < L.length && p < total; k++) { const s = L[k].trim(); if (!s) continue;
    for (const x of s.split(/\s+/)) values[p++] = +x; }
  if (p < total) throw new Error(`cube: ${p} of ${total} grid values`);
  return {frames: [{symbols, xyz, cell, periodic: true, data: {}, labels: null, name}],
          volume: {n, values, cell, name}};
}
function parseChgcar(text, name) {
  // every grid block: 1 = total; 2 = total, magnetization (spin-polarized); 4 = total, mx, my, mz (noncollinear).
  // Between blocks VASP writes augmentation occupancies and per-atom moments; a block starts at a line holding exactly
  // the grid size. CHG-type files are divided by the cell volume (density in e/Å³, like ASE)
  const L = text.split(/\r?\n/), h = poscarHeader(L, 0);
  let k = h.k;
  if (/^\s*s/i.test(L[k])) k++;
  const cart = /^\s*[ck]/i.test(L[k]); k++;
  const xyz = h.symbols.map((_, i) => { const v = nums(L[k + i]).slice(0, 3); return cart ? scale(v, h.scale) : toCart(v, h.cell); });
  k += h.symbols.length;
  while (k < L.length && !L[k].trim()) k++;
  const head = L[k].trim(), n = nums(head).slice(0, 3), total = n[0] * n[1] * n[2];
  const div = /^(CHG|PARCHG|AECCAR)/i.test(name.split("/").pop()) ? Math.abs(det3(h.cell)) : 1;
  const blocks = [];
  while (k < L.length && blocks.length < 4) {
    if (L[k].trim().split(/\s+/).join(" ") !== head.split(/\s+/).join(" ")) { k++; continue; }
    k++;
    const raw = new Float32Array(total);
    let p = 0;
    for (; k < L.length && p < total; k++) for (const x of L[k].trim().split(/\s+/)) { if (x && p < total) raw[p++] = +x; }
    if (p < total) { if (!blocks.length) throw new Error(`${name}: ${p} of ${total} grid values`); break; }
    const values = new Float32Array(total);   // file: x fastest -> C order (z fastest)
    for (let z = 0; z < n[2]; z++) for (let y = 0; y < n[1]; y++) for (let x = 0; x < n[0]; x++)
      values[(x * n[1] + y) * n[2] + z] = raw[x + n[0] * (y + n[1] * z)] / div;
    blocks.push(values);
  }
  const names = blocks.length >= 4 ? ["total", "mx", "my", "mz"] : blocks.length === 2 ? ["total", "magnetization"] : ["total"];
  const channels = {}; names.forEach((c, i) => { channels[c] = blocks[i]; });
  return {frames: [{symbols: h.symbols, xyz, cell: h.cell, periodic: true, data: {}, labels: null, name}],
          volume: {n, values: blocks[0], cell: h.cell, name, channels}};
}
function channelValues(channels, c) {  // a channel of a volumetric file; spin up / down from total and magnetization
  if (channels[c]) return channels[c];
  const t = channels.total, m = channels.magnetization;
  if (!t || !m || (c !== "up" && c !== "down")) return null;
  const out = new Float32Array(t.length), s = c === "up" ? 1 : -1;
  for (let i = 0; i < t.length; i++) out[i] = 0.5 * (t[i] + s * m[i]);
  return out;
}
const CHANNEL_NAMES = {total: "Total", magnetization: "Magnetization (↑−↓)", up: "Spin ↑", down: "Spin ↓", mx: "mx", my: "my", mz: "mz"};
function channelList(channels) {  // what a volume can show
  const c = Object.keys(channels || {});
  return c.includes("magnetization") ? [...c, "up", "down"] : c;
}

// CIF: first data block; symmetry operations applied, duplicates (special positions) merged
function cifTokens(text) {
  const out = [], L = text.split(/\r?\n/);
  for (let i = 0; i < L.length; i++) {
    const line = L[i];
    if (line.startsWith(";")) {                                     // multi-line text field
      const buf = [line.slice(1)];
      while (++i < L.length && !L[i].startsWith(";")) buf.push(L[i]);
      out.push(buf.join("\n")); continue;
    }
    const re = /'((?:[^']|'(?=\S))*)'(?=\s|$)|"((?:[^"]|"(?=\S))*)"(?=\s|$)|(#.*$)|(\S+)/g;
    let m; while ((m = re.exec(line))) { if (m[3]) break; out.push(m[1] ?? m[2] ?? m[4]); }
  }
  return out;
}
function parseCifBlock(text) {
  const t = cifTokens(text), tags = {}, loops = [];
  let i = 0, blocks = 0;
  while (i < t.length) {
    const s = t[i], low = s.toLowerCase();
    if (low.startsWith("data_")) { if (blocks++) break; i++; continue; }
    if (low === "loop_") {
      const names = []; i++;
      while (i < t.length && t[i].startsWith("_")) names.push(t[i++].toLowerCase());
      const vals = [];
      while (i < t.length && !t[i].startsWith("_") && t[i].toLowerCase() !== "loop_" && !t[i].toLowerCase().startsWith("data_")) vals.push(t[i++]);
      const rows = []; for (let r = 0; r + names.length <= vals.length; r += names.length) rows.push(vals.slice(r, r + names.length));
      loops.push({names, rows}); continue;
    }
    if (s.startsWith("_")) { tags[low] = t[i + 1]; i += 2; continue; }
    i++;
  }
  return {tags, loops};
}
const cifNum = s => parseFloat(String(s).replace(/\(\d+\)$/, ""));
function parseSymop(s) {  // "x+1/2,-y,z" -> {R: 3x3, t: [3]}
  const R = [], t = [];
  s.toLowerCase().replace(/\s/g, "").split(",").forEach(part => {
    const row = [0, 0, 0]; let c = 0;
    (part.match(/[+-]?[^+-]+/g) || []).forEach(term => {
      const m = term.match(/^([+-]?)([\d./]*)\*?([xyz])?$/);
      if (!m) throw new Error(`symmetry operation ${s}`);
      const sign = m[1] === "-" ? -1 : 1;
      const num = m[2] ? (m[2].includes("/") ? m[2].split("/").reduce((a, b) => +a / +b) : +m[2]) : 1;
      if (m[3]) row["xyz".indexOf(m[3])] += sign * num; else c += sign * num;
    });
    R.push(row); t.push(c);
  });
  if (R.length !== 3) throw new Error(`symmetry operation ${s}`);
  return {R, t};
}
function parseCif(text, name) {
  const {tags, loops} = parseCifBlock(text);
  const par = ["a", "b", "c", "alpha", "beta", "gamma"].map(k => cifNum(tags[`_cell_length_${k}`] ?? tags[`_cell_angle_${k}`]));
  if (par.some(x => !isFinite(x))) throw new Error("CIF: no cell parameters");
  const cell = cellFromPar(...par), M = inv3(cell);
  const site = loops.find(l => l.names.includes("_atom_site_fract_x"));
  if (!site) throw new Error("CIF: no _atom_site_fract_x loop");
  const col = k => site.names.indexOf(k);
  const opsLoop = loops.find(l => l.names.some(n => n === "_symmetry_equiv_pos_as_xyz" || n === "_space_group_symop_operation_xyz"));
  let ops = [{R: [[1,0,0],[0,1,0],[0,0,1]], t: [0, 0, 0]}], warning = null;
  if (opsLoop) {
    const c = opsLoop.names.findIndex(n => n === "_symmetry_equiv_pos_as_xyz" || n === "_space_group_symop_operation_xyz");
    ops = opsLoop.rows.map(r => parseSymop(r[c]));
  } else if (tags["_symmetry_equiv_pos_as_xyz"] || tags["_space_group_symop_operation_xyz"]) {
    ops = [parseSymop(tags["_symmetry_equiv_pos_as_xyz"] || tags["_space_group_symop_operation_xyz"])];
  } else warning = "CIF has no symmetry operations: sites used as given (P1)";
  const cx = col("_atom_site_fract_x"), cy = col("_atom_site_fract_y"), cz = col("_atom_site_fract_z");
  const ct = col("_atom_site_type_symbol"), cl = col("_atom_site_label"), co = col("_atom_site_occupancy");
  const symbols = [], xyz = [], sites = [], oxi = [], occ = [];
  let hasOxi = false, partial = false;
  site.rows.forEach(r => {
    const raw = ct >= 0 ? r[ct] : r[cl], e = cleanSymbol(raw);
    const q = String(raw).match(/(\d*)([+-])$/), ox = q ? (q[2] === "-" ? -1 : 1) * (q[1] ? +q[1] : 1) : 0;
    if (q) hasOxi = true;
    const o = co >= 0 ? cifNum(r[co]) : 1; if (o < 0.999) partial = true;
    const f0 = [cx, cy, cz].map(c => cifNum(r[c]));
    const mine = [];
    ops.forEach(op => {
      const f = op.R.map((row, i) => { const v = dot(row, f0) + op.t[i]; return v - Math.floor(v + 1e-9); });
      const p = toCart(f, cell);
      if (mine.some(q => { const d = sub(toFrac(q, M), f).map(x => x - Math.round(x)); return norm(toCart(d, cell)) < 0.05; })) return;
      mine.push(p);
    });
    mine.forEach(p => { symbols.push(e); xyz.push(p); sites.push(cl >= 0 ? r[cl] : e); oxi.push(ox); occ.push(isFinite(o) ? o : 1); });
  });
  const data = {site: sites};
  if (hasOxi) data.oxi_states = oxi;
  if (partial) data.occupancy = occ;
  const title = (tags["_chemical_formula_sum"] || "").replace(/\s+/g, "");
  // the file's space group, trusted only when its operations were used (a P1 listing of a Pnma crystal says 'P 1')
  const hm = (tags["_space_group_name_h-m_alt"] || tags["_symmetry_space_group_name_h-m"] || "").replace(/\s+/g, "");
  const num = tags["_space_group_it_number"] || tags["_symmetry_int_tables_number"];
  const spacegroup = ops.length > 1 && hm ? (num ? `${hm} (${num})` : hm) : null;
  return {frames: [{symbols, xyz, cell, periodic: true, data, labels: null, name, spacegroup,
                    warning: warning || (partial ? "partial occupancies: all sites drawn" : null), formula: title}]};
}
function parseGrd(text, name) {  // VESTA .grd (bvlain, gridmlip): comment, a b c α β γ, n0 n1 n2, values in C order; no atoms
  const L = text.split(/\r?\n/).filter(l => l.trim() && !l.trim().startsWith("#"));
  const par = nums(L[0]), n = nums(L[1]).slice(0, 3);
  if (par.length < 6 || n.length < 3 || par.some(x => !isFinite(x))) throw new Error(".grd: expected cell parameters and grid size");
  const cell = cellFromPar(...par.slice(0, 6)), total = n[0] * n[1] * n[2], values = new Float32Array(total);
  let p = 0;
  for (let k = 2; k < L.length && p < total; k++) for (const x of L[k].trim().split(/\s+/)) if (p < total) values[p++] = +x;
  if (p < total) throw new Error(`.grd: ${p} of ${total} grid values`);
  return {frames: [{symbols: [], xyz: [], cell, periodic: true, data: {}, labels: null, name, cellOnly: true}],
          volume: {n, values, cell, name, cellpar: par.slice(0, 6)}};
}
function cellPar(A) {  // a, b, c, alpha, beta, gamma
  const l = A.map(norm), ang = (i, j) => Math.acos(Math.max(-1, Math.min(1, dot(A[i], A[j]) / (l[i] * l[j])))) * 180 / Math.PI;
  return [l[0], l[1], l[2], ang(1, 2), ang(0, 2), ang(0, 1)];
}
const sameCellPar = (p, q, tol = 1e-3) => p.every((x, i) => Math.abs(x - q[i]) < (i < 3 ? tol * Math.max(1, x) : 0.05));

function parseOutcar(text, name) {  // VASP OUTCAR: every ionic step (cell, positions, forces, free energy)
  const L = text.split(/\r?\n/), species = [], potcar = [];
  let counts = null, cell = null;
  const frames = [];
  for (let k = 0; k < L.length; k++) {
    const line = L[k];
    if (line.includes("TITEL  =")) { species.push(cleanSymbol(line.trim().split(/\s+/)[3] || "X")); continue; }
    if (line.includes("POTCAR:")) { potcar.push(cleanSymbol(line.trim().split(/\s+/)[2] || "X")); continue; }
    if (!counts && line.includes("ions per type =")) { counts = nums(line.split("=")[1]); continue; }
    if (line.includes("direct lattice vectors")) { cell = [1, 2, 3].map(d => nums(L[k + d]).slice(0, 3)); k += 3; continue; }
    if (line.includes("POSITION") && line.includes("TOTAL-FORCE")) {
      if (!counts || !cell) throw new Error("OUTCAR: no 'ions per type' or lattice before the positions");
      const n = counts.reduce((a, b) => a + b, 0), xyz = [], force = [];
      for (let i = 0; i < n; i++) { const v = nums(L[k + 2 + i]); xyz.push(v.slice(0, 3)); force.push(v.slice(3, 6)); }
      frames.push({xyz, cell: cell.map(r => r.slice()), periodic: true, data: {force}, labels: null, energy: null});
      k += n + 2; continue;
    }
    if (line.includes("free  energy   TOTEN") && frames.length && frames[frames.length - 1].energy === null)
      frames[frames.length - 1].energy = parseFloat(line.split("=")[1]);
  }
  if (!frames.length) throw new Error("OUTCAR: no finished ionic step (no POSITION / TOTAL-FORCE block)");
  let types = species.length === counts.length ? species : [...new Set(potcar)];
  if (types.length !== counts.length) types = counts.map(() => "X");
  const symbols = []; counts.forEach((c, i) => { for (let j = 0; j < c; j++) symbols.push(types[i]); });
  frames.forEach((f, k) => {
    f.symbols = symbols;
    f.name = `${name}${frames.length > 1 ? " #" + k : ""}${f.energy !== null ? ` · E = ${f.energy.toFixed(4)} eV` : ""}`;
  });
  return {frames};
}

// ASE trajectory (.traj, binary "ULM" format of ase >= 3.11): header, offsets table, one JSON item per frame;
// arrays are {"ndarray": [shape, dtype, file offset]}; numbers and pbc sit in the first frame unless they change
function parseTraj(buf, name) {
  const u8 = new Uint8Array(buf), dv = new DataView(buf), txt = (a, b) => new TextDecoder().decode(u8.subarray(a, b));
  const magic = txt(0, 8);
  if (magic.startsWith("PickleTr")) throw new Error(".traj: an old pickle trajectory (ase < 3.11); convert it with ase first");
  if (magic !== "- of Ulm" && magic !== "AFFormat") throw new Error(".traj: not an ASE trajectory file");
  const tag = txt(8, 24).trim();
  if (tag && tag !== "ASE-Trajectory") throw new Error(`.traj: a '${tag}' file, not an ASE trajectory`);
  const i64 = off => Number(dv.getBigInt64(off, true));
  const nitems = i64(32), pos0 = i64(40);
  const DT = {float64: [8, (o, le) => dv.getFloat64(o, le)], float32: [4, (o, le) => dv.getFloat32(o, le)],
              int64: [8, (o, le) => Number(dv.getBigInt64(o, le))], int32: [4, (o, le) => dv.getInt32(o, le)],
              int16: [2, (o, le) => dv.getInt16(o, le)], int8: [1, o => dv.getInt8(o)], uint8: [1, o => dv.getUint8(o)],
              bool: [1, o => dv.getUint8(o) !== 0]};
  const array = (v, le) => {  // ndarray reference -> nested JS arrays
    const [shape, dtype, off] = v.ndarray, t = DT[dtype];
    if (!t) throw new Error(`.traj: arrays of ${dtype} are not supported`);
    const n = shape.reduce((a, b) => a * b, 1), flat = new Array(n);
    for (let i = 0; i < n; i++) flat[i] = t[1](off + i * t[0], le);
    if (shape.length < 2) return flat;
    const w = shape.slice(1).reduce((a, b) => a * b, 1), rows = [];
    for (let i = 0; i < shape[0]; i++) rows.push(flat.slice(i * w, (i + 1) * w));
    return rows;
  };
  const get = (d, key, le) => (d[key + "."] && d[key + "."].ndarray ? array(d[key + "."], le) : d[key] ?? d[key + "."]);
  const frames = [];
  let numbers = null, pbc = null;
  for (let k = 0; k < nitems; k++) {
    const off = i64(pos0 + 8 * k), size = i64(off);
    const json = txt(off + 8, off + 8 + size).replace(/\bNaN\b|-?\bInfinity\b/g, "null");
    const d = JSON.parse(json), le = d._little_endian !== false;
    numbers = get(d, "numbers", le) || numbers;
    pbc = d.pbc ?? pbc;
    if (!numbers) throw new Error(".traj: no atomic numbers in the first frame");
    const xyz = get(d, "positions", le), cell = get(d, "cell", le);
    const periodic = !!(pbc && pbc.some(Boolean)) && Math.abs(det3(cell)) > 1e-6;
    const data = {};
    for (const key of ["magmoms", "charges"]) { const a = get(d, key, le); if (a) data[key] = a; }
    let energy = null;
    const c = d["calculator."] || d.calculator;
    if (c) {
      const f = get(c, "forces", le); if (f) data.force = f;
      const e = c.free_energy ?? c.energy; if (typeof e === "number") energy = e;
    }
    frames.push({symbols: numbers.map(z => TABLES.symbols[z] || "X"), xyz, cell: periodic ? cell : null, periodic, data, labels: null, energy});
  }
  if (!frames.length) throw new Error(".traj: no frames");
  frames.forEach((f, k) => {
    f.name = `${name}${frames.length > 1 ? " #" + k : ""}${f.energy !== null ? ` · E = ${f.energy.toFixed(4)} eV` : ""}`;
  });
  return {frames};
}
const isBinary = name => /\.traj$/i.test(name);  // read as bytes, not text

function kindOf(name, text) {
  const base = name.split(/[\\/]/).pop(), up = base.toUpperCase(), low = base.toLowerCase();
  if (low.endsWith(".cif") || low.endsWith(".mcif")) return "cif";
  if (low.endsWith(".cube") || low.endsWith(".cub")) return "cube";
  if (low.endsWith(".grd")) return "grd";
  if (low.endsWith(".traj")) return "traj";
  if (low.endsWith(".xyz") || low.endsWith(".extxyz")) return "xyz";
  if (/^XDATCAR/.test(up) || up.endsWith(".XDATCAR")) return "xdatcar";
  if (up.includes("OUTCAR")) return "outcar";
  if (/^(CHGCAR|CHG|LOCPOT|PARCHG|AECCAR\d?|ELFCAR)/.test(up) || /\.(CHGCAR|LOCPOT)$/.test(up)) return "chgcar";
  if (/^(POSCAR|CONTCAR)/.test(up) || /\.(VASP|POSCAR|CONTCAR)$/.test(up)) return "poscar";
  if (/^\s*data_/m.test(text.slice(0, 5000))) return "cif";
  if (/^\s*\d+\s*$/.test(text.split(/\r?\n/, 1)[0])) return "xyz";
  return "poscar";
}
const PARSERS = {cif: parseCif, poscar: parsePoscar, xdatcar: parseXdatcar, xyz: parseXyz, cube: parseCube, chgcar: parseChgcar, grd: parseGrd, outcar: parseOutcar, traj: parseTraj};
function parseFile(name, content) {  // content: text, or an ArrayBuffer for binary formats (isBinary)
  const kind = kindOf(name, typeof content === "string" ? content : "");
  let out;
  try { out = PARSERS[kind](content, name.split(/[\\/]/).pop()); }
  catch (e) { if (e instanceof TypeError || e instanceof RangeError) throw new Error(`could not be read as ${kind}`); throw e; }
  out.kind = kind;
  // a file of another kind (a web page, say) can parse to nothing: say so rather than draw an empty, NaN cell
  const F = out.frames || [];
  if (!F.length || F.every(f => !f.symbols.length && !f.cellOnly)) throw new Error(`no atoms found (read as ${kind})`);
  if (F.some(f => f.periodic && !f.cell.flat().every(Number.isFinite))) throw new Error(`the cell could not be read (read as ${kind})`);
  return out;
}

// ------------------------------------------------------------------ bonds, polyhedra
const NONMETALS = new Set("H He B C N O F Ne Si P S Cl Ar As Se Br Kr Te I Xe At Rn".split(" "));
function rulesFor(symbols, centers, custom) {
  const present = new Set(symbols), rules = {};
  for (const c of present) {
    const r = TABLES.rules[c]; if (!r) continue;
    const kept = {}; for (const l in r) if (present.has(l)) kept[l] = r[l];
    if (Object.keys(kept).length) rules[c] = kept;
  }
  // covalent-radius bonds (1.15 × sum) for pairs no VESTA rule covers, in either direction, when one of the two is
  // a nonmetal (C-H, C-C, O-H in molecules) or neither element has any rule (Al-Al in a metal); never between two
  // different polyhedron centers (no Li-P, Fe-P in LiFePO4) or metal-metal pairs next to rule-covered elements
  const covered = new Set(Object.keys(rules)); Object.values(rules).forEach(r => Object.keys(r).forEach(l => covered.add(l)));
  const els = [...present].sort(), fallback = {};
  els.forEach((a, i) => els.slice(i).forEach(b => {
    if ((rules[a] && rules[a][b] !== undefined) || (rules[b] && rules[b][a] !== undefined)) return;
    if (a !== b && rules[a] && rules[b]) return;
    if (!(NONMETALS.has(a) || NONMETALS.has(b) || (!covered.has(a) && !covered.has(b)))) return;
    fallback[a + "|" + b] = 1.15 * ((TABLES.covalent[a] ?? 1.5) + (TABLES.covalent[b] ?? 1.5));
  }));
  // user bond types: {"A|B": d}; d <= 0 switches the type off. They change an existing type (in either order; a VESTA
  // type keeps forming polyhedra with its new length) or add a new one, which draws bonds only (no polyhedra)
  for (const key in custom || {}) {
    const [a, b] = key.split("|"), d = custom[key];
    if (!present.has(a) || !present.has(b)) continue;
    const fk = a < b ? a + "|" + b : b + "|" + a;
    if (rules[a] && rules[a][b] !== undefined) { if (d > 0) rules[a][b] = d; else delete rules[a][b]; }
    else if (rules[b] && rules[b][a] !== undefined) { if (d > 0) rules[b][a] = d; else delete rules[b][a]; }
    else if (fallback[fk] !== undefined) { if (d > 0) fallback[fk] = d; else delete fallback[fk]; }
    else if (d > 0) fallback[fk] = d;
  }
  for (const c in rules) if (!Object.keys(rules[c]).length) delete rules[c];
  const polyCenters = new Set(Object.keys(rules).filter(c => !centers || centers.includes(c)));
  let cutoff = 0;
  Object.values(rules).forEach(r => Object.values(r).forEach(d => { cutoff = Math.max(cutoff, d); }));
  Object.values(fallback).forEach(d => { cutoff = Math.max(cutoff, d); });
  return {rules, fallback, polyCenters, cutoff};
}
function bondTypes(symbols, custom) {  // [{a, b, d, on, kind}] for the bond panel: defaults, changes, additions
  const base = rulesFor(symbols, null), out = [], seen = new Set();
  const user = key => (custom && custom[key] !== undefined ? custom[key] : undefined);
  const push = (a, b, d0, kind) => {
    const u = user(a + "|" + b) ?? user(b + "|" + a);
    seen.add(a + "|" + b); seen.add(b + "|" + a);
    out.push({a, b, d: u !== undefined ? Math.abs(u) : d0, on: u === undefined || u > 0, kind: u !== undefined ? kind + "*" : kind});
  };
  Object.keys(base.rules).sort().forEach(a => Object.keys(base.rules[a]).sort().forEach(b => push(a, b, base.rules[a][b], "vesta")));
  Object.keys(base.fallback).sort().forEach(k => { const [a, b] = k.split("|"); push(a, b, base.fallback[k], "covalent"); });
  const present = new Set(symbols);
  for (const key in custom || {}) {
    const [a, b] = key.split("|");
    if (!seen.has(key) && present.has(a) && present.has(b)) { seen.add(key); out.push({a, b, d: Math.abs(custom[key]), on: custom[key] > 0, kind: "added"}); }
  }
  return out;
}
function neighbors(frame, cutoff) {  // [{i, j, S, d, p}] with p = xyz[j] + S·cell, 0 < d <= cutoff
  const X = frame.xyz, n = X.length, out = [];
  if (!isPeriodic(frame)) {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) {
      const d = norm(sub(X[j], X[i])); if (d <= cutoff) out.push({i, j, S: [0, 0, 0], d, p: X[j]});
    }
    return out;
  }
  // cell list: atoms wrapped into the cell (x = w + nw·A) and binned; bins at least ~cutoff wide along each
  // plane normal; each atom looks at the bins within reach, periodic images included (shift = whole cells)
  const A = frame.cell, M = inv3(A), V = Math.abs(det3(A));
  const dpl = [0, 1, 2].map(k => V / norm(cross(A[(k + 1) % 3], A[(k + 2) % 3])));  // interplanar spacings
  const nb = dpl.map(d => Math.max(1, Math.floor(d / cutoff)));
  const reach = dpl.map((d, k) => Math.ceil(cutoff / (d / nb[k])));
  const nw = [], bin = [], bins = new Map();
  X.forEach((r, i) => {
    const f = toFrac(r, M), w = f.map(Math.floor), g = f.map((x, k) => Math.min(nb[k] - 1, Math.floor((x - w[k]) * nb[k])));
    nw.push(w); bin.push(g);
    const key = (g[0] * nb[1] + g[1]) * nb[2] + g[2];
    if (!bins.has(key)) bins.set(key, []); bins.get(key).push(i);
  });
  const cellShift = S => toCart(S, A);
  for (let i = 0; i < n; i++) {
    const g = bin[i];
    for (let a = -reach[0]; a <= reach[0]; a++) for (let b = -reach[1]; b <= reach[1]; b++) for (let c = -reach[2]; c <= reach[2]; c++) {
      const t = [g[0] + a, g[1] + b, g[2] + c];
      const sh = t.map((x, k) => Math.floor(x / nb[k])), tb = t.map((x, k) => x - sh[k] * nb[k]);
      const list = bins.get((tb[0] * nb[1] + tb[1]) * nb[2] + tb[2]); if (!list) continue;
      for (const j of list) {
        const S = [sh[0] - nw[j][0] + nw[i][0], sh[1] - nw[j][1] + nw[i][1], sh[2] - nw[j][2] + nw[i][2]];
        const p = add(X[j], cellShift(S)), d = norm(sub(p, X[i]));
        if (d > 0 && d <= cutoff) out.push({i, j, S, d, p});
      }
    }
  }
  return out;
}
function hull(P) {  // convex hull of a few points: {vertices, normals, faces, edges, volume} or null if flat
  const n = P.length; if (n < 4) return null;
  const c = scale(P.reduce(add, [0, 0, 0]), 1 / n), eps = 1e-4, planes = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
    let nv = cross(sub(P[j], P[i]), sub(P[k], P[i])); const L = norm(nv); if (L < 1e-8) continue;
    nv = scale(nv, 1 / L); let off = dot(nv, P[i]);
    if (dot(nv, c) - off > 0) { nv = scale(nv, -1); off = -off; }
    if (P.some(p => dot(nv, p) - off > eps)) continue;
    if (planes.some(q => dot(q.n, nv) > 1 - 1e-6 && Math.abs(q.off - off) < eps)) continue;
    planes.push({n: nv, off});
  }
  if (planes.length < 4) return null;
  const vertices = [], normals = [], faces = [], edges = [], seen = new Set();
  let volume = 0;
  planes.forEach(({n: nv, off}) => {
    const idx = P.map((p, i) => i).filter(i => Math.abs(dot(nv, P[i]) - off) < eps);
    const cen = scale(idx.reduce((s, i) => add(s, P[i]), [0, 0, 0]), 1 / idx.length);
    const u = (() => { const v = sub(P[idx[0]], cen); return scale(v, 1 / norm(v)); })(), w = cross(nv, u);
    idx.sort((a, b) => { const va = sub(P[a], cen), vb = sub(P[b], cen);
      return Math.atan2(dot(va, w), dot(va, u)) - Math.atan2(dot(vb, w), dot(vb, u)); });
    for (let m = 1; m + 1 < idx.length; m++) {
      const tri = [P[idx[0]], P[idx[m]], P[idx[m + 1]]], k = vertices.length;
      tri.forEach(v => { vertices.push(v); normals.push(nv); });
      faces.push(k, k + 1, k + 2);
      volume += Math.abs(dot(sub(tri[0], c), cross(sub(tri[1], c), sub(tri[2], c)))) / 6;
    }
    idx.forEach((a, m) => { const b = idx[(m + 1) % idx.length], key = Math.min(a, b) + "," + Math.max(a, b);
      if (!seen.has(key)) { seen.add(key); edges.push([P[a], P[b]]); } });
  });
  return {vertices, normals, faces, edges, volume};
}
const rkey = p => p.map(x => x.toFixed(3)).join(",");
function scene(frame, centers, custom, minCN = 4) {  // bonds [i, j, p], polyhedra, image atoms {idx, xyz, frac}
  const sym = frame.symbols, X = frame.xyz, periodic = isPeriodic(frame);
  const R = rulesFor(sym, centers, custom), out = {bonds: [], polyhedra: [], images: []};
  if (!R.cutoff || !X.length) return out;
  const M = periodic ? inv3(frame.cell) : null, seenB = new Set(), seenI = new Set(), byCenter = new Map();
  for (const nb of neighbors(frame, R.cutoff)) {
    const a = sym[nb.i], b = sym[nb.j], rule = R.rules[a] && R.rules[a][b];
    const ok = rule !== undefined ? nb.d <= rule : nb.d <= (R.fallback[a < b ? a + "|" + b : b + "|" + a] || 0);
    if (!ok) continue;
    const k1 = rkey(X[nb.i]), k2 = rkey(nb.p), key = k1 < k2 ? k1 + "/" + k2 : k2 + "/" + k1;
    if (!seenB.has(key)) { seenB.add(key); out.bonds.push([nb.i, nb.j, nb.p]); }
    if (rule !== undefined && R.polyCenters.has(a)) { if (!byCenter.has(nb.i)) byCenter.set(nb.i, []); byCenter.get(nb.i).push(nb); }
    if (periodic) {
      const f = toFrac(nb.p, M);
      if (f.some(x => x < -1e-3 || x > 1 - 1e-3) && !seenI.has(k2)) { seenI.add(k2); out.images.push({idx: nb.j, xyz: nb.p, frac: f}); }
    }
  }
  [...byCenter.keys()].sort((a, b) => a - b).forEach(c => {
    const nbs = byCenter.get(c).slice().sort((p, q) => p.d - q.d);
    if (nbs.length < minCN) return;
    const h = hull(nbs.map(nb => nb.p)); if (!h) return;
    const d = nbs.map(nb => nb.d), mean = d.reduce((s, x) => s + x, 0) / d.length;
    out.polyhedra.push({idx: c, elem: sym[c], color: (TABLES.colors[sym[c]] || ["#999999"])[0], ...h,
      ligands: nbs.map(nb => ({idx: nb.j, image: nb.S, d: nb.d})), d_mean: mean,
      distortion: d.reduce((s, x) => s + Math.abs(x - mean), 0) / d.length / mean});  // Baur distortion index
  });
  return out;
}

// ------------------------------------------------------------------ grids
function decodeGrid(b64) {
  const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
  const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new Float32Array(u.buffer);
}
function resample(values, n, limit) {  // periodic trilinear resampling to at most `limit` points
  const total = n[0] * n[1] * n[2]; if (!limit || total <= limit) return {values, n};
  const f = Math.cbrt(limit / total), m = n.map(x => Math.max(2, Math.round(x * f)));
  const out = new Float32Array(m[0] * m[1] * m[2]), at = (i, j, k) => values[((i % n[0]) * n[1] + (j % n[1])) * n[2] + (k % n[2])];
  let p = 0;
  for (let i = 0; i < m[0]; i++) { const x = i * n[0] / m[0], i0 = Math.floor(x), fx = x - i0;
    for (let j = 0; j < m[1]; j++) { const y = j * n[1] / m[1], j0 = Math.floor(y), fy = y - j0;
      for (let k = 0; k < m[2]; k++) { const z = k * n[2] / m[2], k0 = Math.floor(z), fz = z - k0;
        let v = 0;
        for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let c = 0; c < 2; c++)
          v += (a ? fx : 1 - fx) * (b ? fy : 1 - fy) * (c ? fz : 1 - fz) * at(i0 + a, j0 + b, k0 + c);
        out[p++] = v; } } }
  return {values: out, n: m};
}
function gridStats(values) {  // min, mean, 99.5th percentile (non-finite values replaced by the finite max)
  let lo = Infinity, hi = -Infinity, s = 0, c = 0;
  for (const v of values) if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); s += v; c++; }
  if (!c) throw new Error("volume has no finite values");
  for (let i = 0; i < values.length; i++) if (!isFinite(values[i])) values[i] = hi;
  const step = Math.max(1, Math.floor(values.length / 20000)), sample = [];
  for (let i = 0; i < values.length; i += step) sample.push(values[i]);
  sample.sort((a, b) => a - b);
  return {min: lo, mean: s / c, max: sample[Math.min(sample.length - 1, Math.floor(0.995 * sample.length))]};
}

// ------------------------------------------------------------------ symmetry and summary
// Crystals: operations (W, t) in a reduced cell that keep the metric and map every atom onto an atom of the same
// element (within `tol` Å); the rotation parts give the point group (Hermann-Mauguin) from their (det, trace) counts.
// Molecules: orthogonal maps of the centered atoms found from atom triples; the Schoenflies symbol from the axes.
function reduceCell(A) {  // Gauss-Lagrange reduction of the rows (keeps the lattice)
  const B = A.map(r => r.slice());
  for (let it = 0; it < 100; it++) {
    let changed = false;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (i !== j) {
      const k = Math.round(dot(B[i], B[j]) / dot(B[j], B[j]));
      if (k) { B[i] = sub(B[i], scale(B[j], k)); changed = true; }
    }
    if (!changed) break;
  }
  return det3(B) < 0 ? [B[0], B[1], scale(B[2], -1)] : B;
}
const matVec = (W, f) => [dot(W[0], f), dot(W[1], f), dot(W[2], f)];
const wrap = x => x - Math.floor(x);
function crystalRotations(frame, tol = 0.02) {  // unique integer rotation matrices W of the symmetry operations
  const B = reduceCell(frame.cell), M = inv3(B), n = frame.xyz.length;
  if (!n || n > 1500) return null;
  const G = [0, 1, 2].map(i => [0, 1, 2].map(j => dot(B[i], B[j])));
  const F = frame.xyz.map(r => toFrac(r, M).map(wrap));
  const bySym = {}; frame.symbols.forEach((s, i) => (bySym[s] = bySym[s] || []).push(i));
  const rare = Object.values(bySym).sort((p, q) => p.length - q.length)[0], a0 = rare[0];
  const gtol = 0.02 * Math.max(G[0][0], G[1][1], G[2][2]);
  const close = (f, g) => { const d = sub(f, g).map(x => x - Math.round(x)); return norm(toCart(d, B)) < tol; };
  const out = [], v = [-1, 0, 1];
  for (const a of v) for (const b of v) for (const c of v) for (const d of v) for (const e of v) for (const f of v)
  for (const g of v) for (const h of v) for (const k of v) {
    const W = [[a, b, c], [d, e, f], [g, h, k]], dt = det3(W);
    if (Math.abs(dt) !== 1) continue;
    let ok = true;   // metric kept: W G W^T = G (f' = W f, rows of B)
    for (let i = 0; i < 3 && ok; i++) for (let j = 0; j < 3 && ok; j++) {
      let x = 0; for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) x += W[i][p] * G[p][q] * W[j][q];
      if (Math.abs(x - G[i][j]) > gtol) ok = false;
    }
    if (!ok) continue;
    // with f' = W^T-convention: map atoms as f' = W' f where W' acts on columns; use W^T for fractional columns
    const Wc = [[a, d, g], [b, e, h], [c, f, k]];
    const w0 = matVec(Wc, F[a0]);
    for (const b0 of rare) {
      const t = sub(F[b0], w0);
      const all = F.every((fi, i) => { const y = add(matVec(Wc, fi), t); return bySym[frame.symbols[i]].some(j => close(y, F[j])); });
      if (all) { out.push(Wc); break; }
    }
  }
  return out;
}
const PG_TABLE = {  // counts of (det, trace) types -> Hermann-Mauguin point group
  "1": "1", "1,-1": "-1", "1,2": "2", "1,m": "m", "1,2,m,-1": "2/m", "1,2x3": "222", "1,2,mx2": "mm2",
  "1,2x3,mx3,-1": "mmm", "1,2,4x2": "4", "1,2,-4x2": "-4", "1,2,4x2,-1,m,-4x2": "4/m", "1,2x5,4x2": "422",
  "1,2,4x2,mx4": "4mm", "1,2x3,-4x2,mx2": "-42m", "1,2x5,4x2,-1,mx5,-4x2": "4/mmm", "1,3x2": "3",
  "1,3x2,-1,-3x2": "-3", "1,3x2,2x3": "32", "1,3x2,mx3": "3m", "1,3x2,2x3,-1,-3x2,mx3": "-3m",
  "1,2,3x2,6x2": "6", "1,3x2,m,-6x2": "-6", "1,2,3x2,6x2,-1,m,-3x2,-6x2": "6/m", "1,2x7,3x2,6x2": "622",
  "1,2,3x2,6x2,mx6": "6mm", "1,3x2,2x3,mx4,-6x2": "-6m2", "1,2x7,3x2,6x2,-1,mx7,-3x2,-6x2": "6/mmm",
  "1,2x3,3x8": "23", "1,2x3,3x8,-1,mx3,-3x8": "m-3", "1,2x9,3x8,4x6": "432", "1,2x3,3x8,-4x6,mx6": "-43m",
  "1,2x9,3x8,4x6,-1,mx9,-3x8,-4x6": "m-3m"};
const PG_ORDER = ["1", "2", "3", "4", "6", "-1", "m", "-3", "-4", "-6"];
const pgKey = parts => parts.slice().sort((p, q) => PG_ORDER.indexOf(p.split("x")[0]) - PG_ORDER.indexOf(q.split("x")[0])).join(",");
for (const k of Object.keys(PG_TABLE)) { const v = PG_TABLE[k]; delete PG_TABLE[k]; PG_TABLE[pgKey(k.split(","))] = v; }
const PG_SYSTEM = {"1": "triclinic", "-1": "triclinic", "2": "monoclinic", "m": "monoclinic", "2/m": "monoclinic",
  "222": "orthorhombic", "mm2": "orthorhombic", "mmm": "orthorhombic", "3": "trigonal", "-3": "trigonal", "32": "trigonal",
  "3m": "trigonal", "-3m": "trigonal", "23": "cubic", "m-3": "cubic", "432": "cubic", "-43m": "cubic", "m-3m": "cubic"};
function opType(det, tr) {
  const t = Math.round(tr);
  if (det > 0) return {3: "1", "-1": "2", 0: "3", 1: "4", 2: "6"}[t];
  return {"-3": "-1", 1: "m", 0: "-3", "-1": "-4", "-2": "-6"}[t];
}
function crystalPointGroup(frame, tol) {
  const Ws = crystalRotations(frame, tol);
  if (!Ws) return null;
  const c = {}; Ws.forEach(W => { const k = opType(det3(W), W[0][0] + W[1][1] + W[2][2]); c[k] = (c[k] || 0) + 1; });
  const key = PG_ORDER.filter(k => c[k]).map(k => c[k] > 1 ? `${k}x${c[k]}` : k).join(",");
  const pg = PG_TABLE[key] || null;
  return pg && {pg, order: Ws.length, system: PG_SYSTEM[pg] || (pg.includes("6") ? "hexagonal" : "tetragonal")};
}
function solve3(P, Q) {  // R with R·P[:,k] = Q[:,k] for three column vectors given as arrays of rows
  const Pc = [[P[0][0], P[1][0], P[2][0]], [P[0][1], P[1][1], P[2][1]], [P[0][2], P[1][2], P[2][2]]];  // columns as rows
  const Qc = [[Q[0][0], Q[1][0], Q[2][0]], [Q[0][1], Q[1][1], Q[2][1]], [Q[0][2], Q[1][2], Q[2][2]]];
  if (Math.abs(det3(Pc)) < 1e-6) return null;
  const Pi = inv3(Pc);  // Pi rows: Pc^{-1}
  // R = Qc · Pc^{-1} with Pc, Qc having the vectors as columns
  return [0, 1, 2].map(i => [0, 1, 2].map(j => Qc[i][0] * Pi[0][j] + Qc[i][1] * Pi[1][j] + Qc[i][2] * Pi[2][j]));
}
function moleculeOps(frame, tol = 0.05) {
  const n = frame.xyz.length;  // the centroid is fixed by every symmetry operation
  const c = scale(frame.xyz.reduce(add, [0, 0, 0]), 1 / n);
  const X = frame.xyz.map(r => sub(r, c)), S = frame.symbols;
  const bySym = {}; S.forEach((s, i) => (bySym[s] = bySym[s] || []).push(i));
  const maps = R => X.every((p, i) => { const y = matVec(R, p); return bySym[S[i]].some(j => norm(sub(y, X[j])) < tol); });
  const off = X.map((p, i) => i).filter(i => norm(X[i]) > tol);
  if (!off.length) return {linear: false, atom: true, ops: []};
  const p1 = off.reduce((b, i) => (bySym[S[i]].length < bySym[S[b]].length ? i : b), off[0]);
  const lin = off.every(i => norm(cross(X[i], X[p1])) < tol * norm(X[p1]));
  if (lin) return {linear: true, inversion: maps([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])};
  const cand = off.filter(i => S[i] === S[p1] && norm(cross(X[i], X[p1])) > tol * norm(X[p1]));
  const pool = cand.length ? cand : off.filter(i => norm(cross(X[i], X[p1])) > tol * norm(X[p1]));
  const p2 = pool.reduce((b, i) => (norm(sub(X[i], X[p1])) < norm(sub(X[b], X[p1])) ? i : b), pool[0]);
  const v3 = cross(X[p1], X[p2]);
  const p3 = off.find(i => Math.abs(dot(X[i], v3)) > tol * norm(v3));  // undefined: planar molecule
  const same = (i, j) => S[i] === S[j] && Math.abs(norm(X[i]) - norm(X[j])) < tol;
  const dd = (a, b) => norm(sub(X[a], X[b]));
  const ops = [], seen = R => ops.some(Q => Q.every((r, i) => r.every((x, j) => Math.abs(x - R[i][j]) < 1e-3)));
  const tryR = R => {
    if (!R) return;
    const RtR = [0, 1, 2].map(i => [0, 1, 2].map(j => R[0][i] * R[0][j] + R[1][i] * R[1][j] + R[2][i] * R[2][j]));
    if (RtR.some((r, i) => r.some((x, j) => Math.abs(x - (i === j ? 1 : 0)) > 1e-2))) return;
    if (!seen(R) && maps(R)) ops.push(R);
  };
  for (const q1 of bySym[S[p1]]) if (same(p1, q1))
    for (const q2 of bySym[S[p2]]) if (same(p2, q2) && Math.abs(dd(q1, q2) - dd(p1, p2)) < tol) {
      if (p3 !== undefined) {
        for (const q3 of bySym[S[p3]]) if (same(p3, q3) && Math.abs(dd(q1, q3) - dd(p1, p3)) < tol && Math.abs(dd(q2, q3) - dd(p2, p3)) < tol)
          tryR(solve3([X[p1], X[p2], X[p3]], [X[q1], X[q2], X[q3]]));
      } else {
        const w = cross(X[q1], X[q2]);
        tryR(solve3([X[p1], X[p2], v3], [X[q1], X[q2], w]));
        tryR(solve3([X[p1], X[p2], v3], [X[q1], X[q2], scale(w, -1)]));
      }
    }
  return {linear: false, ops};
}
function axisOf(R) {  // rotation axis and angle of a proper rotation
  const tr = R[0][0] + R[1][1] + R[2][2], th = Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2)));
  let ax = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]];
  if (norm(ax) < 1e-3) {  // angle 0 or 180 degrees: axis from R + I
    const P = [0, 1, 2].map(i => [0, 1, 2].map(j => R[i][j] + (i === j ? 1 : 0)));
    ax = P.reduce((b, r) => (norm(r) > norm(b) ? r : b), [0, 0, 0]);
  }
  const l = norm(ax);
  return {axis: l > 1e-9 ? scale(ax, 1 / l) : [0, 0, 1], angle: th};
}
function schoenflies(ops) {
  const proper = [], improper = [];
  ops.forEach(R => { const d = det3(R); (d > 0 ? proper : improper).push(R); });
  const neg = R => R.map(r => r.map(x => -x));
  const axes = [];  // {axis, n}
  proper.forEach(R => {
    const {axis, angle} = axisOf(R); if (angle < 1e-2) return;
    const n = Math.round(2 * Math.PI / angle);
    const a = axes.find(x => Math.abs(Math.abs(dot(x.axis, axis)) - 1) < 1e-3);
    if (a) a.n = Math.max(a.n, n); else axes.push({axis, n});
  });
  const inversion = improper.some(R => Math.abs(R[0][0] + R[1][1] + R[2][2] + 3) < 1e-3);
  const mirrors = improper.filter(R => Math.abs(R[0][0] + R[1][1] + R[2][2] - 1) < 1e-3).map(R => axisOf(neg(R)).axis);
  const high = axes.filter(a => a.n >= 3);
  if (high.length >= 2) {
    if (axes.some(a => a.n === 5)) return inversion ? "Ih" : "I";
    if (axes.some(a => a.n === 4)) return inversion ? "Oh" : "O";
    return inversion ? "Th" : mirrors.length ? "Td" : "T";
  }
  if (!axes.length) return mirrors.length ? "Cs" : inversion ? "Ci" : "C1";
  const nmax = Math.max(...axes.map(a => a.n)), z = axes.find(a => a.n === nmax).axis;
  const perpC2 = axes.filter(a => a.n === 2 && Math.abs(dot(a.axis, z)) < 1e-3).length;
  const sh = mirrors.some(nv => Math.abs(Math.abs(dot(nv, z)) - 1) < 1e-3);
  const sv = mirrors.filter(nv => Math.abs(dot(nv, z)) < 1e-3).length;
  if (perpC2 >= nmax) return sh ? `D${nmax}h` : sv >= nmax ? `D${nmax}d` : `D${nmax}`;
  if (sh) return `C${nmax}h`;
  if (sv >= nmax) return `C${nmax}v`;
  if (improper.length) return `S${2 * nmax}`;
  return `C${nmax}`;
}
function moleculePointGroup(frame, tol) {
  if (frame.xyz.length > 400) return null;
  const r = moleculeOps(frame, tol);
  if (r.atom) return "Kh";
  if (r.linear) return r.inversion ? "D∞h" : "C∞v";
  return schoenflies(r.ops);
}
function gcd(a, b) { return b ? gcd(b, a % b) : a; }
function summary(frame) {  // composition, lattice, volume, Z, symmetry
  const counts = {}; frame.symbols.forEach(s => counts[s] = (counts[s] || 0) + 1);
  const n = frame.symbols.length, out = {natoms: n, counts};
  const vals = Object.values(counts);
  out.Z = vals.length ? vals.reduce(gcd) : 0;
  out.reduced = Object.entries(counts).map(([e, k]) => e + (k / out.Z > 1 ? k / out.Z : "")).join("");
  if (isPeriodic(frame)) {
    out.cellpar = cellPar(frame.cell); out.volume = Math.abs(det3(frame.cell));
    out.spacegroup = frame.spacegroup || null;
    const pg = n ? crystalPointGroup(frame) : null;
    if (pg) { out.pointgroup = pg.pg; out.system = pg.system; }
  } else if (n) {
    out.pointgroup = moleculePointGroup(frame);
    const lo = [0, 1, 2].map(k => Math.min(...frame.xyz.map(p => p[k]))), hi = [0, 1, 2].map(k => Math.max(...frame.xyz.map(p => p[k])));
    out.extent = hi.map((x, k) => x - lo[k]);
  }
  return out;
}

// ------------------------------------------------------------------ writers: the current frame as CIF (P1), POSCAR or extended xyz
const f6 = x => (Math.abs(x) < 5e-10 ? 0 : x).toFixed(6);  // no "-0.000000"
function fracOf(frame) { const M = inv3(frame.cell); return frame.xyz.map(r => toFrac(r, M)); }
function writeCif(frame, name = "chucha") {
  const [a, b, c, al, be, ga] = cellPar(frame.cell), F = fracOf(frame), count = {};
  const rows = frame.symbols.map((s, i) => { count[s] = (count[s] || 0) + 1; return `  ${s}${count[s] - 1} ${s} ${F[i].map(f6).join(" ")} 1`; });
  const block = String(name).replace(/[^\w.-]+/g, "_") || "chucha";
  return [`data_${block}`, "_symmetry_space_group_name_H-M   'P 1'", "_symmetry_Int_Tables_number   1",
    `_cell_length_a   ${f6(a)}`, `_cell_length_b   ${f6(b)}`, `_cell_length_c   ${f6(c)}`,
    `_cell_angle_alpha   ${f6(al)}`, `_cell_angle_beta   ${f6(be)}`, `_cell_angle_gamma   ${f6(ga)}`,
    "loop_", " _symmetry_equiv_pos_as_xyz", "  'x, y, z'",
    "loop_", " _atom_site_label", " _atom_site_type_symbol", " _atom_site_fract_x", " _atom_site_fract_y", " _atom_site_fract_z",
    " _atom_site_occupancy", ...rows, ""].join("\n");
}
function writePoscar(frame, comment = "chucha") {  // atoms grouped by element (first-appearance order), direct coordinates
  const order = [...new Set(frame.symbols)], F = fracOf(frame);
  const idx = order.flatMap(e => frame.symbols.map((s, i) => (s === e ? i : -1)).filter(i => i >= 0));
  return [comment, "1.0", ...frame.cell.map(r => "  " + r.map(f6).join("  ")),
    "  " + order.join("  "), "  " + order.map(e => frame.symbols.filter(s => s === e).length).join("  "),
    "Direct", ...idx.map(i => "  " + F[i].map(f6).join("  ")), ""].join("\n");
}
function writeXyz(frame, comment = "") {  // extended xyz: Lattice and pbc for crystals
  const per = isPeriodic(frame);
  const head = per ? `Lattice="${frame.cell.flat().map(f6).join(" ")}" Properties=species:S:1:pos:R:3 pbc="T T T"` : comment;
  return [String(frame.symbols.length), head, ...frame.symbols.map((s, i) => `${s} ${frame.xyz[i].map(f6).join(" ")}`), ""].join("\n");
}

const api = {setTables, parseFile, kindOf, parsePoscar, parseXdatcar, parseOutcar, parseTraj, isBinary, parseXyz, parseCube, parseChgcar, parseCif, parseSymop,
             writeCif, writePoscar, writeXyz, channelValues, channelList, CHANNEL_NAMES, rulesFor, bondTypes, neighbors, hull, scene, parseGrd, cellPar, sameCellPar, summary, crystalPointGroup, moleculePointGroup, reduceCell, decodeGrid, resample, gridStats, inv3, toFrac, toCart, det3, cellFromPar, isPeriodic, rkey};
root.Chucha = api;
if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
