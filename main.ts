// Sokoban solver: push-optimal per-room A* with Hungarian heuristic and deadlock pruning.
//
// Why rooms: a generated level is a 1-wide hallway with 1-4 rooms off it, each room
// holding 3-4 boxes and as many goals, opening onto the hallway through one door.
// A box pushed out of its room can never be pushed back, so every box stays in its
// room and the level optimum P* is the sum of the rooms' optima. The solver
// decomposes the level into rooms, solves each room independently with push-based
// A*, then stitches the rooms together with walking moves through the hallway.
//
// Per room, the search is A* over pushes: a state is the sorted box cells plus the
// player's reachable region (normalised to its lowest cell). The heuristic is the
// minimum-cost assignment of boxes to goals (Hungarian, by enumerating the at most
// 24 permutations for 4 boxes) over BFS distances on the room floor, which is
// admissible, so A* returns a push-optimal room solution. Pruning: static dead
// squares (reverse-pull reachability from the goals), 2x2 blocks of boxes/walls
// with no goal inside, and a transposition table. A greedy best-first search is
// the fallback if A* exhausts its expansion budget.
//
// Deterministic: no clock, no randomness, no environment reads; the search budget
// is counted in expansions, never in wall time, so repeated runs produce
// byte-identical output.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Expansion budgets (work-based, never the clock, so runs are deterministic).
const ASTAR_CAP = 300_000;
const GREEDY_CAP = 300_000;
const GLOBAL_CAP = 5_000_000;

let totalExpansions = 0;

interface Dir {
  dr: number;
  dc: number;
  ch: string;
}
const DIRS: Dir[] = [
  { dr: -1, dc: 0, ch: 'u' },
  { dr: 1, dc: 0, ch: 'd' },
  { dr: 0, dc: -1, ch: 'l' },
  { dr: 0, dc: 1, ch: 'r' },
];

// Cell one step from c in (dr, dc), or -1 when it leaves the board.
function step(c: number, dr: number, dc: number, w: number, h: number): number {
  const r = Math.floor(c / w) + dr;
  const cc = (c % w) + dc;
  if (r < 0 || r >= h || cc < 0 || cc >= w) return -1;
  return r * w + cc;
}

function letterFor(dr: number, dc: number): string {
  if (dr === -1) return 'u';
  if (dr === 1) return 'd';
  if (dc === -1) return 'l';
  return 'r';
}

interface Level {
  w: number;
  h: number;
  wall: boolean[];
  goals: number[];
  goalSet: boolean[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Array<boolean>(w * h).fill(false);
  const goalSet = new Array<boolean>(w * h).fill(false);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if (ch === '.' || ch === '*' || ch === '+') {
        goalSet[i] = true;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

// BFS over walkable cells avoiding blocked ones. parent[i] is the previous cell,
// -1 at the start, -2 for unvisited.
function bfsParent(start: number, walkable: boolean[], blocked: boolean[], w: number, h: number): Int32Array {
  const parent = new Int32Array(w * h).fill(-2);
  parent[start] = -1;
  const q: number[] = [start];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    for (const d of DIRS) {
      const n = step(c, d.dr, d.dc, w, h);
      if (n === -1 || !walkable[n] || blocked[n] || parent[n] !== -2) continue;
      parent[n] = c;
      q.push(n);
    }
  }
  return parent;
}

// BFS distances from start over walkable cells; -1 where unreachable.
function bfsDist(start: number, walkable: boolean[], w: number, h: number): Int32Array {
  const d = new Int32Array(w * h).fill(-1);
  d[start] = 0;
  const q: number[] = [start];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    for (const dir of DIRS) {
      const n = step(c, dir.dr, dir.dc, w, h);
      if (n === -1 || !walkable[n] || d[n] !== -1) continue;
      d[n] = d[c]! + 1;
      q.push(n);
    }
  }
  return d;
}

// Lowest cell of a reachable region (its canonical representative).
function reachableMin(parent: Int32Array): number {
  for (let i = 0; i < parent.length; i++) if (parent[i] !== -2) return i;
  return -1;
}

// Walking moves from `from` to `to` along a BFS parent array.
function pathMoves(parent: Int32Array, from: number, to: number, w: number): string {
  let moves = '';
  let c = to;
  while (c !== from) {
    const p = parent[c]!;
    const dr = Math.floor(c / w) - Math.floor(p / w);
    const dc = (c % w) - (p % w);
    moves = letterFor(dr, dc) + moves;
    c = p;
  }
  return moves;
}

// Walking moves from `from` to `to`, or null when unreachable.
function bfsPathMoves(from: number, to: number, walkable: boolean[], blocked: boolean[], w: number, h: number): string | null {
  const parent = bfsParent(from, walkable, blocked, w, h);
  if (parent[to] === -2) return null;
  return pathMoves(parent, from, to, w);
}

interface Problem {
  w: number;
  h: number;
  wall: boolean[];
  walkable: boolean[]; // cells the player may stand on
  boxOk: boolean[]; // cells a box may occupy
  goals: number[];
  goalSet: boolean[];
  boxes: number[]; // sorted initial boxes
  startPlayer: number;
  dead: boolean[]; // cells a box on which can never reach a goal
  dist: Int32Array[]; // dist[from] = BFS distances over walkable cells
}

function buildProblem(
  w: number,
  h: number,
  wall: boolean[],
  walkable: boolean[],
  boxOk: boolean[],
  goals: number[],
  goalSet: boolean[],
  boxes: number[],
  startPlayer: number,
): Problem {
  // Static dead squares: reverse-pull reachability from the goals. A cell is live
  // when a box on it can be pulled to a live cell (destination a box cell, the
  // player's cell walkable); this over-approximates reachability, so a cell it
  // leaves dead is truly dead.
  const live = new Array<boolean>(w * h).fill(false);
  for (const g of goals) live[g] = true;
  let changed = true;
  while (changed) {
    changed = false;
    for (let c = 0; c < w * h; c++) {
      if (!boxOk[c] || live[c]) continue;
      for (const d of DIRS) {
        const n = step(c, d.dr, d.dc, w, h);
        const o = step(c, -d.dr, -d.dc, w, h);
        if (n !== -1 && boxOk[n] && live[n] && o !== -1 && walkable[o]) {
          live[c] = true;
          changed = true;
          break;
        }
      }
    }
  }
  const dead = new Array<boolean>(w * h).fill(false);
  for (let c = 0; c < w * h; c++) dead[c] = boxOk[c] && !live[c];
  // All-pairs BFS distances over walkable cells.
  const dist: Int32Array[] = [];
  for (let s = 0; s < w * h; s++) {
    dist.push(walkable[s] ? bfsDist(s, walkable, w, h) : new Int32Array(w * h).fill(-1));
  }
  return { w, h, wall, walkable, boxOk, goals, goalSet, boxes, startPlayer, dead, dist };
}

// True when some box sits in a 2x2 block of walls/boxes holding no goal:
// every box in such a block is frozen.
function hasDeadBlock(bs: number[], filled: boolean[], goalSet: boolean[], w: number, h: number): boolean {
  for (const b of bs) {
    const r = Math.floor(b / w);
    const c = b % w;
    const corners: [number, number][] = [
      [r - 1, c - 1],
      [r - 1, c],
      [r, c - 1],
      [r, c],
    ];
    for (const [br, bc] of corners) {
      if (br < 0 || bc < 0 || br + 1 >= h || bc + 1 >= w) continue;
      const cells = [br * w + bc, br * w + bc + 1, (br + 1) * w + bc, (br + 1) * w + bc + 1];
      if (cells.some((i) => goalSet[i])) continue;
      if (cells.every((i) => filled[i])) return true;
    }
  }
  return false;
}

function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a: number[] = [];
  for (let i = 0; i < n; i++) a.push(i);
  const gen = (k: number): void => {
    if (k === 1) {
      out.push(a.slice());
      return;
    }
    for (let i = 0; i < k; i++) {
      gen(k - 1);
      const j = k % 2 === 1 ? 0 : i;
      const t = a[j]!;
      a[j] = a[k - 1]!;
      a[k - 1] = t;
    }
  };
  gen(n);
  return out;
}

// Push-based A* (greedy=true: best-first on the heuristic alone). Returns the
// push list as [from, to] cell pairs, or null when the budget runs out.
function solvePush(prob: Problem, cap: number, greedy: boolean): [number, number][] | null {
  const { w, h, walkable, boxOk, goals, goalSet } = prob;
  const n = prob.boxes.length;
  if (n === 0) return [];
  const perms = n <= 4 ? permutations(n) : [];
  const hOf = (bs: number[]): number => {
    if (n <= 4) {
      let best = Infinity;
      for (const p of perms) {
        let s = 0;
        let ok = true;
        for (let i = 0; i < n; i++) {
          const d = prob.dist[bs[i]!]![goals[p[i]!]!]!;
          if (d < 0) {
            ok = false;
            break;
          }
          s += d;
        }
        if (ok && s < best) best = s;
      }
      return best;
    }
    let s = 0;
    for (const b of bs) {
      let m = Infinity;
      for (const g of goals) {
        const d = prob.dist[b]![g]!;
        if (d >= 0 && d < m) m = d;
      }
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  };
  const keyOf = (bs: number[], p: number): string => bs.join(',') + '|' + p;

  // Node storage, in parallel arrays.
  const boxArr: number[][] = [];
  const pnorms: number[] = [];
  const gs: number[] = [];
  const fs: number[] = [];
  const seqs: number[] = [];
  const parents: number[] = [];
  const froms: number[] = [];
  const tos: number[] = [];
  const keys: string[] = [];
  const heap: number[] = [];
  let seq = 0;
  const less = (a: number, b: number): boolean => {
    if (fs[a] !== fs[b]) return fs[a]! < fs[b]!;
    if (gs[a] !== gs[b]) return gs[a]! > gs[b]!;
    return seqs[a]! < seqs[b]!;
  };
  const heapPush = (idx: number): void => {
    heap.push(idx);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  const best = new Map<string, number>();
  const done = new Set<string>();
  const addNode = (bs: number[], p: number, g: number, parent: number, from: number, to: number): void => {
    const idx = boxArr.length;
    const hv = hOf(bs);
    boxArr.push(bs);
    pnorms.push(p);
    gs.push(g);
    fs.push(greedy ? hv : g + hv);
    seqs.push(seq++);
    parents.push(parent);
    froms.push(from);
    tos.push(to);
    keys.push(keyOf(bs, p));
    heapPush(idx);
  };

  if (prob.boxes.some((b) => prob.dead[b])) return null;
  const startBlocked = new Array<boolean>(w * h).fill(false);
  for (const b of prob.boxes) startBlocked[b] = true;
  const startParent = bfsParent(prob.startPlayer, walkable, startBlocked, w, h);
  const startP = reachableMin(startParent);
  if (startP === -1) return null;
  if (!isFinite(hOf(prob.boxes))) return null;
  {
    const filled = new Array<boolean>(w * h).fill(false);
    for (let i = 0; i < w * h; i++) filled[i] = prob.wall[i] || startBlocked[i];
    if (hasDeadBlock(prob.boxes, filled, goalSet, w, h)) return null;
  }
  addNode(prob.boxes, startP, 0, -1, -1, -1);
  best.set(keyOf(prob.boxes, startP), 0);

  let expanded = 0;
  while (heap.length > 0) {
    if (expanded >= cap || totalExpansions >= GLOBAL_CAP) return null;
    const idx = heapPop();
    const key = keys[idx]!;
    if (done.has(key)) continue;
    if (best.get(key) !== gs[idx]) continue;
    done.add(key);
    const bs = boxArr[idx]!;
    const g = gs[idx]!;
    if (bs.every((b) => goalSet[b])) {
      const pushes: [number, number][] = [];
      for (let t = idx; parents[t]! !== -1; t = parents[t]!) pushes.push([froms[t]!, tos[t]!]);
      pushes.reverse();
      return pushes;
    }
    expanded++;
    totalExpansions++;
    const blocked = new Array<boolean>(w * h).fill(false);
    for (const b of bs) blocked[b] = true;
    const parent = bfsParent(pnorms[idx]!, walkable, blocked, w, h);
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi]!;
      for (const d of DIRS) {
        const dest = step(b, d.dr, d.dc, w, h);
        const pusher = step(b, -d.dr, -d.dc, w, h);
        if (dest === -1 || pusher === -1) continue;
        if (!boxOk[dest] || blocked[dest]) continue;
        if (!walkable[pusher] || parent[pusher] === -2) continue;
        if (prob.dead[dest]) continue;
        const nbs = bs.slice();
        nbs[bi] = dest;
        nbs.sort((x, y) => x - y);
        const nblocked = blocked.slice();
        nblocked[b] = false;
        nblocked[dest] = true;
        const nparent = bfsParent(b, walkable, nblocked, w, h);
        const np = reachableMin(nparent);
        if (np === -1) continue;
        const filled = new Array<boolean>(w * h).fill(false);
        for (let i = 0; i < w * h; i++) filled[i] = prob.wall[i] || nblocked[i];
        if (hasDeadBlock(nbs, filled, goalSet, w, h)) continue;
        const hv = hOf(nbs);
        if (!isFinite(hv)) continue;
        const ng = g + 1;
        const nkey = keyOf(nbs, np);
        const prev = best.get(nkey);
        if (prev !== undefined && prev <= ng) continue;
        best.set(nkey, ng);
        addNode(nbs, np, ng, idx, b, dest);
      }
    }
  }
  return null;
}

// Turn a push list into walking + push moves, starting with the player at
// startPlayer. Returns the moves and the player's final cell.
function pushesToMoves(
  prob: Problem,
  pushes: [number, number][],
  startPlayer: number,
): { moves: string; endPlayer: number } | null {
  const { w, h, walkable } = prob;
  let moves = '';
  let player = startPlayer;
  const boxes = prob.boxes.slice();
  for (const [from, to] of pushes) {
    const dr = Math.floor(to / w) - Math.floor(from / w);
    const dc = (to % w) - (from % w);
    const pusher = step(from, -dr, -dc, w, h);
    if (pusher === -1) return null;
    const blocked = new Array<boolean>(w * h).fill(false);
    for (const b of boxes) blocked[b] = true;
    const parent = bfsParent(player, walkable, blocked, w, h);
    if (parent[pusher] === -2) return null;
    moves += pathMoves(parent, player, pusher, w);
    moves += letterFor(dr, dc);
    player = from;
    boxes[boxes.indexOf(from)] = to;
    boxes.sort((a, b) => a - b);
  }
  return { moves, endPlayer: player };
}

interface Room {
  cells: number[];
  boxes: number[];
  goals: number[];
  door: number;
  entry: number;
}

// Split the level into its rooms: connected floor components off the hallway row
// (the row the player starts in). Returns null when the shape is not the
// generated hallway-plus-rooms shape, so the caller falls back to a whole-level
// search instead.
function decompose(level: Level): Room[] | null {
  const { w, h } = level;
  const hr = Math.floor(level.player / w);
  const inHall = (i: number): boolean => Math.floor(i / w) === hr;
  const visited = new Array<boolean>(w * h).fill(false);
  const rooms: Room[] = [];
  const roomCells = new Set<number>();
  for (let i = 0; i < w * h; i++) {
    if (level.wall[i] || inHall(i) || visited[i]) continue;
    const cells: number[] = [];
    const q: number[] = [i];
    visited[i] = true;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      cells.push(c);
      for (const d of DIRS) {
        const n = step(c, d.dr, d.dc, w, h);
        if (n === -1 || level.wall[n] || inHall(n) || visited[n]) continue;
        visited[n] = true;
        q.push(n);
      }
    }
    const boxes = cells.filter((c) => level.boxes.includes(c));
    const goals = cells.filter((c) => level.goalSet[c]);
    if (boxes.length === 0 && goals.length === 0) continue;
    if (boxes.length !== goals.length) return null;
    let door = -1;
    let entry = -1;
    for (const c of cells) {
      for (const d of DIRS) {
        const n = step(c, d.dr, d.dc, w, h);
        if (n !== -1 && inHall(n) && !level.wall[n]) {
          door = c;
          entry = n;
          break;
        }
      }
      if (door !== -1) break;
    }
    if (door === -1) return null;
    for (const c of cells) roomCells.add(c);
    rooms.push({ cells, boxes, goals, door, entry });
  }
  for (const b of level.boxes) if (!roomCells.has(b)) return null;
  for (const g of level.goals) if (!roomCells.has(g)) return null;
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

// Whole-level push search, used only when the level is not the generated
// hallway-plus-rooms shape.
function solveWhole(level: Level): string {
  const { w, h } = level;
  const walkable = level.wall.map((x) => !x);
  const prob = buildProblem(w, h, level.wall, walkable, walkable.slice(), level.goals, level.goalSet, level.boxes, level.player);
  let pushes = solvePush(prob, ASTAR_CAP, false);
  if (!pushes) pushes = solvePush(prob, GREEDY_CAP, true);
  if (!pushes) return '';
  const conv = pushesToMoves(prob, pushes, level.player);
  return conv ? conv.moves : '';
}

function solveLevel(level: Level): string {
  const { w, h } = level;
  const rooms = decompose(level);
  if (!rooms) return solveWhole(level);
  const allFloor = level.wall.map((x) => !x);
  const boxAt = new Array<boolean>(w * h).fill(false);
  for (const b of level.boxes) boxAt[b] = true;
  let moves = '';
  let player = level.player;
  for (const room of rooms) {
    const transit = bfsPathMoves(player, room.entry, allFloor, boxAt, w, h);
    if (transit === null) return solveWhole(level);
    moves += transit;
    player = room.entry;
    const walkable = new Array<boolean>(w * h).fill(false);
    for (const c of room.cells) walkable[c] = true;
    walkable[room.entry] = true;
    const boxOk = new Array<boolean>(w * h).fill(false);
    for (const c of room.cells) boxOk[c] = true;
    const prob = buildProblem(w, h, level.wall, walkable, boxOk, room.goals, level.goalSet, room.boxes, room.entry);
    let pushes = solvePush(prob, ASTAR_CAP, false);
    if (!pushes) pushes = solvePush(prob, GREEDY_CAP, true);
    if (!pushes) return solveWhole(level);
    const conv = pushesToMoves(prob, pushes, player);
    if (!conv) return solveWhole(level);
    moves += conv.moves;
    player = conv.endPlayer;
    for (const b of room.boxes) boxAt[b] = false;
    for (const g of room.goals) boxAt[g] = true;
  }
  return moves;
}

function main(): void {
  const ids = readdirSync(INPUT_DIR)
    .filter((id) => {
      try {
        return readdirSync(join(INPUT_DIR, id)).includes('level.txt');
      } catch {
        return false;
      }
    })
    .sort();
  for (const id of ids) {
    const text = readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8');
    const moves = solveLevel(parseLevel(text));
    const outDir = join(OUTPUT_DIR, id);
    mkdirSync(outDir, { recursive: true });
    writeFileSync(join(outDir, 'solution.txt'), moves + '\n', 'utf8');
  }
}

main();
