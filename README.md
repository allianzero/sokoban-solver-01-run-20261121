# Sokoban Solver

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/allianzero/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/allianzero/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver in a single `main.ts` (Node 22, built-ins only, no
dependencies, no network). It reads every level under `/input/<id>/level.txt` and
writes the player's moves to `/output/<id>/solution.txt`.

## How it works

Generated levels are a 1-wide hallway with 1–4 rooms off it, each room holding
3–4 boxes and as many goals behind a single door. A box pushed out of its room
can never be pushed back, so each room is independent and the level optimum P* is
the sum of the rooms' optima. The solver:

1. **Decomposes** the level into rooms (connected floor components off the
   hallway row, each with its door and hallway entry cell).
2. **Solves each room with push-based A\***: a state is the sorted box cells plus
   the player's reachable region (normalised to its lowest cell). The heuristic is
   the minimum-cost assignment of boxes to goals (Hungarian, at most 24
   permutations for 4 boxes) over BFS distances on the room floor — admissible,
   so A* returns a push-optimal room solution.
3. **Prunes dead positions**: static dead squares (reverse-pull reachability from
   the goals), 2×2 blocks of boxes/walls holding no goal, and a transposition
   table. A greedy best-first search is the fallback if A* exhausts its
   expansion budget.
4. **Stitches the rooms together** with walking moves through the hallway and
   reconstructs the full move string from the push sequences.

The search budget is counted in expansions, never in wall time, and there is no
clock, randomness, or environment input — repeated runs produce byte-identical
output.

## Run it

```sh
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Practice score

Practice set (kit `practice/`): **1,000,000 — 5/5 instances at 1000/1000
(push-optimal on every level)**, versus the house baseline plain-astar 1.0.0 at
200,000.
