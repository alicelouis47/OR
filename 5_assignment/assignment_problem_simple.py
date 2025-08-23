# -*- coding: utf-8 -*-
"""
Assignment Problem — Hungarian (Munkres) Algorithm (Simple Version)
Pure-Python, step-by-step. **No FORBIDDEN / BIG_M logic.**

How to use
----------
1) Edit the DATA BLOCK below (MODE, MATRIX).
2) Run:  python assignment_problem_simple.py
   (Optional) Flags:
      --mode min|max   # override MODE in file
      --quiet          # suppress step-by-step (final result only)

Tips
----
- If you want to "forbid" an assignment yourself, just put a very large
  number directly into MATRIX (e.g., 1e9) so the algorithm avoids it.
"""

from __future__ import annotations
from typing import List, Tuple
import argparse

# ======== DATA BLOCK (EDIT HERE) ========
MODE = "min"   # "min" = minimize COST, "max" = maximize PROFIT

M = 1e9
# If MODE="min": MATRIX = costs. If MODE="max": MATRIX = profits.
MATRIX = [
    [820, 810, 840, 960],
    [820, 810, 840, 960],
    [800, 870, M, 920],
    [800, 870, M, 920],
    [740, 900, 810, 840],
]
# =======================================

TOL = 1e-12

# -------------- Utilities --------------
def deepcopy_mat(a: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in a]

def shape(a: List[List[float]]) -> Tuple[int,int]:
    return len(a), len(a[0]) if a else 0

def print_mat(title: str, a: List[List[float]]):
    print(title)
    for r in a:
        print("  ", " ".join(f"{v:9.3f}" for v in r))
    print()

def pad_to_square(a: List[List[float]], pad_value: float=0.0) -> Tuple[List[List[float]], int, int]:
    """Return square matrix padded with pad_value; also original rows, cols."""
    m, n = shape(a)
    N = max(m, n)
    b = [row + [pad_value]*(N - n) for row in deepcopy_mat(a)]
    for _ in range(N - m):
        b.append([pad_value]*N)
    return b, m, n

def to_minimization(a: List[List[float]], mode: str) -> Tuple[List[List[float]], str]:
    """Transform MATRIX into a nonnegative 'cost' matrix to minimize."""
    b = deepcopy_mat(a)
    if mode == "max":
        mx = max(max(row) for row in b) if b else 0.0
        for i in range(len(b)):
            for j in range(len(b[0])):
                b[i][j] = mx - b[i][j]
        return b, f"Transformed for PROFIT (maximize): cost' = max(P)-P (max={mx})"
    else:
        return b, "Using COST directly (minimize)."

# -------------- Hungarian --------------
def hungarian(cost: List[List[float]], verbose=True) -> Tuple[List[int], float]:
    """
    Hungarian for square cost matrix.
    Returns (assignment, total_cost).
    assignment[i] = column chosen for row i (or -1 if dummy row).
    """
    n = len(cost)
    a = deepcopy_mat(cost)

    if verbose:
        print("=== Hungarian Algorithm (step-by-step) ===\n")
        print_mat("Initial cost matrix:", a)

    # Step 1: Row reduction
    for i in range(n):
        rmin = min(a[i])
        for j in range(n):
            a[i][j] -= rmin
    if verbose:
        print_mat("After row reduction:", a)

    # Step 2: Column reduction
    for j in range(n):
        cmin = min(a[i][j] for i in range(n))
        for i in range(n):
            a[i][j] -= cmin
    if verbose:
        print_mat("After column reduction:", a)

    # Masks: 1=star, 2=prime
    mask = [[0]*n for _ in range(n)]
    row_cover = [False]*n
    col_cover = [False]*n

    # Step 3: Star a zero in each row if possible
    for i in range(n):
        for j in range(n):
            if abs(a[i][j]) < TOL and not row_cover[i] and not col_cover[j]:
                mask[i][j] = 1
                row_cover[i] = True
                col_cover[j] = True
                break
    row_cover = [False]*n
    col_cover = [False]*n
    if verbose:
        print("Step 3: initial starred zeros (1=star, 2=prime)")
        for i in range(n): print("  ", mask[i])
        print()

    def cover_columns_of_starred():
        for i in range(n):
            for j in range(n):
                if mask[i][j] == 1:
                    col_cover[j] = True

    def find_a_zero():
        for i in range(n):
            if not row_cover[i]:
                for j in range(n):
                    if abs(a[i][j]) < TOL and not col_cover[j] and mask[i][j] == 0:
                        return i, j
        return None, None

    def find_star_in_row(i):
        for j in range(n):
            if mask[i][j] == 1: return j
        return None

    def find_star_in_col(j):
        for i in range(n):
            if mask[i][j] == 1: return i
        return None

    def find_prime_in_row(i):
        for j in range(n):
            if mask[i][j] == 2: return j
        return None

    def augment_path(path):
        for (r,c) in path:
            if mask[r][c] == 1:
                mask[r][c] = 0
            elif mask[r][c] == 2:
                mask[r][c] = 1

    def clear_primes_and_covers():
        for i in range(n):
            for j in range(n):
                if mask[i][j] == 2:
                    mask[i][j] = 0
        for i in range(n): row_cover[i] = False
        for j in range(n): col_cover[j] = False

    # Step 4
    cover_columns_of_starred()
    if verbose:
        print("Step 4: cover columns containing a starred zero")
        print("  col_cover:", col_cover, "\n")

    while True:
        if sum(col_cover) == n:
            break
        z_i, z_j = find_a_zero()
        while z_i is None:
            # Step 7: Adjust matrix
            min_uncovered = min(
                a[i][j] for i in range(n) for j in range(n) if not row_cover[i] and not col_cover[j]
            )
            for i in range(n):
                if row_cover[i]:
                    for j in range(n):
                        a[i][j] += min_uncovered
            for j in range(n):
                if not col_cover[j]:
                    for i in range(n):
                        a[i][j] -= min_uncovered
            if verbose:
                print(f"Adjust matrix by min uncovered = {min_uncovered:.3f}")
                print_mat("Matrix now:", a)
            z_i, z_j = find_a_zero()

        mask[z_i][z_j] = 2  # prime
        s_col = find_star_in_row(z_i)
        if s_col is None:
            # Step 6: Augmenting path
            if verbose:
                print(f"Found uncovered zero at ({z_i},{z_j}), no star in its row -> build augmenting path")
            path = [(z_i, z_j)]
            row, col = z_i, z_j
            while True:
                r = find_star_in_col(col)
                if r is None:
                    break
                path.append((r, col))
                c = find_prime_in_row(r)
                path.append((r, c))
                row, col = r, c
            if verbose:
                print("  Augmenting path:", path)
            augment_path(path)
            clear_primes_and_covers()
            cover_columns_of_starred()
            if verbose:
                print("  After augmenting, new stars:")
                for i in range(n): print("   ", mask[i])
                print("  col_cover:", col_cover, "\n")
        else:
            row_cover[z_i] = True
            col_cover[s_col] = False
            if verbose:
                print(f"Prime at ({z_i},{z_j}), star in same row at col {s_col} -> cover row {z_i}, uncover col {s_col}")

    # Build assignment
    assignment = [-1]*n
    total = 0.0
    for i in range(n):
        for j in range(n):
            if mask[i][j] == 1:
                assignment[i] = j
                total += cost[i][j]
                break
    return assignment, total

# -------------- Orchestrator --------------
def solve(M: List[List[float]], mode: str, verbose: bool):
    if not M or not M[0]:
        raise SystemExit("MATRIX must be non-empty and rectangular.")
    cols = {len(r) for r in M}
    if len(cols) != 1: raise SystemExit("All rows must have the same length.")

    print("Mode:", "MIN (minimize cost)" if mode=="min" else "MAX (maximize profit)")
    print_mat("Input matrix:", M)

    # Convert to minimization if profit
    M2, note = to_minimization(M, mode)
    print(note); print_mat("Matrix to minimize:", M2)

    # Pad to square
    M3, orig_m, orig_n = pad_to_square(M2, pad_value=0.0)
    if len(M3) != len(M2):
        print(f"Padded to square {len(M3)}x{len(M3)} (original {orig_m}x{orig_n})\n")
        if verbose: print_mat("Square matrix:", M3)

    # Run Hungarian
    assignment, total_cost = hungarian(M3, verbose=verbose)

    # Build mapping back to original shape (ignore dummy rows/cols)
    mapping = []
    real_total_cost = 0.0
    for i in range(orig_m):
        j = assignment[i]
        if j is not None and 0 <= j < orig_n:
            mapping.append((i, j))
            real_total_cost += M2[i][j]  # use minimized matrix to sum cost
        else:
            mapping.append((i, None))  # assigned to dummy

    # Interpret result
    print("\n=== RESULT ===")
    human_map = [(i+1, j+1) if j is not None else (i+1, None) for i, j in mapping]
    print("Assignments (row -> col) [1-based, None=dummy]:", human_map)

    if mode == "max":
        # Recompute profit directly from original MATRIX (profits)
        profit = 0.0
        for i, j in mapping:
            if j is not None:
                profit += M[i][j]
        print(f"Total PROFIT = {profit:.3f}")
    else:
        print(f"Total COST = {real_total_cost:.3f}")

def main():
    # ======== DATA BLOCK (EDIT HERE) ========
    MODE = "min"   # "min" = minimize COST, "max" = maximize PROFIT

    M = 1e4
    # If MODE="min": MATRIX = costs. If MODE="max": MATRIX = profits.
    MATRIX = [
        [8, 6, 3, 7],
        [5, M, 8, 4],
        [6, 3, 9, 6],
        [0, 0, 0, 0],
    ]
    parser = argparse.ArgumentParser(description="Assignment Problem via Hungarian Algorithm (Simple)")
    parser.add_argument("--mode", choices=["min","max"], help="override MODE in file")
    parser.add_argument("--quiet", action="store_true", help="suppress step-by-step details")
    args = parser.parse_args()

    mode = args.mode if args.mode else MODE
    verbose = not args.quiet
    solve(MATRIX, mode=mode, verbose=verbose)

if __name__ == "__main__":
    main()
