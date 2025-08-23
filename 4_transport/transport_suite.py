# -*- coding: utf-8 -*-
"""
Transport Suite (Pure Python): NWC / LCM / VAM + optional Stepping-Stone

This version has **no JSON or inline input parsing**.
👉 Edit the data RIGHT HERE, then run the file.
"""
from __future__ import annotations


# COST = [[-p for p in row] for row in COST]

# =================================

from typing import List, Tuple, Dict, Set
import argparse, math, collections

TOL = 1e-12

# -------------------- Utilities --------------------
def balance_transport(supply: List[float], demand: List[float], cost: List[List[float]]):
    """Balance problem by adding a dummy row/col if needed (cost=0)."""
    S, D = sum(supply), sum(demand)
    m, n = len(supply), len(demand)
    C = [row[:] for row in cost]
    added = {"dummy_row": False, "dummy_col": False}
    if abs(S - D) < 1e-9:
        return supply[:], demand[:], C, added
    if S < D:
        C.append([0.0] * n)
        supply = supply[:] + [D - S]
        added["dummy_row"] = True
    else:
        for r in C:
            r.append(0.0)
        demand = demand[:] + [S - D]
        added["dummy_col"] = True
    return supply, demand, C, added

def total_cost(plan, cost) -> float:
    m, n = len(plan), len(plan[0])
    return sum(plan[i][j] * cost[i][j] for i in range(m) for j in range(n))

def print_transport_table(plan: List[List[float]], cost: List[List[float]], title="Plan (qty@unit_cost)"):
    m, n = len(plan), len(plan[0])
    colw = 13
    print("\n" + title)
    print("-" * colw * (n + 2))
    header = [" "] + [f"Dest{j+1}".center(colw) for j in range(n)] + ["|"]
    print("".join(header))
    print("".join(["-" * colw] * (n + 2)))
    for i in range(m):
        row = [f"Src{i+1}".ljust(colw)]
        for j in range(n):
            cell = f"{plan[i][j]:.4f}@{cost[i][j]:.2f}" if plan[i][j] != 0 else ""
            row.append(cell.center(colw))
        row.append("|")
        print("".join(row))
    print("".join(["-" * colw] * (n + 2)))

# -------------------- Initial methods --------------------
def northwest_corner(supply: List[float], demand: List[float], cost: List[List[float]], verbose=True):
    supply, demand, cost, added = balance_transport(supply, demand, cost)
    m, n = len(supply), len(demand)
    plan = [[0.0 for _ in range(n)] for _ in range(m)]
    if verbose:
        print("=== Northwest Corner (NWC) ===")
        print("Balanced problem:")
        print(f"  supply = {supply}")
        print(f"  demand = {demand}")
        print(f"  added = {added}\n")
    i = j = 0
    step = 0
    while i < m and j < n:
        step += 1
        alloc = min(supply[i], demand[j])
        plan[i][j] = alloc
        supply[i] -= alloc; demand[j] -= alloc
        if verbose:
            print(f"Step {step}: allocate {alloc:.4f} at cell ({i},{j})  cost={cost[i][j]:.4f}")
            print(f"  Remaining: supply[{i}]={supply[i]:.4f}, demand[{j}]={demand[j]:.4f}")
        if abs(supply[i]) < TOL and abs(demand[j]) < TOL:
            if verbose: print("  Both exhausted -> move diagonal (degenerate).")
            i += 1; j += 1
        elif abs(supply[i]) < TOL:
            if verbose: print("  Row exhausted -> move down.")
            i += 1
        elif abs(demand[j]) < TOL:
            if verbose: print("  Column exhausted -> move right.")
            j += 1
        if verbose: print()
    return plan, cost, added

def least_cost_method(supply: List[float], demand: List[float], cost: List[List[float]], verbose=True):
    supply, demand, cost, added = balance_transport(supply, demand, cost)
    m, n = len(supply), len(demand)
    plan = [[0.0 for _ in range(n)] for _ in range(m)]
    active_rows = {i for i in range(m) if supply[i] > 0}
    active_cols = {j for j in range(n) if demand[j] > 0}
    if verbose:
        print("=== Least Cost Method (LCM) ===")
        print("Balanced problem:")
        print(f"  supply = {supply}")
        print(f"  demand = {demand}")
        print(f"  added = {added}\n")
    step = 0
    while active_rows and active_cols:
        step += 1
        best = None
        for i in active_rows:
            for j in active_cols:
                cand = (cost[i][j], i, j)
                if best is None or cand < best:
                    best = cand
        cmin, i, j = best
        alloc = min(supply[i], demand[j])
        plan[i][j] += alloc
        supply[i] -= alloc; demand[j] -= alloc
        if verbose:
            print(f"Step {step}: choose cheapest cell ({i},{j}) with cost={cmin:.4f}")
            print(f"  Allocate {alloc:.4f}")
            print(f"  Remaining: supply[{i}]={supply[i]:.4f}, demand[{j}]={demand[j]:.4f}")
        if abs(supply[i]) < TOL:
            active_rows.discard(i); 
            if verbose: print(f"  Row exhausted -> remove row {i}")
        if abs(demand[j]) < TOL:
            active_cols.discard(j); 
            if verbose: print(f"  Column exhausted -> remove column {j}")
        if verbose: print()
    return plan, cost, added

def _two_smallest(values: List[float]) -> Tuple[float, float]:
    inf = float("inf"); a, b = inf, inf
    for v in values:
        if v < a: a, b = v, a
        elif v < b: b = v
    return a, b

def vogel_approximation(supply: List[float], demand: List[float], cost: List[List[float]], verbose=True):
    supply, demand, cost, added = balance_transport(supply, demand, cost)
    m, n = len(supply), len(demand)
    plan = [[0.0 for _ in range(n)] for _ in range(m)]
    active_rows = {i for i in range(m) if supply[i] > 0}
    active_cols = {j for j in range(n) if demand[j] > 0}
    if verbose:
        print("=== Vogel's Approximation Method (VAM) ===")
        print("Balanced problem:")
        print(f"  supply = {supply}")
        print(f"  demand = {demand}")
        print(f"  added = {added}\n")
    step = 0
    while active_rows and active_cols:
        step += 1
        # penalties
        rpen, cpen = {}, {}
        for i in active_rows:
            vals = [cost[i][j] for j in active_cols]
            m1, m2 = _two_smallest(vals); rpen[i] = (m1, m2, (m2 - m1) if m2 != float("inf") else m1)
        for j in active_cols:
            vals = [cost[i][j] for i in active_rows]
            m1, m2 = _two_smallest(vals); cpen[j] = (m1, m2, (m2 - m1) if m2 != float("inf") else m1)
        if verbose:
            print(f"Step {step}: penalties")
            for j in sorted(active_cols):
                m1,m2,p = cpen[j]; m2s = "∞" if m2 == float('inf') else f"{m2:.4f}"
                print(f"  col[{j}] -> min1={m1:.4f}, min2={m2s}, penalty={p:.4f}")
            for i in sorted(active_rows):
                m1,m2,p = rpen[i]; m2s = "∞" if m2 == float('inf') else f"{m2:.4f}"
                print(f"  row[{i}] -> min1={m1:.4f}, min2={m2s}, penalty={p:.4f}")
            print()
        # choose by max penalty; tie-break by smaller cheapest cell then prefer rows
        best = None  # (penalty, -cheapest, is_row, -idx)
        chosen_type, chosen_idx = None, None
        for i in active_rows:
            m1,m2,p = rpen[i]; score = (p, -m1, 1, -i)
            if best is None or score > best:
                best, chosen_type, chosen_idx = score, 'row', i
        for j in active_cols:
            m1,m2,p = cpen[j]; score = (p, -m1, 0, -j)
            if best is None or score > best:
                best, chosen_type, chosen_idx = score, 'col', j
        if chosen_type == 'row':
            i = chosen_idx
            j = min(active_cols, key=lambda jj: (cost[i][jj], jj))
        else:
            j = chosen_idx
            i = min(active_rows, key=lambda ii: (cost[ii][j], ii))
        alloc = min(supply[i], demand[j])
        plan[i][j] += alloc
        supply[i] -= alloc; demand[j] -= alloc
        if verbose:
            print(f"  Chosen: {chosen_type}[{chosen_idx}] with max penalty {best[0]:.4f}")
            print(f"  Allocate {alloc:.4f} units to cell ({i},{j}) with cost {cost[i][j]:.4f}")
            print(f"  Remaining supply[{i}]={supply[i]:.4f}, demand[{j}]={demand[j]:.4f}\n")
        if abs(supply[i]) < TOL: active_rows.discard(i)
        if abs(demand[j]) < TOL: active_cols.discard(j)
    return plan, cost, added

# add this function to your script

def russel_approximation(supply: List[float], demand: List[float], cost: List[List[float]], verbose=True):
    """Russel's Approximation Method (RAM)"""
    supply, demand, cost, added = balance_transport(supply, demand, cost)
    m, n = len(supply), len(demand)
    plan = [[0.0 for _ in range(n)] for _ in range(m)]

    if verbose:
        print("=== Russel's Approximation Method (RAM) ===")
        print("Balanced problem:")
        print(f"  supply = {supply}")
        print(f"  demand = {demand}")
        print(f"  added = {added}\n")

    step = 0
    while sum(supply) > TOL and sum(demand) > TOL:
        step += 1
        
        # Calculate u_i and v_j
        u = [max([cost[i][j] for j in range(n) if plan[i][j] == 0 and demand[j] > TOL] or [0.0]) for i in range(m)]
        v = [max([cost[i][j] for i in range(m) if plan[i][j] == 0 and supply[i] > TOL] or [0.0]) for j in range(n)]

        # Find the cell with the most negative opportunity cost
        best = None
        for i in range(m):
            if supply[i] <= TOL: continue
            for j in range(n):
                if demand[j] <= TOL: continue
                
                # Check if cell is unallocated and has not been "used"
                if plan[i][j] == 0.0 and supply[i] > TOL and demand[j] > TOL:
                    opportunity_cost = cost[i][j] - u[i] - v[j]
                    if best is None or opportunity_cost < best[0]:
                        best = (opportunity_cost, i, j)
        
        if best is None: break
        
        o_cost, i, j = best
        alloc = min(supply[i], demand[j])
        plan[i][j] += alloc
        supply[i] -= alloc; demand[j] -= alloc

        if verbose:
            print(f"Step {step}: most negative opportunity cost={o_cost:.4f} at cell ({i},{j})")
            print(f"  Allocate {alloc:.4f} units.")
            print(f"  Remaining: supply[{i}]={supply[i]:.4f}, demand[{j}]={demand[j]:.4f}\n")
            
    return plan, cost, added

# -------------------- Stepping-Stone improvement --------------------
def positives_as_basics(plan):
    basics = set()
    for i,row in enumerate(plan):
        for j,x in enumerate(row):
            if x > TOL: basics.add((i,j))
    return basics

class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra; return True
        return False

def ensure_spanning_tree(plan, cost, basic_zeros: Set[Tuple[int,int]]):
    m, n = len(plan), len(plan[0])
    N = m + n
    dsu = DSU(N)
    basics = positives_as_basics(plan).union(basic_zeros)
    edges = []
    for (i,j) in basics:
        if dsu.union(i, m+j):
            edges.append((i,j))
    while len(edges) < m + n - 1:
        candidate = None; best_cost = float('inf')
        for i in range(m):
            for j in range(n):
                if (i,j) in basics: continue
                if dsu.find(i) != dsu.find(m+j):
                    c = cost[i][j]
                    if c < best_cost: best_cost, candidate = c, (i,j)
        if candidate is None: break
        i,j = candidate
        basic_zeros.add((i,j)); basics.add((i,j)); dsu.union(i, m+j); edges.append((i,j))
    return basic_zeros

def build_basis_graph(plan, basic_zeros):
    m, n = len(plan), len(plan[0])
    basics = positives_as_basics(plan).union(basic_zeros)
    g = collections.defaultdict(list)
    for (i,j) in basics:
        g[('r',i)].append(('c',j))
        g[('c',j)].append(('r',i))
    return g

def path_row_to_col(g, i0, j0):
    start = ('r', i0); goal = ('c', j0)
    from collections import deque
    q = deque([start]); prev = {start: None}
    while q:
        u = q.popleft()
        if u == goal: break
        for v in g[u]:
            if v not in prev:
                prev[v] = u; q.append(v)
    if goal not in prev: return None
    nodes = []; cur = goal
    while cur is not None: nodes.append(cur); cur = prev[cur]
    nodes.reverse()
    cells = []
    for a,b in zip(nodes[:-1], nodes[1:]):
        if a[0]=='r' and b[0]=='c': cells.append((a[1], b[1]))
        elif a[0]=='c' and b[0]=='r': cells.append((b[1], a[1]))
        else: raise RuntimeError("Invalid alternation")
    return cells

def loop_for_cell(plan, basic_zeros, start):
    i0, j0 = start
    g = build_basis_graph(plan, basic_zeros)
    path_cells = path_row_to_col(g, i0, j0)
    if not path_cells: return None
    return [start] + path_cells + [start]

def compute_delta(cost, loop):
    delta = 0.0
    for k in range(len(loop)-1):  # ignore last repeated start
        i,j = loop[k]
        delta += cost[i][j] * (1 if k%2==0 else -1)
    return delta

def stepping_stone(plan, cost, verbose=True):
    """Improve 'plan' in-place until no negative Δ remains. Returns improved plan."""
    m, n = len(plan), len(plan[0])
    basic_zeros: Set[Tuple[int,int]] = set()
    basic_zeros = ensure_spanning_tree(plan, cost, basic_zeros)

    iter_no = 0
    while True:
        iter_no += 1
        best = None  # (delta, loop)
        if verbose:
            print(f"Iteration {iter_no}: evaluate Δ for all nonbasic cells")
        basics_now = positives_as_basics(plan).union(basic_zeros)
        for i in range(m):
            for j in range(n):
                if (i,j) in basics_now: continue
                loop = loop_for_cell(plan, basic_zeros, (i,j))
                if not loop: continue
                delta = compute_delta(cost, loop)
                if verbose:
                    print(f"  cell({i},{j}) -> Δ={delta:.4f}  loop={loop}")
                if best is None or delta < best[0]:
                    best = (delta, loop)
        if best is None or best[0] >= -1e-12:
            if verbose: print("\nNo Δ < 0 -> current plan is optimal.\n")
            break
        delta, loop = best
        if verbose:
            print(f"\n=> Choose entering {loop[0]} with most negative Δ={delta:.4f}")
        minus_cells = [loop[k] for k in range(1, len(loop)-1, 2)]
        theta = min(plan[i][j] for (i,j) in minus_cells)
        if verbose:
            print(f"   '-' positions: {minus_cells} -> θ={theta:.4f}")
        # Update along loop
        for k in range(len(loop)-1):
            i,j = loop[k]
            if k%2==0:
                plan[i][j] += theta
            else:
                plan[i][j] -= theta
                if plan[i][j] < TOL: plan[i][j] = 0.0
        # Refresh zero-basics and ensure spanning tree again
        basic_zeros = {c for c in basic_zeros if c not in minus_cells}
        basic_zeros = ensure_spanning_tree(plan, cost, basic_zeros)
        if verbose:
            print(f"   New total cost = {total_cost(plan, cost):.4f}\n")
    return plan

# -------------------- Orchestrator --------------------
# update this function in your script
def solve_transport(supply, demand, cost, method: str, do_optimal: bool, quiet: bool):
    method = method.lower()
    verbose = not quiet
    if method == 'nwc':
        plan, cost_mat, added = northwest_corner(supply[:], demand[:], cost, verbose=verbose)
    elif method == 'lcm':
        plan, cost_mat, added = least_cost_method(supply[:], demand[:], cost, verbose=verbose)
    elif method == 'vam':
        plan, cost_mat, added = vogel_approximation(supply[:], demand[:], cost, verbose=verbose)
    elif method == 'ram': # Add this new elif block
        plan, cost_mat, added = russel_approximation(supply[:], demand[:], cost, verbose=verbose)
    else:
        raise ValueError("method must be one of: nwc, lcm, vam, ram")

    # Show initial plan & cost
    print("\n=== Data ===")
    print("supply =", supply)
    print("demand =", demand)
    print("cost   =", cost)
    print_transport_table(plan, cost_mat, title=f"{method.upper()} Initial Feasible Plan (qty@unit_cost)")
    print(f"\nInitial total cost = {total_cost(plan, cost_mat):.4f}\n")
    if added["dummy_row"] or added["dummy_col"]:
        print("Note: Dummy row/column was added for balancing (cost=0).")

    # Run Stepping-Stone only if requested
    if not do_optimal:
        return
    print("\n=== Stepping-Stone Improvement ===")
    improved = stepping_stone(plan, cost_mat, verbose=verbose)
    print_transport_table(improved, cost_mat, title="Optimal Plan (qty@unit_cost)")
    print(f"\nTotal transportation cost = {total_cost(improved, cost_mat):.4f}")

# update the CLI parser in the main function
def main():
    M = 1e5  # Big-M value for Big-M method (not used here)
    # ====== EDIT YOUR DATA HERE ======
    SUPPLY = [5, 2, 3]
    DEMAND = [3, 3, 2, 2]
    COST = [
        [3, 7, 6, 4],
        [2, 4, 3, 2],
        [4, 3, 8, 5]
]
#
    parser = argparse.ArgumentParser(description="Transport Suite: NWC / LCM / VAM (+ optional Stepping-Stone)")
    parser.add_argument("--method", type=str, default="vam", choices=["nwc", "lcm", "vam", "ram"], help="Initial method") # add 'ram'
    parser.add_argument("--optimal", action="store_true", help="Run Stepping-Stone after the initial method")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    solve_transport(SUPPLY, DEMAND, COST, method=args.method, do_optimal=args.optimal, quiet=args.quiet)

if __name__ == "__main__":
    main()
