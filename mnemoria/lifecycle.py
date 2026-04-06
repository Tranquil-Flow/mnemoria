"""
Lifecycle management for the Unified Memory System.

Provides:
  find_articulation_points(conn) - Tarjan's algorithm to find bridge nodes
  protect_bridge_nodes(conn, articulation_points) - Pin bridge nodes
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Set


# ─── Tarjan Articulation Point Detection ─────────────────────


def find_articulation_points(conn: sqlite3.Connection) -> Set[str]:
    """Find articulation points (bridge nodes) in the memory link graph.

    Uses Tarjan's algorithm via DFS:
      - A root node is an articulation point if it has 2+ DFS tree children.
      - A non-root node u is an articulation point if it has a child v such
        that no vertex in v's subtree has a back-edge to an ancestor of u
        (i.e. low[v] >= disc[u]).

    Returns a set of fact_ids that are articulation points.
    """
    # Build adjacency list from um_links (treat as undirected)
    rows = conn.execute(
        "SELECT source_id, target_id FROM um_links"
    ).fetchall()

    graph: Dict[str, Set[str]] = defaultdict(set)
    all_nodes: Set[str] = set()
    for r in rows:
        src = r[0] if isinstance(r, (list, tuple)) else r["source_id"]
        tgt = r[1] if isinstance(r, (list, tuple)) else r["target_id"]
        graph[src].add(tgt)
        graph[tgt].add(src)
        all_nodes.add(src)
        all_nodes.add(tgt)

    if not all_nodes:
        return set()

    # Tarjan's iterative DFS to find articulation points
    visited: Dict[str, bool] = {}
    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    parent: Dict[str, Optional[str]] = {}
    ap: Set[str] = set()
    timer = [0]

    def dfs(start: str) -> None:
        stack = [(start, None, iter(graph[start]))]
        disc[start] = low[start] = timer[0]
        timer[0] += 1
        visited[start] = True
        parent[start] = None
        child_count: Dict[str, int] = defaultdict(int)

        while stack:
            u, par, neighbors = stack[-1]
            try:
                v = next(neighbors)
                if v not in visited:
                    visited[v] = True
                    disc[v] = low[v] = timer[0]
                    timer[0] += 1
                    parent[v] = u
                    child_count[u] += 1
                    stack.append((v, u, iter(graph[v])))
                elif v != par:
                    # Back edge: update low[u]
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            except StopIteration:
                stack.pop()
                if stack:
                    # Update parent's low value
                    p = stack[-1][0]
                    if low[u] < low[p]:
                        low[p] = low[u]
                    # Check articulation point condition for non-root
                    p_is_root = parent[p] is None
                    if p_is_root:
                        if child_count[p] >= 2:
                            ap.add(p)
                    else:
                        if low[u] >= disc[p]:
                            ap.add(p)

    for node in all_nodes:
        if node not in visited:
            dfs(node)

    return ap


def protect_bridge_nodes(
    conn: sqlite3.Connection,
    articulation_points: Set[str],
) -> int:
    """Set pinned=1 on all articulation point facts.

    Bridge nodes are protected from archive/cold-push during gauge pressure
    cascades by setting the pinned flag. The gauge_check code already skips
    pinned facts.

    Returns the number of facts updated.
    """
    if not articulation_points:
        return 0

    placeholders = ",".join("?" * len(articulation_points))
    result = conn.execute(
        f"UPDATE um_facts SET pinned=1 WHERE id IN ({placeholders})",
        list(articulation_points),
    )
    conn.commit()
    return result.rowcount
