
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


                                                                        
                 
                                                                        

class _Edge:

    __slots__ = ("idx", "u", "v")

    def __init__(self, idx: int, u: int, v: int) -> None:
        self.idx = idx                                                      
        self.u = u                                          
        self.v = v                                                

    def other(self, node: int) -> int:
        return self.v if node == self.u else self.u

    def __repr__(self) -> str:                    
        return f"_Edge(idx={self.idx}, u={self.u}, v={self.v})"


                                                                        
            
                                                                        

class PlacementSearcher:

                                                 
    _BR_FBUS = 0
    _BR_TBUS = 1

    def __init__(self, model: Any) -> None:
        self._model = model
        self._n_bus: int = model.n_bus
        self._n_l: int = model.n_l

                                                                
                                                              
        self._edges: List[_Edge] = []
        self._adj: Dict[int, List[_Edge]] = defaultdict(list)
        self._build_graph(model._active_branches)

                                                     
        self._fundamental_cycles: Optional[List[List[int]]] = None
                                                                 
        self._path_cache: Dict[Tuple[int, int, FrozenSet[int]], bool] = {}

                                                                        
                        
                                                                        

    def _build_graph(self, active_branches: NDArray) -> None:
        ext2int = self._model._ext2int_map

        for l_idx in range(self._n_l):
            fbus_ext = int(active_branches[l_idx, self._BR_FBUS])
            tbus_ext = int(active_branches[l_idx, self._BR_TBUS])
            u = ext2int[fbus_ext]
            v = ext2int[tbus_ext]
            edge = _Edge(idx=l_idx, u=u, v=v)
            self._edges.append(edge)
            self._adj[u].append(edge)
            self._adj[v].append(edge)

                                                                        
                                 
                                                                        

    def find_fundamental_cycles(self) -> List[List[int]]:
        if self._fundamental_cycles is not None:
            return self._fundamental_cycles

        visited: Set[int] = set()
        parent_edge: Dict[int, Optional[_Edge]] = {}                                 
        parent_node: Dict[int, Optional[int]] = {}                          
        cycles: List[List[int]] = []

        for start in range(self._n_bus):
            if start in visited:
                continue
            if start not in self._adj:
                continue

                                            
            visited.add(start)
            parent_edge[start] = None
            parent_node[start] = None
                                                               
            tree_edges: Set[int] = set()
                                 
            stack: List[int] = [start]
                                                                                
            visited_edges: Set[int] = set()

            while stack:
                node = stack[-1]
                found_unvisited = False

                for edge in self._adj[node]:
                    if edge.idx in visited_edges:
                        continue                               

                    neighbour = edge.other(node)

                    if neighbour not in visited:
                                   
                        visited_edges.add(edge.idx)
                        tree_edges.add(edge.idx)
                        visited.add(neighbour)
                        parent_edge[neighbour] = edge
                        parent_node[neighbour] = node
                        stack.append(neighbour)
                        found_unvisited = True
                        break                                 
                    else:
                                                                      
                        visited_edges.add(edge.idx)
                        cycle = self._extract_cycle_from_back_edge(
                            node, neighbour, edge, parent_node, parent_edge
                        )
                        if cycle:
                            cycles.append(cycle)

                if not found_unvisited:
                    stack.pop()

        self._fundamental_cycles = cycles
        return cycles

    def _extract_cycle_from_back_edge(
        self,
        u: int,
        v: int,
        back_edge: _Edge,
        parent_node: Dict[int, Optional[int]],
        parent_edge: Dict[int, Optional[_Edge]],
    ) -> List[int]:
                                
        ancestors_u: Dict[int, int] = {}                 
        node, depth = u, 0
        while node is not None:
            ancestors_u[node] = depth
            node = parent_node.get(node)
            depth += 1

                                        
        lca = v
        depth_v = 0
        while lca not in ancestors_u:
            p = parent_node.get(lca)
            if p is None:
                return []                                     
            lca = p
            depth_v += 1

                                             
        cycle_edges: List[int] = [back_edge.idx]
        node = u
        while node != lca:
            pe = parent_edge.get(node)
            if pe is None:
                break
            cycle_edges.append(pe.idx)
            node = parent_node[node]

                                                          
        path_v: List[int] = []
        node = v
        while node != lca:
            pe = parent_edge.get(node)
            if pe is None:
                break
            path_v.append(pe.idx)
            node = parent_node[node]
        cycle_edges.extend(reversed(path_v))

        return cycle_edges

                                                                        
                   
                                                                        

    def check_rules(
        self,
        df_set: Set[int],
        strict_all_cycles: bool = False,
    ) -> bool:
        if strict_all_cycles:
            return (
                self._check_rule1_all_cycles(df_set)
                and self._check_rule2_all_cycles(df_set)
            )
        return self._check_rules_fundamental(df_set)

    def _check_rules_fundamental(self, df_set: Set[int]) -> bool:
        cycles = self.find_fundamental_cycles()
        for cycle in cycles:
            if not self._check_cycle_rules(cycle, df_set):
                return False
        return True

    def _check_rule1_all_cycles(self, df_set: Set[int]) -> bool:
        for v in range(self._n_bus):
            incident = self._adj.get(v, [])
            ndf_edges = [e for e in incident if e.idx not in df_set]
            if len(ndf_edges) < 2:
                continue

            for i in range(len(ndf_edges)):
                e1 = ndf_edges[i]
                u = e1.other(v)
                if u == v:
                    continue
                for j in range(i + 1, len(ndf_edges)):
                    e2 = ndf_edges[j]
                    w = e2.other(v)
                    if w == v:
                        continue

                                                                            
                                                                     
                    if u == w:
                        return False

                    if self._has_path_excluding(u, w, (v,)):
                        return False
        return True

    def _check_rule2_all_cycles(self, df_set: Set[int]) -> bool:
        for e_mid in self._edges:
            if e_mid.idx not in df_set:
                continue

            v = e_mid.u
            w = e_mid.v

            for e_left in self._adj[v]:
                if e_left.idx == e_mid.idx or e_left.idx not in df_set:
                    continue
                u = e_left.other(v)
                                                                            
                                       
                if u in (v, w):
                    continue

                for e_right in self._adj[w]:
                    if (
                        e_right.idx == e_mid.idx
                        or e_right.idx == e_left.idx
                        or e_right.idx not in df_set
                    ):
                        continue
                    x = e_right.other(w)
                    if x in (v, w):
                        continue

                                                                     
                    if u == x:
                        return False

                    if self._has_path_excluding(u, x, (v, w)):
                        return False

        return True

    def _has_path_excluding(
        self,
        src: int,
        dst: int,
        forbidden_vertices: Tuple[int, ...],
    ) -> bool:
        forbidden = frozenset(forbidden_vertices)
        if src in forbidden or dst in forbidden:
            return False

        a, b = (src, dst) if src <= dst else (dst, src)
        key = (a, b, forbidden)
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached

        if src == dst:
            self._path_cache[key] = True
            return True

        visited: Set[int] = {src}
        queue: deque[int] = deque([src])

        while queue:
            node = queue.popleft()
            for edge in self._adj.get(node, []):
                nb = edge.other(node)
                if nb in forbidden or nb in visited:
                    continue
                if nb == dst:
                    self._path_cache[key] = True
                    return True
                visited.add(nb)
                queue.append(nb)

        self._path_cache[key] = False
        return False

    def _check_cycle_rules(self, cycle: List[int], df_set: Set[int]) -> bool:
        k = len(cycle)
        if k <= 1:
            return True                                            

                                                       
        is_df = [branch_idx in df_set for branch_idx in cycle]

                                                                
                                                                 
                                                                        
                                          
        max_ndf_run = self._max_cyclic_run(is_df, target=False)
        max_df_run = self._max_cyclic_run(is_df, target=True)

                                       
        if max_ndf_run >= 2:
            return False

                                      
        if max_df_run >= 3:
            return False

        return True

    @staticmethod
    def _max_cyclic_run(flags: List[bool], target: bool) -> int:
        k = len(flags)
        if k == 0:
            return 0

        best = 0
        run = 0
        for i in range(2 * k):
            if flags[i % k] == target:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0

        return min(best, k)

                                                                        
                      
                                                                        

    def find_valid_placement(
        self,
        min_df: Optional[int] = None,
        max_df: Optional[int] = None,
        max_attempts: int = 10_000,
        rng: Optional[np.random.Generator] = None,
        strict_all_cycles: bool = False,
    ) -> Optional[Set[int]]:
        if rng is None:
            rng = np.random.default_rng()

        if min_df is None:
            min_df = int(np.ceil(self._n_l / 3))
        if max_df is None:
            max_df = int(np.ceil(2 * self._n_l / 3))

                      
        min_df = max(0, min(min_df, self._n_l))
        max_df = max(min_df, min(max_df, self._n_l))

        all_branches = list(range(self._n_l))
        cycles = self.find_fundamental_cycles()

        for _ in range(max_attempts):
                                                              
            n_df = rng.integers(min_df, max_df + 1)
            df_set = set(rng.choice(all_branches, size=n_df, replace=False).tolist())

                              
            df_set = self._repair_placement(df_set, cycles, rng, max_repair_steps=200)

            if df_set is not None and min_df <= len(df_set) <= max_df:
                if self.check_rules(df_set, strict_all_cycles=strict_all_cycles):
                    return df_set

        return None

    def _repair_placement(
        self,
        df_set: Set[int],
        cycles: List[List[int]],
        rng: np.random.Generator,
        max_repair_steps: int = 200,
    ) -> Optional[Set[int]]:
        df_set = set(df_set)                  

        for _ in range(max_repair_steps):
            violation_found = False

            for cycle in cycles:
                k = len(cycle)
                if k <= 1:
                    continue

                is_df = [b in df_set for b in cycle]

                                                                        
                          
                rule1_violation = self._find_consecutive_run(
                    is_df, target=False, min_run=2
                )
                if rule1_violation is not None:
                    violation_found = True
                                                                           
                    offending_indices = rule1_violation
                    pick = rng.choice(offending_indices)
                    df_set.add(cycle[pick])
                    continue

                                                                       
                rule2_violation = self._find_consecutive_run(
                    is_df, target=True, min_run=3
                )
                if rule2_violation is not None:
                    violation_found = True
                                                                           
                    offending_indices = rule2_violation
                    pick = rng.choice(offending_indices)
                    df_set.discard(cycle[pick])
                    continue

            if not violation_found:
                return df_set

        return None

    @staticmethod
    def _find_consecutive_run(
        flags: List[bool], target: bool, min_run: int
    ) -> Optional[List[int]]:
        k = len(flags)
        if k == 0:
            return None

        run_start = -1
        run_len = 0

        for i in range(2 * k):
            if flags[i % k] == target:
                if run_len == 0:
                    run_start = i
                run_len += 1
                                                                  
                capped = min(run_len, k)
                if capped >= min_run:
                    return [j % k for j in range(i - capped + 1, i + 1)]
            else:
                run_len = 0

        return None

                                                                        
                                         
                                                                        

    def find_placements(
        self,
        n_placements: int = 10,
        min_df: Optional[int] = None,
        max_df: Optional[int] = None,
        max_attempts_per: int = 5_000,
        rng: Optional[np.random.Generator] = None,
        strict_all_cycles: bool = False,
    ) -> List[Set[int]]:
        if rng is None:
            rng = np.random.default_rng()

        found: List[Tuple[int, FrozenSet[int]]] = []
        seen: Set[FrozenSet[int]] = set()

        total_attempts = n_placements * max_attempts_per

        for _ in range(total_attempts):
            if len(found) >= n_placements:
                break

            result = self.find_valid_placement(
                min_df=min_df,
                max_df=max_df,
                max_attempts=1,                           
                rng=rng,
                strict_all_cycles=strict_all_cycles,
            )
            if result is None:
                continue

            frozen = frozenset(result)
            if frozen in seen:
                continue
            seen.add(frozen)

            score = self._score_placement(result)
            found.append((score, frozen))

                                  
        found.sort(key=lambda x: x[0], reverse=True)
        return [set(fs) for _, fs in found]

    def _score_placement(self, df_set: Set[int]) -> int:
        ndf_adj: Dict[int, Set[int]] = defaultdict(set)
        ndf_buses: Set[int] = set()

        for edge in self._edges:
            if edge.idx not in df_set:
                ndf_adj[edge.u].add(edge.v)
                ndf_adj[edge.v].add(edge.u)
                ndf_buses.add(edge.u)
                ndf_buses.add(edge.v)

                                                                        
                                                                           
        all_buses = set()
        for edge in self._edges:
            all_buses.add(edge.u)
            all_buses.add(edge.v)
        for bus in all_buses:
            if bus not in ndf_buses:
                ndf_buses.add(bus)

                                 
        visited: Set[int] = set()
        n_components = 0
        for bus in ndf_buses:
            if bus in visited:
                continue
            n_components += 1
            queue = deque([bus])
            visited.add(bus)
            while queue:
                node = queue.popleft()
                for nb in ndf_adj.get(node, set()):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

        return n_components

                                                                        
                 
                                                                        

    @property
    def n_branches(self) -> int:
        return self._n_l

    @property
    def n_buses(self) -> int:
        return self._n_bus

    @property
    def edges(self) -> List[_Edge]:
        return list(self._edges)

    def __repr__(self) -> str:                    
        return (
            f"PlacementSearcher(n_buses={self._n_bus}, "
            f"n_branches={self._n_l}, "
            f"n_fundamental_cycles="
            f"{len(self._fundamental_cycles) if self._fundamental_cycles else '?'})"
        )


                                                                        
                                   
                                                                        

def find_hidden_placements(
    model: Any,
    cfg: Any,
    n_placements: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> List[Set[int]]:
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    searcher = PlacementSearcher(model)

                                                                        
                                                
    min_df = int(np.ceil(model.n_l / 3))
    max_df = int(np.ceil(2 * model.n_l / 3))

    strict_all_cycles = bool(getattr(cfg, "strict_cycle_rules", False))
    max_attempts_per = int(
        getattr(cfg, "placement_max_attempts_per", 5000)
    )
    if strict_all_cycles:
        max_attempts_per = int(
            getattr(cfg, "placement_max_attempts_per_strict", max_attempts_per)
        )

    placements = searcher.find_placements(
        n_placements=n_placements,
        min_df=min_df,
        max_df=max_df,
        rng=rng,
        strict_all_cycles=strict_all_cycles,
        max_attempts_per=max_attempts_per,
    )
    if strict_all_cycles and not placements:
        logger.warning(
            "No placement satisfies strict all-cycle Rule 1/2 checks. "
            "No relaxed fallback is performed in strict mode."
        )
    return placements
