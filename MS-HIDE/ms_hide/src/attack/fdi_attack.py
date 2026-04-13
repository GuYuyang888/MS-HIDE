
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


                                                                    
                       
                                                                    

@dataclass(frozen=True)
class Hypothesis:

    id: int
    E_h: Optional[NDArray[np.float64]]
    attack_direction: Optional[NDArray[np.float64]]


                                                                    
                   
                                                                    

class FDIAttackModel:

                                                                             
    def __init__(self, H0: NDArray[np.float64], n: int) -> None:
        H0 = np.asarray(H0, dtype=np.float64)
        if H0.ndim != 2:
            raise ValueError(
                f"H0 must be a 2-D array; got shape {H0.shape}."
            )
        if H0.shape[1] != n:
            raise ValueError(
                f"H0 has {H0.shape[1]} columns but n={n} was specified."
            )

        self._H0: NDArray[np.float64] = H0
        self._n: int = n
        self._m: int = H0.shape[0]
        self._hypotheses: List[Hypothesis] = self._build_hypothesis_library()

                                                                             
    @property
    def H0(self) -> NDArray[np.float64]:
        return self._H0

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    @property
    def hypotheses(self) -> List[Hypothesis]:
        return list(self._hypotheses)

    @property
    def n_hypotheses(self) -> int:
        return self._n + 1

                                                                             
    def _build_hypothesis_library(self) -> List[Hypothesis]:
        library: List[Hypothesis] = []

                                
        library.append(Hypothesis(id=0, E_h=None, attack_direction=None))

                                        
        for k in range(1, self._n + 1):
            E_h = np.zeros(self._n, dtype=np.float64)
            E_h[k - 1] = 1.0                                                    
            direction = self._H0[:, k - 1].copy()                          
            library.append(
                Hypothesis(id=k, E_h=E_h, attack_direction=direction)
            )

        return library

                                                                           
    def construct_attack_vector(
        self, h: int, c_h: float
    ) -> NDArray[np.float64]:
        if h < 0 or h > self._n:
            raise IndexError(
                f"Hypothesis index h={h} is out of range [0, {self._n}]."
            )

        if h == 0:
            return np.zeros(self._m, dtype=np.float64)

        direction = self._hypotheses[h].attack_direction
        assert direction is not None                         
        return direction * float(c_h)

                                                                          
    def get_attack_direction(self, h: int) -> NDArray[np.float64]:
        if h < 0 or h > self._n:
            raise IndexError(
                f"Hypothesis index h={h} is out of range [0, {self._n}]."
            )

        if h == 0:
            return np.zeros(self._m, dtype=np.float64)

        direction = self._hypotheses[h].attack_direction
        assert direction is not None
        return direction.copy()

                                                                              
    def sample_true_attack(
        self,
        cfg: Any,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, float, NDArray[np.float64]]:
        if rng is None:
            rng = np.random.default_rng()

        p_0: float = float(cfg.prior_null)
        n_hyp: int = self.n_hypotheses                       

                                                        
        probs = np.empty(n_hyp, dtype=np.float64)
        probs[0] = p_0
        if n_hyp > 1:
            probs[1:] = (1.0 - p_0) / (n_hyp - 1)

        h_star: int = int(rng.choice(n_hyp, p=probs))

                                                        
        if h_star == 0:
            c_star = 0.0
        else:
            sign = rng.choice([-1.0, 1.0])
            c_star = sign * float(cfg.true_c)

                                                        
        a_star = self.construct_attack_vector(h_star, c_star)

        return h_star, c_star, a_star

                                                                       
    def __repr__(self) -> str:                    
        return (
            f"FDIAttackModel(m={self._m}, n={self._n}, "
            f"|H|={self.n_hypotheses})"
        )
