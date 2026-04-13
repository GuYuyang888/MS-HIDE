
import sys
import os

import numpy as np
import pytest

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.power.matpower_parser import parse_matpower_case
from src.power.dc_model import DCModel
from src.power.dc_se import DCSE
from src.attack.fdi_attack import FDIAttackModel
from src.placement.dfs_placement import PlacementSearcher
from src.hmtd.component_basis import build_component_basis_U
from src.hmtd.action_library import ActionLibrary
from src.estimation.residual_subspace import ResidualSubspace
from src.estimation.bdd import BDDDetector
from config.cfg_global import GlobalConfig

CASE14_PATH = os.path.normpath(
    os.path.join(_PROJECT_ROOT, os.pardir, "matpower", "data", "case14.m")
)

_MIN_ABS_FLOW = 1e-6


def _find_safe_placement(model, max_seeds=200):
    d0 = model.A @ model.theta0
    zero_flow = set(int(i) for i in np.where(np.abs(d0) < _MIN_ABS_FLOW)[0])

    for seed in range(1, max_seeds + 1):
        rng = np.random.default_rng(seed)
        searcher = PlacementSearcher(model)
        df_set = searcher.find_valid_placement(rng=rng, max_attempts=400)
        if df_set is not None and df_set.isdisjoint(zero_flow):
            U, _, s = build_component_basis_U(df_set, model)
            if s > 0:
                return df_set, seed

    raise AssertionError(
        "Could not find a valid placement with non-zero flow on all DF branches."
    )


def _total_raw_detectability(
    g_dict: dict,
) -> float:
    return sum(float(np.sum(g_h ** 2)) for g_h in g_dict.values())


class TestDetectabilityGain:

    @pytest.fixture(autouse=True)
    def setup_all(self):
        case_data = parse_matpower_case(CASE14_PATH)
        self.model = DCModel(case_data)

        df_set, seed = _find_safe_placement(self.model)
        self.cfg = GlobalConfig(seed=seed)
        self.df_set = df_set

        self.se = DCSE(self.model.z_nom)
        self.attack_model = FDIAttackModel(self.model.H0, self.model.n)

        U, components, s = build_component_basis_U(df_set, self.model)
        self.U = U
        self.s = s

        if s == 0:
            pytest.skip("s=0 -- no hidden dimensions; cannot test MTD actions.")

        self.library = ActionLibrary(
            self.model, self.df_set, self.U, self.cfg
        )

                                                                     
                                          
                                                                     
    def test_d_raw_zero_at_u0(self):
        rs_zero = ResidualSubspace(
            self.model.H0, self.se.W_sqrt, self.model.H0
        )
        g_dict_zero = rs_zero.compute_all_g(self.attack_model)
        D_raw_zero = _total_raw_detectability(g_dict_zero)

        assert D_raw_zero < 1e-15, (
            f"D_raw at u=0 should be ~0, got {D_raw_zero:.2e}"
        )

                                                                     
                                                    
                                                                     
    def test_at_least_one_action_improves_detectability(self):
        D_raw_values = []

        for action in self.library.actions:
            if np.allclose(action["w"], 0.0):
                continue                    

            H_t = action["H_t"]
            rs = ResidualSubspace(H_t, self.se.W_sqrt, self.model.H0)
            g_dict = rs.compute_all_g(self.attack_model)
            D_raw = _total_raw_detectability(g_dict)
            D_raw_values.append(D_raw)

        assert len(D_raw_values) > 0, (
            "No non-zero actions found in the library."
        )

        max_D_raw = max(D_raw_values)
        assert max_D_raw > 1e-6, (
            f"Best non-zero action has D_raw = {max_D_raw:.2e}, "
            f"which is not a meaningful improvement."
        )

                                                                     
                                                                   
                                                                     
    def test_nonzero_actions_have_nonzero_sensitivity(self):
        for action in self.library.actions:
            if np.allclose(action["w"], 0.0):
                continue

            H_t = action["H_t"]
            rs = ResidualSubspace(H_t, self.se.W_sqrt, self.model.H0)
            g_dict = rs.compute_all_g(self.attack_model)

            max_g_norm = max(
                float(np.linalg.norm(g_h)) for g_h in g_dict.values()
            )
            assert max_g_norm > 1e-10, (
                f"Non-zero action w={action['w']} has all g_h ~ 0. "
                f"max ||g_h|| = {max_g_norm:.2e}"
            )

                                                                     
                                                                    
                                                                     
    def test_detection_probability_improvement(self):
        c_h = self.cfg.true_c

        best_pd = 0.0

        for action in self.library.actions:
            if np.allclose(action["w"], 0.0):
                continue

            H_t = action["H_t"]
            rs = ResidualSubspace(H_t, self.se.W_sqrt, self.model.H0)
            bdd = BDDDetector(rs.nu, self.cfg.alpha_fa)
            g_dict = rs.compute_all_g(self.attack_model)

            for h, g_h in g_dict.items():
                pd = bdd.detection_probability(g_h, c_h)
                if pd > best_pd:
                    best_pd = pd

        assert best_pd > self.cfg.alpha_fa + 0.01, (
            f"Best detection probability {best_pd:.4f} is not meaningfully "
            f"above alpha_fa={self.cfg.alpha_fa:.4f}. MTD is ineffective."
        )

                                                                     
                                                                  
                                                                     
    def test_d_raw_equals_sum_noncentrality_at_unit_c(self):
        for action in self.library.actions:
            if np.allclose(action["w"], 0.0):
                continue

            H_t = action["H_t"]
            rs = ResidualSubspace(H_t, self.se.W_sqrt, self.model.H0)
            g_dict = rs.compute_all_g(self.attack_model)

            D_raw = _total_raw_detectability(g_dict)

                                                                   
            sum_lambda = sum(
                BDDDetector.noncentrality(g_h, 1.0)
                for g_h in g_dict.values()
            )
            assert np.isclose(D_raw, sum_lambda, rtol=1e-10), (
                f"D_raw={D_raw:.6e} != sum(lambda_h)={sum_lambda:.6e}"
            )
            break                                                       
