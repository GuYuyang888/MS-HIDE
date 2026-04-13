
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
from src.placement.dfs_placement import PlacementSearcher
from src.hmtd.component_basis import build_component_basis_U
from src.hmtd.susceptance_update import compute_susceptance, verify_hiddenness
from src.hmtd.action_library import ActionLibrary
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


class TestHiddennessExact:

    @pytest.fixture(autouse=True)
    def setup_all(self):
        case_data = parse_matpower_case(CASE14_PATH)
        self.model = DCModel(case_data)

        df_set, seed = _find_safe_placement(self.model)
        self.cfg = GlobalConfig(seed=seed)
        self.df_set = df_set

        self.U, self.components, self.s = build_component_basis_U(
            df_set, self.model
        )

                                                             
        if self.s > 0:
            self.library = ActionLibrary(
                self.model, self.df_set, self.U, self.cfg
            )
        else:
            self.library = None

                                                                     
                                                           
                                                                     
    def test_zero_action_zero_violation(self):
        model = self.model
        w_zero = np.zeros(self.s, dtype=np.float64)
        b_zero, feasible = compute_susceptance(
            w_zero, model, self.df_set, self.U, self.cfg
        )

                                                   
        assert feasible, "Zero action must be feasible."

        is_hidden, max_violation = verify_hiddenness(
            w_zero, b_zero, model, self.U, self.cfg
        )
        assert is_hidden, (
            f"Zero action must be hidden; violation = {max_violation:.2e}"
        )
        assert max_violation < 1e-14, (
            f"Zero action violation should be ~0; got {max_violation:.2e}"
        )

                                                                     
                                                   
                                                                     
    def test_all_library_actions_hidden(self):
        if self.library is None:
            pytest.skip("s=0 -- no hidden dimensions, library not built.")

        model = self.model
        hide_tol = self.cfg.hide_tol

        for idx, action in enumerate(self.library.actions):
            w = action["w"]
            b_new = action["b"]

            is_hidden, max_violation = verify_hiddenness(
                w, b_new, model, self.U, self.cfg
            )

            assert is_hidden, (
                f"Action {idx}: hiddenness violated. "
                f"||delta_z||_inf = {max_violation:.2e} > hide_tol = {hide_tol:.2e}"
            )
            assert max_violation <= hide_tol, (
                f"Action {idx}: max_violation={max_violation:.2e} exceeds "
                f"hide_tol={hide_tol:.2e}"
            )

                                                                     
                                                                   
                                                                     
    def test_all_library_actions_feasible_and_hidden_flags(self):
        if self.library is None:
            pytest.skip("s=0 -- no hidden dimensions, library not built.")

        for idx, action in enumerate(self.library.actions):
            assert action["feasible"], (
                f"Action {idx} in library is not feasible."
            )
            assert action["hidden"], (
                f"Action {idx} in library is not hidden."
            )

                                                                     
                                                               
                                                                     
    def test_zero_action_is_first(self):
        if self.library is None:
            pytest.skip("s=0 -- no hidden dimensions, library not built.")

        zero_act = self.library.zero_action
        assert np.allclose(zero_act["w"], 0.0), (
            "The zero_action's w vector is not all zeros."
        )

                                                                     
                                                                        
                                                                     
    def test_independent_hiddenness_recomputation(self):
        if self.library is None:
            pytest.skip("s=0 -- no hidden dimensions, library not built.")

        model = self.model

        for idx, action in enumerate(self.library.actions):
            w = action["w"]
            theta_shifted = model.theta0 + self.U @ w
            H_t = action["H_t"]

            z_new = H_t @ theta_shifted
            delta = z_new - model.z_nom
            max_err = float(np.max(np.abs(delta)))

            assert max_err <= self.cfg.hide_tol, (
                f"Action {idx}: independent recomputation gives "
                f"||delta_z||_inf = {max_err:.2e}"
            )
