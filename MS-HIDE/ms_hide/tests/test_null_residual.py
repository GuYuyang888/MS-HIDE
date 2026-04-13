
import sys
import os

import numpy as np
import pytest
from scipy import stats

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.power.matpower_parser import parse_matpower_case
from src.power.dc_model import DCModel
from src.power.dc_se import DCSE
from src.placement.dfs_placement import PlacementSearcher
from src.hmtd.component_basis import build_component_basis_U
from src.hmtd.action_library import ActionLibrary
from src.estimation.residual_subspace import ResidualSubspace
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


class TestNullResidualDistribution:

    @pytest.fixture(autouse=True)
    def setup_all(self):
        case_data = parse_matpower_case(CASE14_PATH)
        self.model = DCModel(case_data)

        df_set, seed = _find_safe_placement(self.model)
        self.cfg = GlobalConfig(seed=seed)
        self.df_set = df_set

        U, components, s = build_component_basis_U(df_set, self.model)
        self.U = U
        self.s = s

        if s == 0:
            pytest.skip("s=0 -- no hidden dimensions; cannot test HMTD action.")

        self.library = ActionLibrary(
            self.model, self.df_set, self.U, self.cfg
        )

                                                                  
        self.action = None
        for act in self.library.actions[1:]:
            if not np.allclose(act["w"], 0.0):
                self.action = act
                break

        if self.action is None:
            pytest.skip("No non-zero action found in the library.")

        self.se = DCSE(self.model.z_nom)

                                                             
        self.rs = ResidualSubspace(
            self.action["H_t"], self.se.W_sqrt, self.model.H0
        )

                                                                     
                       
                                                                     
    def test_mean_near_zero(self):
        N = 2000
        rng = np.random.default_rng(2024)
        se = self.se
        rs = self.rs
        model = self.model

        Y = np.zeros((N, rs.nu))
        for i in range(N):
            noise = rng.normal(scale=se.sigma)
            z_t = model.z_nom + noise             
            Y[i, :] = rs.project(z_t)

        sample_mean = np.mean(Y, axis=0)

                                                                    
                                                                      
        threshold = 4.0 / np.sqrt(N)
        assert np.all(np.abs(sample_mean) < threshold), (
            f"Sample mean exceeds {threshold:.4f}: "
            f"max |mean_j| = {np.max(np.abs(sample_mean)):.4f}"
        )

                                                                     
                           
                                                                     
    def test_covariance_near_identity(self):
        N = 3000
        rng = np.random.default_rng(2025)
        se = self.se
        rs = self.rs
        model = self.model

        Y = np.zeros((N, rs.nu))
        for i in range(N):
            noise = rng.normal(scale=se.sigma)
            z_t = model.z_nom + noise
            Y[i, :] = rs.project(z_t)

        sample_cov = np.cov(Y, rowvar=False)            
        I_nu = np.eye(rs.nu)

                                                                     
                                                                     
        max_err = np.max(np.abs(sample_cov - I_nu))
        assert max_err < 0.15, (
            f"Covariance deviates from I_nu: max |err| = {max_err:.4f}"
        )

                                                                     
                                                          
                                                                     
    def test_chi_squared_distribution(self):
        N = 2000
        rng = np.random.default_rng(2026)
        se = self.se
        rs = self.rs
        model = self.model

        xi_values = np.zeros(N)
        for i in range(N):
            noise = rng.normal(scale=se.sigma)
            z_t = model.z_nom + noise
            y_t = rs.project(z_t)
            xi_values[i] = float(np.sum(y_t ** 2))

                                                   
        ks_stat, p_value = stats.kstest(xi_values, "chi2", args=(rs.nu,))

                                                            
                                                                   
        assert p_value > 0.01, (
            f"KS test rejects chi^2({rs.nu}) at alpha=0.01: "
            f"KS={ks_stat:.4f}, p={p_value:.4f}"
        )

                                                                     
                                                  
                                                                     
    def test_marginal_normality(self):
        N = 2000
        rng = np.random.default_rng(2027)
        se = self.se
        rs = self.rs
        model = self.model

        Y = np.zeros((N, rs.nu))
        for i in range(N):
            noise = rng.normal(scale=se.sigma)
            z_t = model.z_nom + noise
            Y[i, :] = rs.project(z_t)

                                                            
        for j in range(rs.nu):
            ks_stat, p_value = stats.kstest(Y[:, j], "norm", args=(0.0, 1.0))
            assert p_value > 0.005, (
                f"Component y[{j}] rejects N(0,1): "
                f"KS={ks_stat:.4f}, p={p_value:.4f}"
            )
