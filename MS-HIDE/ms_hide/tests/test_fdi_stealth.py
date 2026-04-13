
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
from src.estimation.residual_subspace import ResidualSubspace
from src.estimation.bdd import BDDDetector

CASE14_PATH = os.path.normpath(
    os.path.join(_PROJECT_ROOT, os.pardir, "matpower", "data", "case14.m")
)


class TestFDIStealthWithoutMTD:

    @pytest.fixture(autouse=True)
    def setup_all(self):
        case_data = parse_matpower_case(CASE14_PATH)
        self.model = DCModel(case_data)
        self.se = DCSE(self.model.z_nom)
        self.attack_model = FDIAttackModel(self.model.H0, self.model.n)

                          
        self.rs = ResidualSubspace(
            self.model.H0, self.se.W_sqrt, self.model.H0
        )
        self.bdd = BDDDetector(self.rs.nu)

                                                                     
                                                     
                                                                     
    def test_attack_in_column_space(self):
        model = self.model
        c_h = 0.04

        for h in range(1, model.n + 1):
            a = self.attack_model.construct_attack_vector(h, c_h)
                                                   
            expected = model.H0[:, h - 1] * c_h
            assert np.allclose(a, expected), (
                f"h={h}: attack vector != H0[:, h-1] * c"
            )

                                                                     
                                                          
                                                                     
    def test_residual_unchanged_by_attack(self):
        model = self.model
        se = self.se
        c_h = 0.04
        rng = np.random.default_rng(42)

        for h in range(1, model.n + 1):
            a = self.attack_model.construct_attack_vector(h, c_h)

                                                    
            noise = rng.normal(scale=se.sigma)

                                           
            z_clean = model.z_nom + noise
            result_clean = se.estimate(z_clean, model.H0)
            r_clean_norm = np.linalg.norm(result_clean.residual)

                                         
            z_attack = model.z_nom + a + noise
            result_attack = se.estimate(z_attack, model.H0)
            r_attack_norm = np.linalg.norm(result_attack.residual)

            assert np.isclose(r_clean_norm, r_attack_norm, rtol=1e-10), (
                f"h={h}: residual norms differ -- clean={r_clean_norm:.6e}, "
                f"attack={r_attack_norm:.6e}"
            )

                                                                     
                                                                  
                                                                     
    def test_projected_residual_unchanged(self):
        model = self.model
        se = self.se
        rs = self.rs
        c_h = 0.04
        rng = np.random.default_rng(99)

        for h in range(1, model.n + 1):
            a = self.attack_model.construct_attack_vector(h, c_h)
            noise = rng.normal(scale=se.sigma)

            z_clean = model.z_nom + noise
            z_attack = model.z_nom + a + noise

            y_clean = rs.project(z_clean)
            y_attack = rs.project(z_attack)

            assert np.allclose(y_clean, y_attack, atol=1e-10), (
                f"h={h}: projected residuals differ by "
                f"max |err| = {np.max(np.abs(y_clean - y_attack)):.2e}"
            )

                                                                     
                                                                           
                                                                     
    def test_bdd_rate_matches_false_alarm(self):
        model = self.model
        se = self.se
        rs = self.rs
        bdd = self.bdd
        c_h = 0.04
        h_target = 1
        N = 2000
        rng = np.random.default_rng(123)

        a = self.attack_model.construct_attack_vector(h_target, c_h)

        detections = 0
        for _ in range(N):
            noise = rng.normal(scale=se.sigma)
            z_attack = model.z_nom + a + noise
            y = rs.project(z_attack)
            if bdd.detect(y):
                detections += 1

        empirical_rate = detections / N
                                                                      
        assert abs(empirical_rate - bdd.alpha_fa) < 0.04, (
            f"Detection rate {empirical_rate:.3f} deviates too much from "
            f"alpha_fa={bdd.alpha_fa:.3f} (stealth attack should be invisible)."
        )

                                                                     
                                                                 
                                                                     
    def test_sensitivity_vector_zero_at_u0(self):
        rs = self.rs

        g_dict = rs.compute_all_g(self.attack_model)
        for h, g_h in g_dict.items():
            assert np.allclose(g_h, 0.0, atol=1e-10), (
                f"h={h}: ||g_h|| = {np.linalg.norm(g_h):.2e} should be ~0 "
                f"at u=0 (H_t = H0)"
            )
