
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

                                                          
CASE14_PATH = os.path.normpath(
    os.path.join(_PROJECT_ROOT, os.pardir, "matpower", "data", "case14.m")
)


class TestDCModelConsistency:

    @pytest.fixture(autouse=True)
    def setup_model(self):
        case_data = parse_matpower_case(CASE14_PATH)
        self.model = DCModel(case_data)

                                                                     
                             
                                                                     
    def test_H_f0_equals_diag_b0_at_A(self):
        model = self.model
        H_f_check = model.b0[:, None] * model.A
        assert np.allclose(model.H_f0, H_f_check), (
            f"H_f0 mismatch: max |err| = {np.max(np.abs(model.H_f0 - H_f_check)):.2e}"
        )

                                                                     
                                   
                                                                     
    def test_H_p0_equals_At_diag_b0_A(self):
        model = self.model
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            H_p_check = model.A.T @ (model.b0[:, None] * model.A)
        assert np.allclose(model.H_p0, H_p_check), (
            f"H_p0 mismatch: max |err| = {np.max(np.abs(model.H_p0 - H_p_check)):.2e}"
        )

                                                                     
                           
                                                                     
    def test_H0_stacking(self):
        model = self.model
        H0_check = np.vstack([model.H_f0, model.H_p0])
        assert np.allclose(model.H0, H0_check), "H0 stacking mismatch"

                                                                     
                             
                                                                     
    def test_z_nom_equals_H0_theta0(self):
        model = self.model
        z_check = model.H0 @ model.theta0
        assert np.allclose(model.z_nom, z_check), (
            f"z_nom mismatch: max |err| = {np.max(np.abs(model.z_nom - z_check)):.2e}"
        )

                                                                     
                     
                                                                     
    def test_m_equals_nl_plus_n(self):
        model = self.model
        assert model.m == model.n_l + model.n, (
            f"m={model.m} != n_l + n = {model.n_l} + {model.n}"
        )

                                                                     
                                                      
                                                                     
    def test_perturbed_b_structure(self):
        model = self.model
        b_pert = model.b0 * 1.1
        H_pert = model.build_H(b_pert)
        H_f_pert = model.build_H_f(b_pert)
        H_p_pert = model.build_H_p(b_pert)

                                                
        assert np.allclose(H_pert, np.vstack([H_f_pert, H_p_pert])), (
            "Perturbed H does not stack correctly from H_f and H_p."
        )

                                        
        assert np.allclose(H_f_pert, b_pert[:, None] * model.A), (
            "Perturbed H_f != diag(b_pert) @ A"
        )

                                              
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            H_p_check_pert = model.A.T @ (b_pert[:, None] * model.A)
        assert np.allclose(H_p_pert, H_p_check_pert), (
            "Perturbed H_p != A^T @ diag(b_pert) @ A"
        )

                                                                     
                                                            
                                                                     
    def test_dimension_shapes(self):
        model = self.model
        assert model.A.shape == (model.n_l, model.n)
        assert model.b0.shape == (model.n_l,)
        assert model.H_f0.shape == (model.n_l, model.n)
        assert model.H_p0.shape == (model.n, model.n)
        assert model.H0.shape == (model.m, model.n)
        assert model.theta0.shape == (model.n,)
        assert model.z_nom.shape == (model.m,)

                                                                     
                                                               
                                                                     
    def test_H_p0_symmetry(self):
        model = self.model
        assert np.allclose(model.H_p0, model.H_p0.T), "H_p0 is not symmetric"
