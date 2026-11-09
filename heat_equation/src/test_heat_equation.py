"""Independent invariants and accuracy checks; run with unittest discovery."""
import unittest
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy import sparse
from scipy.sparse.linalg import spsolve
from heat_equation import (dirichlet_system, full_field, solve, exact_square,
                           robin_fv, reconstruct, heat_balance,
                           manufactured_robin, robin_first_order, diagnostic_callback)


class HeatEquationTests(unittest.TestCase):
    def test_boundary_orientation_and_cholesky(self):
        a, b = dirichlet_system(4)
        u = cho_solve(cho_factor(a.toarray()), b)
        field = full_field(u, 4)
        self.assertAlmostEqual(field[2, 2], 2.5, places=13)
        np.testing.assert_equal(field[1:-1, 0], 1.)
        np.testing.assert_equal(field[1:-1, -1], 3.)
        np.testing.assert_equal(field[0, 1:-1], 4.)
        np.testing.assert_equal(field[-1, 1:-1], 2.)
        self.assertTrue(np.all((u >= 1)&(u <= 4)))

    def test_all_iterative_methods(self):
        n = 8
        a, b = dirichlet_system(n)
        reference = cho_solve(cho_factor(a.toarray()), b)
        for method in ("Jacobi", "GS", "SOR", "SD", "CG", "BiCGStab"):
            with self.subTest(method=method):
                u, hist, _ = solve(a, b, method, tol=1e-10, omega=1.5, n=n)
                self.assertLessEqual(np.max(abs(b-a@u)), 1e-10)
                self.assertEqual(hist[0], np.max(abs(b)))
                np.testing.assert_allclose(u, reference, atol=4e-10, rtol=0)

    def test_lexicographic_sweep_matches_matrix_splitting(self):
        a, b = dirichlet_system(6)
        for omega in (1., 1.4, 1.8):
            fast, h1, _ = solve(a, b, "SOR", tol=1e-8, omega=omega, n=6)
            generic, h2, _ = solve(a, b, "SOR", tol=1e-8, omega=omega)
            np.testing.assert_allclose(h1, h2, atol=1e-14, rtol=1e-5)
            np.testing.assert_allclose(fast, generic, atol=1e-13, rtol=0)

    def test_fourier_reference_and_second_order_interior(self):
        self.assertAlmostEqual(float(exact_square(.5, .5)), 2.5, places=13)
        # Equal boundary data give a constant harmonic extension.
        x, y = np.meshgrid([.1, .3, .6, .9], [.1, .4, .7, .9])
        np.testing.assert_allclose(exact_square(x, y, (2, 2, 2, 2)), 2, atol=1e-13)
        errors = []
        for n in (16, 32):
            a, b = dirichlet_system(n)
            u = spsolve(a, b).reshape(n-1, n-1)
            # Stay off x=y: these particular side data give u=2.5 there.
            errors.append(abs(u[3*n//8-1, n//4-1]-exact_square(.25, .375)))
        self.assertGreater(np.log2(errors[0]/errors[1]), 1.9)

    def test_manufactured_robin_accuracy(self):
        first, second = [], []
        for n in (16, 32):
            a, b, exact = robin_first_order(n)
            first.append(np.max(abs(spsolve(a, b).reshape(exact.shape)-exact)))
            a, b, meta, exact = manufactured_robin(n)
            self.assertEqual((a-a.T).nnz, 0)
            second.append(np.max(abs(reconstruct(spsolve(a, b), meta)-exact)))
        self.assertTrue(.9 < np.log2(first[0]/first[1]) < 1.1)
        self.assertTrue(1.9 < np.log2(second[0]/second[1]) < 2.1)

    def test_constant_robin_equilibrium_and_mixed_balance(self):
        n = 8
        a, b, meta = robin_fv(n, 0., dict.fromkeys("LRBT", 17.))
        np.testing.assert_allclose(reconstruct(spsolve(a, b), meta), 17., atol=1e-12)
        mask = np.zeros((n+1, n+1), dtype=bool)
        mask[3:6, 0] = True
        a, b, meta = robin_fv(n, 200., dict.fromkeys("LRBT", -10.), dirichlet=(mask, -10.))
        cho_factor(a.toarray())  # positive definiteness, not just symmetry
        u = reconstruct(spsolve(a, b), meta)
        balance = heat_balance(u, meta)
        self.assertAlmostEqual(balance["source_W"], 200., places=12)
        self.assertLess(abs(balance["imbalance_W"]), 1e-10)
        np.testing.assert_equal(u[mask], -10.)
        self.assertGreaterEqual(u.min(), -10.-1e-12)

    def test_invalid_parameters_and_zero_rhs(self):
        a, b = dirichlet_system(4)
        with self.assertRaises(ValueError):
            solve(a, b, "SOR", omega=2.)
        with self.assertRaises(RuntimeError):
            solve(a, b, "Jacobi", maxiter=1)
        u, hist, _ = solve(a, np.zeros_like(b), "CG")
        self.assertEqual(len(hist), 1)
        np.testing.assert_equal(u, 0)

    def test_second_order_one_sided_boundary(self):
        errors = []
        for n in (16, 32, 64):
            a, b, exact = robin_first_order(n, order=2)
            self.assertGreater((a-a.T).nnz, 0)
            errors.append(np.max(abs(spsolve(a, b).reshape(exact.shape)-exact)))
        self.assertTrue(1.9 < np.log2(errors[-2]/errors[-1]) < 2.1)

    def test_robin_order_across_biot_numbers(self):
        for beta in (.1, 10.):
            errors = []
            for n in (16, 32):
                a, b, meta, exact = manufactured_robin(n, beta)
                errors.append(np.max(abs(reconstruct(spsolve(a, b), meta)-exact)))
            self.assertTrue(1.9 < np.log2(errors[0]/errors[1]) < 2.1)

    def test_bicgstab_half_step_and_nonsymmetric_system(self):
        a = sparse.eye(3, format="csr")
        b = np.array([1., 2., -3.])
        u, history, _ = solve(a, b, "BiCGStab")
        np.testing.assert_equal(u, b)
        self.assertEqual(len(history), 2)
        a, b, _ = robin_first_order(8, order=2)
        u, history, _ = solve(a, b, "BiCGStab", tol=1e-9)
        np.testing.assert_allclose(u, spsolve(a, b), atol=2e-8, rtol=0)
        self.assertLessEqual(history[-1], 1e-9)
        self.assertTrue(np.all(history[:-1] > 1e-9))

    def test_cg_energy_and_true_residual_diagnostics(self):
        a, b = dirichlet_system(12)
        reference = spsolve(a, b)
        records = []
        _, history, _ = solve(a, b, "CG", callback=diagnostic_callback(a, reference, "CG", records))
        np.testing.assert_equal(history, [r["residual_inf"] for r in records])
        energy = np.array([r["energy_error"] for r in records])
        self.assertTrue(np.all(np.diff(energy) <= 1e-12))

    def test_window_flux_excludes_robin_even_when_temperatures_differ(self):
        n = 8
        fixed = np.zeros((n+1, n+1), dtype=bool)
        fixed[2:7, 0] = True
        a, b, meta = robin_fv(n, 20., dict.fromkeys("LRBT", -10.), dirichlet=(fixed, 7.))
        u = reconstruct(spsolve(a, b), meta)
        ids, g, _ = meta["boundary"]["L"]
        np.testing.assert_equal(g[fixed.ravel()[ids]], 0.)
        # Independent window power: conductive edges into the fixed set,
        # plus source generated inside its control volumes.
        expected = float(np.sum((meta["source"]*meta["volumes"])[fixed]))
        for j in range(n+1):
            for i in range(n+1):
                for jj, ii in ((j+1, i), (j, i+1)):
                    if jj > n or ii > n or fixed[j, i] == fixed[jj, ii]:
                        continue
                    conductance = .5 if (i == ii and i in (0, n)) or (j == jj and j in (0, n)) else 1.
                    sign = 1 if fixed[j, i] else -1
                    expected += sign*conductance*(u[jj, ii]-u[j, i])
        balance = heat_balance(u, meta)
        self.assertAlmostEqual(balance["dirichlet_reaction_W"], expected, places=11)
        self.assertAlmostEqual(balance["imbalance_W"], 0., places=10)
        # Conservation defect is exactly the negative sum of free residuals,
        # also away from the solution.
        u[3, 3] += .7
        balance = heat_balance(u, meta)
        self.assertAlmostEqual(balance["imbalance_W"], -balance["free_residual_sum_W"], places=10)

    def test_affine_robin_solution_with_dimensional_coefficients(self):
        n, length, k, alpha = 6, 2., 3., 4.
        y, x = np.mgrid[0:n+1, 0:n+1]*length/n
        exact = 7+2*x-3*y
        ambient = dict(L=exact[:, 0]-2*k/alpha, R=exact[:, -1]+2*k/alpha,
                       B=exact[0]+3*k/alpha, T=exact[-1]-3*k/alpha)
        a, b, meta = robin_fv(n, 0., ambient, alpha, k, length)
        np.testing.assert_allclose(reconstruct(spsolve(a, b), meta), exact, atol=5e-13, rtol=0)

    def test_reject_invalid_inputs_and_incompatible_fast_sweep(self):
        a, b = dirichlet_system(4)
        for bad in (float("nan"), float("inf"), 0., -1.):
            with self.assertRaises(ValueError):
                solve(a, b, "CG", tol=bad)
        for n in (True, 1, 3.5):
            with self.assertRaises(ValueError):
                dirichlet_system(n)
            with self.assertRaises(ValueError):
                robin_fv(n, 0., dict.fromkeys("LRBT", 0.))
        with self.assertRaises(ValueError):
            solve(2*a, b, "SOR", n=4)
        with self.assertRaises(ValueError):
            solve(a, b[:, None], "CG")
        with self.assertRaises(RuntimeError):
            solve(-sparse.eye(2), np.ones(2), "CG")
        with self.assertRaises(RuntimeError):
            solve(sparse.csr_matrix((2, 2)), np.ones(2), "BiCGStab")


if __name__ == "__main__":
    unittest.main(verbosity=2)
