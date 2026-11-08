# chosen variant - m
"""II.10.5, variant 'м': nine required SLAE solvers + optional GMRES.

Usage:
  python solve_slau.py
  python solve_slau.py --input my_system.npz --out my_results
  python solve_slau.py --tol 1e-12 --maxiter 1500 --show

Custom data file must contain two arrays named 'A' and 'b':
  np.savez('my_system.npz', A=my_matrix, b=my_rhs)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

import sys
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class IterationResult:
    name: str
    x: np.ndarray
    history: np.ndarray  # h[k] = ||b - A @ x_k||_2 / ||b||_2; includes k=0
    status: str           # 'converged', 'maxiter', 'diverged', 'breakdown', 'skipped'
    details: str = ""

    @property
    def iterations(self) -> int:
        return max(0, len(self.history) - 1)

    @property
    def rel_residual(self) -> float:
        return float(self.history[-1]) if len(self.history) else float("nan")


@dataclass
class DirectResult:
    name: str
    x: np.ndarray | None
    status: str
    details: str = ""
    reconstruction_error: float | None = None


def variant_m(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """II.10.5 м): main diagonal 10, +/-1 diagonals 1, +/-2 diagonals 0.1."""
    if n < 3:
        raise ValueError("Variant 'м' needs n >= 3.")
    A = 10.0 * np.eye(n)
    for offset, coefficient in ((1, 1.0), (2, 0.1)):
        z = np.full(n - offset, coefficient, dtype=float)
        A += np.diag(z, offset) + np.diag(z, -offset)
    b = np.arange(1, n + 1, dtype=float)
    return A, b


def validate_system(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return safe copies and reject invalid input instead of silently reshaping."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix")
    if len(b) != A.shape[0]:
        raise ValueError("b must have exactly as many entries as rows of A")
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("A and b must have finite real values")
    return A.copy(), b.copy()


def relative_residual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """Always check the true residual of the ORIGINAL linear system."""
    denominator = np.linalg.norm(b)
    if denominator == 0.0:
        denominator = 1.0
    return float(np.linalg.norm(b - A @ x) / denominator)


def make_result(
    name: str, x: np.ndarray, history: list[float], status: str, details: str = ""
) -> IterationResult:
    return IterationResult(name, x.copy(), np.asarray(history), status, details)


def inspect_system(A: np.ndarray) -> dict[str, object]:
    """Matrix diagnostics: spectrum, spectral conditioning, diagonal dominance."""
    diagonal = np.diag(A)
    margin = np.abs(diagonal) - (np.sum(np.abs(A), axis=1) - np.abs(diagonal))
    symmetric = bool(np.allclose(A, A.T, rtol=1e-12, atol=1e-12))
    eigs = np.linalg.eigvalsh(A) if symmetric else np.linalg.eigvals(A)
    positive_definite = bool(symmetric and np.all(eigs > 0.0))
    return {
        "n": A.shape[0],
        "symmetric": symmetric,
        "spd": positive_definite,
        "diag_dominant": bool(np.all(margin > 0)),
        "min_diag_margin": float(margin.min()),
        "eigenvalues": eigs,
        "condition_2": float(np.linalg.cond(A)),
        "nonzero_elements": int(np.count_nonzero(A)),
    }


# -------------------------- Direct solvers --------------------------

def triangular_forward(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Forward substitution for a nonsingular lower triangular matrix."""
    n = len(b)
    x = np.empty(n)
    for i in range(n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x


def triangular_backward(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Backward substitution for a nonsingular upper triangular matrix."""
    n = len(b)
    x = np.empty(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def pivot_threshold(A: np.ndarray) -> float:
    """Relative singularity threshold (not a fixed threshold in absolute units)."""
    scale = np.linalg.norm(A, ord=np.inf)
    return np.finfo(float).eps * max(scale, np.finfo(float).tiny) * len(A)


def gaussian_elimination_pivot(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gaussian elimination with PARTIAL pivoting + backward substitution."""
    U, rhs = A.copy(), b.copy()
    n = len(b)
    threshold = pivot_threshold(A)
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[pivot, k]) <= threshold:
            raise np.linalg.LinAlgError(f"Near-zero pivot at column {k + 1}")
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            rhs[[k, pivot]] = rhs[[pivot, k]]
        if k + 1 < n:
            multipliers = U[k + 1:, k] / U[k, k]
            U[k + 1:, k + 1:] -= np.outer(multipliers, U[k, k + 1:])
            rhs[k + 1:] -= multipliers * rhs[k]
            U[k + 1:, k] = 0.0
    return triangular_backward(U, rhs)


def lu_doolittle(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpivoted LU (Doolittle), A=L@U, diag(L)=1.

    If a pivot is zero/small, this particular LU without row permutations
    does not exist numerically; the solver reports a skip rather than hiding it.
    """
    n = len(A)
    L = np.eye(n)
    U = A.copy()
    threshold = pivot_threshold(A)
    for k in range(n):
        if abs(U[k, k]) <= threshold:
            raise np.linalg.LinAlgError(
                f"LU without permutations: zero/tiny pivot at index {k + 1}"
            )
        if k + 1 < n:
            multipliers = U[k + 1:, k] / U[k, k]
            L[k + 1:, k] = multipliers
            U[k + 1:, k + 1:] -= np.outer(multipliers, U[k, k + 1:])
            U[k + 1:, k] = 0.0
    return L, U


def solve_lu(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    L, U = lu_doolittle(A)
    y = triangular_forward(L, b)
    x = triangular_backward(U, y)
    return x, float(np.linalg.norm(A - L @ U, ord=np.inf))


# -------------------------- Iterative solvers --------------------------

def jacobi(A: np.ndarray, b: np.ndarray, tol: float, maxiter: int) -> IterationResult:
    name = "Якоби"
    diagonal = np.diag(A)
    x = np.zeros_like(b)
    history = [relative_residual(A, b, x)]
    if history[-1] <= tol:
        return make_result(name, x, history, "converged")
    if np.any(diagonal == 0.0):
        return make_result(name, x, history, "skipped", "Zero on the diagonal")
    for _ in range(maxiter):
        x = x + (b - A @ x) / diagonal
        error = relative_residual(A, b, x)
        history.append(error)
        if error <= tol:
            return make_result(name, x, history, "converged")
        if not np.isfinite(error) or error > 1e10:
            return make_result(name, x, history, "diverged")
    return make_result(name, x, history, "maxiter")


def sor(
    A: np.ndarray, b: np.ndarray, tol: float, maxiter: int, omega: float
) -> IterationResult:
    name = "Зейдель" if omega == 1.0 else "Верхняя релаксация (SOR)"
    x = np.zeros_like(b)
    history = [relative_residual(A, b, x)]
    if history[-1] <= tol:
        return make_result(name, x, history, "converged")
    diagonal = np.diag(A)
    if np.any(diagonal == 0):
        return make_result(name, x, history, "skipped", "Zero on diagonal")
    if not 0 < omega < 2:
        raise ValueError("SOR requires 0 < omega < 2")
    for _ in range(maxiter):
        old = x.copy()
        for i in range(len(b)):
            low_sum = A[i, :i] @ x[:i]       # New components from this sweep
            high_sum = A[i, i + 1:] @ old[i + 1:]  # Old components
            gs_value = (b[i] - low_sum - high_sum) / diagonal[i]
            x[i] = (1.0 - omega) * old[i] + omega * gs_value
        error = relative_residual(A, b, x)
        history.append(error)
        if error <= tol:
            return make_result(name, x, history, "converged", f"omega={omega:.6f}")
        if not np.isfinite(error) or error > 1e10:
            return make_result(name, x, history, "diverged", f"omega={omega:.6f}")
    return make_result(name, x, history, "maxiter", f"omega={omega:.6f}")


def steepest_descent(
    A: np.ndarray, b: np.ndarray, tol: float, maxiter: int
) -> IterationResult:
    """Steepest descent on .5*x.T@A@x - b.T@x; requires SPD A."""
    name = "Градиентный спуск"
    x = np.zeros_like(b)
    history = [relative_residual(A, b, x)]
    if history[-1] <= tol:
        return make_result(name, x, history, "converged")
    for _ in range(maxiter):
        r = b - A @ x                  # Negative gradient of objective
        Ar = A @ r
        denominator = float(r @ Ar)
        if denominator <= 0.0 or not np.isfinite(denominator):
            return make_result(name, x, history, "breakdown", "Nonpositive step denominator")
        alpha = float((r @ r) / denominator)
        x += alpha * r
        error = relative_residual(A, b, x)
        history.append(error)
        if error <= tol:
            return make_result(name, x, history, "converged")
        if not np.isfinite(error) or error > 1e10:
            return make_result(name, x, history, "diverged")
    return make_result(name, x, history, "maxiter")


def minimum_residuals(
    A: np.ndarray, b: np.ndarray, tol: float, maxiter: int
) -> IterationResult:
    """One-direction minimal residual: minimize ||r-alpha*A*r|| in alpha."""
    name = "Минимальные невязки"
    x = np.zeros_like(b)
    history = [relative_residual(A, b, x)]
    if history[-1] <= tol:
        return make_result(name, x, history, "converged")
    for _ in range(maxiter):
        r = b - A @ x
        Ar = A @ r
        denominator = float(Ar @ Ar)
        if denominator <= 0.0 or not np.isfinite(denominator):
            return make_result(name, x, history, "breakdown", "A @ r vanished")
        numerator = float(Ar @ r)
        if numerator == 0.0:
            return make_result(name, x, history, "breakdown", "Stagnation: step is zero")
        alpha = numerator / denominator
        x += alpha * r
        error = relative_residual(A, b, x)
        history.append(error)
        if error <= tol:
            return make_result(name, x, history, "converged")
        if not np.isfinite(error) or error > 1e10:
            return make_result(name, x, history, "diverged")
    return make_result(name, x, history, "maxiter")


def conjugate_gradients(
    A: np.ndarray, b: np.ndarray, tol: float, maxiter: int
) -> IterationResult:
    """CG with recurrence of A-conjugate search directions; requires SPD A."""
    name = "Сопряжённые градиенты (CG)"
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rr = float(r @ r)
    history = [relative_residual(A, b, x)]
    if history[-1] <= tol:
        return make_result(name, x, history, "converged")
    for _ in range(maxiter):
        Ap = A @ p
        pAp = float(p @ Ap)
        if pAp <= 0.0 or not np.isfinite(pAp):
            return make_result(name, x, history, "breakdown", "p.T @ A @ p not positive")
        alpha = rr / pAp
        x += alpha * p
        r -= alpha * Ap
        error = relative_residual(A, b, x)  # True residual, not just recurrence
        history.append(error)
        if error <= tol:
            return make_result(name, x, history, "converged")
        if not np.isfinite(error):
            return make_result(name, x, history, "diverged")
        next_rr = float(r @ r)
        if rr == 0:
            return make_result(name, x, history, "breakdown", "Zero residual denominator")
        beta = next_rr / rr
        p = r + beta * p
        rr = next_rr
    return make_result(name, x, history, "maxiter")


def bicgstab(
    A: np.ndarray, b: np.ndarray, tol: float, maxiter: int
) -> IterationResult:
    """Unpreconditioned BiCGStab for general nonsymmetric, nonsingular A."""
    name = "BiCGStab"
    x = np.zeros_like(b)
    r = b - A @ x
    shadow = r.copy()              # Fixed shadow residual
    p = np.zeros_like(r)
    v = np.zeros_like(r)
    rho_old = alpha = omega = 1.0
    history = [relative_residual(A, b, x)]
    if history[-1] <= tol:
        return make_result(name, x, history, "converged")
    tiny = 1e-30

    for k in range(maxiter):
        rho = float(shadow @ r)
        if abs(rho) <= tiny:
            return make_result(name, x, history, "breakdown", "rho=0")
        if k == 0:
            p = r.copy()
        else:
            if abs(omega) <= tiny:
                return make_result(name, x, history, "breakdown", "omega=0")
            beta = (rho / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)
        v = A @ p
        denom_alpha = float(shadow @ v)
        if abs(denom_alpha) <= tiny:
            return make_result(name, x, history, "breakdown", "(r_shadow, A*p)=0")
        alpha = rho / denom_alpha
        s = r - alpha * v
        x_trial = x + alpha * p
        if relative_residual(A, b, x_trial) <= tol:
            x = x_trial
            history.append(relative_residual(A, b, x))
            return make_result(name, x, history, "converged")
        t = A @ s
        denom_omega = float(t @ t)
        if denom_omega <= tiny:
            return make_result(name, x, history, "breakdown", "(A*s,A*s)=0")
        omega = float((t @ s) / denom_omega)
        if abs(omega) <= tiny:
            return make_result(name, x, history, "breakdown", "omega=0")
        x = x_trial + omega * s
        r = s - omega * t
        rho_old = rho
        error = relative_residual(A, b, x)
        history.append(error)
        if error <= tol:
            return make_result(name, x, history, "converged")
        if not np.isfinite(error) or error > 1e10:
            return make_result(name, x, history, "diverged")
    return make_result(name, x, history, "maxiter")


def gmres(
    A: np.ndarray, b: np.ndarray, tol: float, maxiter: int
) -> IterationResult:
    """Extra method: unrestarted GMRES using Arnoldi / modified Gram-Schmidt.

    For compact teaching problems this is convenient. For huge matrices use
    restarted GMRES from scipy.sparse.linalg instead of storing O(n*iterations).
    """
    name = "GMRES (доп.)"
    n = len(b)
    x0 = np.zeros_like(b)
    r0 = b - A @ x0
    beta = float(np.linalg.norm(r0))
    history = [relative_residual(A, b, x0)]
    if history[-1] <= tol:
        return make_result(name, x0, history, "converged")
    V = np.zeros((n, n + 1))
    H = np.zeros((n + 1, n))
    V[:, 0] = r0 / beta
    x = x0.copy()
    for j in range(min(maxiter, n)):
        w = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = V[:, i] @ w
            w -= H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] > 1e-14 * np.linalg.norm(A, 2):
            V[:, j + 1] = w / H[j + 1, j]
        rhs = np.zeros(j + 2)
        rhs[0] = beta
        coefficients, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], rhs, rcond=None)
        x = x0 + V[:, :j + 1] @ coefficients
        error = relative_residual(A, b, x)
        history.append(error)
        if error <= tol:
            return make_result(name, x, history, "converged")
        if H[j + 1, j] <= 1e-14 * np.linalg.norm(A, 2):
            return make_result(name, x, history, "breakdown", "Arnoldi happy breakdown")
    return make_result(name, x, history, "maxiter", "Maximum GMRES subspace dimension reached")


def iteration_spectral_radius(A: np.ndarray, method: str, omega: float = 1) -> float:
    """rho(T), where the stationary method has x_next = T @ x + c."""
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    if method == "jacobi":
        T = np.linalg.solve(D, -(L + U))
    elif method == "sor":
        T = np.linalg.solve(D + omega * L, (1 - omega) * D - omega * U)
    else:
        raise ValueError(f"Unknown stationary method: {method}")
    return float(np.max(np.abs(np.linalg.eigvals(T))))


def select_sor_omega(A: np.ndarray) -> tuple[float, float]:
    """Numerical optimization of rho(T_SOR) on 1<omega<2 (no invalid textbook shortcut).

    First a coarse scan, then a local scan. This is an experiment for this
    matrix, not a claim of universal analytic optimality.
    """
    grid = np.linspace(1.00001, 1.95, 95)
    radii = np.array([iteration_spectral_radius(A, "sor", w) for w in grid])
    best = int(np.argmin(radii))
    width = max(grid[1] - grid[0], 0.01)
    grid2 = np.linspace(max(1.000001, grid[best] - width), min(1.99, grid[best] + width), 151)
    radii2 = np.array([iteration_spectral_radius(A, "sor", w) for w in grid2])
    where = int(np.argmin(radii2))
    return float(grid2[where]), float(radii2[where])


def safe_direct(name: str, fn: Callable[[], object]) -> DirectResult:
    try:
        output = fn()
        if isinstance(output, tuple):
            x, reconstruction = output
        else:
            x, reconstruction = output, None
        return DirectResult(name, x, "converged", "", reconstruction)
    except np.linalg.LinAlgError as exc:
        return DirectResult(name, None, "skipped", str(exc))


def run_solvers(
    A: np.ndarray, b: np.ndarray, tol: float, maxiter: int, omega: float | None
) -> tuple[list[DirectResult], list[IterationResult], dict[str, object]]:
    info = inspect_system(A)
    direct = [
        safe_direct("Гаусс (главный элемент)", lambda: gaussian_elimination_pivot(A, b)),
        safe_direct("LU (Дулиттл, без перестановок)", lambda: solve_lu(A, b)),
    ]
    if omega is None:
        if info["spd"]:
            omega, rho = select_sor_omega(A)
            info["omega_selection"] = "numerical spectral-radius optimization"
            info["rho_sor"] = rho
        else:
            omega = 1.0  # For a general matrix overrelaxation is not guaranteed.
            info["omega_selection"] = "1.0 (auto fallback for non-SPD)"
    else:
        info["omega_selection"] = "user-specified"
    info["omega"] = omega

    if np.all(np.diag(A) != 0.0):
        info["rho_jacobi"] = iteration_spectral_radius(A, "jacobi")
        info["rho_gauss_seidel"] = iteration_spectral_radius(A, "sor", 1.0)
        info["rho_sor"] = iteration_spectral_radius(A, "sor", omega)
    results = [jacobi(A, b, tol, maxiter), sor(A, b, tol, maxiter, 1.0)]
    # SOR with omega=1 is exactly Gauss-Seidel: do not mislabel as a new solver.
    if omega > 1.0:
        results.append(sor(A, b, tol, maxiter, omega))
    else:
        results.append(make_result(
            "Верхняя релаксация (SOR)", np.zeros_like(b),
            [relative_residual(A, b, np.zeros_like(b))], "skipped",
            "Need 1 < omega < 2 to call it over-relaxation",
        ))
    if info["spd"]:
        results.extend([
            steepest_descent(A, b, tol, maxiter),
            minimum_residuals(A, b, tol, maxiter),
            conjugate_gradients(A, b, tol, maxiter),
        ])
    else:
        # No silent switch to normal equations! Their condition number is squared.
        for name in ("Градиентный спуск", "Сопряжённые градиенты (CG)"):
            results.append(make_result(
                name, np.zeros_like(b), [relative_residual(A, b, np.zeros_like(b))],
                "skipped", "Requires symmetric positive definite A",
            ))
        results.append(minimum_residuals(A, b, tol, maxiter))
    results += [bicgstab(A, b, tol, maxiter), gmres(A, b, tol, maxiter)]
    return direct, results, info


# -------------------------- Plots and exports --------------------------
COLORS = {
    "Якоби": "#3b82f6",
    "Зейдель": "#f97316",
    "Верхняя релаксация (SOR)": "#10b981",
    "Градиентный спуск": "#a855f7",
    "Минимальные невязки": "#d946ef",
    "Сопряжённые градиенты (CG)": "#ef4444",
    "BiCGStab": "#0891b2",
    "GMRES (доп.)": "#64748b",
}
PLOT_TITLES = {
    "01_all_methods": "Сравнение всех итерационных методов",
    "02_classical_methods": "Классические методы: Якоби, Зейдель, SOR",
    "03_descent_and_residual": "Методы градиентного спуска и минимальных невязок",
    "04_krylov_methods": "Методы Крылова: CG, BiCGStab и GMRES",
}
PLOT_GROUPS = {
    "01_all_methods": None,
    "02_classical_methods": ["Якоби", "Зейдель", "Верхняя релаксация (SOR)"],
    "03_descent_and_residual": ["Градиентный спуск", "Минимальные невязки"],
    "04_krylov_methods": ["Сопряжённые градиенты (CG)", "BiCGStab", "GMRES (доп.)"],
}


def draw_residual_plot(
    data: list[IterationResult], path: Path, title: str, tol: float
) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    for result in data:
        if result.status == "skipped":
            continue
        y = np.where(result.history > 0, result.history, np.nan)
        ax.semilogy(
            np.arange(len(y)), y, linewidth=2.1, marker="o", markersize=3.7,
            label=f"{result.name} ({result.iterations} ит.)",
            color=COLORS.get(result.name),
        )
    ax.axhline(tol, color="#374151", ls="--", lw=1.2, label=f"Порог {tol:g}")
    ax.set(title=title, xlabel="Номер итерации k", ylabel=r"Относительная невязка $\|b-Ax_k\|_2/\|b\|_2$")
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", axis="y", alpha=0.13)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_matrix(A: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 6.5))
    image = ax.imshow(A, cmap="viridis", interpolation="nearest")
    ax.set(title="Структура матрицы A, вариант м)", xlabel="Столбец j", ylabel="Строка i")
    ax.set_xticks(np.arange(0, len(A), max(1, len(A)//10)))
    ax.set_yticks(np.arange(0, len(A), max(1, len(A)//10)))
    ax.tick_params(labelsize=10)
    cb = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cb.set_label("Значение элемента")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    A: np.ndarray, b: np.ndarray, direct: list[DirectResult],
    iterative: list[IterationResult], info: dict[str, object],
    tol: float, out: Path
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plot_dir = out / "plots"
    (plot_dir / "individual").mkdir(parents=True, exist_ok=True)
    np.savez(out / "system.npz", A=A, b=b)
    np.savetxt(out / "matrix_A.csv", A, delimiter=",", fmt="%.16g")
    np.savetxt(out / "vector_b.csv", b.reshape(-1, 1), delimiter=",", fmt="%.16g")
    draw_matrix(A, plot_dir / "05_matrix.png")

    for path_name, subset in PLOT_GROUPS.items():
        data = iterative if subset is None else [r for r in iterative if r.name in subset]
        if data:
            draw_residual_plot(data, plot_dir / f"{path_name}.png", PLOT_TITLES[path_name], tol)

    for j, result in enumerate(iterative, 1):
        if result.status != "skipped":
            draw_residual_plot([result], plot_dir / "individual" / f"method_{j:02}.png", result.name, tol)

    # Reference computed ONLY for validating our implementations.
    try:
        reference = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        reference = None

    with (out / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "type", "iterations", "relative_residual", "error_vs_reference", "status", "notes"])
        for result in direct:
            err = relative_residual(A, b, result.x) if result.x is not None else float("nan")
            vs_ref = np.linalg.norm(result.x - reference) if result.x is not None and reference is not None else float("nan")
            writer.writerow([result.name, "direct", 0, f"{err:.15e}", f"{vs_ref:.15e}", result.status, result.details])
        for result in iterative:
            vs_ref = np.linalg.norm(result.x - reference) if reference is not None and result.status != "skipped" else float("nan")
            writer.writerow([result.name, "iterative", result.iterations,
                             f"{result.rel_residual:.15e}", f"{vs_ref:.15e}", result.status, result.details])

    with (out / "residual_histories.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration"] + [r.name for r in iterative])
        for k in range(1 + max(r.iterations for r in iterative)):
            writer.writerow([k] + [f"{r.history[k]:.16e}" if k < len(r.history) and r.status != "skipped" else "" for r in iterative])

    if reference is not None:
        with (out / "solution.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["i", "x_numpy_reference", "x_gauss_pivot", "x_LU"])
            x_g = direct[0].x
            x_lu = direct[1].x
            for k in range(len(reference)):
                writer.writerow([k + 1, f"{reference[k]:.15g}",
                                 f"{x_g[k]:.15g}" if x_g is not None else "",
                                 f"{x_lu[k]:.15g}" if x_lu is not None else ""])

    with (out / "diagnostics.txt").open("w", encoding="utf-8") as f:
        f.write("II.10.5, variant м); for user-supplied data see system.npz\n")
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
        f.write(f"relative_residual_tolerance: {tol:g}\n")
        for r in direct:
            f.write(f"{r.name}: {r.status}; ||b-Ax||/||b||="
                    + (f"{relative_residual(A,b,r.x):.4e}" if r.x is not None else "n/a") + "\n")
        for r in iterative:
            f.write(f"{r.name}: {r.status}; iter={r.iterations}; res={r.rel_residual:.4e}; {r.details}\n")


def print_report(
    A: np.ndarray, b: np.ndarray, direct: list[DirectResult],
    iterative: list[IterationResult], info: dict[str, object], tol: float
) -> None:
    print("=" * 91)
    print(f"A: {len(A)}x{len(A)} | sym={info['symmetric']} | SPD={info['spd']} | cond2={info['condition_2']:.6f}")
    if info["spd"]:
        print(f"lambda_min={info['eigenvalues'][0]:.8f}; lambda_max={info['eigenvalues'][-1]:.8f}")
    print(f"omega(SOR)={info['omega']:.7f} | target residual={tol:g}")
    if "rho_jacobi" in info:
        print(f"rho(Jacobi)={info['rho_jacobi']:.6f}; rho(GS)={info['rho_gauss_seidel']:.6f}; rho(SOR)={info['rho_sor']:.6f}")
    print("-" * 91)
    print(f"{'Метод':34s} {'Итер.':>6s} {'Невязка (отн.)':>19s}  Статус")
    print("-" * 91)
    for r in direct:
        res = relative_residual(A,b,r.x) if r.x is not None else float("nan")
        print(f"{r.name:34.34s} {0:6d} {res:19.3e}  {r.status}")
    for r in iterative:
        print(f"{r.name:34.34s} {r.iterations:6d} {r.rel_residual:19.3e}  {r.status}")
    print("=" * 91)


def main() -> None:
    parser = argparse.ArgumentParser(description="II.10.5 СЛАУ: 9 методов + GMRES, графики и CSV")
    parser.add_argument("--input", type=Path, default=None, help="NPZ with arrays A and b; default: variant 'м'")
    parser.add_argument("--matrix", type=Path, default=None, help="A as comma-delimited CSV; use with --rhs")
    parser.add_argument("--rhs", type=Path, default=None, help="b as comma-delimited CSV; use with --matrix")
    parser.add_argument("--out", type=Path, default=Path("slau_results"), help="Output folder")
    parser.add_argument("--tol", type=float, default=1e-10, help="Relative residual convergence threshold")
    parser.add_argument("--maxiter", type=int, default=1000, help="Iterations per solver")
    parser.add_argument("--omega", type=float, default=None, help="SOR omega; automatic when omitted")
    parser.add_argument("--show", action="store_true", help="Show the combined residual plot interactively")
    args = parser.parse_args()
    if not 0 < args.tol < 1:
        parser.error("--tol must be in (0, 1)")
    if args.maxiter < 1:
        parser.error("--maxiter must be positive")
    if args.input is not None and (args.matrix or args.rhs):
        parser.error("Use --input OR --matrix/--rhs, not both")
    if (args.matrix is None) != (args.rhs is None):
        parser.error("--matrix and --rhs must be provided together")
    if args.input is not None:
        with np.load(args.input, allow_pickle=False) as loaded:
            A, b = validate_system(loaded["A"], loaded["b"])
    elif args.matrix is not None:
        A, b = validate_system(
            np.loadtxt(args.matrix, delimiter=",", ndmin=2),
            np.loadtxt(args.rhs, delimiter=",", ndmin=1),
        )
    else:
        A, b = validate_system(*variant_m())

    direct, iterative, info = run_solvers(A, b, args.tol, args.maxiter, args.omega)
    save_outputs(A, b, direct, iterative, info, args.tol, args.out)
    print_report(A, b, direct, iterative, info, args.tol)
    print(f"Files saved to: {args.out.resolve()}")
    if args.show:
        img = plt.imread(args.out / "plots" / "01_all_methods.png")
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
