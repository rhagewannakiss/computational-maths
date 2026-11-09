"""Stationary heat conduction: textbook problem 16 and Robin extension.

Arrays are indexed [y, x]; N is the number of intervals, not nodes.
Iteration recurrences are explicit; sparse direct and triangular solves use SciPy.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/heat_equation_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.signal import lfilter
from scipy.sparse.linalg import spsolve, spsolve_triangular

METHOD_NAMES = {"Jacobi": "Якоби", "GS": "Гаусс–Зейдель", "SOR": "SOR",
                "SD": "Наискорейший спуск", "CG": "CG", "BiCGStab": "BiCGStab"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.prop_cycle": plt.cycler(color=["#147d92", "#db7437", "#6957a5",
                                                          "#77923e", "#254969", "#be4c70"])})


def validate_grid(n):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")


def dirichlet_system(n, sides=(1., 3., 4., 2.)):
    """h^2 (-Delta_h) u=b on unit coordinates; sides L,R,B,T."""
    validate_grid(n)
    if np.shape(sides) != (4,) or not np.all(np.isfinite(sides)):
        raise ValueError("sides must contain four finite temperatures in L,R,B,T order")
    m = n - 1
    d = sparse.diags([-np.ones(m-1), 2*np.ones(m), -np.ones(m-1)],
                     [-1, 0, 1], format="csr")
    a = sparse.kronsum(d, d, format="csr")
    b = np.zeros((m, m))
    left, right, bottom, top = sides
    b[:, 0] += left
    b[:, -1] += right
    b[0, :] += bottom
    b[-1, :] += top
    return a, b.ravel()


def full_field(v, n, sides=(1., 3., 4., 2.)):
    left, right, bottom, top = sides
    u = np.zeros((n+1, n+1))
    u[1:-1, 1:-1] = np.asarray(v).reshape(n-1, n-1)
    u[:, 0], u[:, -1], u[0, :], u[-1, :] = left, right, bottom, top
    # Corner values are for plotting only and never enter the interior stencil.
    u[0, 0], u[0, -1] = (left+bottom)/2, (right+bottom)/2
    u[-1, 0], u[-1, -1] = (left+top)/2, (right+top)/2
    return u


def solve(a, b, method, tol=1e-10, omega=1., maxiter=100000, n=None, callback=None):
    """Zero initial guess, absolute infinity norm of the true residual.

    n enables a validated square Dirichlet-grid SOR sweep using a row recurrence.
    General SOR uses the equivalent sparse triangular splitting.
    callback(iteration, x, true_residual) receives copies, including iteration 0.
    """
    if not np.isfinite(tol) or tol <= 0 or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("tol must be finite and positive; maxiter must be a positive integer")
    a = sparse.csr_matrix(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if b.ndim != 1 or b.size == 0 or a.shape != (b.size, b.size):
        raise ValueError("a must be square and match a nonempty one-dimensional b")
    if not np.all(np.isfinite(a.data)) or not np.all(np.isfinite(b)):
        raise ValueError("a and b must be finite")
    if method not in {"Jacobi", "GS", "SOR", "SD", "CG", "BiCGStab"}:
        raise ValueError(f"Unknown method: {method}")
    if method == "GS":
        omega = 1.
    if method == "SOR" and not 0 < omega < 2:
        raise ValueError("SOR requires 0 < omega < 2")
    if method in {"SD", "CG"} and (a-a.T).nnz:
        raise ValueError(f"{method} requires a symmetric positive definite matrix")
    if n is not None and method in {"GS", "SOR"}:
        expected, _ = dirichlet_system(n)
        if a.shape != expected.shape or (a-expected).nnz:
            raise ValueError("n enables only the standard square Dirichlet stencil")
    x = np.zeros_like(b, dtype=float)
    norm = lambda r: float(np.linalg.norm(r, np.inf))
    history = [norm(b)]
    if callback is not None:
        callback(0, x.copy(), b.copy())
    start = perf_counter()
    if history[0] <= tol:
        return x, np.array(history), perf_counter()-start
    diag = a.diagonal()
    if method in {"Jacobi", "GS", "SOR"} and np.any(diag == 0):
        raise ValueError("stationary iteration requires a nonzero diagonal")
    if method in {"SOR", "GS"} and n is None:
        lower = sparse.tril(a, -1, format="csr")
        upper = sparse.triu(a, 1, format="csr")
        triangular = sparse.diags(diag) + omega*lower
        remainder = (1-omega)*sparse.diags(diag) - omega*upper
    r, p = b.copy(), b.copy()
    shadow = r.copy()
    rho_old = alpha = stabilization = 1.
    v = np.zeros_like(b)

    def divisor(value, name, positive=False):
        if not np.isfinite(value) or value == 0 or (positive and value < 0):
            raise RuntimeError(f"{method} breakdown: {name}={value}; SPD required for SD/CG")
        return value

    for iteration in range(1, maxiter+1):
        if method == "Jacobi":
            x += (b-a@x)/diag
        elif method in {"SOR", "GS"}:
            if n is None:
                x = spsolve_triangular(triangular, omega*b+remainder@x,
                                      lower=True)
            else:
                m = n-1
                old = x.reshape(m, m)
                new = np.empty_like(old)
                rhs = b.reshape(m, m)
                for j in range(m):
                    row = (1-omega)*old[j] + omega/4*rhs[j]
                    if j:
                        row += omega/4*new[j-1]
                    if j+1 < m:
                        row += omega/4*old[j+1]
                    row[:-1] += omega/4*old[j, 1:]
                    new[j] = lfilter([1.], [1., -omega/4], row)
                x = new.ravel()
        elif method == "SD":
            r = b-a@x
            ar = a@r
            x += (r@r)/divisor(r@ar, "r.Ar", positive=True)*r
        elif method == "CG":
            ap = a@p
            rr = r@r
            alpha = rr/divisor(p@ap, "p.Ap", positive=True)
            x += alpha*p
            rnew = r-alpha*ap
            p = rnew + (rnew@rnew)/rr*p
            r = rnew
        elif method == "BiCGStab":
            rho = divisor(shadow@r, "shadow.r")
            if iteration == 1:
                p = r.copy()
            else:
                beta = (rho/rho_old)*(alpha/divisor(stabilization, "omega"))
                p = r+beta*(p-stabilization*v)
            v = a@p
            alpha = rho/divisor(shadow@v, "shadow.v")
            s = r-alpha*v
            half = x+alpha*p
            # Exactly the same true infinity-norm test at a possible half-step.
            if norm(b-a@half) <= tol:
                x = half
            else:
                t = a@s
                stabilization = divisor((t@s)/divisor(t@t, "t.t"), "omega")
                x = half+stabilization*s
                r = s-stabilization*t
                rho_old = rho
        true_r = b-a@x
        residual = norm(true_r)
        history.append(residual)
        if callback is not None:
            callback(iteration, x.copy(), true_r.copy())
        if not np.isfinite(residual):
            raise RuntimeError(f"{method} produced a nonfinite residual")
        if residual <= tol:
            return x, np.array(history), perf_counter()-start
    raise RuntimeError(f"{method} exceeded {maxiter} iterations; r={history[-1]}")


def exact_square(x, y, sides=(1., 3., 4., 2.), terms=1200):
    """Fourier harmonic extension on open square, with stable sinh ratios."""
    x, y = np.broadcast_arrays(x, y)
    out = np.zeros_like(x, dtype=float)
    left, right, bottom, top = sides
    for m in range(1, 2*terms, 2):
        k = np.pi*m
        def ratio(t):
            return np.exp(k*(t-1))*(-np.expm1(-2*k*t))/(-np.expm1(-2*k))
        out += 4/k*(np.sin(k*y)*(left*ratio(1-x)+right*ratio(x))
                    + np.sin(k*x)*(bottom*ratio(1-y)+top*ratio(y)))
    return out


def robin_fv(n, source, ambient, alpha=1., conductivity=1., length=1.,
             dirichlet=None):
    """Node-centered finite volumes, half cells on sides, quarter at corners.

    ambient is a dict L,R,B,T of scalar or boundary-array temperatures.
    Return reduced symmetric matrix, rhs, and reconstruction/flux metadata.
    Source is volumetric W/m^3; depth is 1 m. Boundary conductances are W/K.
    """
    validate_grid(n)
    if not np.all(np.isfinite([alpha, conductivity, length])) or min(alpha, conductivity, length) <= 0:
        raise ValueError("n>=2 and positive physical coefficients are required")
    h = length/n
    w = np.ones(n+1)
    w[[0, -1]] = .5
    volumes = h*h*np.outer(w, w)
    q = np.broadcast_to(source, (n+1, n+1))
    if not np.all(np.isfinite(q)):
        raise ValueError("source must be finite")
    count = (n+1)**2
    fixed = np.zeros(count, dtype=bool)
    values = np.zeros(count)
    if dirichlet is not None:
        mask, temperatures = dirichlet
        if np.shape(mask) != (n+1, n+1) or np.asarray(mask).dtype != bool:
            raise ValueError("Dirichlet mask must be a boolean grid of shape (n+1,n+1)")
        fixed = np.asarray(mask).ravel().copy()
        values[fixed] = np.broadcast_to(temperatures, (n+1, n+1)).ravel()[fixed]
        if not np.all(np.isfinite(values)) or np.all(fixed):
            raise ValueError("Dirichlet values must be finite and at least one node must be free")
    rows, cols, data = [], [], []
    diagonal = np.zeros((n+1, n+1))
    b = (q*volumes).ravel().copy()
    def edge(j, i, jj, ii, g):
        u, v = j*(n+1)+i, jj*(n+1)+ii
        rows.extend((u, v)); cols.extend((v, u)); data.extend((-g, -g))
        diagonal[j, i] += g; diagonal[jj, ii] += g
    for j in range(n+1):
        for i in range(n+1):
            if i < n:
                edge(j, i, j, i+1, conductivity*w[j])
            if j < n:
                edge(j, i, j+1, i, conductivity*w[i])
    boundary = {}
    index = np.arange((n+1)**2).reshape(n+1, n+1)
    for side, ids in zip("LRBT", (index[:, 0], index[:, -1], index[0], index[-1])):
        g = alpha*h*w
        g = np.where(fixed[ids], 0., g)  # Robin and Dirichlet faces are disjoint.
        t = np.broadcast_to(ambient[side], (n+1,)).copy()
        if not np.all(np.isfinite(t)):
            raise ValueError("ambient temperatures must be finite")
        diagonal.ravel()[ids] += g
        b[ids] += g*t
        boundary[side] = (ids, g, t)
    rows.extend(range(count)); cols.extend(range(count)); data.extend(diagonal.ravel())
    a = sparse.csr_matrix((data, (rows, cols)), shape=(count, count))
    free = ~fixed
    reduced_b = b[free]-(a@values)[free]
    meta = dict(a=a, b=b, fixed=fixed, free=free, values=values,
                boundary=boundary, volumes=volumes, source=q, n=n)
    return a[free][:, free].tocsr(), reduced_b, meta


def reconstruct(v, meta):
    full = meta["values"].copy()
    full[meta["free"]] = v
    return full.reshape(meta["n"]+1, meta["n"]+1)


def heat_balance(u, meta):
    """Robin flux plus Dirichlet reaction equals integrated volume source."""
    flat = u.ravel()
    fluxes = {side: float(np.sum(g*(flat[ids]-t)))
              for side, (ids, g, t) in meta["boundary"].items()}
    # At fixed nodes the additional reaction removes the residual energy.
    reaction = float(np.sum((meta["b"]-meta["a"]@flat)[meta["fixed"]]))
    supplied = float(np.sum(meta["source"]*meta["volumes"]))
    outward = sum(fluxes.values())+reaction
    return dict(source_W=supplied, robin_flux_W=fluxes,
                dirichlet_reaction_W=reaction, outward_W=outward,
                imbalance_W=outward-supplied,
                free_residual_sum_W=float(np.sum((meta["b"]-meta["a"]@flat)[meta["free"]])))


def robin_first_order(n, beta=1., order=1):
    """Manufactured exp(x+y/2), one-sided outward differences of order 1 or 2.

    -Delta u=-1.25 exp(x+y/2), dn u+beta*u=g. At corners sum both
    adjacent Robin equations. This full matrix is generally nonsymmetric.
    """
    validate_grid(n)
    if order not in (1, 2) or not np.isfinite(beta) or beta <= 0:
        raise ValueError("order must be 1 or 2 and beta finite and positive")
    h = 1/n
    y, x = np.mgrid[0:n+1, 0:n+1]/n
    exact = np.exp(x+.5*y)
    a = sparse.lil_matrix(((n+1)**2, (n+1)**2))
    b = np.zeros((n+1)**2)
    for j in range(n+1):
        for i in range(n+1):
            p = j*(n+1)+i
            faces = []
            if i == 0: faces.append((p+1, -1.))
            if i == n: faces.append((p-1, 1.))
            if j == 0: faces.append((p+n+1, -.5))
            if j == n: faces.append((p-n-1, .5))
            if faces:
                for neighbor, derivative in faces:
                    if order == 1:
                        a[p, p] += 1/h+beta
                        a[p, neighbor] -= 1/h
                    else:
                        a[p, p] += 1.5/h+beta
                        a[p, neighbor] -= 2/h
                        a[p, p+2*(neighbor-p)] += .5/h
                    b[p] += (derivative+beta)*exact[j, i]
            else:
                a[p, p] = 4/h**2
                for neighbor in (p-1, p+1, p-n-1, p+n+1):
                    a[p, neighbor] = -1/h**2
                b[p] = -1.25*exact[j, i]
    return a.tocsr(), b, exact


def manufactured_robin(n, beta=1.):
    validate_grid(n)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    y, x = np.mgrid[0:n+1, 0:n+1]/n
    exact = np.exp(x+.5*y)
    # k=1, alpha=beta; ambient=g/beta for this manufactured problem.
    ambient = dict(L=(1-1/beta)*exact[:, 0], R=(1+1/beta)*exact[:, -1],
                   B=(1-.5/beta)*exact[0], T=(1+.5/beta)*exact[-1])
    a, b, meta = robin_fv(n, -1.25*exact, ambient, alpha=beta)
    return a, b, meta, exact


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_plot(fig, folder, name):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(folder/f"{name}.{extension}", dpi=170)
    plt.close(fig)


def diagnostic_callback(a, reference, method, rows):
    """Measure true residuals and energy error, without changing the iteration."""
    def record(iteration, x, residual):
        error = x-reference
        rows.append(dict(method=method, iteration=iteration,
                         residual_inf=float(np.linalg.norm(residual, np.inf)),
                         residual_l2=float(np.linalg.norm(residual)),
                         energy_error=float(np.sqrt(max(0., error@(a@error)))),
                         max_residual_node=int(np.argmax(abs(residual)))))
    return record


def peak_summary(rows):
    summaries = []
    for method in dict.fromkeys(row["method"] for row in rows):
        data = [r for r in rows if r["method"] == method]
        ratios = np.array([r["residual_inf"] for r in data[1:]])/np.array(
            [r["residual_inf"] for r in data[:-1]])
        j = int(np.argmax(ratios))+1
        summaries.append(dict(method=method, increases=int(np.sum(ratios > 1+1e-12)),
                              largest_growth=float(ratios[j-1]), iteration=data[j]["iteration"],
                              energy_ratio=data[j]["energy_error"]/data[j-1]["energy_error"],
                              max_node_changed=data[j]["max_residual_node"] != data[j-1]["max_residual_node"]))
    return summaries


def run(out, n=32, tol=1e-10):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    plots = out/"plots"; plots.mkdir(exist_ok=True)
    a, b = dirichlet_system(n)
    start = perf_counter(); reference = spsolve(a, b); direct_time = perf_counter()-start
    omega = 2/(1+np.sin(np.pi/n))
    summary, histories = [], []
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in ("Jacobi", "GS", "SOR", "SD", "CG", "BiCGStab"):
        v, history, elapsed = solve(a, b, method, tol, omega, n=n)
        ratio = float(np.exp(np.mean(np.diff(np.log(history[-min(51, len(history)):])))))
        theory = {"Jacobi": np.cos(np.pi/n), "GS": np.cos(np.pi/n)**2,
                  "SOR": omega-1}.get(method, np.nan)
        summary.append(dict(method=method, N=n, iterations=len(history)-1,
                            seconds=elapsed, residual_inf=history[-1],
                            error_vs_direct_inf=float(np.max(abs(v-reference))),
                            rho_theory=theory, tail_residual_factor=ratio))
        histories.extend(dict(method=method, iteration=k, residual_inf=r)
                         for k, r in enumerate(history))
        ax.semilogy(np.arange(len(history)), history, label=METHOD_NAMES[method])
        print(f"{method}: {len(history)-1} iterations, r={history[-1]:.3e}", flush=True)
    direct_row = dict(method="LU", N=n, iterations=0,
                        seconds=direct_time, residual_inf=float(np.max(abs(b-a@reference))),
                        error_vs_direct_inf=0., rho_theory=np.nan, tail_residual_factor=np.nan)
    write_csv(out/"direct_reference.csv", [direct_row])
    write_csv(out/"methods.csv", summary); write_csv(out/"residuals.csv", histories)
    ax.axhline(tol, ls="--", c="black", label="Порог остановки")
    ax.set(xlabel="Итерация (включая завершающий полушаг BiCGStab)", ylabel=r"$\|b-Au\|_\infty$",
           title=f"Задача 16: N={n}, нулевое начальное приближение")
    ax.legend(ncol=2); ax.grid(alpha=.25)
    save_plot(fig, plots, "convergence")

    sweeps = []
    for w in sorted(set(np.r_[np.linspace(.5, 1.95, 21), omega, omega-.015, omega+.015])):
        _, hist, seconds = solve(a, b, "SOR", tol, float(w), n=n)
        sweeps.append(dict(omega=w, iterations=len(hist)-1, seconds=seconds,
                           residual_inf=hist[-1]))
    write_csv(out/"sor_sweep.csv", sweeps)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["omega"] for r in sweeps], [r["iterations"] for r in sweeps], "o-")
    ax.axvline(omega, c="black", ls="--", label=f"Теория: {omega:.5f}")
    ax.set(xlabel=r"$\omega$ (в условии $\tau$)", ylabel="Число итераций", title="Выбор параметра релаксации SOR")
    ax.legend(); ax.grid(alpha=.25); save_plot(fig, plots, "sor_parameter")
    print("SOR sweep complete", flush=True)

    grid_rows = []
    for grid in (16, 32, 64, 128):
        aa, bb = dirichlet_system(grid)
        uu = spsolve(aa, bb).reshape(grid-1, grid-1)
        y, x = np.mgrid[1:grid, 1:grid]/grid
        exact = exact_square(x, y)
        error = np.abs(uu-exact)
        core = (x >= .125)&(x <= .875)&(y >= .125)&(y <= .875)
        row = dict(N=grid, h_m=.1/grid, error_all_inf=error.max(),
                   error_rms=np.sqrt(np.mean(error**2)), error_core_inf=error[core].max(),
                   observed_core_order=np.nan, residual_inf=np.max(abs(bb-aa@uu.ravel())))
        if grid_rows:
            row["observed_core_order"] = np.log2(grid_rows[-1]["error_core_inf"]/row["error_core_inf"])
        grid_rows.append(row)
        if grid == 128:
            field = full_field(uu, grid)
            np.savez_compressed(out/"temperature.npz", u=field, T_C=100*field,
                                x_m=np.linspace(0, .1, grid+1), y_m=np.linspace(0, .1, grid+1))
    write_csv(out/"grid_convergence.csv", grid_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    hs = [r["h_m"] for r in grid_rows]
    for key, label in (("error_all_inf", "Максимум: все внутренние узлы"),
                       ("error_rms", "Среднеквадратичная: все внутренние узлы"), ("error_core_inf", "Максимум: фиксированный внутренний квадрат")):
        ax.loglog(hs, [r[key] for r in grid_rows], "o-", label=label)
    ax.set(xlabel="h, м", ylabel="Ошибка безразмерной температуры u", title="Сравнение с решением в виде ряда Фурье")
    ax.legend(fontsize=8); ax.grid(alpha=.25); save_plot(fig, plots, "grid_convergence")
    coordinates = np.linspace(0, .1, 129)
    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(coordinates, coordinates, 100*field, levels=np.linspace(100, 400, 25), cmap="inferno")
    lines = ax.contour(coordinates, coordinates, 100*field, levels=np.arange(125, 400, 25), colors="white", linewidths=.55)
    ax.clabel(lines, fontsize=7); fig.colorbar(contour, ax=ax, label="T, °C", ticks=np.arange(100, 401, 50))
    ax.set(xlabel="x, м", ylabel="y, м", title="Задача 16: температура и изотермы", aspect="equal")
    save_plot(fig, plots, "temperature")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for fraction in (.25, .5, .75):
        index = round(128*fraction)
        axes[0].plot(coordinates, 100*field[index, :], label=f"y={.1*fraction:.3f} м")
        axes[1].plot(coordinates, 100*field[:, index], label=f"x={.1*fraction:.3f} м")
    for ax, coordinate in zip(axes, ("x", "y")):
        ax.set(xlabel=f"{coordinate}, м", ylabel="T, °C"); ax.legend(); ax.grid(alpha=.25)
    save_plot(fig, plots, "profiles")

    robin_rows = []
    for grid in (8, 16, 32, 64):
        a1, b1, exact = robin_first_order(grid)
        error1 = np.max(abs(spsolve(a1, b1).reshape(exact.shape)-exact))
        a3, b3, _ = robin_first_order(grid, order=2)
        error3 = np.max(abs(spsolve(a3, b3).reshape(exact.shape)-exact))
        a2, b2, meta, exact = manufactured_robin(grid)
        error2 = np.max(abs(reconstruct(spsolve(a2, b2), meta)-exact))
        row = dict(N=grid, error_first_order=error1, error_second_order=error2,
                   error_one_sided_second=error3, p_first=np.nan, p_second=np.nan, p_one_sided=np.nan)
        if robin_rows:
            row["p_first"] = np.log2(robin_rows[-1]["error_first_order"]/error1)
            row["p_second"] = np.log2(robin_rows[-1]["error_second_order"]/error2)
            row["p_one_sided"] = np.log2(robin_rows[-1]["error_one_sided_second"]/error3)
        robin_rows.append(row)
    write_csv(out/"robin_accuracy.csv", robin_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("error_first_order", "Односторонняя, порядок 1"),
                       ("error_one_sided_second", "Односторонняя, порядок 2"),
                       ("error_second_order", "Фиктивный узел / баланс")):
        ax.loglog([1/r["N"] for r in robin_rows], [r[key] for r in robin_rows], "o-", label=label)
    ax.set(xlabel="h", ylabel="Максимальная ошибка по всем узлам", title="Условие Робина: проверка на точном решении")
    ax.legend(); ax.grid(alpha=.25); save_plot(fig, plots, "robin_accuracy")
    validation = []
    for beta in (.1, 1., 10.):
        previous = None
        for grid in (8, 16, 32, 64):
            av, bv, mv, exact = manufactured_robin(grid, beta)
            numerical = reconstruct(spsolve(av, bv), mv)
            defect = (av@exact.ravel()-bv).reshape(exact.shape)/mv["volumes"]
            boundary = np.ones_like(exact, dtype=bool)
            boundary[1:-1, 1:-1] = False
            error = np.max(abs(numerical-exact))
            validation.append(dict(beta=beta, N=grid, error_inf=error,
                                   order=np.nan if previous is None else np.log2(previous/error),
                                   boundary_defect_inf=np.max(abs(defect[boundary])),
                                   interior_defect_inf=np.max(abs(defect[~boundary]))))
            previous = error
    write_csv(out/"robin_validation.csv", validation)
    print("Grid and Robin accuracy studies complete", flush=True)

    room_n = 40
    y, x = np.mgrid[0:room_n+1, 0:room_n+1]/room_n
    source_mask = (x >= .7)&(x <= .9)&(y >= .35)&(y <= .65)
    w = np.ones(room_n+1); w[[0, -1]] = .5
    q = source_mask*200/np.sum(source_mask*np.outer(w, w)/room_n**2)
    fixed = (x == 0)&(y >= .4)&(y <= .6)
    aa, bb, meta = robin_fv(room_n, q, dict.fromkeys("LRBT", -10.),
                           dirichlet=(fixed, -10.))
    direct = spsolve(aa, bb)
    room_rows, room_histories, diagnostics = [], [], []
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in ("CG", "BiCGStab", "SOR"):
        vv, hist, elapsed = solve(aa, bb, method, tol=1e-9, omega=1.8,
                                 callback=diagnostic_callback(aa, direct, method, diagnostics))
        room_rows.append(dict(method=method, iterations=len(hist)-1, seconds=elapsed,
                              residual_inf=hist[-1], error_vs_direct_inf=np.max(abs(vv-direct))))
        room_histories.extend(dict(method=method, iteration=k, residual_inf=r) for k, r in enumerate(hist))
        ax.semilogy(range(len(hist)), hist, label=method)
    write_csv(out/"room_methods.csv", room_rows)
    write_csv(out/"room_residuals.csv", room_histories)
    write_csv(out/"room_diagnostics.csv", diagnostics)
    write_csv(out/"residual_peaks.csv", peak_summary(diagnostics))
    ax.set(xlabel="Итерация (включая завершающий полушаг BiCGStab)", ylabel="Абсолютная невязка, Вт", title="Модель с окном и источником")
    ax.legend(); ax.grid(alpha=.25); save_plot(fig, plots, "room_convergence")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for method in ("CG", "BiCGStab"):
        records = [r for r in diagnostics if r["method"] == method]
        steps = [r["iteration"] for r in records]
        for key, style in (("residual_inf", "-"), ("residual_l2", "--")):
            label = r"$\|r\|_\infty$" if key == "residual_inf" else r"$\|r\|_2$"
            axes[0].semilogy(steps, [r[key] for r in records], style, label=method+": "+label)
        axes[1].semilogy(steps, [r["energy_error"] for r in records], label=method)
    axes[0].set(xlabel="Итерация", ylabel="Невязка, Вт", title="Две нормы истинной невязки")
    axes[1].set(xlabel="Итерация", ylabel=r"$\|T^k-T_{LU}\|_A$", title="Энергетическая норма ошибки")
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    save_plot(fig, plots, "residual_diagnostics")
    room = reconstruct(direct, meta)
    balance = heat_balance(room, meta)
    balance.update(min_temperature_C=float(room.min()), max_temperature_C=float(room.max()))
    (out/"heat_balance.json").write_text(json.dumps(balance, indent=2)+"\n")
    np.savez_compressed(out/"room.npz", T_C=room, x_m=x, y_m=y, source_W_m3=q)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(x, y, room, levels=np.linspace(room.min(), room.max(), 25), cmap="inferno")
    fig.colorbar(im, ax=ax, label="T, °C")
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((.7, .35), .2, .3, fill=False, ec="cyan", ls="--", label="Источник 200 Вт"))
    ax.plot([0, 0], [.4, .6], c="cyan", lw=4, label="Окно: −10 °C", clip_on=False)
    ax.set(xlabel="x, м", ylabel="y, м", title="Температура: теплоотдача, окно и источник", aspect="equal")
    ax.legend(fontsize=8); save_plot(fig, plots, "room_temperature")
    settings = dict(N=n, tolerance=tol, omega_optimal=omega, L_m=.1,
                    T_scale_C=100, sides_LRBT=[1, 3, 4, 2],
                    room_N=room_n, room_k_W_mK=1., room_alpha_W_m2K=1., room_depth_m=1.,
                    room_tolerance_W=1e-9, room_omega=1.8,
                    grid_intervals=[16, 32, 64, 128], robin_intervals=[8, 16, 32, 64],
                    numpy_version=np.__version__, scipy_version=__import__("scipy").__version__,
                    matplotlib_version=matplotlib.__version__)
    (out/"settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    print(json.dumps(balance, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"results")
    parser.add_argument("--n", type=int, default=32, help="intervals for method comparison (>= 8)")
    parser.add_argument("--tol", type=float, default=1e-10)
    args = parser.parse_args()
    if args.n < 8 or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("--n must be >= 8 and --tol finite and positive")
    run(args.out, args.n, args.tol)
