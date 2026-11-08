#!/usr/bin/env python3
""" IV.12.3 (д, и), IV.12.4 (в), IV.12.6 (г).

    python src/lab3_nonlinear.py --out results --show
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ScalarFunction = Callable[[float], float]
VectorFunction = Callable[[np.ndarray], np.ndarray]

TOL = 1e-10
MAX_ITER = 800


@dataclass(frozen=True)
class ScalarCase:
    key: str
    title: str
    f: ScalarFunction
    derivative: ScalarFunction
    fixed_point: ScalarFunction
    bracket: tuple[float, float]
    display: str


@dataclass(frozen=True)
class SystemCase:
    key: str
    title: str
    f: VectorFunction
    jacobian: VectorFunction
    fixed_point: VectorFunction
    reduced_f: ScalarFunction
    reconstruct: Callable[[float], np.ndarray]
    bracket: tuple[float, float]
    initial: tuple[float, float]
    display: str


@dataclass
class Result:
    name: str
    point: float | np.ndarray
    residuals: list[float]
    converged: bool
    iterations: int
    note: str = ""

    @property
    def residual(self) -> float:
        return float(self.residuals[-1])


def scalar_problems() -> list[ScalarCase]:
    f_i = lambda x: x * x - np.exp(x / 5.0)
    df_i = lambda x: 2.0 * x - 0.2 * np.exp(x / 5.0)
    return [
        ScalarCase(
            "IV_12_3_d", "IV.12.3 (д): arctan(x-1) + 2x = 0",
            lambda x: np.arctan(x - 1.0) + 2.0 * x,
            lambda x: 1.0 / (1.0 + (x - 1.0) ** 2) + 2.0,
            lambda x: -0.5 * np.arctan(x - 1.0),
            (0.0, 1.0), "Единственный корень",
        ),
        ScalarCase(
            "IV_12_3_i_neg", "IV.12.3 (и): x² - exp(x/5) = 0",
            f_i, df_i, lambda x: -np.exp(x / 10.0),
            (-1.0, 0.0), "Отрицательный корень",
        ),
        ScalarCase(
            "IV_12_3_i_small", "IV.12.3 (и): x² - exp(x/5) = 0",
            f_i, df_i, lambda x: np.exp(x / 10.0),
            (1.0, 2.0), "Малый положительный корень",
        ),
        ScalarCase(
            "IV_12_3_i_large", "IV.12.3 (и): x² - exp(x/5) = 0",
            f_i, df_i, lambda x: 10.0 * np.log(x),
            (35.0, 40.0), "Большой положительный корень",
        ),
    ]


def system_problems() -> list[SystemCase]:
    def f_124(p):
        x, y = p
        return np.array([np.cos(x - 1.0) + y - 0.5,
                         x - np.cos(y) - 3.0], dtype=float)

    def j_124(p):
        x, y = p
        return np.array([[-np.sin(x - 1.0), 1.0],
                         [1.0, np.sin(y)]], dtype=float)

    def t_124(p):
        x, y = p
        return np.array([3.0 + np.cos(y),
                         0.5 - np.cos(x - 1.0)], dtype=float)

    def f_126(p):
        x, y = p
        return np.array([np.sin(x) - y - 1.32,
                         np.cos(y) - x + 0.85], dtype=float)

    def j_126(p):
        x, y = p
        return np.array([[np.cos(x), -1.0],
                         [-1.0, -np.sin(y)]], dtype=float)

    def t_126(p):
        x, y = p
        return np.array([0.85 + np.cos(y),
                         np.sin(x) - 1.32], dtype=float)

    return [
        SystemCase(
            "IV_12_4_v", "IV.12.4 (в): пересечение двух кривых",
            f_124, j_124, t_124,
            lambda x: x - 3.0 - np.cos(0.5 - np.cos(x - 1.0)),
            lambda x: np.array([x, 0.5 - np.cos(x - 1.0)]),
            (3.0, 4.0), (3.5, 1.2),
            "cos(x-1)+y=0.5; x-cos(y)=3",
        ),
        SystemCase(
            "IV_12_6_g", "IV.12.6 (г): пересечение двух кривых",
            f_126, j_126, t_126,
            lambda x: x - 0.85 - np.cos(np.sin(x) - 1.32),
            lambda x: np.array([x, np.sin(x) - 1.32]),
            (0.85, 1.85), (1.4, -0.4),
            "sin(x)-y=1.32; cos(y)-x=-0.85",
        ),
    ]


def residual(f, x) -> float:
    return float(np.linalg.norm(np.atleast_1d(f(x))))


def bisection(f: ScalarFunction, bracket: tuple[float, float],
              *, tol: float = TOL, max_iter: int = MAX_ITER,
              lift: Callable[[float], float | np.ndarray] | None = None,
              original_f=None) -> Result:

    a, b = map(float, bracket)
    fa, fb = float(f(a)), float(f(b))
    if not np.isfinite([fa, fb]).all() or fa * fb > 0:
        raise ValueError(f"Нужен корректный интервал смены знака: [{a}, {b}]")

    lift = lift or (lambda x: float(x))
    original_f = original_f or f
    if fa == 0.0:
        return Result("Половинное деление", lift(a),
                      [residual(original_f, lift(a))], True, 0)
    if fb == 0.0:
        return Result("Половинное деление", lift(b),
                      [residual(original_f, lift(b))], True, 0)
    mid = (a + b) / 2.0
    hist = [residual(original_f, lift(mid))]
    if hist[-1] <= tol:
        return Result("Половинное деление", lift(mid), hist, True, 0)

    for k in range(1, max_iter + 1):
        fm = float(f(mid))
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
        mid = (a + b) / 2.0
        hist.append(residual(original_f, lift(mid)))
        if hist[-1] <= tol:
            return Result("Половинное деление", lift(mid), hist, True, k)
        if mid == a or mid == b:
            return Result("Половинное деление", lift(mid), hist,
                          False, k, "Машинная точность предельного интервала")
    return Result("Половинное деление", lift(mid), hist, False, max_iter,
                  "Превышен лимит итераций")


def simple_iteration_scalar(case: ScalarCase, *, tol=TOL, max_iter=MAX_ITER) -> Result:
    x = sum(case.bracket) / 2.0
    hist = [residual(case.f, x)]
    for k in range(1, max_iter + 1):
        next_x = float(case.fixed_point(x))
        if not np.isfinite(next_x):
            return Result("Простая итерация", x, hist, False, k - 1,
                          "Выход из области определения")
        x = next_x
        hist.append(residual(case.f, x))
        if hist[-1] <= tol:
            return Result("Простая итерация", x, hist, True, k)
        if abs(x) > 1e8 or hist[-1] > 1e16:
            return Result("Простая итерация", x, hist, False, k, "Расходимость")
    return Result("Простая итерация", x, hist, False, max_iter,
                  "Превышен лимит итераций")


def newton_scalar(case: ScalarCase, *, modified=False,
                  tol=TOL, max_iter=MAX_ITER) -> Result:
    x = sum(case.bracket) / 2.0
    label = "Модифицированный Ньютон" if modified else "Ньютон"
    df0 = float(case.derivative(x))
    hist = [residual(case.f, x)]
    for k in range(1, max_iter + 1):
        dfx = df0 if modified else float(case.derivative(x))
        if abs(dfx) < 1e-14:
            return Result(label, x, hist, False, k - 1, "Слишком малая производная")
        x -= float(case.f(x)) / dfx
        if not np.isfinite(x):
            return Result(label, x, hist, False, k, "Выход в бесконечность")
        hist.append(residual(case.f, x))
        if hist[-1] <= tol:
            return Result(label, x, hist, True, k)
    return Result(label, x, hist, False, max_iter, "Превышен лимит итераций")


def variational_scalar(case: ScalarCase, *, tol=TOL, max_iter=MAX_ITER) -> Result:
    x = sum(case.bracket) / 2.0
    hist = [residual(case.f, x)]
    for k in range(1, max_iter + 1):
        fx = float(case.f(x))
        grad = fx * float(case.derivative(x))
        phi = 0.5 * fx * fx
        if not np.isfinite(grad):
            return Result("Вариационный метод", x, hist, False, k - 1,
                          "Некорректное значение градиента")
        alpha = 0.6 / (1.0 + float(case.derivative(x)) ** 2)
        accepted = False
        for _ in range(55):
            trial = x - alpha * grad
            ft = float(case.f(trial))
            if np.isfinite(ft) and 0.5 * ft * ft <= phi - 1e-4 * alpha * grad * grad:
                x = trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return Result("Вариационный метод", x, hist, False, k - 1,
                          "Линейный поиск не дал убывания функционала")
        hist.append(residual(case.f, x))
        if hist[-1] <= tol:
            return Result("Вариационный метод", x, hist, True, k)
    return Result("Вариационный метод", x, hist, False, max_iter,
                  "Превышен лимит итераций")


def simple_iteration_system(case: SystemCase, *, tol=TOL,
                            max_iter=MAX_ITER) -> Result:
    x = np.asarray(case.initial, dtype=float)
    hist = [residual(case.f, x)]
    for k in range(1, max_iter + 1):
        x = np.asarray(case.fixed_point(x), dtype=float)
        if not np.isfinite(x).all():
            return Result("Простая итерация", x, hist, False, k, "Некорректное x")
        hist.append(residual(case.f, x))
        if hist[-1] <= tol:
            return Result("Простая итерация", x, hist, True, k)
        if np.linalg.norm(x) > 1e8:
            return Result("Простая итерация", x, hist, False, k, "Расходимость")
    return Result("Простая итерация", x, hist, False, max_iter,
                  "Превышен лимит итераций")


def newton_system(case: SystemCase, *, modified=False,
                  tol=TOL, max_iter=MAX_ITER) -> Result:
    x = np.asarray(case.initial, dtype=float)
    label = "Модифицированный Ньютон" if modified else "Ньютон"
    jac0 = case.jacobian(x)
    hist = [residual(case.f, x)]
    for k in range(1, max_iter + 1):
        jac = jac0 if modified else case.jacobian(x)
        try:
            delta = np.linalg.solve(jac, -case.f(x))
        except np.linalg.LinAlgError:
            return Result(label, x, hist, False, k - 1, "Вырожденный якобиан")
        x = x + delta
        if not np.isfinite(x).all():
            return Result(label, x, hist, False, k, "Выход в бесконечность")
        hist.append(residual(case.f, x))
        if hist[-1] <= tol:
            return Result(label, x, hist, True, k)
    return Result(label, x, hist, False, max_iter, "Превышен лимит итераций")


def variational_system(case: SystemCase, *, tol=TOL,
                       max_iter=MAX_ITER) -> Result:
    x = np.asarray(case.initial, dtype=float)
    hist = [residual(case.f, x)]
    for k in range(1, max_iter + 1):
        r = case.f(x)
        jac = case.jacobian(x)
        grad = jac.T @ r
        phi = 0.5 * float(r @ r)
        alpha = 0.8 / max(1.0, float(np.linalg.norm(jac, ord='fro') ** 2))
        accepted = False
        for _ in range(55):
            trial = x - alpha * grad
            rt = case.f(trial)
            if np.isfinite(rt).all() and 0.5 * (rt @ rt) <= phi - 1e-4 * alpha * (grad @ grad):
                x = trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return Result("Вариационный метод", x, hist, False, k - 1,
                          "Линейный поиск не дал убывания функционала")
        hist.append(residual(case.f, x))
        if hist[-1] <= tol:
            return Result("Вариационный метод", x, hist, True, k)
    return Result("Вариационный метод", x, hist, False, max_iter,
                  "Превышен лимит итераций")


def run_scalar(case: ScalarCase):
    return [
        bisection(case.f, case.bracket),
        simple_iteration_scalar(case),
        newton_scalar(case),
        newton_scalar(case, modified=True),
        variational_scalar(case),
    ]


def run_system(case: SystemCase):
    return [
        bisection(case.reduced_f, case.bracket,
                  lift=case.reconstruct, original_f=case.f),
        simple_iteration_system(case),
        newton_system(case),
        newton_system(case, modified=True),
        variational_system(case),
    ]


def verify_problem_definitions() -> None:
    for case in system_problems():
        p = np.asarray(case.initial, dtype=float)
        step = 1e-6
        J_fd = np.column_stack([(case.f(p + np.eye(2)[j] * step)
                                - case.f(p - np.eye(2)[j] * step)) / (2 * step)
                                for j in range(2)])
        if not np.allclose(case.jacobian(p), J_fd, rtol=1e-7, atol=1e-9):
            raise AssertionError(f"Неправильно выписан якобиан: {case.key}")
        for x in np.linspace(*case.bracket, 7):
            point = case.reconstruct(x)
            # Две записи одного уравнения должны иметь равную норму невязки
            if not np.isclose(residual(case.f, point), abs(case.reduced_f(x)), atol=1e-10):
                raise AssertionError("Ошибка исключения переменной")


def scan_sign_changes(f, a: float, b: float, divisions: int=10000):
    xs = np.linspace(a, b, divisions + 1)
    with np.errstate(over='ignore', invalid='ignore'):
        ys = f(xs)
    out = []
    for i in range(divisions):
        if np.isfinite(ys[i]) and np.isfinite(ys[i+1]) and ys[i]*ys[i+1] < 0:
            out.append((float(xs[i]), float(xs[i+1])))
    return out


def configure_plots():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 125, "savefig.dpi": 210,
        "svg.fonttype": "none",
    })


def plot_histories(case, results: list[Result], out: Path):
    fig, ax = plt.subplots(figsize=(9.3, 5.6))
    markers = ['o', 's', '^', 'D', 'x']
    for result, marker in zip(results, markers):
        r = np.asarray(result.residuals)
        r = np.where(r > 0, r, np.nan)
        every = max(1, (len(r) - 1)//13)
        ax.semilogy(np.arange(len(r)), r, label=f"{result.name} ({result.iterations})",
                    linewidth=1.9, marker=marker, markersize=4.1, markevery=every)
    ax.axhline(TOL, linestyle='--', color='black', linewidth=1.3,
               alpha=0.6, label=r"Допуск $10^{-10}$")
    ax.set_xlabel('Номер итерации k')
    ax.set_ylabel(r'Норма невязки $|f(x_k)|$ или $\|F(z_k)\|_2$')
    ax.set_title(f'{case.title}\n{case.display if isinstance(case, ScalarCase) else case.display}')
    ax.legend(loc='best', fontsize=8.7, framealpha=.9)
    ax.grid(which='major', alpha=.32)
    ax.grid(which='minor', alpha=.13)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix('.svg'))
    return fig


def plot_scalar_equations(cases, results_by_case, output: Path):
    fig, axs = plt.subplots(1, 3, figsize=(14.4, 4.6))
    d = cases[0]
    xs = np.linspace(-0.15, 1.1, 650)
    axs[0].plot(xs, d.f(xs), color='#285999', linewidth=2.3)
    axs[0].axhline(0, color='black', alpha=.48)
    axs[0].scatter([float(results_by_case[d.key][0].point)], [0],
                   color='#b3432f', zorder=4, s=40)
    axs[0].axvspan(*d.bracket, color='#dbe8f6', alpha=.55)
    axs[0].set(title='IV.12.3 (д): один корень', xlabel='x', ylabel='f(x)')
    axs[0].grid(alpha=.25)

    i = cases[1]
    xnear = np.linspace(-1.5, 2.8, 700)
    axs[1].plot(xnear, i.f(xnear), color='#285999', linewidth=2.3)
    axs[1].axhline(0, color='black', alpha=.48)
    axs[1].axvspan(-1,0,color='#dbe8f6',alpha=.55)
    axs[1].axvspan(1,2,color='#dbe8f6',alpha=.55)
    for index in (1,2):
        root = float(results_by_case[cases[index].key][0].point)
        axs[1].scatter([root],[0], color='#b3432f', zorder=4,s=40)
    axs[1].set(title='IV.12.3 (и): два близких корня', xlabel='x', ylabel='f(x)')
    axs[1].grid(alpha=.25)

    xfar = np.linspace(33,40,700)
    axs[2].plot(xfar,i.f(xfar),color='#285999',linewidth=2.3)
    axs[2].axhline(0,color='black',alpha=.48)
    axs[2].axvspan(35,40,color='#dbe8f6',alpha=.55)
    axs[2].scatter([float(results_by_case[cases[3].key][0].point)],
                   [0],color='#b3432f',zorder=4,s=40)
    axs[2].set(title='IV.12.3 (и): большой корень', xlabel='x', ylabel='f(x)')
    axs[2].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output)
    return fig


def plot_system_geometry(case: SystemCase, results: list[Result], output: Path):
    if case.key.endswith('4_v'):
        xx, yy = np.meshgrid(np.linspace(3.0,3.9,300),np.linspace(.8,1.55,300))
    else:
        xx, yy = np.meshgrid(np.linspace(1.48,1.95,300),np.linspace(-.62,-.2,300))
    z1,z2=case.f(np.stack([xx,yy]))
    fig,ax=plt.subplots(figsize=(7.5,5.3))
    a=ax.contour(xx,yy,z1,levels=[0],colors=['#2d6a99'],linewidths=[2.4])
    b=ax.contour(xx,yy,z2,levels=[0],colors=['#ba5833'],linewidths=[2.4])
    from matplotlib.lines import Line2D
    legend_curves = [Line2D([0], [0], color='#2d6a99', lw=2.4, label='Первое уравнение: F₁=0'),
                     Line2D([0], [0], color='#ba5833', lw=2.4, label='Второе уравнение: F₂=0')]
    root=np.asarray(results[0].point)
    ax.scatter(*root,s=85,color='black',marker='*',zorder=6)
    legend_curves.append(Line2D([0], [0], color='black', linestyle='None', marker='*',
                                markersize=10, label=f'Пересечение ({root[0]:.6f}; {root[1]:.6f})'))
    ax.set(xlabel='x',ylabel='y',title=case.title)
    ax.grid(alpha=.18)
    ax.legend(handles=legend_curves,fontsize=9,loc='best')
    fig.tight_layout()
    fig.savefig(output)
    return fig


def write_csv_rows(out: Path, name: str, rows: list[dict]):
    with (out / name).open('w', newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(output: Path, show: bool = False):
    # TOL determines line on plots; CLI --tol currently intentionally not offered
    verify_problem_definitions()
    configure_plots()
    plot_dir=output/'plots'
    plot_dir.mkdir(parents=True,exist_ok=True)
    scalar_cases = scalar_problems()
    system_cases = system_problems()
    all_cases = scalar_cases + system_cases
    by_case = {}
    summary_rows = []
    hist_rows = []
    scalar_roots=[]
    system_roots=[]
    failures=[]

    for case in all_cases:
        methods = run_scalar(case) if isinstance(case,ScalarCase) else run_system(case)
        by_case[case.key]=methods
        reference=np.atleast_1d(methods[0].point)
        print('\n'+ '='*81)
        print(case.title, '|', case.display)
        print('Reference (bisection after elimination for systems):',reference)
        print(f'{"Метод":<30} {"Итерации":>10} {"||невязка||":>15}  Статус')
        for result in methods:
            x=np.atleast_1d(result.point)
            true_residual = residual(case.f, result.point)
            delta=float(np.linalg.norm(x-reference))
            print(f'{result.name:<30} {result.iterations:>10} {true_residual:>15.4e}  {"OK" if result.converged else "FAIL"}')
            if not result.converged or true_residual>1.01*TOL:
                failures.append((case.key,result.name,result.note,true_residual))
            summary_rows.append(dict(task=case.key,root=case.display,method=result.name,
                                     iterations=result.iterations,final_residual=f'{true_residual:.16e}',
                                     deviation_from_reference=f'{delta:.16e}',
                                     success=result.converged,note=result.note))
            for it, r in enumerate(result.residuals):
                hist_rows.append(dict(task=case.key,root=case.display,method=result.name,
                                      iteration=it,residual=f'{r:.16e}'))
        if isinstance(case,ScalarCase):
            scalar_roots.append(dict(task=case.key,root=case.display,
                                     value=f'{float(reference[0]):.14f}',f_at_root=f'{residual(case.f,float(reference[0])):.3e}'))
        else:
            system_roots.append(dict(task=case.key,x=f'{reference[0]:.14f}',
                                     y=f'{reference[1]:.14f}',norm_f=f'{residual(case.f,reference):.3e}'))
        plot_histories(case,methods,plot_dir/f'{case.key}_convergence.png')

    write_csv_rows(output,'summary.csv',summary_rows)
    write_csv_rows(output,'histories.csv',hist_rows)
    write_csv_rows(output,'scalar_roots.csv',scalar_roots)
    write_csv_rows(output,'system_roots.csv',system_roots)
    plot_scalar_equations(scalar_cases,by_case,plot_dir/'scalar_roots.png')
    for case in system_cases:
        plot_system_geometry(case,by_case[case.key],plot_dir/f'{case.key}_geometry.png')

    # Overview: 4 separate mathematical problems (for (и) choose the hardest root)
    picked = [scalar_cases[0],scalar_cases[3],system_cases[0],system_cases[1]]
    fig,axs=plt.subplots(2,2,figsize=(13.8,9.5))
    for ax,case in zip(axs.flat,picked):
        for result in by_case[case.key]:
            data=np.asarray(result.residuals)
            ax.semilogy(np.arange(len(data)),np.where(data>0,data,np.nan),lw=1.7,
                        label=result.name)
        ax.axhline(TOL, color='black',ls='--',alpha=.55)
        ax.set(title=case.title, xlabel='Итерации',ylabel='Норма невязки (log)')
        ax.grid(alpha=.27,which='both')
        ax.legend(fontsize=7.8)
    fig.tight_layout()
    fig.savefig(plot_dir/'overview.png',dpi=210)

    if failures:
        for failure in failures:
            print('FAILED: ', failure)
        raise RuntimeError(f'Не достигли точности {len(failures)} запусков')
    print('\nВсе 30 вычислительных запусков достигли заданного критерия.')
    if show:
        plt.show()
    else:
        plt.close('all')
    return by_case


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent.parent/'results',
                        help='Папка для графиков и таблиц')
    parser.add_argument('--show',action='store_true',help='Показать графики в окнах')
    args=parser.parse_args()
    main(args.out,args.show)
