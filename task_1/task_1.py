import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FunctionCase:

    title: str
    filename: str
    f: Callable
    df: Callable


FUNCTIONS = (
    FunctionCase(
        r"$f(x)=\sin(x^2)$",
        "01_sin_x2.png",
        lambda x: np.sin(x**2),     # func itself
        lambda x: 2 * x * np.cos(x**2), #its derivative
    ),
    FunctionCase(
        r"$f(x)=\cos(\sin x)$",
        "02_cos_sin_x.png",
        lambda x: np.cos(np.sin(x)),
        lambda x: -np.sin(np.sin(x)) * np.cos(x),
    ),
    FunctionCase(
        r"$f(x)=\exp(\sin(\cos x))$",
        "03_exp_sin_cos_x.png",
        lambda x: np.exp(np.sin(np.cos(x))),
        lambda x: -np.exp(np.sin(np.cos(x))) * np.cos(np.cos(x)) * np.sin(x),
    ),
    FunctionCase(
        r"$f(x)=\ln(x+3)$",
        "04_ln_x_plus_3.png",
        lambda x: np.log(x + 3),
        lambda x: 1 / (x + 3),
    ),
    FunctionCase(
        r"$f(x)=\sqrt{x+3}$",
        "05_sqrt_x_plus_3.png",
        lambda x: np.sqrt(x + 3),
        lambda x: 1 / (2 * np.sqrt(x + 3)),
    ),
)

METHOD_NAMES = (
    "1. Правая разность, O(h)",
    "2. Левая разность, O(h)",
    "3. Центральная, O(h²)",
    "4. Экстраполяция, O(h⁴)",
    "5. Шестой порядок, O(h⁶)",
)
MARKERS = ("o", "s", "^", "D", "v")


def numerical_derivatives(f: Callable, x: float, h: np.ndarray) -> tuple:
    f0 = f(x)
    fp1, fm1 = f(x + h), f(x - h)
    fp2, fm2 = f(x + 2 * h), f(x - 2 * h)
    fp3, fm3 = f(x + 3 * h), f(x - 3 * h)

    # Симметричная разность с шагами h, 2h и 3h.
    c1 = (fp1 - fm1) / (2 * h)
    c2 = (fp2 - fm2) / (4 * h)
    c3 = (fp3 - fm3) / (6 * h)

    return (
        (fp1 - f0) / h,                 # Метод 1: O(h)
        (f0 - fm1) / h,                 # Метод 2: O(h)
        c1,                             # Метод 3: O(h²)
        (4 / 3) * c1 - (1 / 3) * c2,   # Метод 4: O(h⁴)
        (3 / 2) * c1 - (3 / 5) * c2
        + (1 / 10) * c3,               # Метод 5: O(h⁶)
    )


def build_figure(case: FunctionCase, x0: float, h: np.ndarray, output: Path) -> None:
    exact = case.df(x0)
    approximations = numerical_derivatives(case.f, x0, h)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    print(f"\n{case.title}: аналитическая производная = {exact:.12g}")

    for name, marker, approximate in zip(METHOD_NAMES, MARKERS, approximations):
        error = np.abs(approximate - exact)

        plotted_error = np.where(error > 0, error, np.nan)
        ax.loglog(h, plotted_error, marker=marker, markersize=4,
                  linewidth=1.7, label=name)

        best = int(np.argmin(error))
        print(f"  {name}: min ошибка = {error[best]:.3e}, h = {h[best]:.3e}")

    ax.set_title(f"{case.title}    ($x_0={x0:g}$)", fontsize=15, pad=14)
    ax.set_xlabel(r"Шаг численного дифференцирования $h$", fontsize=12)
    ax.set_ylabel(r"Абсолютная ошибка $|D_h - f'(x_0)|$", fontsize=12)
    ax.grid(True, which="major", alpha=0.45)
    ax.grid(True, which="minor", alpha=0.15)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / case.filename, dpi=200, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=1.1,
                        help="точка дифференцирования (по умолчанию 1.1)")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"),
                        help="папка для пяти PNG (по умолчанию plots)")
    parser.add_argument("--no-show", action="store_true",
                        help="не открывать графические окна")
    args = parser.parse_args()

    # При h_max=1 метод 5 использует x-3h; для log/sqrt нужно x > 0.
    if not np.isfinite(args.x) or args.x <= 0:
        parser.error("Для всех пяти функций и всех шагов необходимо x > 0.")

    n = np.arange(1, 22)
    h = 2.0 / (2.0**n)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for case in FUNCTIONS:
        build_figure(case, args.x, h, args.output_dir)

    print(f"\nСохранено 5 графиков в папку: {args.output_dir.resolve()}")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
