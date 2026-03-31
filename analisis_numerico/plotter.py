"""
plotter.py — Visualizaciones con Matplotlib para todos los métodos
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sympy as sp
import io
from typing import List, Optional


plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 13,
})

COLORS = {
    'primary': '#5B3FD4',
    'secondary': '#1D9E75',
    'accent': '#D85A30',
    'warn': '#BA7517',
    'gray': '#888780',
}


def fig_to_bytes(fig) -> bytes:
    """Convierte figura matplotlib a bytes PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def eval_expr(expr_str: str, x_arr: np.ndarray) -> np.ndarray:
    """Evalúa expresión simbólica en array de x."""
    x = sp.Symbol('x')
    expr = sp.sympify(expr_str)
    f = sp.lambdify(x, expr, modules=['numpy'])
    return np.array(f(x_arr), dtype=float)


# ─── RAÍCES ──────────────────────────────────────────────────────────────────

def plot_root_method(
    expr_str: str,
    iterations: list,
    root: float,
    method_name: str,
    interval: Optional[tuple] = None
) -> bytes:
    """Gráfica para métodos de búsqueda de raíces."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel izquierdo: función y raíz ---
    if interval:
        margin = abs(interval[1] - interval[0]) * 0.3
        x_min = interval[0] - margin
        x_max = interval[1] + margin
    else:
        x_min = root - 3
        x_max = root + 3

    x_plot = np.linspace(x_min, x_max, 500)
    try:
        y_plot = eval_expr(expr_str, x_plot)
        # Clip para evitar asíntotas
        y_plot = np.clip(y_plot, -50, 50)
    except Exception:
        y_plot = np.zeros_like(x_plot)

    ax1.plot(x_plot, y_plot, color=COLORS['primary'], linewidth=2, label=f'f(x) = {expr_str}')
    ax1.axhline(y=0, color=COLORS['gray'], linewidth=0.8, linestyle='--')
    ax1.axvline(x=root, color=COLORS['accent'], linewidth=1.5, linestyle=':', alpha=0.7)
    ax1.scatter([root], [0], color=COLORS['accent'], zorder=5, s=100, label=f'Raíz ≈ {root:.6f}')

    # Marcar iteraciones (últimas 5)
    if iterations and 'x_{n+1}' in iterations[0]:
        last_iters = iterations[-min(5, len(iterations)):]
        for it in last_iters:
            xi = it.get('x_{n+1}', it.get('x_n', 0))
            try:
                yi = eval_expr(expr_str, np.array([xi]))[0]
                ax1.scatter([xi], [yi], color=COLORS['warn'], zorder=4, s=30, alpha=0.6)
            except Exception:
                pass

    ax1.set_title(f'{method_name} — Función y raíz')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend(fontsize=9)
    ax1.set_ylim(-max(abs(y_plot[np.isfinite(y_plot)])) * 1.1 if np.any(np.isfinite(y_plot)) else -10,
                  max(abs(y_plot[np.isfinite(y_plot)])) * 1.1 if np.any(np.isfinite(y_plot)) else 10)

    # --- Panel derecho: convergencia del error ---
    if iterations and 'error' in iterations[0]:
        errors = [it['error'] for it in iterations if it['error'] > 0]
        iters_num = list(range(1, len(errors) + 1))

        ax2.semilogy(iters_num, errors, color=COLORS['secondary'], linewidth=2, marker='o', markersize=4)
        ax2.set_title('Convergencia del error')
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Error (escala log)')
        ax2.set_xticks(range(1, len(errors) + 1, max(1, len(errors) // 10)))
    else:
        ax2.text(0.5, 0.5, 'Sin datos\nde convergencia', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12, color=COLORS['gray'])

    fig.suptitle(f'Análisis Numérico — {method_name}', fontsize=14, y=1.01)
    fig.tight_layout()
    return fig_to_bytes(fig)


# ─── INTEGRALES ───────────────────────────────────────────────────────────────

def plot_integral(
    expr_str: str,
    a: float,
    b: float,
    method: str,
    n: int,
    result: float
) -> bytes:
    """Visualiza el área bajo la curva con el método de integración."""
    fig, ax = plt.subplots(figsize=(10, 5))

    margin = abs(b - a) * 0.15
    x_plot = np.linspace(a - margin, b + margin, 600)
    try:
        y_plot = eval_expr(expr_str, x_plot)
        y_plot = np.clip(y_plot, -1e6, 1e6)
    except Exception:
        y_plot = np.zeros_like(x_plot)

    ax.plot(x_plot, y_plot, color=COLORS['primary'], linewidth=2.5, zorder=3, label=f'f(x) = {expr_str}')

    # Dibujar rectángulos/trapecios
    x_parts = np.linspace(a, b, n + 1)
    try:
        y_parts = eval_expr(expr_str, x_parts)
    except Exception:
        y_parts = np.zeros_like(x_parts)

    display_n = min(n, 50)
    step = max(1, n // display_n)
    x_display = x_parts[::step]
    y_display = y_parts[::step]

    for i in range(len(x_display) - 1):
        if method.lower() in ('trapecio', 'trapezoidal'):
            poly_x = [x_display[i], x_display[i], x_display[i + 1], x_display[i + 1]]
            poly_y = [0, y_display[i], y_display[i + 1], 0]
            ax.fill(poly_x, poly_y, alpha=0.25, color=COLORS['secondary'], zorder=2)
            ax.plot([x_display[i], x_display[i + 1]], [y_display[i], y_display[i + 1]],
                    color=COLORS['secondary'], linewidth=0.5, alpha=0.5)
        else:  # Simpson
            mid = (x_display[i] + x_display[i + 1]) / 2
            ax.fill_between([x_display[i], x_display[i + 1]], 0,
                             np.clip(eval_expr(expr_str, np.linspace(x_display[i], x_display[i + 1], 10)), -1e6, 1e6),
                             alpha=0.2, color=COLORS['secondary'])

    ax.axhline(y=0, color=COLORS['gray'], linewidth=0.8)
    ax.axvline(x=a, color=COLORS['accent'], linewidth=1, linestyle='--', alpha=0.7)
    ax.axvline(x=b, color=COLORS['accent'], linewidth=1, linestyle='--', alpha=0.7)

    ax.set_title(f'Integración por {method}  |  ∫f(x)dx ≈ {result:.6f}  (n={n})')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig_to_bytes(fig)


# ─── INTERPOLACIÓN ────────────────────────────────────────────────────────────

def plot_interpolation(
    x_data: List[float],
    y_data: List[float],
    polynomial_str: str,
    method_name: str,
    x_eval: Optional[float] = None
) -> bytes:
    """Visualiza el polinomio de interpolación y los puntos de datos."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x_arr = np.array(x_data)
    y_arr = np.array(y_data)

    margin = (max(x_arr) - min(x_arr)) * 0.2 + 0.5
    x_plot = np.linspace(min(x_arr) - margin, max(x_arr) + margin, 500)

    try:
        y_plot = eval_expr(polynomial_str, x_plot)
        y_plot = np.clip(y_plot, min(y_arr) * 2 - 1, max(y_arr) * 2 + 1)
        ax.plot(x_plot, y_plot, color=COLORS['primary'], linewidth=2, label='Polinomio P(x)', zorder=2)
    except Exception:
        pass

    ax.scatter(x_arr, y_arr, color=COLORS['accent'], zorder=5, s=80, label='Puntos conocidos', edgecolors='white', linewidth=1)

    if x_eval is not None:
        try:
            y_eval = eval_expr(polynomial_str, np.array([x_eval]))[0]
            ax.scatter([x_eval], [y_eval], color=COLORS['secondary'], zorder=6, s=120,
                       marker='*', label=f'P({x_eval}) = {y_eval:.4f}')
            ax.axvline(x=x_eval, color=COLORS['secondary'], linewidth=1, linestyle=':', alpha=0.6)
        except Exception:
            pass

    ax.set_title(f'Interpolación {method_name}  |  Grado {len(x_data) - 1}')
    ax.set_xlabel('x')
    ax.set_ylabel('P(x)')
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig_to_bytes(fig)


# ─── SUMATORIAS ───────────────────────────────────────────────────────────────

def plot_series(table_rows: list, title: str = "Sumatoria") -> bytes:
    """Visualiza los términos y la suma parcial de una serie."""
    if not table_rows:
        return b""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ns = [r['n'] for r in table_rows]
    terms = [r['término'] for r in table_rows]
    partial = [r['suma parcial'] for r in table_rows]

    ax1.bar(ns, terms, color=COLORS['primary'], alpha=0.75, edgecolor='white', linewidth=0.5)
    ax1.set_title('Términos de la serie')
    ax1.set_xlabel('n')
    ax1.set_ylabel('a_n')
    ax1.axhline(y=0, color=COLORS['gray'], linewidth=0.6)

    ax2.plot(ns, partial, color=COLORS['secondary'], linewidth=2, marker='o', markersize=4)
    ax2.set_title('Sumas parciales')
    ax2.set_xlabel('n')
    ax2.set_ylabel('S_n')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig_to_bytes(fig)
