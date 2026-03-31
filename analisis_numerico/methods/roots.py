"""
methods/roots.py — Bisección, Punto Fijo, Newton-Raphson
"""
import numpy as np
import sympy as sp
from typing import Callable, Optional
import pandas as pd


def safe_eval(expr_str: str, x_val: float) -> float:
    """Evalúa una expresión simbólica de forma segura."""
    x = sp.Symbol('x')
    try:
        expr = sp.sympify(expr_str)
        return float(expr.subs(x, x_val))
    except Exception as e:
        raise ValueError(f"No se pudo evaluar f({x_val}): {e}")


def get_derivative(expr_str: str) -> str:
    """Calcula la derivada simbólica de una expresión."""
    x = sp.Symbol('x')
    expr = sp.sympify(expr_str)
    deriv = sp.diff(expr, x)
    return str(deriv)


# ─── BISECCIÓN ───────────────────────────────────────────────────────────────

def biseccion(
    expr_str: str,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> dict:
    """
    Método de Bisección para encontrar raíces.

    Parámetros:
        expr_str : expresión de f(x) como string (ej: 'x**3 - x - 2')
        a, b     : extremos del intervalo [a, b]
        tol      : tolerancia para el criterio de parada
        max_iter : número máximo de iteraciones

    Retorna dict con: raiz, iteraciones (tabla), convergió, mensaje
    """
    fa = safe_eval(expr_str, a)
    fb = safe_eval(expr_str, b)

    if fa * fb > 0:
        return {
            "success": False,
            "error": f"f(a) y f(b) tienen el mismo signo. f({a})={fa:.6f}, f({b})={fb:.6f}. No se garantiza raíz en el intervalo.",
            "root": None,
            "iterations": [],
            "converged": False
        }

    iterations = []

    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = safe_eval(expr_str, c)

        error_abs = abs(b - a) / 2.0

        iterations.append({
            "n": i,
            "a": round(a, 8),
            "b": round(b, 8),
            "c (xm)": round(c, 8),
            "f(c)": round(fc, 8),
            "error": round(error_abs, 8),
        })

        if abs(fc) < 1e-14 or error_abs < tol:
            return {
                "success": True,
                "root": round(c, 10),
                "iterations": iterations,
                "converged": True,
                "num_iterations": i,
                "final_error": error_abs,
                "f_root": fc,
                "message": f"Convergió en {i} iteraciones. Raíz ≈ {c:.8f}"
            }

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    c = (a + b) / 2.0
    return {
        "success": True,
        "root": round(c, 10),
        "iterations": iterations,
        "converged": False,
        "num_iterations": max_iter,
        "final_error": abs(b - a) / 2.0,
        "f_root": safe_eval(expr_str, c),
        "message": f"Se alcanzó el máximo de iteraciones ({max_iter}). Raíz aproximada ≈ {c:.8f}"
    }


# ─── PUNTO FIJO ──────────────────────────────────────────────────────────────

def punto_fijo(
    g_expr_str: str,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> dict:
    """
    Método de Punto Fijo: x_{n+1} = g(x_n)

    Parámetros:
        g_expr_str : expresión de g(x) tal que x = g(x)
        x0         : valor inicial
        tol        : tolerancia
        max_iter   : máximo de iteraciones
    """
    iterations = []
    x_prev = x0

    for i in range(1, max_iter + 1):
        try:
            x_new = safe_eval(g_expr_str, x_prev)
        except Exception as e:
            return {
                "success": False,
                "error": f"Error evaluando g({x_prev}): {e}",
                "root": None,
                "iterations": iterations,
                "converged": False
            }

        error = abs(x_new - x_prev)

        iterations.append({
            "n": i,
            "x_n": round(x_prev, 8),
            "g(x_n)": round(x_new, 8),
            "error": round(error, 8),
        })

        if error < tol:
            return {
                "success": True,
                "root": round(x_new, 10),
                "iterations": iterations,
                "converged": True,
                "num_iterations": i,
                "final_error": error,
                "message": f"Convergió en {i} iteraciones. Punto fijo ≈ {x_new:.8f}"
            }

        # Detectar divergencia
        if abs(x_new) > 1e10:
            return {
                "success": False,
                "error": "El método diverge. Verificar que |g'(x)| < 1 en el entorno de la raíz.",
                "root": None,
                "iterations": iterations,
                "converged": False
            }

        x_prev = x_new

    return {
        "success": True,
        "root": round(x_prev, 10),
        "iterations": iterations,
        "converged": False,
        "num_iterations": max_iter,
        "final_error": error,
        "message": f"Se alcanzó el máximo de iteraciones ({max_iter}). Aproximación ≈ {x_prev:.8f}"
    }


# ─── NEWTON-RAPHSON ───────────────────────────────────────────────────────────

def newton_raphson(
    f_expr_str: str,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    df_expr_str: Optional[str] = None
) -> dict:
    """
    Método de Newton-Raphson.

    Parámetros:
        f_expr_str  : expresión de f(x)
        x0          : valor inicial
        tol         : tolerancia
        max_iter    : máximo de iteraciones
        df_expr_str : derivada de f(x) (si es None se calcula simbólicamente)
    """
    # Calcular derivada si no se provee
    if df_expr_str is None:
        try:
            df_expr_str = get_derivative(f_expr_str)
        except Exception as e:
            return {
                "success": False,
                "error": f"No se pudo calcular la derivada: {e}",
                "root": None,
                "iterations": [],
                "converged": False
            }

    iterations = []
    x = x0

    for i in range(1, max_iter + 1):
        try:
            fx = safe_eval(f_expr_str, x)
            dfx = safe_eval(df_expr_str, x)
        except Exception as e:
            return {
                "success": False,
                "error": f"Error en iteración {i}: {e}",
                "root": None,
                "iterations": iterations,
                "converged": False
            }

        if abs(dfx) < 1e-14:
            return {
                "success": False,
                "error": f"La derivada es cero en x={x:.6f}. El método falla.",
                "root": None,
                "iterations": iterations,
                "converged": False
            }

        x_new = x - fx / dfx
        error = abs(x_new - x)

        iterations.append({
            "n": i,
            "x_n": round(x, 8),
            "f(x_n)": round(fx, 8),
            "f'(x_n)": round(dfx, 8),
            "x_{n+1}": round(x_new, 8),
            "error": round(error, 8),
        })

        if error < tol or abs(fx) < 1e-14:
            return {
                "success": True,
                "root": round(x_new, 10),
                "iterations": iterations,
                "converged": True,
                "num_iterations": i,
                "final_error": error,
                "derivative_used": df_expr_str,
                "message": f"Convergió en {i} iteraciones. Raíz ≈ {x_new:.8f}"
            }

        x = x_new

    return {
        "success": True,
        "root": round(x, 10),
        "iterations": iterations,
        "converged": False,
        "num_iterations": max_iter,
        "final_error": error,
        "derivative_used": df_expr_str,
        "message": f"Se alcanzó el máximo de iteraciones. Aproximación ≈ {x:.8f}"
    }
