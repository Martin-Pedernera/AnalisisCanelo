"""
methods/integrals.py — Integración numérica: Regla del Trapecio y Simpson 1/3
"""
import numpy as np
import sympy as sp
from typing import Optional


def safe_eval_array(expr_str: str, x_arr: np.ndarray) -> np.ndarray:
    """Evalúa una expresión en un array de valores x."""
    x = sp.Symbol('x')
    expr = sp.sympify(expr_str)
    f = sp.lambdify(x, expr, modules=['numpy'])
    return np.array(f(x_arr), dtype=float)


def exact_integral(expr_str: str, a: float, b: float) -> Optional[float]:
    """Calcula la integral exacta simbólicamente (para comparación de error)."""
    try:
        x = sp.Symbol('x')
        expr = sp.sympify(expr_str)
        result = sp.integrate(expr, (x, a, b))
        return float(result.evalf())
    except Exception:
        return None


# ─── REGLA DEL TRAPECIO ───────────────────────────────────────────────────────

def trapecio(
    expr_str: str,
    a: float,
    b: float,
    n: int = 100
) -> dict:
    """
    Regla del Trapecio Compuesta.

    ∫[a,b] f(x)dx ≈ (h/2)[f(x0) + 2f(x1) + ... + 2f(x_{n-1}) + f(xn)]

    Parámetros:
        expr_str : expresión de f(x)
        a, b     : límites de integración
        n        : número de subintervalos (debe ser >= 1)
    """
    if n < 1:
        return {"success": False, "error": "n debe ser >= 1"}
    if a >= b:
        return {"success": False, "error": "a debe ser menor que b"}

    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)

    try:
        y_vals = safe_eval_array(expr_str, x_vals)
    except Exception as e:
        return {"success": False, "error": f"Error evaluando la función: {e}"}

    # Fórmula del trapecio
    integral = h * (y_vals[0] / 2 + np.sum(y_vals[1:-1]) + y_vals[-1] / 2)

    # Tabla de puntos (mostrar máx 20 para no saturar la UI)
    step = max(1, (n + 1) // 20)
    table_rows = []
    for i in range(0, n + 1, step):
        coef = 1 if (i == 0 or i == n) else 2
        table_rows.append({
            "i": i,
            "x_i": round(float(x_vals[i]), 6),
            "f(x_i)": round(float(y_vals[i]), 8),
            "coeficiente": coef,
        })

    exact = exact_integral(expr_str, a, b)
    error_rel = abs((integral - exact) / exact) * 100 if exact and exact != 0 else None

    return {
        "success": True,
        "result": round(float(integral), 10),
        "n": n,
        "h": h,
        "table": table_rows,
        "exact": round(exact, 10) if exact is not None else None,
        "error_relative_pct": round(error_rel, 6) if error_rel is not None else None,
        "formula": f"h/2 · [f(x₀) + 2f(x₁) + ... + 2f(xₙ₋₁) + f(xₙ)], h = {h:.6f}",
        "message": f"∫ f(x)dx de {a} a {b} ≈ {integral:.8f}  (n={n} subintervalos)"
    }


# ─── SIMPSON 1/3 ──────────────────────────────────────────────────────────────

def simpson_13(
    expr_str: str,
    a: float,
    b: float,
    n: int = 100
) -> dict:
    """
    Regla de Simpson 1/3 Compuesta.

    ∫[a,b] f(x)dx ≈ (h/3)[f(x0) + 4f(x1) + 2f(x2) + 4f(x3) + ... + f(xn)]

    n debe ser par. Si se pasa impar se incrementa en 1.

    Parámetros:
        expr_str : expresión de f(x)
        a, b     : límites de integración
        n        : número de subintervalos (debe ser par)
    """
    if n < 2:
        n = 2
    if n % 2 != 0:
        n += 1  # Simpson requiere n par
    if a >= b:
        return {"success": False, "error": "a debe ser menor que b"}

    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)

    try:
        y_vals = safe_eval_array(expr_str, x_vals)
    except Exception as e:
        return {"success": False, "error": f"Error evaluando la función: {e}"}

    # Fórmula de Simpson 1/3 compuesta
    integral = y_vals[0] + y_vals[-1]
    integral += 4 * np.sum(y_vals[1:-1:2])   # índices impares × 4
    integral += 2 * np.sum(y_vals[2:-2:2])   # índices pares × 2
    integral *= h / 3

    # Tabla de puntos
    step = max(1, (n + 1) // 20)
    table_rows = []
    for i in range(0, n + 1, step):
        if i == 0 or i == n:
            coef = 1
        elif i % 2 == 1:
            coef = 4
        else:
            coef = 2
        table_rows.append({
            "i": i,
            "x_i": round(float(x_vals[i]), 6),
            "f(x_i)": round(float(y_vals[i]), 8),
            "coeficiente": coef,
        })

    exact = exact_integral(expr_str, a, b)
    error_rel = abs((integral - exact) / exact) * 100 if exact and exact != 0 else None

    return {
        "success": True,
        "result": round(float(integral), 10),
        "n": n,
        "h": h,
        "table": table_rows,
        "exact": round(exact, 10) if exact is not None else None,
        "error_relative_pct": round(error_rel, 6) if error_rel is not None else None,
        "formula": f"h/3 · [f(x₀) + 4f(x₁) + 2f(x₂) + ... + f(xₙ)], h = {h:.6f}",
        "note": "n fue ajustado a par" if (n % 2 == 0) else "",
        "message": f"∫ f(x)dx de {a} a {b} ≈ {integral:.8f}  (n={n} subintervalos)"
    }


# ─── SIMPSON 3/8 (bonus) ─────────────────────────────────────────────────────

def simpson_38(
    expr_str: str,
    a: float,
    b: float,
    n: int = 99
) -> dict:
    """
    Regla de Simpson 3/8 Compuesta. n debe ser múltiplo de 3.
    """
    if n < 3:
        n = 3
    while n % 3 != 0:
        n += 1
    if a >= b:
        return {"success": False, "error": "a debe ser menor que b"}

    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)

    try:
        y_vals = safe_eval_array(expr_str, x_vals)
    except Exception as e:
        return {"success": False, "error": f"Error evaluando la función: {e}"}

    integral = y_vals[0] + y_vals[-1]
    for i in range(1, n):
        if i % 3 == 0:
            integral += 2 * y_vals[i]
        else:
            integral += 3 * y_vals[i]
    integral *= 3 * h / 8

    exact = exact_integral(expr_str, a, b)
    error_rel = abs((integral - exact) / exact) * 100 if exact and exact != 0 else None

    return {
        "success": True,
        "result": round(float(integral), 10),
        "n": n,
        "h": h,
        "exact": round(exact, 10) if exact is not None else None,
        "error_relative_pct": round(error_rel, 6) if error_rel is not None else None,
        "message": f"∫ f(x)dx de {a} a {b} ≈ {integral:.8f}  (n={n} subintervalos)"
    }
