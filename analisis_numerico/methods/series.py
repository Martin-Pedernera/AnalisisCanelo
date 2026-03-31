"""
methods/series.py — Sumatorias y series numéricas
"""
import sympy as sp
import numpy as np
from typing import Optional


def calcular_sumatoria(
    expr_str: str,
    var: str = "n",
    start: int = 1,
    end: int = 10
) -> dict:
    """
    Calcula Σ expr para var desde start hasta end.

    Parámetros:
        expr_str : expresión a sumar (ej: '1/n**2', 'n*(n+1)', 'x**n/factorial(n)')
        var      : variable de suma (por defecto 'n')
        start    : límite inferior
        end      : límite superior (puede ser 'inf' para series infinitas)
    """
    n = sp.Symbol(var)

    try:
        expr = sp.sympify(expr_str)
    except Exception as e:
        return {"success": False, "error": f"Expresión inválida: {e}"}

    # Tabla de términos
    table_rows = []
    partial_sum = 0.0
    terms = []

    display_limit = min(end, start + 49) if end != sp.oo else start + 49

    for i in range(start, int(display_limit) + 1):
        try:
            val = float(expr.subs(n, i).evalf())
            partial_sum += val
            terms.append(val)
            table_rows.append({
                "n": i,
                "término": round(val, 10),
                "suma parcial": round(partial_sum, 10),
            })
        except Exception:
            break

    # Suma simbólica exacta
    try:
        exact_sum = sp.summation(expr, (n, start, end))
        exact_val = float(exact_sum.evalf()) if exact_sum.is_number else None
        exact_str = str(exact_sum)
    except Exception:
        exact_val = None
        exact_str = "No disponible"

    # Verificar convergencia (para series infinitas)
    convergence_info = None
    if end == sp.oo or str(end).lower() == 'inf':
        convergence_info = check_convergence(expr_str, var)

    return {
        "success": True,
        "result": round(partial_sum, 10),
        "exact_symbolic": exact_str,
        "exact_value": round(exact_val, 10) if exact_val is not None else None,
        "table": table_rows,
        "terms_shown": len(table_rows),
        "total_terms": end - start + 1 if end != sp.oo else "∞",
        "convergence": convergence_info,
        "message": f"Σ ({expr_str}), n={start} hasta {end} = {partial_sum:.8f}"
    }


def check_convergence(expr_str: str, var: str = "n") -> dict:
    """
    Verifica convergencia de una serie por criterio del cociente (D'Alembert).
    """
    n = sp.Symbol(var)
    try:
        expr = sp.sympify(expr_str)
        # Criterio del límite: si lim |a_n| -> 0 es necesario pero no suficiente
        lim_val = sp.limit(sp.Abs(expr), n, sp.oo)

        # Criterio del cociente
        ratio = sp.Abs(expr.subs(n, n + 1) / expr)
        ratio_limit = sp.limit(ratio, n, sp.oo)

        if ratio_limit < 1:
            verdict = "Converge (criterio del cociente: L < 1)"
        elif ratio_limit > 1:
            verdict = "Diverge (criterio del cociente: L > 1)"
        elif ratio_limit == 1:
            verdict = "Criterio del cociente inconcluso (L = 1)"
        else:
            verdict = "No determinado"

        return {
            "limit_an": str(lim_val),
            "ratio_limit": str(ratio_limit),
            "verdict": verdict
        }
    except Exception as e:
        return {"verdict": f"No se pudo determinar: {e}"}


def series_geometrica(a: float, r: float, n: int = 10) -> dict:
    """
    Serie geométrica: Σ a·r^k, k=0..n
    Si |r| < 1 calcula también la suma infinita exacta.
    """
    terms = [a * (r ** k) for k in range(n + 1)]
    partial_sums = [sum(terms[:i+1]) for i in range(n + 1)]

    table_rows = [{"k": k, "término": round(terms[k], 8), "suma parcial": round(partial_sums[k], 8)}
                  for k in range(n + 1)]

    result = {
        "success": True,
        "result": partial_sums[-1],
        "table": table_rows,
        "a": a, "r": r, "n": n,
    }

    if abs(r) < 1:
        infinite_sum = a / (1 - r)
        result["infinite_sum"] = round(infinite_sum, 10)
        result["message"] = f"Suma finita ({n+1} términos) = {partial_sums[-1]:.6f} | Suma infinita = {infinite_sum:.6f}"
    else:
        result["message"] = f"Suma ({n+1} términos) = {partial_sums[-1]:.6f} | La serie diverge (|r| ≥ 1)"

    return result


def series_taylor(
    expr_str: str,
    punto: float = 0,
    grado: int = 5,
    x_eval: Optional[float] = None
) -> dict:
    """
    Calcula la serie de Taylor de f(x) alrededor de 'punto' hasta grado n.
    """
    x = sp.Symbol('x')
    try:
        expr = sp.sympify(expr_str)
        taylor = sp.series(expr, x, punto, grado + 1).removeO()
        taylor_str = str(taylor)

        # Tabla de coeficientes
        poly = sp.Poly(taylor, x)
        coeffs = poly.all_coeffs()

        result = {
            "success": True,
            "polynomial": taylor_str,
            "degree": grado,
            "around": punto,
        }

        if x_eval is not None:
            approx = float(taylor.subs(x, x_eval))
            exact = float(expr.subs(x, x_eval).evalf())
            result["eval_x"] = x_eval
            result["approx_value"] = round(approx, 8)
            result["exact_value"] = round(exact, 8)
            result["error"] = round(abs(approx - exact), 10)

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
