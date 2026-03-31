"""
methods/interpolation.py — Interpolación de Lagrange y Newton (diferencias divididas)
"""
import numpy as np
import sympy as sp
from typing import List, Optional


# ─── LAGRANGE ────────────────────────────────────────────────────────────────

def lagrange(
    x_data: List[float],
    y_data: List[float],
    x_eval: Optional[float] = None
) -> dict:
    """
    Interpolación de Lagrange.

    P(x) = Σ y_i · L_i(x)
    donde L_i(x) = Π (x - x_j)/(x_i - x_j), j≠i

    Parámetros:
        x_data : lista de puntos x conocidos
        y_data : lista de valores y = f(x)
        x_eval : punto a interpolar (opcional)
    """
    x_arr = np.array(x_data, dtype=float)
    y_arr = np.array(y_data, dtype=float)
    n = len(x_arr)

    if len(x_arr) != len(y_arr):
        return {"success": False, "error": "x_data e y_data deben tener la misma longitud"}
    if n < 2:
        return {"success": False, "error": "Se necesitan al menos 2 puntos"}

    # Construir polinomio simbólico
    x = sp.Symbol('x')
    P = sp.Integer(0)
    L_terms = []

    for i in range(n):
        L_i = sp.Integer(1)
        for j in range(n):
            if j != i:
                L_i *= (x - x_arr[j]) / (x_arr[i] - x_arr[j])
        L_terms.append(sp.simplify(L_i))
        P += y_arr[i] * L_i

    P_simplified = sp.expand(P)

    # Tabla de bases L_i evaluadas (si hay punto de evaluación)
    table_rows = []
    for i in range(n):
        row = {
            "i": i,
            "x_i": round(x_arr[i], 6),
            "y_i": round(y_arr[i], 6),
            "L_i(x)": str(sp.simplify(L_terms[i])),
        }
        if x_eval is not None:
            row["L_i(x_eval)"] = round(float(L_terms[i].subs(x, x_eval)), 8)
        table_rows.append(row)

    result = {
        "success": True,
        "polynomial": str(P_simplified),
        "degree": n - 1,
        "table": table_rows,
        "n_points": n,
    }

    if x_eval is not None:
        value = float(P_simplified.subs(x, x_eval))
        result["eval_x"] = x_eval
        result["eval_value"] = round(value, 8)
        result["message"] = f"P({x_eval}) = {value:.8f}"

    return result


# ─── NEWTON DIFERENCIAS DIVIDIDAS ─────────────────────────────────────────────

def newton_diferencias_divididas(
    x_data: List[float],
    y_data: List[float],
    x_eval: Optional[float] = None
) -> dict:
    """
    Interpolación de Newton con diferencias divididas.

    P(x) = f[x0] + f[x0,x1](x-x0) + f[x0,x1,x2](x-x0)(x-x1) + ...

    Parámetros:
        x_data : lista de puntos x conocidos
        y_data : lista de valores y = f(x)
        x_eval : punto a interpolar (opcional)
    """
    x_arr = np.array(x_data, dtype=float)
    y_arr = np.array(y_data, dtype=float)
    n = len(x_arr)

    if len(x_arr) != len(y_arr):
        return {"success": False, "error": "x_data e y_data deben tener la misma longitud"}
    if n < 2:
        return {"success": False, "error": "Se necesitan al menos 2 puntos"}

    # Tabla de diferencias divididas
    dd = np.zeros((n, n))
    dd[:, 0] = y_arr

    for j in range(1, n):
        for i in range(n - j):
            dd[i][j] = (dd[i + 1][j - 1] - dd[i][j - 1]) / (x_arr[i + j] - x_arr[i])

    coeffs = dd[0, :]  # Coeficientes del polinomio de Newton

    # Tabla de diferencias divididas para mostrar
    table_rows = []
    for i in range(n):
        row = {"x_i": round(x_arr[i], 6), "f[x_i]": round(dd[i][0], 8)}
        for j in range(1, n - i):
            row[f"f[x_{i}..x_{i+j}]"] = round(dd[i][j], 8)
        table_rows.append(row)

    # Construir polinomio simbólico
    x = sp.Symbol('x')
    P = sp.Float(coeffs[0])
    prod = sp.Integer(1)

    poly_terms = [f"{coeffs[0]:.6f}"]

    for k in range(1, n):
        prod *= (x - x_arr[k - 1])
        P += coeffs[k] * prod

        term_str = f"+ ({coeffs[k]:.6f})" + "".join([f"(x - {x_arr[j]:.4f})" for j in range(k)])
        poly_terms.append(term_str)

    P_simplified = sp.expand(P)

    result = {
        "success": True,
        "polynomial": str(P_simplified),
        "coefficients": [round(c, 8) for c in coeffs],
        "difference_table": table_rows,
        "degree": n - 1,
        "n_points": n,
        "polynomial_newton_form": "\n".join(poly_terms),
    }

    if x_eval is not None:
        # Evaluación eficiente con método de Horner hacia atrás
        value = coeffs[-1]
        for k in range(n - 2, -1, -1):
            value = coeffs[k] + (x_eval - x_arr[k]) * value

        result["eval_x"] = x_eval
        result["eval_value"] = round(float(value), 8)
        result["message"] = f"P({x_eval}) = {value:.8f}"

    return result


# ─── DIFERENCIAS FINITAS (Newton hacia adelante) ─────────────────────────────

def newton_diferencias_finitas(
    x_data: List[float],
    y_data: List[float],
    x_eval: Optional[float] = None
) -> dict:
    """
    Newton hacia adelante con diferencias finitas (espaciado uniforme).
    """
    x_arr = np.array(x_data, dtype=float)
    y_arr = np.array(y_data, dtype=float)
    n = len(x_arr)
    h = x_arr[1] - x_arr[0]

    # Verificar espaciado uniforme
    diffs = np.diff(x_arr)
    if not np.allclose(diffs, h, rtol=1e-5):
        return {"success": False, "error": "Los puntos no están igualmente espaciados. Use Newton Diferencias Divididas."}

    # Tabla de diferencias finitas
    delta = np.zeros((n, n))
    delta[:, 0] = y_arr

    for j in range(1, n):
        for i in range(n - j):
            delta[i][j] = delta[i + 1][j - 1] - delta[i][j - 1]

    # Tabla para mostrar
    table_rows = []
    for i in range(n):
        row = {"x_i": round(x_arr[i], 6), "y_i": round(y_arr[i], 8)}
        for j in range(1, n - i):
            row[f"Δ^{j}y_{i}"] = round(delta[i][j], 8)
        table_rows.append(row)

    result = {
        "success": True,
        "h": h,
        "difference_table": table_rows,
        "n_points": n,
    }

    if x_eval is not None:
        s = (x_eval - x_arr[0]) / h
        value = delta[0][0]
        s_term = 1.0
        factorial = 1

        for k in range(1, n):
            s_term *= (s - (k - 1))
            factorial *= k
            value += (s_term / factorial) * delta[0][k]

        result["eval_x"] = x_eval
        result["eval_value"] = round(float(value), 8)
        result["s_value"] = round(float(s), 6)
        result["message"] = f"P({x_eval}) = {value:.8f}  (s = {s:.6f})"

    return result
