"""
parser_module.py — Detecta el tipo de problema y extrae parámetros
"""
import re
from typing import Optional


# Patrones de detección por tipo de problema
PATTERNS = {
    "biseccion": [
        r'\bbisecci[oó]n\b', r'\bintervalo\b.*\bfunci[oó]n\b',
        r'\b\[.*,.*\]\b.*\bf\(', r'\braíces?\b.*\bintervalo\b',
        r'\bceros?\b.*\bintervalo\b', r'\bmétodo\s+de\s+bisecci',
    ],
    "punto_fijo": [
        r'\bpunto\s+fijo\b', r'\bg\(x\)', r'\biteraci[oó]n\b.*\bx\s*=\s*g',
        r'\bx\s*=\s*g\(x\)', r'\bconvergencia\b.*\biteraci',
    ],
    "newton": [
        r'\bnewton\b', r'\bnewton[\s-]raphson\b', r'\bderivada\b.*\braíz\b',
        r'\bf\'\(', r'\bdf\b', r'\btangente\b.*\bcero\b',
    ],
    "integral": [
        r'\bintegral?\b', r'∫', r'\bintegr[ae]',
        r'\bárea\b.*\bcurva\b', r'\bsimpson\b', r'\btrapecio\b',
        r'\bint\s*\(', r'\[a,\s*b\].*\bdx\b',
    ],
    "sumatoria": [
        r'\bsumatoria\b', r'\bserie\b', r'\bsuma\b.*\btérminos\b',
        r'Σ', r'\bsum\b.*\bi\s*=', r'\bn\s*=.*\bsuma\b',
        r'\bsumar\b.*\bdes?de\b',
    ],
    "interpolacion": [
        r'\binterpolaci[oó]n\b', r'\blagrange\b', r'\bpolinomio\b.*\bpuntos\b',
        r'\bdif[ea]rencias\s+divididas\b', r'\bnewton\b.*\binterpolaci',
        r'\bajuste\b.*\bpuntos\b', r'\btabla\b.*\bx.*y\b',
    ],
}

METHOD_NAMES = {
    "biseccion": "Bisección",
    "punto_fijo": "Punto Fijo",
    "newton": "Newton-Raphson",
    "integral": "Integral Numérica",
    "sumatoria": "Sumatoria / Serie",
    "interpolacion": "Interpolación",
}


def detect_problem_type(text: str) -> dict:
    """
    Analiza el texto y detecta el tipo de problema matemático.
    Devuelve el tipo detectado y un score de confianza.
    """
    text_lower = text.lower()
    scores = {}

    for method, patterns in PATTERNS.items():
        score = 0
        matched = []
        for pattern in patterns:
            if re.search(pattern, text_lower):
                score += 1
                matched.append(pattern)

        if score > 0:
            scores[method] = {"score": score, "matches": matched}

    if not scores:
        return {
            "detected": None,
            "confidence": 0.0,
            "all_scores": {},
            "display_name": "No detectado"
        }

    # 🔧 Ajuste: penalizar "newton" si también aparece "interpolacion"
    if "newton" in scores and "interpolacion" in scores:
        scores["newton"]["score"] -= 1

    # Elegir mejor
    best = max(scores, key=lambda k: scores[k]["score"])
    total_patterns = len(PATTERNS[best])
    confidence = min(scores[best]["score"] / total_patterns, 1.0)

    return {
        "detected": best,
        "confidence": confidence,
        "all_scores": scores,
        "display_name": METHOD_NAMES.get(best, best)
    }


def extract_function_from_text(text: str) -> Optional[str]:
    patterns = [
        r'f\(x\)\s*=\s*([^\n,;]+)',
        r'g\(x\)\s*=\s*([^\n,;]+)',
        r'y\s*=\s*([^\n,;]+)',
        r'f\s*=\s*([^\n,;]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            expr = match.group(1).strip()

            # 🔧 limpiar cosas típicas de integrales
            expr = re.sub(r'dx.*$', '', expr)
            expr = re.sub(r'=', '', expr)

            expr = clean_expression(expr)
            return expr

    return None

def extract_interval(text: str) -> Optional[tuple]:
    """Extrae un intervalo [a, b] del texto de forma más segura."""

    patterns = [
        r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]',  # [a, b]
        r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)',  # (a, b)
        r'intervalo\s*(?:es|:)?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2))

    return None


def extract_tolerance(text: str) -> float:
    """Extrae la tolerancia/epsilon del texto, por defecto 1e-6."""
    patterns = [
        r'tol(?:erancia)?\s*[=:]\s*([\d.e\-]+)',
        r'epsilon\s*[=:]\s*([\d.e\-]+)',
        r'(?:error|precisión)\s*[=:]\s*([\d.e\-]+)',
        r'10\^?[-−]\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                if '10^' in pattern or '10\^' in pattern:
                    return 10 ** (-float(match.group(1)))
                return float(match.group(1))
            except ValueError:
                pass
    return 1e-6


def extract_points_table(text: str) -> Optional[dict]:
    """
    Extrae una tabla de puntos (x, y) del texto para interpolación.
    Formatos soportados: 'x: 1 2 3, y: 4 5 6' o pares '(1,4) (2,5)...'
    """
    # Formato (x, y) pares
    pair_pattern = r'\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)'
    pairs = re.findall(pair_pattern, text)
    if len(pairs) >= 2:
        xs = [float(p[0]) for p in pairs]
        ys = [float(p[1]) for p in pairs]
        return {"x": xs, "y": ys}

    # Formato tabla x: ... y: ...
    x_match = re.search(r'x\s*[=:]\s*([\d.\s,\-]+)', text, re.IGNORECASE)
    y_match = re.search(r'y\s*[=:]\s*([\d.\s,\-]+)', text, re.IGNORECASE)
    if x_match and y_match:
        try:
            xs = [float(v) for v in re.split(r'[\s,]+', x_match.group(1).strip()) if v]
            ys = [float(v) for v in re.split(r'[\s,]+', y_match.group(1).strip()) if v]
            if len(xs) == len(ys) and len(xs) >= 2:
                return {"x": xs, "y": ys}
        except ValueError:
            pass
    return None


def extract_sum_params(text: str) -> dict:
    params = {"expr": None, "start": 1, "end": 10, "var": "n"}

    # desde
    from_match = re.search(r'(?:desde|de|from|i\s*=|n\s*=)\s*(-?\d+)', text, re.IGNORECASE)
    if from_match:
        params["start"] = int(from_match.group(1))

    # hasta
    to_match = re.search(r'(?:hasta|a|to)\s*(-?\d+)', text, re.IGNORECASE)
    if to_match:
        params["end"] = int(to_match.group(1))

    # formato i=1..10
    range_match = re.search(r'(\w)\s*=\s*(\d+)\s*\.\.\s*(\d+)', text)
    if range_match:
        params["var"] = range_match.group(1)
        params["start"] = int(range_match.group(2))
        params["end"] = int(range_match.group(3))

    return params


def clean_expression(expr: str) -> str:
    """Limpia una expresión matemática de artefactos OCR."""
    # Reemplazos comunes
    expr = expr.replace('^', '**')
    expr = expr.replace('×', '*')
    expr = expr.replace('÷', '/')
    expr = expr.replace('π', 'pi')
    expr = re.sub(r'\s+', '', expr)
    # Quitar caracteres no matemáticos al final
    expr = re.sub(r'[^\w\+\-\*/\.\(\)\^\|]+$', '', expr)
    return expr


def parse_full_problem(text: str) -> dict:
    problem_type = detect_problem_type(text)
    function_expr = extract_function_from_text(text)
    interval = extract_interval(text)
    tolerance = extract_tolerance(text)
    points = extract_points_table(text)
    sum_params = extract_sum_params(text)

    errors = []

    if problem_type["detected"] in ["biseccion", "newton", "integral"] and not function_expr:
        errors.append("No se pudo detectar la función")

    return {
        "type": problem_type,
        "function": function_expr,
        "interval": interval,
        "tolerance": tolerance,
        "points": points,
        "sum_params": sum_params,
        "errors": errors,
        "raw_text": text,
    }