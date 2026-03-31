"""
app.py — Aplicación principal de Análisis Numérico con Streamlit
Ejecutar con: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
from PIL import Image
import io

# Módulos propios
from ocr_module import extract_text_from_image, image_bytes_from_uploaded
from parser_module import parse_full_problem, detect_problem_type

from methods.roots import biseccion, punto_fijo, newton_raphson, get_derivative
from methods.integrals import trapecio, simpson_13, simpson_38
from methods.series import calcular_sumatoria, series_geometrica, series_taylor
from methods.interpolation import (
    lagrange, newton_diferencias_divididas, newton_diferencias_finitas
)
from plotter import (
    plot_root_method, plot_integral, plot_interpolation, plot_series
)

# ─── CONFIGURACIÓN DE PÁGINA ─────────────────────────────────────────────────

st.set_page_config(
    page_title="Análisis Numérico",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #5B3FD4;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background: linear-gradient(135deg, #f0edff 0%, #e8f5ee 100%);
        border-left: 4px solid #5B3FD4;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .detected-badge {
        background: #5B3FD4;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .error-box {
        background: #fff0f0;
        border-left: 4px solid #D85A30;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
    }
    .info-box {
        background: #f0f7ff;
        border-left: 4px solid #185FA5;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    div[data-testid="stTabs"] button {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ─── FUNCIONES AUXILIARES ─────────────────────────────────────────────────────

def show_result_box(label: str, value: str):
    st.markdown(f"""
    <div class="result-box">
        <strong>{label}</strong><br>
        <span style="font-size:1.3rem;font-weight:700;color:#5B3FD4">{value}</span>
    </div>
    """, unsafe_allow_html=True)


def show_error(msg: str):
    st.markdown(f'<div class="error-box">⚠️ {msg}</div>', unsafe_allow_html=True)


def show_info(msg: str):
    st.markdown(f'<div class="info-box">ℹ️ {msg}</div>', unsafe_allow_html=True)


def show_iteration_table(iterations: list):
    if iterations:
        df = pd.DataFrame(iterations)
        st.dataframe(df, use_container_width=True, height=min(400, 50 + 35 * len(iterations)))


def show_plot_bytes(img_bytes: bytes, caption: str = ""):
    if img_bytes is not None:
        st.image(img_bytes, caption=caption, use_container_width=True)


def parse_points_input(text: str) -> tuple[list, list] | None:
    """Parsea texto de puntos en formato 'x1,y1; x2,y2; ...'"""
    try:
        pairs = [p.strip() for p in text.split(';') if p.strip()]
        xs, ys = [], []
        for pair in pairs:
            vals = [v.strip() for v in pair.split(',')]
            if len(vals) == 2:
                xs.append(float(vals[0]))
                ys.append(float(vals[1]))
        if len(xs) >= 2:
            return xs, ys
    except Exception:
        pass
    return None


# ─── SECCIÓN OCR ─────────────────────────────────────────────────────────────

def section_ocr():
    st.markdown("### 📷 Reconocimiento de imagen (OCR)")
    st.success("Subí una foto o captura de pantalla de tu problema. La aplicación intentará detectar el tipo de ejercicio y extraer los parámetros automáticamente.")

    col_up, col_prev = st.columns([1, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Subir imagen del problema",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Formatos: PNG, JPG, JPEG, BMP, TIFF"
        )

    if uploaded:
        img_bytes = image_bytes_from_uploaded(uploaded)

        with col_prev:
            st.image(uploaded, caption="Imagen subida", use_container_width=True)

        with st.spinner("Procesando imagen con Tesseract OCR..."):
            ocr_result = extract_text_from_image(img_bytes)

        if not ocr_result["success"]:
            show_error(f"Error en OCR: {ocr_result['error']}. Asegurate de tener Tesseract instalado.")
            st.markdown("**Instalación de Tesseract:**")
            st.code("# Windows: descargar desde https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "# Linux: sudo apt install tesseract-ocr tesseract-ocr-spa\n"
                    "# Mac: brew install tesseract", language="bash")
            return

        raw = ocr_result["raw_text"]
        normalized = ocr_result["normalized_text"]

        st.markdown("#### Texto extraído")
        with st.expander("Ver texto crudo (OCR)", expanded=True):
            st.text_area("Texto detectado", value=raw, height=120, key="ocr_raw", disabled=True)

        # Detectar tipo de problema
        parsed = parse_full_problem(normalized)
        prob_type = parsed["type"]

        st.markdown("#### Análisis del problema")
        col1, col2, col3 = st.columns(3)
        with col1:
            detected = prob_type.get("display_name", "No detectado")
            conf = prob_type.get("confidence", 0)
            st.metric("Tipo detectado", detected, f"Confianza: {conf:.0%}")
        with col2:
            if parsed.get("function"):
                st.metric("Función detectada", parsed["function"])
        with col3:
            if parsed.get("interval"):
                a, b = parsed["interval"]
                st.metric("Intervalo detectado", f"[{a}, {b}]")

        # Permitir editar el texto antes de procesar
        st.markdown("#### Texto para procesar")
        text_to_use = st.text_area(
            "Podés editar el texto antes de resolver",
            value=normalized,
            height=100,
            key="ocr_edit"
        )

        if st.button("🔄 Re-analizar texto editado", key="reanalyze"):
            st.session_state['ocr_parsed'] = parse_full_problem(text_to_use)
            st.rerun()

        # Guardar en session state para usar en la calculadora
        if st.button("✅ Usar este texto en la calculadora", type="primary", key="use_ocr"):
            st.session_state['ocr_text'] = text_to_use
            st.session_state['ocr_parsed'] = parsed

            st.success("✅ Datos cargados en la calculadora.")

            # 🔽 NUEVO: sugerencia automática de navegación
            detected_type = parsed["type"].get("detected")

            if detected_type == "biseccion":
                st.info("👉 Ir a la pestaña: Raíces → Bisección")
            elif detected_type == "punto_fijo":
                st.info("👉 Ir a la pestaña: Raíces → Punto Fijo")
            elif detected_type == "newton":
                st.info("👉 Ir a la pestaña: Raíces → Newton-Raphson")
            elif detected_type == "integral":
                st.info("👉 Ir a la pestaña: Integrales")
            elif detected_type == "sumatoria":
                st.info("👉 Ir a la pestaña: Sumatorias")
            elif detected_type == "interpolacion":
                st.info("👉 Ir a la pestaña: Interpolación")
            else:
                st.info("👉 Revisá las pestañas manualmente.")


# ─── SECCIÓN RAÍCES ──────────────────────────────────────────────────────────

def section_roots():
    st.markdown("### 🎯 Búsqueda de Raíces")

    # Pre-cargar desde OCR si disponible
    ocr_parsed = st.session_state.get('ocr_parsed', {})
    default_func = ocr_parsed.get('function') or 'x**3 - x - 2'
    default_a = ocr_parsed.get('interval', (1.0, 2.0))[0] if ocr_parsed.get('interval') else 1.0
    default_b = ocr_parsed.get('interval', (1.0, 2.0))[1] if ocr_parsed.get('interval') else 2.0
    default_tol = ocr_parsed.get('tolerance', 1e-6)

    tab_bis, tab_pf, tab_nr = st.tabs(["Bisección", "Punto Fijo", "Newton-Raphson"])

    # ─── BISECCIÓN ───
    with tab_bis:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Parámetros de Bisección**")
            st.success("Requiere f(a)·f(b) < 0 (signos opuestos en el intervalo)")

            f_expr = st.text_input("f(x) =", value=default_func, key="bis_f", help="Ej: x**3 - x - 2, sin(x) - x/2, exp(x) - 3")
            col_a, col_b = st.columns(2)
            with col_a:
                a_val = st.number_input("a", value=float(default_a), step=0.1, key="bis_a")
            with col_b:
                b_val = st.number_input("b", value=float(default_b), step=0.1, key="bis_b")

            tol = st.number_input("Tolerancia", value=float(default_tol), format="%.2e", min_value=1e-15, max_value=0.1, key="bis_tol")
            max_it = st.slider("Máx. iteraciones", 5, 200, 100, key="bis_max")

            if st.button("▶ Calcular Bisección", type="primary", key="run_bis"):
                with st.spinner("Calculando..."):
                    result = biseccion(f_expr, a_val, b_val, tol, max_it)
                st.session_state['bis_result'] = result
                st.session_state['bis_expr'] = f_expr
                st.session_state['bis_interval'] = (a_val, b_val)

        with col2:
            result = st.session_state.get('bis_result')
            if result:
                if result['success'] and result['root'] is not None:
                    show_result_box("Raíz encontrada", f"{result['root']:.10f}")
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric("Iteraciones", result.get('num_iterations', '—'))
                    col_r2.metric("Error final", f"{result.get('final_error', 0):.2e}")
                    col_r3.metric("Convergió", "✅" if result['converged'] else "⚠️")
                    st.info(result.get('message', ''))
                else:
                    show_error(result.get('error', 'Error desconocido'))

        if st.session_state.get('bis_result') and st.session_state['bis_result']['success']:
            r = st.session_state['bis_result']
            expr = st.session_state.get('bis_expr', f_expr)
            intv = st.session_state.get('bis_interval', (a_val, b_val))

            tab_iter, tab_graf = st.tabs(["Tabla de iteraciones", "Gráfica"])
            with tab_iter:
                show_iteration_table(r.get('iterations', []))
            with tab_graf:
                with st.spinner("Generando gráfica..."):
                    img = plot_root_method(expr, r.get('iterations', []), r['root'], "Bisección", intv)
                show_plot_bytes(img)

    # ─── PUNTO FIJO ───
    with tab_pf:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Parámetros de Punto Fijo**")
            st.success("Expresar como x = g(x). La convergencia requiere |g'(x)| < 1.")

            g_expr = st.text_input("g(x) =", value='(x + 2/x)/2', key="pf_g",
                                    help="Ej: (x + 2/x)/2, (x**2 + 2)/(2*x + 1)")
            x0_pf = st.number_input("x₀ (valor inicial)", value=1.5, step=0.1, key="pf_x0")
            tol_pf = st.number_input("Tolerancia", value=1e-6, format="%.2e", min_value=1e-15, key="pf_tol")
            max_it_pf = st.slider("Máx. iteraciones", 5, 200, 100, key="pf_max")

            if st.button("▶ Calcular Punto Fijo", type="primary", key="run_pf"):
                with st.spinner("Calculando..."):
                    result = punto_fijo(g_expr, x0_pf, tol_pf, max_it_pf)
                st.session_state['pf_result'] = result
                st.session_state['pf_expr'] = g_expr

        with col2:
            result = st.session_state.get('pf_result')
            if result:
                if result['success'] and result['root'] is not None:
                    show_result_box("Punto fijo encontrado", f"{result['root']:.10f}")
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric("Iteraciones", result.get('num_iterations', '—'))
                    col_r2.metric("Error final", f"{result.get('final_error', 0):.2e}")
                    col_r3.metric("Convergió", "✅" if result['converged'] else "⚠️")
                    st.info(result.get('message', ''))
                else:
                    show_error(result.get('error', 'Error desconocido'))

        if st.session_state.get('pf_result') and st.session_state['pf_result']['success']:
            r = st.session_state['pf_result']
            tab_iter, tab_graf = st.tabs(["Tabla de iteraciones", "Gráfica"])
            with tab_iter:
                show_iteration_table(r.get('iterations', []))
            with tab_graf:
                g_expr_plot = st.session_state.get('pf_expr', g_expr)
                with st.spinner("Generando gráfica..."):
                    img = plot_root_method(g_expr_plot, r.get('iterations', []), r['root'], "Punto Fijo")
                show_plot_bytes(img)

    # ─── NEWTON-RAPHSON ───
    with tab_nr:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Parámetros de Newton-Raphson**")
            st.success("La derivada se calcula automáticamente con SymPy si no la ingresás.")

            f_nr = st.text_input("f(x) =", value=default_func, key="nr_f")
            df_nr = st.text_input("f'(x) = (opcional, se calcula si está vacío)",
                                   value='', key="nr_df",
                                   help="Dejar vacío para cálculo automático")
            x0_nr = st.number_input("x₀ (valor inicial)", value=float(default_a), step=0.1, key="nr_x0")
            tol_nr = st.number_input("Tolerancia", value=1e-6, format="%.2e",
                                      min_value=1e-15, key="nr_tol")
            max_it_nr = st.slider("Máx. iteraciones", 5, 100, 50, key="nr_max")

            # Mostrar derivada calculada
            if f_nr:
                try:
                    deriv_preview = get_derivative(f_nr)
                    st.markdown(f"**Derivada calculada:** `f'(x) = {deriv_preview}`")
                except Exception:
                    pass

            if st.button("▶ Calcular Newton-Raphson", type="primary", key="run_nr"):
                df_input = df_nr.strip() if df_nr.strip() else None
                with st.spinner("Calculando..."):
                    result = newton_raphson(f_nr, x0_nr, tol_nr, max_it_nr, df_input)
                st.session_state['nr_result'] = result
                st.session_state['nr_expr'] = f_nr

        with col2:
            result = st.session_state.get('nr_result')
            if result:
                if result['success'] and result['root'] is not None:
                    show_result_box("Raíz encontrada", f"{result['root']:.10f}")
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric("Iteraciones", result.get('num_iterations', '—'))
                    col_r2.metric("Error final", f"{result.get('final_error', 0):.2e}")
                    col_r3.metric("Convergió", "✅" if result['converged'] else "⚠️")
                    if result.get('derivative_used'):
                        st.markdown(f"**Derivada usada:** `{result['derivative_used']}`")
                    st.info(result.get('message', ''))
                else:
                    show_error(result.get('error', 'Error desconocido'))

        if st.session_state.get('nr_result') and st.session_state['nr_result']['success']:
            r = st.session_state['nr_result']
            expr = st.session_state.get('nr_expr', f_nr)
            tab_iter, tab_graf = st.tabs(["Tabla de iteraciones", "Gráfica"])
            with tab_iter:
                show_iteration_table(r.get('iterations', []))
            with tab_graf:
                with st.spinner("Generando gráfica..."):
                    img = plot_root_method(expr, r.get('iterations', []),
                                           r['root'], "Newton-Raphson")
                show_plot_bytes(img)


# ─── SECCIÓN INTEGRALES ───────────────────────────────────────────────────────

def section_integrals():
    st.markdown("### ∫ Integración Numérica")

    ocr_parsed = st.session_state.get('ocr_parsed', {})
    default_func = ocr_parsed.get('function') or 'x**2 + 1'
    default_a = ocr_parsed.get('interval', (0.0, 1.0))[0] if ocr_parsed.get('interval') else 0.0
    default_b = ocr_parsed.get('interval', (0.0, 1.0))[1] if ocr_parsed.get('interval') else 1.0

    tab_trap, tab_simp13, tab_simp38 = st.tabs(["Trapecio", "Simpson 1/3", "Simpson 3/8"])

    for tab, method_name, method_func, default_n, n_key in [
        (tab_trap,   "Trapecio",   trapecio,   100, "trap"),
        (tab_simp13, "Simpson 1/3", simpson_13, 100, "s13"),
        (tab_simp38, "Simpson 3/8", simpson_38, 99,  "s38"),
    ]:
        with tab:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**Parámetros — {method_name}**")
                if method_name == "Simpson 1/3":
                    st.success("n debe ser par. Se ajustará automáticamente si es impar.")
                elif method_name == "Simpson 3/8":
                    show_info("n debe ser múltiplo de 3. Se ajustará automáticamente.")

                f_int = st.text_input("f(x) =", value=default_func, key=f"{n_key}_f")
                col_a, col_b = st.columns(2)
                with col_a:
                    a_int = st.number_input("a (límite inferior)", value=float(default_a), step=0.1, key=f"{n_key}_a")
                with col_b:
                    b_int = st.number_input("b (límite superior)", value=float(default_b), step=0.1, key=f"{n_key}_b")
                n_int = st.slider("Número de subintervalos (n)", 2, 1000, default_n, key=f"{n_key}_n")

                if st.button(f"▶ Calcular {method_name}", type="primary", key=f"run_{n_key}"):
                    with st.spinner("Calculando..."):
                        result = method_func(f_int, a_int, b_int, n_int)
                    st.session_state[f'{n_key}_result'] = result
                    st.session_state[f'{n_key}_params'] = (f_int, a_int, b_int, n_int)

            with col2:
                result = st.session_state.get(f'{n_key}_result')
                if result:
                    if result['success']:
                        show_result_box("Resultado de la integral", f"{result['result']:.10f}")
                        if result.get('exact') is not None:
                            col_e1, col_e2 = st.columns(2)
                            col_e1.metric("Valor exacto (SymPy)", f"{result['exact']:.10f}")
                            if result.get('error_relative_pct') is not None:
                                col_e2.metric("Error relativo", f"{result['error_relative_pct']:.4f}%")
                        st.markdown(f"*{result.get('formula', '')}*")
                        st.info(result.get('message', ''))
                    else:
                        show_error(result.get('error', 'Error desconocido'))

            if st.session_state.get(f'{n_key}_result') and st.session_state[f'{n_key}_result']['success']:
                r = st.session_state[f'{n_key}_result']
                params = st.session_state.get(f'{n_key}_params', (f_int, a_int, b_int, n_int))

                tab_tb, tab_gr = st.tabs(["Tabla de puntos", "Gráfica"])
                with tab_tb:
                    if r.get('table'):
                        df = pd.DataFrame(r['table'])
                        st.dataframe(df, use_container_width=True)
                with tab_gr:
                    with st.spinner("Generando gráfica..."):
                        img = plot_integral(params[0], params[1], params[2],
                                            method_name, params[3], r['result'])
                    show_plot_bytes(img)


# ─── SECCIÓN SUMATORIAS ───────────────────────────────────────────────────────

def section_series():
    st.markdown("### Σ Sumatorias y Series")

    tab_gen, tab_geom, tab_taylor = st.tabs(["Sumatoria general", "Serie geométrica", "Serie de Taylor"])

    with tab_gen:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Sumatoria: Σ expr(n), n = inicio..fin**")
            st.success("Usá 'n' como variable. Ejemplo: 1/n**2, n*(n+1)/2, (-1)**n/n")

            expr_s = st.text_input("Expresión a sumar", value='1/n**2', key="sum_expr")
            var_s = st.text_input("Variable", value='n', key="sum_var")
            col_s, col_e = st.columns(2)
            with col_s:
                start_s = st.number_input("Desde (n =)", value=1, step=1, key="sum_start")
            with col_e:
                end_input = st.text_input("Hasta (n =)", value='10', key="sum_end", help="Número entero o 'inf' para series infinitas")

            if st.button("▶ Calcular Sumatoria", type="primary", key="run_sum"):
                try:
                    if end_input.strip().lower() in ('inf', '∞', 'infinity'):
                        end_val = sp.oo
                    else:
                        end_val = int(end_input)
                except ValueError:
                    show_error("El valor final debe ser un número entero o 'inf'")
                    return

                try:
                    with st.spinner("Calculando..."):
                        result = calcular_sumatoria(expr_s, var_s, int(start_s), end_val)
                    st.session_state['sum_result'] = result
                except Exception as e:
                    show_error(f"Error al calcular la sumatoria: {str(e)}")

        with col2:
            result = st.session_state.get('sum_result')
            if result:
                if result['success']:
                    show_result_box("Resultado", f"{result['result']:.10f}")
                    if result.get('exact_symbolic') and result['exact_symbolic'] != 'No disponible':
                        st.markdown(f"**Forma exacta:** `{result['exact_symbolic']}`")
                    if result.get('exact_value') is not None:
                        st.metric("Valor exacto", f"{result['exact_value']:.10f}")
                    if result.get('convergence'):
                        conv = result['convergence']
                        st.markdown(f"**Convergencia:** {conv.get('verdict', '—')}")
                    st.info(result.get('message', ''))
                else:
                    show_error(result.get('error', 'Error desconocido'))

        if st.session_state.get('sum_result') and st.session_state['sum_result']['success']:
            r = st.session_state['sum_result']
            tab_tb, tab_gr = st.tabs(["Tabla de términos", "Gráfica"])
            with tab_tb:
                if r.get('table'):
                    df = pd.DataFrame(r['table'])
                    st.dataframe(df, use_container_width=True)
            with tab_gr:
                with st.spinner("Generando gráfica..."):
                    img = plot_series(r.get('table', []), "Sumatoria")
                show_plot_bytes(img)

    with tab_geom:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Serie geométrica: Σ a·r^k**")
            a_g = st.number_input("a (primer término)", value=1.0, step=0.1, key="geom_a")
            r_g = st.number_input("r (razón)", value=0.5, step=0.1, key="geom_r")
            n_g = st.slider("Número de términos", 2, 100, 10, key="geom_n")

            if st.button("▶ Calcular Serie Geométrica", type="primary", key="run_geom"):
                with st.spinner("Calculando..."):
                    result = series_geometrica(a_g, r_g, n_g)
                st.session_state['geom_result'] = result

        with col2:
            result = st.session_state.get('geom_result')
            if result and result['success']:
                show_result_box("Suma finita", f"{result['result']:.8f}")
                if result.get('infinite_sum') is not None:
                    st.metric("Suma infinita exacta", f"{result['infinite_sum']:.8f}")
                st.info(result.get('message', ''))

        if st.session_state.get('geom_result') and st.session_state['geom_result']['success']:
            r = st.session_state['geom_result']
            tab_tb, tab_gr = st.tabs(["Tabla", "Gráfica"])
            with tab_tb:
                df = pd.DataFrame(r.get('table', []))
                st.dataframe(df, use_container_width=True)
            with tab_gr:
                img = plot_series(r.get('table', []), "Serie Geométrica")
                show_plot_bytes(img)

    with tab_taylor:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Serie de Taylor**")
            f_tay = st.text_input("f(x) =", value='sin(x)', key="tay_f")
            punto_tay = st.number_input("Punto de expansión (x₀)", value=0.0, step=0.1, key="tay_pt")
            grado_tay = st.slider("Grado del polinomio", 1, 15, 5, key="tay_deg")
            x_eval_tay = st.text_input("Evaluar en x = (opcional)", value='', key="tay_eval")

            if st.button("▶ Calcular Taylor", type="primary", key="run_tay"):
                try:
                    x_eval_val = float(x_eval_tay) if x_eval_tay.strip() else None
                except ValueError:
                    show_error("El valor de x debe ser numérico")
                    return

                try:
                    with st.spinner("Calculando..."):
                        result = series_taylor(f_tay, punto_tay, grado_tay, x_eval_val)
                    st.session_state['tay_result'] = result
                except Exception as e:
                    show_error(f"Error al calcular la serie de Taylor: {str(e)}")

        with col2:
            result = st.session_state.get('tay_result')
            if result:
                if result.get('success'):
                    st.markdown("**Polinomio de Taylor:**")
                    st.code(f"P(x) = {result['polynomial']}", language="python")
                    if result.get('eval_value') is not None:
                        show_result_box(f"P({result['eval_x']})", f"{result['eval_value']:.8f}")
                        col_t1, col_t2 = st.columns(2)
                        col_t1.metric("Valor exacto f(x)", f"{result['exact_value']:.8f}")
                        col_t2.metric("Error de aproximación", f"{result['error']:.2e}")
                else:
                    show_error(result.get('error', 'Error'))


# ─── SECCIÓN INTERPOLACIÓN ───────────────────────────────────────────────────

def section_interpolation():
    st.markdown("### 📈 Interpolación")

    ocr_parsed = st.session_state.get('ocr_parsed', {})
    default_points = ""
    if ocr_parsed.get('points'):
        pts = ocr_parsed['points']
        default_points = "; ".join([f"{x},{y}" for x, y in zip(pts['x'], pts['y'])])
    if not default_points:
        default_points = "0,1; 1,3; 2,7; 3,13"

    tab_lag, tab_nd, tab_nf = st.tabs(["Lagrange", "Newton Dif. Divididas", "Newton Dif. Finitas"])

    for tab, method_name, method_func, method_key in [
        (tab_lag, "Lagrange",                lagrange,                         "lag"),
        (tab_nd,  "Newton Dif. Divididas",    newton_diferencias_divididas,     "nd"),
        (tab_nf,  "Newton Dif. Finitas",      newton_diferencias_finitas,       "nf"),
    ]:
        with tab:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**Parámetros — {method_name}**")
                if method_name == "Newton Dif. Finitas":
                    st.success("Requiere puntos igualmente espaciados en x.")

                points_input = st.text_area(
                    "Puntos (formato: x1,y1; x2,y2; ...)",
                    value=default_points,
                    height=100,
                    key=f"{method_key}_pts",
                    help="Ej: 0,1; 1,3; 2,7; 3,13"
                )
                x_eval_str = st.text_input("Interpolar en x =", value='1.5', key=f"{method_key}_xev")

            if st.button(f"▶ Calcular {method_name}", type="primary", key=f"run_{method_key}"):
                pts = parse_points_input(points_input)

                if pts is None:
                    show_error("Formato incorrecto. Usar: x1,y1; x2,y2; ...")
                else:
                    try:
                        x_eval_val = float(x_eval_str) if x_eval_str.strip() else None
                    except ValueError:
                        show_error("El valor de x debe ser numérico")
                        return

                    try:
                        with st.spinner("Calculando..."):
                            result = method_func(pts[0], pts[1], x_eval_val)

                        st.session_state[f'{method_key}_result'] = result
                        st.session_state[f'{method_key}_pts'] = pts

                    except Exception as e:
                        show_error(f"Error en el cálculo: {str(e)}")

            with col2:
                result = st.session_state.get(f'{method_key}_result')
                if result:
                    if result['success']:
                        st.markdown("**Polinomio interpolador:**")
                        st.code(f"P(x) = {result['polynomial']}", language="python")
                        if result.get('eval_value') is not None:
                            show_result_box(f"P({result['eval_x']})", f"{result['eval_value']:.8f}")
                        st.info(result.get('message', ''))
                    else:
                        show_error(result.get('error', 'Error desconocido'))

            if st.session_state.get(f'{method_key}_result') and \
                st.session_state[f'{method_key}_result']['success']:
                r = st.session_state[f'{method_key}_result']
                pts_saved = st.session_state.get(f'{method_key}_pts')

                show_tabs = ["Tabla de diferencias", "Gráfica"]
                tb_dif, tb_gr = st.tabs(show_tabs)

                with tb_dif:
                    table_key = 'difference_table' if 'difference_table' in r else 'table'
                    if r.get(table_key):
                        df = pd.DataFrame(r[table_key])
                        st.dataframe(df, use_container_width=True)

                with tb_gr:
                    if pts_saved:
                        with st.spinner("Generando gráfica..."):
                            img = plot_interpolation(
                                pts_saved[0], pts_saved[1],
                                r['polynomial'], method_name,
                                r.get('eval_x')
                            )
                        show_plot_bytes(img)


# ─── SECCIÓN AYUDA / REFERENCIA ───────────────────────────────────────────────

def section_help():
    st.markdown("### 📚 Referencia rápida")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("🔤 Sintaxis de funciones", expanded=True):
            st.markdown("""
| Escribir | Significa |
|----------|-----------|
| `x**2` | x² |
| `x**3` | x³ |
| `sqrt(x)` | √x |
| `exp(x)` | eˣ |
| `log(x)` | ln(x) |
| `log(x, 10)` | log₁₀(x) |
| `sin(x)`, `cos(x)`, `tan(x)` | sen, cos, tan |
| `pi` | π ≈ 3.14159 |
| `E` | e ≈ 2.71828 |
| `abs(x)` | \|x\| |
| `factorial(n)` | n! |
| `1/x` | 1/x (fracciones) |
""")

        with st.expander("📐 Guía de métodos de raíces"):
            st.markdown("""
**Bisección**
- Garantiza convergencia si f(a)·f(b) < 0
- Convergencia lineal, más lento
- Bueno para encontrar el intervalo correcto

**Punto Fijo**
- Reescribir f(x) = 0 como x = g(x)
- Converge si |g'(x)| < 1
- Puede divergir si la condición no se cumple

**Newton-Raphson**
- Convergencia cuadrática (muy rápido)
- Requiere f'(x) ≠ 0 en la vecindad
- Puede oscilar si x₀ está lejos de la raíz
""")

    with col2:
        with st.expander("∫ Guía de integración"):
            st.markdown("""
**Regla del Trapecio**
- Error: O(h²), donde h = (b-a)/n
- Funciona para cualquier función continua
- n = 100 da buena precisión en general

**Simpson 1/3**
- Error: O(h⁴), más preciso que Trapecio
- n debe ser par
- Mejor para funciones suaves

**Simpson 3/8**
- Similar a 1/3 pero n múltiplo de 3
- Útil cuando los datos tienen ese espaciado
""")

        with st.expander("📈 Guía de interpolación"):
            st.markdown("""
**Lagrange**
- No requiere espaciado uniforme
- Construye explícitamente L_i(x)
- Cuidado con el fenómeno de Runge en grados altos

**Newton Diferencias Divididas**
- No requiere espaciado uniforme
- Fácil de actualizar con nuevos puntos
- Forma jerárquica del polinomio

**Newton Diferencias Finitas**
- Requiere espaciado uniforme en x
- Más eficiente computacionalmente
- Usa el parámetro s = (x-x₀)/h
""")

    st.markdown("---")
    st.markdown("### ℹ️ Sobre la aplicación")
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown("**Tecnologías**\n- Streamlit\n- NumPy / SciPy\n- SymPy\n- Matplotlib\n- Tesseract OCR")
    col_b.markdown("**Métodos incluidos**\n- Bisección\n- Punto Fijo\n- Newton-Raphson\n- Trapecio / Simpson\n- Sumatorias / Taylor\n- Lagrange / Newton")
    col_c.markdown("**Materia**\n- Análisis Numérico\n\n**OCR requiere**\n- Tesseract instalado\n- `pip install pytesseract`")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Inicializar session state
    if 'ocr_parsed' not in st.session_state:
        st.session_state['ocr_parsed'] = {}

    # Header
    st.markdown('<div class="main-title">🔢 Análisis Numérico</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Calculadora de métodos numéricos con reconocimiento de imágenes</div>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🧮 Navegación")
        st.markdown("---")
        st.markdown("**Subir imagen** → OCR detecta el problema")
        st.markdown("**Calculadora** → Ingresar manualmente")
        st.markdown("---")
        st.markdown("### Funciones de ejemplo")
        examples = {
            "x³ - x - 2": "x**3 - x - 2",
            "sin(x) - x/2": "sin(x) - x/2",
            "e^x - 3": "exp(x) - 3",
            "x² - 4": "x**2 - 4",
            "cos(x) - x": "cos(x) - x",
        }
        for label, expr in examples.items():
            if st.button(label, key=f"ex_{label}"):
                st.session_state['quick_expr'] = expr
                st.info(f"Cargado: {expr}")

        st.markdown("---")
        if st.session_state.get('ocr_parsed', {}).get('function'):
            st.success(f"📷 OCR cargado:\n`{st.session_state['ocr_parsed']['function']}`")

    # Tabs principales
    tab_ocr, tab_roots, tab_integ, tab_series, tab_interp, tab_help = st.tabs([
        "📷 OCR / Imagen",
        "🎯 Raíces",
        "∫ Integrales",
        "Σ Sumatorias",
        "📈 Interpolación",
        "📚 Ayuda",
    ])


    with tab_ocr:
        section_ocr()

    with tab_roots:
        section_roots()

    with tab_integ:
        section_integrals()

    with tab_series:
        section_series()

    with tab_interp:
        section_interpolation()

    with tab_help:
        section_help()


if __name__ == "__main__":
    main()
