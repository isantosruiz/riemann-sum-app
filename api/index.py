import base64
import io
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
import sympy as sp
from flask import Flask, render_template, request
from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

x = sp.symbols("x", real=True)
n = sp.symbols("n", integer=True, positive=True)
i = sp.symbols("i", integer=True)

METHODS = {
    "left": "Izquierda",
    "right": "Derecha",
    "midpoint": "Centro",
    "trapezoidal": "Trapezoidal",
}

METHOD_DESCRIPTORS = {
    "left": "por la izquierda",
    "right": "por la derecha",
    "midpoint": "por el centro",
    "trapezoidal": "trapezoidal",
}

ALLOWED_SYMBOLS = {
    "x": x,
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "log10": lambda arg: sp.log(arg, 10),
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "sin": sp.sin,
    "sen": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "acot": sp.acot,
    "asec": sp.asec,
    "acsc": sp.acsc,
    "sinh": sp.sinh,
    "senh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "coth": sp.coth,
    "sech": sp.sech,
    "csch": sp.csch,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "acoth": sp.acoth,
    "asech": sp.asech,
    "acsch": sp.acsch,
}

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)
PARSER_GLOBALS = {
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
}


def parse_expression(expr_text: str) -> sp.Expr:
    return parse_expr(
        expr_text,
        local_dict=ALLOWED_SYMBOLS,
        global_dict=PARSER_GLOBALS,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def symbolic_riemann_sum_components(
    expr: sp.Expr, a: sp.Expr, b: sp.Expr, method: str
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    delta = (b - a) / n

    if method == "left":
        f_xi = sp.simplify(expr.subs(x, a + (i - 1) * delta))
        term = sp.simplify(f_xi * delta)
        sum_notation = sp.Sum(term, (i, 1, n))
        return sum_notation, sp.simplify(sp.summation(term, (i, 1, n))), f_xi, sp.simplify(delta)

    if method == "right":
        f_xi = sp.simplify(expr.subs(x, a + i * delta))
        term = sp.simplify(f_xi * delta)
        sum_notation = sp.Sum(term, (i, 1, n))
        return sum_notation, sp.simplify(sp.summation(term, (i, 1, n))), f_xi, sp.simplify(delta)

    if method == "midpoint":
        f_xi = sp.simplify(expr.subs(x, a + (i - sp.Rational(1, 2)) * delta))
        term = sp.simplify(f_xi * delta)
        sum_notation = sp.Sum(term, (i, 1, n))
        return sum_notation, sp.simplify(sp.summation(term, (i, 1, n))), f_xi, sp.simplify(delta)

    if method == "trapezoidal":
        f_xi = sp.simplify(
            (
                expr.subs(x, a + (i - 1) * delta)
                + expr.subs(x, a + i * delta)
            )
            / 2
        )
        term = sp.simplify(f_xi * delta)
        sum_notation = sp.Sum(term, (i, 1, n))
        return sum_notation, sp.simplify(sp.summation(term, (i, 1, n))), f_xi, sp.simplify(delta)

    raise ValueError("Método no soportado.")


def symbolic_riemann_sum(expr: sp.Expr, a: sp.Expr, b: sp.Expr, method: str) -> sp.Expr:
    _, result, _, _ = symbolic_riemann_sum_components(expr, a, b, method)
    return result


def eval_real_values(expr: sp.Expr, points: np.ndarray) -> np.ndarray:
    values = []
    for point in points:
        value = sp.N(expr.subs(x, sp.Float(point)))
        if value.is_real is False:
            values.append(np.nan)
            continue
        try:
            as_complex = complex(value.evalf())
        except Exception:
            values.append(np.nan)
            continue
        if abs(as_complex.imag) > 1e-10:
            values.append(np.nan)
            continue
        values.append(float(as_complex.real))
    return np.array(values, dtype=float)


def build_plot(expr: sp.Expr, a: sp.Expr, b: sp.Expr, n_partitions: int, method: str) -> str:
    a_float = float(sp.N(a))
    b_float = float(sp.N(b))
    edges = np.linspace(a_float, b_float, n_partitions + 1)

    x_curve = np.linspace(a_float, b_float, 800)
    y_curve = eval_real_values(expr, x_curve)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color="#4d4d4d", linewidth=1)
    ax.plot(x_curve, y_curve, color="#0b1f3a", linewidth=2, label="f(x)")

    if method in {"left", "right", "midpoint"}:
        if method == "left":
            sample_points = edges[:-1]
        elif method == "right":
            sample_points = edges[1:]
        else:
            sample_points = (edges[:-1] + edges[1:]) / 2

        heights = eval_real_values(expr, sample_points)
        for x0, x1, h in zip(edges[:-1], edges[1:], heights):
            if np.isnan(h):
                continue
            ax.fill(
                [x0, x0, x1, x1],
                [0, h, h, 0],
                color="#5fb3da",
                alpha=0.68,
                edgecolor="#219ebc",
                linewidth=1,
            )

    elif method == "trapezoidal":
        left_heights = eval_real_values(expr, edges[:-1])
        right_heights = eval_real_values(expr, edges[1:])
        for x0, x1, h0, h1 in zip(edges[:-1], edges[1:], left_heights, right_heights):
            if np.isnan(h0) or np.isnan(h1):
                continue
            ax.fill(
                [x0, x0, x1, x1],
                [0, h0, h1, 0],
                color="#5fb3da",
                alpha=0.68,
                edgecolor="#219ebc",
                linewidth=1,
            )
            ax.plot([x0, x1], [h0, h1], color="#219ebc", linewidth=1.2)

    ax.set_xlim(a_float, b_float)
    ax.set_title(f"Suma de Riemann ({METHODS[method]}) con n={n_partitions}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def unresolved_parameters(expr: sp.Expr, a: sp.Expr, b: sp.Expr) -> list[str]:
    return sorted(str(sym) for sym in ((expr.free_symbols | a.free_symbols | b.free_symbols) - {x}))


def numeric_approximation(expr: sp.Expr) -> sp.Float | None:
    try:
        value = sp.N(expr, 12)
        if value.free_symbols or value.is_real is False:
            return None
        as_complex = complex(value.evalf())
        if abs(as_complex.imag) > 1e-10:
            return None
        return sp.Float(as_complex.real, 12)
    except Exception:
        return None


def should_show_approximation(exact_value: sp.Expr, approx_value: sp.Float | None) -> bool:
    if approx_value is None:
        return False
    if not exact_value.free_symbols and exact_value.is_integer is True:
        return False
    try:
        as_float = float(approx_value)
        if abs(as_float - round(as_float)) < 1e-10:
            return False
    except Exception:
        return True
    return True


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "function_input": "x^2",
        "a_input": "-2",
        "b_input": "2",
        "n_input": "8",
        "method_input": "right",
        "action_input": "calculate",
        "limit_requested": False,
        "methods": METHODS,
        "error": None,
        "latex_problem_statement": None,
        "latex_fx_i": None,
        "latex_delta_x": None,
        "latex_symbolic": None,
        "latex_n_value": None,
        "latex_limit_value": None,
        "limit_error": None,
        "symbolic_notice": None,
        "plot_warning": None,
        "plot_base64": None,
        "n_partitions": None,
    }

    if request.method == "POST":
        context["function_input"] = request.form.get("function", "").strip()
        context["a_input"] = request.form.get("a", "").strip()
        context["b_input"] = request.form.get("b", "").strip()
        context["n_input"] = request.form.get("n", "").strip()
        context["method_input"] = request.form.get("method", "right")
        context["action_input"] = request.form.get("action", "calculate")
        context["limit_requested"] = context["action_input"] == "limit"

        try:
            if context["method_input"] not in METHODS:
                raise ValueError("Método inválido.")

            expr = parse_expression(context["function_input"])
            a = parse_expression(context["a_input"])
            b = parse_expression(context["b_input"])

            n_partitions = int(context["n_input"])
            if n_partitions <= 0:
                raise ValueError("El número de particiones debe ser mayor que cero.")

            symbolic_sum_notation, symbolic_sum, f_xi_expr, delta = symbolic_riemann_sum_components(
                expr, a, b, context["method_input"]
            )
            n_sum = sp.simplify(symbolic_sum.subs(n, n_partitions))
            unresolved = unresolved_parameters(expr, a, b)

            descriptor = METHOD_DESCRIPTORS[context["method_input"]]
            partition_word = (
                "trapecios" if context["method_input"] == "trapezoidal" else "rectángulos"
            )
            context["latex_problem_statement"] = (
                f"\\text{{Suma de Riemann {descriptor} de }} f(x) = {sp.latex(expr)}"
                f"\\text{{ en }} [{sp.latex(a)}, {sp.latex(b)}]"
                f"\\text{{ con {n_partitions} {partition_word}.}}"
            )

            x_i = sp.Symbol("x_i")
            context["latex_fx_i"] = sp.latex(sp.Eq(sp.Function("f")(x_i), f_xi_expr))
            context["latex_delta_x"] = "\\Delta x = " + sp.latex(delta)

            lhs_symbolic = sp.latex(sp.Symbol("S_n"))
            generic_sum = "\\sum_{i=1}^{n} f\\left(x_i\\right)\\,\\Delta x"
            context["latex_symbolic"] = (
                f"{lhs_symbolic} = {generic_sum} = {sp.latex(symbolic_sum_notation)} = {sp.latex(symbolic_sum)}"
            )
            lhs_n = sp.latex(sp.Symbol(f"S_{n_partitions}"))
            rhs_n = sp.latex(n_sum)
            approx_value = None if unresolved else numeric_approximation(n_sum)
            if should_show_approximation(n_sum, approx_value):
                context["latex_n_value"] = f"{lhs_n} = {rhs_n} \\approx {sp.latex(approx_value)}"
            else:
                context["latex_n_value"] = f"{lhs_n} = {rhs_n}"
            context["n_partitions"] = n_partitions

            if context["limit_requested"]:
                try:
                    limit_value = sp.simplify(sp.limit(symbolic_sum, n, sp.oo))
                    limit_eq = sp.Eq(sp.Limit(sp.Symbol("S_n"), n, sp.oo), limit_value, evaluate=False)
                    context["latex_limit_value"] = sp.latex(limit_eq)
                except Exception:
                    context["limit_error"] = (
                        "No se pudo calcular el límite de S_n cuando n tiende a infinito."
                    )

            if unresolved:
                context["symbolic_notice"] = (
                    "Se detectaron parámetros simbólicos ("
                    + ", ".join(unresolved)
                    + "). Se muestra solo el resultado simbólico."
                )
                context["plot_warning"] = (
                    "No se puede dibujar la suma de Riemann sin valores numéricos para todos los parámetros."
                )
            else:
                a_numeric = sp.N(a)
                b_numeric = sp.N(b)
                if a_numeric.is_real is False or b_numeric.is_real is False:
                    raise ValueError("Los límites a y b deben ser reales.")
                if float(a_numeric) >= float(b_numeric):
                    raise ValueError("Debe cumplirse que a < b.")
                context["plot_base64"] = build_plot(expr, a, b, n_partitions, context["method_input"])

        except Exception as exc:
            context["error"] = str(exc)

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
