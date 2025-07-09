# integracion_interactiva.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify
from scipy.integrate import quad
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_image("logo_tecnm.png")

st.markdown(f"""
    <div style='text-align: center; margin-top: 10px; margin-bottom: -20px;'>
        <img src="data:image/png;base64,{img_base64}" width="250">
    </div>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Métodos de Integración Numérica", layout="centered")

st.markdown("""
    <h1 style='text-align: center;'> ∫ Métodos de Integración Numérica ∫ </h1>
    <h3 style='text-align: center; color: gray;'>Instituto Tecnológico Nacional de México</h3>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center; color: gray; font-size: 12px; margin-top: 20px;'>"
    "Desarrollado en el Laboratorio de Simulación Matematica del Instituto Tecnológico de Morelia por el Ing. Juan David López Regalado"
    "</p>",
    unsafe_allow_html=True
)


# Entrada de usuario
func_str = st.text_input("Escribe la función f(x):", value="x**2")
a = st.number_input("Límite inferior (a):", value=0.0)
b = st.number_input("Límite superior (b):", value=2.0)
n = st.slider("Número de subintervalos (n):", min_value=1, max_value=100, step=1, value=10)

method = st.selectbox("Selecciona el método:", ["Riemann Izquierda", "Riemann Derecha", "Punto Medio", "Trapecio", "Simpson"])

with st.expander("Ver teoría del método seleccionado..."):
    if method == "Riemann Izquierda":
        st.markdown(r"""
        ### 🟦 Suma de Riemann por la izquierda
        Aproxima el área bajo la curva usando rectángulos.

        La altura del rectángulo es **f(xₖ)** donde **xₖ** es el extremo izquierdo de cada subintervalo.

        **Fórmula:**
        $$
        \int_a^b f(x)\,dx \approx \sum_{k=0}^{n-1} f(x_k)\cdot\Delta x
        $$
        """)

    elif method == "Riemann Derecha":
        st.markdown(r"""
        ### 🟥 Suma de Riemann por la derecha
        Usa el valor de f(x) en el extremo derecho de cada subintervalo.

        **Fórmula:**
        $$
        \int_a^b f(x)\,dx \approx \sum_{k=1}^{n} f(x_k)\cdot\Delta x
        $$
        """)

    elif method == "Punto Medio":
        st.markdown(r"""
        ### 🟨 Método del Punto Medio
        La altura del rectángulo es **f(mₖ)** donde **mₖ** es el punto medio de cada subintervalo.

        **Fórmula:**
        $$
        \int_a^b f(x)\,dx \approx \sum_{k=0}^{n-1} f\left(\frac{x_k + x_{k+1}}{2}\right)\cdot\Delta x
        $$
        """)

    elif method == "Trapecio":
        st.markdown(r"""
        ### 🔺 Método del Trapecio
        Se usa un trapecio en lugar de un rectángulo para cada subintervalo.

        **Fórmula general:**
        $$
        \int_a^b f(x)\,dx \approx \frac{\Delta x}{2} \left[f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right]
        $$
        """)

    elif method == "Simpson":
        st.markdown(r"""
        ### 🟣 Método de Simpson 1/3
        Aproxima la curva con parábolas. Requiere que n sea par.

        **Fórmula:**
        $$
        \int_a^b f(x)\,dx \approx \frac{\Delta x}{3} \left[f(x_0) + 4\sum_{i=1,\,3,\,5,\dots}^{n-1} f(x_i) + 2\sum_{i=2,\,4,\,6,\dots}^{n-2} f(x_i) + f(x_n)\right]
        $$
        """)
x = symbols('x')
try:
    func_expr = sympify(func_str)
    f = lambdify(x, func_expr, 'numpy')
    X = np.linspace(a, b, n+1)
    dx = (b - a) / n
    approx = 0

    if method == "Riemann Izquierda":
        approx = np.sum(f(X[:-1]) * dx)
    elif method == "Riemann Derecha":
        approx = np.sum(f(X[1:]) * dx)
    elif method == "Punto Medio":
        midpoints = (X[:-1] + X[1:]) / 2
        approx = np.sum(f(midpoints) * dx)
    elif method == "Trapecio":
        approx = (dx / 2) * np.sum(f(X[:-1]) + f(X[1:]))
    elif method == "Simpson":
        if n % 2 != 0:
            st.warning("⚠️ Simpson requiere un número par de subintervalos.")
            approx = np.nan
        else:
            x_simp = np.linspace(a, b, n+1)
            y = f(x_simp)
            approx = dx / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

    # Valor exacto con scipy
    exact, _ = quad(f, a, b)
    error = abs(approx - exact)
    error_porcentual = (error/ abs(exact)) * 100

    # Mostrar resultados
    st.subheader("📈 Resultados")
    st.write(f"**Valor aproximado:** {approx:.6f}")
    st.write(f"**Valor exacto:** {exact:.6f}")
    st.write(f"**Error absoluto:** {error:.6f}")
    st.write(f"**Error porcentual:** {error_porcentual:.6f} %")


    # Gráfica
    fig, ax = plt.subplots(figsize=(8, 4))
    X_fine = np.linspace(a, b, 1000)
    ax.plot(X_fine, f(X_fine), label='f(x)', color='blue', linewidth=2)

    if method == "Trapecio":
        for i in range(n):
            xi0, xi1 = X[i], X[i+1]
            yi0, yi1 = f(xi0), f(xi1)
            
            # Coordenadas del trapecio
            x_trapecio = [xi0, xi0, xi1, xi1, xi0]
            y_trapecio = [0,  yi0, yi1, 0, 0]
            
            # Dibujar contorno del trapecio (línea verde)
            ax.plot([xi0, xi1], [yi0, yi1], color='green', linewidth=2)  # parte superior del trapecio

            # Rellenar el trapecio
            ax.fill(x_trapecio, y_trapecio, color='orange', alpha=0.5, edgecolor='green')


    elif method == "Simpson":
        if n % 2 != 0:
            st.warning("⚠️ Simpson requiere un número par de subintervalos.")
        else:
            for i in range(0, n, 2):
                xi = X[i:i+3]
                yi = f(xi)
                coef = np.polyfit(xi, yi, 2)
                poly = np.poly1d(coef)
                x_parabola = np.linspace(xi[0], xi[-1], 100)
                y_parabola = poly(x_parabola)
                # Dibuja la parábola con línea roja discontinua
                ax.plot(x_parabola, y_parabola, 'r--', linewidth=2)
                # Rellena el área bajo la parábola
                ax.fill_between(x_parabola, y_parabola, alpha=0.3, color='orange')


    else:
        # Para los métodos de Riemann y Punto Medio, sigue igual (barras)
        for i in range(n):
            xi, xf = X[i], X[i+1]
            if method == "Riemann Izquierda":
                ax.bar(xi, f(xi), width=dx, alpha=0.3, align='edge', edgecolor='k')
            elif method == "Riemann Derecha":
                ax.bar(xf, f(xf), width=-dx, alpha=0.3, align='edge', edgecolor='k')
            elif method == "Punto Medio":
                xm = (xi + xf) / 2
                ax.bar(xm, f(xm), width=dx, alpha=0.3, align='center', edgecolor='k')

    ax.set_title(f"Método: {method}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)


except Exception as e:
    st.error(f"Ocurrió un error: {e}")
