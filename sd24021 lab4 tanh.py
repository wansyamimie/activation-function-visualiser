import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
 
st.set_page_config(page_title="Tanh Activation Function")
 
# the activation function and its slope (derivative)
def activation(x):
    return np.tanh(x)
 
def slope(x):
    return 1 - np.tanh(x) ** 2
 
 
st.title("Hyperbolic Tangent (Tanh) Activation Function")
st.write("Tanh changes any input into an output between -1 and 1.")
st.latex(r"\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}")
 
# choose the range of x to plot
x_min, x_max = st.slider("Range of x", -20.0, 20.0, (-10.0, 10.0), 0.5)
 
# choose one value of x to calculate
x_value = st.number_input("Input value (x)", value=0.0, step=0.5)
 
x = np.linspace(x_min, x_max, 400)
y = activation(x)
 
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, label="Tanh")
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.scatter([x_value], [activation(x_value)], color="red", zorder=5, label="chosen x")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("Hyperbolic Tangent (Tanh) Activation Function")
ax.grid(alpha=0.3)
ax.legend()
st.pyplot(fig)
plt.close(fig)
 
col1, col2 = st.columns(2)
col1.metric("Tanh output f(x)", f"{activation(x_value):.4f}")
col2.metric("Slope f'(x)", f"{slope(x_value):.4f}")
 
st.info("Large negative inputs give outputs close to -1 and large positive inputs give outputs close to 1. At x = 0 the output is 0.")