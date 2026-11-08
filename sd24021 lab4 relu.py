import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
 
st.set_page_config(page_title="ReLU Activation Function")
 
# the activation function and its slope (derivative)
def activation(x):
    return np.maximum(0, x)
 
def slope(x):
    return np.where(x > 0, 1.0, 0.0)
 
 
st.title("ReLU Activation Function")
st.write("ReLU (Rectified Linear Unit) gives 0 for negative inputs and gives the input itself for positive inputs.")
st.latex(r"f(x) = \max(0, x)")
 
# choose the range of x to plot
x_min, x_max = st.slider("Range of x", -20.0, 20.0, (-10.0, 10.0), 0.5)
 
# choose one value of x to calculate
x_value = st.number_input("Input value (x)", value=-2.0, step=0.5)
 
x = np.linspace(x_min, x_max, 400)
y = activation(x)
 
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, label="ReLU")
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.scatter([x_value], [activation(x_value)], color="red", zorder=5, label="chosen x")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("ReLU Activation Function")
ax.grid(alpha=0.3)
ax.legend()
st.pyplot(fig)
plt.close(fig)
 
col1, col2 = st.columns(2)
col1.metric("ReLU output f(x)", f"{activation(x_value):.4f}")
col2.metric("Slope f'(x)", f"{slope(x_value):.4f}")
 
st.info("Negative inputs become 0, positive inputs pass through unchanged. The slope is 0 for negative x and 1 for positive x.")