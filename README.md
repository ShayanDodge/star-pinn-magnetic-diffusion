# A STacked Adaptive Residual PINN (STAR-PINN) Approach to 2D Nonlinear Magnetic Diffusion

[![Project Website](https://img.shields.io/badge/Website-STAR--PINN-blue)](https://shayandodge.github.io/)
[![Paper](https://img.shields.io/badge/Paper-IEEE_Access-red)](https://ieeexplore.ieee.org/document/11122441/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

# 📌 Official Implementation

This repository contains the official TensorFlow implementation of:

> **A STacked Adaptive Residual PINN (STAR-PINN) Approach to 2D Time-Domain Magnetic Diffusion in Nonlinear Materials**  
> *IEEE Access, 2025*

📄 Paper: https://ieeexplore.ieee.org/document/11122441/

The project introduces **STAR-PINN**, a stacked residual Physics-Informed Neural Network architecture for solving the **2D nonlinear time-domain magnetic diffusion equation** in conductive magnetic materials.

<p align="center">
  <img src="https://github.com/user-attachments/assets/76ff0ac5-5085-40f1-bb37-2d42a213c998" 
       alt="STAR-PINN Architecture" width="850"/>
</p>

---

# 📖 Overview

This repository focuses on solving **nonlinear magnetic diffusion problems** using Physics-Informed Neural Networks (PINNs).

Unlike conventional numerical solvers, the proposed approach embeds the governing Maxwell equations directly into the loss function using automatic differentiation.

The proposed **STAR-PINN** architecture improves conventional PINNs by introducing:

- stacked residual PINN blocks
- adaptive residual learning
- improved convergence stability
- enhanced nonlinear approximation capability

The method is designed specifically for **nonlinear magnetic materials**, where permeability depends on the magnetic flux density.

---

# ⭐ STAR-PINN Architecture

STAR-PINN consists of multiple lightweight PINN blocks connected through adaptive residual corrections.

Instead of relying on a single deep neural network, each block progressively refines the solution learned by the previous block.

## Residual Formulation

\[
A_z^{(2)} = A_z^{(1)} + \alpha_1 A_z^{(0)}
\]

\[
A_z = A_z^{(2)} + \alpha_2 A_z^{(3)}
\]

where:

- \(A_z^{(i)}\) denotes the output of each PINN block
- \(\alpha_i\) are trainable adaptive coefficients

This residual formulation improves:

- training stability
- gradient propagation
- nonlinear feature learning
- convergence accuracy

---

# 🧲 Nonlinear Magnetic Diffusion Equation

The nonlinear PDE solved in this work is:

\[
\frac{\partial}{\partial x}
\left(
\nu_B \frac{\partial A_z}{\partial x}
\right)
+
\frac{\partial}{\partial y}
\left(
\nu_B \frac{\partial A_z}{\partial y}
\right)
=
\sigma \frac{\partial A_z}{\partial t}
\]

where:

- \(A_z\) = magnetic vector potential
- \(\nu_B\) = nonlinear magnetic reluctivity
- \(\sigma\) = electrical conductivity

The nonlinear reluctivity is defined as:

\[
\nu_B =
\frac{1}
{\mu_r \mu_0 (1 - b|B|)}
\]

---

# 🚀 Key Features

- STAR-PINN residual architecture
- Nonlinear magnetic material modeling
- Time-domain Maxwell diffusion solver
- Automatic differentiation with TensorFlow
- Physics-informed learning without datasets
- Adaptive residual refinement
- Mesh-free PDE solution
- L-BFGS optimization

---

# 📂 Repository Structure

```text
.
├── 2D_Diffusion_STARPINN.ipynb   # Main nonlinear STAR-PINN notebook
├── pinn.py                       # STAR-PINN implementation
├── network.py                    # Neural network architecture
├── optimizer.py                  # L-BFGS optimizer
├── layer.py                      # Custom layers and utilities
├── README.md
└── LICENSE
```

---

# ⚙️ Installation

## Requirements

- Python 3.10+
- TensorFlow
- NumPy
- Matplotlib
- SciPy

## Install dependencies

```bash
pip install tensorflow numpy matplotlib scipy
```

---

# ▶️ Running the Code

Launch the notebook:

```bash
jupyter notebook 2D_Diffusion_STARPINN.ipynb
```

---

# 🧪 Example

```python
from pinn import PINN

model = PINN()

model.train()
model.predict()
```

---

# 📊 Results

STAR-PINN achieves:

- significantly lower error than standard PINNs
- improved nonlinear magnetic field prediction
- stable convergence in transient simulations
- strong agreement with FEM solutions

The architecture is particularly effective for magnetic saturation problems and nonlinear diffusion dynamics.

---

# 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{dodge2025starpinn,
  title={A STacked Adaptive Residual PINN (STAR-PINN) Approach to 2D Time-Domain Magnetic Diffusion in Nonlinear Materials},
  author={Dodge, Shayan and Barmada, Sami and Formisano, Alessandro},
  journal={IEEE Access},
  year={2025}
}
```

---

# 📬 Contact

**Shayan Dodge**  
PhD Researcher — University of Pisa  

🌐 Website: https://shayandodge.github.io/

For questions, collaborations, or issues, please open a GitHub issue.
