# A STacked Adaptive Residual PINN (STAR-PINN) Approach to 2D Time-Domain Magnetic Diffusion in Nonlinear Materials

[![Project Website](https://img.shields.io/badge/website-STAR_PINN-blue)](https://shayandodge.github.io/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 🔗 Official Implementation

This repository is the **official implementation** of the paper:

> **“A STacked Adaptive Residual PINN (STAR-PINN) Approach to 2D Time-Domain Magnetic Diffusion in Nonlinear Materials”**  
> Published in IEEE Access, 2025  
> [https://ieeexplore.ieee.org/document/10755077 ](https://ieeexplore.ieee.org/abstract/document/11122441/) 

It provides a Physics-Informed Neural Network (PINN) designed to solve the **2D time-domain magnetic diffusion (Maxwell) equation** in linear and nonlinear materials.

<p align="center">
  <img src="https://github.com/user-attachments/assets/76ff0ac5-5085-40f1-bb37-2d42a213c998" alt="STAR-PINN" width="800"/>
</p>

---

## 📖 Overview

This repository implements **Physics-Informed Neural Networks (PINNs)** and the proposed **STAR-PINN architecture** for solving time-domain magnetic diffusion problems.

Unlike traditional data-driven models, PINNs embed the governing PDEs directly into the loss function—removing the need for labeled datasets.

**STAR-PINN extends standard PINNs** by stacking lightweight residual blocks that iteratively refine the solution.

### ✅ Key Advantages
- Improved convergence stability  
- Higher accuracy for nonlinear problems  
- Lower computational complexity vs hybrid architectures (e.g., CNN/RNN-based PINNs)

---

## 🚀 Key Features

- Physics-informed learning of PDEs (no dataset required)  
- Time-domain PINN (TD-PINN) for linear problems  
- STAR-PINN for nonlinear magnetic diffusion  
- Adaptive residual stacking mechanism  
- Automatic differentiation for PDE residuals  
- Applications to 1D and 2D electromagnetic problems  

---

## ▶️ Running Experiments

### Case III — 2D Nonlinear Diffusion (STAR-PINN)  
python experiments/case3_2D_nonlinear.py  

---

## 🧠 Method

### STAR-PINN
- Inputs: spatial coordinates *(x, y)* and time *(t)*  
- Learns PDE solution via loss minimization  
- Uses automatic differentiation for derivatives  
- Stacked residual PINN blocks  
- Each block learns a correction (residual)  
- Adaptive weighting improves convergence  

**Residual formulation:**  
A_z = A_2 + α · A_3  



---

## 🧪 Minimal Example

from models.star_pinn import StarPINN

model = StarPINN()  
model.train()  

---

## 📬 Contact

For questions, issues, or collaborations, please open an issue on GitHub.
