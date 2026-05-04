# STAR-PINN  
**Stacked Adaptive Residual Physics-Informed Neural Network for 2D Time-Domain Magnetic Diffusion**

Official implementation of **STAR-PINN**, a Physics-Informed Neural Network (PINN) designed to solve the **2D time-domain magnetic diffusion (Maxwell) equation** in linear and nonlinear materials.

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

### Case I — 1D Linear Diffusion  
python experiments/case1_1D_linear.py  

### Case II — 2D Linear Diffusion  
python experiments/case2_2D_linear.py  

### Case III — 2D Nonlinear Diffusion (STAR-PINN)  
python experiments/case3_2D_nonlinear.py  

---

## 🧠 Method

### TD-PINN
- Inputs: spatial coordinates *(x, y)* and time *(t)*  
- Learns PDE solution via loss minimization  
- Uses automatic differentiation for derivatives  

### STAR-PINN
- Stacked residual PINN blocks  
- Each block learns a correction (residual)  
- Adaptive weighting improves convergence  

**Residual formulation:**  
A_z = A_2 + α · A_3  

---

## 🔗 Paper-to-Code Mapping

| Problem Type              | Script |
|--------------------------|--------|
| 1D Linear Case           | `case1_1D_linear.py` |
| 2D Linear Case           | `case2_2D_linear.py` |
| 2D Nonlinear (STAR-PINN) | `case3_2D_nonlinear.py` |

---

## 🧪 Minimal Example

from models.star_pinn import StarPINN

model = StarPINN()  
model.train()  

---

## 📬 Contact

For questions, issues, or collaborations, please open an issue on GitHub.
