# STAR-PINN: a Physics-Informed Neural Network (PINN) for solving the 2D time-domain magnetic diffusion (Maxwell) equation in nonlinear materials.
Official implementation of STAR-PINN: a Physics-Informed Neural Network (PINN) for solving the 2D time-domain magnetic diffusion (Maxwell) equation in nonlinear materials.

<img width="1701" height="1080" alt="STARPINN_V2" src="https://github.com/user-attachments/assets/76ff0ac5-5085-40f1-bb37-2d42a213c998" />




==================================================
CITATION
==================================================

If you use this code, please cite:

@article{dodge2025starpinn,
  title={A STacked Adaptive Residual PINN (STAR-PINN) Approach to 2D Time-Domain Magnetic Diffusion in Nonlinear Materials},
  author={Dodge, Shayan and Barmada, Sami and Formisano, Alessandro},
  journal={IEEE Access},
  year={2025}
}


==================================================
OVERVIEW
==================================================

This repository implements Physics-Informed Neural Networks (PINNs) and the proposed STAR-PINN architecture for solving time-domain magnetic diffusion problems in linear and nonlinear materials.

The method embeds the governing PDEs directly into the loss function, eliminating the need for labeled datasets. The proposed STAR-PINN improves standard PINNs by stacking lightweight residual blocks that iteratively refine the solution.

Key advantages:
- Improved convergence stability
- Higher accuracy in nonlinear problems
- Lower computational complexity compared to hybrid architectures (e.g., CNN/RNN-based PINNs)


==================================================
KEY FEATURES
==================================================

- Physics-informed learning of PDEs (no dataset required)
- Time-domain PINN (TD-PINN) for linear problems
- STAR-PINN for nonlinear magnetic diffusion
- Adaptive residual stacking mechanism
- Automatic differentiation for computing PDE residuals
- Applications to 1D and 2D electromagnetic problems


==================================================
REPOSITORY STRUCTURE
==================================================

data/                 -> Sampling points and domain definitions
models/               -> PINN and STAR-PINN architectures
utils/                -> Loss functions and helper utilities
experiments/          -> Scripts for reproducing results
  case1_1D_linear.py
  case2_2D_linear.py
  case3_2D_nonlinear.py
results/              -> Output plots and predictions
notebooks/            -> Optional demos
requirements.txt
README.md


==================================================
INSTALLATION
==================================================

git clone https://github.com/yourusername/star-pinn.git
cd star-pinn
pip install -r requirements.txt


==================================================
RUNNING EXPERIMENTS
==================================================

Case I: 1D Linear Diffusion
python experiments/case1_1D_linear.py

Case II: 2D Linear Diffusion
python experiments/case2_2D_linear.py

Case III: 2D Nonlinear Diffusion (STAR-PINN)
python experiments/case3_2D_nonlinear.py


==================================================
METHOD
==================================================

TD-PINN:
- Inputs: spatial coordinates (x, y) and time (t)
- Learns PDE solution via loss minimization
- Uses automatic differentiation for derivatives

STAR-PINN:
- Stacked residual PINN blocks
- Each block learns correction (residual)
- Adaptive weights improve convergence

Residual formulation:
Az = A2 + α * A3


==================================================
RESULTS
==================================================

- Strong agreement with analytical solutions in 1D
- Close match with FEM simulations in 2D
- STAR-PINN reduces error below 1% in nonlinear cases

See "results/" folder for plots and outputs.


==================================================
REPRODUCIBILITY
==================================================

Framework: TensorFlow
Optimizer: L-BFGS
Activation: tanh

Training points:
- ~12,000 (1D case)
- ~31,500 (2D cases)


==================================================
PAPER TO CODE MAPPING
==================================================

1D Linear Case              -> case1_1D_linear.py
2D Linear Case              -> case2_2D_linear.py
2D Nonlinear (STAR-PINN)    -> case3_2D_nonlinear.py


==================================================
MINIMAL EXAMPLE
==================================================

from models.star_pinn import StarPINN

model = StarPINN()
model.train()


==================================================
ACKNOWLEDGMENTS
==================================================

Supported by:
- PRIN 2022 (Italian Ministry for Education, University and Research)
- European Union – NextGenerationEU


==================================================
CONTACT
==================================================

For questions or collaborations, please open an issue on GitHub.
