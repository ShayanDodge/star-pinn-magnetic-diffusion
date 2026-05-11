import tensorflow as tf
import numpy as np
from layer import GradientLayer

class PINN:
    """
    A Physics Informed Neural Network (PINN) model for the diffusion equation.

    Attributes:
        network: keras network model with input (t, z, y) and output u(t, z, y).
        grads: GradientLayer object for computing the gradients.
    """

    def __init__(self, network, D, b):
        """
        Initializes the PINN model.

        Args:
            network: keras network model with input (t, z, y) and output u(t, z, y).
            c: Diffusion coefficient (default 0.5).
        """
        self.network = network
        self.c = D # sigma * mu_0 * mu_r
        self.b = b
        self.grads = GradientLayer(self.network)  # Initialize the gradient computation layer

    def build(self):
        """
        Build a PINN model for the diffusion equation.

        The model computes outputs for:
            - Equation: u(t, z, y) satisfies the diffusion equation.
            - Initial Condition: u(t = 0, z, y) and du/dt(t=0, z, y) for the initial condition.
            - Boundary Conditions: u(t, z=bounds, y=bounds) and gradients at the boundaries.
        
        Returns:
            PINN model for the diffusion equation with inputs:
                1. (t, z, y) for the PDE.
                2. (t = 0, z, y) for initial condition.
                3. Boundary condition inputs for the four boundaries.
                
            And outputs:
                1. u(t, z, y) from the PDE.
                2. u(t = 0, z, y) for the initial condition.
                3. du/dt at t=0, z, y for initial derivative.
                4. u at boundary conditions and corresponding gradients.
        """
        
        # Input layers
        tzy_pde = tf.keras.layers.Input(shape=(3,))  # Input for PDE: t, z, y
        tzy_ini = tf.keras.layers.Input(shape=(3,))  # Input for Initial Condition: t=0, z, y
        tzy_y_t = tf.keras.layers.Input(shape=(3,))  # Boundary: y = y_f (top boundary)
        tzy_y_b = tf.keras.layers.Input(shape=(3,))  # Boundary: y = y_ini (bottom boundary)
        tzy_z_r = tf.keras.layers.Input(shape=(3,))  # Boundary: z = z_f (right boundary)
        tzy_z_l = tf.keras.layers.Input(shape=(3,))  # Boundary: z = z_ini (left boundary)
        
        tzy_y_t_int = tf.keras.layers.Input(shape=(3,))  # Boundary: y = y_f (top boundary)
        tzy_y_b_int = tf.keras.layers.Input(shape=(3,))  # Boundary: y = y_ini (bottom boundary)
        tzy_z_r_int = tf.keras.layers.Input(shape=(3,))  # Boundary: z = z_f (right boundary)
        tzy_z_l_int = tf.keras.layers.Input(shape=(3,))  # Boundary: z = z_ini (left boundary)

        # Compute gradients for the PDE equation
        _, du_dt, _, _, _, d2u_dz2, d2u_dy2 = self.grads(tzy_pde, self.b)

        # Equation output: Diffusion equation, set to zero
        u_pde = self.c * du_dt - (d2u_dz2 + d2u_dy2) 

        # Compute gradients for the initial condition (t=0)
        u_ini, _, _, _, _, _, _ = self.grads(tzy_ini, self.b)

        # Boundary condition computations
        u_t, _, du_dz_t, _, _, _, _ = self.grads(tzy_y_t, self.b)  # Top boundary (y = y_f)
        u_b, _, du_dz_b, _, _, _, _ = self.grads(tzy_y_b, self.b)  # Bottom boundary (y = y_ini)

        u_r, _, _, du_dy_r, _, _, _ = self.grads(tzy_z_r, self.b)  # Right boundary (z = z_f)
        u_l, _, _, du_dy_l, _, _, _ = self.grads(tzy_z_l, self.b)  # Left boundary (z = z_ini)

                # Boundary condition computations
        _, _, _, du_dy_t_int, _, _, _ = self.grads(tzy_y_t_int, self.b)  # Top boundary (y = y_f)
        _, _, _, du_dy_b_int, _, _, _ = self.grads(tzy_y_b_int, self.b)  # Bottom boundary (y = y_ini)

        _, _, du_dz_r_int, _, _, _, _ = self.grads(tzy_z_r_int, self.b)  # Right boundary (z = z_f)
        _, _, du_dz_l_int, _, _, _, _ = self.grads(tzy_z_l_int, self.b)  # Left boundary (z = z_ini)

        # Build and return the PINN model for the diffusion equation
        return tf.keras.models.Model(
            inputs= [tzy_pde, tzy_ini, tzy_y_b, tzy_y_t, tzy_z_l, tzy_z_r, tzy_y_b_int, tzy_y_t_int, tzy_z_l_int, tzy_z_r_int],
            outputs=[u_pde  , u_ini  ,du_dz_b, du_dz_t, du_dy_l, du_dy_r, du_dy_b_int, du_dy_t_int, du_dz_l_int, du_dz_r_int
                     , u_b, u_t, u_l, u_r])  # Output includes all PDE and boundary condition results
