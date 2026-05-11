import scipy.optimize
import numpy as np
import tensorflow as tf
from tqdm import tqdm  # Import tqdm for progress bar
from tensorflow.keras.optimizers import Adam
from skopt import gp_minimize



class L_BFGS_B:
    """
    Optimize the keras network model using L-BFGS-B algorithm.
    Attributes:
        model: optimization target model.
        samples: training samples.
        factr: convergence condition. Typical values for factr are:
               1e12 for low accuracy, 1e7 for moderate accuracy, 10 for extremely high accuracy.
        m: maximum number of variable metric corrections used to define the limited memory matrix.
        maxls: maximum number of line search steps (per iteration).
        maxiter: maximum number of iterations.
        metrics: logging metrics.
        pbar: progress bar (tqdm instance).
    """
 
    def __init__(self, model, x_train, y_train, dr ,mu ,I , factr = 0.1, m = 50, maxls = 50, maxiter = 35000, epsilon = 1e-10, pgtol=1e-9, baseNmunber = 15, timeNmunber = 15, b = 1e-8): # 35000
        """
            Args:
                model: 
                    The model to be optimized. This is the target optimization model.
                x_train, y_train: 
                    Training samples. `x_train` contains the input features, and `y_train` contains the corresponding labels/targets.
                factr: 
                    Convergence condition (stopping criterion). This is a factor that scales the tolerance for the stopping condition:
                    - `factr=1e12` for low accuracy (faster convergence).
                    - `factr=1e7` for moderate accuracy (good balance between speed and accuracy).
                    - `factr=10` for extremely high accuracy (slower convergence but more precise results).
                    **Default**: `10`.
                m: 
                    Maximum number of variable metric corrections used to define the limited-memory BFGS (L-BFGS) matrix. 
                    Increasing `m` allows for a more accurate approximation of the inverse Hessian but requires more memory.
                    **Typical values**: `50` or `100`. 
                    **Default**: `50`.
                maxls: 
                    Maximum number of line search steps allowed per iteration. The line search is used to find an appropriate step size during optimization.
                    **Typical values**: `20-50`.
                    **Default**: `50`.
                maxiter: 
                    Maximum number of iterations to run the optimization. If the optimization reaches this number of iterations without converging, it stops.
                    For large problems, this may need to be increased.
                    **Default**: `35000`.
                epsilon: 
                    The step size used in numerical gradient approximation when the gradient is not provided analytically. 
                    Smaller values lead to more accurate gradients but slower computation.
                    **Typical value**: `1e-6`. This is a reasonable choice for most problems.
                pgtol: 
                    Gradient tolerance. Optimization will stop when the gradient norm is smaller than this value.
                    **Typical value**: `1e-4`. 
                    Smaller values lead to more precise convergence, but might take longer to compute.
                    **Default**: `1e-4`.
            """

        # Set attributes
        self.model = model
        self.x_train = [tf.constant(x, dtype=tf.float32) for x in x_train]
        self.y_train = [tf.constant(y, dtype=tf.float32) for y in y_train]
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.epsilon = epsilon
        self.metrics = ['loss']
        self.mu = mu
        self.I = I
        self.pgtol = pgtol
        # Initialize the progress bar
        self.pbar = tqdm(total=maxiter, desc="L-BFGS-B Optimization Progress", unit="iteration")
        self.dr = dr
        self.baseNmunber = baseNmunber
        self.timeNmunber = timeNmunber
        self.b = b

    def set_weights(self, flat_weights):
        """
        Set weights to the model.
        Args:
            flat_weights: flatten weights.
        """
        shapes = [w.shape for w in self.model.get_weights()]
        split_ids = np.cumsum([np.prod(shape) for shape in [0] + shapes])
        weights = [flat_weights[from_id:to_id].reshape(shape)
                   for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes)]
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):
        """
        Evaluate loss and gradients for weights as tf.Tensor.
        Args:
            x, y: input data and labels.
        Returns:
            Loss and gradients for weights as tf.Tensor.
        """
        with tf.GradientTape() as g:
            u = self.model(x)

            # du_dy_b_grouped = tf.reshape(u[6], (self.timeNmunber, self.baseNmunber, -1))
            # du_dy_b_summed = tf.reduce_sum(du_dy_b_grouped, axis=1) * self.dr  

            # du_dy_t_grouped = tf.reshape(u[7], (self.timeNmunber, self.baseNmunber, -1))
            # du_dy_t_summed = tf.reduce_sum(du_dy_t_grouped, axis=1) * self.dr  
            
            # du_dz_l_grouped = tf.reshape(u[8], (self.timeNmunber, self.baseNmunber, -1))
            # du_dz_l_summed = tf.reduce_sum(du_dz_l_grouped, axis=1) * self.dr  

            # du_dz_r_grouped = tf.reshape(u[9], (self.timeNmunber, self.baseNmunber, -1))
            # du_dz_r_summed = tf.reduce_sum(du_dz_r_grouped, axis=1) * self.dr 

            du_dy_b_grouped = tf.reshape((1/(1-(self.b*abs(u[6]))))*u[6], (self.timeNmunber, self.baseNmunber, -1))
            du_dy_b_summed = tf.reduce_sum(du_dy_b_grouped, axis=1) * self.dr  

            du_dy_t_grouped = tf.reshape((1/(1-(self.b*abs(u[7]))))*u[7], (self.timeNmunber, self.baseNmunber, -1))
            du_dy_t_summed = tf.reduce_sum(du_dy_t_grouped, axis=1) * self.dr  
            
            du_dz_l_grouped = tf.reshape((1/(1-(self.b*abs(u[8]))))*u[8], (self.timeNmunber, self.baseNmunber, -1))
            du_dz_l_summed = tf.reduce_sum(du_dz_l_grouped, axis=1) * self.dr  

            du_dz_r_grouped = tf.reshape((1/(1-(self.b*abs(u[9]))))*u[9], (self.timeNmunber, self.baseNmunber, -1))
            du_dz_r_summed = tf.reduce_sum(du_dz_r_grouped, axis=1) * self.dr 
            
            ones_tensor = tf.ones_like(du_dz_r_summed)
            
            loss_int = tf.reduce_mean(tf.keras.losses.mse(-du_dy_t_summed  
                                                            +du_dy_b_summed
                                                            +du_dz_l_summed
                                                            -du_dz_r_summed,
                                                            self.I * self.mu * (ones_tensor) ))
                           
            lossPde = 1 * tf.reduce_mean(tf.keras.losses.mse(u[0], y[0]))
            lossBoun =  (1 * tf.reduce_mean(tf.keras.losses.mse(u[2:6], y[2:6])) + 1 * tf.reduce_mean(tf.keras.losses.mse(u[1], y[1])) +
                         0 * tf.reduce_mean(tf.keras.losses.mse(u[10:14], y[10:14])) )


            loss = tf.reduce_mean(1 * lossPde + 1 * lossBoun + 1 * loss_int)

        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads, lossPde, lossBoun, loss_int

    def evaluate(self, weights):
        """
        Evaluate loss and gradients for weights as ndarray.
        Args:
            weights: flatten weights.
        Returns:
            Loss and gradients for weights as ndarray.
        """
        self.set_weights(weights)
        loss, grads, lossPde, lossBoun, loss_int = self.tf_evaluate(self.x_train, self.y_train)
        loss = loss.numpy().astype('float64')
        lossPde = lossPde.numpy().astype('float64')
        lossBoun = lossBoun.numpy().astype('float64')
        loss_int = loss_int.numpy().astype('float64')
        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype('float64')
        return loss, grads, lossPde, lossBoun, loss_int

    def callback(self, weights):
        """
        Callback that updates the progress bar and logs the loss.
        Args:
            weights: flatten weights.
        """
        self.trainable_vars = self.model.trainable_variables

        loss, grads, lossPde, lossBoun, loss_int = self.evaluate(weights)
        self.pbar.set_postfix({"loss": loss,"lossPde": lossPde,"lossBoun": lossBoun,"loss_int": loss_int})
        # Log trainable variables

        # for var in self.trainable_vars:
        #     if "alpha" in var.name:  # Check if the variable name contains "alpha"
        #         tf.print(f"{var.name}: {var.numpy()}")

        self.pbar.update(1)

    def fit(self):
        """
        Train the model using L-BFGS-B algorithm.
        """
        initial_weights = np.concatenate([w.flatten() for w in self.model.get_weights()])
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        # scipy.optimize.fmin_l_bfgs_b(
        #     func=self.evaluate,
        #     x0=initial_weights,
        #     factr=self.factr,
        #     m=self.m,
        #     maxls=self.maxls,
        #     maxiter=self.maxiter,
        #     callback=self.callback,
        #     pgtol=0  # Prevent early stopping by setting tolerance to zero
        # )


        # Function for optimization
        scipy.optimize.fmin_l_bfgs_b(
            func=self.evaluate,           # Objective function to minimize
            x0=initial_weights,           # Initial guess for the parameters (weights)
            
            # **pgtol**: Gradient Tolerance - determines when optimization stops based on gradient size.
            # If the gradient norm becomes smaller than this value, the optimization will stop.
            # Small values (e.g., 1e-6) will result in more precise convergence, but it will take longer.
            # Larger values (e.g., 1e-2) will stop optimization earlier, which may not guarantee a precise solution.
            pgtol = self.pgtol,                   # Gradient tolerance (stopping criterion based on gradient magnitude)

            # **epsilon**: Step size for numerical gradient approximation in case the gradient is not provided analytically.
            # Smaller values (e.g., 1e-6) give more precise gradients, but too small a value can cause numerical instability.
            # Larger values (e.g., 1e-4) can speed up the computation but might affect gradient accuracy.
            epsilon=self.epsilon,                 # Numerical gradient step size
            
            # **factr**: A parameter controlling the accuracy of the solution. Smaller values mean higher precision.
            factr=self.factr,             # Factor for controlling stopping criterion (default is very high for precise optimization)

            # **m**: The number of corrections to use in the approximation of the inverse Hessian matrix.
            # A larger value uses more memory and computations, but might result in a more accurate solution.
            m=self.m,                     # Number of corrections (typically set to 50 or larger)

            # **maxls**: Maximum number of line search steps allowed.
            # It controls how many attempts the algorithm makes to find a suitable step length at each iteration.
            maxls=self.maxls,             # Max number of line search steps

            # **maxiter**: Maximum number of iterations to run the optimization.
            # Increasing this allows the optimization to run longer, but you should balance it with convergence criteria.
            maxiter=self.maxiter,         # Maximum iterations allowed (set to higher value for large problems)

            # **callback**: Function called at each iteration to monitor the progress of the optimization.
            callback=self.callback       # Optional callback function to monitor optimization progress
        )




        self.pbar.close()