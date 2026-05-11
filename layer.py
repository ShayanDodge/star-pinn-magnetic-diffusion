import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives for the diffusion equation.
    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """
        # Initialize the parent class and store the model
        self.model = model
        super().__init__(**kwargs)

    def call(self, tzy, b = 1e-8):
        """
        Compute 1st and 2nd derivatives for the diffusion equation.
        Args:
            tzy: Input variables (e.g., time t, spatial variables z and y).
        Returns:
            u: Network output.
            du_dt: 1st derivative with respect to t.
            du_dz: 1st derivative with respect to z.
            du_dy: 1st derivative with respect to y.
            d2u_dt2: 2nd derivative with respect to t.
            d2u_dz2: 2nd derivative with respect to z.
            d2u_dy2: 2nd derivative with respect to y.
        """

        # Use an outer gradient tape to compute second-order derivatives
        with tf.GradientTape() as g:
            g.watch(tzy)  # Ensure tzy is tracked by the gradient tape

            # Use an inner gradient tape to compute first-order derivatives
            with tf.GradientTape() as gg:
                gg.watch(tzy)  # Ensure tzy is tracked by the inner gradient tape
                u = self.model(tzy)  # Compute the model output

            # Compute first-order derivatives (Jacobian) with respect to tzy
            du_dtzy = gg.batch_jacobian(u, tzy)
            du_dt = du_dtzy[..., 0]  # Extract derivative with respect to t
            du_dz = du_dtzy[..., 1]  # Extract derivative with respect to z
            du_dy = du_dtzy[..., 2]  # Extract derivative with respect to y

            # Optimized nu calculation
            B_squared = du_dy**2 + du_dz**2  # Avoid computing the square root
            nu = 1 / (1 - (b * tf.sqrt(B_squared)))
            nu_du_dtzy = tf.stack([du_dt, nu * du_dz, nu * du_dy], axis=-1)

        # Compute second-order derivatives using the outer gradient tape
        d2u_dtzy2 = g.batch_jacobian(nu_du_dtzy, tzy)
        d2u_dt2 = d2u_dtzy2[..., 0, 0]  # Second derivative with respect to t
        d2u_dz2 = d2u_dtzy2[..., 1, 1]  # Second derivative with respect to z
        d2u_dy2 = d2u_dtzy2[..., 2, 2]  # Second derivative with respect to y

        # Return the model output and computed derivatives
        return u, du_dt, du_dz, du_dy, d2u_dt2, d2u_dz2, d2u_dy2
    