import tensorflow as tf
from tensorflow.keras import models, layers

class Network:
    @classmethod
    def build(cls, num_inputs=3, main_layers=[48, 48, 48], res_layers=[48, 48, 48], 
              main_activation='tanh', res_activation='tanh', num_outputs=1,
              kernel_initializer='he_normal', num_stacks=2):

        # Input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,))

        # Helper function for a dense block
        def dense_block(x, units, activation, kernel_initializer):
            return tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer)(x)

        # Main PINN dense layers
        x = inputs
        for units in main_layers:
            x = dense_block(x, units, main_activation, kernel_initializer)

        # Initial output
        u_prev = tf.keras.layers.Dense(num_outputs, kernel_initializer=kernel_initializer)(x)

        alphas = []

        # Stacked residual blocks
        for i in range(num_stacks):
            # Trainable scaling factor for residuals
            alpha = tf.keras.layers.Dense(1, activation=None, use_bias=False, name=f"alpha_{i}")(u_prev)
            alphas.append(alpha)

            # Concatenate inputs and previous prediction
            pinn_input = tf.keras.layers.Concatenate()([inputs, u_prev])
            x = pinn_input

            # Residual network with its own architecture
            for units in res_layers:
                x = dense_block(x, units, res_activation, kernel_initializer)

            # Output of residual block
            u = tf.keras.layers.Dense(num_outputs, kernel_initializer=kernel_initializer)(x)

            # Residual connection: u_prev = u_prev + alpha * u
            u_prev = tf.keras.layers.add([u_prev, alpha * u])

        # Final output
        return models.Model(inputs=inputs, outputs=u_prev)
