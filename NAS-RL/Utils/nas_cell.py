import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class NASCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units):
        self._num_units = num_units

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "nas_cell", reuse=tf.AUTO_REUSE):
            input_size = inputs.get_shape().as_list()[-1]

            W = tf.get_variable(
                "W",
                shape=[input_size + self._num_units, self._num_units],
                initializer=tf.glorot_uniform_initializer()
            )
            b = tf.get_variable(
                "b",
                shape=[self._num_units],
                initializer=tf.zeros_initializer()
            )

            concat = tf.concat([inputs, state], axis=1)  # [B, 24]
            hidden = tf.tanh(tf.matmul(concat, W) + b)

        return hidden, hidden
