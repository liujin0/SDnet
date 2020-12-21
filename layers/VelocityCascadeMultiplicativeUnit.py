import tensorflow as tf
from layers.MultiplicativeUnit import MultiplicativeUnit


class VelocityCascadeMultiplicativeUnit():
    """Initialize the causal multiplicative unit.
    Args:
       layer_name: layer names for different causal multiplicative unit.
       filter_size: int tuple of the height and width of the filter.
       num_hidden: number of units in output tensor.
    """
    def __init__(self, layer_name, num_hidden, filter_size):
        self.layer_name = layer_name
        self.num_features = num_hidden
        self.filter_size = filter_size

    def __call__(self, h1, h2, stride=False, reuse=False):
        with tf.variable_scope(self.layer_name, reuse=reuse):
            hl = MultiplicativeUnit('multiplicative_unit_1', self.num_features, self.filter_size)(h1, reuse=reuse)
            if not stride:
                hl = MultiplicativeUnit('multiplicative_unit_1', self.num_features, self.filter_size)(hl, reuse=True)
            hr = MultiplicativeUnit('multiplicative_unit_2', self.num_features, self.filter_size)(h2, reuse=reuse)
            h = tf.add(hl, hr)
            h_sig = tf.layers.conv2d(
                h, self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='sig')
            h_tan = tf.layers.conv2d(
                h, self.num_features, self.filter_size, padding='same', activation=tf.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='tanh')
            v = h2-h1
            h0 = tf.multiply(h_sig, h_tan)
            h = h0 + v
            return h
