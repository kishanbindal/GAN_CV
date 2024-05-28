import tensorflow as tf

class ReflectionPad2D(tf.keras.layers.Layer):

    def __init__(self, padding =(1, 1)):
        super().__init__()
        if type(padding) is int:
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)

    def call(self, input_tensor):
        p_vertical, p_horizontal = self.padding
        # accounting for dims : b_size, x, y, channels
        paddings = tf.constant([[0, 0],[p_vertical, p_vertical], [p_horizontal, p_horizontal], [0, 0]], dtype=tf.int32)
        return tf.pad(input_tensor, paddings, mode="REFLECT")
