import tensorflow as tf

from LIDCArtifactReduction.math_helpers.math_mixin import MathMixin


class TensorflowMathMixin(MathMixin):
    def shape(self, img):
        return tf.shape(img)

    def log(self, x):
        return tf.math.log(x)

    def exp(self, x):
        return tf.math.exp(x)

    def random_normal(self, mean, stddev, size):
        return tf.random.normal(size, mean, stddev, dtype=tf.float32)

    def random_poisson(self, mean):
        mean_tf = tf.convert_to_tensor(mean, dtype=tf.float32)
        return tf.random.poisson([], mean_tf, dtype=tf.float32)

    def as_array(self, x):
        return tf.convert_to_tensor(x, dtype=tf.float32)
        # The row below generally would accept x to be Tensor with dtype=tf.float64 and cast it.
        # tf.convert_to_tensor with dtype=tf.float32 does not accept such input but it is better to have this option,
        # because tf.float64 should not be present in the system.
        # return tf.cast(tf.convert_to_tensor(x), dtype=tf.float32)

    def where(self, condition, x, y):
        return tf.where(condition, x, y)
