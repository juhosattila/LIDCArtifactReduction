from tensorflow.keras.losses import Loss, MeanSquaredError
from LIDCArtifactReduction.tf_image import TV_square_diff_norm

class MSE_TV_square_diff_loss(Loss):
    def __init__(self, tv_weight, name):
        super().__init__(name=name)
        self._mse_loss = MeanSquaredError()
        self._tv_weight = tv_weight

    def call(self, imgs_true, imgs_pred):
        result = self._mse_loss(imgs_true, imgs_pred)
        result = result + self._tv_weight * TV_square_diff_norm(imgs_true, imgs_pred)
        return result
