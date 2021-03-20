from tensorflow.keras.losses import Loss, MeanSquaredError

from LIDCArtifactReduction.image.tf_image import total_variation_squared_mean_norm, total_variation_mean_norm, \
    logarithmic_total_variation_objective_function


class MSE_TV_squared_diff_loss(Loss):
    def __init__(self, tv_weight, name):
        super().__init__(name=name)
        self._mse_loss = MeanSquaredError()
        self._tv_weight = tv_weight

    def call(self, imgs_true, imgs_pred):
        result = self._mse_loss(imgs_true, imgs_pred)
        result = result + self._tv_weight * total_variation_squared_mean_norm(imgs_true - imgs_pred)
        return result


class MSE_TV_diff_loss(Loss):
    def __init__(self, tv_weight, name):
        super().__init__(name=name)
        self._mse_loss = MeanSquaredError()
        self._tv_weight = tv_weight

    def call(self, imgs_true, imgs_pred):
        result = self._mse_loss(imgs_true, imgs_pred)
        result = result + self._tv_weight * total_variation_mean_norm(imgs_true - imgs_pred)
        return result


class CustomReconstructionLoss(Loss):
    def __init__(self,
                 rec_mse_weight: float = 1.0,
                 rec_tv_l1_weight: float or None = None,
                 rec_log_tv_weight: float or None = None,
                 rec_log_tv_eps: float = 0.001,
                 name='custom_rec_loss'):
        super().__init__(name=name)

        self.rec_mse_weight: float = rec_mse_weight
        self.rec_tv_l1_weight: float or None = rec_tv_l1_weight
        self.rec_log_tv_weight: float or None = rec_log_tv_weight
        self.rec_log_tv_eps: float = rec_log_tv_eps

        self.mse = MeanSquaredError()

    def call(self, rec_true, rec_pred):
        loss = self.mse(rec_true, rec_pred) * self.rec_mse_weight

        if self.rec_tv_l1_weight is not None:
            loss += total_variation_mean_norm(rec_true - rec_pred) * self.rec_tv_l1_weight

        if self.rec_log_tv_weight is not None:
            loss += logarithmic_total_variation_objective_function(rec_pred, self.rec_log_tv_eps) * self.rec_log_tv_weight

        return loss


class ScaledMeanSquaredError(MeanSquaredError):
    def __init__(self, scale=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def call(self, y_true, y_pred):
        return super().call(y_true, y_pred) * self.scale
