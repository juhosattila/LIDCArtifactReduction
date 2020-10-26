import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from LIDCArtifactReduction.metrics import HU_MAE, RadioSNR, SSIM, MeanSquare
from LIDCArtifactReduction.neural_nets.ModelInterface import ModelInterface
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet import IterativeARTResNet
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTRestNet_training_loss import \
    IterativeARTRestNetTrainingLoss, RecSinoGradientLoss
from LIDCArtifactReduction.neural_nets.radon_layer import ForwardRadonLayer
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform


class IterativeARTResNetTraining(ModelInterface):
    def __init__(self, radon_transformation: ForwardprojectionRadonTransform,
                 target_model: IterativeARTResNet,
                 name=None):
        super().__init__(name=name)

        self._radon_transformation = radon_transformation
        self._target_model = target_model

        self._build()

    sinos_output_name = 'sinos_output_layer'

    def _build(self):
        target_imgs_input_layer = self._target_model.imgs_input_layer
        target_sinos_input_layer = self._target_model.sinos_input_layer
        target_imgs_output_layer, target_difference_layer = self._target_model.output_layer_or_layers

        radon_layer = ForwardRadonLayer(radon_transformation=self._radon_transformation,
                                        name=IterativeARTResNetTraining.sinos_output_name)\
                                        (target_difference_layer)

        self._model = IterativeARTResNetTraningCustomTrainStepModel(
                        inputs=[target_imgs_input_layer, target_sinos_input_layer],
                        outputs=[target_imgs_output_layer, radon_layer])

    def compile(self, lr, reconstructions_output_weight, error_singrom_weight, gradient_weight):
        # Setting losses.
        loss_object = RecSinoGradientLoss(reconstructions_output_weight, error_singrom_weight, gradient_weight)
        self._model.compile(lr=lr, loss=loss_object)

    def fit(self):
        raise NotImplementedError()
        # TODO: continue here


class IterativeARTResNetTraningCustomTrainStepModel(Model):
    def __init__(self, custom_loss: IterativeARTRestNetTrainingLoss = None, **kwargs):
        super().__init__(**kwargs)
        self._custom_loss: IterativeARTRestNetTrainingLoss = custom_loss
        self._all_metrics = None

    def _input_data_decoder(self, data):
        return data[IterativeARTResNet.imgs_input_name], \
               data[IterativeARTResNet.sinos_input_name], \
               data[IterativeARTResNet.resnet_name]

    def train_step(self, data):
        actual_reconstructions, bad_sinograms, good_reconstructions = self._input_data_decoder(data)
        inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                  IterativeARTResNet.sinos_input_name: bad_sinograms}

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                tape2.watch(actual_reconstructions)
                #reconstructions_output, error_sinogram = self([actual_reconstructions, bad_sinograms], training=True)
                reconstructions_output, error_sinogram = self(inputs, training=True)
                # TODO: derivative wrt input or art output?
            doutput_dinput = tape2.gradient(reconstructions_output, actual_reconstructions)

            lossvalue = self._custom_loss(
                            reconstructions_output=reconstructions_output,
                            error_sinogram=error_sinogram,
                            doutput_dinput=doutput_dinput,
                            good_reconstructions=good_reconstructions)
        trainable_variable_gradients = tape1.gradient(lossvalue, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(trainable_variable_gradients, self.trainable_variables))

        # Update metrics
        for m in self._metrics_reconstruction:
            m.update_state(good_reconstructions, reconstructions_output)
        for m in self._metrics_error_sino:
            m.update_state(error_sinogram)
        for m in self._metrics_gradient:
            m.update_state(doutput_dinput)

        # If you had compiled metrics.
        # self.compiled_metrics.update_state(good_reconstructions, reconstructions_output)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    def compile(self, lr, loss):
        # Setting losses
        self._custom_loss = loss

        # Possibilities: Rec: MSE, RMSE, MAE, MSE in HU, MAE in HU, RMSE in HU, SNR, SSIM
        #        Choose: SNR, SSIM, HU_MAE, MSE
        #                sino_error: MS
        #                differential: MS
        self._metrics_reconstruction = [metrics.MeanSquaredError('rec_mse'),
                                  HU_MAE('rec_HU_mae'),
                                  RadioSNR('rec_radionsnr'),
                                  SSIM('rec_ssim')]
        self._metrics_error_sino = [MeanSquare('error_sino_ms')]
        self._metrics_gradient = [MeanSquare('gradient_ms')]

        self._all_metrics = self._metrics_reconstruction + self._metrics_error_sino + self._metrics_gradient

        super().compile(optimizer=optimizers.Adam(lr))

    @property
    def metrics(self):
        return self._all_metrics
