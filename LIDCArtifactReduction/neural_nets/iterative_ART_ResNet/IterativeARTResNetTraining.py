import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError

from LIDCArtifactReduction.neural_nets.ModelInterface import ModelInterface
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet import IterativeARTResNet
from LIDCArtifactReduction.neural_nets.radon_layer import ForwardRadonLayer
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform


class IterativeARTRestNetTrainingLoss:
    def __init__(self, reconstructions_output_weight, error_singrom_weight, derivative_weight):
        self._reconstructions_output_weight = tf.convert_to_tensor(reconstructions_output_weight, dtype=tf.float32)
        self._error_sinogram_weight = tf.convert_to_tensor(error_singrom_weight, dtype=tf.float32)
        self._derivative_weight = tf.convert_to_tensor(derivative_weight, dtype=tf.float32)

        self._mse = MeanSquaredError()

    def __call__(self, reconstructions_output, error_sinogram, good_reconstructions, doutput_dinput):
        return self._reconstructions_output_weight * self._mse(reconstructions_output, good_reconstructions) + \
               self._error_sinogram_weight *  tf.reduce_mean(tf.square(error_sinogram)) + \
               self._derivative_weight * tf.reduce_mean(tf.square(doutput_dinput))


class IterativeARTResNetTraningCustomTrainStepModel(Model):
    def __init__(self, custom_loss: IterativeARTRestNetTrainingLoss, **kwargs):
        super().__init__(**kwargs)
        self._custom_loss = custom_loss

    def _input_data_decoder(self, data):
        return data[IterativeARTResNet.imgs_input_name], \
               data[IterativeARTResNet.sinos_input_name], \
               data[IterativeARTResNet.resnet_name]

    def train_step(self, data):
        # TODO: somehow pack input of network into dictionary
        actual_reconstructions, bad_sinograms, good_reconstructions = self._input_data_decoder(data)

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                tape2.watch(actual_reconstructions)
                reconstructions_output, error_sinogram = self([actual_reconstructions, bad_sinograms], training=True)
            # TODO: derivative wrt input or art output?
            doutput_dinput = tape2.gradient(reconstructions_output, actual_reconstructions)

            lossvalue = self._custom_loss(reconstructions_output, error_sinogram, doutput_dinput, good_reconstructions)
        trainable_variable_gradients = tape1.gradient(lossvalue, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(trainable_variable_gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(good_reconstructions, reconstructions_output)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}




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

        self._model = Model(inputs=[target_imgs_input_layer, target_sinos_input_layer],
                            outputs=[target_imgs_output_layer, radon_layer])


