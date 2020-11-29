import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Progbar

from LIDCArtifactReduction.directory_system import DirectorySystem
from LIDCArtifactReduction.metrics import HU_MAE, RadioSNR, SSIM, MeanSquare, RelativeError, MeanSumSquare
from LIDCArtifactReduction.neural_nets.ModelInterface import ModelInterface
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet import IterativeARTResNet
from LIDCArtifactReduction.neural_nets.radon_layer import ForwardRadonLayer
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform


class IterativeARTResNetTrainingV3(ModelInterface):
    def __init__(self, radon_transformation: ForwardprojectionRadonTransform,
                 target_model: IterativeARTResNet, dir_system: DirectorySystem,
                 name=None):
        super().__init__(name=name)

        self._radon_transformation = radon_transformation
        self._target_model = target_model
        self._weight_dir = dir_system.MODEL_WEIGHTS_DIRECTORY

        self._build()

    # TODO: delete this comment, if naming did not cause weightloading problems
    # previous name 'sinos_output_layer'
    difference_radon_layer_name = 'difference_radon_layer'
    output_radon_layer_name = 'output_radon_layer'

    def _build(self):
        target_imgs_input_layer = self._target_model.imgs_input_layer
        target_sinos_input_layer = self._target_model.sinos_input_layer
        target_imgs_output_layer, target_difference_layer = self._target_model.output_layer_or_layers

        difference_radon_layer = ForwardRadonLayer(radon_transformation=self._radon_transformation,
                                                   name=IterativeARTResNetTrainingV3.difference_radon_layer_name)(target_difference_layer)
        output_radon_layer = ForwardRadonLayer(radon_transformation=self._radon_transformation,
                                               name=IterativeARTResNetTrainingV3.output_radon_layer_name)(target_imgs_output_layer)

        self._model = Model(
            inputs=[target_imgs_input_layer, target_sinos_input_layer],
            outputs=[target_imgs_output_layer, difference_radon_layer, output_radon_layer],
            name=self._target_model.name
        )


    @staticmethod
    def output_data_formatter(actual_reconstructions, bad_sinograms, good_reconstructions, good_sinograms, **kwargs):
        return {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                IterativeARTResNet.sinos_input_name: bad_sinograms,
                IterativeARTResNet.resnet_name: good_reconstructions,
                IterativeARTResNetTrainingV3.output_radon_layer_name: good_sinograms}


    @staticmethod
    def input_data_decoder(data):
        return data[IterativeARTResNet.imgs_input_name], \
               data[IterativeARTResNet.sinos_input_name], \
               data[IterativeARTResNet.resnet_name], \
               data[IterativeARTResNetTrainingV3.output_radon_layer_name]


    def compile(self, final_level=4, reconstructions_output_weight=1.0, reconstruction_weight_amplifier=1.0,
                measurement_consistency_weight=None, kernel_error_weight=None, gradient_weight=None,
                lr=1e-3):
        """
        Args:
            reconstruction_weight_amplifier: the amplifier constant for reconstruction loss throughout iterations
            measurement_consistency_weight: weight factor of the error between the Radon transformed output
                and expected sinogram. Low error ensures consistency with measurements. If None, it is not considered.
            kernel_error_weight: weight for error regarding the sinogram of the UNet output. 0 error would mean, that
                the resnet part of the model only makes transformation in the kernel space of the Radon operator.
                If None, no loss is calculated.
            gradient_weight: error on the tangent derivative of the network wrt the input. If None, it is not considered.
        """
        self.final_level = final_level

        self.reconstruction_weight_amplifier = tf.convert_to_tensor(reconstruction_weight_amplifier, dtype=tf.float32)
        self.reconstructions_output_weight = reconstructions_output_weight
        self.measurement_consistency_weight = measurement_consistency_weight
        self.kernel_error_weight = kernel_error_weight
        self.gradient_weight = gradient_weight

        # Setting metrics.
        self._all_metrics = []
        # Possibilities: Rec: MSE, RMSE, MAE, MSE in HU, MAE in HU, RMSE in HU, SNR, SSIM, relative error
        #        Choose: SNR, SSIM, HU_MAE, MSE, rel_error
        #                sino_error: MS
        #                differential: MS
        self._metrics_iteration_reconstructions = []
        for i in range(self.final_level):
            self._metrics_iteration_reconstructions.append(metrics.MeanSquaredError(f'rec_mse_{i+1}'))
        self._metrics_final_reconstruction = [HU_MAE('rec_HU_mae'),
                                              RadioSNR('rec_snr'),
                                              SSIM('rec_ssim'),
                                              RelativeError('rec_rel_err')]
        self._monitored_metric = self._metrics_final_reconstruction[0]
        self._all_metrics += self._metrics_final_reconstruction

        if self.measurement_consistency_weight is not None:
            self._metrics_measurement_consistency = [keras.metrics.MeanSquaredError('radon_mse')]
            self._all_metrics += self._metrics_measurement_consistency

        if self.kernel_error_weight is not None:
            self._metrics_kernel_error = [MeanSquare('kernel_error_ms')]
            self._all_metrics += self._metrics_kernel_error

        if self.gradient_weight is not None:
            self._metrics_gradient = [MeanSumSquare('gradient_mss')]
            self._all_metrics += self._metrics_gradient

        self._model.compile(optimizer=optimizers.Adam(lr))


    # TODO: is tf.function needed, or not?
    #@tf.function
    def _train_step(self, data):
        actual_reconstructions_0, bad_sinograms, good_reconstructions, good_sinograms = \
            IterativeARTResNetTrainingV3.input_data_decoder(data)

        mse = keras.losses.MeanSquaredError()


        def step_level_i(i, actual_reconstructions, total_loss):
        
            inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                    IterativeARTResNet.sinos_input_name: bad_sinograms}

            if self.gradient_weight is not None:
                with tf.autodiff.ForwardAccumulator(
                    primals=actual_reconstructions,
                    tangents=tf.math.l2_normalize(good_reconstructions - actual_reconstructions, axis=[1, 2, 3])
                ) as acc:
                    reconstruction_output, kernel_error_output, radon_output = self._model(inputs, training=True)

                tangent_der = acc.jvp(reconstruction_output)
                total_loss += self.gradient_weight * tangent_der
                for m in self._metrics_gradient:
                    m.update_state(tangent_der)

            else:
                reconstruction_output, kernel_error_output, radon_output = self._model(inputs, training=True)

            loss_amplifier = self.reconstruction_weight_amplifier ** tf.cast(i, dtype=tf.float32)
            total_loss += loss_amplifier * mse(reconstruction_output, good_reconstructions)

            self._metrics_iteration_reconstructions[i].update_state(good_reconstructions, reconstruction_output)

            if self.measurement_consistency_weight is not None:
                total_loss += self.measurement_consistency_weight * mse(radon_output, good_sinograms)
                for m in self._metrics_measurement_consistency:
                    m.update_state(radon_output, good_sinograms)

            if self.kernel_error_weight is not None:
                total_loss += self.kernel_error_weight * tf.reduce_mean(tf.square(kernel_error_output))
                for m in self._metrics_kernel_error:
                    m.update_state(kernel_error_output)

            return i+1, reconstruction_output, bad_sinograms, good_reconstructions, total_loss


        with tf.GradientTape() as tape:
            _, actual_reconstructions_final, _, _, total_loss = \
                tf.while_loop(cond=lambda i, *args, **kwargs: i < self.final_level,
                          body=step_level_i,
                          loop_vars=(0, actual_reconstructions_0, tf.constant(0.0)),
                          swap_memory=True)

        loss_gradient = tape.gradient(total_loss, self._model.trainable_variables)
        # Update weights
        self._model.optimizer.apply_gradients(zip(loss_gradient, self._model.trainable_variables))

        # Update other metrics
        for m in self._metrics_final_reconstruction:
            m.update_state(good_reconstructions, actual_reconstructions_final)


    def train(self, train_iterator, epochs, steps_per_epoch):
        print(f"Training network {self.name}.")
        for epoch in range(epochs):
            print(f"Epoch {epoch}:")
            progbar = Progbar(steps_per_epoch, verbose=1, stateful_metrics=[m.name for m in self._all_metrics])
            for step in range(1, steps_per_epoch + 1):
                data = next(train_iterator)
                self._train_step(data)

                if step % 1 == 0:
                    progbar.update(step, values=[(m.name, m.result().numpy()) for m in self._all_metrics])

            print("Saving model")
            self._model.save_weights(
                filepath=os.path.join(self._weight_dir, self._name) + '-' + self._monitored_metric.name + '-' +
                         '{a:.3f}'.format(a=self._monitored_metric.result().numpy()[0]) + '.hdf5')

            for m in self._all_metrics:
                m.reset_states()
            # Possible validation could be added here.
