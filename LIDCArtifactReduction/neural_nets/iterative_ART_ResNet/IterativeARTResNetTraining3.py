import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Progbar

from LIDCArtifactReduction.directory_system import DirectorySystem
from LIDCArtifactReduction.metrics import HU_MAE, RadioSNR, SSIM, MeanSquare, RelativeError
from LIDCArtifactReduction.neural_nets.ModelInterface import ModelInterface
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet import IterativeARTResNet
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTRestNet_training_loss \
    import IterativeARTRestNetTrainingLoss, RecSinoGradientLoss
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.data_formatter import input_data_decoder
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.iterator import RecSinoSuperIterator, RecSinoArrayIterator
from LIDCArtifactReduction.neural_nets.radon_layer import ForwardRadonLayer
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform


class IterativeARTResNetTraining(ModelInterface):
    def __init__(self, radon_transformation: ForwardprojectionRadonTransform,
                 target_model: IterativeARTResNet, dir_system: DirectorySystem,
                 name=None):
        super().__init__(name=name)

        self._radon_transformation = radon_transformation
        self._target_model = target_model
        self._weight_dir = dir_system.MODEL_WEIGHTS_DIRECTORY

        self._build()

    sinos_output_name = 'sinos_output_layer'

    def _build(self):
        target_imgs_input_layer = self._target_model.imgs_input_layer
        target_sinos_input_layer = self._target_model.sinos_input_layer
        target_imgs_output_layer, target_difference_layer = self._target_model.output_layer_or_layers

        radon_layer = ForwardRadonLayer(radon_transformation=self._radon_transformation,
                                        name=IterativeARTResNetTraining.sinos_output_name)(target_difference_layer)

        self._model = Model(
            inputs=[target_imgs_input_layer, target_sinos_input_layer],
            outputs=[target_imgs_output_layer, radon_layer],
            name=self._target_model.name
        )


    # TODO: provide meaningful defaults
    def compile(self, lr, reconstructions_output_weight, error_singrom_weight, gradient_weight):
        # Setting losses.
        self._custom_loss = RecSinoGradientLoss(reconstructions_output_weight, error_singrom_weight, gradient_weight)

        # Setting metrics.
        # Possibilities: Rec: MSE, RMSE, MAE, MSE in HU, MAE in HU, RMSE in HU, SNR, SSIM, relative error
        #        Choose: SNR, SSIM, HU_MAE, MSE, rel_error
        #                sino_error: MS
        #                differential: MS
        self._metrics_reconstruction = [metrics.MeanSquaredError('rec_mse'),
                                        HU_MAE('rec_HU_mae'),
                                        RadioSNR('rec_snr'),
                                        SSIM('rec_ssim'),
                                        RelativeError('rec_rel_err')]
        self._monitored_metric = self._metrics_reconstruction[1]
        self._metrics_error_sino = [MeanSquare('error_sino_ms')]
        self._metrics_gradient = [MeanSquare('gradient_ms')]

        self._all_metrics = self._metrics_reconstruction + self._metrics_error_sino + self._metrics_gradient

        self._model.compile(optimizer=optimizers.Adam(lr))


    # TODO: implement
    def apply(self):
        # kiertekel kezdeti iteraciokat es futtat halot k-szor
        raise NotImplementedError()




    # TODO: implement entire loss function
    def _loss_function(self, actual_reconstructions, bad_sinograms, good_reconstructions):
        inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                  IterativeARTResNet.sinos_input_name: bad_sinograms}

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(actual_reconstructions)
            reconstruction_output, error_sinogram = self._model(inputs, training=True)
            from tensorflow.keras import losses
            loss = losses.MeanSquaredError()(reconstruction_output, good_reconstructions)

        loss_gradient = tape.gradient(loss, self._model.trainable_variables)

        return reconstruction_output, loss_gradient, tape

    def _eval(self, actual_reconstructions, bad_sinograms):
        inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                  IterativeARTResNet.sinos_input_name: bad_sinograms}
        reconstruction_output, error_sinogram = self._model(inputs, training=True)
        return reconstruction_output, bad_sinograms





    def _step_level(self, actual_reconstructions, bad_sinograms, good_reconstructions, total_loss):
        inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                  IterativeARTResNet.sinos_input_name: bad_sinograms}

        reconstruction_output, error_sinogram = self._model(inputs, training=True)
        # TODO: refactor
        from tensorflow.keras import losses
        total_loss += losses.MeanSquaredError()(reconstruction_output, good_reconstructions)

        return reconstruction_output, bad_sinograms, good_reconstructions, total_loss

    #@tf.function
    def _train_step(self, data):
        actual_reconstructions_0, bad_sinograms, good_reconstructions = input_data_decoder(data)

        # actual_reconstructions_0 = tf.Variable(actual_reconstructions_0, trainable=False)
        # bad_sinograms = tf.Variable(bad_sinograms, trainable=False)
        # good_reconstructions = tf.Variable(good_reconstructions, trainable=False)

        # with tf.GradientTape() as tape:
        #     actual_reconstructions_final, _, _, total_loss = \
        #         tf.while_loop(cond=lambda *args, **kwargs: True,
        #                 # TODO: take out level
        #                   maximum_iterations=5,
        #                   body=self._step_level,
        #                   loop_vars=(actual_reconstructions_0, bad_sinograms, good_reconstructions, tf.constant(0.0)),
        #                   swap_memory=True)

        # loss_gradient = tape.gradient(total_loss, self._model.trainable_variables)
        # # Update weights
        # self._model.optimizer.apply_gradients(zip(loss_gradient, self._model.trainable_variables))

        def step_level_i(i, actual_reconstructions, bad_sinograms, good_reconstructions, total_loss):
        
            inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                    IterativeARTResNet.sinos_input_name: bad_sinograms}

            reconstruction_output, error_sinogram = self._model(inputs, training=True)
            # TODO: refactor
            from tensorflow.keras import losses
            total_loss += losses.MeanSquaredError()(reconstruction_output, good_reconstructions)

            return i+1, reconstruction_output, bad_sinograms, good_reconstructions, total_loss

        with tf.GradientTape() as tape:
            _, actual_reconstructions_final, _, _, total_loss = \
                tf.while_loop(cond=lambda i, *args, **kwargs: i < 6,
                          body=step_level_i,
                          loop_vars=(0, actual_reconstructions_0, bad_sinograms, good_reconstructions, tf.constant(0.0)),
                          swap_memory=True)

        # with tf.GradientTape() as tape:
        #     total_loss = tf.constant(0.0)
        #     for _ in tf.range(5):
        #         tf.autograph.experimental.set_loop_options(swap_memory=True)
        #         inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions_0,
        #             IterativeARTResNet.sinos_input_name: bad_sinograms}

        #         actual_reconstructions_0, error_sinogram = self._model(inputs, training=True)
        #         # TODO: refactor
        #         from tensorflow.keras import losses
        #         total_loss += losses.MeanSquaredError()(actual_reconstructions_0, good_reconstructions)

        loss_gradient = tape.gradient(total_loss, self._model.trainable_variables)
        # Update weights
        self._model.optimizer.apply_gradients(zip(loss_gradient, self._model.trainable_variables))


        # actual_reconstructions_final, _, _, total_loss = \
        #         tf.while_loop(cond=lambda *args, **kwargs: True,
        #                 # TODO: take out level
        #                   maximum_iterations=5,
        #                   body=self._step_level,
        #                   loop_vars=(actual_reconstructions_0, bad_sinograms, good_reconstructions, tf.constant(0.0)),
        #                   # TODO: try allowing swapping
        #                   swap_memory=False)

        # loss_gradient = tf.gradients(total_loss, self._model.trainable_variables)
        # # # Update weights
        # self._model.optimizer.apply_gradients(zip(loss_gradient, self._model.trainable_variables))

        # Update metrics
        for m in self._metrics_reconstruction:
            m.update_state(good_reconstructions, actual_reconstructions_final)
        # for m in self._metrics_error_sino:
        #     m.update_state(errors_sinogram)
        # for m in self._metrics_gradient:
        #     m.update_state(doutput_dinput)

        # If you had compiled metrics.
        # self.compiled_metrics.update_state(good_reconstructions, reconstructions_output)

        # For overriding train_step in Tf>=2.2 return a dict mapping metric names to current value
        # return {m.name: m.result() for m in self.metrics}

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
    # TODO: add logic for increasing number of levels, if necessary
