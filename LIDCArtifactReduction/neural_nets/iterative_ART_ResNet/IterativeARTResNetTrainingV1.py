import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Progbar
from tensorflow_core.python.keras.preprocessing.image import Iterator as KerasImgIterator

from LIDCArtifactReduction.directory_system import DirectorySystem
from LIDCArtifactReduction.metrics import HU_MAE, RadioSNR, SSIM, MeanSquare, RelativeError
from LIDCArtifactReduction.neural_nets.ModelInterface import ModelInterface
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet import IterativeARTResNet
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTRestNet_training_loss \
    import IterativeARTRestNetTrainingLoss, RecSinoGradientLoss
from LIDCArtifactReduction.neural_nets.radon_layer import ForwardRadonLayer
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform


class IterativeARTResNetTrainingV1(ModelInterface):
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
                                        name=IterativeARTResNetTrainingV1.sinos_output_name)(target_difference_layer)

        self._model = Model(
            inputs=[target_imgs_input_layer, target_sinos_input_layer],
            outputs=[target_imgs_output_layer, radon_layer],
            name=self._target_model.name
        )


    @staticmethod
    def output_data_formatter(actual_reconstructions, bad_sinograms, good_reconstructions, **kwargs):
        return {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                IterativeARTResNet.sinos_input_name: bad_sinograms,
                IterativeARTResNet.resnet_name: good_reconstructions}


    @staticmethod
    def input_data_decoder(data):
        return data[IterativeARTResNet.imgs_input_name], \
               data[IterativeARTResNet.sinos_input_name], \
               data[IterativeARTResNet.resnet_name]


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


    # Toggle, if performance needed.
    @tf.function
    def _predict_depth_generator_step(self, data_batch):
        actual_reconstructions, bad_sinograms, good_reconstructions = \
            IterativeARTResNetTrainingV1.input_data_decoder(data_batch)
        inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                  IterativeARTResNet.sinos_input_name: bad_sinograms}

        reconstructions_output, errors_sinogram = self._model(inputs, training=False)

        return reconstructions_output, bad_sinograms, good_reconstructions


    def predict_depth_generator(self, data_iterator, depth, steps, progbar: Progbar = None):
        """
        Args:
            data_iterator: should provide data in the form of a dict containing keys
                IterativeARTResNet.imgs_input_name and IterativeARTResNet.sinos_input_name
        """
        if progbar is not None:
            progbar.add(1)

        if depth == 0:
            return data_iterator

        new_actual = []
        new_sino = []
        new_good_reco = []

        # Lots of optimisation needed. Outputting sinograms and good_reconstructions could be optimized.
        nr_steps = steps or len(data_iterator)
        progbar_sublevel = Progbar(target=nr_steps, verbose=1)
        for i in range(nr_steps):
            data_batch = next(data_iterator)
            reconstructions_output, bad_sinograms, good_reconstructions = \
                self._predict_depth_generator_step(data_batch)
            new_actual.append(reconstructions_output.numpy())
            new_sino.append(bad_sinograms.numpy())
            new_good_reco.append(good_reconstructions.numpy())

            progbar_sublevel.update(i + 1)

        new_data_iterator = RecSinoArrayIterator(new_actual, new_sino, new_good_reco)
        return self.predict_depth_generator(new_data_iterator, depth - 1, steps=None, progbar=progbar)


    # Toggle for debugging or deployment.
    @tf.function
    def _train_depth_step(self, data):
        actual_reconstructions, bad_sinograms, good_reconstructions = \
            IterativeARTResNetTrainingV1.input_data_decoder(data)
        inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                  IterativeARTResNet.sinos_input_name: bad_sinograms}

        with tf.GradientTape() as tape1:

            # TODO: mean computation of tangent gradient must be changed so that mean is only on batch dimension,
            #  not on the entire batch. On other dimensions standard summation is needed.

            with tf.autodiff.ForwardAccumulator(
                primals=actual_reconstructions,
                tangents=tf.math.l2_normalize(good_reconstructions - actual_reconstructions, axis=[1, 2, 3])
            ) as acc:
                # If not dictionary:
                # reconstructions_output, errors_sinogram = self([actual_reconstructions, bad_sinograms], training=True)
                reconstructions_output, errors_sinogram = self._model(inputs, training=True)
                # reconstructions_output, errors_sinogram = self([reconstructions_output, bad_sinograms], training=True)

            # TODO: delete unnecessary prints
            print("-----SHAPES----")
            print(tf.shape(actual_reconstructions))
            print(tf.shape(reconstructions_output))

            doutput_dinput = acc.jvp(reconstructions_output)

            print(tf.shape(doutput_dinput))
            print("-----SHAPES END----")
            # doutput_dinput = [grad if grad is not None else tf.zeros_like(var) for grad, var in zip(doutput_dinput, actual_reconstructions)]

            lossvalue = self._custom_loss(
                reconstructions_output=reconstructions_output,
                error_sinogram=errors_sinogram,
                doutput_dinput=doutput_dinput,
                good_reconstructions=good_reconstructions)
        trainable_variable_gradients = tape1.gradient(lossvalue, self._model.trainable_variables)

        # Update weights
        self._model.optimizer.apply_gradients(zip(trainable_variable_gradients, self._model.trainable_variables))

        # Update metrics
        for m in self._metrics_reconstruction:
            m.update_state(good_reconstructions, reconstructions_output)
        for m in self._metrics_error_sino:
            m.update_state(errors_sinogram)
        for m in self._metrics_gradient:
            m.update_state(doutput_dinput)

        # If you had compiled metrics.
        # self.compiled_metrics.update_state(good_reconstructions, reconstructions_output)

        # For overriding train_step in Tf>=2.2 return a dict mapping metric names to current value
        # return {m.name: m.result() for m in self.metrics}

    def _train_depth(self, iterator, epochs, steps_per_epoch, weights_filepath):
        print(f"Training network {self.name}.")
        for epoch in range(epochs):
            print(f"Epoch {epoch}:")
            progbar = Progbar(steps_per_epoch, verbose=1, stateful_metrics=[m.name for m in self._all_metrics])
            for step in range(1, steps_per_epoch + 1):
                data = next(iterator)
                self._train_depth_step(data)

                if step % 1 == 0:
                    progbar.update(step, values=[(m.name, m.result().numpy()) for m in self._all_metrics])

            print("Saving model")
            self._model.save_weights(
                filepath=weights_filepath + '-' + self._monitored_metric.name + '-' + '{a:.3f}'.format(
                    a=self._monitored_metric.result().numpy()[0]) + '.hdf5')

            for m in self._all_metrics:
                m.reset_states()
            # Possible validation could be added here.


    def train(self, train_iterator, final_depth, data_epoch, steps_per_epoch, start_depth=1, start_epoch=1):
        """
        Args:
            final_depth: the depth the system should be trained to reach. Should be at least 1.
        """
        _start_epoch = start_epoch

        print("---------------------------------")
        print("--- Training to final level {}----".format(final_depth))
        print("---------------------------------")

        # What capability level is taught now.
        for actual_depth in range(start_depth, final_depth + 1):  # actual depth <= final_depth
            print("---------------------------------")
            print("--- Training to actual level {}---".format(actual_depth))
            for de in range(_start_epoch, data_epoch + 1):
                print("--- Data epoch {}----------------".format(de))
                print("--- Starting data preparation ---")

                # Data needs to be generated for each sublevel.
                iterators = []
                for data_level in range(actual_depth):
                    print("---Data level {}---".format(data_level))
                    iterators.append(
                        self.predict_depth_generator(train_iterator, depth=data_level, steps=steps_per_epoch,
                                                     progbar=None))

                super_iterator = RecSinoSuperIterator(iterators)

                self._train_depth(super_iterator, epochs=1, steps_per_epoch=actual_depth * steps_per_epoch,
                                  weights_filepath=os.path.join(self._weight_dir, self._name))

            # In upcoming levels we start from epoch 1.
            _start_epoch = 1


class RecSinoArrayIterator(KerasImgIterator):
    def __init__(self, actual_reconstructions_batches, sinograms_batches, good_reconstructions_batches):
        super().__init__(len(actual_reconstructions_batches), 1, shuffle=False, seed=None)
        self._actual_reconstructions_batches = actual_reconstructions_batches
        self._sinograms_batches = sinograms_batches
        self._good_reconstructions_batches = good_reconstructions_batches

    def _get_batches_of_transformed_samples(self, index_array):
        idx = index_array[0]
        return IterativeARTResNetTrainingV1.output_data_formatter(
            actual_reconstructions=tf.convert_to_tensor(self._actual_reconstructions_batches[idx], dtype=tf.float32),
            bad_sinograms=tf.convert_to_tensor(self._sinograms_batches[idx], dtype=tf.float32),
            good_reconstructions=tf.convert_to_tensor(self._good_reconstructions_batches[idx], dtype=tf.float32)
        )


class RecSinoSuperIterator(KerasImgIterator):
    def __init__(self, iterators):
        self._iterators = iterators
        super().__init__(len(iterators), batch_size=1, shuffle=True, seed=None)

    def _get_batches_of_transformed_samples(self, index_array):
        idx = index_array[0]
        return next(self._iterators[idx])
