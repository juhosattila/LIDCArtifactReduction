import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from LIDCArtifactReduction.metrics import HU_MAE, RadioSNR, SSIM, MeanSquare, RelativeError
from LIDCArtifactReduction.neural_nets.ModelInterface import ModelInterface
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet import IterativeARTResNet
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTRestNet_training_loss import \
    IterativeARTRestNetTrainingLoss, RecSinoGradientLoss
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.data_formatter import input_data_decoder
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.iterator import RecSinoSuperIterator, RecSinoArrayIterator
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

    # TODO: provide meaningful defaults
    def compile(self, lr, reconstructions_output_weight, error_singrom_weight, gradient_weight):
        # Setting losses.
        loss_object = RecSinoGradientLoss(reconstructions_output_weight, error_singrom_weight, gradient_weight)
        self._model.compile(lr=lr, loss=loss_object)


    # TODO: implement
    def apply(self):
        # kiertekel kezdeti iteraciokat es futtat halot k-szor
        raise NotImplementedError()



    # Toggle, if performance needed.
    #@tf.function
    def predict_depth_generator_step(self, data_batch):
        actual_reconstructions, bad_sinograms, good_reconstructions = input_data_decoder(data_batch)
        inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                  IterativeARTResNet.sinos_input_name: bad_sinograms}

        reconstructions_output, errors_sinogram = self._model(inputs, training=False)

        return reconstructions_output, bad_sinograms, good_reconstructions


    def predict_depth_generator(self, data_iterator, depth, steps):
        """
        Args:
            data_iterator: should provide data in the form of a dict containing keys
                IterativeARTResNet.imgs_input_name and IterativeARTResNet.sinos_input_name
        """
        if depth == 0:
            return data_iterator

        new_actual = []
        new_sino = []
        new_good_reco = []

        # Lots of optimisation needed. Outputting sinograms and good_reconstructions could be optimized.
        nr_steps = steps or len(data_iterator)
        for i in range(nr_steps):
            data_batch = next(data_iterator)
            reconstructions_output, bad_sinograms, good_reconstructions = \
                self.predict_depth_generator_step(data_batch)
            new_actual.append(reconstructions_output)
            new_sino.append(bad_sinograms)
            new_good_reco.append(good_reconstructions)

        new_data_iterator = RecSinoArrayIterator(new_actual, new_sino, new_good_reco)
        return self.predict_depth_generator(new_data_iterator, depth-1, steps=None)


    def train(self, train_iterator, final_depth, steps_per_epoch):
        """
        Args:
            final_depth: the depth the system should be trained to reach. Should be at least 1.
        """
        # What capability level is taught now.
        for actual_depth in range(1, final_depth+1):  # actual depth <= final_depth
            # Data needs to be generated for each sublevel.
            iterators = []
            for data_level in range(actual_depth):
                iterators.append(self.predict_depth_generator(train_iterator, depth=data_level, steps=steps_per_epoch))

            super_iterator = RecSinoSuperIterator(iterators)

            # TODO: checkpointer is needed to save weights
            self._model.fit(super_iterator, epochs=1, steps_per_epoch=len(iterators) * steps_per_epoch)


class IterativeARTResNetTraningCustomTrainStepModel(Model):
    def __init__(self, custom_loss: IterativeARTRestNetTrainingLoss = None, **kwargs):
        super().__init__(**kwargs)
        self._custom_loss: IterativeARTRestNetTrainingLoss = custom_loss
        self._all_metrics = None


    def train_step(self, data):
        actual_reconstructions, bad_sinograms, good_reconstructions = input_data_decoder(data)
        inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions,
                  IterativeARTResNet.sinos_input_name: bad_sinograms}

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                tape2.watch(actual_reconstructions)
                #reconstructions_output, errors_sinogram = self([actual_reconstructions, bad_sinograms], training=True)
                reconstructions_output, errors_sinogram = self(inputs, training=True)
                # TODO: derivative wrt input or art output?
            doutput_dinput = tape2.gradient(reconstructions_output, actual_reconstructions)

            lossvalue = self._custom_loss(
                            reconstructions_output=reconstructions_output,
                            error_sinogram=errors_sinogram,
                            doutput_dinput=doutput_dinput,
                            good_reconstructions=good_reconstructions)
        trainable_variable_gradients = tape1.gradient(lossvalue, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(trainable_variable_gradients, self.trainable_variables))

        # Update metrics
        for m in self._metrics_reconstruction:
            m.update_state(good_reconstructions, reconstructions_output)
        for m in self._metrics_error_sino:
            m.update_state(errors_sinogram)
        for m in self._metrics_gradient:
            m.update_state(doutput_dinput)

        # If you had compiled metrics.
        # self.compiled_metrics.update_state(good_reconstructions, reconstructions_output)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    # TODO: if necessary, use, otherwise delete. predict_step overriden
    # def predict_step(self, data):
    #     actual_reconstructions, bad_sinograms, good_reconstructions = \
    #         IterativeARTResNetTraningCustomTrainStepModel.input_data_decoder(data)
    #     inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions,
    #               IterativeARTResNet.sinos_input_name: bad_sinograms}
    #     reconstructions_output, errors_sinogram = self(inputs, training=False)
    #     return reconstructions_output, errors_sinogram


    def compile(self, lr, loss):
        # Setting losses
        self._custom_loss = loss

        # Possibilities: Rec: MSE, RMSE, MAE, MSE in HU, MAE in HU, RMSE in HU, SNR, SSIM, relative error
        #        Choose: SNR, SSIM, HU_MAE, MSE, rel_error
        #                sino_error: MS
        #                differential: MS
        self._metrics_reconstruction = [metrics.MeanSquaredError('rec_mse'),
                                  HU_MAE('rec_HU_mae'),
                                  RadioSNR('rec_snr'),
                                  SSIM('rec_ssim'),
                                  RelativeError('rec_rel_err')]
        self._metrics_error_sino = [MeanSquare('error_sino_ms')]
        self._metrics_gradient = [MeanSquare('gradient_ms')]

        self._all_metrics = self._metrics_reconstruction + self._metrics_error_sino + self._metrics_gradient

        super().compile(optimizer=optimizers.Adam(lr))

    @property
    def metrics(self):
        return self._all_metrics
