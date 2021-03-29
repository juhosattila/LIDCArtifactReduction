import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, LambdaCallback

from LIDCArtifactReduction import utility
from LIDCArtifactReduction.directory_system import DirectorySystem
from LIDCArtifactReduction.metrics import HU_RMSE, ReconstructionReference2Noise, SSIM, HU_MAE, Signal2Noise, \
    Signal2NoiseStandardDeviance, RelativeError, RadonMeanSquaredError, RadonRelativeError
from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform
from LIDCArtifactReduction.neural_nets.ModelInterface import ModelInterface
from LIDCArtifactReduction.neural_nets.FBPPostProcessingResNet.target_networks import DCAR_TargetInterface
from LIDCArtifactReduction.neural_nets.radon_layer import ForwardRadonLayer


class DCAR_TrainingNetwork(ModelInterface):
    def __init__(self, radon_geometry: RadonGeometry, radon_transformation: ForwardprojectionRadonTransform,
                 target_model: DCAR_TargetInterface, dir_system: DirectorySystem, name=None):
        super().__init__(name=name, weight_dir=dir_system.MODEL_WEIGHTS_DIRECTORY)
        self._target_model = target_model
        self._input_shape = target_model.input_shape

        self._radon_geometry = radon_geometry
        self._radon_transformation = radon_transformation

        self._input_layer = None
        self._expected_output_layer = None
        self._expected_Radon_layer = None

        self._total_variation_loss_set = False

        self._directory_system = dir_system

        self._build_model()

    sino_output_name = 'radon_layer'
    input_name = DCAR_TargetInterface.input_name
    reconstruction_output_name = DCAR_TargetInterface.reconstruction_output_name


    def _build_model(self):
        target_input_layer = self._target_model.input_layer
        target_output_layer = self._target_model.output_layer

        expected_output_layer = target_output_layer
        expected_Radon_layer = ForwardRadonLayer(self._radon_transformation,
                                                 name=DCAR_TrainingNetwork.sino_output_name)(expected_output_layer)

        model = Model(
            inputs=target_input_layer,
            outputs=[expected_output_layer, expected_Radon_layer],
            name=self.name)

        self._input_layer = target_input_layer
        self._expected_output_layer = expected_output_layer
        self._expected_Radon_layer = expected_Radon_layer
        self._model = model

    @property
    def target_model(self):
        return self._target_model


    @staticmethod
    def input_data_encoder(bad_reconstructions, bad_sinograms, good_reconstructions, good_sinograms, **kwargs):
        # TODO: if upgraded to TF 2.2 remove [None]
        # These 'None'-s correspond to weights attached to the outputs.
        return ({DCAR_TrainingNetwork.input_name: bad_reconstructions},
                {DCAR_TrainingNetwork.reconstruction_output_name: good_reconstructions,
                 DCAR_TrainingNetwork.sino_output_name: good_sinograms}, [None, None])


    @staticmethod
    def input_data_decoder(data):
        return data[0][DCAR_TrainingNetwork.input_name], \
               data[1][DCAR_TrainingNetwork.reconstruction_output_name], \
               data[1][DCAR_TrainingNetwork.sino_output_name]


    def compile(self, lr=1e-3, reconstruction_loss=None, sinogram_loss=None, **kwargs):
        # Losses:
        losses = {DCAR_TrainingNetwork.reconstruction_output_name: reconstruction_loss,
                  DCAR_TrainingNetwork.sino_output_name: sinogram_loss}
        loss_weights = None

        # Metrics

        # Possibilities: Rec: MSE, RMSE, MAE, MSE in HU, MAE in HU, RMSE in HU, RNR, SNR, SSIM, relative error
        #           New: SNR, Standard Variance on SNR (or any)
        #        Choose: RNR, SNR, SNR STD, SSIM, HU_MAE, MSE, rel_error
        #                sino_error: MSE, rel_error
        metrics = { DCAR_TrainingNetwork.reconstruction_output_name:
                        [HU_MAE(name='hu_mae'),
                         keras.losses.MeanSquaredError(name='rec_mse'),
                         ReconstructionReference2Noise(name='rec_rnr'),
                         Signal2Noise(name='rec_snr'),
                         Signal2NoiseStandardDeviance(name='rec_snr_std'),
                         SSIM(name='rec_ssim'),
                         RelativeError(name='rec_rel_err'),
                         RadonMeanSquaredError(radon_transformation=self._radon_transformation, name='radon_mse'),
                         RadonRelativeError(radon_transformation=self._radon_transformation, name='radon_rel_error')
                         ],
                    DCAR_TrainingNetwork.sino_output_name:
                        [keras.losses.MeanSquaredError(name='sino_mse'),
                         RelativeError(name='sino_rel_error')]}


        self._model.compile(optimizer=Adam(lr),
                            loss=losses,
                            loss_weights=loss_weights,
                            metrics=metrics)

    def fit(self, train_iterator, validation_iterator,
            epochs: int, steps_per_epoch=None, validation_steps=None,
            early_stoppig_patience: int or None = None, csv_logging: bool = False,
            test_data=None,
            verbose=1, initial_epoch=0):

        ## Callbacks
        # Modelcheckpointer
        name_datetime = self.name + '-' + datetime.now().strftime("%Y%m%d-%H%M")
        monitored_value = 'val_' + DCAR_TrainingNetwork.reconstruction_output_name + '_hu_mae'

        file = os.path.join(self.weight_dir, name_datetime)
        #file = file + '.{epoch:02d}-{' + monitored_value + ':.1f}' + self._model_weights_extension
        file = file + '-HUMAE-{' + monitored_value + ':.1f}' + self._model_weights_extension
        checkpointer = ModelCheckpoint(
                        monitor=monitored_value,
                        filepath=file, save_best_only=True,
                        save_weights_only=True, verbose=1,
                        save_freq='epoch')

        # Tensorboard
        tensorboard_logdir = utility.direc(self._directory_system.TENSORBOARD_LOGDIR, "fit", name_datetime)
        tensorboard = TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1, write_graph=True)

        callbacks = [checkpointer, tensorboard]

        if early_stoppig_patience is not None:
            earlystopping = EarlyStopping(patience=early_stoppig_patience, verbose=1)
            callbacks.append(earlystopping)

        if csv_logging:
            txt_logdir = utility.direc(self._directory_system.CSV_LOGDIR, "fit")
            txt_filename = os.path.join(txt_logdir, name_datetime + '.log')
            csvlogger = CSVLogger(filename=txt_filename)
            callbacks.append(csvlogger)

        if test_data is not None:
            test_image_direc = utility.direc(tensorboard_logdir, 'case_studies')
            tb_test_image_writer = tf.summary.create_file_writer(test_image_direc)
            tb_test_image_writer.set_as_default()

            bad_rec, good_rec, good_sino = DCAR_TrainingNetwork.input_data_decoder(test_data)
            nr_imgs = len(bad_rec)

            def log_test_data(epoch, logs):
                test_reconstructions, test_sinograms = self._model(bad_rec)
                for i in range(0, nr_imgs):
                    tf.summary.image(
                        name=f"test_image_{i}", 
                        data=tf.stack([good_rec[i], bad_rec[i], test_reconstructions[i]]),
                        step=epoch)
            test_data_cb = LambdaCallback(on_epoch_end=log_test_data)
            callbacks.append(test_data_cb)


        # Number of batches used.
        # Use entire dataset once.
        if steps_per_epoch is None:
            steps_per_epoch: int = len(train_iterator)
        if validation_steps is None:
            validation_steps: int = len(validation_iterator)

        return self._model.fit(x=train_iterator, validation_data=validation_iterator,
                        epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                        callbacks=callbacks, verbose=verbose, initial_epoch=initial_epoch)


    def predict(self, data_iterator, steps=None,
                tensorboard_logging: bool = False, csv_logging: bool = False,
                verbose=1):
        callbacks = []

        name_datetime = self.name + '-' + datetime.now().strftime("%Y%m%d-%H%M")
        # Tensorboard
        if tensorboard_logging:
            tensorboard_logdir = utility.direc(self._directory_system.TENSORBOARD_LOGDIR, "predict", name_datetime)
            tensorboard = TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1, write_graph=True)
            callbacks.append(tensorboard)

        # CSV logging
        if csv_logging:
            txt_logdir = utility.direc(self._directory_system.CSV_LOGDIR, "predict")
            txt_filename = os.path.join(txt_logdir, name_datetime + '.log')
            csvlogger = CSVLogger(filename=txt_filename)
            callbacks.append(csvlogger)

        # Number of batches used.
        # Use entire dataset once.
        if steps is None:
            steps: int = len(data_iterator)

        return self._model.predict(x=data_iterator, steps=steps, callbacks=callbacks, verbose=verbose)


    def evaluate(self, data_iterator, steps=None,
                tensorboard_logging: bool = False, csv_logging: bool = False,
                verbose=1):
        callbacks = []

        name_datetime = self.name + '-' + datetime.now().strftime("%Y%m%d-%H%M")
        # Tensorboard
        if tensorboard_logging:
            tensorboard_logdir = utility.direc(self._directory_system.TENSORBOARD_LOGDIR, "evaluate", name_datetime)
            tensorboard = TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1, write_graph=True)
            callbacks.append(tensorboard)

        # CSV logging
        if csv_logging:
            txt_logdir = utility.direc(self._directory_system.CSV_LOGDIR, "evaluate")
            txt_filename = os.path.join(txt_logdir, name_datetime + '.log')
            csvlogger = CSVLogger(filename=txt_filename)
            callbacks.append(csvlogger)

        # Number of batches used.
        # Use entire dataset once.
        if steps is None:
            steps: int = len(data_iterator)

        return self._model.evaluate(x=data_iterator, steps=steps, callbacks=callbacks, verbose=verbose)
