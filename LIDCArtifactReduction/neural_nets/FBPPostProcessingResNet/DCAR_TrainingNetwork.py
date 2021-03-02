import os
from datetime import datetime

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger

from LIDCArtifactReduction import utility
from LIDCArtifactReduction.directory_system import DirectorySystem
from LIDCArtifactReduction.metrics import HU_RMSE, ReconstructionReference2Noise, SSIM, HU_MAE, Signal2Noise, \
    Signal2NoiseStandardDeviance, RelativeError
from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform
from LIDCArtifactReduction.tf_image import LogarithmicTotalVariationObjectiveFunction
from LIDCArtifactReduction.neural_nets.ModelInterface import ModelInterface
from LIDCArtifactReduction.neural_nets.FBPPostProcessingResNet.target_networks import DCAR_TargetInterface
from LIDCArtifactReduction.neural_nets.radon_layer import ForwardRadonLayer
from LIDCArtifactReduction.neural_nets.FBPPostProcessingResNet.losses import MSE_TV_diff_loss


class DCAR_TrainingNetwork(ModelInterface):
    def __init__(self, radon_geometry: RadonGeometry, radon_transformation: ForwardprojectionRadonTransform,
                 target_model: DCAR_TargetInterface, dir_system: DirectorySystem, name=None):
        super().__init__(name)
        self._target_model = target_model
        self._input_shape = target_model.input_shape

        self._radon_geometry = radon_geometry
        self._radon_transformation = radon_transformation

        self._input_layer = None
        self._expected_output_layer = None
        self._expected_Radon_layer = None

        self._total_variation_loss_set = False

        self._weight_dir = dir_system.MODEL_WEIGHTS_DIRECTORY
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


    def compile(self, lr=1e-3, reconstruction_loss=None, sinogram_loss=None,
                reconstruction_output_weight=1.0,
                sino_output_weight : float or 'auto' or None = 'auto',
                add_total_variation=True, total_variation_eps=1.0, tot_var_loss_weight=1e-3,
                mse_tv_weight=3.0):
        """
        Args:
             sino_output_weight: if 'auto', it is specified as 1.0 / self._radon_geometry.nr_projections
        """
        # _sino_output_weight: float or 'auto' or None = sino_output_weight
        # if _sino_output_weight == 'auto':
        #     _sino_output_weight = 1.0 / self._radon_geometry.nr_projections

        # TODO: delete if not necessary
        # # Losses

        # # This is added only once, because it becomes part of the topology.
        # # TODO: make it removeable.
        # # !!! If once added, it will stay in the topology, hence a second call to compile with add_tv=false will have
        # # no effect.
        # if add_total_variation and not self._total_variation_loss_set:
        #     tot_var_regualizer = LogarithmicTotalVariationObjectiveFunction(total_variation_eps)
        #
        #     self._model.add_loss(tot_var_regualizer(self._expected_output_layer) * tot_var_loss_weight)
        #     self._total_variation_loss_set = True
        #
        #
        # # TODO: if necessary, make it switcheable from API
        # # Loss settings:
        #
        # # First setting.
        # # losses = {DCAR_TrainingNetwork.reconstruction_output_name : MeanSquaredError(name='mse_reconstrction'),
        # #           DCAR_TrainingNetwork.sino_output_name : MeanSquaredError(name='mse_radon_space')}
        #
        # # The second one contains a TV L2 squared loss on the differenceimage in order to preserve edges.
        # # losses = {DCAR_TrainingNetwork.reconstruction_output_name:
        # #               LIDCArtifactReduction.losses.MSE_TV_squared_diff_loss(tv_weight=mse_tv_weight,
        # #                                                                     name='mse_tv_square_diff'),
        # #           DCAR_TrainingNetwork.sino_output_name: MeanSquaredError(name='mse_radon_space')}
        #
        # # Third setting: TV L1.
        # losses = {DCAR_TrainingNetwork.reconstruction_output_name:
        #               MSE_TV_diff_loss(tv_weight=mse_tv_weight, name='mse_tv_square_diff'),
        #           DCAR_TrainingNetwork.sino_output_name:
        #               keras.losses.MeanSquaredError(name='mse_radon_space')}
        #
        #
        # loss_weights = {DCAR_TrainingNetwork.reconstruction_output_name : reconstruction_output_weight,
        #                 DCAR_TrainingNetwork.sino_output_name : sino_output_weight}

        # Losses:
        losses = {DCAR_TrainingNetwork.reconstruction_output_name: reconstruction_loss,
                  DCAR_TrainingNetwork.sino_output_name: sinogram_loss}
        loss_weights = None

        # Metrics

        # Possibilities: Rec: MSE, RMSE, MAE, MSE in HU, MAE in HU, RMSE in HU, RNR, SNR, SSIM, relative error
        #           New: SNR, Standard Variance on SNR (or any)
        #        Choose: RNR, SNR, SNR STD, SSIM, HU_MAE, MSE, rel_error
        #                sino_error: MSE
        metrics = { DCAR_TrainingNetwork.reconstruction_output_name:
                        [HU_MAE(name='hu_mae'),
                         keras.losses.MeanSquaredError(name='rec_mse'),
                         ReconstructionReference2Noise(name='rec_rnr'),
                         Signal2Noise(name='rec_snr'),
                         Signal2NoiseStandardDeviance(name='rec_snr_std'),
                         SSIM(name='rec_ssim'),
                         RelativeError(name='rec_rel_err')
                         ],
                    DCAR_TrainingNetwork.sino_output_name:
                        [keras.losses.MeanSquaredError(name='sino_mse')]}


        self._model.compile(optimizer=Adam(lr),
                            loss=losses,
                            loss_weights=loss_weights,
                            metrics=metrics)

    def fit(self, train_iterator, validation_iterator,
            epochs: int, steps_per_epoch=None, validation_steps=None,
            early_stoppig_patience=5, verbose=1, initial_epoch=0):

        # We are going to use early stopping and model saving mechanism.
        monitored_value = 'val_' + DCAR_TrainingNetwork.reconstruction_output_name + '_loss'
        file = os.path.join(self._weight_dir, self._name)
        file = file + '.{epoch:02d}-{' + monitored_value + ':.4f}' + self._model_weights_extension
        checkpointer = ModelCheckpoint(
                        monitor=monitored_value,
                        filepath=file, save_best_only=True,
                        save_weights_only=True, verbose=1,
                        save_freq='epoch')
        earlystopping = EarlyStopping(patience=early_stoppig_patience, verbose=1)

        # Tensorboard and logging
        datetimenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_logdir = utility.direc(self._directory_system.TENSORBOARD_LOGDIR, "fit", datetimenow)
        tensorboard = TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1, write_graph=True)
        txt_logdir = utility.direc(self._directory_system.CSV_LOGDIR, "fit")
        txt_filename = os.path.join(txt_logdir, datetimenow + '.log')
        csvlogger = CSVLogger(filename=txt_filename)

        # callbacks = [checkpointer, earlystopping, tensorboard, csvlogger]
        callbacks = [tensorboard]

        # Number of batches used.
        # Use entire dataset once.
        if steps_per_epoch is None:
            steps_per_epoch: int = len(train_iterator)
        if validation_steps is None:
            validation_steps: int = len(validation_iterator)

        return self._model.fit(x=train_iterator, validation_data=validation_iterator,
                        epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                        callbacks=callbacks, verbose=verbose, initial_epoch=initial_epoch)

    def predict(self, data_iterator, steps=None, verbose=1):
        # Tensorboard and logging
        datetimenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_logdir = utility.direc(self._directory_system.TENSORBOARD_LOGDIR, "predict", datetimenow)
        tensorboard = TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1, write_graph=True)
        txt_logdir = utility.direc(self._directory_system.CSV_LOGDIR, "predict")
        txt_filename = os.path.join(txt_logdir, datetimenow + '.log')
        csvlogger = CSVLogger(filename=txt_filename)

        callbacks = [tensorboard, csvlogger]

        # Number of batches used.
        # Use entire dataset once.
        if steps is None:
            steps: int = len(data_iterator)

        return self._model.predict(x=data_iterator, steps=steps, callbacks=callbacks, verbose=verbose)

    # TODO: evaluation
