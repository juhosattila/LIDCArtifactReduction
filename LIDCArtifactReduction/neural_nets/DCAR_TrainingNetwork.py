import os
from datetime import datetime

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger

from LIDCArtifactReduction import utility
from LIDCArtifactReduction import parameters
from LIDCArtifactReduction.metrics import HU_RMSE, RadioSNR, SSIM
from LIDCArtifactReduction.tf_image import SparseTotalVariationObjectiveFunction, TotalVariationNormObjectiveFunction
from LIDCArtifactReduction.neural_nets.interfaces import ModelInterface, DCAR_TargetInterface
from LIDCArtifactReduction.radon_layer import RadonLayer
from LIDCArtifactReduction.radon_params import RadonParams
import LIDCArtifactReduction.losses


class DCAR_TrainingNetwork(ModelInterface):
    def __init__(self, radon_params: RadonParams,
                 target_model: DCAR_TargetInterface, name=None):
        super().__init__(name)
        self._target_model = target_model
        self._input_shape = target_model.input_shape

        self._radon_params = radon_params

        self._input_layer = None
        self._expected_output_layer = None
        self._expected_Radon_layer = None

        self._total_variation_loss_set = False

        self._build_model()

    sino_output_name = 'radon_layer'
    input_name = DCAR_TargetInterface.input_name
    reconstruction_output_name = DCAR_TargetInterface.reconstruction_output_name

    def _build_model(self):
        target_input_layer = self._target_model.input_layer
        target_output_layer = self._target_model.output_layer

        expected_output_layer = target_output_layer
        expected_Radon_layer = RadonLayer(self._radon_params, name=DCAR_TrainingNetwork.sino_output_name)\
                                (expected_output_layer)

        model = Model(inputs=target_input_layer, outputs=[expected_output_layer, expected_Radon_layer],
                      name=self.name)

        self._input_layer = target_input_layer
        self._expected_output_layer = expected_output_layer
        self._expected_Radon_layer = expected_Radon_layer
        self._model = model

    @property
    def target_model(self):
        return self._target_model

    def compile(self, adam_lr=1e-3, reconstruction_output_weight=1.0,
                sino_output_weight=1.0 / parameters.NR_OF_SPARSE_ANGLES,
                add_total_variation=True, total_variation_eps=1.0, tot_var_loss_weight=1e-3,
                mse_tv_weight = 3.0):
        # Losses
        if add_total_variation and not self._total_variation_loss_set:
            #tot_var_regualizer = SparseTotalVariationObjectiveFunction(eps=100.0 / parameters.HU_TO_CT_SCALING)  # 5HU / scaling
            tot_var_regualizer = SparseTotalVariationObjectiveFunction(total_variation_eps)
            # TODO: not tested and does not work
            #tot_var_regualizer = TotalVariationNormObjectiveFunction()

            self._model.add_loss(tot_var_regualizer(self._expected_output_layer) * tot_var_loss_weight)
            self._total_variation_loss_set = True

         #TODO: change back
        # losses = {DCAR_TrainingNetwork.reconstruction_output_name : MeanSquaredError(name='mse_reconstrction'),
        #           DCAR_TrainingNetwork.sino_output_name : MeanSquaredError(name='mse_radon_space')}

        losses = {DCAR_TrainingNetwork.reconstruction_output_name:
                      LIDCArtifactReduction.losses.MSE_TV_square_diff_loss(tv_weight=mse_tv_weight, name='mse_tv_square_diff'),
                  DCAR_TrainingNetwork.sino_output_name: MeanSquaredError(name='mse_radon_space')}

        loss_weights = {DCAR_TrainingNetwork.reconstruction_output_name : reconstruction_output_weight,
                        DCAR_TrainingNetwork.sino_output_name : sino_output_weight}

        # Metrics
        metrics = { DCAR_TrainingNetwork.reconstruction_output_name:
                        [RootMeanSquaredError(name='rmse_reconstruction'),
                         HU_RMSE(name='rmse_HU_recontruction'),
                         RadioSNR(name='SNR_reconstruction'),
                         SSIM(name='ssim')],
                    DCAR_TrainingNetwork.sino_output_name:
                        [RootMeanSquaredError(name='rmse_radon_space')]}

        self._model.compile(optimizer=Adam(adam_lr),
                            loss=losses,
                            loss_weights=loss_weights,
                            metrics=metrics)

    def fit(self, train_iterator, validation_iterator,
            epochs: int, steps_per_epoch=None, validation_steps=None,
            early_stoppig_patience = 5, verbose=1, initial_epoch=0):

        self.set_training(training=True)

        # We are going to use early stopping and model saving mechanism.
        monitored_value = 'val_' + DCAR_TrainingNetwork.reconstruction_output_name + '_loss'
        file = os.path.join(parameters.MODEL_WEIGHTS_DIRECTORY, self._name)
        file = file + '.{epoch:02d}-{' + monitored_value + ':.4f}' + self._model_weights_extension
        checkpointer = ModelCheckpoint(
                        monitor=monitored_value,
                        filepath=file, save_best_only=True,
                        save_weights_only=True, verbose=1,
                        save_freq='epoch')
        earlystopping = EarlyStopping(patience=early_stoppig_patience, verbose=1)

        # Tensorboard and logging
        datetimenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_logdir = utility.direc(parameters.TENSORBOARD_LOGDIR, "fit", datetimenow)
        tensorboard = TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1, write_graph=True)
        txt_logdir = utility.direc(parameters.CSV_LOGDIR, "fit")
        txt_filename = os.path.join(txt_logdir, datetimenow + '.log')
        csvlogger = CSVLogger(filename=txt_filename)

        callbacks = [checkpointer, earlystopping, tensorboard, csvlogger]

        # Number of batches used.
        # Use entire dataset once.
        if steps_per_epoch is None:
            steps_per_epoch: int = len(train_iterator)
        if validation_steps is None:
            validation_steps: int = len(validation_iterator)

        return self._model.fit(x=train_iterator, validation_data=validation_iterator,
                        epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                        callbacks=callbacks, verbose=verbose, initial_epoch=initial_epoch)

    def set_training(self, training : bool):
        self.target_model.set_training(training)

    def predict(self, data_iterator, steps=None, verbose=1):
        self.set_training(training=False)

        # Tensorboard and logging
        datetimenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_logdir = utility.direc(parameters.TENSORBOARD_LOGDIR, "predict", datetimenow)
        tensorboard = TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1, write_graph=True)
        txt_logdir = utility.direc(parameters.CSV_LOGDIR, "predict")
        txt_filename = os.path.join(txt_logdir, datetimenow + '.log')
        csvlogger = CSVLogger(filename=txt_filename)

        callbacks = [tensorboard, csvlogger]

        # Number of batches used.
        # Use entire dataset once.
        if steps is None:
            steps: int = len(data_iterator)

        return self._model.fit(x=data_iterator, steps=steps, callbacks=callbacks, verbose=verbose)
