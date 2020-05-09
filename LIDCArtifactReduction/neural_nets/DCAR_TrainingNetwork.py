import os
from datetime import datetime

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from LIDCArtifactReduction import parameters
from LIDCArtifactReduction.generator import LIDCDataIterator
from LIDCArtifactReduction.tf_image import SparseTotalVariationObjectiveFunction
from LIDCArtifactReduction.neural_nets.interfaces import ModelInterface, DCAR_TargetInterface
from LIDCArtifactReduction.radon_layer import RadonLayer
from LIDCArtifactReduction.radon_params import RadonParams


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

    def compile(self, adam_lr=1e-3):
        # TODO: specify metrics

        # Custom losses and metrics
        if not self._total_variation_loss_set:
            # change based on projection values. This goes for 256
            tot_var_loss_weight = 1e-2
            tot_var_regualizer = SparseTotalVariationObjectiveFunction(eps=5.0 / parameters.HU_TO_CT_SCALING)  # 5HU / scaling
            self._model.add_loss(tot_var_regualizer(self._expected_output_layer) * tot_var_loss_weight)
            self._total_variation_loss_set = True

        losses = {DCAR_TrainingNetwork.reconstruction_output_name : MeanSquaredError(name='mse_reconstrction'),
                  DCAR_TrainingNetwork.sino_output_name : MeanSquaredError(name='mse_radon_space')}
        loss_weights = {DCAR_TrainingNetwork.reconstruction_output_name : 1.0,
                        DCAR_TrainingNetwork.sino_output_name : 1.0 / parameters.NR_OF_SPARSE_ANGLES}

        # TODO: continue
        metrics = {}

        self._model.compile(optimizer=Adam(adam_lr),
                            loss=losses,
                            loss_weights=loss_weights,
                            metrics=metrics)

    def fit(self, train_iterator: LIDCDataIterator, validation_iterator: LIDCDataIterator,
            epochs: int,
            verbose=1, adam_lr=1e-3, initial_epoch=0):

        self.set_training(training=True)
        self.compile(adam_lr)

        # TODO schedule learning decay, scheduler or manually

        # We are going to use early stopping and model saving-reloading mechanism.
        file = os.path.join(parameters.MODEL_WEIGHTS_DIRECTORY, self._name)
        file = file + '.{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpointer = ModelCheckpoint(filepath=file, save_best_only=True,
                                       save_weights_only=True, verbose=1)
        earlystopping = EarlyStopping(patience=10, verbose=1)

        # Tensorboard
        # TODO: supplement tensorboard with metrics and others
        logdir = os.path.join(parameters.PROJECT_DIRECTORY,
                              "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard = TensorBoard(logdir=logdir, histogram_freq=1, write_graph=True)

        callbacks = [checkpointer, earlystopping, tensorboard]

        # Number of batches used.
        # Use entire dataset once
        steps_per_epoch: int = len(train_iterator)
        validation_steps: int = len(validation_iterator)

        return self._model.fit(x=train_iterator, validation_data=validation_iterator,
                        epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                        callbacks=callbacks, verbose=verbose, initial_epoch=initial_epoch)

    def set_training(self, training : bool):
        self.target_model.set_training(training)
