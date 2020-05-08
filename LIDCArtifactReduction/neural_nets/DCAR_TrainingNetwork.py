from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from LIDCArtifactReduction import parameters
from LIDCArtifactReduction.tf_image import SparseTotalVariationObjectiveFunction
from LIDCArtifactReduction.neural_nets import ModelInterface, DCAR_TargetInterface

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

        self._build_model()

    def _build_model(self):
        target_input_layer = self._target_model.input_layer
        target_output_layer = self._target_model.output_layer

        expected_output_layer = target_output_layer

        # TODO: set the one that is needed
        expected_Radon_layer = RadonLayer(self._radon_params)(expected_output_layer)
        #expected_Radon_layer = expected_output_layer

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
        # TODO: metrics
        # TODO schedule learning decay

        # change based on projection values. This goes for 256
        tot_var_loss_weight = 1e-2

        tot_var_regualizer = SparseTotalVariationObjectiveFunction(eps=5.0 / parameters.HU_TO_CT_SCALING)  # 5HU / scaling
        self._model.add_loss( tot_var_regualizer(self._expected_output_layer) * tot_var_loss_weight)

        # TODO: szotarszeruen rendez loss-okat es metrikakat
        self._model.compile(optimizer=Adam(adam_lr),
                            loss=[MeanSquaredError(), MeanSquaredError()],
                            loss_weights=[1.0, 1.0 / parameters.NR_OF_SPARSE_ANGLES])

    def set_training(self, training : bool):
        self.target_model.set_training(training)
