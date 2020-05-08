import os
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from abc import abstractmethod

from LIDCArtifactReduction import parameters


class ModelInterface:
    object_counter = {}

    def __init__(self, name=None):
        valid_name = name
        if valid_name is None:
            class_name = type(self).__name__
            ModelInterface.object_counter[class_name] = ModelInterface.object_counter.get(class_name, 0) + 1
            valid_name = class_name + '_' + str(ModelInterface.object_counter[class_name])
        self._name = valid_name
        self._model : Model = None

    @property
    def name(self):
        return self._name

    def summary(self):
        self._model.summary()

    def plot_model(self, to_file=None, show_shapes=True):
        direc = parameters.MODEL_PLOTS_DIRECTORY
        if not os.path.exists(direc):
            os.mkdir(direc)
        valid_to_file = to_file if to_file is not None else (self.name + '.png')
        final_to_file = os.path.join(direc, valid_to_file)
        plot_model(self._model, final_to_file, show_shapes)

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def set_training(self, training : bool):
        pass

    def save(self):
        file = os.path.join(parameters.MODEL_DIRECTORY, self._name)

        # If format='h5' is used, losses and custom objects need to be handled separately.
        self._model.save(file, save_format='tf')

    def save_weights(self):
        file = os.path.join(parameters.MODEL_WEIGHTS_DIRECTORY, self._name)
        self._model.save_weights(file + '.h5')

    def load_weights(self, name=None):
        valid_name = name if name is not None else self._name
        file = os.path.join(parameters.MODEL_WEIGHTS_DIRECTORY, valid_name)
        self._model.load_weights(file + '.h5')


class DCAR_TargetInterface(ModelInterface):
    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def input_layer(self):
        pass

    @property
    @abstractmethod
    def output_layer(self):
        pass

    @abstractmethod
    def set_training(self, training: bool):
        pass
