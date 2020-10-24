import os
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from abc import abstractmethod

from LIDCArtifactReduction import directory_system, utility


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

        self._model_weights_extension = '.hdf5'
        self._weight_dir = directory_system.MODEL_WEIGHTS_DIRECTORY

    @property
    def name(self):
        return self._name

    def summary(self):
        self._model.summary()

    def plot_model(self, to_file=None, show_shapes=True):
        direc = directory_system.MODEL_PLOTS_DIRECTORY
        valid_to_file = to_file if to_file is not None else (self.name + '.png')
        final_to_file = os.path.join(direc, valid_to_file)
        plot_model(self._model, final_to_file, show_shapes)

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def set_training(self, training : bool):
        """After set_training we usually need recompilation."""
        pass

    # TODO: this is not needed
    def set_inference_mode(self):
        """Sets model to inference mode. Recompilation done."""
        self.set_training(training=False)
        self.compile()

    def set_learning_mode(self):
        """After this we usually need recompilation."""
        self.set_training(training=True)

    def __call__(self, inputs):
        return self._model(inputs)

    # TODO: not yet used, beacuse Radon layers are not serialisable
    def save(self):
        raise NotImplementedError()
        file = os.path.join(directory_system.MODEL_DIRECTORY, self._name)
        # If format='h5' is used, losses and custom objects need to be handled separately.
        self._model.save(file, save_format='tf')

    def save_weights(self):
        file = os.path.join(self._weight_dir, self._name)
        self._model.save_weights(file + self._model_weights_extension)

    def load_weights(self, name=None, latest=False):
        """
        Args:
            name: filename containing the weights. Could have full path or not. Extension attached,
                if it does not already have. See extension in class.
            latest: If true, and there is a weight file in the weight directory, then the latest of them is loaded,
                ignoring 'name'. If there is not a file, than 'name' is loaded.
        """
        file = utility.get_filepath(name=name, latest=latest,
                             directory=self._weight_dir, extension=self._model_weights_extension)

        print("----------------------------------")
        print("Loading model weights contained in file:")
        print(file)
        print("----------------------------------")

        self._model.load_weights(file)
        return self

