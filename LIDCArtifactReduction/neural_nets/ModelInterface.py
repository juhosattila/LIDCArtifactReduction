import os
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from abc import abstractmethod

from LIDCArtifactReduction import directory_system, utility


class ModelInterface:
    object_counter = {}

    def __init__(self, name=None, weight_dir=None):
        valid_name = name
        if valid_name is None:
            class_name = type(self).__name__
            ModelInterface.object_counter[class_name] = ModelInterface.object_counter.get(class_name, 0) + 1
            valid_name = class_name + '_' + str(ModelInterface.object_counter[class_name])
        self._name = valid_name
        self._model : Model = None

        self._model_weights_extension = '.hdf5'
        self._weight_dir = weight_dir or directory_system.BASE_MODEL_WEIGHTS_DIRECTORY

    @property
    def name(self):
        return self._name

    @property
    def weight_dir(self):
        return self._weight_dir

    @weight_dir.setter
    def weight_dir(self, value):
        self._weight_dir = value

    @property
    def model_weights_extension(self):
        return self._model_weights_extension

    def summary(self):
        self._model.summary()

    def plot_model(self, to_file=None, show_shapes=True):
        direc = directory_system.MODEL_PLOTS_DIRECTORY
        valid_to_file = to_file if to_file is not None else (self.name + '.png')
        final_to_file = os.path.join(direc, valid_to_file)
        plot_model(self._model, final_to_file, show_shapes)

    def __call__(self, inputs, training=False):
        return self._model(inputs, training=training)

    # # TODO: not yet used, beacuse Radon layers are not serialisable
    # def save(self):
    #     raise NotImplementedError()
    #     file = os.path.join(directory_system.MODEL_DIRECTORY, self._name)
    #     # If format='h5' is used, losses and custom objects need to be handled separately.
    #     self._model.save(file, save_format='tf')

    def save_weights(self):
        filename = os.path.join(self.weight_dir, self.name) + self.model_weights_extension
        self._model.save_weights(filename)

    def load_weights(self, name=None, latest=False):
        """
        Args:
            name: filename containing the weights. Could have full path or not. Extension attached,
                if it does not already have. See extension in class.
            latest: If true, and there is a weight file in the weight directory, then the latest of them is loaded,
                ignoring 'name'. If there is not a file, than 'name' is loaded.
        """
        file = utility.get_filepath(name=name, latest=latest,
                             directory=self.weight_dir, extension=self.model_weights_extension)

        print("----------------------------------")
        print("Loading model weights contained in file:")
        print(file)
        print("----------------------------------")

        self._model.load_weights(file)
        return self
