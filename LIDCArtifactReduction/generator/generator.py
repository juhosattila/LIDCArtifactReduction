import numpy as np
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import Iterator as KerasImgIterator

from LIDCArtifactReduction import utility
from LIDCArtifactReduction.array_streams import ArrayStream
from LIDCArtifactReduction.generator.generator_transform import LIDCGeneratorTransform


class LIDCDataGenerator:
    """Custom generator class for LIDC data.

    Data is assumed to be found in a directory passed to init.
    Capable of shuffling data, augmenting it with possible Poisson noise.
    It can provide iterator objects for training, validation and testing.
    """
    def __init__(self, array_stream: ArrayStream, data_configuration_dir,
                 validation_split=0.1, test_split=0.1, batch_size=16,
                 shuffle : bool = True,
                 verbose : bool = True,
                 load_data_config : bool or str = False):
        """
        Args:
            load_data_config: boolean or string. If False, then data is shuffled.
                If True, then latest configuration is loaded. If string, than a filename is specified.
        """
        self._batch_size = batch_size
        self._array_stream = array_stream
        self._shuffle = shuffle

        self._load_data_configuration = load_data_config
        self._data_configuraion_file_format = '.json'
        self._data_configuration_dir = data_configuration_dir

        validation_test_split = validation_split + test_split

        if self._load_data_configuration is False:
            patient_dirs = self._array_stream.get_directory_names()

            train_directories, valid_test_directories = train_test_split(patient_dirs,
                train_size=1.0-validation_test_split, shuffle=self._shuffle)
            valid_directories, test_directories = train_test_split(valid_test_directories,
                train_size=validation_split / validation_test_split, shuffle=self._shuffle)

            # saving filenames
            if verbose:
                print("### LIDC Generator ###")
                print("-----------------------------")
                print("Created data configuration, saving to disk...")
                print("-----------------------------")
            self._save_data_configuration(train_directories, valid_directories, test_directories)

        else:
            if self._load_data_configuration is True:  # load previously created data
                latest = True
                filename = None
                if verbose:
                    print("### LIDC Generator ###")
                    print("-----------------------------")
                    print("Found data configuration, loading latest...")
                    print("-----------------------------")

            else:  # load_data_configuration is a str specifying a filename
                print("### LIDC Generator ###")
                print("-----------------------------")
                print("Found data configuration, loading specified: {}".format(self._load_data_configuration))
                print("-----------------------------")
                latest = False
                filename = self._load_data_configuration

            train_directories, valid_directories, test_directories = \
                self._load_data_config(filename=filename, latest=latest)

        train_arrnames = self._array_stream.get_names_with_dir(train_directories)
        valid_arrnames = self._array_stream.get_names_with_dir(valid_directories)
        test_arrnames = self._array_stream.get_names_with_dir(test_directories)

        if verbose:
            print("### LIDC Generator ###")
            print("-----------------------------")
            print("Train dirs: ", len(train_directories))
            print("E.g.: ", train_directories[:5])
            print("Train arrs: ", len(train_arrnames))
            print("-----------------------------")
            print("Valid dirs: ", len(valid_directories))
            print("E.g.: ", valid_directories[:5])
            print("Valid arrs: ", len(valid_arrnames))
            print("-----------------------------")
            print("Test dirs: ", len(test_directories))
            print("E.g.: ", test_directories[:5])
            print("Test arrs: ", len(test_arrnames))
            print("-----------------------------")

        self._train_arrnames = train_arrnames
        self._valid_arrnames = valid_arrnames
        self._test_arrnames = test_arrnames


    def _save_data_configuration(self, train_directories, valid_directories, test_directories):
        data_config = {'train': train_directories, 'valid': valid_directories, 'test': test_directories}
        filename = 'config' + datetime.now().strftime("%Y%m%d-%H%M%S") + self._data_configuraion_file_format
        filepath = os.path.join(self._data_configuration_dir, filename)
        with open(filepath, 'w') as file:
            json.dump(data_config, file)

    def _load_data_config(self, filename=None, latest=False):
        filepath = utility.get_filepath(name=filename, directory=self._data_configuration_dir,
                                        latest=latest, extension=self._data_configuraion_file_format)
        with open(filepath) as file:
            config = json.load(file)
        return config['train'], config['valid'], config['test']

    def _get_iterator(self, transformer, arrnames, seed=None):
        return LIDCDataIterator(arrnames=arrnames, batch_size=self._batch_size, array_stream=self._array_stream,
                                shuffle=self._shuffle, transformer=transformer, seed=seed)

    def get_new_train_iterator(self, transformer: LIDCGeneratorTransform, seed=None):
        return self._get_iterator(transformer, self._train_arrnames, seed=seed)

    def get_new_validation_iterator(self, transformer: LIDCGeneratorTransform, seed=None):
        return self._get_iterator(transformer, self._valid_arrnames, seed=seed)

    def get_new_test_iterator(self, transformer: LIDCGeneratorTransform, seed=None):
        return self._get_iterator(transformer, self._test_arrnames, seed=seed)


class LIDCDataIterator(KerasImgIterator):
    def __init__(self, arrnames, batch_size, array_stream: ArrayStream,
                 shuffle: bool, transformer: LIDCGeneratorTransform, seed=None):
        self._arrnames = arrnames
        self._array_stream = array_stream
        self._transformer = transformer

        super().__init__(len(arrnames), batch_size, shuffle=shuffle, seed=seed)

    def _get_batches_of_transformed_samples(self, index_array):
        """Get images with indices in index array, transform them.
        """
        # self._arrnames[index_array] might just be enough
        relevant_arr_names = [self._arrnames[idx] for idx in index_array]
        good_rec_sino_pairs = [self._array_stream.load_arrays(name)
                              for name in relevant_arr_names]
        good_recs = np.stack([pair[0] for pair in good_rec_sino_pairs], axis=0)
        good_sinos = np.stack([pair[1] for pair in good_rec_sino_pairs], axis=0)

        return self._transformer.transform(reconstructions=good_recs, sinograms=good_sinos)
