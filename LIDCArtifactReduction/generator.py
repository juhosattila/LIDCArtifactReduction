import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import iradon

from tensorflow.keras.preprocessing.image import Iterator as KerasImgIterator

from LIDCArtifactReduction import parameters, utility
from LIDCArtifactReduction.array_streams import RecSinoArrayStream
from LIDCArtifactReduction.neural_nets.DCAR_TrainingNetwork import DCAR_TrainingNetwork


class LIDCDataGenerator:
    """Custom generator class for LIDC data.

    Data is assumed to be found in a directory passed to init.
    Capable of shuffling data, augmenting it with possible Poisson noise.
    It can provide iterator objects for training, validation and testing.
    """
    def __init__(self, validation_split=0.1, test_split=0.1, batch_size=16,
                 array_stream: RecSinoArrayStream = RecSinoArrayStream(), mode='patient',
                 shuffle : bool = True, add_noise_train : bool = True,
                 verbose : bool = False, test_mode : bool = False):
        """
        :param mode: should be either 'patient' or 'combined'.
        """
        self._batch_size = batch_size
        self._array_stream = array_stream
        self._shuffle = shuffle
        self._add_noise_train = add_noise_train
        self._test_mode = test_mode

        validation_test_split = validation_split + test_split

        if mode == 'combined':
            raise NotImplementedError()
            # arrnames = self._array_stream.get_names_with_dir()
            # train_arrnames, valid_test_arrnames = train_test_split(arrnames,
            #     train_size=1.0-validation_test_split, shuffle=True)
            # valid_arrnames, test_arrnames = train_test_split(valid_test_arrnames,
            #     train_size=validation_split / validation_test_split, shuffle=True)

        else:  # mode == 'patient'
            patient_dirs = self._array_stream.get_directory_names()

            train_directories, valid_test_directories = train_test_split(patient_dirs,
                train_size=1.0-validation_test_split, shuffle=self._shuffle)
            valid_directories, test_directories = train_test_split(valid_test_directories,
                train_size=validation_split / validation_test_split, shuffle=self._shuffle)

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

        self._train_iterator = self._get_iterator(train_arrnames, add_noise=self._add_noise_train)
        self._valid_iterator = self._get_iterator(valid_arrnames, add_noise=False)
        self._test_iterator = self._get_iterator(test_arrnames, add_noise=False)

    def _get_iterator(self, arrnames, add_noise):
        return LIDCDataIterator(arrnames, self._batch_size, add_noise, self._array_stream,
                                self._shuffle, self._test_mode)

    @property
    def train_iterator(self):
        return self._train_iterator

    @property
    def valid_iterator(self):
        return self._valid_iterator

    @property
    def test_iterator(self):
        return self._test_iterator


class LIDCDataIterator(KerasImgIterator):
    def __init__(self, arrnames, batch_size, add_noise: bool, array_stream: RecSinoArrayStream,
                 shuffle: bool, test_mode : bool):
        self._arrnames = arrnames
        self._add_noise = add_noise
        self._array_stream = array_stream
        self._test_mode = test_mode

        # Testing
        self.analysed = False

        super().__init__(len(arrnames), batch_size, shuffle=shuffle, seed=None)

    def _get_batches_of_transformed_samples(self, index_array):
        """Get images with indices in index array, transform them.
        """
        relevant_arr_names = [self._arrnames[idx] for idx in index_array]
        good_rec_sino_pairs = [self._array_stream.load_arrays(name)
                              for name in relevant_arr_names]
        good_recs = [pair[0] for pair in good_rec_sino_pairs]
        good_sinos = [pair[1] for pair in good_rec_sino_pairs]

        if self._test_mode:
            all = [self._transform(good_sino) for good_sino in good_sinos]
            bad_recs = [a[0] for a in all]
            bad_sinos = [a[1] for a in all]
            return bad_recs, [good_recs, good_sinos], bad_sinos

        bad_recs = [self._transform(good_sino) for good_sino in good_sinos]
        return self._output_format_manager(bad_recs, good_recs, good_sinos)

    def _output_format_manager(self, bad_recs, good_recs, good_sinos):
        return ({DCAR_TrainingNetwork.input_name : bad_recs},
                {DCAR_TrainingNetwork.reconstruction_output_name : good_recs,
                 DCAR_TrainingNetwork.sino_output_name : good_sinos})

    def _transform(self, sino):
        # cut 3rd channel dimension, which has C=1
        sino2D = np.reshape(sino, newshape=np.shape(sino)[:2])

        if self._add_noise:
            sino2D = self._generate_noise(sino2D)

        bad_rec = self._inverted_radon(sino2D)
        # add back channel dimension
        bad_rec = np.expand_dims(bad_rec, axis=-1)

        if self._test_mode:
            return bad_rec, sino2D

        return bad_rec

    def _inverted_radon(self, sino):
        res = np.transpose(sino)
        res = iradon(res, theta=np.linspace(0., 180., parameters.NR_OF_SPARSE_ANGLES),
                     output_size=parameters.IMG_SIDE_LENGTH, circle=True)
        return res

    def _generate_noise(self, sino):
        """See documentation for explananation."""

        # For scaling 1000HU to 1CT, we have (relevant sinomax) ~ IMG_SIDE_LENGTH
        pmax = parameters.IMG_SIDE_LENGTH


        # base_intensity in logarithm. This definies sum scaling.
        # These should changed together.
        lnI0 = 10 * np.log(5)
        sum_scaling = 5.0

        # further scale parameter
        scale = parameters.HU_TO_CT_SCALING / 1000.0 * parameters.IMG_SIDE_LENGTH / sum_scaling

        # scaling of noise deviation parameter
        alfa = 0.3


        # deviation of noise
        sigma_I0 = alfa * np.exp(lnI0 - 1.0 / scale * pmax)

        I_normal_noise = np.random.normal(loc=0.0, scale=sigma_I0, size=np.shape(sino))
        lnI = lnI0 - 1.0 / scale * sino
        I_no_noise = np.exp(lnI)
        I_added_noise = I_no_noise + I_normal_noise

        # some elements might be too low
        too_low = I_added_noise < I_no_noise / 2.0
        I_added_noise[too_low] = I_no_noise[too_low]

        I_Poisson = np.random.poisson(I_added_noise)

        lnI_added_noise_and_Poisson = np.log(I_Poisson)
        sino_added_noise = scale * (lnI0 - lnI_added_noise_and_Poisson)

        # Testing
        if self._test_mode and not self.analysed:
            self.analysed = True
            utility.analyse(I_no_noise, "I no noise")
            utility.analyse(I_added_noise, "I normal noise")
            utility.analyse(I_Poisson, "I Poisson")

        return sino_added_noise
