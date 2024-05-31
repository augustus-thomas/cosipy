import copy
import numpy as np
import astropy.units as u
from tqdm.autonotebook import tqdm

from histpy import Histogram

from .deconvolution_algorithm_base import DeconvolutionAlgorithmBase

class RichardsonLucy(DeconvolutionAlgorithmBase):
    """
    A class for the RichardsonLucy algorithm. 
    The algorithm here is based on Knoedlseder+99, Knoedlseder+05, Siegert+20.
    """

    def __init__(self, initial_model_map, data, parameter):

        DeconvolutionAlgorithmBase.__init__(self, initial_model_map, data, parameter)

        self.loglikelihood = None

        self.do_acceleration = parameter.get('acceleration', False)

        self.alpha_max = parameter.get('alpha_max', 1.0)

        self.do_response_weighting = parameter.get('response_weighting', False)

        self.do_smoothing = parameter.get('smoothing', False)

        self.do_bkg_norm_fitting = parameter.get('background_normalization_fitting', False)

        if self.do_bkg_norm_fitting:
            self.bkg_norm_range = parameter.get('background_normalization_range', [0.5, 1.5])

        print("... calculating the expected events with the initial model map ...")
        self.expectation = self.calc_expectation(self.initial_model_map, self.data)

        if self.do_response_weighting:
            print("... calculating the response weighting filter...")

            response_weighting_index = parameter.get('response_weighting_index', 0.5)

            self.response_weighting_filter = data.image_response_dense_projected.contents / np.max(data.image_response_dense_projected.contents) 

            self.response_weighting_filter = self.response_weighting_filter**response_weighting_index

        if self.do_smoothing:
            self.smoothing_fwhm = parameter['smoothing_FWHM'] * u.deg
            print(f"... We will apply the Gaussian filter with FWHM of {self.smoothing_fwhm} to delta images ...")

    def pre_processing(self):
        pass

    def Estep(self):
        """
        Notes
        -----
        Expect count histogram is calculated in the post processing.
        """
        print("... skip E-step ...")

    def Mstep(self):
        """
        M-step in RL algorithm.

        Notes
        -----
        Background normalization is also optimized based on a generalized RL algirithm.
        Currenly we use a signle normalization parameter. 
        In the future, the normalization will be optimized for each background group defined in some file.
        """
        # Currenly (2024-01-12) this method can work for both local coordinate CDS and in galactic coordinate CDS.
        # This is just because in DC2 the rotate response for galactic coordinate CDS does not have an axis for time/scatt binning.
        # However it is likely that it will have such an axis in the future in order to consider background variability depending on time and pointign direction etc.
        # Then, the implementation here will not work. Thus, keep in mind that we need to modify it once the response format is fixed.

        diff = self.data.event_dense / self.expectation - 1
        
        # part1
        delta_map_part1 = self.model_map / self.data.image_response_dense_projected

        # part2
        delta_map_part2 = Histogram(self.model_map.axes, unit = self.data.image_response_dense_projected.unit)

        diff_x_response = np.tensordot(diff.contents, self.data.image_response_dense.contents, axes = ([1,2,3], [2,3,4])) 
            # [Time/ScAtt, Em, Phi, PsiChi] x [NuLambda, Ei, Em, Phi, PsiChi] -> [Time/ScAtt, NuLambda, Ei]

        delta_map_part2[:] = np.tensordot(self.data.coordsys_conv_matrix.contents, diff_x_response, axes = ([0,2], [0,1])) \
                             * diff_x_response.unit * self.data.coordsys_conv_matrix.unit
            # [Time/ScAtt, lb, NuLambda] x [Time/ScAtt, NuLambda, Ei] -> [lb, Ei]
            # note that coordsys_conv_matrix is the sparse, so the unit should be recovered.
        
        # delta map
        self.delta_map = delta_map_part1 * delta_map_part2

        # mask for zero-exposure pixels
        if self.data.mask is not None:
            self.delta_map.mask_pixels(self.data.mask, 0)

        if self.do_bkg_norm_fitting:
            self.bkg_norm += self.bkg_norm * np.sum(diff * self.data.bkg_dense) / np.sum(self.data.bkg_dense)

            if self.bkg_norm < self.bkg_norm_range[0]:
                self.bkg_norm = self.bkg_norm_range[0]
            elif self.bkg_norm > self.bkg_norm_range[1]:
                self.bkg_norm = self.bkg_norm_range[1]

    def post_processing(self):
        """
        Here three processes will be performed.
        - response weighting filter: the delta map is renormalized as pixels with large exposure times will have more feedback.
        - gaussian smoothing filter: the delta map is blurred with a Gaussian function.
        - acceleration of RL algirithm: the normalization of delta map is increased as long as the updated image has no non-negative components.
        """

        if self.do_response_weighting:
            self.delta_map[:,:] *= self.response_weighting_filter

        if self.do_smoothing:
            self.delta_map = self.delta_map.smoothing(fwhm = self.smoothing_fwhm)
        
        if self.do_acceleration:
            self.alpha = self.calc_alpha(self.delta_map, self.model_map)
        else:
            self.alpha = 1.0

        model_map_new = self.model_map + self.delta_map * self.alpha
        model_map_new[:] = np.where(model_map_new.contents < self.minimum_flux * model_map_new.unit, 
                                    self.minimum_flux * model_map_new.unit, model_map_new.contents)

        self.processed_delta_map = model_map_new - self.model_map
        self.model_map = model_map_new

        print("... calculating the expected events with the updated model map ...")
        self.expectation = self.calc_expectation(self.model_map, self.data)

    def check_stopping_criteria(self, i_iteration):
        """
        If i_iteration is smaller than iteration_max, the iterative process will continue.

        Returns
        -------
        bool
        """
        if i_iteration < self.iteration_max:
            return False
        return True

    def register_result(self, i_iteration):
        """
        The values below are stored at the end of each iteration.
        - iteration: iteration number
        - model_map: updated image
        - delta_map: delta map after M-step 
        - processed_delta_map: delta map after post-processing
        - alpha: acceleration parameter in RL algirithm
        - background_normalization: optimized background normalization
        - loglikelihood: log-likelihood
        """
        loglikelihood = self.calc_loglikelihood(self.data, self.model_map, self.expectation)

        this_result = {"iteration": i_iteration, 
                       "model_map": copy.deepcopy(self.model_map), 
                       "delta_map": copy.deepcopy(self.delta_map),
                       "processed_delta_map": copy.copy(self.processed_delta_map),
                       "alpha": self.alpha, 
                       "background_normalization": self.bkg_norm,
                       "loglikelihood": loglikelihood}

        self.result = this_result

    def save_result(self, i_iteration):
        self.result["model_map"].write(f"model_map_itr{i_iteration}.hdf5", overwrite = True)
        self.result["delta_map"].write(f"delta_map_itr{i_iteration}.hdf5", overwrite = True)
        self.result["processed_delta_map"].write(f"processed_delta_map_itr{i_iteration}.hdf5", overwrite = True)

        with open(f"result_itr{i_iteration}.dat", "w") as f:
            f.write(f'alpha: {self.result["alpha"]}\n')
            f.write(f'loglikelihood: {self.result["loglikelihood"]}\n')
            f.write(f'background_normalization: {self.result["background_normalization"]}\n')

    def show_result(self, i_iteration):
        print(f'    alpha: {self.result["alpha"]}')
        print(f'    loglikelihood: {self.result["loglikelihood"]}')
        print(f'    background_normalization: {self.result["background_normalization"]}')

    def calc_alpha(self, delta, model_map):
        """
        Calculate the acceleration parameter in RL algorithm.

        Returns
        -------
        float
            Acceleration parameter
        """
        diff = -1 * (model_map / delta).contents

        diff[(diff <= 0) | (delta.contents == 0)] = np.inf

        if self.data.mask is not None:
            diff[np.invert(self.data.mask.contents)] = np.inf

        alpha = min(np.min(diff), self.alpha_max)

        if alpha < 1.0:
            alpha = 1.0

        return alpha
