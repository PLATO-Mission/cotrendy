"""
Cotrending Basis Vectors compoents for Cotrendy
"""
# standard imports
import gc
import sys
import traceback
import logging
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
from scipy.linalg import svd
import scipy.optimize as optimization
from scipy.stats import median_absolute_deviation
# imports for entropy from TASOC
import bottleneck as bn
from scipy.special import xlogy
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from sklearn.decomposition import PCA
# import for plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# my own imports
import jastro.lightcurves as jlc
from cotrendy.map import MAP
from cotrendy.utils import picklify, depicklify

# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: disable=logging-fstring-interpolation

# TODO: move making theta to inside MAP.
#       make it encompass both cond and prior vals so it doesn't fail!

class CBVs():
    """
    A class to generate a series of cotrending basis vectors from
    an array of light curves. This class is based on the Kepler PDC
    method of cotrending

    https://iopscience.iop.org/article/10.1086/667697/pdf

    Take in an array of light curves one object on each row is more intutive
    however SVD needs each light curve to be a column in the matrix. We will
    handle that conversion here silently

    Steps to generate CBVs:
       1. Estimate the variability in each light curve
          1. This is done using a 3rd order polynomial
          1. The variability V = sigma_y / (delta_y * Vbar)
          1. Where sigma_y is the polynomial corrected RMS, delta_y is the
             uncertainty of the flux data and Vbar is the median variability
             over all light curves in the sample
          1. They ignore anything with normalised variability > cotrend.normalised_variability_limit
          1. Anything < 0.5 is considered very quiet
       1. After the above cut, the remaining stars are pearson correlated and
          the 50% most correlated are used for generating the CBV via SVD
       1. The light curves for SVD are dithered slightly by 0 mean Gaussian to
          prevent an artificial node occuring at 0
       1. Once the CBVs are obtained several checks are performed
          1. A SNR check is made for each CBV, SNR(dB) = 10*log10(Asig^2/Anoise^2)
             1. Asig is the RMS of the light curve
             1. Anoise is the noise floor and is the first differences between adjacent fluxes
             1. Any CBVs < 5dB are removed
          1. An entropy check is made to eliminate domineering stars influencing the CBVs
             1. Entropy is calculated using Eqs 13-16 in the Smith paper
                1. They find anything with an entropy less than -0.7 was bad
                1. Bad targets are removed and the SVD is done again until all are better than 0.7
    """
    def __init__(self, config, timesteps, targets):
        """
        Take a list of target objects
        and prepare them for CBV generation

        Parameters
        ----------
        config : object
            configuration for simulation
        timesteps : array-like
            array of partial bjd times for a given dataset
        targets : Lightcurves object
            collection of light curves for a given camera

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.camera_id = config['global']['camera_id']
        self.direc = config['global']['root']
        self.phot_file = config['data']['flux_file'].split('.pkl')[0]
        self.timesteps = timesteps
        self.targets = targets
        self.lc_idx = np.arange(0, len(targets))

        # multiprocessing pool size
        self.pool_size = config['cotrend']['pool_size']

        # enable debugging mode for alllll the plots
        self.debug = config['global']['debug']

        # shortlist of stars to analyse more closely
        self.test_stars = config['cotrend']['test_stars']

        # placeholder for normalise variability stats
        self.normalised_variability_limit = config['cotrend']['normalised_variability_limit']
        self.prior_normalised_variability_limit = config['cotrend']['prior_normalised_variability_limit']
        self.variability_normalisation_order = config['cotrend']['variability_normalisation_order']

        # the max number to look for
        self.max_n_cbvs = config['cotrend']['max_n_cbvs']
        # the actual number extracted
        self.n_cbvs = 0

        # limiting SNR for useful CBVs
        self.cbvs_snr_limit = config['cotrend']['cbv_snr_limit'] # dB

        # check if we want to load externally calculated variability
        # or do it on the fly. If it's empty we work it out, otherwise
        # we just use the array from the pickle file
        if config['data']['variability_file']:
            self.variability = depicklify(config['data']['variability_file'])
        else:
            self.variability = None

        # get the correlation threshold. the top fraction of stars for the CBVs
        self.correlation_threshold = config['cotrend']['correlation_threshold']

        # get the threshold for entropy cleaning
        self.entropy_threshold = config['cotrend']['entropy_threshold']
        # get the limit on number of stars being rejected by entropy cleaning
        max_entropy_rejections = config['cotrend']['max_entropy_rejections']
        # if there is no limit on number of rejections, set it to 1 > size of lc_idx array
        if max_entropy_rejections < 0:
            self.max_entropy_rejections = len(self.lc_idx) + 1
        else:
            self.max_entropy_rejections = config['cotrend']['max_entropy_rejections']

        # placeholders for variable star ids
        self.non_variable_star_idx = []
        self.variable_star_idx = []

        # are we working in pixels or degrees for RA/Dec?
        # important for the fit coeff correlations plots
        self.coords_units = config['catalog']['coords_units']

        # params for generating CBVs
        self.median_abs_correlations = []
        self.highly_correlated_star_idx = []
        self.lowly_correlated_star_idx = []

        # placeholder for the dithered array used in SVD
        self.norm_flux_array_for_cbvs_dithered = None

        # initial CBV / SVD parameters, pre entropy cleaning
        self.U0, self.s0, self.VT0 = None, None, None
        self.cbvs0 = defaultdict(list)
        # storage for the CBV SNRs
        self.cbvs0_snr = {}

        # final CBV / SVD params, post entropy cleaning
        self.U, self.s, self.VT = None, None, None
        self.cbvs = defaultdict(list)
        # storage for the CBV SNRs
        self.cbvs_snr = {}

        # keep track of rejected stars
        self.entropy_rejected_idx = []

        # this is a numpy array that's easier to broadcast on
        self.vect_store = None

        # CBV fit coefficients
        self.fit_coeffs = defaultdict(list)

        # keep the theta grids for the PDFs in MAP
        self.theta = defaultdict(np.array)

        # store the MAP execution times for analysis of mp
        self.map_exec_times = None

        # keep placeholder for cotrending and cotrended arrays
        self.cotrending_flux_array = None
        self.cotrended_flux_array = None

        # loop over the targets and make the flux array for working on
        # the light curve normalisation is done at this point
        norm_flux_array = []
        norm_fluxerr_array = []

        for lc in targets:
            # divide by the median to normalise to 1
            # subtract the median, then divide by the median to normalise to 0
            # kepler appears to do the latter
            norm_flux_array.append((lc.flux_wtrend-lc.median_flux)/lc.median_flux)
            # propagate the errors for use in the LS fit later
            norm_fluxerr_array.append(lc.fluxerr_wtrend/lc.median_flux)
        # make it into a numpy array
        self.norm_flux_array = np.array(norm_flux_array)
        self.norm_fluxerr_array = np.array(norm_fluxerr_array)

        # initialise some storage for the CBVs stars
        # these arrays will be whittled down during CBV star selection
        self.norm_flux_array_for_cbvs = np.copy(self.norm_flux_array)
        self.lc_idx_for_cbvs = np.copy(self.lc_idx)

        # here were take in some parameters which are passed to MAP if needed
        # TODO: remove the try after these parameters are standard in the config files
        try:
            self.prior_cond_snapping = config['cotrend']['prior_cond_snapping']
            self.prior_raw_goodness_weight = config['cotrend']['prior_raw_goodness_weight']
            self.prior_raw_goodness_exponent = config['cotrend']['prior_raw_goodness_exponent']
            self.prior_noise_goodness_weight = config['cotrend']['prior_noise_goodness_weight']
            self.prior_pdf_variability_weight = config['cotrend']['prior_pdf_variability_weight']
            self.prior_pdf_goodness_gain = config['cotrend']['prior_pdf_goodness_gain']
            self.prior_pdf_goodness_weight = config['cotrend']['prior_pdf_goodness_weight']
        except KeyError:
            self.prior_cond_snapping = False
            self.prior_raw_goodness_weight = 5.0
            self.prior_raw_goodness_exponent = 3.0
            self.prior_noise_goodness_weight = 0.002
            self.prior_pdf_variability_weight = 2.0
            self.prior_pdf_goodness_gain = 1.0
            self.prior_pdf_goodness_weight = 0.5

    def calculate_normalised_variability(self):
        """
        First apply a filter to remove variable stars

        NOTE: This may be part of why PDC struggled with
        long term variability

        V = sigma_y / (delta_y * Vbar)

        Loop over all the stars, subtract a 3rd order polynomial
        Work out the RMS
        """
        logging.info("Finding variable stars...")

        if self.variability is None:
            logging.info("Calculating normalised variability...")
            sigma_y, delta_y = [], []
            for target in self.targets:
                delta_y.append(np.average(target.fluxerr_wtrend))
                coeffs = np.polyfit(self.timesteps, target.flux_wtrend, self.variability_normalisation_order)
                besty = np.polyval(coeffs, self.timesteps)
                # this was original flux_wtrend - besty, which seemed wrong
                # as all the stars for CBVs were chosen as the fainest ones.
                flattened = target.flux_wtrend - besty
                sigma_y.append(np.std(flattened))
            sigma_y = np.array(sigma_y)
            delta_y = np.array(delta_y)

            # calculate the normalised variability
            V = sigma_y / delta_y
            median_V = np.median(V)
            self.variability = V / median_V

        try:
            # set log V for plotting
            logV = np.log10(self.variability)

            fig, ax = plt.subplots(1)
            ax.hist(logV, bins=100)
            ax.axvline(np.log10(self.normalised_variability_limit), ls='--', color='red')
            ax.set_xlabel('log(Normalised variability)')
            ax.set_ylabel('Number')
            fig.tight_layout()
            fig.savefig(f"{self.direc}/variability_camera_{self.camera_id}_{self.phot_file}.pdf")
            fig.clf()
            plt.close()
            gc.collect()
        except ValueError:
            logging.error('Catching odd error with making variability histogram. Skipping...')

    def calculate_pearson_correlation(self):
        """
        There is a much faster way to get the median absolute correlation

        It is explained nicely here:
        https://towardsdatascience.com/x%E1%B5%80x-covariance-correlation-and-cosine-matrices-d2230997fb7
        And can also be seen in the Kepler pipeline here:
        https://github.com/nasa/kepler-pipeline/blob/f58b21df2c82969d8bd3e26a269bd7f5b9a770e1/source-code/matlab/pdc/mfiles/pdc_compute_correlation.m#L63

        By taking an array of normalised fluxes (scaling them by their per star RMS)
        and then multiplying the array by its transpose and dividing by the
        number of observations, we get a correlation maxtrix equivilant to the pearson
        correlation.

        Note: As of cotrendy version 0.1.0.dev this function calculates correlations
        for the stars that make the variability cut only.
        """
        n_cadences = len(self.norm_flux_array_for_cbvs[0])
        n_targets = len(self.norm_flux_array_for_cbvs)
        per_object_rms = np.std(self.norm_flux_array_for_cbvs, axis=1).reshape(n_targets, -1)
        unit_norm_flux = self.norm_flux_array_for_cbvs / per_object_rms
        correlation_matrix = (unit_norm_flux @ unit_norm_flux.T) / n_cadences
        self.median_abs_correlations = np.median(np.abs(correlation_matrix), axis=0)

    def _apply_zero_mean_dither(self):
        """
        In order for avoid an artifical node where the lcs
        cross zero we dither them slightly before making the
        CBVs. This is only done when making the CBVs, not
        when analysing the actual light curves
        """
        logging.info("Applying zero mean dither...")
        n_light_curves = len(self.norm_flux_array_for_cbvs)
        dither = np.random.normal(loc=0.0, scale=0.001, size=n_light_curves)
        self.norm_flux_array_for_cbvs_dithered = self.norm_flux_array_for_cbvs + dither.reshape(n_light_curves, 1)

    def calculate_svd(self, initial=False):
        """
        Take the previously filtered flux array and calculate the SVD

        We apply a zero mean dither before hand

        Initial flag allows us to store results in arrays with 0 suffix, indicating initial pass.
        .
        This allows a pre and post entropy cleaning check on the CBVs
        """
        # now we take the SVD of the highest correlated light curves
        # NOTE fluxes need to be in columns in order for the left-singular maxtrix (U)
        # to return us the CBVs in order
        if initial:
            logging.info("Calculating the initial SVD...")
            self.U0, self.s0, self.VT0 = svd(self.norm_flux_array_for_cbvs_dithered.T)
            output_filename = f"{self.direc}/singular_values_{self.camera_id}_{self.phot_file}_initial.pdf"
            sing_vals = self.s0
            logging.info(f"Matrix shapes -  U: {self.U0.shape}, s: {self.s0.shape}, VT: {self.VT0.shape}")
        else:
            logging.info("Calculating the final SVD...")
            self.U, self.s, self.VT = svd(self.norm_flux_array_for_cbvs_dithered.T)
            output_filename = f"{self.direc}/singular_values_{self.camera_id}_{self.phot_file}.pdf"
            sing_vals = self.s
            logging.info(f"Matrix shapes -  U: {self.U.shape}, s: {self.s.shape}, VT: {self.VT.shape}")

        # plot the first ~50 singular values against their index
        fig, ax = plt.subplots(1, figsize=(5, 5))

        # quickly check can we plot 50 singlular values? If <50, plot them all
        if initial:
            n_vecs = self.U0.shape[0]
        else:
            n_vecs = self.U.shape[0]

        # check can we do 50 or not?
        if n_vecs >= 50:
            n_plot = 50
        else:
            n_plot = n_vecs

        inds = np.arange(1, n_plot+1)
        ax.loglog(inds, sing_vals[:n_plot], 'ko')
        ax.loglog(inds, sing_vals[:n_plot], 'k-')
        ax.set_xlabel('Singular value index [1 ind]')
        ax.set_ylabel('Singular vale')
        fig.tight_layout()
        fig.savefig(output_filename)
        fig.clf()
        plt.close()
        gc.collect()

    def calculate_cbv_snr(self, initial=False):
        """
        Kepler implements as check on the SNR for each
        CBV. If the SNR (dB) is < 5dB, they are removed.

        They typically use 8 and only some CBVs across
        the entire FOV were ever removed
        """
        logging.info(f"Calculating {self.max_n_cbvs} CBV SNRs...")

        if initial:
            cbvs = self.cbvs0
        else:
            cbvs = self.cbvs

        for i in sorted(cbvs):
            Anoise = np.std(np.diff(cbvs[i]))
            Asignal = np.std(cbvs[i])
            snr = 10*np.log10((Asignal**2) / (Anoise**2))
            if initial:
                self.cbvs0_snr[i] = snr
            else:
                self.cbvs_snr[i] = snr
            logging.info(f"CBV {i} SNR: {snr:.3f} dB")

    @staticmethod
    def _compute_entropy(U):
        """
        Taken from TASOC code

        Compute the entropy of each CBV compared to a Guassian
        """
        HGauss0 = 0.5 + 0.5*np.log(2*np.pi)
        nSingVals = U.shape[1]
        H = np.empty(nSingVals, dtype='float64')

        for iBasisVector in range(nSingVals):
            kde = KDE(np.abs(U[:, iBasisVector]))
            kde.fit(gridsize=1000)

            pdf = kde.density
            x = kde.support

            dx = x[1]-x[0]

            # Calculate the Gaussian entropy
            pdfMean = bn.nansum(x * pdf)*dx
            with np.errstate(invalid='ignore'):
                sigma = np.sqrt(bn.nansum(((x-pdfMean)**2) * pdf) * dx)
            HGauss = HGauss0 + np.log(sigma)

            # Calculate vMatrix entropy
            pdf_pos = (pdf > 0)
            HVMatrix = -np.sum(xlogy(pdf[pdf_pos], pdf[pdf_pos])) * dx

            # Returned entropy is difference between V-Matrix entropy and Gaussian entropy of similar width (sigma)
            H[iBasisVector] = HVMatrix - HGauss

        return H

    def entropy_cleaning(self, ncomponents, random_state=999):
        """
        Entropy-cleaning of lightcurve matrix using the SVD U-matrix.
        Identify stars which are increading the entropy of the CBVs, remove them
        and repeat until there is no overrepresentation (typically entropy < -0.7)

        This is taken directly from the TASOC pipeline
        """
        # conversion constant from MAD to Sigma. Constant is 1/norm.ppf(3/4)
        mad_to_sigma = 1.482602218505602
        # added to track those rejected - jmcc
        rejected_idx = []

        # Calculate the principle components:
        pca = PCA(ncomponents, random_state=random_state)
        U, _, _ = pca._fit(self.norm_flux_array_for_cbvs_dithered)

        ent = self._compute_entropy(U)
        logging.info(f"Entropy start: {ent}")

        targets_removed = 0
        components = np.arange(ncomponents)

        with np.errstate(invalid='ignore'):
            while np.any(ent < self.entropy_threshold):
                com = components[ent < self.entropy_threshold][0]

                # Remove highest relative weight target
                m = bn.nanmedian(U[:, com])
                s = mad_to_sigma*bn.nanmedian(np.abs(U[:, com] - m))
                dev = np.abs(U[:, com] - m) / s

                idx0 = np.argmax(dev)
                logging.info(f"{len(dev)}, {idx0}, {len(self.lc_idx_for_cbvs)}")

                # store the id from the original matrix, so we can see which were rejected
                rejected_idx.append(self.lc_idx_for_cbvs[idx0])
                # then remove that element from the list of original IDs so the list length
                # follows that of the matrix.
                self.lc_idx_for_cbvs = np.delete(self.lc_idx_for_cbvs, idx0)

                # Remove the star from the lightcurve matrix:
                star_no = np.ones(U.shape[0], dtype=bool)
                star_no[idx0] = False
                # remove the star from both the dithered and undithered arrays
                self.norm_flux_array_for_cbvs = self.norm_flux_array_for_cbvs[star_no, :]
                self.norm_flux_array_for_cbvs_dithered = self.norm_flux_array_for_cbvs_dithered[star_no, :]

                targets_removed += 1
                if targets_removed >= self.max_entropy_rejections:
                    logging.warning(f"Entropy cleaning rejection limit {self.max_entropy_rejections} reached, breaking")
                    break
                elif len(self.norm_flux_array_for_cbvs_dithered) == 0:
                    logging.critical("Entropy cleaning has removed all stars, quitting!")
                    sys.exit(1)

                U, _, _ = pca._fit(self.norm_flux_array_for_cbvs_dithered)
                ent = self._compute_entropy(U)

        logging.info(f"Entropy end: {ent}")
        logging.info(f"Targets removed: {targets_removed}")
        self.entropy_rejected_idx = rejected_idx

    def calculate_cbvs(self):
        """
        Generate the CBVs from the input array normalised fluxes

        1. Calculate normalised variability, remove those above variability limit
        2. Calculate the median absolute correlation for surviving objects, keep top N%
        3. Dither surviving light curves
        4. Do an initial SVD, check the SNR of each CBV, remove those < SNR limit
        5. Check the entropy of each CBV, remove any stars that are over representend
        6. Iterate SVD, entropy rejection until CBVs have entropy < entropy limit
        7. Store the final vectors for use and make some summary plots of singular values etc
        """
        logging.info("Calculating CBVs...")

        # When we start we have the two _for_cbvs arrays, which are
        # copies of the original full lc array and their indexes
        # we want to work on those arrays and whittle them down

        # step 1, calculate the normalised variability
        self.calculate_normalised_variability()
        # make a cut of the _for_cbvs flux array, removing anything >= Var lim
        non_var_loc = np.where(self.variability < self.normalised_variability_limit)[0]
        self.non_variable_star_idx = non_var_loc
        # for completeness, lets keep the ids of the variable objects too
        var_loc = np.where(self.variability >= self.normalised_variability_limit)[0]
        self.variable_star_idx = var_loc

        self.norm_flux_array_for_cbvs = self.norm_flux_array_for_cbvs[non_var_loc]
        self.lc_idx_for_cbvs = self.lc_idx_for_cbvs[non_var_loc]

        # the surviving, non-variable stars then go into the correlation stage
        self.calculate_pearson_correlation()
        # now sort the objects by their correlations and keep the top 50%
        temp = zip(self.median_abs_correlations, self.lc_idx_for_cbvs)
        temp = sorted(temp, reverse=True)

        median_abs_correlations_sorted, lc_idx_for_cbvs_sorted = map(np.array, zip(*temp))
        # find the position that corresponds to the limiting correlation
        loc_correlation_limit = int(len(median_abs_correlations_sorted)*self.correlation_threshold)
        correlation_limit = median_abs_correlations_sorted[loc_correlation_limit]
        # store the ids of the highly correlated stars
        self.highly_correlated_star_idx = lc_idx_for_cbvs_sorted[:loc_correlation_limit]
        # keep the ids of the lowly correlated stars for completeness
        self.lowly_correlated_star_idx = lc_idx_for_cbvs_sorted[loc_correlation_limit:]
        # keep all the stars with correlations > the correlation limit
        cor_loc = np.where(self.median_abs_correlations >= correlation_limit)[0]
        # next whittle the _for_cbvs arrays down to only those highly correlated stars
        self.norm_flux_array_for_cbvs = self.norm_flux_array_for_cbvs[cor_loc]
        self.lc_idx_for_cbvs = self.lc_idx_for_cbvs[cor_loc]

        # we need to apply a small dither to the CBV lcs to avoid a node
        # where the normalised flux crosses zero
        self._apply_zero_mean_dither()

        # now calculate the initial svd, pre entropy cleaning
        self.calculate_svd(initial=True)
        for i in range(self.max_n_cbvs):
            self.cbvs0[i] = self.U0[:, i]

        # check the initial CBVs SNRs
        self.calculate_cbv_snr(initial=True)

        # It seems to make sense that we only do an entropy check for the
        # highest SNR CBVs, so let's add an initial check here to determine how
        # many components to entropy check
        #n_components_to_entropy_check = self.max_n_cbvs

        n_components_to_entropy_check = 0
        for i in sorted(self.cbvs0):
            if i == 0 or self.cbvs0_snr[i] >= self.cbvs_snr_limit:
                n_components_to_entropy_check += 1

        # do the entropy cleaning
        self.entropy_cleaning(n_components_to_entropy_check)

        # now calculate the final svd, post entropy cleaning
        self.calculate_svd(initial=False)
        for i in range(self.max_n_cbvs):
            self.cbvs[i] = self.U[:, i]

        # check the initial CBVs SNRs
        self.calculate_cbv_snr(initial=False)

        # check the final CBV SNRs and remove (or not) any
        to_delete = []
        for i in sorted(self.cbvs):
            if i != 0 and self.cbvs_snr[i] < self.cbvs_snr_limit:
                logging.info(f"Low SNR CBV {i}, deleting!")
                to_delete.append(i)

        # do the actual deletion
        for i in to_delete:
            del self.cbvs[i]

        # get the final number of high SNR CBVs
        self.n_cbvs = len(self.cbvs)

        # make a convenient numpy array of the vectors for using later
        # grab the vectors from their dictionary
        vect_store = []
        for c in sorted(self.cbvs):
            vect_store.append(self.cbvs[c])
        # make them into a numpy array
        self.vect_store = np.array(vect_store)

        # TODO: calculate the CBV effectiveness scores as per TASOC
        # plot the CBVs for everyone to see
        output_filename = f"{self.direc}/cbvs_{self.camera_id}_{self.phot_file}.pdf"
        fig, ax = plt.subplots(self.max_n_cbvs, figsize=(10, 10), sharex=True, sharey=True)

        # plot the initial CBVs, pre entropy cleaning
        for cbv_id in sorted(self.cbvs0.keys()):
            ax[cbv_id].plot(self.cbvs0[cbv_id], label=f'CBV0 {cbv_id}', color='orange')
            ax[cbv_id].legend(loc='upper right', fontsize='small')
        # plot the final CBVs post entropy cleaning
        for cbv_id in sorted(self.cbvs.keys()):
            ax[cbv_id].plot(self.cbvs[cbv_id], label=f'CBV {cbv_id}', color='blue')
            ax[cbv_id].legend(loc='upper right', fontsize='small')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.0)
        fig.savefig(output_filename)
        fig.clf()
        plt.close()
        gc.collect()

    @staticmethod
    def _fit_cbvs_to_data(x, y, vectors):
        """
        Similar to _fit_cbv_to_data
        but simultaneously does all coeffs

        Parameters
        ----------
        x : array
            list of fit coeffs
        y : array
            light curve array
        vectors : array
            list of co trending basis vectors

        Returns
        -------
        res : array
            residuals after subtracting the model
        """
        # x comes in as a numpy array of guess coeffs
        # we need to make it a column array to scale each vector
        xc = x.reshape(-1, 1)
        # scale them by the fit coeffs
        scaled_vectors = vectors * xc
        # sum the scaled cbvs to make the correction light curve
        model = np.sum(scaled_vectors, axis=0)
        # subtract the model from the data and return the residuals
        return y - model

    def calculate_robust_fit_coeffs_simult(self):
        """
        Calculate the fit coefficients, but instead of one by
        one in turn of their position in U, do them simultaneously
        """
        logging.info("Calculating CBVs fit coeffs...")

        # initial guess at fit coeffs
        x0 = [-0.1]*len(self.vect_store)
        cbv_ids = sorted(self.cbvs.keys())

        for i in range(0, len(self.norm_flux_array)):
            res = optimization.least_squares(self._fit_cbvs_to_data, x0,
                                             args=(self.norm_flux_array[i], self.vect_store),
                                             loss='soft_l1')
            xc = res['x']
            # store the best fitting coeffs for each CBV
            for coeff, cbv_id in zip(xc, cbv_ids):
                self.fit_coeffs[cbv_id].append(coeff)

        # now we have the full range of fit coeffs, we can make
        # theta which will be used in MAP for generating PDFs
        # do this for each CBV
        for j in cbv_ids:
            # remember to cast the fit_coeffs as an array for later use
            self.fit_coeffs[j] = np.array(self.fit_coeffs[j])
            theta_range = np.percentile(self.fit_coeffs[j], 98) - np.percentile(self.fit_coeffs[j], 2)
            theta_llim = np.percentile(self.fit_coeffs[j], 2) - theta_range
            theta_ulim = np.percentile(self.fit_coeffs[j], 98) + theta_range
            self.theta[j] = np.linspace(theta_llim, theta_ulim, 1000)

    @staticmethod
    def _fit_cbv_to_data(x, y, cbv):
        """
        Function to subtract CBV from data.
        Used when fitting coefficients with
        scipy.optimize.least_squares
        """
        return y - (x[0]* cbv)

    def calculate_robust_fit_coeffs_sequen(self):
        """
        Calculate a fit coefficient for fitting each CBV
        to each lc directly.

        If >1 CBV is found, the previous CBVs must be fitted and
        removed from the data before fitting the next one.
        """
        logging.info("Calculating CBVs fit coeffs...")

        # initial guess at fit coeff
        x0 = [-0.1]

        # keep a list of partially corrected lcs to use if > 1 CBV
        # CBVs are fitted and removed sequentially before the next is fit
        partial_correction_array = np.copy(self.norm_flux_array)

        # loop over the existing CBVs
        for j in sorted(self.cbvs):
            for i in range(0, len(partial_correction_array)):
                # we fit with the loss function = soft_l1, which gives a robust fit
                # wrt to outliers in the data
                res = optimization.least_squares(self._fit_cbv_to_data,
                                                 x0,
                                                 args=(partial_correction_array[i], self.cbvs[j]),
                                                 loss='soft_l1')
                # now take the result and store it, but use it now
                # to correct the data in the partial correction array.
                # This partially corrected data can be fit for the next CBV
                fc = res['x'][0]
                self.fit_coeffs[j].append(fc)
                partial_correction_array[i] -= self.cbvs[j]*fc
            # cast this as a lovely numpy array when done, lists suck
            self.fit_coeffs[j] = np.array(self.fit_coeffs[j])
            # now we have the full range of fit coeffs, we can make
            # theta which will be used in MAP for generating PDFs
            theta_range = np.percentile(self.fit_coeffs[j], 95) - np.percentile(self.fit_coeffs[j], 5)
            theta_llim = np.percentile(self.fit_coeffs[j], 5) - theta_range
            theta_ulim = np.percentile(self.fit_coeffs[j], 95) + theta_range
            self.theta[j] = np.linspace(theta_llim, theta_ulim, 2500)

    def plot_fit_coeff_correlations(self, catalog):
        """
        Take the fitted coeffs and the input catalog and plot
        the coeffs against ra, dec, mag etc to look for correlations
        for each basis vector
        """
        logging.info("Plotting fit coefficients correlations...")

        # determine binning scales for plots
        if self.coords_units == 'pix':
            coords_scale = 200 # pixels
        else:
            coords_scale = 0.5 # degrees

        with PdfPages(f"{self.direc}/fit_coeff_correlations_{self.camera_id}_{self.phot_file}.pdf") as pdf:
            # loop over ra, dec and mag for each CBV and plot the
            # coeffs for all stars but also for the CBV only stars
            for cbv_id in sorted(self.cbvs.keys()):

                fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

                # do some stats to limit the coeffs axis
                med = np.median(self.fit_coeffs[cbv_id])
                mad = median_absolute_deviation(self.fit_coeffs[cbv_id])
                llim = med - 5*mad
                ulim = med + 5*mad

                # plot some running averages also, sort by ra, dec, mag
                # then bin and plot over the data

                # RA running averages
                ra_cor = np.vstack((catalog.ra, self.fit_coeffs[cbv_id]))
                ra_cor_s = ra_cor[:, ra_cor[0].argsort()]
                ra_cor_m = np.vstack((catalog.ra[self.lc_idx_for_cbvs], self.fit_coeffs[cbv_id][self.lc_idx_for_cbvs]))
                ra_cor_m_s = ra_cor_m[:, ra_cor_m[0].argsort()]

                ra_sorted_bin, ra_coeff_sorted_bin, _ = jlc.pc_bin(ra_cor_s[0], ra_cor_s[1],
                                                                   ra_cor_s[1], coords_scale,
                                                                   mode="median")
                ra_mask_sorted_bin, ra_coeff_mask_sorted_bin, _ = jlc.pc_bin(ra_cor_m_s[0],
                                                                             ra_cor_m_s[1],
                                                                             ra_cor_m_s[1],
                                                                             coords_scale)

                # DEC running averages
                dec_cor = np.vstack((catalog.dec, self.fit_coeffs[cbv_id]))
                dec_cor_s = dec_cor[:, dec_cor[0].argsort()]
                dec_cor_m = np.vstack((catalog.dec[self.lc_idx_for_cbvs], self.fit_coeffs[cbv_id][self.lc_idx_for_cbvs]))
                dec_cor_m_s = dec_cor_m[:, dec_cor_m[0].argsort()]

                dec_sorted_bin, dec_coeff_sorted_bin, _ = jlc.pc_bin(dec_cor_s[0], dec_cor_s[1],
                                                                     dec_cor_s[1], coords_scale,
                                                                     mode="median")
                dec_mask_sorted_bin, dec_coeff_mask_sorted_bin, _ = jlc.pc_bin(dec_cor_m_s[0],
                                                                               dec_cor_m_s[1],
                                                                               dec_cor_m_s[1],
                                                                               coords_scale)

                # Mag running averages
                mag_cor = np.vstack((catalog.mag, self.fit_coeffs[cbv_id]))
                mag_cor_s = mag_cor[:, mag_cor[0].argsort()]
                mag_cor_m = np.vstack((catalog.mag[self.lc_idx_for_cbvs], self.fit_coeffs[cbv_id][self.lc_idx_for_cbvs]))
                mag_cor_m_s = mag_cor_m[:, mag_cor_m[0].argsort()]

                mag_sorted_bin, mag_coeff_sorted_bin, _ = jlc.pc_bin(mag_cor_s[0], mag_cor_s[1],
                                                                     mag_cor_s[1], 0.5, mode="median")
                mag_mask_sorted_bin, mag_coeff_mask_sorted_bin, _ = jlc.pc_bin(mag_cor_m_s[0],
                                                                               mag_cor_m_s[1],
                                                                               mag_cor_m_s[1],
                                                                               0.5)

                # plot against ra
                ax[0].plot(self.fit_coeffs[cbv_id], catalog.ra, '.', color='grey', label='All')
                ax[0].plot(self.fit_coeffs[cbv_id][self.lc_idx_for_cbvs], catalog.ra[self.lc_idx_for_cbvs],
                           'k.', label='SVD')
                ax[0].plot(ra_coeff_sorted_bin, ra_sorted_bin, 'b-', label="All")
                ax[0].plot(ra_coeff_mask_sorted_bin, ra_mask_sorted_bin, 'r-', label="SVD")
                ax[0].set_ylabel("R.A.")
                ax[0].set_xlim(llim, ulim)
                ax[0].legend(fontsize='small', loc='upper right')

                # plot against dec
                ax[1].plot(self.fit_coeffs[cbv_id], catalog.dec, '.', color='grey', label='All')
                ax[1].plot(self.fit_coeffs[cbv_id][self.lc_idx_for_cbvs], catalog.dec[self.lc_idx_for_cbvs],
                           'k.', label='SVD')
                ax[1].plot(dec_coeff_sorted_bin, dec_sorted_bin, 'b-', label="All")
                ax[1].plot(dec_coeff_mask_sorted_bin, dec_mask_sorted_bin, 'r-', label="SVD")
                ax[1].set_ylabel("Dec.")
                ax[1].set_xlim(llim, ulim)
                ax[1].legend(fontsize='small', loc='upper right')

                # plot against mag
                ax[2].plot(self.fit_coeffs[cbv_id], catalog.mag, '.', color='grey', label='All')
                ax[2].plot(self.fit_coeffs[cbv_id][self.lc_idx_for_cbvs], catalog.mag[self.lc_idx_for_cbvs],
                           'k.', label='SVD')
                ax[2].plot(mag_coeff_sorted_bin, mag_sorted_bin, 'b-', label="All")
                ax[2].plot(mag_coeff_mask_sorted_bin, mag_mask_sorted_bin, 'r-', label="SVD")
                ax[2].set_ylabel("Mag")
                ax[2].set_xlabel(f"Coeff value, CBV {cbv_id}")
                ax[2].set_xlim(llim, ulim)
                ax[2].legend(fontsize='small', loc='upper right')

                fig.tight_layout()
                fig.subplots_adjust(hspace=0.0)
                pdf.savefig()
                plt.close()

    def cotrend_data_map_no_mp(self, catalog, camera_id, timeslot):
        """
        Process the data using MAP but in a cingle core

        This is mainly used for testing the times to execution for different stars
        """
        target_ids = np.arange(0, len(self.norm_flux_array))
        n_data_points = len(self.norm_flux_array[0])

        # make an empty array for holding the correction
        correction = np.empty((len(target_ids), n_data_points))

        # collect together constants for giving to pool
        logging.info("Running detrending in single core, making const...")
        #const = (catalog, self, camera_id, timeslot)

        # make the new massive const tuple
        constants = (camera_id,
                     timeslot,
                     self.test_stars,
                     self.debug,
                     self.direc,
                     self.cbvs,
                     self.fit_coeffs,
                     catalog,
                     self.lc_idx_for_cbvs,
                     self.theta,
                     self.n_cbvs,
                     self.variability_normalisation_order,
                     self.prior_normalised_variability_limit,
                     self.prior_cond_snapping,
                     self.prior_raw_goodness_weight,
                     self.prior_raw_goodness_exponent,
                     self.prior_noise_goodness_weight,
                     self.prior_pdf_variability_weight,
                     self.prior_pdf_goodness_gain,
                     self.prior_pdf_goodness_weight)

        # loop over all target ids and call worker_fn manually and catch the results
        for target_id in target_ids:
            r, c, t = worker_fn(target_id, self.norm_flux_array[target_id],
                                self.variability[target_id], constants)
            correction[r] = c
            exec_times.append(t)

        # finally cotrend the lightcurves
        self.cotrending_flux_array = correction
        self.cotrended_flux_array = self.norm_flux_array - self.cotrending_flux_array
        self.map_exec_times = np.array(exec_times)


    def cotrend_data_map_mp(self, catalog, camera_id, timeslot):
        """
        Use multiprocessing to speed up the cotrending of many lcs
        """
        target_ids = np.arange(0, len(self.norm_flux_array))
        n_data_points = len(self.norm_flux_array[0])

        # store the execution times
        exec_times = []

        # make an empty array for holding the correction
        correction = np.empty((len(target_ids), n_data_points))

        # collect together constants for giving to pool
        logging.info("Before pool, making const...")
        #const = (catalog, self, camera_id, timeslot)

        # make the new massive const tuple
        constants = (camera_id,
                     timeslot,
                     self.test_stars,
                     self.debug,
                     self.direc,
                     self.cbvs,
                     self.fit_coeffs,
                     catalog,
                     self.lc_idx_for_cbvs,
                     self.theta,
                     self.n_cbvs,
                     self.variability_normalisation_order,
                     self.prior_normalised_variability_limit,
                     self.prior_cond_snapping,
                     self.prior_raw_goodness_weight,
                     self.prior_raw_goodness_exponent,
                     self.prior_noise_goodness_weight,
                     self.prior_pdf_variability_weight,
                     self.prior_pdf_goodness_gain,
                     self.prior_pdf_goodness_weight)

        # make the new large itterable for star_ids, fluxes and variability
        # TODO: Is there an issue here?
        big_map_itterable = []
        for i, j, k in zip(target_ids, self.norm_flux_array, self.variability):
            big_map_itterable.append([i, j, k])

        # the old way of plotting with no thread locking
        # however, we just plot with a different script now for speed
        # make a partial function with the constants baked in
        logging.info("Before pool, making fn...")
        fn = partial(worker_fn, constants=constants)

        # run a pool of N workers and set them detrending
        with Pool(self.pool_size) as pool:
            logging.info("Inside pool, trying to map...")
            #results = pool.map(fn, target_ids)
            results = pool.starmap(fn, big_map_itterable)

        # debugging statement
        logging.info("Outside pool, did it work?...")

        # collect the results and make a correction array
        for r, c, t in results:
            correction[r] = c
            exec_times.append(t)

        # finally cotrend the lightcurves
        self.cotrending_flux_array = correction
        self.cotrended_flux_array = self.norm_flux_array - self.cotrending_flux_array
        self.map_exec_times = np.array(exec_times)

    def cotrend_data_ls(self):
        """
        Directly fit the CBVs to the data and ignore prior information
        """
        cotrending_flux_array = []

        for target_id in np.arange(0, len(self.norm_flux_array)):
            logging.info(f"Cotrending {target_id}/{len(self.norm_flux_array)}...")
            correction_to_apply = []
            for cbv_id in sorted(self.cbvs):
                component = self.cbvs[cbv_id]*self.fit_coeffs[cbv_id][target_id]
                correction_to_apply.append(component)

            correction_to_apply = np.sum(np.array(correction_to_apply), axis=0)
            cotrending_flux_array.append(correction_to_apply)

        # finally cotrend the lightcurves
        self.cotrending_flux_array = np.array(cotrending_flux_array)
        self.cotrended_flux_array = self.norm_flux_array - self.cotrending_flux_array

# multiprocessing worker function
def worker_fn(star_id, norm_flux, variability, constants):
    """
    Take in the constant items under the constants tuple
    and the star_id as a changing variable to select the
    right parts of the constant data to perform the cotrending
    """
    logging.info("In worker_fn...")
    # get the time the function started to check concurrency
    start = datetime.utcnow()

    # unpack constants
    #catalog, cbvs, camera_id, timeslot = constants

    # lazy unpack the new large const tuple
    # packing order with these at the start allows us to lazy unpack here
    # as we only need some info for this function, rest is destined for MAP
    camera_id, timeslot, test_stars, debug, direc, cbvs, fit_coeffs, *_ = constants

    try:
        # TODO: fix this call and then catching info in MAP
        mapp = MAP(star_id, norm_flux, variability, constants)
    except Exception:
        # return the star_id and all zeros for the correction
        # if the mapp processing fails
        traceback.print_exc(file=sys.stdout)
        logging.warning("MAP failed, skipping...")
        n_data_points = len(norm_flux)
        return star_id, np.zeros(n_data_points)

    # make plots for the test stars
    if star_id in test_stars or debug:
        # debug plotting moved out of the main script
        # pickle the MAP object also for inspection
        map_filename = f"{direc}/{camera_id}_{timeslot}_{star_id}_map.pkl"
        picklify(map_filename, mapp)

    # try to use our super duper new posterior PDF
    correction_to_apply = []
    for cbv_id in sorted(cbvs.keys()):
        # if MAP succeeded, use the posterior, else fall back to LS results
        if mapp.mode == "MAP":
            final_coeff = mapp.posterior_peak_theta[cbv_id]
        else:
            final_coeff = fit_coeffs[cbv_id][star_id]

        component = cbvs[cbv_id]*final_coeff
        correction_to_apply.append(component)

    # sum up the scaled CBVs then do the correction
    correction_to_apply = np.sum(np.array(correction_to_apply), axis=0)

    # check the run time for this star
    end = datetime.utcnow()
    diff_time = (end - start).seconds
    logging.info(f"[{star_id}] Started: {start} - Finished: {end} - Runtime: {diff_time} sec")

    return star_id, correction_to_apply, diff_time
