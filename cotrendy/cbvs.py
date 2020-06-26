"""
Cotrending Basis Vectors compoents for Cotrendy
"""
import gc
import sys
import traceback
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr
from scipy.linalg import svd
import scipy.optimize as optimization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cotrendy.map import MAP

# pylint: disable=invalid-name

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
             1. The right hand singular vectors contain the contribution of the signal in the
                basis vectors
             1. These are the ROWS in VT
             1. Entropy is calculated using Eqs 13-16 in the Smith paper
                1. They find anything with an entropy less than -0.7 was bad
                1. Bad targets are removed and the SVD is done again until all are better than 0.7
             1. However, I spoke with Tom Marsh and it seems like a simple itterative sigma
                clip done on the outliers of objects in VT will do the same thing.
                1. I've implemented this itterative sigma clip below
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
        self.phot_file = config['data']['flux_file'].split('.')[0]
        self.timesteps = timesteps
        self.targets = targets
        self.lc_idx = np.arange(0, len(targets))

        # shortlist of stars to analyse more closely
        self.test_stars = config['cotrend']['test_stars']

        # placeholder for normalise variability stats
        self.normalised_variability_limit = config['cotrend']['normalised_variability_limit']
        self.variability = None

        # placeholder for CBV domination checks
        # starts with ones, if a star dominates it
        # gets set to zero later and CBVs redone
        self.cbv_domination = np.ones(len(targets))

        # some mask placeholders
        self.variability_mask = np.ones(len(targets))
        self.pre_cbv_mask = None
        self.cbv_mask = None

        # params for generating CBVs
        self.correlations = defaultdict(list)
        self.median_correlations = []
        self.high_correlation_mask = []
        self.norm_flux_dithered_array = None

        # SVD params
        self.U, self.s, self.VT = None, None, None

        # hold the CBVs
        self.cbvs = defaultdict(list)
        # this is a numpy array that's easier to broadcast on
        self.vect_store = None
        # the max number to look for
        self.max_n_cbvs = config['cotrend']['max_n_cbvs']
        # the actual number extracted
        self.n_cbvs = 0

        # storage for the CBV SNRs
        self.cbvs_snr = {}
        self.cbvs_snr_limit = config['cotrend']['cbv_snr_limit'] # dB

        # CBV fit coefficients
        self.fit_coeffs = defaultdict(list)

        # keep the theta grids for the PDFs in MAP
        self.theta = defaultdict(np.array)

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

    def calculate_normalised_variability(self):
        """
        First apply a filter to remove variable stars

        NOTE: This may be part of why PDC struggled with
        long term variability

        V = sigma_y / (delta_y * Vbar)

        Loop over all the stars, subtract a 3rd order polynomial
        Work out the RMS
        """
        print("Finding variable stars...")

        sigma_y, delta_y = [], []
        for target in self.targets:
            delta_y.append(np.average(target.fluxerr_wtrend))
            coeffs = np.polyfit(self.timesteps, target.flux_wtrend, 3)
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
        var_loc = np.where(self.variability > self.normalised_variability_limit)[0]
        self.variability_mask[var_loc] = 0.0

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

    def calculate_pearson_correlation(self):
        """
        Take the array of targets and find the most correlated

        TODO: This is terrible code, fix this when happy with method!!!

        # old method:
        # NOTE: This original stat of median correlation is too basic for real data
        # It is dominated by faint stars. We need to find those that highly correlate
        # more carefully and then compare them to each other, take that as their correlation
        # statistic. Finally when selecting the best ones, we want the top 50% of ONLY the
        # good ones, not 50% of all of them!
        #self.median_correlations = np.array([np.median(self.correlations[i]) for i in self.correlations])
        """

        print("Finding correlated stars...")
        n_stars_to_correlate = len(self.norm_flux_array)
        self.high_correlation_mask = []

        for j in range(0, n_stars_to_correlate):
            print(f"{j+1}/{n_stars_to_correlate}")
            for i in range(0, n_stars_to_correlate):
                if i != j:
                    r = pearsonr(self.norm_flux_array[j], self.norm_flux_array[i])
                    self.correlations[j].append(r[0])
            # check if this star has high correlation with at least some stars
            high_corr = np.where(np.array(self.correlations[j]) > 0.5)[0]
            if len(high_corr) > 0.10*n_stars_to_correlate:
                self.high_correlation_mask.append(j)

        # now we know which are highly correlated
        # loop over and make a mask to keep the best ones
        # setting the not-good-ones to zero
        print("Second pass on correlation stars...")
        for j in range(0, n_stars_to_correlate):
            # if the star is in the high correlation pile, continue
            local_correlations = []
            if j in self.high_correlation_mask:
                for i in range(0, n_stars_to_correlate):
                    if i in self.high_correlation_mask and i != j:
                        local_correlations.append(pearsonr(self.norm_flux_array[j], self.norm_flux_array[i]))
                self.median_correlations.append(np.median(local_correlations))
            # else give it a score of 0 for correlation
            else:
                self.median_correlations.append(0)

        # finally set the median correlations list as an array for masking
        self.median_correlations = np.array(self.median_correlations)

    def _mask_stars_for_cbvs(self):
        """
        Take the variability mask and combine it
        with the pearson correlations to find
        the best CBV stars

        Use the binary variability mask and multiply
        by the correlation values. Those that are variable
        go to 0.
        """
        print("Masking poor CBV stars...")
        self.pre_cbv_mask = self.median_correlations * self.variability_mask * self.cbv_domination
        # sort the lcs by combined mask score
        temp = zip(self.pre_cbv_mask, self.lc_idx)
        temp = sorted(temp, reverse=True)
        # TODO: Make a plot here to show the stars scores
        # maybe the top 50% is not correct?
        pre_cbv_mask_sorted, lc_idx_sorted = map(np.array, zip(*temp))
        # find the targets with non-zero scores
        n = np.where(pre_cbv_mask_sorted != 0)[0]
        # keep the top 50% most correlated light curves, non-zero light curves
        self.cbv_mask = lc_idx_sorted[:len(n)//2]

    def _apply_zero_mean_dither(self):
        """
        In order for avoid an artifical node where the lcs
        cross zero we dither them slightly before making the
        CBVs. This is only done when making the CBVs, not
        when analysing the actual light curves
        """
        print("Applying zero mean dither...")
        n_light_curves = len(self.norm_flux_array)
        dither = np.random.normal(loc=0.0, scale=0.001, size=n_light_curves)
        self.norm_flux_dithered_array = self.norm_flux_array + dither.reshape(n_light_curves, 1)

    def calculate_svd(self, cbv_pass):
        """
        Take the previously filtered flux array and calculate the SVD

        We apply a zero mean dither before hand
        """
        print("Calculating the SVD...")
        # now we take the SVD of the highest correlated light curves
        # NOTE fluxes need to be in columns in order for the left-singular maxtrix (U)
        # to return us the CBVs in order
        # NOTE we only need to work out variability, correlation and zero dither once
        if cbv_pass == 0:
            self.calculate_normalised_variability()
            self.calculate_pearson_correlation()
            self._apply_zero_mean_dither()
        self._mask_stars_for_cbvs()
        print("Calculating SVD...")
        self.U, self.s, self.VT = svd(self.norm_flux_dithered_array[self.cbv_mask].T)
        print(f"Matrix shapes -  U: {self.U.shape}, s: {self.s.shape}, VT: {self.VT.shape}")

    def calculate_cbv_snr(self):
        """
        Kepler implements as check on the SNR for each
        CBV. If the SNR (dB) is < 5dB, they are removed.

        They typically use 8 and only some CBVs across
        the entire FOV were ever removed
        """
        to_delete = []
        print(f"Calculating {self.max_n_cbvs} CBV SNRs...")
        for i in sorted(self.cbvs):
            Anoise = np.std(np.diff(self.cbvs[i]))
            Asignal = np.std(self.cbvs[i])
            snr = 10*np.log10((Asignal**2) / (Anoise**2))
            self.cbvs_snr[i] = snr
            print(f"CBV {i} SNR: {snr:.3f} dB")
            if snr < self.cbvs_snr_limit:
                print("Low SNR CBV, deleting!")
                to_delete.append(i)
        # do the deleting after the loop, otherwise, splosions
        for i in to_delete:
            del self.cbvs[i]
        # store the final number of CBVs
        self.n_cbvs = len(self.cbvs)

    def calculate_cbvs(self):
        """
        Generate the CBVs from the SVD output
        """
        print("Calculating CBVs...")
        # fudge domination level to massive number to force looping
        n_dominating = 1E6
        # keep track of the number of passes on CBV generation
        cbv_pass = 0
        # loop until there are no more dominating stars
        while n_dominating != 0:
            print(f"Starting CBV pass {cbv_pass}...")
            self.calculate_svd(cbv_pass)
            for i in range(self.max_n_cbvs):
                self.cbvs[i] = self.U[:, i]

            # check they have sufficient SNR
            self.calculate_cbv_snr()
            n_dominating = self._check_for_dominating_stars(cbv_pass)
            cbv_pass += 1

        # make a numpy array of the vectors for using later
        # grab the vectors from their dictionary
        vect_store = []
        for c in sorted(self.cbvs):
            vect_store.append(self.cbvs[c])
        # make them into a numpy array
        self.vect_store = np.array(vect_store)

    def _check_for_dominating_stars(self, cbv_pass):
        """
        Kepler implements a check for stars
        that dominate in the creation of CBVs

        Run stats on the right singular vectors for
        each CBV. Any stars that contribute overly
        to the CBV should be removed and the CBVs
        redone
        """
        print("Checking for dominating stars...")
        with PdfPages(f"{self.direc}/dominating_stars_camera_{self.camera_id}_{self.phot_file}_pass_{cbv_pass}.pdf") as pdf:
            # loop over each CVB and check individual star's contributions
            # dominating stars will stick out from the rest
            for i, v in enumerate(self.VT[:len(self.cbvs)]):
                print(f"Checking for dominating stars in CBV {i}")
                mean = np.mean(self.VT[i])
                std = np.std(self.VT[i])
                n = np.where(((v > mean+3*std) | (v < mean-3*std)))[0]
                n_dominating = len(n)

                # make a plot
                fig, ax = plt.subplots(1)
                ax.plot(v, 'k-')
                ax.axhline(mean+3*std, lw=1, color='red')
                ax.axhline(mean-3*std, lw=1, color='red')

                # flag the bad objects
                for j in n:
                    # j is their CBV star index
                    # we need to keep their object index so we can mask them out
                    bad_indx = self.cbv_mask[j]
                    print(f"Removing star {bad_indx} from CBV list")
                    self.cbv_domination[self.cbv_mask[j]] = 0
                    # mark them on the plot also
                    ax.axvline(j, lw=1, color='orange')

                # complete the plotting
                ax.set_xlabel('CBV Star ID')
                ax.set_ylabel(f'Contribution to CVB {i}')
                fig.tight_layout()
                pdf.savefig()
                plt.close()
        return n_dominating

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
        print("Calculating CBVs fit coeffs...")

        # initial guess at fit coeffs
        x0 = [-0.1]*self.n_cbvs
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
            theta_range = abs(max(self.fit_coeffs[j]) - min(self.fit_coeffs[j]))
            # generate generous range of theta
            self.theta[j] = np.linspace(min(self.fit_coeffs[j])-0.10*theta_range,
                                        max(self.fit_coeffs[j])+0.10*theta_range, 2500)

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
        print("Calculating CBVs fit coeffs...")

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
            theta_range = abs(max(self.fit_coeffs[j]) - min(self.fit_coeffs[j]))
            # generate generous range of theta
            self.theta[j] = np.linspace(min(self.fit_coeffs[j])-0.10*theta_range,
                                        max(self.fit_coeffs[j])+0.10*theta_range, 2500)

    def cotrend_data_map_mp(self, catalog):
        """
        Use multiprocessing to speed up the cotrending of many lcs
        """
        target_ids = np.arange(0, len(self.norm_flux_array))
        n_data_points = len(self.norm_flux_array[0])

        # make an empty array for holding the correction
        correction = np.empty((len(target_ids), n_data_points))

        # collect together constants for giving to pool
        const = (catalog, self)

        # make a partial function with the constants baked in
        fn = partial(worker_fn, constants=const)

        # run a pool of 6 workers and set them detrending
        with Pool(6) as pool:
            results = pool.map(fn, target_ids)

        # collect the results and make a correction array
        for r, c in results:
            correction[r] = c

        # finally cotrend the lightcurves
        self.cotrending_flux_array = correction
        self.cotrended_flux_array = self.norm_flux_array - self.cotrending_flux_array

    def cotrend_data_ls(self):
        """
        Directly fit the CBVs to the data and ignore prior information

        This is the simpler approach while the kinks in MAP are worked out
        """
        cotrending_flux_array = []

        for target_id in np.arange(0, len(self.norm_flux_array)):
            print(f"Cotrending {target_id}/{len(self.norm_flux_array)}...")
            correction_to_apply = []
            for cbv_id in sorted(self.cbvs):
                component = self.cbvs[cbv_id]*self.fit_coeffs[cbv_id][target_id]
                correction_to_apply.append(component)

            correction_to_apply = np.sum(np.array(correction_to_apply), axis=0)
            cotrending_flux_array.append(correction_to_apply)

        # finally cotrend the lightcurves
        self.cotrending_flux_array = np.array(cotrending_flux_array)
        self.cotrended_flux_array = self.norm_flux_array - self.cotrending_flux_array

    #def cotrend_data_map(self, catalog, store_map=True):
    #    """
    #    Take the CBVs, generate MAP PDFs and
    #    determine the best fitting values
    #    then scale and correct the lightcurves
    #    """
    #    cotrending_flux_array = []
    #    mapp_store = []
    #
    #    for target_id in np.arange(0, len(self.norm_flux_array)):
    #        print(f"Cotrending {target_id}/{len(self.norm_flux_array)}...")
    #        # generate a MAP object
    #        # this generates the prior, conditional and posterior pdfs
    #        try:
    #            mapp = MAP(catalog, self, target_id)
    #        except Exception:
    #            traceback.print_exc(file=sys.stdout)
    #            print("MAP failed, skipping...")
    #            n_data_points = len(self.norm_flux_array[0])
    #            cotrending_flux_array.append(np.zeros(n_data_points))
    #            if store_map:
    #                mapp_store.append(None)
    #            continue
    #
    #        # make plots for the test stars
    #        if target_id in self.test_stars:
    #            mapp.plot_prior_pdf(self)
    #            mapp.plot_conditional_pdf(self)
    #            mapp.plot_posterior_pdf(self)
    #
    #        # work out very crudely if we want to use the prior or not
    #        # then detrend the lightcurve and store the results
    #        correction_to_apply = []
    #        for cbv_id in sorted(self.cbvs):
    #            sigma = mapp.prior_sigma[cbv_id]
    #            cond_peak_theta = mapp.cond_peak_theta[cbv_id]
    #            prior_peak_theta = mapp.prior_peak_theta[cbv_id]
    #            prior_cond_diff = abs(prior_peak_theta-cond_peak_theta)
    #
    #            # so for simplicity here, we do the following
    #            # if the star is > variability_limit and the conditional
    #            # is > 5 sigma from the prior, this indicates a bad fit
    #            # use the prior. Else just use the conditional fit
    #            if self.variability[target_id] > self.normalised_variability_limit and \
    #                prior_cond_diff > 5*sigma:
    #                best_theta = prior_peak_theta
    #            else:
    #                best_theta = cond_peak_theta
    #
    #            component = self.cbvs[cbv_id]*best_theta
    #            correction_to_apply.append(component)
    #
    #        # sum up the scaled CBVs then do the correction
    #        correction_to_apply = np.sum(np.array(correction_to_apply), axis=0)
    #        cotrending_flux_array.append(correction_to_apply)
    #
    #        # store the mapp objects for debugging
    #        if store_map:
    #            mapp_store.append(mapp)
    #
    #    # pickle the mapp_store for analysis later
    #    if store_map:
    #        cuts.picklify(f"{self.direc}/map.pkl", mapp_store)
    #
    #    # finally cotrend the lightcurves
    #    self.cotrending_flux_array = np.array(cotrending_flux_array)
    #    self.cotrended_flux_array = self.norm_flux_array - self.cotrending_flux_array

# multiprocessing worker function
def worker_fn(star_id, constants):
    """
    Take in the constant items under the constants tuple
    and the star_id as a changing variable to select the
    right parts of the constant data to perform the cotrending
    """
    catalog, cbvs = constants

    try:
        mapp = MAP(catalog, cbvs, star_id)
    except Exception:
        # return the star_id and all zeros for the correction
        # if the mapp processing fails
        traceback.print_exc(file=sys.stdout)
        print("MAP failed, skipping...")
        n_data_points = len(cbvs.norm_flux_array[0])
        return star_id, np.zeros(n_data_points)

    # make plots for the test stars
    if star_id in cbvs.test_stars:
        mapp.plot_prior_pdf(cbvs)
        mapp.plot_conditional_pdf(cbvs)
        mapp.plot_posterior_pdf(cbvs)

    # work out very crudely if we want to use the prior or not
    # then detrend the lightcurve and store the results
    correction_to_apply = []
    for cbv_id in sorted(cbvs.cbvs.keys()):
        sigma = mapp.prior_sigma[cbv_id]
        cond_peak_theta = mapp.cond_peak_theta[cbv_id]
        prior_peak_theta = mapp.prior_peak_theta[cbv_id]
        prior_cond_diff = abs(prior_peak_theta-cond_peak_theta)

        # so for simplicity here, we do the following
        # if the star is > variability_limit and the conditional
        # is > 5 sigma from the prior, this indicates a bad fit
        # use the prior. Else just use the conditional fit
        if cbvs.variability[star_id] > cbvs.normalised_variability_limit and \
            prior_cond_diff > 5*sigma:
            best_theta = prior_peak_theta
        else:
            best_theta = cond_peak_theta

        component = cbvs.cbvs[cbv_id]*best_theta
        correction_to_apply.append(component)

    # sum up the scaled CBVs then do the correction
    correction_to_apply = np.sum(np.array(correction_to_apply), axis=0)
    return star_id, correction_to_apply
