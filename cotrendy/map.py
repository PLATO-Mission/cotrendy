"""
MAP components for Cotrendy
"""
import logging
from collections import defaultdict
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import simps
from scipy.stats import median_absolute_deviation as mad
from scipy.signal import periodogram

# pylint: disable=invalid-name

class MAP():
    """
    Take a set of CBVs fit coeffs and use the MAP method
    to cotrend a series of light curves.

    There is a MAP instance for each light curve

    PDFs and parameters are indexed by CBV_id
    """

    # TODO: Break all coeffs out into config file

    def __init__(self, catalog, cbvs, tus_id):
        """
        Initialise the MAP class

        Parameters
        ----------
        catalog : Catalog object
            Contains information about the targets
            RA, Dec, Mag and ID
        cbvs : CBVs object
            Contains information about the basis vectors
        tus_id : int
            Current target index being analysed
        """
        self.tus_id = tus_id
        self.direc = cbvs.direc
        self.id = int(catalog.ids[tus_id])
        self.mag = round(catalog.mag[tus_id], 2)

        # overall flags for MAP
        self.mode = "MAP"

        # prior PDF
        self.distances = None
        self.prior_sigma = defaultdict(float)
        self.prior_pdf = defaultdict(np.array)
        self.prior_peak_theta = defaultdict(float)
        self.prior_peak_pdf = defaultdict(float)
        self.prior_pdf_integral = defaultdict(float)
        self.hist_bins = 50
        self.prior_max_success = defaultdict(bool)
        self.prior_sigma_mask = None
        self.prior_general_goodness = 0.
        self.prior_noise_goodness = 0.
        self.prior_weight = 0.
        self.prior_weight_pt_var = 0.
        self.prior_weight_pt_gen_good = 0.
        self.prior_snapped_to_cond = defaultdict(bool)

        # conditioanl PDF
        self.cond_pdf = defaultdict(np.array)
        self.cond_peak_theta = defaultdict(float)
        self.cond_peak_pdf = defaultdict(float)
        self.cond_max_success = defaultdict(bool)

        # posterior PDF
        self.posterior_pdf = defaultdict(np.array)
        self.posterior_peak_theta = defaultdict(list)
        self.posterior_peak_pdf = defaultdict(list)
        self.posterior_max_success = defaultdict(list)

        # make a mask to exclude the current tus
        # if there is a cbv mask we check this object is not in there
        mask = np.where(cbvs.lc_idx_for_cbvs != tus_id)[0]
        self.prior_mask = cbvs.lc_idx_for_cbvs[mask]

        # calculate the PDFs
        self.calculate_prior_pdfs(catalog, cbvs)

        # analyse the info we have, if the prior_goodness is poor
        # or the star is quiet, break back to a pure LS fit and
        # ignore the rest of the MAP analysis
        if self.prior_general_goodness < 0.1 or \
                cbvs.variability[self.tus_id] < cbvs.prior_normalised_variability_limit:
            self.mode = "LS"
            # we stop here and MAP defaults to LS
        else:
            self.calculate_conditional_pdfs(cbvs)

            # finally combine the PDFs above into the posterior and maximise
            self.calculate_posterior_pdfs(cbvs)

            failures = 0
            # check if any maximising failed
            for cbv_id in sorted(cbvs.cbvs.keys()):
                if not self.prior_max_success[cbv_id] or not self.cond_max_success[cbv_id] or not \
                    self.posterior_max_success[cbv_id]:
                    failures += 1
            if failures > 0:
                self.mode = "LS"

    def calculate_prior_pdfs(self, catalog, cbvs):
        """
        Calculate the weights for all SVD stars compared to the TUS

        Taken from eqs 17 and 18 of Smith et al. 2012 Kepler PDC paper.

        Note: here we have ra, dec and gmag equally weighted

        Parameters
        ----------
        catalog : Catalog object
            Contains information about the targets
            RA, Dec, Mag and ID
        cbvs : CBVs object
            Contains information about the basis vectors
        """
        lamb = np.matrix(np.diag([mad(catalog.ra[self.prior_mask])/catalog.ra_weight,
                                  mad(catalog.dec[self.prior_mask])/catalog.dec_weight,
                                  mad(catalog.mag[self.prior_mask])/catalog.mag_weight]))
        lamb_inv = np.linalg.inv(lamb)

        # work out the distances in the 3D parameter space
        distances = []
        for i in self.prior_mask:
            ra_diff = catalog.ra[self.tus_id] - catalog.ra[i]
            dec_diff = catalog.dec[self.tus_id] - catalog.dec[i]
            mag_diff = catalog.mag[self.tus_id] - catalog.mag[i]
            diff = np.matrix([[ra_diff], [dec_diff], [mag_diff]])
            distances.append(float(np.sqrt(np.dot(np.dot(diff.T, lamb_inv), diff))))

        # make distances an array
        distances = np.array(distances)
        self.distances = distances

        # work out which stars to measure sigma for
        # using a similar method to Kepler
        # start with a small window around this object, expand it to
        # get > 10 objects, then take the sigma for this CBV from there
        hw = 0.25
        n_close = 0
        while n_close < 10:
            close_loc = np.where(self.distances < hw)[0]
            n_close = len(close_loc)
            hw *= 2
        self.prior_sigma_mask = self.prior_mask[close_loc]

        # the large distances should be a low weight, so invert them
        # Kepler PDC uses the inverse_square
        weights_unnormalised = 1./(distances**2)

        # now normalise the weights so their total gives 1
        weights_norm_const = np.sum(weights_unnormalised)
        weights = weights_unnormalised / weights_norm_const

        for cbv_id in sorted(cbvs.cbvs):
            self.prior_pdf[cbv_id] = self._kde_scipy(cbvs.fit_coeffs[cbv_id][self.prior_mask],
                                                     cbvs.theta[cbv_id], weights=weights)

            # maximise the prior PDF
            peak_theta, peak_pdf = self._maximise_pdf(cbvs.theta[cbv_id],
                                                      self.prior_pdf[cbv_id],
                                                      'prior')

            # check if maximising failed
            if peak_theta is None or peak_pdf is None:
                self.prior_max_success[cbv_id] = False
                self.prior_peak_theta[cbv_id] = 0.0
                self.prior_peak_pdf[cbv_id] = 0.0
            else:
                self.prior_max_success[cbv_id] = True
                self.prior_peak_theta[cbv_id] = peak_theta
                self.prior_peak_pdf[cbv_id] = peak_pdf

            # calculate the integral of the weighted PDF, it should be close to 1
            dx = cbvs.theta[cbv_id][1] - cbvs.theta[cbv_id][0]
            self.prior_pdf_integral[cbv_id] = simps(self.prior_pdf[cbv_id], dx=dx)

            # work out sigma for this cbv
            sigma_fit_coeffs = cbvs.fit_coeffs[cbv_id][self.prior_sigma_mask]
            self.prior_sigma[cbv_id] = np.std(sigma_fit_coeffs)

        self.prior_general_goodness, self.prior_noise_goodness = self.calculate_prior_goodness(cbvs)
        self.prior_weight, self.prior_weight_pt_var, self.prior_weight_pt_gen_good = self.calculate_prior_weight(cbvs)

    def calculate_prior_goodness(self, cbvs):
        """
        Analyse the prior fit and see if it is better or
        worse than expected.

        This is done in two parts, a general goodness (compared to a low
        order polyfit) and a noise goodness, by comparing periodograms
        of the diffs of the uncorrected data and the prior fit
        """
        # generate the prior fit light curve
        prior_coeffs = []
        for cbv_id in sorted(cbvs.cbvs.keys()):
            prior_coeffs.append(self.prior_peak_theta[cbv_id])
        prior_coeffs = np.array(prior_coeffs).reshape(cbvs.n_cbvs, -1)
        prior_components = cbvs.vect_store * prior_coeffs
        prior_fit = np.sum(prior_components, axis=0)

        # calculate the general trend goodness
        x = np.arange(0, len(prior_fit))
        coeffs = np.polyfit(x, cbvs.norm_flux_array[self.tus_id], 3)
        poly_fit = np.polyval(coeffs, x)

        # get the difference between the prior fit light curve and
        # a low order detrended light curve
        diff_prior_to_poly = prior_fit - poly_fit

        # normalise by the mad of the polyfit removed light curve
        abs_dev = mad(cbvs.norm_flux_array[self.tus_id] - poly_fit)
        std_diff_prior_to_poly = np.std((diff_prior_to_poly/abs_dev) - 1)

        # the scaling values come from Kepler
        # a value near 0 is a bad prior goodness, a value near 1 is a good prior goodness
        # TODO: Pull the scaling factors out to the config file
        # and also workout out how to choose those values!?
        prior_general_goodness = 1 - (std_diff_prior_to_poly/5.)**(3)

        # set it to 0 if it's < 0
        if prior_general_goodness < 0.0:
            prior_general_goodness = 0.0

        # Now calculate the noise goodness metric
        _, psd_flux = periodogram(np.diff(cbvs.norm_flux_array[self.tus_id]), detrend=False)
        _, psd_prior = periodogram(np.diff(prior_fit), detrend=False)

        # TODO: same here, why is the noise 2e-4?
        psd_ratio = psd_prior / psd_flux
        noise_weight = 2e-4
        prior_noise_goodness = noise_weight * np.sum(np.log(psd_ratio[psd_ratio > 1]**2))
        prior_noise_goodness = 1. / (prior_noise_goodness + 1)

        return prior_general_goodness, prior_noise_goodness


    def calculate_conditional_pdfs(self, cbvs):
        """
        Generate the conditional PDF for each CBV

        This represents the direct LS fit of the CBVs to the data

        Parameters
        ----------
        cbvs : CBVs object
            Contains information about the basis vectors
        """
        # keep a running total of the fit residuals, starting with the data
        data = np.copy(cbvs.norm_flux_array[self.tus_id])
        # CBV ID 0 may not always be there, get the smallest number
        valid_cbv_id = sorted(cbvs.theta.keys())[0]
        # number of thetas to try
        n_theta = len(cbvs.theta[valid_cbv_id])
        # grab the stddev of the data
        sigma = np.std(data)

        # loop over the CBVs
        for cbv_id in sorted(cbvs.cbvs):
            # make a 2D copy of the light curve theta times
            # each successive CBV will be subtracted and a conditional made
            fit_residuals = data.repeat(n_theta).reshape(len(data), n_theta)

            # scale CBV by each theta and subtract it to get the residuals
            for i, t in enumerate(cbvs.theta[cbv_id]):
                fit_residuals[:, i] -= cbvs.cbvs[cbv_id]*t

            # generate the conditional PDF from the fit residuals
            self.cond_pdf[cbv_id] = -1./(2.*(sigma**2)) * np.diag(np.dot(fit_residuals.T,
                                                                         fit_residuals))

            # maximise the conditional PDF
            peak_theta, peak_pdf = self._maximise_pdf(cbvs.theta[cbv_id],
                                                      self.cond_pdf[cbv_id],
                                                      'cond')

            # check if maximising failed
            if peak_theta is None or peak_pdf is None:
                self.cond_max_success[cbv_id] = False
                self.cond_peak_theta[cbv_id] = 0.0
                self.cond_peak_pdf[cbv_id] = 0.0
            else:
                self.cond_max_success[cbv_id] = True
                self.cond_peak_theta[cbv_id] = peak_theta
                self.cond_peak_pdf[cbv_id] = peak_pdf

            # update the data and sigma for the next round
            data = data - (cbvs.cbvs[cbv_id] * self.cond_peak_theta[cbv_id])
            sigma = np.std(data)

    def calculate_prior_weight(self, cbvs):
        """
        Calculate the weighting of the prior information for the posterior PDF
        """
        # calculate the variability part
        # TODO: pull out this scaling coeff. Why is is 2 in Kepler PDC?
        prior_pdf_variability_weight = 2.0
        part_var = (1 + cbvs.variability[self.tus_id])**prior_pdf_variability_weight

        # calculate the prior goodness part
        prior_goodness_gain = 1.0
        prior_goodness_weight = 0.5
        part_gen_goodness = prior_goodness_gain * (self.prior_general_goodness**prior_goodness_weight)
        prior_weight = part_var * part_gen_goodness
        return prior_weight, part_var, part_gen_goodness

    def calculate_posterior_pdfs(self, cbvs):
        """
        Generate the posterior PDF for each CBV

        This is a combination of the conditional PDF and the weighted* prior PDF

        Parameters
        ----------
        cbvs : CBVs object
            Contains information about the basis vectors
        """
        for cbv_id in sorted(cbvs.cbvs):
            # make the check for conditional and prior being within 1 sigma
            # if so, skip the posterior and snap to the conditional only
            sigma_snap_llim = self.prior_peak_theta[cbv_id] - self.prior_sigma[cbv_id]
            sigma_snap_ulim = self.prior_peak_theta[cbv_id] + self.prior_sigma[cbv_id]
            if sigma_snap_llim <= self.cond_peak_theta[cbv_id] <= sigma_snap_ulim:
                self.posterior_pdf[cbv_id] = self.cond_pdf[cbv_id]
                peak_theta = self.cond_peak_theta[cbv_id]
                peak_pdf = self.cond_peak_pdf[cbv_id]
                self.prior_snapped_to_cond[cbv_id] = True
            else:
                posterior = self.cond_pdf[cbv_id] + np.log(self.prior_pdf[cbv_id])*self.prior_weight
                self.posterior_pdf[cbv_id] = posterior
                peak_theta, peak_pdf = self._maximise_pdf(cbvs.theta[cbv_id],
                                                          posterior,
                                                          'posterior')
                self.prior_snapped_to_cond[cbv_id] = False

            if peak_theta is None or peak_pdf is None:
                self.posterior_max_success[cbv_id].append(False)
                self.posterior_peak_theta[cbv_id] = 0.0
                self.posterior_peak_pdf[cbv_id] = 0.0
            else:
                self.posterior_max_success[cbv_id].append(True)
                self.posterior_peak_theta[cbv_id] = peak_theta
                self.posterior_peak_pdf[cbv_id] = peak_pdf

    def _maximise_pdf(self, theta, pdf, pdf_type):
        """
        Take a PDF and maximise it to find the best fitting coefficient

        Parameters
        ----------
        theta : array
            Array of fit coefficients
        pdf : array
            Probability of each fit coeff in theta
        pdf_type : string
            Type of PDF being maximised

        Returns
        -------
        theta_peak : float
            Theta value at max of PDF
        pdf_peak : float
            PDF value at max

        Raises
        ------
        None
        """
        # find the maximum of the PDF then perform a quadratic fit
        # of the 3 values surrounding the maximum to find the best
        # fitting theta
        tm = np.where(pdf == np.max(pdf))[0][0]
        if tm >= 1 and tm <= len(pdf) - 2:
            peak_slice = pdf[tm-1: tm+2]
            theta_slice = theta[tm-1: tm+2]
            coeffs = np.polyfit(theta_slice, peak_slice, 2)
            # y = ax^2 + bx + c
            # peak is where dy/dx = 0
            # 2ax + b = 0, x = -b / (2a)
            theta_peak = -coeffs[1] / (2*coeffs[0])
            pdf_peak = np.polyval(coeffs, theta_peak)
        else:
            logging.warning(f"[{self.tus_id}:{pdf_type}] PDF peak location unbound!")
            theta_peak = None
            pdf_peak = None
        return theta_peak, pdf_peak

    @staticmethod
    def _kde_scipy(x, x_grid, weights=None):
        """
        Kernel Density Estimation with Scipy

        Parameters
        ----------
        x : array
            Array of values for generate PDF
        x_grid : array
            Array of value to evaluate PDF over

        Returns
        -------
        pdf : array
            PDF of values in x

        Raises
        ------
        None
        """
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # we divide the bandwidth by the sample standard deviation here.

        # I removed the bandwidth variable as it works it out itself - jmcc
        kde = gaussian_kde(x, weights=weights)
        return kde.evaluate(x_grid)
