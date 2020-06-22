"""
MAP components for Cotrendy
"""
import gc
from collections import defaultdict
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import simps
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

# pylint: disable=invalid-name

class MAP():
    """
    Take a set of CBVs and use the MAP method
    to cotrend a series of light curves.

    There is a MAP CLASS FOR EACH OBJECT!

    PDFs and parameters are indexed by CBV_id
    """
    def __init__(self, catalog, cbvs, tus_id, direc):
        """
        Initialise the MAP class
        """
        self.tus_id = tus_id
        self.direc = direc

        # prior PDF
        self.distances = None
        self.prior_sigma = defaultdict(float)
        self.prior_pdf = defaultdict(np.array)
        self.prior_peak_theta = defaultdict(float)
        self.prior_peak_pdf = defaultdict(float)
        self.prior_pdf_integral = defaultdict(float)
        self.hist_bins = 50
        self.prior_max_success = defaultdict(bool)

        # conditioanl PDF
        self.cond_pdf = defaultdict(np.array)
        self.cond_peak_theta = defaultdict(float)
        self.cond_peak_pdf = defaultdict(float)
        self.cond_max_success = defaultdict(bool)

        # posterior PDF
        self.posterior_pdf = defaultdict(list)
        self.posterior_peak_theta = defaultdict(list)
        self.posterior_peak_pdf = defaultdict(list)
        self.posterior_max_success = defaultdict(list)

        # the prior weights need working out emperically
        # for now we just take a list of them and see how they
        # effect the posterior pdf
        self.prior_pdf_weights = np.arange(1, 15)

        # make a mask to exclude the current tus
        # no masking actually occurs if tus is not an SVD star
        mask = np.where(cbvs.cbv_mask != tus_id)[0]
        self.prior_mask = cbvs.cbv_mask[mask]

        # calculate the PDFs
        self.calculate_prior_pdfs(catalog, cbvs)
        self.calculate_conditional_pdfs(cbvs)
        self.calculate_posterior_pdfs(cbvs)

        # take some notes on the success of maximising the PDFs
        self.all_max_success = False
        failures = 0
        # check if any maximising failed
        for cbv_id in cbvs.cbvs.keys():
            if not self.prior_max_success[cbv_id] or not self.cond_max_success[cbv_id] or not \
                self.posterior_max_success[cbv_id]:
                failures += 1
        if failures == 0:
            self.all_max_success = True

    def calculate_prior_pdfs(self, catalog, cbvs):
        """
        Calculate the weights for all SVD stars compared to the TUS

        Taken from eqs 17 and 18 of Smith et al. 2012 Kepler PDC paper.

        Note: here we have ra, dec and gmag equally weighted
        """
        lamb = np.matrix(np.diag([np.var(catalog.ra[self.prior_mask]),
                                  np.var(catalog.dec[self.prior_mask]),
                                  np.var(catalog.mag[self.prior_mask])]))
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

        # here we work out which stars are closest in distance
        # we find those below some cut (say 20%) and then
        # take the stddev of the fit coeffs of those stars, then use this
        # stddev to snap the prior to the conditional if the prior is within
        # some sigma of the conditional
        ds = np.copy(distances)
        obj_id_at_limit = int(len(distances)*0.2)
        ds.sort()
        distance_limit_for_sigma = ds[obj_id_at_limit]

        # work out which stars to measure sigma for
        sigma_mask = np.where(distances <= distance_limit_for_sigma)[0]

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
            else:
                self.prior_max_success[cbv_id] = True

            # store the values for later use
            self.prior_peak_theta[cbv_id] = peak_theta
            self.prior_peak_pdf[cbv_id] = peak_pdf

            # calculate the integral of the weighted PDF, it should be close to 1
            dx = cbvs.theta[cbv_id][1] - cbvs.theta[cbv_id][0]
            self.prior_pdf_integral[cbv_id] = simps(self.prior_pdf[cbv_id], dx=dx)

            # work out sigma for this cbv
            sigma_fit_coeffs = cbvs.fit_coeffs[cbv_id][self.prior_mask][sigma_mask]
            self.prior_sigma[cbv_id] = np.std(sigma_fit_coeffs)

    def calculate_conditional_pdfs(self, cbvs):
        """
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
                correction_factor = 0
            else:
                self.cond_max_success[cbv_id] = True
                correction_factor = peak_theta

            self.cond_peak_theta[cbv_id] = peak_theta
            self.cond_peak_pdf[cbv_id] = peak_pdf

            # update the data and sigma for the next round
            data = data - (cbvs.cbvs[cbv_id] * correction_factor)
            sigma = np.std(data)

    def calculate_posterior_pdfs(self, cbvs):
        """
        """
        for cbv_id in sorted(cbvs.cbvs):
            for weight in self.prior_pdf_weights:

                # make the check for conditional and prior being within 1 sigma
                sigma_snap_llim = self.prior_peak_theta[cbv_id] - self.prior_sigma[cbv_id]
                sigma_snap_ulim = self.prior_peak_theta[cbv_id] + self.prior_sigma[cbv_id]
                if self.cond_peak_theta[cbv_id] >= sigma_snap_llim and \
                    self.cond_peak_theta[cbv_id] <= sigma_snap_ulim:
                    posterior = self.cond_pdf[cbv_id]
                else:
                    posterior = self.cond_pdf[cbv_id] + self.prior_pdf[cbv_id]*weight
                self.posterior_pdf[cbv_id].append(posterior)
                peak_theta, peak_pdf = self._maximise_pdf(cbvs.theta[cbv_id],
                                                          posterior,
                                                          'posterior')

                if peak_theta is None or peak_pdf is None:
                    self.posterior_max_success[cbv_id].append(False)
                else:
                    self.posterior_max_success[cbv_id].append(True)

                # store the peak etc for this weight
                self.posterior_peak_theta[cbv_id].append(peak_theta)
                self.posterior_peak_pdf[cbv_id].append(peak_pdf)

    def _maximise_pdf(self, theta, pdf, pdf_type):
        """
        Take a PDF and maximise it to find the best fitting coefficient
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
            print(f"[{self.tus_id}:{pdf_type}] PDF peak location unbound!")
            theta_peak = None
            pdf_peak = None
        return theta_peak, pdf_peak

    @staticmethod
    def _kde_scipy(x, x_grid, weights=None):
        """Kernel Density Estimation with Scipy"""
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # we divide the bandwidth by the sample standard deviation here.

        # I removed the bandwidth variable as it works it out itself - jmcc
        kde = gaussian_kde(x, weights=weights)
        return kde.evaluate(x_grid)

    def plot_prior_pdf(self, cbvs):
        """
        Plot a histogram of the coeffs, then over plot the
        generated weighted PDF

        Call this function with the same coeffs used to generate the
        weighted prior PDF otherwise it doesn't make sense
        """
        fig, axar = plt.subplots(len(cbvs.cbvs.keys()), figsize=(10, 10))

        # make the axis a list if there is only one
        if len(cbvs.cbvs.keys()) == 1:
            axar = [axar]

        for i, ax in zip(sorted(cbvs.cbvs.keys()), axar):
            _ = ax.hist(cbvs.fit_coeffs[i], bins=self.hist_bins,
                        density=True, label='Theta Histogram')
            # draw a vertical line for the max of each PDF
            _ = ax.axvline(self.prior_peak_theta[i], color='blue', ls='--')
            _ = ax.axvline(self.cond_peak_theta[i], color='red', ls='--')
            _ = ax.axvline(min(self.posterior_peak_theta[i]), color='green', ls='--')
            _ = ax.axvline(max(self.posterior_peak_theta[i]), color='green', ls='--')
            label = f"Theta PDFw ({self.prior_peak_theta[i]:.4f})"
            _ = ax.plot(cbvs.theta[i], self.prior_pdf[i], 'r-',
                        label=label)
            #_ = ax.set_xlabel('Theta')
            #_ = ax.set_ylabel('Frequency')
            #_ = ax.legend()

        fig.tight_layout()
        fig.savefig(f"{self.direc}/prior_pdfs_star{self.tus_id:06d}.png")
        fig.clf()
        plt.close()
        gc.collect()

    def plot_conditional_pdf(self, cbvs):
        """
        Plot the conditional PDF and overplot the maximum
        """
        fig, axar = plt.subplots(len(cbvs.cbvs.keys()), figsize=(10, 10))
        if len(cbvs.cbvs.keys()) == 1:
            axar = [axar]

        for i, ax in zip(sorted(cbvs.cbvs.keys()), axar):
            # draw a vertical line for the max of each PDF
            _ = ax.axvline(self.prior_peak_theta[i], color='blue', ls='--')
            _ = ax.axvline(self.cond_peak_theta[i], color='red', ls='--')
            _ = ax.axvline(min(self.posterior_peak_theta[i]), color='green', ls='--')
            _ = ax.axvline(max(self.posterior_peak_theta[i]), color='green', ls='--')
            label = f"Cond ({self.cond_peak_theta[i]:.4f})"
            _ = ax.plot(cbvs.theta[i], self.cond_pdf[i], 'k-',
                        label=label)
            #_ = ax.set_xlabel('Theta')
            #_ = ax.set_ylabel('Arbitrary units')
            #_ = ax.legend()

        fig.tight_layout()
        fig.savefig(f"{self.direc}/conditional_pdfs_star{self.tus_id:06d}.png")
        fig.clf()
        plt.close()
        gc.collect()

    def plot_posterior_pdf(self, cbvs):
        """
        Plot the posterior PDF
        """
        fig, axar = plt.subplots(len(cbvs.cbvs.keys()), figsize=(10, 10))
        if len(cbvs.cbvs.keys()) == 1:
            axar = [axar]

        for i, ax in zip(sorted(cbvs.cbvs.keys()), axar):
            for j, pdf in enumerate(self.posterior_pdf[i]):
                label = f"Pr_w={self.prior_pdf_weights[j]}"
                _ = ax.plot(cbvs.theta[i], pdf, label=label)
            # draw a vertical line for the max of each PDF
            _ = ax.axvline(self.prior_peak_theta[i], color='blue', ls='--')
            _ = ax.axvline(self.cond_peak_theta[i], color='red', ls='--')
            _ = ax.axvline(min(self.posterior_peak_theta[i]), color='green', ls='--')
            _ = ax.axvline(max(self.posterior_peak_theta[i]), color='green', ls='--')
            #_ = ax.set_xlabel('Theta')
            #_ = ax.set_ylabel('Arbitrary units')
            #_ = ax.legend()

        fig.tight_layout()
        fig.savefig(f"{self.direc}/posterior_pdfs_star{self.tus_id:06d}.png")
        fig.clf()
        plt.close()
        gc.collect()
