"""
Light curves components for Cotrendy
"""
import sys
import logging
import numpy as np
from scipy.stats import median_abs_deviation
import cotrendy.utils as cuts

def load_photometry(config, apply_object_mask=True):
    """
    Read in a photometry file

    Parameters
    ----------
    config : dict
        Configuration file loaded via TOML
    apply_object_mask : boolean
        Mask our a subset of stars?

    Returns
    -------
    times : array
        Array of times of observation
    lightcurves : list
        List of Lightcurve objects, one per star

    Raises
    ------
    None
    """
    root = config['global']['root']
    time_file = config['data']['time_file']
    flux_file = config['data']['flux_file']
    error_file = config['data']['error_file']

    times = cuts.depicklify(f"{root}/{time_file}")
    if times is None:
        logging.critical(f"Could not load {root}/{time_file}...")
        sys.exit(1)
    fluxes = cuts.depicklify(f"{root}/{flux_file}")
    if fluxes is None:
        logging.critical(f"Could not load {root}/{flux_file}...")
        sys.exit(1)
    errors = cuts.depicklify(f"{root}/{error_file}")
    if errors is None:
        logging.critical(f"Could not load {root}/{error_file}...")
        sys.exit(1)

    if fluxes.shape != errors.shape or len(times) != len(fluxes[0]):
        logging.critical("Data arrays have mismatched shapes...")
        sys.exit(1)

    # now apply the mask if needed
    if apply_object_mask:
        objects_mask_file = config['data']['objects_mask_file']
        mask = cuts.depicklify(f"{root}/{objects_mask_file}")
        fluxes = fluxes[mask]
        errors = errors[mask]

    # now make list of Lightcurves objects
    lightcurves = []
    n_stars = len(fluxes)
    i = 0
    for star, star_err in zip(fluxes, errors):
        logging.info(f"{i+1}/{n_stars}")
        lightcurves.append(Lightcurve(star, star_err, config['data']['reject_outliers']))
        i += 1

    return times, lightcurves

class Lightcurve():
    """
    Lightcurve object of real object
    """
    def __init__(self, flux, flux_err, filter_outliers=False):
        """
        Initialise the class

        Parameters
        ----------
        flux : array-like
            list of flux values
        flux_err : array-like
            list of flux error values
        filter_outliers : boolean
            turn on PLATO outlier rejection?
            default = False

        Returns
        -------
        None

        Raises
        ------
        None
        """
        # Initialise variables to hold data when trend is applied
        self.flux_wtrend = flux
        self.fluxerr_wtrend = flux_err
        self.median_flux = np.median(flux)
        self.outlier_indices = None
        # store the lightcurve after removing outliers
        if filter_outliers:
            self.filter_outliers()

    def filter_outliers(self, alpha=5, beta=12):
        """
        Filter out data points that are > alpha*local MAD
        within a window ±beta around a given data point.
        Replace the data point with the local median
        as to not introduce gaps

        Parameters
        ----------
        alpha : int
            Scaling factor for number of MADs to reject outside
        beta : int
            Half width of sliding window for MAD rejection

        Returns
        -------
        None

        Outliers indices are included in self.outlier_indices

        Raises
        ------
        None
        """
        # could imaging this having a voting system where each beta*2+1 slice
        # votes on an outlier and if >N votes it gets nuked
        outlier_indices = []
        for i in np.arange(beta, len(self.flux_wtrend)-beta-1):
            window = self.flux_wtrend[i-beta: i+beta+1]
            med = np.median(window)
            mad = median_abs_deviation(window)
            outlier_positions = np.where(((window >= med+alpha*mad) |
                                          (window <= med-alpha*mad)))[0] + i - beta

            # gather them up and then correct them with a median
            # window centered on them
            for outlier_position in outlier_positions:
                if outlier_position not in outlier_indices:
                    outlier_indices.append(outlier_position)

        # now go back and fix the outliers
        for outlier in outlier_indices:
            lower = outlier-beta
            upper = outlier+beta+1
            if lower < 0:
                lower = 0
            if upper > len(self.flux_wtrend):
                upper = len(self.flux_wtrend)
            med = np.median(self.flux_wtrend[lower:upper])
            self.flux_wtrend[outlier] = med

        self.outlier_indices = outlier_indices
