"""
Catalog components for Cotrendy
"""
import sys
import traceback
import numpy as np
from astropy.io import fits

# pylint: disable=no-member
# pylint: disable=broad-except
# pylint: disable=invalid-name

class Catalog():
    """
    External information on the targets to detrend
    """
    def __init__(self, config):
        """
        Initialise the class

        Parameters
        ----------

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.ra = None
        self.dec = None
        self.mag = None
        self.load_catalog(config)

    def load_catalog(self, cfg):
        """
        Load the RA, Dec and Gaia G mag from the input catalog
        """
        try:
            path = cfg['global']['root']
            filename = cfg['catalog']['cat_file']
            frmt = cfg['catalog']['format']
        except Exception:
            print("Missing catalog.path or catalog.format from configuration file")
            traceback.print_exc(file=sys.stdout)
            sys.exit(1)

        if frmt == "fits":
            print(f"Loading fits catalog {path}/{filename}...")
            try:
                with fits.open(f"{path}/{filename}") as ff:
                    ra = ff[1].data['ra']
                    dec = ff[1].data['dec']
                    mag = ff[1].data['mag']
            except Exception:
                print("Catalog loading failed, exiting...")
                traceback.print_exc(file=sys.stdout)
                sys.exit(1)
        else:
            print(f"Loading text catalog {path}/{filename}")
            try:
                ra, dec, mag = np.loadtxt(f"{path}/{filename}", usecols=[0, 1, 2], unpack=True)
            except Exception:
                print("Catalog loading failed, exiting...")
                traceback.print_exc(file=sys.stdout)
                sys.exit(1)

        # check if RA spans 0h
        min_ra = np.min(ra)
        max_ra = np.max(ra)
        diff_ra = max_ra - min_ra
        if diff_ra > 180:
            ra[ra > 180] -= 360

        # store these for further use
        self.ra = ra
        self.dec = dec
        self.mag = mag
