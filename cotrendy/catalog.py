"""
Catalog components for Cotrendy
"""
import sys
import traceback
import numpy as np
import cotrendy.utils as cuts

# pylint: disable=no-member
# pylint: disable=broad-except
# pylint: disable=invalid-name

class Catalog():
    """
    External information on the targets to detrend
    """
    def __init__(self, config, apply_mask=True):
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
        self.ids = None
        self.load_catalog(config, apply_mask)

    def load_catalog(self, cfg, apply_mask):
        """
        Load the RA, Dec and Gaia G mag from the input catalog
        """
        try:
            path = cfg['global']['root']
            filename = cfg['catalog']['input_cat_file']
        except Exception:
            print("Missing catalog info from configuration file")
            traceback.print_exc(file=sys.stdout)
            sys.exit(1)

        cat = cuts.depicklify(f"{path}/{filename}")
        ra = cat[0]
        dec = cat[1]
        mag = cat[2]
        ids = cat[3]

        # mask the catalog if required
        if apply_mask:
            objects_mask_file = cfg['data']['objects_mask_file']
            mask_file = f"{path}/{objects_mask_file}"
            mask = cuts.depicklify(mask_file)
            ra = ra[mask]
            dec = dec[mask]
            mag = mag[mask]
            ids = ids[mask]

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
        self.ids = ids
