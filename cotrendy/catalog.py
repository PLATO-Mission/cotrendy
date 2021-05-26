"""
Catalog components for Cotrendy
"""
import sys
import logging
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
    def __init__(self, config, apply_object_mask=True):
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
        self.coords_units = None
        self.ra = None
        self.dec = None
        self.mag = None
        self.ids = None
        self.ra_weight = float(config['catalog']['dim_weights'][0])
        self.dec_weight = float(config['catalog']['dim_weights'][1])
        self.mag_weight = float(config['catalog']['dim_weights'][2])
        self.load_catalog(config, apply_object_mask)

    def load_catalog(self, cfg, apply_object_mask):
        """
        Load the RA, Dec and Gaia G mag from the input catalog
        """
        try:
            path = cfg['global']['root']
            filename = cfg['catalog']['input_cat_file']
            # this can be pixels for CCD coords or degrees for RA/Dec
            # if pixels, ra is X and dec is Y by default
            # the actual units don't matter, but the RA correction below
            # is wrong when pixels, so we distinuish them
            coords_units = cfg['catalog']['coords_units']
        except Exception:
            logging.error("Missing catalog info from configuration file")
            traceback.print_exc(file=sys.stdout)
            sys.exit(1)

        cat = cuts.depicklify(f"{path}/{filename}")
        ra = cat[0]
        dec = cat[1]
        mag = cat[2]
        ids = cat[3]

        # mask the catalog if required
        if apply_object_mask:
            objects_mask_file = cfg['data']['objects_mask_file']
            mask_file = f"{path}/{objects_mask_file}"
            mask = cuts.depicklify(mask_file)
            ra = ra[mask]
            dec = dec[mask]
            mag = mag[mask]
            ids = ids[mask]

        if coords_units == "deg":
            # check if RA spans 0h
            min_ra = np.min(ra)
            max_ra = np.max(ra)
            diff_ra = max_ra - min_ra
            if diff_ra > 180:
                ra[ra > 180] -= 360
        else:
            # do soemthing special to pixels coords, if necessary
            pass # for now

        # store these for further use
        self.ra = ra
        self.dec = dec
        self.mag = mag
        self.ids = ids
