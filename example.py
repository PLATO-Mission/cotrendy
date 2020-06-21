"""
Example layout of using Cotrendy
"""
import gc
import argparse as ap
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import cotrendy.utils as cuts
import cotrendy.lightcurves as tlc
from cotrendy.catalog import Catalog
from cotrendy.cbvs import CBVs

# pylint: disable=invalid-name

# TODO: check the weighting is ok and the prior is being used when expected?

def arg_parse():
    """
    Parse the command line arguments
    """
    p = ap.ArgumentParser()
    p.add_argument('config',
                   help='path to config file')
    return p.parse_args()

if __name__ == "__main__":
    # load the command line arguments
    args = arg_parse()

    # load the configuration
    config = cuts.load_config(args.config)
    camera_id = config['global']['camera_id']

    # grab the locations of the data
    root = config['global']['root']
    cbv_pickle_file = config['data']['cbv_file']
    cbv_pickle_file_output = f"{root}/{cbv_pickle_file}"
    cbv_fit_method = config['cotrend']['cbv_fit_method']

    # load the external catalog
    catalog = Catalog(config)

    # check if we have the cbv pickle file
    print(f"Looking for pickle file {cbv_pickle_file_output}...")
    cbvs = cuts.depicklify(f"{cbv_pickle_file_output}")
    # if there is no pickle file, extract the CBVs from scratch
    if cbvs is None:
        print(f"Pickle file {cbv_pickle_file_output} not found, doing detrending from scratch...")

        # step 1, load the photometry, this is done here and not prior to
        # the check for CBVs because the flux array is pickled with the CBVs
        times, lightcurves = tlc.load_photometry(config)

        # create a CBVs object for our targets, we want the top n_CBVs
        cbvs = CBVs(config, times, lightcurves)
        # pickle the intermediate CBVs object incase it crashes later
        cuts.picklify(cbv_pickle_file_output, cbvs)

        # calculate the basis vectors
        cbvs.calculate_cbvs()
        # pickle the intermediate CBVs object incase it crashes later
        cuts.picklify(cbv_pickle_file_output, cbvs)

        # work out the fit coefficients, needed for the Prior PDF
        # calculate them either simultaneously for all CBVs or sequentially
        # from the first to last
        if cbv_fit_method == "sequential":
            cbvs.calculate_robust_fit_coeffs_sequen()
        else:
            cbvs.calculate_robust_fit_coeffs_simult()
        # pickle the intermediate CBVs object incase it crashes later
        cuts.picklify(cbv_pickle_file_output, cbvs)

        # cotrend the data using the MAP method
        cbvs.cotrend_data(catalog)
        # pickle the intermediate CBVs object incase it crashes later
        cuts.picklify(cbv_pickle_file_output, cbvs)

        # this extra stuff needs baked in somehwere
        tic_file = '/Users/jmcc/Dropbox/PythonScripts/TESS/mono_example_data/tic_ids_kept.pkl'
        tic_ids_kept = cuts.depicklify(tic_file)

        # loop over the light curves and plot the raw and corrected versions
        for i in range(0, len(cbvs.norm_flux_array)):
            outfile = f"{root}/{tic_ids_kept[i].split('.')[0]}"
            fig, ax = plt.subplots(2, figsize=(10, 10), sharex=True, sharey=True)
            ax[0].plot(cbvs.norm_flux_array[i], 'g.')
            ax[0].set_ylabel('Flux norm')
            ax[1].plot(cbvs.cotrended_flux_array[i], 'r.')
            ax[1].set_xlabel('Image number')
            ax[1].set_ylabel('Flux norm')
            fig.tight_layout()
            fig.savefig(outfile)
            fig.clf()
            plt.close()
            gc.collect()
