# Cotrendy

A complete rewrite and generalisation of Trendy

# Key components of Cotrendy

### Configuration file

Cotrendy is setup using a configuration file in the [TOML format](https://github.com/toml-lang/toml). Below is an example configuration file. Each parameter is explained using inline comments.

```toml
# This is a TOML config file for TESS Sector 5
[owner]
name = "James McCormac"
version = "0.0.1"

[global]
# working directory
root = "/Users/jmcc/Dropbox/PythonScripts/TESS/mono_example_data"
# time slot identifier, quarter, night etc
timeslot = "S5"
# camera_id
camera_id = "1.1"

[data]
# a file containing times, this should be the same length as each star row below
time_file = "tess_s05_times.pkl"
# a file containing fluxes (1 row per star)
flux_file = "tess_s05_fluxes.pkl"
# a file containing errors on flxues (1 row per star)
error_file = "tess_s05_errors.pkl"
# mask file to exclude cadences from CBV fitting
cadence_mask_file = "S05_mask.fits"
# name of the cbv output pickle file
cbv_file = "tess_s05_cbvs.pkl"
# file with ids of objects considered for CVBs
# if there are many man objects to detrend, this mask can select a subset
# to make the CBVs (brightest, whatever) and then the vectors can be used to
# cotrend everything
objects_mask_file = "objects_mask.pkl"
# reject outliers in the data, as per PLATO outlier rejection?
reject_outliers = false

[catalog]
# Master input catalog
master_cat_file = "S05_1-1.fits"
#s CBV input catalog - these are the stars kept for making the CBVs
# ra, dec, mag, id
input_cat_file = "tess_s05_cat.pkl"

[cotrend]
# maximum number of CBVs to attempt extracting
max_n_cbvs = 8
# SNR limit for significant cbvs, those with lower SNR are excluded
cbv_snr_limit = 5
# CBV fitting method, sequential or simultaneous
#cbv_fit_method = "simultaneous"
cbv_fit_method = "sequential"
# set if we want LS or MAP fitting - NOTE: MAP still needs some work
cbv_mode = "MAP"
# set the normalised variability limit
normalised_variability_limit = 1.3
# take a few test case stars to plot PDFs etc
test_stars = [10,100,1000]
```

### Catalog

The bayesian maximum a posteriori (MAP) method of cotrending requires supplimentary information about the target under study and its neighbours. This is supplied as a catalog of information (currently, RA, Dec, Magnitude).

   1. The location of the input catalog is given in the configuration file under ```{global.root}/{catalog.cat_file}```
   1. Supported formats are fits and text inputs
   1. The catalog is assumed to contain 3 colums, Ra, Dec and Mag in that order
   1. If a text file is supplied it should be space separated columns
   1. If a fits file is supplied the fits table should be in the first fits extension and the column headers must be 'ra', 'dec' and 'mag'

### Photometry

Cotrendy needs times, fluxes for a list of stars and errors on those flux values. To be as agnostic
towards different instruments as possible we expect the input data in the following format.

   1. A pickled 1D numpy array containing the timestamps for the data
   1. A pickled 2D numpy array containing flux measurements, 1 row per star
   1. A pickled 2D numpy array of the same shape as above, containing errors on the flxues

Do whatever you like to extract your data, but present it to Cotrendy as pickled numpy arrays.

Note: you can import the pickling functions to pickle your own numpy arrays using you own data extraction 
tool.

```python
from cotrendy.utils import picklify

# times is your 1D timestamp array
# fluxes and errors are your two 2D flux and error arrays
picklify('times.pkl', times)
picklify('fluxes.pkl', fluxes)
picklify('errors.pkl', errors)
```

I have made a couple of functions in ```cotrendy.INSTRUMENT_NAME``` to help prepare the data in the correct formats.
There are functions currently available for:

   1. NGTS QLP
