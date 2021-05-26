# Cotrendy

A complete rewrite and generalisation of Trendy

# Key components of Cotrendy

### Configuration file

Cotrendy is setup using a configuration file in the [TOML format](https://github.com/toml-lang/toml). Below is an example configuration file. Each parameter is explained using inline comments.

```toml
# This is a TOML config file for c01_q00
[owner]
name = "James McCormac"
version = "0.1.0"

[global]
# enable debugging mode?
# default = true
debug = true
# working directory
root = "/ngts/scratch/jmcc/psls_simulations_2021/BOL4_2k_bm"
# time slot identifier, quarter, night etc
timeslot = "c01_q00"
# camera_id
camera_id = "c01"

[data]
# a file containing times, this should be the same length as each star row below
time_file = "c01_q00_times.pkl"
# a file containing fluxes (1 row per star)
flux_file = "c01_q00_fluxes.pkl"
# a file containing errors on flxues (1 row per star)
error_file = "c01_q00_errors.pkl"
# normalised variability file (1 value per star). This is used to override internal
# normalised variability calculations
variability_file = ""
# mask file to exclude cadences from CBV fitting
# default = ""
cadence_mask_file = ""
# name of the output cbv pickle file
cbv_file = "c01_q00_cbvs.pkl"
# file with ids of objects considered for CVBs
objects_mask_file = "c01_objects_mask.pkl"
# reject outliers in the data, as per PLATO outlier rejection?
# default = false
reject_outliers = false

[catalog]
# CBV input catalog. Format is currently ra, dec, mag, id
input_cat_file = "c01_catalog.pkl"
# coords units, are they in degrees for ra/dec or pix for X/Y, "radec" or "pix"
# default = "radec"
coords_units = "pix"
# MAP weights for ra, dec and mag
# default = [1, 1, 2]
dim_weights = [1, 1, 2]

[cotrend]
# number of workers in multiprocessing pool
# default = 2
pool_size = 2
# maximum number of CBVs to attempt extracting
# default = 12
max_n_cbvs = 12
# SNR limit for significant cbvs, those with lower SNR are excluded
# default = 5
cbv_snr_limit = 5
# set if we want LS or MAP fitting
# default = "LS"
cbv_mode = "LS"
# set the variability normalisation order, this is the order of coarse detrend
# when determining the normalised variability for each star in a set
# default = 3
variability_normalisation_order = 3
# set the normalised variability limit
# default = 1.3
normalised_variability_limit = 1.3
# correlation threshold, the top fraction of correlated stars for CBVs
# default = 0.5
correlation_threshold = 0.5                                           
# entropy cleaning threshold   
# default = -0.7
entropy_threshold = -0.7         
# entropy cleaning rejection limit - how many stars can be rejected?
# set a -ve number for no limit
# default = -1
max_entropy_rejections = -1
# set the normalised variability limit below which priors are not used
# default = 0.7
prior_normalised_variability_limit = 0.7
# enable or disable snapping of posterior to conditional if within prior_sigma of prior
# default = false
prior_cond_snapping = false
# prior raw goodness weight, emperical parameter from PDC pipeline see eq. 19 Smith et al. 2012
# default = 5.0
prior_raw_goodness_weight = 5.0
# prior raw goodness exponent from PDC pipeline see eq. 19 Smith et al. 2012
# default = 3.0
prior_raw_goodness_exponent = 3.0
# prior noise weight (extra param, post Smith 2012)
# default = 0.0002
prior_noise_goodness_weight = 0.0002
# scaling parameter for the variability part of the prior weight calculation
# default = 2.0
prior_pdf_variability_weight = 2.0
# gain for the goodness part of prior pdf weighting
# default = 1.0
prior_pdf_goodness_gain = 1.0
# weight for the goodness part of prior pdf weighting
# default = 0.5
prior_pdf_goodness_weight = 0.5
# take a few test case stars to plot PDFs etc
# default = []
test_stars = []
```

### Catalog

The bayesian maximum a posteriori (MAP) method of cotrending requires supplimentary information about the target under study and its neighbours. This is supplied as a catalog of information (currently, RA, Dec, Magnitude).

   1. The location of the input catalog is given in the configuration file under ```{global.root}/{catalog.input_cat_file}```
   1. The catalog should be a pickled numpy array with 4 rows:
      1. Ra
      1. Dec
      1. Mag
      1. Id

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
