# Cotrendy

A complete rewrite of generalisation of Trendy

# Key components of Cotrendy

### Configuration file

Cotrendy is setup using a configuration file in the [TOML format](https://github.com/toml-lang/toml). Below is an example configuration file. Each parameter is explained using inline comments.

*TODO ADD EXAMPLE TOML FILE*

### Catalog

The bayesian maximum a posteriori (MAP) method of cotrending requires supplimentary information about the target under study and its neighbours. This is supplied as a catalog of information (currently, RA, Dec, Magnitude).

   1. The location of the input catalog is given in the configuration file under ```catalog.path```
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
from trendier.utils import picklify

# times is your 1D timestamp array
# fluxes and errors are your two 2D flux and error arrays
picklify('times.pkl', times)
picklify('fluxes.pkl', fluxes)
picklify('errors.pkl', errors)
```

I have written tools in ```trendy.lightcurves``` to extract and pickle data from various instruments
(those I've played with). There is currently support for:

   1. The NGTS quick look pipelines (photX.X) files
   1. TESS full frame image photometry from S. Gill
