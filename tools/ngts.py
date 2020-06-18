"""
Code for handling NGTS data
"""
import numpy as np
import cotrendy.utils as cuts

def read_phot_file(phot_file):
    """
    Read in the phot files from process
    """
    # shorten the names of somethings
    nppc = 9
    npoc = 7

    try:
        lcs = pd.read_csv(phot_file, header=None, skiprows=1, delim_whitespace=True)
        n_stars = (len(lcs.columns)-nppc)//npoc
    except FileNotFoundError:
        return None, None, None

    # select the different times and columns of interest
    jd = lcs[1]
    bjd = lcs[2]
    jd0 = int(jd[0])
    bjd = bjd - jd0

    # get all the comparisons into a stackable numpy array
    # make a check on the comparison stars before continuing
    stack, stack_w_sky, stack_max_pix, stack_x, stack_y = [], [], [], [], []
    for i in range(1, n_stars+1):
        stack.append(np.array(lcs[nppc + (npoc*i - 5)]))
        stack_w_sky.append(np.array(lcs[nppc + (npoc*i - 5)]) + np.array(lcs[nppc + (npoc*i - 3)]))
        stack_max_pix.append(np.array(lcs[nppc + (npoc*i - 1)]))
        stack_x.append(np.array(lcs[nppc + (npoc*i - 7)]))
        stack_y.append(np.array(lcs[nppc + (npoc*i - 6)]))

    # only use the good comparisons from here on
    comparisons = np.vstack(stack)
    comparisons_w_sky = np.vstack(stack_w_sky)
    comparisons_max_pix = np.vstack(stack_max_pix)
    stack_x = np.vstack(stack_x)
    stack_y = np.vstack(stack_y)

    # take the error as just the photon and sky level for now
    comparisons_err = np.sqrt(comparisons_w_sky)

    return bjd, comparisons, comparisons_err

def make_pickled_input_files(path):
    """
    Take an NGTS QLP photfile and make the pickled numpy
    arrays expected by Cotrendy
    """

    times, fluxes, errors = read_phot_file(path)

    if times is None or fluxes is None or errors is None:
        print(f"Could not load {}")

    times_file = f"{path}_times.pkl"
    fluxes_file = f"{path}_fluxes.pkl"
    errors_file = f"{path}_errors.pkl"
    cuts.picklify(time_file, times)
    cuts.picklify(fluxes_file, fluxes)
    cuts.picklify(errors_file, errors)
