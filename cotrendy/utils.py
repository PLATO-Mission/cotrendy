"""
Utility functions for Cotrendy
"""
import pickle
import toml
import numpy as np

# pylint: disable=invalid-name

def load_config(filename):
    """
    Load the trend simulator config file
    """
    return toml.load(filename)

def picklify(pickle_file, pickle_object):
    """
    Take an object and pickle it for later use
    """
    of = open(pickle_file, 'wb')
    pickle.dump(pickle_object, of)
    of.close()

def depicklify(pickle_file):
    """
    Take a pickled file and return the object
    """
    try:
        of = open(pickle_file, 'rb')
        res = pickle.load(of)
        of.close()
    except FileNotFoundError:
        print(f"{pickle_file} not found...")
        res = None
    return res

def find_star_in_list(cat_x, cat_y, tar_x, tar_y):
    """
    Take in catalogue postions and
    a target position and find the
    target index in the list, return also
    it's Euclidean separation
    """
    dx = abs(cat_x - tar_x)
    dy = abs(cat_y - tar_y)
    rad = np.sqrt(dx**2 + dy**2)
    match = np.where(rad == min(rad))[0][0]
    return match, min(rad)
