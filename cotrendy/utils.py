"""
Utility functions for Cotrendy
"""
import sys
import logging
import traceback
import pickle
import toml
import numpy as np

# pylint: disable=invalid-name

def load_config(filename):
    """
    Load the trend simulator config file

    Parameters
    ----------
    filename : string
        Name of the configuration file to load

    Returns
    -------
    configuration : dict
        Configuration information

    Raises
    ------
    None
    """
    return toml.load(filename)

def picklify(pickle_file, pickle_object):
    """
    Take an object and pickle it for later use

    Parameters
    ----------
    pickle_file : string
        Path to pickled output file
    pickle_object : misc
        Item to pickle

    Returns
    -------
    None

    Raises
    ------
    None
    """
    of = open(pickle_file, 'wb')
    pickle.dump(pickle_object, of, protocol=4)
    of.close()

def depicklify(pickle_file):
    """
    Take a pickled file and return the object

    Parameters
    ----------
    pickle_file : string
        Path to pickled file

    Returns
    -------
    res : misc
        Unpickled contents of pickled file

    Raises
    ------
    None
    """
    try:
        of = open(pickle_file, 'rb')
        res = pickle.load(of)
        of.close()
    except FileNotFoundError:
        logging.warning(f"{pickle_file} not found...")
        traceback.print_exc(file=sys.stdout)
        res = None
    return res

def find_star_in_list(cat_x, cat_y, tar_x, tar_y):
    """
    Take in catalogue postions and
    a target position and find the
    target index in the list, return also
    it's Euclidean separation

    Parameters
    ----------
    cat_x : array
        array of catalog x positions
    cat_y : array
        array of catalog y positions
    tar_x : float
        target x position
    tar_y : float
        target y position

    Returns
    -------
    match : int
        Array index of matching object
    rad L float
        Radius of separation between target and closest match

    Raises
    ------
    None
    """
    dx = abs(cat_x - tar_x)
    dy = abs(cat_y - tar_y)
    rad = np.sqrt(dx**2 + dy**2)
    match = np.where(rad == min(rad))[0][0]
    return match, min(rad)
