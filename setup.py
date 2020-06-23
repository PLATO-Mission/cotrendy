"""
cotrendy setup.py file

Inspiration is taken from the donuts package setup
"""
from setuptools import setup

# Get some values from the setup.cfg
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser
    conf = ConfigParser()
    conf.read(['setup.cfg'])
    metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata.get('package_name', 'packagename')
DESCRIPTION = metadata.get('description', '')
AUTHOR = metadata.get('author', '')
AUTHOR_EMAIL = metadata.get('author_email', '')
LICENSE = metadata.get('license', '')
URL = metadata.get('url', 'https://github.com/PLATO-Mission/cotrendy')
LONG_DESCRIPTION = open('README.md').read()

from cotrendy_version import VERSION, RELEASE

# add sphinx build_docs integration
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

setup(name=PACKAGENAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=['cotrendy',],
    install_requires=['numpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    long_description=LONG_DESCRIPTION,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Astronomy',
    ]
)
