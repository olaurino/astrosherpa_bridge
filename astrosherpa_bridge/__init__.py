# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

<<<<<<< HEAD
from main import SherpaFitter, SherpaMCMC, Stat, OptMethod, EstMethod, Dataset, ConvertedModel
=======
from main import SherpaFitter, SherpaMCMC, Stat, OptMethod, EstMethod
>>>>>>> b70820ceda946bb9d61bda09ee745308050a94ba
# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    pass


