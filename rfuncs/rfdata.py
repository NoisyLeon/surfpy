# -*- coding: utf-8 -*-
"""
ASDF for retrieving receiver function data
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
try:
    import surfpy.refuncs.rfbase as rfbase
except:
    import rfbase


import numpy as np
import obspy

class dataASDF(rfbase.baseASDF):
    """ Class for data retrieval 
    =================================================================================================================

    =================================================================================================================
    """
    