talib_wrappers
==============

ta-lib has a nice abstraction layer
(http://ta-lib.org/d_api/d_api.html#Abstraction). This library uses the
abstraction layer to expose all of the TA-lib functions to Python.


The Python wrappers simplify the usage of the functions by taking care of all
the allocation process for outputs, input checks, etc. It makes an efficient
use of NumPy arrays all over the place.
