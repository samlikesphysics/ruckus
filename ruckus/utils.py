import numpy as _np
from sklearn.preprocessing import KernelCenterer as _KernelCenterer
        
class _DecoyCenterer(_KernelCenterer):
    r"""
    Does nothing. Meant to be called in child classes of PCA algorithms
    where centering the data is not desired.
    """

    def fit(self, K, y=None):
        """
        Fits nothing.
        """
        return self

    def transform(self, K, copy=True):
        """
        Does not center K.
        """
        return K