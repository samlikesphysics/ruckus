from ruckus.base import RKHS as _RKHS

import numpy as _np
from numpy.lib.stride_tricks import sliding_window_view as _sliding_window_view

class ConvolutionalRKHS(_RKHS):
    r"""
    Kernels can be applied directly to data, or as a filter to a convolution [1]. 
    This class allows one to convolve an RKHS filter over :math:`N\mathrm{D}` data.

    The dimension of the sample indices is set by the length of the ``window_shape`` 
    and ``stride`` parameters, which must match. For instance, one can take a :math:`2\times 3`
    dimension window over the first two dimensions of the data by setting ``window_shape = (2,3)``
    and ``stride=(1,1)``. After pulling the sliding window data, it is fitted to or transformed by 
    the ``rkhs`` specified by the parameters.

    1. `Mairal, J., Koniusz, P., Harchaoui, Z., Schmid, C. "Convolutional Kernel Networks." Advances in Neural Information Processing Systems 27 (NIPS 2014) <https://papers.nips.cc/paper/2014/hash/81ca0262c82e712e50c580c032d99b60-Abstract.html>`_
    
    ==========
    Parameters
    ==========
    :param window_shape: Default = (2,).
        Specifies the shape of the sliding window to be passed over the first ``len(window_shape)`` axes of the data.
    :type window_shape: ``tuple``
    :param stride: Default = (1,).
        Specifies how many steps the window takes in each direction during the convolution.
    :type stride: ``tuple``
    :param rkhs: Default = ``None``.
        Specifies the :py:class:`RKHS` to be applied to the convolved data. If ``None``, a base :py:class:`RKHS` instance is generated. 
    :type rkhs: :py:class:`RKHS` or ``None``
    :param flatten_samples: Default = ``True``.
        If ``True``, the axes which the window was applied to are flattened after the convolution. Ideal for passing to other :py:class:`RKHS` instances which only recognize one sample dimension.
    :type flatten_samples: ``bool``
    :param flatten_features: Default = ``False``.
        If ``True``, the original features of ``X`` and the new window axes are flattened together.
    :type flatten_features: ``bool``
    :param copy_X: Default = ``True``.
        If ``True``, input ``X`` is copied and stored by the model in the ``X_fit_`` attribute. If no further changes will be done to ``X``, setting ``copy_X=False`` saves memory by storing a reference.
    :type copy_X: ``bool``

    ==========
    Attributes
    ==========
    :param shape_in\_: The required shape of the input datapoints, aka the shape of the domain space :math:`X`.
    :type shape_in\_: ``tuple``
    :param shape_out\_: The final shape of the transformed datapoints, aka the shape of the Hilbert space :math:`H`.
    :type shape_out\_: ``tuple``
    :param X_fit\_: The data which was used to fit the model.
    :type X_fit\_: :py:class:`numpy.ndarray` of shape `(n_samples,)+self.shape_in_`
    """
    def __init__(
        self,
        window_shape=(2,),
        stride=(1,),
        rkhs=None,
        *,
        flatten_samples=True,
        flatten_features=False,
        # Misc.
        copy_X=True,
    ):
        self.window_shape = window_shape
        self.stride = stride
        if rkhs is None:
            self.rkhs = _RKHS()
        else:
            self.rkhs = rkhs
        self.flatten_samples = flatten_samples
        self.flatten_features = flatten_features
        self.copy_X = copy_X

    def _sliding_window(self,X):
        """
        Applies a sliding window to the data and reshapes it according to ``self.flatten_samples`` and ``self.flatten_features``.
        """
        sample_dims = len(self.window_shape)

        stride_inds = _np.ix_(*[_np.arange(0,X.shape[i]-self.window_shape[i]+1,self.stride[i]) for i in range(sample_dims)])
        X_slide = _sliding_window_view(X,
            self.window_shape,
            axis=tuple(range(sample_dims))
        )[stride_inds]

        if self.flatten_samples:
            X_slide = X_slide.reshape((_np.prod(X_slide.shape[:sample_dims],dtype=int),)+X_slide.shape[sample_dims:])
            sample_dims = 1

        if self.flatten_features:
            X_slide = X_slide.reshape(X_slide.shape[:sample_dims]+(_np.prod(X_slide.shape[sample_dims:],dtype=int),))

        return X_slide

    def fit_transform(self,X,y=None):
        """
        Fit the model from data in ``X`` and transform ``X``.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in `self.rkhs.take` and `self.rkhs.filter` after convolution.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``        

        :returns: The transformed data
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_out_``
        """
        if self.copy_X:
            X = X.copy()
        self.X_fit_ = X
        
        convX = self._sliding_window(self.X_fit_)
        self.shape_in_ = self.X_fit_.shape[1:]
        X_transformed = self.rkhs.fit_transform(convX)
        self.shape_out_ = X_transformed.shape[1:]
        return X_transformed
    
    def fit(self,X,y=None):
        """
        Fit the model from data in ``X``.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in `self.rkhs.take` and `self.rkhs.filter` after convolution.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``        

        :returns: The instance itself
        :rtype: :py:class:`RKHS`
        """
        if self.copy_X:
            X = X.copy()
        self.X_fit_ = X

        convX = self._sliding_window(self.X_fit_)
        self.shape_in_ = self.X_fit_.shape[1:]
        self.rkhs.fit(convX)
        self.shape_out_ = self.rkhs.shape_out_
        return self
    
    def transform(self,X,):
        """
        Transform ``X``.

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The transformed data
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_out_``
        """
        if self.copy_X:
            X = X.copy()
        self.X_fit_ = X
        return self.rkhs.transform(self._sliding_window(X))