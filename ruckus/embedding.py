from ruckus.utils import _DecoyCenterer
from ruckus.base import RKHS as _RKHS
import numpy as _np
from sklearn.decomposition import KernelPCA as _KernelPCA
from sklearn.preprocessing import KernelCenterer as _KernelCenterer
from sklearn.cluster import KMeans as _KMeans
from sklearn.metrics.pairwise import pairwise_kernels as _pairwise_kernels
from sklearn.utils.validation import check_is_fitted as _check_is_fitted
import scipy.stats as _st


# SPECIFIC RKHS'S

## EigenRKHS
class EigenRKHS(_KernelPCA,_RKHS):
    r"""
    ``EigenRKHS`` is a child class of :py:class:`sklearn.decomposition._kernel_pca.KernelPCA`, 
    which adapts it to our :py:class:`RKHS` class formula, allowing interactivity with other
    RKHS's. We also add new options regarding centering and Nyström sampling for efficiency.
    Because of this dependency, our code and documentation inherits notably from that of ``KernelPCA``,
    particularly in methods where only minor revisions were made.

    ``EigenRKHS`` is initialized with a kernel :math:`k(x,y)`---which now defaults to a Gaussian RBF---and 
    computes the eigenvector decomposition :math:`k(x,y) = \sum_a \lambda_a \phi_a(x)\phi_a(x)` 
    to determine the
    feature mappings :math:`\phi(x)` into the Hilbert space :math:`\mathcal{H}`. Because computing the
    eigenvectors scales cubically with the number of samples, we have added options for Nyström sampling,
    which selects a smaller subset of the data to use for the eigenvector computation, and then uses those
    eigenvectors to transform the remaining data [1].

    1. `Williams, C., Seeger, M. "Using the Nyström Method to Speed Up Kernel Machines." Advances in Neural Information Processing Systems 13 (NIPS 2000) <https://papers.nips.cc/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html>`_
    
    ==========
    Parameters
    ==========

    :param use_kernel: Default = ``"rbf"``.
        See :py:class:`sklearn.decomposition._kernel_pca.KernelPCA` for kernel options.
    :type use_kernel: ``str`` or ``callable``
    :param gamma: Default = ``None``. Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other kernels. If ``gamma`` is ``None``, then it is set to ``1/n_features``.
    :type gamma: ``float``
    :param degree: Default = 3. Degree for poly kernels. Ignored by other kernels.
    :type degree: ``int``
    :param coef0: Default = 1. Independent term in poly and sigmoid kernels. Ignored by other kernels.
    :type coef0: ``float``
    :param kernel_params: Default = ``None``. Parameters (keyword arguments) and values for kernel passed as callable object. Ignored by other kernels.
    :type kernel_params: ``dict``
    :param n_jobs: Default = ``None``. Number of parallel jobs to run. See :py:class:`sklearn.decomposition._kernel_pca.KernelPCA` for details.
    :type n_jobs: ``int``
    :param n_nystrom_samples: Default = ``1.0``. The number of samples to draw from ``X`` to compute the SVD. If ``int``, then draw ``n_nystrom_samples`` samples. If float, then draw ``n_nystrom_samples * X.shape[0]`` samples.
    :type n_nystrom_samples: ``int`` or ``float``
    :param sample_method: Default = ``"random"``. How to draw the Nyström samples. If ``"random"``, then subsample randomly with replacement. If ``"kmeans"``, then find the ``n_nystrom_samples`` optimal means.
    :type sample_method: ``str``
    :param sample_iter: Default = 300. If ``sample_method = "kmeans"``, the number of times to iterate the algorithm.
    :param n_components: Default = ``None``. Number of components. If None, all non-zero components are kept.
    :type n_components: ``int``
    :param centered: Default = ``False``. Whether to center the kernel before computing the SVD. This must be ``False`` for embeddings of distributions to be valid.
    :param eigen_solver: Default = ``"auto"``. Solver to use for eigenvector computation. See :py:class:`sklearn.decomposition._kernel_pca.KernelPCA` for details.
    :type eigen_solver: {``"auto"``, ``"dense"``, ``"arpack"``, ``"randomized"``}
    :param tol: Default = 0. Convergence tolerance for arpack. If 0, optimal value will be chosen by arpack.
    :type tol: ``float``
    :param max_iter: Default = ``None``. Maximum number of iterations for arpack. If None, optimal value will be chosen by arpack.
    :type max_iter: ``int``
    :param iterated_power: Default = ``"auto"``. Number of iterations for the power method computed by ``svd_solver == "randomized"``. When ``"auto"``, it is set to 7 when ``n_components < 0.1 * min(X.shape)``, other it is set to 4.
    :type iterated_power: ``int >= 0`` or ``"auto"``
    :param remove_zero_eig: Default = ``False``.
        If True, then all components with zero eigenvalues are removed, so that the number of components in the output may be < n_components (and sometimes even zero due to numerical instability). When n_components is None, this parameter is ignored and components with zero eigenvalues are removed regardless.
    :type remove_zero_eig: ``bool``
    :param random_state: Used when eigen_solver == "arpack" or "randomized". :py:class:`sklearn.decomposition._kernel_pca.KernelPCA` for more details.
    :type random_state: ``int``
    :param take: Default = ``None``.
        Specifies which values to take from the datapoint for transformation.
        If ``None``, the entire datapoint will be taken in its original shape.
        If ``bool`` array, acts as a mask setting values marked ``False`` to ``0`` and leaving values marked True unchanged.
        If ``int`` array, the integers specify the indices (along the first feature dimension) which are to be taken, in the order/shape of the desired input.
        If ``tuple`` of ``int`` arrays, allows for drawing indices across multiple dimensions, similar to passing a ``tuple`` to a ``numpy`` array.
    :type take: :py:class:`numpy.ndarray` of ``dtype int`` or ``bool``, or ``tuple`` of :py:class:`numpy.ndarray` instances of type ``int``, or ``None``
    :param filter: Default = ``None``.
        Specifies a linear preprocessing of the data. Applied after take.
        If ``None``, no changes are made to the input data.
        If the same shape as the input datapoints, ``filter`` and the datapoint are multiplied elementwise. 
        If ``filter`` has a larger dimension than the datapoint, then its first dimensions will be contracted with the datapoint via :py:func:`numpy.tensordot`. The final shape is determined by the remaining dimensions of filter.
    :type filter: :py:class:`numpy.ndarray` of ``dtype float`` or ``None``
    :param copy_X: Default = ``True``.
        If ``True``, input ``X`` is copied and stored by the model in the ``X_fit_`` attribute. If no further changes will be done to ``X``, setting ``copy_X=False`` saves memory by storing a reference.
    :type copy_X: ``bool``

    ==========
    Attributes
    ==========

    :param eigenvalues\_: Eigenvalues of the centered kernel matrix in decreasing order. If ``n_components`` and ``remove_zero_eig`` are not set, then all values are stored.
    :type eigenvalues\_: :py:class:`numpy.ndarray` of shape ``(n_components,)``
    :param eigenvectors\_: Eigenvectors of the kernel matrix. If ``n_components`` and ``remove_zero_eig`` are not set, then all components are stored.
    :type eigenvectors\_: :py:class:`numpy.ndarray` of shape ``(n_samples,n_components)``
    :param shape_in\_: The required shape of the input datapoints, aka the shape of the domain space :math:`X`.
    :type shape_in\_: ``tuple``
    :param shape_out\_: The final shape of the transformed datapoints, aka the shape of the Hilbert space :math:`H`.
    :type shape_out\_: ``tuple``
    :param X_fit\_: The data which was used to fit the model.
    :type X_fit\_: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_in_``
    :param X_nys\_: The nystrom subsamples of the data used to fit the model.
    :type X_nys\_: :py:class:`numpy.ndarray` of shape ``(n_nystrom_samples,n_features_in_)``
    :param n_features_in\_: The size of the features after preprocessing.
    :type n_features_in\_: ``int``
    """
    def __init__(
        self,
        use_kernel="rbf",
        *,
        # Data/Kernel options
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=None,
        # Nyström options
        n_nystrom_samples = 1.0,
        sample_method = 'random',
        sample_iter = 300,
        # SVD options
        n_components=None,
        centered = False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        random_state=None,
        # Misc.
        take=None,
        filter=None,
        copy_X=True,
    ):
        self.use_kernel = use_kernel
        
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

        self.n_nystrom_samples = n_nystrom_samples
        self.sample_iter = sample_iter
        self.sample_method = sample_method

        self.n_components = n_components
        self.centered = centered
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state

        self.fit_inverse_transform = False
        
        self.take = take
        self.filter = filter
        self.copy_X = copy_X

    def _get_kernel(self, X, Y=None):
        """
        Only difference with :py:func:`sklearn.decomposition._kernel_pca.KernelPCA._get_kernel` is that the ``metric`` is ``self.use_kernel`` and not ``self.kernel``.
        """
        if callable(self.use_kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return _pairwise_kernels(
            X, Y, metric=self.use_kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    def kernel(self, X, Y=None):
        """
        Applies ``self.take`` and ``self.filter`` to data, then
        calls :py:func:`_get_kernel` for kernel evaluation.

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the number of features. Must be consistent with preprocessing instructions in ``self.take`` and ``self.factors``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :param Y: Default = ``None``.
            Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in ``self.take`` and ``self.factors``. If ``None``, ``X`` is used.
        :type Y: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The matrix ``K[i,j] = k(X[i],Y[j])`` 
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples_1,n_samples_2)``
        """
        X = self._validate_data(X, accept_sparse="csr", copy=self.copy_X,allow_nd=True,)
        X = self._apply_filter(X)
        X = X.reshape(X.shape[0],_np.prod(X.shape[1:],dtype=int))
        if Y is None:
            Y = X
        else:
            Y = self._validate_data(Y, accept_sparse="csr", copy=self.copy_X,allow_nd=True,)
            Y = self._apply_filter(Y)
            Y = Y.reshape(Y.shape[0],_np.prod(Y.shape[1:],dtype=int))

        return self._get_kernel(X, Y=Y)
    
    def _get_nys_samples(self, X):
        """
        Takes Nyström subsample of data: random samples with replacement if ``self.sample_method == "random"`` and optimal means if ``self.sample_method == "kmeans"``.
        """
        if isinstance(self.n_nystrom_samples,int):
            n_nystrom_samples = self.n_nystrom_samples
        elif self.n_nystrom_samples<=1.0:
            n_nystrom_samples = int(self.n_nystrom_samples*X.shape[0])
        else:
            raise ValueError("n_nystrom_samples must be either an integer or between 0 and 1.")
            
        if self.sample_method is None:
            self.sample_method == 'random'
        
        if self.sample_method == 'random':
            rng = _np.random.default_rng()
            inds = _np.sort(rng.permutation(X.shape[0])[0:n_nystrom_samples])
            X_samples = X[inds]
        elif self.sample_method == 'kmeans':
            kmeans = _KMeans(n_clusters=n_nystrom_samples,max_iter=self.sample_iter).fit(X)
            X_samples = kmeans.cluster_centers_

        return X_samples

    def fit(self, X, y=None):
        """
        Fit the model from data in ``X``. This method filters the data, determines whether it is to be centered, and takes the specified Nyström subsamples. 
        After this, :py:func:`sklearn.decomposition._kernel_pca.KernelPCA._fit_transform` is invoked.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in ``self.take`` and ``self.filter``. Final filtered data will be flattened on the feature axes.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``

        :param y: Not used, present for API consistency by convention.
        :type y: Ignored            

        :returns: The instance itself
        :rtype: :py:class:`RKHS`
        """
        self.X_fit_ = self._validate_data(X, accept_sparse="csr", copy=self.copy_X,allow_nd=True,)
        self.shape_in_ = self.X_fit_.shape[1:]
        X = self._apply_filter(self.X_fit_)
        X = X.reshape(X.shape[0],_np.prod(X.shape[1:],dtype=int))
        self.n_features_in_ = X.shape[1]

        if self.centered:
            self._centerer = _KernelCenterer()
        else:
            self._centerer = _DecoyCenterer()

        self.X_nys_ = self._get_nys_samples(X)
        K = self._get_kernel(self.X_nys_)
        self._fit_transform(K)
    
        self.shape_out_ = self.eigenvalues_.shape
        return self

    def transform(self, X):
        """
        Transform ``X``. This differs from :py:func:`sklearn.decomposition._kernel_pca.KernelPCA.transform` in the data preprocessing.

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The transformed data
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_out_``
        """
        _check_is_fitted(self)
        X = self._validate_data(X, accept_sparse="csr", copy=self.copy_X,allow_nd=True,)
        X = self._apply_filter(X)
        X = X.reshape(X.shape[0],_np.prod(X.shape[1:],dtype=int))
        #return super().transform(X)
        # Compute centered gram matrix between X and training data X_fit_
        K = self._centerer.transform(self._get_kernel(X, self.X_nys_))

        # scale eigenvectors (properly account for null-space for dot product)
        non_zeros = _np.flatnonzero(self.eigenvalues_)
        scaled_alphas = _np.zeros_like(self.eigenvectors_)
        scaled_alphas[:, non_zeros] = self.eigenvectors_[:, non_zeros] / _np.sqrt(
            self.eigenvalues_[non_zeros]
        )

        # Project with a scalar product between K and the scaled eigenvectors
        return _np.dot(K, scaled_alphas)

    def fit_transform(self,X,y=None):
        """
        Fit the model from data in ``X`` and transform ``X``.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in ``self.take`` and ``self.filter``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``        

        :returns: The transformed data
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_out_``
        """
        self.fit(X, y)
        X_transformed = self.transform(X)
        return X_transformed

## RandomFourier
class RandomFourierRBF(_RKHS):
    r"""
    ``RandomFourierRBF`` generates an embedding map :math:`\phi:X\rightarrow H` by constructing random Fourier
    phase signals; that is,

    .. math::

        \phi(x) = \frac{1}{\sqrt{K}}\begin{bmatrix}
            e^{i x\cdot w_1} \\
            \vdots \\
            e^{i x\cdot w_K}
        \end{bmatrix}

    where :math:`K` is the specified ``n_components`` and :math:`(w_1,\dots,w_K)` is drawn from a multivariate
    normal with covariance matrix :math:`\mathrm{diag}(\gamma,\dots,\gamma)`. The result that the kernel
    :math:`k(x,y) = \left<\phi(x),\phi(y)\right>` is approximately a Gaussian RBF with scale parameter :math:`\gamma` [1].

    Rather than drawing a truly random set of phase vectors (which converges :math:`O(n^{-1/2})`)
    we use quasi-Monte Carlo sampling via :py:func:`scipy.stats.qmc.QMCEngine`, which converges :math:`O((\log n)^d n^{-1})`
    where :math:`d` corresponds to the number of features in :math:`X`.

    1. `Rahimi, A., Recht, B. "Random Features for Large-Scale Kernel Machines." Advances in Neural Information Processing Systems 20 (NIPS 2007) <https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html>`_
    
    ==========
    Parameters
    ==========
    
    :param n_components: Default = 100. The number of random Fourier features to generate.
    :type n_components: ``int``
    :param gamma: Default = ``None``. Specifies the scale parameter of the Gaussian kernel to be approximated. If ``None``, set to ``1/n_features``.
    :type gamma: ``float``
    :param complex: Default = ``False``. If ``False``, the output vector has shape ``(n_samples,2*n_components)``, where real and imaginary parts are written in pairs.
    :type complex: ``bool``
    :param engine: Default = ``None``. The sampler class to use. If ``None``, set to :py:func:`scipy.stats.qmc.Sobol`.
    :type engine: child class of :py:func:`scipy.stats.qmc.QMCEngine`
    :param engine_params: Default = ``None``. Initialization parameters to use for ``engine``.
    :type engine_params: ``dict``
    :param take: Default = ``None``.
        Specifies which values to take from the datapoint for transformation.
        If ``None``, the entire datapoint will be taken in its original shape.
        If ``bool`` array, acts as a mask setting values marked ``False`` to ``0`` and leaving values marked True unchanged.
        If ``int`` array, the integers specify the indices (along the first feature dimension) which are to be taken, in the order/shape of the desired input.
        If ``tuple`` of ``int`` arrays, allows for drawing indices across multiple dimensions, similar to passing a ``tuple`` to a ``numpy`` array.
    :type take: :py:class:`numpy.ndarray` of ``dtype int`` or ``bool``, or ``tuple`` of :py:class:`numpy.ndarray` instances of type ``int``, or ``None``
    :param filter: Default = ``None``.
        Specifies a linear preprocessing of the data. Applied after take.
        If ``None``, no changes are made to the input data.
        If the same shape as the input datapoints, ``filter`` and the datapoint are multiplied elementwise. 
        If ``filter`` has a larger dimension than the datapoint, then its first dimensions will be contracted with the datapoint via :py:func:`numpy.tensordot`. The final shape is determined by the remaining dimensions of filter.
    :type filter: :py:class:`numpy.ndarray` of ``dtype float`` or ``None``
    :param copy_X: Default = ``True``.
        If ``True``, input ``X`` is copied and stored by the model in the ``X_fit_`` attribute. If no further changes will be done to ``X``, setting ``copy_X=False`` saves memory by storing a reference.
    :type copy_X: ``bool``

    ==========
    Attributes
    ==========

    :param ws\_: Randomly-selected phase coefficients used to generate Fourier features.
    :type ws\_: :py:class:`numpy.ndarray` of shape ``(n_components,n_features)``
    :param shape_in\_: The required shape of the input datapoints, aka the shape of the domain space :math:`X`.
    :type shape_in\_: ``tuple``
    :param shape_out\_: The final shape of the transformed datapoints, aka the shape of the Hilbert space :math:`H`.
    :type shape_out\_: ``tuple``
    :param X_fit\_: The data which was used to fit the model.
    :type X_fit\_: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_in_``
    """
    def __init__(self,
        n_components = 100,
        gamma = None,
        complex = False,
        engine = None,
        engine_params = None,
        take=None,
        filter=None,
        copy_X = True
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.complex = complex
        self.engine = engine
        self.engine_params = engine_params
        self.take = take
        self.filter = filter
        self.copy_X = copy_X
    
    def fit(self,X,y=None):
        """
        Fit the model from data in ``X``.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in ``self.take`` and ``self.filter``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``

        :param y: Not used, present for API consistency by convention.
        :type y: Ignored            

        :returns: The instance itself
        :rtype: :py:class:`RKHS`
        """
        self.X_fit_ = self._validate_data(X, accept_sparse="csr", copy=self.copy_X,allow_nd=True,)
        self.shape_in_ = self.X_fit_.shape[1:]
        X = self._apply_filter(self.X_fit_)
        n_features_in = _np.prod(X.shape[1:],dtype=int)
        if self.gamma is None:
            self.gamma = 1/n_features_in
        d = max(n_features_in,2)
        
        if self.engine_params is None:
            params = {'d':d}
        else:
            params = self.engine_params
            params['d'] = d
        
        if self.engine is None:
            engine = _st.qmc.Sobol(**params)
        else:
            engine = self.engine(**params)

        var = 2*self.gamma
        cov = _np.diag([var]*d)
        sampler = _st.qmc.MultivariateNormalQMC(mean=_np.zeros([d]),cov=cov,engine=engine)
        self.ws_ = sampler.random(self.n_components)[:,:n_features_in].reshape((self.n_components,)+X.shape[1:])

        if self.complex:
            self.shape_out_ = (self.n_components,)
        else:
            self.shape_out_ = (2*self.n_components,)
        return self

    def transform(self,X):
        """
        Transform ``X``. 

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The transformed data
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_out_``
        """
        _check_is_fitted(self)
        X = self._validate_data(X, accept_sparse="csr", copy=self.copy_X,allow_nd=True,)

        if X.shape[1:] != self.shape_in_:
            raise ValueError('Shape of X is %s but must be equal to %s' % (str(X.shape[1:]),str(self.shape_in_)))

        X = self._apply_filter(X)

        featuremap = _np.exp(
            1j*_np.tensordot(
                X,self.ws_,
                axes=[tuple(range(1,len(self.ws_.shape)))]*2
            )
        )/_np.sqrt(self.n_components)
        if self.complex:
            return featuremap
        else:
            featuremap_real = _np.zeros([featuremap.shape[0],featuremap.shape[1]*2])
            featuremap_real[:,::2] = _np.real(featuremap)
            featuremap_real[:,1::2] = _np.imag(featuremap)
            return featuremap_real
        
## OneHot
class OneHotRKHS(_RKHS):
    r"""
    ``OneHotRKHS`` is for processing categorical data.
    If :math:`X` is a discrete set, this generates an embedding map :math:`\phi:X\rightarrow H` 
    into a Hilbert space :math:`H` whose dimension is the cardinality of :math:`X`, such that
    :math:`\phi(x)` maps the element :math:`x` to a one-hot vector with the 1-valued component 
    in the dimension which uniquely corresponds to :math:`x`.

    This is particularly advantageous when working with kernel embeddings of distributions, as
    the embedded distribution vector is itself a probability vector (positive components and sums to 1).
    
    ==========
    Parameters
    ==========

    :param axis: Default = ``None``.
        Specifies the axis or axes along which unique entries will be determined. 
        The alphabet will be taken as the unique subarrays indexed by the given axes,
        and the transformed vector will have the shape of the given axes + an additional
        axis indexing the alphabet. If ``None``, defaults to all axes. The 0 axis (that is,
        the sample axis) will always be included, even if not given.
    :type axis: int or tuple of ints
    :param take: Default = ``None``.
        Specifies which values to take from the datapoint for transformation.
        If ``None``, the entire datapoint will be taken in its original shape.
        If ``bool`` array, acts as a mask setting values marked ``False`` to ``0`` and leaving values marked True unchanged.
        If ``int`` array, the integers specify the indices (along the first feature dimension) which are to be taken, in the order/shape of the desired input.
        If ``tuple`` of ``int`` arrays, allows for drawing indices across multiple dimensions, similar to passing a ``tuple`` to a ``numpy`` array.
    :type take: :py:class:`numpy.ndarray` of ``dtype int`` or ``bool``, or ``tuple`` of :py:class:`numpy.ndarray` instances of type ``int``, or ``None``
    :param copy_X: Default = ``True``.
        If ``True``, input ``X`` is copied and stored by the model in the ``X_fit_`` attribute. If no further changes will be done to ``X``, setting ``copy_X=False`` saves memory by storing a reference.
    :type copy_X: ``bool``

    ==========
    Attributes
    ==========

    :param alphabet\_: The unique elements from ``self.X_fit_``.
    :type alphabet\_: :py:class:`numpy.ndarray` of ``objects``
    :param shape_in\_: The required shape of the input datapoints, aka the shape of the domain space :math:`X`.
    :type shape_in\_: ``tuple``
    :param shape_out\_: The final shape of the transformed datapoints, aka the shape of the Hilbert space :math:`H`.
    :type shape_out\_: ``tuple``
    :param X_fit\_: The data which was used to fit the model.
    :type X_fit\_: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_in_``
    """
    def __init__(
        self,
        axis=None,
        *,
        take=None,
        copy_X=True,
    ):
        self.take=take
        self.copy_X = copy_X
        self.axis = axis

    def fit(self,X,y=None):
        """
        Fit the model from data in ``X``.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in ``self.take``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``

        :param y: Not used, present for API consistency by convention.
        :type y: Ignored            

        :returns: The instance itself
        :rtype: :py:class:`RKHS`
        """
        if self.copy_X:
            X = X.copy()
        self.X_fit_ = X
        self.shape_in_ = self.X_fit_.shape[1:]
        X = self._apply_take(X)

        if self.axis is None:
            self.alphabet_ = _np.unique(self.X_fit_)
            self.shape_out_ = self.shape_in_+(len(self.alphabet_),)
        elif self.axis == 0:
            self.alphabet_ = _np.unique(self.X_fit_,axis=0)
            self.shape_out_ = (self.alphabet_.shape[0],)
        else:
            self.axis = tuple(self.axis)
            if not(0 in self.axis):
                self.axis = (0,)+self.axis
            flatten_dim = len(self.axis)
            X_moved = _np.moveaxis(
                self.X_fit_,self.axis,
                tuple(range(flatten_dim))
            )
            self.alphabet_ = _np.unique(
                X_moved.reshape((_np.prod(X_moved.shape[:flatten_dim],dtype=int),)
                                +X_moved.shape[flatten_dim:]),
                axis=0
            )
            self.shape_out_ = X_moved.shape[1:]+(self.alphabet_.shape[0],)
        return self
    
    def transform(self,X):
        """
        Transform ``X``. 

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The transformed data
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_out_``
        """
        X = self._apply_take(X)

        if self.axis is None:
            X_transformed = (X[...,None] == self.alphabet_.reshape((1,)*X.ndim+self.alphabet_.shape)).astype(float)
        elif self.axis == 0:
            X_transformed = _np.all(X[:,None] == self.alphabet_[None],axis=tuple(range(2,X.ndim+1))).astype(float)
        else:
            flatten_dim = len(self.axis)
            X_moved = _np.moveaxis(
                X,self.axis,
                tuple(range(flatten_dim))
            )
            X_transformed = _np.all(
                X_moved.reshape(X_moved.shape[:flatten_dim]+(1,)+X_moved.shape[flatten_dim:]) == self.alphabet_.reshape((1,)*flatten_dim+self.alphabet_.shape),
                axis=tuple(range(flatten_dim+1,X_moved.ndim+1))
            ).astype(float)

        X_squeezed = _np.squeeze(X_transformed)
        if X_squeezed.ndim >1:
            return X_squeezed
        else:
            return X_squeezed[:,None]