import numpy as _np
from functools import reduce as _reduce

from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.utils.validation import check_is_fitted as _check_is_fitted
from sklearn.exceptions import NotFittedError as _NotFittedError
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.linear_model import Ridge as _Ridge
# UTILITY RKHS's

class RKHS(_TransformerMixin,_BaseEstimator):
    r"""
    Base instance of a Reproducing Kernel Hilbert Space [1]. An RKHS consists of a
    Hilbert space :math:`H`, a feature mapping :math:`\phi:X \rightarrow H` from the data
    space :math:`X` into :math:`H`, and a kernel :math:`k(x,y)` on :math:`X^2` defined by 
    :math:`k(x,y) = \left<\phi(x),\phi(y)\right>_H`. This base RKHS sets :math:`H=X` by default, with
    :math:`\phi(x)=x` and :math:`k(x,y)=x^T y`.

    Certain functions :math:`f` may be represented in :math:`H` with a vector :math:`F`
    satisfying :math:`\left<F,\phi(x)\right>_H=f(x)` for all :math:`x \in X`. This representation can
    be discovered using ridge regression [2]. The set of valid functions depends 
    on :math:`H` and :math:`k`. This base RKHS class can only represent *linear* functions. 

    The :py:func:`fit` method will typically determine the dimensions and shapes of :math:`H`
    and :math:`X`, as well as any other necessary parameters for determining the
    feature mapping :math:`\phi`. The :py:func:`transform` method will implement the feature
    mapping :math:`\phi`. The :py:func:`kernel` method will evaluate the kernel :math:`k`. The
    :py:func:`fit_function` method will find the representation of a function :math:`f` given
    the vector :math:`y_i=f(x_i)` of its values on the predictor variables.

    RKHS instances can be combined with one another via composition, direct sum
    and tensor product. These produce compound RKHS classes, :py:class:`CompositeRKHS`,
    :py:class:`DirectSumRKHS`, and :py:class:`ProductRKHS`. These combinations can be instantiated
    with the corresponding class, or generated from arbitrary RKHS instances
    using the operations ``@`` for composition, ``+`` for direct sum, and ``*`` for
    tensor product. See the corresponding classes for further details.

    1. `Aronszajn, N. "Theory of reproducing kernels." Trans. Amer. Math. Soc. 68 (1950), 337-404. <https://www.ams.org/journals/tran/1950-068-03/S0002-9947-1950-0051437-7/>`_
    2. Murphy, K. P. "Machine Learning: A Probabilistic Perspective", The MIT Press. chapter 14.4.3, pp. 492-493
    
    ==========
    Parameters
    ==========
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
    :param shape_in\_: The required shape of the input datapoints, aka the shape of the domain space :math:`X`.
    :type shape_in\_: ``tuple``
    :param shape_out\_: The final shape of the transformed datapoints, aka the shape of the Hilbert space :math:`H`.
    :type shape_out\_: ``tuple``
    :param X_fit\_: The data which was used to fit the model.
    :type X_fit\_: :py:class:`numpy.ndarray` of shape `(n_samples,)+self.shape_in_`
    """

    def __init__(self,*,take=None,filter=None,copy_X = True):
        self.take = take
        self.filter = filter
        self.copy_X = copy_X
        return None
    
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
        self.X_fit_ = self._validate_data(X, accept_sparse="csr", copy=self.copy_X,allow_nd=True,ensure_2d=False)
        self.shape_in_ = self.X_fit_.shape[1:]
        self.shape_out_ = self._apply_filter(self.X_fit_).shape[1:]
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
        X = self._validate_data(X, accept_sparse="csr", copy=self.copy_X,allow_nd=True,ensure_2d=False)
        if X.shape[1:] == self.shape_in_:
            return self._apply_filter(X).reshape((X.shape[0],)+self.shape_out_)
        else:
            raise ValueError('The input shape of the data, %s, does not match the required input type, %s' % (str(X.shape[1]),str(self.shape_in_)))

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

    def kernel(self,X,Y=None):
        """
        Evaluates the kernel on ``X`` and ``Y`` (or ``X`` and ``X``).

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :param Y: Default = ``None``.
            Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``. If ``None``, ``X`` is used.
        :type Y: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The matrix ``K[i,j] = k(X[i],Y[j])`` 
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples_1,n_samples_2)``
        """
        X = self._validate_data(X, accept_sparse="csr", copy=self.copy_X,allow_nd=True,ensure_2d=False)
        if Y is None:
            Y = X
        else:
            Y = self._validate_data(Y, accept_sparse="csr", copy=self.copy_X,allow_nd=True,ensure_2d=False)
        return _np.tensordot(self.transform(X),self.transform(Y),axes=[tuple(range(1,len(self.shape_out_)+1))]*2)

    def fit_function(self,y,X=None,regressor=None,alpha=1):
        """
        Fit a function using its values on the predictor data and a regressor.

        :param y: Target vector, where ``n_samples`` is the number of samples and ``n_targets`` is the number of target functions.
        :type y: :py:class:`numpy.ndarray` of shape ``(n_samples, n_targets)``   

        :param X: Default = ``None``.
            Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``. If ``None``, ``self.X_fit_`` is used.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :param regressor: The regressor object to use to fit the function. If ``None``, a :py:class:`sklearn.linear_model.Ridge` instance is used with ``fit_intercept=False`` and ``alpha`` specified below.
        :type regressor: :py:class:`sklearn.base.BaseEstimator`

        :param alpha: The ridge parameter used in the default Ridge regressor.
        :param type: float

        :returns: ``regressor``, fitted to provide the function representation.
        :rtype: object
        """
        if X is None:
            X = self.X_fit_
        else:
            X = self._validate_data(X, accept_sparse="csr", copy=self.copy_X,allow_nd=True,ensure_2d=False)
            
        if regressor is None:
            regressor = _Ridge(alpha=alpha,fit_intercept=False)

        X_reps = self.transform(X)

        y = self._validate_data(y, accept_sparse="csr", copy=self.copy_X,allow_nd=True,ensure_2d=False)
        
        return regressor.fit(X_reps,y)

    def __matmul__(self,other):
        """
        Constructs the :py:class:`CompositeRKHS` from ``self`` and ``other``.
        """
        return CompositeRKHS([other,self],copy_X = other.copy_X)

    def __mul__(self,other):
        """
        Constructs the :py:class:`ProductRKHS` from ``self`` and ``other``.
        """
        return ProductRKHS([self,other],copy_X = self.copy_X or other.copy_X)

    def __add__(self,other):
        """
        Constructs the :py:class:`DirectSumRKHS` from ``self`` and ``other``.
        """
        return DirectSumRKHS([self,other],copy_X = self.copy_X or other.copy_X)

    def _apply_filter(self,X,):
        """
        Applies ``self.take`` and ``self.filter`` to the input data as a preprocessing step.
        """
        X = self._apply_take(X,)
        if self.filter is None:
            return X
        elif self.filter.ndim == X.ndim-1:
            return X*self.filter[None]
        elif self.filter.ndim > X.ndim-1:
            if self.filter.shape[:X.ndim-1] == X.shape[1:]:
                return _np.tensordot(X,self.filter,axes=(tuple(range(1,X.ndim)),tuple(range(0,X.ndim-1))))
            else:
                raise ValueError('First %d axes of filter must have same shape as the last %d axes of apply_take(X,take).' % (X.ndim-1,)*2)
        else:
            raise ValueError('Dimension of filter must be at least %d' % (X.ndim-1,))
            
    def _apply_take(self,X,):
        """
        Applies ``take`` to the input data as a preprocessing step.
        """
        if self.take is None:
            return X
        elif type(self.take) is tuple:
            return _np.array([X[k][self.take] for k in range(X.shape[0])])
        elif self.take.dtype is bool or self.take.dtype is _np.dtype('int64'):
            return X[:,self.take]
        else:
            raise ValueError('take is not of the right form (must either be a boolean mask, an array of indices, or a tuple of integer arrays')

# Compound RKHS's
class CompositeRKHS(RKHS):
    r"""
    Given a sequence of RKHS's with Hilbert spaces :math:`H_1`, ..., :math:`H_n` and feature
    maps :math:`\phi_1`, ..., :math:`\phi_n`, their composition lives in the final Hilbert
    space :math:`H_n` but has feature map :math:`\phi_n \circ \dots \circ \phi_1` [1].
    Correspondingly, a ``CompositeRKHS`` class has the ``shape_out_`` of its final
    component, the ``shape_in_`` of its first component, and :py:func:`transform` is applied to the data
    by implementing ``transform`` sequentially for each of the component spaces.
    This is useful for building pipelines and deep kernels.

    1. `Cho, Y., Lawrence, S. "Kernel Methods for Deep Learning." Advances in Neural Information Processing Systems 22 (NIPS 2009) <https://papers.nips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html>`_

    ==========
    Parameters
    ==========
    :param components:  The component :py:class:`RKHS` objects, listed from the first to be applied to the last.
    :type components: list of :py:class:`RKHS` objects
               
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
        components,
        *,
        copy_X=True
    ):
        self.components = components
        self.copy_X = copy_X

    def fit_transform(self,X,y=None):
        """
        Fit the model from data in ``X`` and transform ``X``.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in `self.components[0].take` and `self.components[0].filter`.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``        

        :returns: The transformed data
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_out_``
        """
        if self.copy_X:
            X = X.copy()
        self.X_fit_ = X
        self.shape_in_ = self.X_fit_.shape[1:]

        current_X = self.X_fit_
        for rkhs in self.components[:-1]:
            new_X = rkhs.fit_transform(current_X)
            current_X = new_X
        self.components[-1].fit(current_X)

        self.shape_out_ = self.components[-1].shape_out_

        return current_X

    def fit(self,X,y=None):
        """
        Fit the model from data in ``X`.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in `self.components[0].take` and `self.components[0].filter`.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``        

        :returns: The instance itself
        :rtype: :py:class:`RKHS`
        """
        self.fit_transform(X,y)
        return self

    def transform(self,X):
        """
        Transform ``X``.

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The transformed data
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_out_``
        """
        if self.copy_X:
            X = X.copy()
        current_X = X

        for rkhs in self.components:
            new_X = rkhs.transform(current_X)
            current_X = new_X

        return current_X

    def kernel(self,X,Y=None):
        """
        Evaluates the kernel on ``X`` and ``Y`` (or ``X`` and ``X``) by iterating over component
        embeddings. As such, ``CompositeRKHS`` kernels can only be evaluated after fitting to data.

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :param Y: Default = ``None``.
            Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``. If ``None``, ``X`` is used.
        :type Y: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The matrix ``K[i,j] = k(X[i],Y[j])`` 
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples_1,n_samples_2)``
        """
        try:
            _check_is_fitted(self)
            if self.copy_X:
                X = X.copy()
            current_X = X

            if Y is None:
                current_Y = current_X
            else:
                if self.copy_X:
                    Y = Y.copy()
                current_Y = Y

            for rkhs in self.components[:-1]:
                current_X = rkhs.transform(current_X)
                current_Y = rkhs.transform(current_Y)

            return self.components[-1](current_X,current_Y)

        except:
            raise _NotFittedError("Composite RKHS's must be fitted before their kernels can be evaluated.")
        
    def __matmul__(self,other):
        """
        Constructs the ``CompositeRKHS`` from ``self` and `other``. Flattens the component list to avoid unnecessary recursion.
        """
        if type(other) is CompositeRKHS:
            return CompositeRKHS(other.components+self.components,copy_X = other.copy_X)
        else:
            return CompositeRKHS(self.components.insert(0,other),copy_X = self.copy_X)

class ProductRKHS(RKHS):
    r"""
    Given a sequence of RKHS's with Hilbert spaces :math:`H_1`, ..., :math:`H_n` and feature
    maps :math:`\phi_1`, ..., :math:`\phi_n`, their composition lives in the tensor product Hilbert
    space :math:`H_1\otimes \dots \otimes H_n` and has feature map 
    :math:`\phi_1 \otimes \dots \otimes \phi_n` [1].
    Correspondingly, the ``shape_out_`` of a ``ProductRKHS`` instance is the
    tuple-sum of the ``shape_out_`` tuples of its factors, while all its factors share
    the same ``shape_in_``.

    Product RKHS's are particularly useful for working with kernel embeddings of
    distributions and their conditional probabilities [2]. A ``ProductRKHS`` can
    be reduced to its marginal along a set of factors using the :py:func:`marginal`
    method, and can be reduced into a marginal space paired with a
    ridge-regressed conditional map using the :py:func:`conditional` method.

    1. `Aronszajn, N. "Theory of reproducing kernels." Trans. Amer. Math. Soc. 68 (1950), 337-404. <https://www.ams.org/journals/tran/1950-068-03/S0002-9947-1950-0051437-7/>`_
    2. `Muandet, K., Fukuzimu, K., Sriperumbudur, B., Schölkopf, B. "Kernel Mean Embedding of Distributions: A Review and Beyond." Foundations and Trends in Machine Learning: Vol. 10: No. 1-2, pp 1-141 (2017) <https://arxiv.org/abs/1605.09522/>`_

    ==========
    Parameters
    ==========
    :param factors:  The factor :py:class:`RKHS` objects, listed in the order that their dimensions will appear in indexing.
    :type factors: list of :py:class:`RKHS` objects
               
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
        factors,
        *,
        copy_X=True
    ):
        self.factors = factors
        self.copy_X = copy_X

    def fit(self,X,y=None):
        """
        Fit the model from data in ``X``.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in `fac.take` and `fac.filter` for each `fac` in `self.factors`.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``        

        :returns: The instance itself
        :rtype: :py:class:`RKHS`
        """
        if self.copy_X:
            X = X.copy()
        self.X_fit_ = X
        self.shape_in_ = self.X_fit_.shape[1:]

        for j in range(len(self.factors)):
            self.factors[j].fit(self.X_fit_)
        
        self.shape_out_ = _reduce(lambda x,y:x+y, [f.shape_out_ for f in self.factors], ())
        return self
    
    def transform(self,X):
        """
        Transform ``X``.

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The transformed data
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_out_``
        """
        if self.copy_X:
            X = X.copy()
        
        Ys = []
        for j in range(len(self.factors)):
            Ys.append(self.factors[j].transform(X))
        
        # Performs a vectorized tensor product of the feature dimensions
        tensor_func = lambda A,B: A.reshape(A.shape+(1,)*(B.ndim-1))*B.reshape((B.shape[0],)+(1,)*(A.ndim-1)+B.shape[1:])
        return _reduce(tensor_func,Ys[1:],Ys[0])
    
    def kernel(self,X,Y=None):
        """
        Evaluates the kernel on ``X`` and ``Y`` (or ``X`` and ``X``) by multiplying the kernels of the factors.

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :param Y: Default = ``None``.
            Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``. If ``None``, ``X`` is used.
        :type Y: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The matrix ``K[i,j] = k(X[i],Y[j])`` 
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples_1,n_samples_2)``
        """
        if self.copy_X:
            X = X.copy()
        if Y is None:
            Y = X
        else:
            if self.copy_X:
                Y = Y.copy()
        
        Ks = []
        for j in range(len(self.factors)):
            Ks.append(self.factors[j].kernel(X,Y))
        return _np.product(Ks,axis=0)

    def marginal(self,var_inds,copy_X=False):
        """
        Construct a ``ProductRKHS`` from only the factors specified by ``var_inds``.
        Only to be used if ``ProductRKHS`` is already fit, and you'd rather not
        fit again.

        :param var_inds: List of indices of the factors in ``self.factors`` from which to the marginal ``ProductRKHS``.
        :type var_inds: ``array`` -like of ``int``
        :param copy_X: Default = ``True``.
            If ``True``, input ``self.X_fit_`` is copied and stored as the new model's ``X_fit_`` attribute. If no further changes will be done to ``X``, setting ``copy_X=False`` saves memory by storing a reference.
        :type copy_X: ``bool``

        :returns: The marginal ``ProductRKHS`` of the ``var_inds``.
        :rtype: ``ProductRKHS``
        """
        new_rkhs = ProductRKHS(list(map(self.factors.__getitem__, var_inds)))
        if copy_X:
            new_rkhs.X_fit_ = self.X_fit_.copy()
        else:
            new_rkhs.X_fit_ = self.X_fit_
        new_rkhs.shape_in_ = self.X_fit_.shape[1:]
        new_rkhs.shape_out_ = _reduce(lambda x,y:x+y, [f.shape_out_ for f in new_rkhs.factors], ())
        return new_rkhs

    def conditional(self,predictor_inds,response_inds,regressor=None,alpha=1.0):
        """
        Returns a pair of outputs, the first being a :py:class:`sklearn.pipelines.Pipeline` 
        consisting of the marginal RKHS of ``predictor_inds`` and a regressor which represents 
        the conditional distribution embedding, and the second being the marginal RKHS 
        of ``response_inds``.

        For two systems :math:`X` and :math:`Y`, embedded in Hilbert spaces 
        :math:`H_1` and :math:`H_2` respectively, the conditional distribution 
        embedding is a linear map :math:`C_{Y|X}:H_1\\rightarrow H_2` such that 
        :math:`C_{Y|X}\phi_1(x)` gives the kernel embedding of the distribution
        of :math:`Y` conditioned on :math:`X=x`. This is typically determined
        by using a ridge regression, though we allow the user to pass a custom 
        regressor for model selection purposes. See [1] for details.

        1. `Muandet, K., Fukuzimu, K., Sriperumbudur, B., Schölkopf, B. "Kernel Mean Embedding of Distributions: A Review and Beyond." Foundations and Trends in Machine Learning: Vol. 10: No. 1-2, pp 1-141 (2017) <https://arxiv.org/abs/1605.09522/>`_

        :param predictor_inds: List of indices of the factors in ``self.factors`` on which the ``response_inds`` will be conditioned.
        :type predictor_inds: ``array`` -like of ``int``
        :param response_inds: List of indices of the factors in ``self.factors`` which are to be conditioned on the ``predictor_inds``.
        :type predictor_inds: ``array`` -like of ``int``
        :param regressor: The regressor object to use to fit the conditional embedding. If ``None``, a :py:class:`sklearn.linear_model.Ridge` instance is used with ``fit_intercept=False`` and ``alpha`` specified below.
        :type regressor: :py:class:`sklearn.base.BaseEstimator`
        :param alpha: The ridge parameter used in the default Ridge regressor.
        :type alpha: float

        :returns: (``pipe``,``response``), where ``pipe`` is a pipeline consisting of the marginal of ``predictor_inds`` and the fitted ``regressor``, and ``response`` is the marginal of ``response_inds``.
        :rtype: (:py:class:`sklearn.pipelines.Pipeline`, ``ProductRKHS``)
        """
        if regressor is None:
            regressor = _Ridge(fit_intercept=False,alpha=alpha)

        rkhs_predictor = self.marginal(predictor_inds)
        rkhs_response = self.marginal(response_inds)

        X_in = rkhs_predictor.transform(rkhs_predictor.X_fit_)
        y_in = rkhs_response.transform(rkhs_response.X_fit_)
        regressor.fit(X_in.reshape([X_in.shape[0],_np.prod(X_in.shape[1:],dtype=int)]),
                      y_in.reshape([y_in.shape[0],_np.prod(y_in.shape[1:],dtype=int)]))

        pipe = _Pipeline([('embedding',rkhs_predictor),('regressor',regressor)])
        return pipe,rkhs_response
    
    def __mul__(self,other):
        """
        Constructs the ``ProductRKHS`` from ``self` and `other``. Flattens the factor list to avoid unnecessary recursion.
        """
        if type(other) is ProductRKHS:
            return ProductRKHS(self.factors+other.factors,self.filters+other.filters,copy_X = self.copy_X or other.copy_X)
        else:
            return ProductRKHS(self.factors.append[other],self.filters.append[None],copy_X = self.copy_X)

class DirectSumRKHS(RKHS):
    r"""
    Given a sequence of RKHS's with Hilbert spaces :math:`H_1`, ..., :math:`H_n` and feature
    maps :math:`\phi_1`, ..., :math:`\phi_n`, their direct sum lives in the tensor product Hilbert
    space :math:`H_1\oplus \dots \oplus H_n` and has feature map of stacked vectors
    :math:`[\phi_1^T,\ \dots,\ \phi_n^T]^T` [1].
    Correspondingly, the ``shape_out_`` of a ``DirectRKHS`` instance is determined the the same manner
    as when using :py:func:`numpy.concatenate` on the specified axis, while all its subspaces share
    the same ``shape_in_``.

    1. `Aronszajn, N. "Theory of reproducing kernels." Trans. Amer. Math. Soc. 68 (1950), 337-404. <https://www.ams.org/journals/tran/1950-068-03/S0002-9947-1950-0051437-7/>`_

    ==========
    Parameters
    ==========
    :param subspaces:  The subspace :py:class:`RKHS` objects, listed in the order that their indices will appear along the first axis.
    :type subspaces: list of :py:class:`RKHS` objects

    :param axis:  The axis along which the data will be concatenated. Data dimension must match on all other axes.
    :type axis: int
               
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
        subspaces,
        axis=0,
        *,
        copy_X=True
    ):
        self.subspaces = subspaces
        self.axis = axis
        self.copy_X = copy_X

    def fit(self,X,y=None):
        """
        Fit the model from data in ``X``.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in `sub.take` and `sub.filter` for each `sub` in `self.subspaces`.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``        

        :returns: The instance itself
        :rtype: :py:class:`RKHS`
        """
        if self.copy_X:
            X = X.copy()
        self.X_fit_ = X
        self.shape_in_ = self.X_fit_.shape[1:]

        for j in range(len(self.subspaces)):
            self.subspaces[j].fit(X)
        
        shapes_out = _np.array([list(s.shape_out_) for s in self.subspaces])
        axes = list(range(shapes_out.shape[1]))
        axes.remove(self.axis)
        axis_mask = _np.zeros(shapes_out.shape[1])
        axis_mask[self.axis] = 1
        if _np.all(shapes_out[1:,axes]==shapes_out[None,0,axes]):
            shapesum = lambda sh1,sh2:sh1+axis_mask*sh2
            self.shape_out_ = tuple(_reduce(shapesum,shapes_out[1:],shapes_out[0]))
        else:
            raise ValueError('Subspaces have incompatible shapes for direct sum')
            
        return self
    
    def transform(self,X):
        """
        Transform ``X``.

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The transformed data
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples,)+self.shape_out_``
        """
        if self.copy_X:
            X = X.copy()
        Ys = []
        for j in range(len(self.subspaces)):
            self.subspaces[j].fit(X)
            Ys.append(self.subspaces[j].transform(X))
        return _np.concatenate(Ys,axis=1+self.axis)
    
    def kernel(self,X,Y=None):
        """
        Evaluates the kernel on ``X`` and ``Y`` (or ``X`` and ``X``) by summing the kernels of the factors.

        :param X: Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :param Y: Default = ``None``.
            Data vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. These must match ``self.shape_in_``. If ``None``, ``X`` is used.
        :type Y: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``   

        :returns: The matrix ``K[i,j] = k(X[i],Y[j])`` 
        :rtype: :py:class:`numpy.ndarray` of shape ``(n_samples_1,n_samples_2)``
        """
        if self.copy_X:
            X = X.copy()
        if Y is None:
            Y = X
        else:
            if self.copy_X:
                Y = Y.copy()
        
        Ks = []
        for j in range(len(self.subspaces)):
            Ks.append(self.subspaces[j].kernel(X,Y))
        return _np.sum(Ks,axis=0)
    
    def __add__(self,other):
        """
        Constructs the ``DirectSumRKHS`` from ``self` and `other``. Flattens the subspace list to avoid unnecessary recursion.
        """
        if type(other) is DirectSumRKHS:
            return DirectSumRKHS(self.subspaces+other.subspaces,self.filters+other.filters,copy_X = self.copy_X or other.copy_X)
        else:
            return DirectSumRKHS(self.subspaces.append[other],self.filters.append[None],copy_X = self.copy_X)