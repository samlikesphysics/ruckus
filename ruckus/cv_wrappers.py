from ruckus.scoring import joint_probs_hilbert_schmidt_scorer as _joint_probs_hilbert_schmidt_scorer
from sklearn.base import BaseEstimator as _BaseEstimator

class ConditionalMapWrapper(_BaseEstimator):
    """
    Cross-validation wrapper for constructing a :py:class:`ProductRKHS` and conditioning
    some of its factor spaces on the others.

    For two systems :math:`X` and :math:`Y`, embedded in Hilbert spaces 
    :math:`H_1` and :math:`H_2` respectively, the conditional distribution 
    embedding is a linear map :math:`C_{Y|X}:H_1\\rightarrow H_2` such that 
    :math:`C_{Y|X}\phi_1(x)` gives the kernel embedding of the distribution
    of :math:`Y` conditioned on :math:`X=x`. This is typically determined
    by using a ridge regression, though we allow the user to pass a custom 
    regressor for model selection purposes. See [1] for details.

    1. `Muandet, K., Fukuzimu, K., Sriperumbudur, B., Schölkopf, B. "Kernel Mean Embedding of Distributions: A Review and Beyond." Foundations and Trends in Machine Learning: Vol. 10: No. 1-2, pp 1-141 (2017) <https://arxiv.org/abs/1605.09522/>`_

    ==========
    Parameters
    ==========
    :param prod_rkhs: The :py:class:`ProductRKHS` instance to fit to the data.
    :type prod_rkhs: :py:class:`ProductRKHS`
    :param predictor_inds: List of indices of the factors in ``prod_rkhs.factors`` on which the ``response_inds`` will be conditioned.
    :type predictor_inds: ``array`` -like of ``int``
    :param response_inds: List of indices of the factors in ``prod_rkhs.factors`` which are to be conditioned on the ``predictor_inds``.
    :type predictor_inds: ``array`` -like of ``int``
    :param regressor: The regressor object to use to fit the conditional embedding. If ``None``, a :py:class:`sklearn.linear_model.Ridge` instance is used with ``fit_intercept=False`` and ``alpha`` specified below.
    :type regressor: :py:class:`sklearn.base.BaseEstimator`
    :param alpha: The ridge parameter used in the default Ridge regressor.
    :type alpha: float
    :param scoring: The scoring function which will be applied to the ``regressor``. If ``None``, :py:func:`joint_probs_hilbert_schmidt_scorer` is used.
    :type scoring: callable

    ==========
    Attributes
    ==========

    :param conditional_map_: A pipeline consisting of the marginal of ``predictor_inds`` and the fitted ``regressor``.
    :type conditional_map_: :py:class:`sklearn.pipelines.Pipeline`
    :param marginal_response_: The marginal of ``response_inds``.
    :type marginal_response_: :py:class:`ProductRKHS`
    """
    def __init__(
        self,
        prod_rkhs,
        predictor_inds,
        response_inds,
        regressor=None,
        alpha=1.0,
        scoring=None
    ):
        self.prod_rkhs = prod_rkhs
        self.predictor_inds = predictor_inds
        self.response_inds = response_inds
        self.regressor = regressor
        self.alpha = alpha
        self.scoring=scoring
    
    def fit(self,X,y=None):
        """
        Fit the model from data in ``X``.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in ``fac.take`` and ``fac.filter`` for each ``fac`` in ``prod_rkhs.factors``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``        

        :returns: The instance itself
        :rtype: :py:class:`ConditionalMapWrapper`
        """
        self.prod_rkhs.fit(X)
        self.conditional_map_, self.marginal_response_ = self.prod_rkhs.conditional(
            self.predictor_inds,
            self.response_inds,
            self.regressor,
            self.alpha
        )
        return self

    def score(self,X):
        """
        Scores the model's performance on data ``X`` using the specified ``scoring`` function.

        :param X: Training vector, where ``n_samples`` is the number of samples and ``(n_features_1,...,n_features_d)`` is the shape of the input data. Must be consistent with preprocessing instructions in ``fac.take`` and ``fac.filter`` for each ``fac`` in ``prod_rkhs.factors``.
        :type X: :py:class:`numpy.ndarray` of shape ``(n_samples, n_features_1,...,n_features_d)``        

        :returns: The score.
        :rtype: float
        """
        if self.scoring is None:
            scoring = _joint_probs_hilbert_schmidt_scorer
        else:
            scoring = self.scoring

        X_in = self.conditional_map_.named_steps['embedding'].transform(X)
        y_in = self.marginal_response_.transform(X)
        return scoring(self.conditional_map_.named_steps['regressor'],X_in,y_in)
