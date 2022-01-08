import numpy as _np
from scipy.linalg import eig as _eig

def joint_probs_hilbert_schmidt_scorer(regressor,X,y):
    r"""
    Determines if a conditional embedding accurately represents the correlations in the
    original dataset.

    If we have samples over two spaces :math:`X` and :math:`Y` embedded into :math:`H_1\otimes H_2` as

    .. math::

        \mu_{XY} = \frac{1}{N}\sum_i \phi_1(x_i)\otimes \phi_2(y_i)

    and an estimated conditional embedding :math:`\hat{C}_{Y|X}`, we reconstruct the joint estimator as

    .. math::

        \hat{\mu}_{XY} = \frac{1}{N}\sum_i \phi_1(x_i)\otimes \hat{C}_{Y|X}\phi(x_i)

    and return the normalized Hilbert-Schmidt inner product:

    .. math::

        \mathrm{Score} = \frac{\left<\mu_{XY},\hat{\mu}_{XY}\right>_{\mathrm{HS}}}{\sqrt{\left<\hat{\mu}_{XY},\hat{\mu}_{XY}\right>_{\mathrm{HS}}\left<\mu_{XY},\mu_{XY}\right>_{\mathrm{HS}}}}

    :param regressor: The fitted ``regressor`` which computes the conditional map.
    :type regressor: :py:class:`sklearn.base.BaseEstimator`
    :param X: The training data.
    :type X: :py:class:`numpy.ndarray`
    :param y: The target data.
    :type y: :py:class:`numpy.ndarray`
    """
    y_pred = regressor.predict(X)
    cov_Q = X.T@y/X.shape[0]
    cov_pred = X.T@(y_pred)/X.shape[0]
    return (cov_Q*cov_pred).sum()/_np.sqrt((cov_Q**2).sum()*(cov_pred**2).sum())

def joint_probs_euclidean_scorer(regressor,X,y):
    r"""
    Determines if a conditional embedding accurately represents the correlations in the
    original dataset.

    If we have samples over two spaces :math:`X` and :math:`Y` embedded into :math:`H_1\otimes H_2` as

    .. math::

        \mu_{XY} = \frac{1}{N}\sum_i \phi_1(x_i)\otimes \phi_2(y_i)

    and an estimated conditional embedding :math:`\hat{C}_{Y|X}`, we reconstruct the joint estimator as

    .. math::

        \hat{\mu}_{XY} = \frac{1}{N}\sum_i \phi_1(x_i)\otimes \hat{C}_{Y|X}\phi(x_i)

    and return the (negative) Euclidean distance:

    .. math::

        \mathrm{Score} = -\left\|\mu_{XY}-\hat{\mu}_{XY}\right\|_{\mathrm{HS}}

    :param regressor: The fitted ``regressor`` which computes the conditional map.
    :type regressor: :py:class:`sklearn.base.BaseEstimator`
    :param X: The training data.
    :type X: :py:class:`numpy.ndarray`
    :param y: The target data.
    :type y: :py:class:`numpy.ndarray`
    """
    y_pred = regressor.predict(X)
    cov_Q = X.T@y/X.shape[0]
    cov_pred = X.T@(y_pred)/X.shape[0]
    return -_np.sqrt(((cov_Q-cov_pred)**2).sum())

def ghmm_score(P,Q):
    r"""
    Compares two sets of transition matrices for a hidden Markov model (or generalized hidden Markov model)
    and computes a score between them which is 1 if they generate the same process and :math:`<1` otherwise.

    Let :math:`P^{(x)}_{ij}` and :math:`Q^{(x)}_{ij}` be the symbol-labeled state transition matrices of two separate HMMs.
    Construct the block matrix :math:`E^{(P,Q)}_{ij,kl}` as: 

    .. math::

        E^{(P,Q)}_{ij,kl} = \sum_x P^{(x)}_{ik}Q^{(x)}_{jl}
    
    Denote the leading eigenvalue of this matrix across the comma as :math:`\lambda^{(P,Q)}`.
    Then the score is computed as

    .. math::

        \mathrm{Score} = \frac{\lambda^{(P,Q)}}{\sqrt{\lambda^{(P,P)},\lambda^{(Q,Q)}}}

    :param P: The symbol-labeled transition matrices of the first HMM.
    :type P: :py:class:`numpy.ndarray` of shape ``(n_symbols,n_states_1,n_states_1)``
    :param Q: The symbol-labeled transition matrices of the second HMM. Can have different number of states from ``P`` but must have same number of symbols.
    :type Q: :py:class:`numpy.ndarray` of shape ``(n_symbols,n_states_2,n_states_2)``
    """
    pair_cross = _np.moveaxis(
        (P[:,:,:,None,None]*Q[:,None,None,:,:]).sum(axis=0),
        [1,2],[2,1]
    ).reshape([P.shape[1]*Q.shape[1]]*2)
    pred_cross = _np.moveaxis(
        (P[:,:,:,None,None]*P[:,None,None,:,:]).sum(axis=0),
        [1,2],[2,1]
    ).reshape([P.shape[1]**2]*2)
    act_cross = _np.moveaxis(
        (Q[:,:,:,None,None]*Q[:,None,None,:,:]).sum(axis=0),
        [1,2],[2,1]
    ).reshape([Q.shape[1]**2]*2)

    e,_ = _eig(pair_cross)
    lam_cross = _np.max(_np.abs(e))
    e,_ = _eig(pred_cross)
    lam_P = _np.max(_np.abs(e))
    e,_ = _eig(act_cross)
    lam_Q = _np.max(_np.abs(e))
    
    return lam_cross/_np.sqrt(lam_P*lam_Q)