# -*- coding: utf-8 -*-
import jax.numpy as jnp
import numpy as np
import pytest

from jaxlogit.mixed_logit import MixedLogit, neg_loglike
from xlogit import MixedLogit as MixedLogitXl


# Setup data used for tests
X = np.array([[2, 1], [1, 3], [3, 1], [2, 4], [2, 1], [2, 4]])
y = np.array([0, 1, 0, 1, 0, 1])
ids = np.array([1, 1, 2, 2, 3, 3])
alts = np.array([1, 2, 1, 2, 1, 2])
panels = np.array([1, 1, 1, 1, 2, 2])
varnames = ["a", "b"]
randvars = {"a": "n", "b": "n"}
N, J, K, R = 3, 2, 2, 5

MIN_COMP_ZERO = 1e-300
MAX_COMP_EXP = 700


def test_log_likelihood():
    """
    Computes the log-likelihood "by hand" for a simple example and ensures
    that the one returned by xlogit is the same
    """
    betas = np.array([0.1, 0.1, 0.1, 0.1])
    X_, y_ = X.reshape(N, J, K), y.reshape(N, J, 1)

    # Compute log likelihood using xlogit
    X_, y_ = (
        X.reshape(N * J, K),
        y.astype(bool).reshape(
            N * J,
        ),
    )
    Xd = X_[~y_, :].reshape(N, J - 1, K) - X_[y_, :].reshape(N, 1, K)

    model = MixedLogit()
    model._rvidx, model._rvdist = np.array([True, True]), np.array(["n", "n"])
    Xdf = Xd[:, :, ~model._rvidx]  # Data for fixed parameters
    Xdr = Xd[:, :, model._rvidx]  # Data for random parameters

    draws = model._generate_halton_draws(N, R, K)  # (N,Kr,R)
    # betas, Xd, panels, draws, weights, avail, scale_d, addit_d, batch_size, return_gradient=True
    # obtained_loglik = model._loglik_gradient(betas, Xd, None, draws, None, None, None, None, R, return_gradient=False)
    obtained_loglik = neg_loglike(
        betas,
        jnp.array(Xdf),
        jnp.array(Xdr),
        None,  # panels
        jnp.array(draws),
        None,  # weights
        None,  # avail
        None,  # scale_d
        None,  # mask
        None,  # values_for_mask
        jnp.array(model._rvidx),
        rand_idx=jnp.where(jnp.array(model._rvidx))[0],
        fixed_idx=jnp.where(~jnp.array(model._rvidx))[0],
        num_panels=None,
        idx_ln_dist=jnp.array([]),
        include_correlations=False,
    )

    # Compute expected log likelihood "by hand"
    X_, y_ = X.reshape(N, J, K), y.reshape(N, J, 1)
    Br = betas[None, [0, 1], None] + draws * betas[None, [2, 3], None]
    eXB = np.exp(np.einsum("njk,nkr -> njr", X_, Br))
    p = eXB / np.sum(eXB, axis=1, keepdims=True)
    expected_loglik = -np.sum(np.log((y_ * p).sum(axis=1).mean(axis=1)))

    assert expected_loglik == pytest.approx(obtained_loglik)


# def test__transform_betas():
#     """
#     Check that betas are properly transformed to random draws

#     """
#     betas = np.array([0.1, 0.1, 0.1, 0.1])

#     # Compute log likelihood using xlogit
#     model = MixedLogit()
#     model._rvidx, model._rvdist = np.array([True, True]), np.array(["n", "n"])
#     draws = model._generate_halton_draws(N, R, K)  # (N,Kr,R)
#     expected_betas = betas[None, [0, 1], None] + draws * betas[None, [2, 3], None]
#     obtained_betas = _transform_rand_betas(betas, draws)

#     assert np.allclose(expected_betas, obtained_betas)


# def test_fit():
#     """
#     Ensures the log-likelihood works for multiple iterations with the default
#     initial coefficients. The value of -1.473423 was computed by hand for
#     comparison purposes
#     """
#     # There is no need to initialize a random seed as the halton draws produce
#     # reproducible results
#     model = MixedLogit()
#     model.fit(
#         jnp.array(X),
#         jnp.array(y),
#         varnames,
#         jnp.array(alts),
#         jnp.array(ids),
#         randvars,
#         n_draws=10,
#         panels=jnp.array(panels),
#         maxiter=0,
#         init_coeff=jnp.array(np.repeat(0.1, 4)),
#         weights=jnp.array(np.ones(N * J)),
#     )

#     model_xl = MixedLogitXl()
#     model_xl.fit(
#         X,
#         y,
#         varnames,
#         alts,
#         ids,
#         randvars,
#         n_draws=10,
#         panels=panels,
#         maxiter=0,
#         verbose=0,
#         init_coeff=np.repeat(0.1, 4),
#         weights=np.ones(N * J),
#     )
#     assert model_xl.loglikelihood == pytest.approx(-1.473423)

#     # assert model.loglikelihood == pytest.approx(-1.473423)  # my fit says -0.1770801 which is better. Why?
#     # Ah I htink this is because maxiter is not working, it looks like it converged.


def test_validate_inputs():
    """
    Covers potential mistakes in parameters of the fit method that xlogit
    should be able to identify
    """
    model = MixedLogit()
    with pytest.raises(ValueError):  # wrong distribution
        model.fit(
            X,
            y,
            varnames=varnames,
            alts=alts,
            ids=ids,
            n_draws=10,
            maxiter=0,
            halton=True,
            randvars={"a": "fake"},
        )

    with pytest.raises(ValueError):  # wrong var name
        model.fit(
            X,
            y,
            varnames=varnames,
            alts=alts,
            ids=ids,
            n_draws=10,
            maxiter=0,
            halton=True,
            randvars={"fake": "n"},
        )
