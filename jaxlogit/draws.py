import logging

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np

from jax._src.numpy.util import promote_args_inexact
from jax._src.scipy.special import log_ndtr, ndtri
from jax._src.scipy.stats.truncnorm import _log_gauss_mass

logger = logging.getLogger(__name__)


# not implemented in jax.scipy.stats.truncnorm, implemented here based on
# https://github.com/scipy/scipy/blob/09195e4e02feedd1835a2db335f10e0e151b7909/scipy/stats/_continuous_distns.py#L10313
# and using https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/truncnorm.py
# def _truncnorm_ppf(q, a, b):
#     q, a, b = promote_args_inexact("truncnorm_ppf", q, a, b)
#     q, a, b = jnp.broadcast_arrays(q, a, b)

#     case_left = a < 0
#     case_right = ~case_left

#     def ppf_left(q, a, b):
#         log_Phi_x = jnp.logaddexp(log_ndtr(a), jnp.log(q) + _log_gauss_mass(a, b))
#         return ndtri(jnp.exp(log_Phi_x))

#     def ppf_right(q, a, b):
#         log_Phi_x = jnp.logaddexp(log_ndtr(-b), jnp.log1p(-q) + _log_gauss_mass(a, b))
#         return -ndtri(jnp.exp(log_Phi_x))

#     out = jnp.select([case_left, case_right], [ppf_left(q, a, b), ppf_right(q, a, b)])
#     return out

# This seems considerably faster based on a small-ish test case. TODO: proper performance testing.
def _truncnorm_ppf(u, a, b):
    """
    Compute the percent point function (inverse of cdf) for a truncated normal distribution.
    u is a uniform random variable in [0, 1).
    """
    phi_a = jstats.norm.cdf(a)
    phi_b = jstats.norm.cdf(b)
    val_ = phi_a + u * (phi_b - phi_a)
    return jstats.norm.ppf(val_)

def truncnorm_ppf(q, loc, scale):
    # Note I hard-coded upper and lower bound here because using -jnp.inf and (a - loc) / scale led to nan gradients
    #  # hard-code, a=-jnp.inf, b=0.0, gradient
    # q, a, b = promote_args_inexact("truncnorm_ppf", q, a, b)
    # q, a, b = jnp.broadcast_arrays(q, a, b)
    if jax.config.jax_enable_x64:
        LOG_PROB_MIN = 1e-300
        LOG_PROB_MAX = 1.0 - 1e-16
    else:
        LOG_PROB_MIN = 1e-37
        LOG_PROB_MAX = 1.0 - 1e-7

    lb = -jnp.inf  # (a - loc) / scale
    ub = -loc / scale  # (b - loc) / scale

    # clipping for numerical stability.
    # In case of ub = 0.0 , upper bound should be 1.0 and not 1-eps but result is eps instead of 0 so should be fine
    return _truncnorm_ppf(q.clip(LOG_PROB_MIN, LOG_PROB_MAX), lb, ub) * scale + loc


# TODO: have a look at scipy.stats.qmc, has sobol draws for large number of variables
def generate_draws(sample_size, n_draws, _rvdist, halton=True, halton_opts=None):
    """Generate draws based on the given mixing distributions. Note that we generate
    independent standard normals for normal and log-normal distributions and
    uniform randoms for uniform and truncated normal distributions.
    The actual distributtion during estimation is applied in _apply_distribution.
    """
    if halton:
        draws = generate_halton_draws(
            sample_size,
            n_draws,
            len(_rvdist),
            **halton_opts if halton_opts is not None else {},
        )
    else:
        draws = generate_random_draws(sample_size, n_draws, len(_rvdist))

    for k, dist in enumerate(_rvdist):
        if dist in ["n", "ln"]:  # Normal based
            draws[:, k, :] = jstats.norm.ppf(draws[:, k, :])
        # elif dist == "t":  # Triangular
        #     draws_k = draws[:, k, :]
        #     draws[:, k, :] = (np.sqrt(2 * draws_k) - 1) * (draws_k <= 0.5) + (1 - np.sqrt(2 * (1 - draws_k))) * (
        #         draws_k > 0.5
        #     )
        elif dist in ["u", "n_trunc"]:  # Uniform or truncated normal
            pass  # keep [0, 1] for trunc norm
            # draws[:, k, :] = 2 * draws[:, k, :] - 1
        else:
            raise ValueError(f"Mixing distribution {dist} for random variable {k} not implemented yet.")

    return draws  # (N,Kr,R)


def generate_random_draws(sample_size, n_draws, n_vars):
    """Generate random uniform draws between 0 and 1."""
    return np.random.uniform(size=(sample_size, n_vars, n_draws))


def generate_halton_draws(sample_size, n_draws, n_vars, shuffle=False, drop=100, primes=None):
    """Generate Halton draws for multiple random variables using different primes as base"""
    if primes is None:
        primes = [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            71,
            73,
            79,
            83,
            89,
            97,
            101,
            103,
            107,
            109,
            113,
            127,
            131,
            137,
            139,
            149,
            151,
            157,
            163,
            167,
            173,
            179,
            181,
            191,
            193,
            197,
            199,
            211,
            223,
            227,
            229,
            233,
            239,
            241,
            251,
            257,
            263,
            269,
            271,
            277,
            281,
            283,
            293,
            307,
            311,
        ]

    def halton_seq(length, prime=3, shuffle=False, drop=100):
        """Generates a halton sequence while handling memory efficiently.

        Memory is efficiently handled by creating a single array ``seq`` that is iteratively filled without using
        intermidiate arrays.
        """
        req_length = length + drop
        seq = np.empty(req_length)
        seq[0] = 0
        seq_idx = 1
        t = 1
        while seq_idx < req_length:
            d = 1 / prime**t
            seq_size = seq_idx
            i = 1
            while i < prime and seq_idx < req_length:
                max_seq = min(req_length - seq_idx, seq_size)
                seq[seq_idx : seq_idx + max_seq] = seq[:max_seq] + d * i
                seq_idx += max_seq
                i += 1
            t += 1
        seq = seq[drop : length + drop]
        if shuffle:
            np.random.shuffle(seq)
        return seq

    draws = [
        halton_seq(
            sample_size * n_draws,
            prime=primes[i % len(primes)],
            shuffle=shuffle,
            drop=drop,
        ).reshape(sample_size, n_draws)
        for i in range(n_vars)
    ]
    draws = np.stack(draws, axis=1)
    return draws  # (N,Kr,R)
