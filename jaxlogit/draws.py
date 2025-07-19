import logging

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np

logger = logging.getLogger(__name__)


# TODO: have a look at scipy.stats.qmc, has sobol draws for large number of variables
def generate_draws(sample_size, n_draws, _rvdist, halton=True, halton_opts=None):
    """Generate draws based on the given mixing distributions."""
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
        # elif dist == "u":  # Uniform
        #     draws[:, k, :] = 2 * draws[:, k, :] - 1
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


def van_der_corput_jax(k, base=2, perm=None):
    """JAX-traceable van der Corput value for integer k and base, with optional digit permutation."""

    def cond_fun(state):
        k, val, denom = state
        return k > 0

    def body_fun(state):
        k, val, denom = state
        k, remainder = divmod(k, base)
        if perm is not None:
            remainder = perm[remainder]
        val = val + remainder / denom
        denom = denom * base
        return (k, val, denom)

    _, val, _ = jax.lax.while_loop(cond_fun, body_fun, (k, 0.0, base.astype(float)))
    return val


def halton_seq_jax(length, base=2, drop=0, shuffle=False, key=None, scramble=False, perm=None):
    # idxs = jnp.arange(length + drop)[drop:]
    MAX_DRAW = 100_000_000_000_000
    idxs = jax.lax.dynamic_slice(jnp.arange(MAX_DRAW), (drop,), (length,))
    if scramble:
        if perm is None:
            raise ValueError("A permutation array must be provided for scrambling.")
        seq = jax.vmap(lambda k: van_der_corput_jax(k, base, perm))(idxs)
    else:
        seq = jax.vmap(lambda k: van_der_corput_jax(k, base))(idxs)
    if shuffle:
        if key is None:
            raise ValueError("A JAX PRNG key must be provided for shuffling.")
        seq = jax.random.permutation(key, seq)
    return seq


# @partial(jax.jit, static_argnames=["sample_size", "n_draws", "n_vars"])
def get_normal_halton_draws_jax(sample_size, n_draws, n_vars, drop=100, shuffle=False, key=None, primes=None):
    if primes is None:
        primes = jnp.array(
            [
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
        )

    # Optionally split the key for each variable
    if shuffle:
        if key is None:
            raise ValueError("A JAX PRNG key must be provided for shuffling.")
        keys = jax.random.split(key, n_vars)

    def one_var(i):
        base = primes[i % len(primes)]
        k = keys[i] if shuffle else None
        # if scramble:
        #     # Generate a random permutation for this base
        #     perm = jax.random.permutation(k, jnp.arange(base))
        #     seq = halton_seq_jax(
        #         sample_size * n_draws, base, drop=drop, shuffle=shuffle, key=k, scramble=True, perm=perm
        #     )
        # else:
        seq = halton_seq_jax(sample_size * n_draws, base, drop=drop, shuffle=shuffle, key=k)
        return jax.scipy.stats.norm.ppf(seq.reshape(sample_size, n_draws))

    draws = jax.vmap(one_var)(jnp.arange(n_vars))
    draws = jnp.transpose(draws, (1, 0, 2))  # (sample_size, n_vars, n_draws)
    return draws


# def get_normal_halton_draws_jax(sample_size, n_draws, n_vars, drop=100, shuffle=False, key=None):
#     """Generate Halton draws and transform them to normal distribution."""
#     draws = generate_halton_draws_jax(
#         sample_size,
#         n_draws,
#         n_vars,
#         drop=drop,
#         shuffle=shuffle,
#         key=key,
#     )
#     # Transform to normal distribution
#     for k in range(n_vars):
#         draws = draws.at[:, k, :].set(jstats.norm.ppf(draws[:, k, :]))
#     return draws  # (sample_size, n_vars, n_draws)
