import logging

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np

logger = logging.getLogger(__name__)

## currently a pr at https://github.com/jax-ml/jax/pull/23808/files
def _newton_raphson(f, x, iters):
    """Use the Newton-Raphson method to find a root of the given function."""

    def update(x, _):
        y = x - f(x) / jax.grad(f)(x)
        return y, None

    x, _ = jax.lax.scan(update, x, length=iters)
    return x


def roberts_sequence(
    num: int,
    dim: int,
    root_iters: int = 10_000,
    complement_basis: bool = True,
    key=None,  ## ArrayLike | None = None,
    perturb: bool = False,
    shuffle: bool = False,
    dtype=float,  # DTypeLikeFloat = float,
):
    """Returns the Roberts sequence, a low-discrepancy quasi-random sequence:
    Low-discrepancy sequences are useful for quasi-Monte Carlo methods.
    Reference:
    Martin Roberts. The Unreasonable Effectiveness of Quasirandom Sequences.
    extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences
    Args:
      num: Number of points to return.
      dim: The dimensionality of each point in the sequence.
      root_iters: Number of iterations to use to find the root.
      complement_basis: Complement the basis to improve precision, as described
        in https://www.martysmods.com/a-better-r2-sequence.
      key: a PRNG key.
      perturb: Apply a uniformly random perturbation to the entire sequence,
        followed by modulo 1.
      shuffle: Shuffle the elements of the sequence before returning them.
        Warning: This degrades the low-discrepancy property for prefixes of
        the output sequence.
      dtype: optional, a float dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).
    Returns:
      An array of shape (num, dim) containing the sequence.
    """

    def f(x):
        return x ** (dim + 1) - x - 1

    root = _newton_raphson(f, jnp.astype(1.0, dtype), root_iters)

    basis = 1 / root ** (1 + jnp.arange(dim, dtype=dtype))

    if complement_basis:
        basis = 1 - basis

    n = jnp.arange(num, dtype=dtype)
    x = n[:, None] * basis[None, :]

    if perturb:
        if key is None:
            raise ValueError("key for roberts_sequence cannot be None when perturb=True")
        key, subkey = jax.random.split(key)
        x += jax.random.uniform(subkey, (dim,), dtype)[None]

    x, _ = jnp.modf(x)

    if shuffle:
        if key is None:
            raise ValueError("key for roberts_sequence cannot be None when shuffle=True")
        x = jax.random.permutation(key, x)

    return x


# TODO: have a look at scipy.stats.qmc, has sobol draws
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
