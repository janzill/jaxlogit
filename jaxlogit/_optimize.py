import logging

import jax.numpy as jnp
from scipy.optimize import minimize, approx_fprime

logger = logging.getLogger(__name__)


def _minimize(loglik_fn, x, args, method, tol, options, bounds=None):
    logger.info(f"Running minimization with method {method}")

    nit = 0  # global counter for display callback

    def display_callback(optim_res):
        nonlocal nit, loglik_fn, args
        nit += 1
        val, grad = loglik_fn(optim_res, *args)
        g_norm = jnp.linalg.norm(grad)
        logger.info(f"Iter {nit}, fun = {val:.3f}, |grad| = {g_norm:.3f}")  # , current sol = {optim_res}")

    if method == "L-BFGS-B":
        return minimize(
            loglik_fn,
            x,
            args=args,
            jac=True,
            method="L-BFGS-B",
            tol=tol,
            options=options,
            bounds=bounds,
            callback=display_callback if options["disp"] else None,
        )
    elif method == "BFGS":
        return minimize(
            loglik_fn,
            x,
            args=args,
            jac=True,
            method="BFGS",
            options=options,
            callback=display_callback if options["disp"] else None,
        )
    else:
        logger.error(f"Unknown optimization method: {method} exiting gracefully")
        return None


def _numerical_hessian(x, fn, args):
    H = jnp.empty((len(x), len(x)))
    eps = 1.4901161193847656e-08  # From scipy 1.8 defaults

    for i in range(len(x)):
        fn_call = lambda x_: fn(x_, *args)[1][i]
        hess_row = approx_fprime(x, fn_call, epsilon=eps)
        H = H.at[i, :].set(hess_row)

    Hinv = jnp.linalg.inv(H)
    return Hinv


def fd_grad(function, x, *args):
    """Finite difference gradient approximation."""
    eps = 1e-6
    n = x.size
    grad_shape = function(x, *args).size

    grad = jnp.zeros((grad_shape, n))
    for i in range(n):
        x_plus = x.at[i].set(x[i] + eps)
        x_minus = x.at[i].set(x[i] - eps)
        grad = grad.at[:, i].set((function(x_plus, *args) - function(x_minus, *args)) / (2 * eps))
    return grad
