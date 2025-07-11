import logging

import jax
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
        fn_call = lambda x_: fn(x_, *args)[1][i]  # noqa: B023, E731
        hess_row = approx_fprime(x, fn_call, epsilon=eps)
        H = H.at[i, :].set(hess_row)

    Hinv = jnp.linalg.inv(H)
    return Hinv


def gradient(funct, x, *args):
    """Finite difference gradient approximation."""
    
    # # memory intensive for large x and large sample sizes
    # grad = jax.jacobian(funct, argnums=0)(jnp.array(x), *fargs)

    # TODO: implement batching

    # Finite differences, lowest memory usage but slowest
    eps = 1e-6
    n = x.size
    grad_shape = funct(x, *args).size
    grad = jnp.zeros((grad_shape, n))
    for i in range(n):
        x_plus = x.at[i].set(x[i] + eps)
        x_minus = x.at[i].set(x[i] - eps)
        grad = grad.at[:, i].set((funct(x_plus, *args) - funct(x_minus, *args)) / (2 * eps))
    return grad


def hessian(funct, x, *args):
    """Compute the Hessian of funct for variables x."""

    # # this is memory intensive for large x.
    #hess_fn = jax.jacfwd(jax.grad(funct))  # jax.hessian(neg_loglike)
    #H = hess_fn(jnp.array(x), *args)

    # # slower but less memory intensive - hessian_by_rows
    grad_funct = jax.grad(funct)
    def row(i):
        return jax.grad(lambda x_: grad_funct(x_, *args)[i])(x)
    H = jax.vmap(row)(jnp.arange(x.size))

    return H

