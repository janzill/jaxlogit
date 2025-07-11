import logging

import jax
import jax.numpy as jnp
import optimistix as optx

from collections.abc import Callable, Set
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

    logger.info(f"Running optimization with method: {method}")
    if method in ["L-BFGS-B", "BFGS"]:
        neg_loglik_and_grad = jax.value_and_grad(loglik_fn, argnums=0)
        def neg_loglike_scipy(betas, *args):
            """Wrapper for neg_loglike to use with scipy."""
            x = jnp.array(betas)
            return neg_loglik_and_grad(x, *args)

    if method == "L-BFGS-B":
        return minimize(
            neg_loglike_scipy,
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
            neg_loglike_scipy,
            x,
            args=args,
            jac=True,
            method="BFGS",
            options=options,
            callback=display_callback if options["disp"] else None,
        )
    elif method == "trust-region":
        class HybridSolver(optx.AbstractBFGS):
            rtol: float 
            atol: float
            norm: Callable = optx.max_norm  # max_norm, rms_norm, l2_norm
            use_inverse: bool = False  # need to set to false when using trust region methods.
            verbose: Set = frozenset({})
            descent: optx.AbstractDescent = optx.DoglegDescent()
            search: optx.AbstractSearch = optx.ClassicalTrustRegion()
            # standard BFGS uses optx.AbstractSearch = optx.BacktrackingArmijo()

        def neg_loglike_optx(betas, args):
           """Wrapper for neg_loglike to use with optx."""
           return loglik_fn(betas, *args)
        solver_optx = HybridSolver(rtol=1e-6, atol=1e-6, verbose=frozenset({"step_size", "loss"}))
        optx_result = optx.minimise(neg_loglike_optx, solver_optx, x, max_steps=2500, args=args)
        # TODO: wrap things up in proper result class, for now just use scipy's structure
        return {
            "x": optx_result.value,
            "fun": optx_result.state.f_info.f,
            "jac": optx_result.state.f_info.grad,
            "success": optx_result.result == optx.RESULTS.successful,
            "nit": optx_result.state.num_accepted_steps,
            "nfev": optx_result.stats["num_steps"],
            "njev": optx_result.state.num_accepted_steps,
            "message": "",
        }
    else:
        logger.error(f"Unknown optimization method: {method} exiting gracefully")
        return None


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


def hessian(funct, x, use_finite_diffs, *args):
    """Compute the Hessian of funct for variables x."""

    # # this is memory intensive for large x.
    #hess_fn = jax.jacfwd(jax.grad(funct))  # jax.hessian(neg_loglike)
    #H = hess_fn(jnp.array(x), *args)

    # # slower but less memory intensive - hessian_by_rows
    if not use_finite_diffs:
        grad_funct = jax.grad(funct)
        def row(i):
            return jax.grad(lambda x_: grad_funct(x_, *args)[i])(x)
        H = jax.vmap(row)(jnp.arange(x.size))
    else:
        # # even slower but lowest memory usage
        H = jnp.empty((len(x), len(x)))
        eps = 1.4901161193847656e-08  # From scipy 1.8 defaults
        for i in range(len(x)):
            fn_call = lambda x_: fn(x_, *args)[1][i]  # noqa: B023, E731
            hess_row = approx_fprime(x, fn_call, epsilon=eps)
            H = H.at[i, :].set(hess_row)
    return H
