import logging

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats

from ._choice_model import ChoiceModel, diff_nonchosen_chosen
from ._optimize import _minimize, fd_grad
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

"""
Notations
---------
    N : Number of choice situations
    P : Number of observations per panel
    J : Number of alternatives
    K : Number of variables (Kf: fixed, Kr: random)
"""


class MixedLogit(ChoiceModel):
    """Class for estimation of Mixed Logit Models."""

    def __init__(self):
        super(MixedLogit, self).__init__()
        self._rvidx = None  # Index of random variables (True when random var)
        self._rvdist = None  # List of mixing distributions of rand vars

    def _setup_input_data(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        randvars,
        isvars=None,
        weights=None,
        avail=None,
        panels=None,
        init_coeff=None,
        random_state=None,
        n_draws=200,
        halton=True,
        predict_mode=False,
        halton_opts=None,
        scale_factor=None,
        include_correlations=False,
    ):
        # TODO: replace numpy random structure with jax
        if random_state is not None:
            np.random.seed(random_state)

        self._check_long_format_consistency(ids, alts)
        y = self._format_choice_var(y, alts) if not predict_mode else None
        X, Xnames = self._setup_design_matrix(X)
        self._model_specific_validations(randvars, Xnames)

        N, J, K = X.shape[0], X.shape[1], X.shape[2]
        num_random_params = len(randvars)
        Ks = 1 if scale_factor is not None else 0
        # lower triangular matrix elements of correlations for random variables, minus the diagonal
        num_cholesky_params = (
            0 if not include_correlations else num_random_params * (num_random_params + 1) // 2 - num_random_params
        )

        if panels is not None:
            # Convert panel ids to indexes
            panels = panels.reshape(N, J)[:, 0]
            panels_idx = np.empty(N)
            for i, u in enumerate(np.unique(panels)):
                panels_idx[np.where(panels == u)] = i
            panels = panels_idx.astype(int)

        # Reshape arrays in the format required for the rest of the estimation
        X = X.reshape(N, J, K)
        y = y.reshape(N, J, 1) if not predict_mode else None

        if not predict_mode:
            self._setup_randvars_info(randvars, Xnames)
        self.n_draws = n_draws

        if avail is not None:
            avail = avail.reshape(N, J)

        # Generate draws
        n_samples = N if panels is None else np.max(panels) + 1
        draws = self._generate_draws(n_samples, n_draws, halton, halton_opts=halton_opts)
        draws = draws if panels is None else draws[panels]  # (N,num_random_params,n_draws)

        if weights is not None:  # Reshape weights to match input data
            weights = weights.reshape(N, J)[:, 0]
            if panels is not None:
                panel_change_idx = np.concatenate(([0], np.where(panels[:-1] != panels[1:])[0] + 1))
                weights = weights[panel_change_idx]

        # initial values for coefficients. One for each provided variable, plus a std dev for each random variable,
        # plus a scale factor if provided, plus correlation coefficients for random variables if requested.
        num_coeffs = K + num_random_params + num_cholesky_params + Ks
        if init_coeff is None:
            betas = np.repeat(0.1, num_coeffs)
        else:
            betas = init_coeff
            if len(init_coeff) != num_coeffs:
                raise ValueError(f"The length of init_coeff must be {num_coeffs}, but got {len(init_coeff)}.")

        # Add std dev and correlation coefficients to the coefficient names
        coef_names = np.append(Xnames, np.char.add("sd.", Xnames[self._rvidx]))
        if include_correlations:
            corr_names = [
                f"corr.{i}.{j}" for num_, i in enumerate(Xnames[self._rvidx]) for j in Xnames[self._rvidx][num_ + 1 :]
            ]
            coef_names = np.append(coef_names, corr_names)

        if scale_factor is not None:
            coef_names = np.append(coef_names, "_scale_factor")

        assert len(coef_names) == num_coeffs, (
            f"Wrong number of coefficients set up, this is a data prep bug. Expected {num_coeffs}, got {len(coef_names)}. {coef_names}."
        )
        logger.debug(f"Set up {num_coeffs} initial coefficients for the model: {dict(zip(coef_names, betas))}")

        scale = None if scale_factor is None else scale_factor.reshape(N, J)

        return (
            jnp.array(betas),
            jnp.array(X),
            jnp.array(y),
            jnp.array(panels) if panels is not None else None,
            jnp.array(draws),
            jnp.array(weights) if weights is not None else None,
            jnp.array(avail) if avail is not None else None,
            Xnames,
            jnp.array(scale) if scale is not None else None,
            coef_names,
        )

    # TODO: split this into generic data prep and estimation specific data prep so we can reduce code duplication
    # for predictions. We should also wrap the data and model in different classes at some point.
    def data_prep_for_fit(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        randvars,
        isvars=None,
        weights=None,
        avail=None,
        panels=None,
        base_alt=None,
        fit_intercept=False,
        init_coeff=None,
        maxiter=2000,
        random_state=None,
        n_draws=1000,
        halton=True,
        halton_opts=None,
        fixedvars=None,
        scale_factor=None,
        include_correlations=False,
    ):
          # Handle array-like inputs by converting everything to numpy arrays
        (
            X,
            y,
            varnames,
            alts,
            isvars,
            ids,
            weights,
            panels,
            avail,
            scale_factor,
        ) = self._as_array(
            X,
            y,
            varnames,
            alts,
            isvars,
            ids,
            weights,
            panels,
            avail,
            scale_factor,
        )

        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights)

        # if mnl_init and init_coeff is None:
        #     # Initialize coefficients using a multinomial logit model
        #     logger.info("Pre-fitting MNL model as inital guess for MXL coefficients.")
        #     mnl = MultinomialLogit()
        #     mnl.fit(
        #         X,
        #         y,
        #         varnames,
        #         alts,
        #         ids,
        #         isvars=isvars,
        #         weights=weights,
        #         avail=avail,
        #         base_alt=base_alt,
        #         fit_intercept=fit_intercept,
        #         skip_std_errs=True,
        #     )
        #     init_coeff = np.concatenate((mnl.coeff_, np.repeat(0.1, len(randvars))))
        #     init_coeff = (
        #         init_coeff if scale_factor is None else np.append(init_coeff, 1.0)
        #     )

        logger.info(
            f"Starting data preparation, including generation of {n_draws} random draws for each random variable and observation."
        )

        self._pre_fit(alts, varnames, isvars, base_alt, fit_intercept, maxiter)

        betas, X, y, panels, draws, weights, avail, Xnames, scale, coef_names = self._setup_input_data(
            X,
            y,
            varnames,
            alts,
            ids,
            randvars,
            isvars=isvars,
            weights=weights,
            avail=avail,
            panels=panels,
            init_coeff=init_coeff,
            random_state=random_state,
            n_draws=n_draws,
            halton=halton,
            predict_mode=False,
            halton_opts=halton_opts,
            scale_factor=scale_factor,
            include_correlations=include_correlations,
        )

        # Mask fixed coefficients and set up array with values for the loglikelihood function
        mask = None
        values_for_mask = None
        if fixedvars is not None:
            mask = np.zeros(len(fixedvars), dtype=np.int32)
            values_for_mask = np.zeros(len(fixedvars), dtype=np.int32)
            for i, (k, v) in enumerate(fixedvars.items()):
                idx = np.where(coef_names == k)[0]
                if len(idx) == 0:
                    raise ValueError(f"Variable {k} not found in the model.")
                if len(idx) > 1:
                    raise ValueError(f"Variable {k} found more than once, this should never happen.")
                idx = idx[0]
                mask[i] = idx
                if v is not None:
                    betas = betas.at[idx].set(v)
                    values_for_mask[i] = v

            mask = jnp.array(mask)
            values_for_mask = jnp.array(values_for_mask)

        # panels are 0-based and contiguous by construction, so we can use the maximum value to determine the number
        # of panels. We provide this number explicitly to the log-likelihood function for jit compilation of
        # segment_sum (product of probabilities over panels)
        num_panels = 0 if panels is None else int(jnp.max(panels)) + 1

        # Set up index into _rvdist for lognormal distributions. This is used to apply the lognormal transformation
        # to the random betas
        idx_ln_dist = jnp.array([i for i, x in enumerate(self._rvdist) if x == "ln"], dtype=jnp.int32)

        # This here is estimation specific - we compute the difference between the chosen and non-chosen
        # alternatives because we only need the probability of the chosen alternative in the log-likelihood
        Xd, scale_d, avail = diff_nonchosen_chosen(X, y, scale, avail)  # Setup Xd as Xij - Xi*
        if scale_d is not None:
            # Multiply data by lambda coefficient when scaling is in use
            Xd = Xd * betas[-1]

        # split data for fixed and random parameters to speed up estimation
        rvidx = jnp.array(self._rvidx, dtype=bool)
        rand_idx = jnp.where(rvidx)[0]
        fixed_idx = jnp.where(~rvidx)[0]
        Xdf = Xd[:, :, ~rvidx]  # Data for fixed parameters
        Xdr = Xd[:, :, rvidx]  # Data for random parameters

        return (
            betas,
            Xdf,
            Xdr,
            panels,
            draws,
            weights,
            avail,
            scale_d,
            mask,
            values_for_mask,
            rvidx,
            rand_idx,
            fixed_idx,
            num_panels,
            idx_ln_dist,
            coef_names,
        )

    def fit(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        randvars,
        isvars=None,
        weights=None,
        avail=None,
        panels=None,
        base_alt=None,
        fit_intercept=False,
        init_coeff=None,
        maxiter=2000,
        random_state=None,
        n_draws=1000,
        halton=True,
        halton_opts=None,
        tol_opts=None,
        robust=False,
        num_hess=False,
        fixedvars=None,
        scale_factor=None,
        optim_method="L-BFGS-B",
        skip_std_errs=False,
        include_correlations=False,
    ):

        (
            betas,
            Xdf,
            Xdr,
            panels,
            draws,
            weights,
            avail,
            scale_d,
            mask,
            values_for_mask,
            rvidx,
            rand_idx,
            fixed_idx,
            num_panels,
            idx_ln_dist,
            coef_names,
        ) = self.data_prep_for_fit(
            X,
            y,
            varnames,
            alts,
            ids,
            randvars,
            isvars=isvars,
            weights=weights,
            avail=avail,
            panels=panels,
            base_alt=base_alt,
            fit_intercept=fit_intercept,
            init_coeff=init_coeff,
            maxiter=maxiter,
            random_state=random_state,
            n_draws=n_draws,
            halton=halton,
            halton_opts=halton_opts,
            fixedvars=fixedvars,
            scale_factor=scale_factor,
            include_correlations=include_correlations,
        )

        fargs = (
            Xdf,
            Xdr,
            panels,
            draws,
            weights,
            avail,
            scale_d,
            mask,
            values_for_mask,
            rvidx,
            rand_idx,
            fixed_idx,
            num_panels,
            idx_ln_dist,
            include_correlations,
        )

        logger.info("Compiling log-likelihood function.")
        jit_neg_loglike = jax.jit(neg_loglike, static_argnames=["num_panels", "include_correlations"])
        neg_loglik_and_grad = jax.value_and_grad(jit_neg_loglike, argnums=0)
        init_loglik = neg_loglik_and_grad(betas, *fargs)
        logger.info(f"Compilation finished, init neg_loglike = {init_loglik[0]:.2f}")

        def neg_loglike_scipy(betas, *args):
            """Wrapper for neg_loglike to use with scipy."""
            x = jnp.array(betas)
            return neg_loglik_and_grad(x, *args)

        tol = {
            "ftol": 1e-10,
            "gtol": 1e-6,
        }
        if tol_opts is not None:
            tol.update(tol_opts)

        # std dev needs to be positive
        bounds = [(-np.inf, np.inf) for _ in range(len(betas))]
        sd_start_idx = len(rvidx)
        sd_slice_size = len(rand_idx)
        for i in range(sd_start_idx, sd_start_idx + sd_slice_size):
            bounds[i] = (0, np.inf)

        optim_res = _minimize(
            neg_loglike_scipy,
            betas,
            args=fargs,
            method=optim_method,
            tol=tol["ftol"],
            options={
                "gtol": tol["gtol"],
                "maxiter": maxiter,
                "disp": True,
            },
            bounds=bounds,
        )
        if optim_res is None:
            logger.error("Optimization failed, returning None.")
            return None

        logger.info(f"Optimization finished, success = {optim_res['success']}, final loglike = {-optim_res['fun']:.2f}")

        # num_hess = num_hess if scale_factor is None else True

        # this is problematic because it consumes a lot of memory. Use finite diffs for now, implement batching later.
        # optim_res["grad_n"] = jax.jacobian(loglike_individual, argnums=0)(
        #     jnp.array(optim_res["x"]), *fargs
        # )

        if skip_std_errs:
            logger.info("Skipping H_inv and grad_n calculation due to skip_std_errs=True")
        else:
            try:
                logger.info("Calculating gradient of individual log-likelihood contributions")
                optim_res["grad_n"] = fd_grad(loglike_individual, jnp.array(optim_res["x"]), *fargs)

                logger.info("Calculating H_inv")
                hess_fn = jax.jacfwd(jax.grad(neg_loglike))  # jax.hessian(neg_loglike)
                H = hess_fn(jnp.array(optim_res["x"]), *fargs)
                # remove masked parameters to make it invertible
                if mask is not None:
                    mask_for_hessian = jnp.array([x for x in range(0, H.shape[0]) if x not in mask])
                    h_free = H[jnp.ix_(mask_for_hessian, mask_for_hessian)]
                    h_inv_nonfixed = jnp.linalg.inv(h_free)
                    h_inv = jnp.zeros_like(H)
                    h_inv = h_inv.at[jnp.ix_(mask_for_hessian, mask_for_hessian)].set(h_inv_nonfixed)
                else:
                    h_inv = jnp.linalg.inv(H)

                optim_res["hess_inv"] = h_inv
            # TODO: narrow down to actual error here
            except Exception as e:
                logger.error(f"Numerical Hessian calculation failed with {e} - parameters might not be identified")
                optim_res["hess_inv"] = jnp.eye(len(optim_res["x"]))

        self._post_fit(optim_res, coef_names, Xdf.shape[0], mask, fixedvars, skip_std_errs)
        return optim_res

    def _setup_randvars_info(self, randvars, Xnames):
        """Set up information about random variables and their mixing distributions.
        _rvidx: boolean array indicating which variables are random
        _rvdist: list of mixing distributions for each random variable
        """
        self.randvars = randvars
        self._rvidx, self._rvdist = [], []
        for var in Xnames:
            if var in self.randvars.keys():
                self._rvidx.append(True)
                self._rvdist.append(self.randvars[var])
            else:
                self._rvidx.append(False)
        self._rvidx = np.array(self._rvidx)

    # TODO: move draws to a separate file, use scipy.stats.qmc
    def _generate_draws(self, sample_size, n_draws, halton=True, halton_opts=None):
        """Generate draws based on the given mixing distributions."""
        if halton:
            draws = self._generate_halton_draws(
                sample_size,
                n_draws,
                len(self._rvdist),
                **halton_opts if halton_opts is not None else {},
            )
        else:
            draws = self._generate_random_draws(sample_size, n_draws, len(self._rvdist))

        for k, dist in enumerate(self._rvdist):
            if dist in ["n", "ln"]:  # Normal based
                draws[:, k, :] = jstats.norm.ppf(draws[:, k, :])
            elif dist == "t":  # Triangular
                draws_k = draws[:, k, :]
                draws[:, k, :] = (np.sqrt(2 * draws_k) - 1) * (draws_k <= 0.5) + (1 - np.sqrt(2 * (1 - draws_k))) * (
                    draws_k > 0.5
                )
            elif dist == "u":  # Uniform
                draws[:, k, :] = 2 * draws[:, k, :] - 1

        return draws  # (N,Kr,R)

    def _generate_random_draws(self, sample_size, n_draws, n_vars):
        """Generate random uniform draws between 0 and 1."""
        return np.random.uniform(size=(sample_size, n_vars, n_draws))

    def _generate_halton_draws(self, sample_size, n_draws, n_vars, shuffle=False, drop=100, primes=None):
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

    def _model_specific_validations(self, randvars, Xnames):
        """Conduct validations specific for mixed logit models."""
        if randvars is None:
            raise ValueError("The 'randvars' parameter is required for Mixed Logit estimation")
        if not set(randvars.keys()).issubset(Xnames):
            raise ValueError("Some variable names in 'randvars' were not found in the list of variable names")
        if not set(randvars.values()).issubset(["n", "ln", "t", "tn", "n_trunc", "u"]):
            raise ValueError("Wrong mixing distribution in 'randvars'. Accepted distrubtions are n, ln, t, u, tn")

    def summary(self):
        """Show estimation results in console."""
        super(MixedLogit, self).summary()


def _apply_distribution(betas_random, idx_ln_dist):
    """Apply the mixing distribution to the random betas."""

    if jax.config.jax_enable_x64:
        UTIL_MAX = 700  # ONLY IF 64bit precision is used
    else:
        UTIL_MAX = 87

    for i in idx_ln_dist:
        betas_random = betas_random.at[:, i, :].set(jnp.exp(betas_random[:, i, :].clip(-UTIL_MAX, UTIL_MAX)))
    return betas_random

def _transform_rand_betas(
    betas,
    draws,
    rand_idx,
    sd_start_index,
    sd_slice_size,
    chol_start_idx,
    chol_slice_size,
    idx_ln_dist,
    include_correlations,
):
    """Compute the products between the betas and the random coefficients.

    This method also applies the associated mixing distributions
    """
    br_mean = betas[rand_idx]
    # diag_vals = jax.nn.softplus(jax.lax.dynamic_slice(betas, (sd_start_index,), (sd_slice_size,)))
    diag_vals = jax.lax.dynamic_slice(betas, (sd_start_index,), (sd_slice_size,))

    if include_correlations:
        # Build lower-triangular Cholesky matrix
        tril_rows, tril_cols = jnp.tril_indices(sd_slice_size)
        L = jnp.zeros((sd_slice_size, sd_slice_size), dtype=betas.dtype)
        diag_mask = tril_rows == tril_cols
        # now using bounds in L-BFGS-B, use softmax when using optimization algorithm that does not support bounds
        L = L.at[tril_rows[diag_mask], tril_cols[diag_mask]].set(diag_vals)
        L = L.at[tril_rows[~diag_mask], tril_cols[~diag_mask]].set(
            jax.lax.dynamic_slice(betas, (chol_start_idx,), (chol_slice_size,))
        )
        N, _, R = draws.shape
        draws_flat = draws.transpose(0, 2, 1).reshape(-1, sd_slice_size)
        correlated_flat = (L @ draws_flat.T).T
        cov = correlated_flat.reshape(N, R, sd_slice_size).transpose(0, 2, 1)
    else:
        cov = draws * diag_vals[None, :, None]

    betas_random = br_mean[None, :, None] + cov
    # TODO: correlations are not straight forward when using anything but normal here
    betas_random = _apply_distribution(betas_random, idx_ln_dist)

    return betas_random

### TODO: re-write for JAX, make whole class derive from pytree, etc. Until then, this is a separate method.
def neg_loglike(
    betas,
    Xdf,
    Xdr,
    panels,
    draws,
    weights,
    avail,
    scale_d,
    mask,
    values_for_mask,
    rvdix,
    rand_idx,
    fixed_idx,
    num_panels,
    idx_ln_dist,
    include_correlations,
):
    loglik_individ = loglike_individual(
        betas,
        Xdf,
        Xdr,
        panels,
        draws,
        weights,
        avail,
        scale_d,
        mask,
        values_for_mask,
        rvdix,
        rand_idx,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        include_correlations,
    )

    loglik = loglik_individ.sum()
    return -loglik


def loglike_individual(
    betas,
    Xdf,
    Xdr,
    panels,
    draws,
    weights,
    avail,
    scale_d,
    mask,
    values_for_mask,
    rvdix,
    rand_idx,
    fixed_idx,
    num_panels,
    idx_ln_dist,
    include_correlations,
):
    """Compute the log-likelihood and gradient.

    Fixed and random parameters are handled separately to speed up the estimation and the results are concatenated.
    """

    if jax.config.jax_enable_x64:
        UTIL_MAX = 700  # ONLY IF 64bit precision is used
        LOG_PROB_MIN = 1e-300
    else:
        UTIL_MAX = 87
        LOG_PROB_MIN = 1e-30

    R = draws.shape[2]

    # mask for asserted parameters.
    if mask is not None:
        betas = betas.at[mask].set(values_for_mask)

    # Utility for fixed parameters
    Bf = betas[fixed_idx]  # Fixed betas
    Vdf = jnp.einsum("njk,k -> nj", Xdf, Bf)  # (N, J-1)

    sd_start_idx = len(rvdix)
    sd_slice_size = len(rand_idx)

    # these are onlu accessed when include_correlations is True
    chol_start_idx = sd_start_idx + sd_slice_size
    chol_slice_size = (sd_slice_size * (sd_slice_size + 1)) // 2 - sd_slice_size

    # Utility for random parameters
    Br = _transform_rand_betas(
        betas,
        draws,
        rand_idx,
        sd_start_idx,
        sd_slice_size,
        chol_start_idx,
        chol_slice_size,
        idx_ln_dist,
        include_correlations,
    )

    # Vdr shape: (N,J-1,R)
    Vd = Vdf[:, :, None] + jnp.einsum("njk,nkr -> njr", Xdr, Br)
    if scale_d is not None:
        Vd = Vd - (betas[-1] * scale_d)[:, :, None]
    eVd = jnp.exp(jnp.clip(Vd, -UTIL_MAX, UTIL_MAX))
    eVd = eVd if avail is None else eVd * avail[:, :, None]
    proba_n = 1 / (1 + eVd.sum(axis=1))  # (N,R)

    if panels is not None:
        # # no grads for segment_prod for non-unique panels. need to use sum of logs and then exp as workaround
        # proba_ = jax.ops.segment_prod(proba_n, panels, num_segments=num_panels)
        proba_n = jnp.exp(
            jnp.clip(
                jax.ops.segment_sum(
                    jnp.log(jnp.clip(proba_n, LOG_PROB_MIN, jnp.inf)),
                    panels,
                    num_segments=num_panels,
                ),
                -UTIL_MAX,
                UTIL_MAX,
            )
        )

    lik = proba_n.sum(axis=1) / R

    loglik = jnp.log(jnp.clip(lik, LOG_PROB_MIN, jnp.inf))

    if weights is not None:
        loglik = loglik * weights

    return loglik
    # loglik = loglik.sum()
    # return -loglik
