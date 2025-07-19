import logging
import jax
import jax.numpy as jnp
import numpy as np

from ._choice_model import ChoiceModel, diff_nonchosen_chosen
from ._optimize import _minimize, gradient, hessian
# from .draws import roberts_sequence  # generate_draws

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

    def set_class_variables(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        randvars,
        weights,
        avail,
        panels,
        init_coeff,
        maxiter,
        random_state,
        n_draws,
        halton,
        halton_opts,
        tol_opts,
        num_hess,
        fixedvars,
        scale_factor,
        optim_method,
        skip_std_errs,
        include_correlations,
        force_positive_chol_diag,
        hessian_by_row,
    ):
        # Set class variables to enable simple pickling and running things post-estimation for analysis. This will be
        # replaced by proper database/dataseries structure in the future.
        self.X_raw = X
        self.y_raw = y
        self.varnames_raw = varnames
        self.alts_raw = (alts,)
        self.ids_raw = (ids,)
        self.randvars_raw = (randvars,)
        self.weights_raw = (weights,)
        self.avail_raw = (avail,)
        self.panels_raw = (panels,)
        self.init_coeff_raw = (init_coeff,)
        self.maxiter_raw = (maxiter,)
        self.random_state_raw = (random_state,)
        self.n_draws_raw = (n_draws,)
        self.halton_raw = (halton,)
        self.halton_opts_raw = (halton_opts,)
        self.tol_opts_raw = (tol_opts,)
        self.num_hess_raw = (num_hess,)
        self.fixedvars_raw = (fixedvars,)
        self.scale_factor_raw = (scale_factor,)
        self.optim_method_raw = (optim_method,)
        self.skip_std_errs_raw = (skip_std_errs,)
        self.include_correlations_raw = (include_correlations,)
        self.force_positive_chol_diag_raw = (force_positive_chol_diag,)
        self.hessian_by_row_raw = (hessian_by_row,)

    def _setup_input_data(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        randvars,
        weights=None,
        avail=None,
        panels=None,
        init_coeff=None,
        predict_mode=False,
        scale_factor=None,
        include_correlations=False,
    ):
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

        # if not predict_mode:
        self._setup_randvars_info(randvars, Xnames)
        # self.n_draws = n_draws

        if avail is not None:
            avail = avail.reshape(N, J)

        # Generate draws
        # n_samples = N if panels is None else np.max(panels) + 1
        # draws = generate_draws(n_samples, n_draws, self._rvdist, halton, halton_opts=halton_opts)
        # draws = draws if panels is None else draws[panels]  # (N,num_random_params,n_draws)

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
                f"chol.{i}.{j}" for idx_j, j in enumerate(Xnames[self._rvidx]) for i in Xnames[self._rvidx][:idx_j]
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
            None if predict_mode else jnp.array(y),
            jnp.array(panels) if panels is not None else None,
            jnp.array(weights) if weights is not None else None,
            jnp.array(avail) if avail is not None else None,
            Xnames,
            jnp.array(scale) if scale is not None else None,
            coef_names,
        )

    def data_prep(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        randvars,
        weights=None,
        avail=None,
        panels=None,
        init_coeff=None,
        maxiter=2000,
        fixedvars=None,
        scale_factor=None,
        include_correlations=False,
        predict_mode=False,
    ):
        # Handle array-like inputs by converting everything to numpy arrays
        (
            X,
            y,
            varnames,
            alts,
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
            ids,
            weights,
            panels,
            avail,
            scale_factor,
        )

        self._validate_inputs(X, y, alts, varnames, ids, weights, predict_mode=predict_mode)

        # logger.info(
        #     f"Starting data preparation, including generation of {n_draws} random draws for each random variable and observation."
        # )

        self._pre_fit(alts, varnames, maxiter)

        betas, X, y, panels, weights, avail, Xnames, scale, coef_names = self._setup_input_data(
            X,
            y,
            varnames,
            alts,
            ids,
            randvars,
            weights=weights,
            avail=avail,
            panels=panels,
            init_coeff=init_coeff,
            predict_mode=predict_mode,
            scale_factor=scale_factor,
            include_correlations=include_correlations,
        )

        # Mask fixed coefficients and set up array with values for the loglikelihood function
        mask = None
        values_for_mask = None
        # separate mask for fixing values of cholesky coeffs after softplus transformation
        mask_chol = []
        values_for_chol_mask = []
        sd_start_idx = len(self._rvidx)
        sd_slice_size = len(jnp.where(self._rvidx)[0])

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
                assert v is not None
                betas = betas.at[idx].set(v)
                values_for_mask[i] = v

                if (idx >= sd_start_idx) & (idx < sd_start_idx + sd_slice_size):
                    mask_chol.append(idx - sd_start_idx)
                    values_for_chol_mask.append(v)

            mask = jnp.array(mask)
            values_for_mask = jnp.array(values_for_mask)
            mask_chol = jnp.array(mask_chol, dtype=jnp.int32)
            values_for_chol_mask = jnp.array(values_for_chol_mask)

        if (fixedvars is None) or (len(mask_chol) == 0):
            mask_chol = None
            values_for_chol_mask = None

        # panels are 0-based and contiguous by construction, so we can use the maximum value to determine the number
        # of panels. We provide this number explicitly to the log-likelihood function for jit compilation of
        # segment_sum (product of probabilities over panels)
        num_panels = 0 if panels is None else int(jnp.max(panels)) + 1

        # Set up index into _rvdist for lognormal distributions. This is used to apply the lognormal transformation
        # to the random betas
        idx_ln_dist = jnp.array([i for i, x in enumerate(self._rvdist) if x == "ln"], dtype=jnp.int32)

        if not predict_mode:
            # This here is estimation specific - we compute the difference between the chosen and non-chosen
            # alternatives because we only need the probability of the chosen alternative in the log-likelihood
            Xd, scale_d, avail = diff_nonchosen_chosen(X, y, scale, avail)  # Setup Xd as Xij - Xi*
        else:
            scale_d = scale
            Xd = X

        if scale_d is not None:
            # Multiply data by lambda coefficient when scaling is in use
            Xd = Xd * betas[-1]

        # split data for fixed and random parameters to speed up calculations
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
            weights,
            avail,
            scale_d,
            mask,
            values_for_mask,
            mask_chol,
            values_for_chol_mask,
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
        randvars,  # TODO: check if this works for zero randvars
        weights=None,
        avail=None,
        panels=None,
        init_coeff=None,
        maxiter=2000,
        random_state=None,
        n_draws=1000,
        halton=True,
        halton_opts=None,
        tol_opts=None,
        num_hess=False,
        fixedvars=None,
        scale_factor=None,
        optim_method="trust-region",  # "trust-region", "L-BFGS-B", "BFGS"
        skip_std_errs=False,
        include_correlations=False,
        force_positive_chol_diag=True,  # use softplus for the cholesky diagonal elements
        hessian_by_row=True,  # calculate the hessian row by row in a for loop to save memory at the expense of runtime
        batch_size=None,
    ):
        # Set class variables to enable simple pickling and running things post-estimation for analysis. This will be
        # replaced by proper database/dataseries structure in the future.
        self.set_class_variables(
            X,
            y,
            varnames,
            alts,
            ids,
            randvars,
            weights,
            avail,
            panels,
            init_coeff,
            maxiter,
            random_state,
            n_draws,
            halton,
            halton_opts,
            tol_opts,
            num_hess,
            fixedvars,
            scale_factor,
            optim_method,
            skip_std_errs,
            include_correlations,
            force_positive_chol_diag,
            hessian_by_row,
        )

        (
            betas,
            Xdf,
            Xdr,
            panels,
            weights,
            avail,
            scale_d,
            mask,
            values_for_mask,
            mask_chol,
            values_for_chol_mask,
            rvidx,
            rand_idx,
            fixed_idx,
            num_panels,
            idx_ln_dist,
            coef_names,
        ) = self.data_prep(
            X,
            y,
            varnames,
            alts,
            ids,
            randvars,
            weights=weights,
            avail=avail,
            panels=panels,
            init_coeff=init_coeff,
            maxiter=maxiter,
            fixedvars=fixedvars,
            scale_factor=scale_factor,
            include_correlations=include_correlations,
        )

        if panels is None:
            N = Xdf.shape[0]  # Number of observations
        else:
            N = num_panels

        if batch_size is None:
            logger.info(f"Number of draws: {n_draws}.")
            num_batches = 1
            batch_shape = (N, len(rand_idx), n_draws)
        else:
            # TODO: has to be the same shape for all batches and equally divide all rands for jax.lax.scan
            assert n_draws % batch_size == 0, (
                f"Batch size {batch_size} does not divide the number of draws {n_draws} evenly "
                " but this is currently required."
            )
            num_batches = len(range(0, n_draws, batch_size))
            batch_shape = (N, len(rand_idx), batch_size)
            logger.info(
                f"Batch size {batch_size} for {n_draws} draws, {num_batches} batches, batch_shape={batch_shape}."
            )

        fargs = (
            Xdf,
            Xdr,
            panels,
            n_draws,
            weights,
            avail,
            scale_d,
            mask,
            values_for_mask,
            mask_chol,
            values_for_chol_mask,
            rvidx,
            rand_idx,
            fixed_idx,
            num_panels,
            idx_ln_dist,
            include_correlations,
            force_positive_chol_diag,
            num_batches,
            batch_shape,
        )

        if idx_ln_dist.shape[0] > 0:
            logger.info(
                f"Lognormal distributions found for {idx_ln_dist.shape[0]} random variables, applying transformation."
            )

        if scale_d is not None:
            logger.info("Scaling is in use, scaling the data by the scale factor.")

        if panels is not None:
            logger.info(f"Data contains {num_panels} panels.")

        logger.info(f"Shape of Xdf: {Xdf.shape}, shape of Xdr: {Xdr.shape}")

        tol = {
            "ftol": 1e-10,
            "gtol": 1e-6,
        }
        if tol_opts is not None:
            tol.update(tol_opts)

        init_loglike = neg_loglike(betas, *fargs)
        logger.info(f"Init loglike = {-init_loglike:.2f}.")

        # init_grad = jax.jacfwd(neg_loglike, argnums=0)(betas, *fargs)
        # init_grad_norm = jax.lax.stop_gradient(jnp.linalg.norm(init_grad))
        # logger.info(f"Init gradient_fwd norm = {init_grad_norm:.2f}, shape of grad = {init_grad.shape}.")
        init_grad = jax.grad(neg_loglike, argnums=0)(betas, *fargs)
        init_grad_norm = jax.lax.stop_gradient(jnp.linalg.norm(init_grad))
        logger.info(f"Init gradient norm = {init_grad_norm:.2f}, shape of grad = {init_grad.shape}.")

        optim_res = _minimize(
            neg_loglike,
            betas,
            args=fargs,
            method=optim_method,
            tol=tol["ftol"],
            options={
                "gtol": tol["gtol"],
                "maxiter": maxiter,
                "disp": True,
            },
        )
        if optim_res is None:
            logger.error("Optimization failed, returning None.")
            return None

        logger.info(
            f"Optimization finished, success = {optim_res['success']}, final loglike = {-optim_res['fun']:.2f}"
            + f", final gradient max = {optim_res['jac'].max():.2e}, norm = {jnp.linalg.norm(optim_res['jac']):.2e}."
        )

        if skip_std_errs:
            logger.info("Skipping H_inv and grad_n calculation due to skip_std_errs=True")
        else:
            logger.info("Calculating gradient of individual log-likelihood contributions")
            optim_res["grad_n"] = gradient(loglike_individual, jnp.array(optim_res["x"]), *fargs)

            try:
                logger.info("Calculating Hessian")
                H = hessian(neg_loglike, jnp.array(optim_res["x"]), hessian_by_row, *fargs)

                logger.info("Inverting Hessian")
                # remove masked parameters to make it invertible
                if mask is not None:
                    mask_for_hessian = jnp.array([x for x in range(0, H.shape[0]) if x not in mask])
                    h_free = H[jnp.ix_(mask_for_hessian, mask_for_hessian)]
                    h_inv_nonfixed = jax.lax.stop_gradient(jnp.linalg.inv(h_free))
                    h_inv = jnp.zeros_like(H)
                    h_inv = h_inv.at[jnp.ix_(mask_for_hessian, mask_for_hessian)].set(h_inv_nonfixed)
                else:
                    h_inv = jax.lax.stop_gradient(jnp.linalg.inv(H))

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

    def predict(
        self,
        X,
        varnames,
        alts,
        ids,
        randvars,
        init_coeff,
        weights=None,
        avail=None,
        panels=None,
        maxiter=2000,
        random_state=None,
        n_draws=1000,
        halton=True,
        halton_opts=None,
        scale_factor=None,
        include_correlations=False,
        softplus_chol_diag=True,
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
            mask_chol,
            values_for_chol_mask,
            rvidx,
            rand_idx,
            fixed_idx,
            num_panels,
            idx_ln_dist,
            coef_names,
        ) = self.data_prep(
            X,
            None,
            varnames,
            alts,
            ids,
            randvars,
            weights=weights,
            avail=avail,
            panels=panels,
            init_coeff=init_coeff,
            maxiter=maxiter,
            random_state=random_state,
            n_draws=n_draws,
            halton=halton,
            halton_opts=halton_opts,
            fixedvars=None,
            scale_factor=scale_factor,
            include_correlations=include_correlations,
            predict_mode=True,
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
            mask_chol,
            values_for_chol_mask,
            rvidx,
            rand_idx,
            fixed_idx,
            num_panels,
            idx_ln_dist,
            include_correlations,
            softplus_chol_diag,
        )

        probs = probability_individual(betas, *fargs)
        # uq_alts, idx = np.unique(alts, return_index=True)
        # uq_alts = uq_alts[np.argsort(idx)]
        # return pd.DataFrame(probs, columns=uq_alts)
        return probs


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
    idx_ln_dist,
    include_correlations,
    force_positive_chol_diag,
    mask_chol,
    values_for_chol_mask,
):
    """Compute the products between the betas and the random coefficients.

    This method also applies the associated mixing distributions
    """
    br_mean = betas[rand_idx]
    diag_vals = jax.lax.dynamic_slice(betas, (sd_start_index,), (sd_slice_size,))
    if force_positive_chol_diag:
        diag_vals = jax.nn.softplus(diag_vals)
        if mask_chol is not None:
            # Apply mask to the diagonal values of the Cholesky matrix again.
            # Could work around this by setting asserted params to softplus-1(x) but we also want to ensure
            # 0 values are propagated correctly for, e.g., ECs with less than full rank cov matrix.
            diag_vals = diag_vals.at[mask_chol].set(values_for_chol_mask)

    if include_correlations:
        chol_start_idx = sd_start_index + sd_slice_size
        chol_slice_size = (sd_slice_size * (sd_slice_size + 1)) // 2 - sd_slice_size

        # Build lower-triangular Cholesky matrix
        tril_rows, tril_cols = jnp.tril_indices(sd_slice_size)
        L = jnp.zeros((sd_slice_size, sd_slice_size), dtype=betas.dtype)
        diag_mask = tril_rows == tril_cols
        off_diag_mask = ~diag_mask
        off_diag_vals = jax.lax.dynamic_slice(betas, (chol_start_idx,), (chol_slice_size,))

        tril_vals = jnp.where(diag_mask, diag_vals[tril_rows], off_diag_vals[jnp.cumsum(off_diag_mask) - 1])
        L = L.at[tril_rows, tril_cols].set(tril_vals)

        N, _, R = draws.shape
        draws_flat = draws.transpose(0, 2, 1).reshape(-1, sd_slice_size)
        correlated_flat = (L @ draws_flat.T).T
        cov = correlated_flat.reshape(N, R, sd_slice_size).transpose(0, 2, 1)
    else:
        cov = draws * diag_vals[None, :, None]

    betas_random = br_mean[None, :, None] + cov
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
    mask_chol,
    values_for_chol_mask,
    rvdix,
    rand_idx,
    fixed_idx,
    num_panels,
    idx_ln_dist,
    include_correlations,
    force_positive_chol_diag,
    num_batches,
    batch_shape,
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
        mask_chol,
        values_for_chol_mask,
        rvdix,
        rand_idx,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        include_correlations,
        force_positive_chol_diag,
        num_batches,
        batch_shape,
    )

    loglik = loglik_individ.sum()
    return -loglik


def loglike_individual(
    betas,
    Xdf,
    Xdr,
    panels,
    draws,  # TEST: number to indicate number of draws, dynamically generated to avoid memory issues for large models
    weights,
    avail,
    scale_d,
    mask,
    values_for_mask,
    mask_chol,
    values_for_chol_mask,
    rvdix,
    rand_idx,
    fixed_idx,
    num_panels,
    idx_ln_dist,
    include_correlations,
    force_positive_chol_diag,
    num_batches,
    batch_shape,
):
    """Compute the log-likelihood.

    Fixed and random parameters are handled separately to speed up the estimation and the results are concatenated.
    """

    if jax.config.jax_enable_x64:
        UTIL_MAX = 700  # ONLY IF 64bit precision is used
        LOG_PROB_MIN = 1e-300
    else:
        UTIL_MAX = 87
        LOG_PROB_MIN = 1e-30

    if panels is None:
        N = Xdf.shape[0]  # Number of observations
    else:
        N = num_panels

    seed = 999
    key = jax.random.key(seed)
    subkeys = jax.random.split(key, num_batches)

    # mask for asserted parameters.
    if mask is not None:
        betas = betas.at[mask].set(values_for_mask)

    # Utility for fixed parameters
    Bf = betas[fixed_idx]  # Fixed betas
    Vdf = jnp.einsum("njk,k -> nj", Xdf, Bf)  # (N, J-1)

    sd_start_idx = len(rvdix)
    sd_slice_size = len(rand_idx)

    def batch_body(carry, subkey):
        ### TODO: try halton draws with normla.ppf and drop batch_number * (num_obs * num_rand_vars) for each batch;
        ###  will need array of batch numbers to compile
        draws_batched = jax.random.normal(subkey, shape=batch_shape)
        ###
        # draws_batched = roberts_sequence(batch_shape[2] * batch_shape[0], batch_shape[1], key=subkey).reshape(
        #     batch_shape
        # )  # shuffle=True
        # for k in range(batch_shape[1]):
        #     draws_batched = draws_batched.at[:, k, :].set(jax.scipy.stats.norm.ppf(draws_batched[:, k, :]))

        if panels is not None:
            draws_batched = draws_batched[panels]
        Br = _transform_rand_betas(
            betas,
            draws_batched,
            rand_idx,
            sd_start_idx,
            sd_slice_size,
            idx_ln_dist,
            include_correlations,
            force_positive_chol_diag,
            mask_chol,
            values_for_chol_mask,
        )
        Vd = Vdf[:, :, None] + jnp.einsum("njk,nkr -> njr", Xdr, Br)
        if scale_d is not None:
            Vd = Vd - (betas[-1] * scale_d)[:, :, None]
        eVd = jnp.exp(jnp.clip(Vd, -UTIL_MAX, UTIL_MAX))
        if avail is not None:
            eVd = eVd * avail[:, :, None]
        proba_n = 1 / (1 + eVd.sum(axis=1))

        if panels is not None:
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
        carry = carry + proba_n.sum(axis=1)
        return carry, None

    total_lik, _ = jax.lax.scan(batch_body, jnp.zeros((N,)), subkeys)

    loglik = jnp.log(jnp.clip(total_lik / draws, LOG_PROB_MIN, jnp.inf))

    if weights is not None:
        loglik = loglik * weights

    return loglik


def probability_individual(
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
    mask_chol,
    values_for_chol_mask,
    rvdix,
    rand_idx,
    fixed_idx,
    num_panels,
    idx_ln_dist,
    include_correlations,
    force_positive_chol_diag=True,
):
    """Compute the probabilities of all alternatives."""

    if jax.config.jax_enable_x64:
        UTIL_MAX = 700  # ONLY IF 64bit precision is used
    else:
        UTIL_MAX = 87

    R = draws.shape[2]

    # mask for asserted parameters.
    if mask is not None:
        betas = betas.at[mask].set(values_for_mask)

    # Utility for fixed parameters
    Bf = betas[fixed_idx]  # Fixed betas
    Vdf = jnp.einsum("njk,k -> nj", Xdf, Bf)  # (N, J)

    sd_start_idx = len(rvdix)
    sd_slice_size = len(rand_idx)
    Br = _transform_rand_betas(
        betas,
        draws,
        rand_idx,
        sd_start_idx,
        sd_slice_size,
        idx_ln_dist,
        include_correlations,
        force_positive_chol_diag,
        mask_chol,
        values_for_chol_mask,
    )

    # Vdr shape: (N,J,R)
    Vd = Vdf[:, :, None] + jnp.einsum("njk,nkr -> njr", Xdr, Br)
    if scale_d is not None:
        Vd = Vd - (betas[-1] * scale_d)[:, :, None]
    eVd = jnp.exp(jnp.clip(Vd, -UTIL_MAX, UTIL_MAX))
    eVd = eVd if avail is None else eVd * avail[:, :, None]

    proba_n = eVd / eVd.sum(axis=1)[:, None, :]  # (N,J,R)

    # TODO: check if this is still correct - might need to be over different dimension? - Let's leave this out for now
    # if panels is not None:
    #     # # no grads for segment_prod for non-unique panels. need to use sum of logs and then exp as workaround
    #     # proba_ = jax.ops.segment_prod(proba_n, panels, num_segments=num_panels)
    #     proba_n = jnp.exp(
    #         jnp.clip(
    #             jax.ops.segment_sum(
    #                 jnp.log(jnp.clip(proba_n, LOG_PROB_MIN, jnp.inf)),
    #                 panels,
    #                 num_segments=num_panels,
    #             ),
    #             -UTIL_MAX,
    #             UTIL_MAX,
    #         )
    #     )

    mean_proba_n = proba_n.sum(axis=2) / R

    return mean_proba_n
