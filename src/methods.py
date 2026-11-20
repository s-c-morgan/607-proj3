"""Estimators from Freedman (2008): logit MLE and the plug-in estimator."""
import warnings

import numpy as np

CONVERGED, MAX_ITER, SINGULAR_HESSIAN, SEPARATED, NONFINITE, FITTED_BOUNDARY = range(6)
STATUS = {CONVERGED: "converged", MAX_ITER: "max_iter",
          SINGULAR_HESSIAN: "singular_hessian", SEPARATED: "separated",
          NONFINITE: "nonfinite", FITTED_BOUNDARY: "fitted_boundary"}

BOUNDARY_ETA = 30.0

CHUNK_ELEMENTS = 1_000_000


class ConvergenceWarning(RuntimeWarning):
    """logit_mle did not converge to a finite MLE."""


def _expit_parts(t):
    """Return (pos, d, ed) with expit(|t|) = d and expit(-|t|) = ed.

    exp is only evaluated at -|t| <= 0, so its value lies in (0, 1] and cannot
    overflow. expit(t) is d where t >= 0 and ed elsewhere; expit(-t) is the
    other one, so 1 - p is never formed by subtraction.
    """
    t = np.asarray(t, dtype=float)
    e = np.exp(-np.abs(t))
    d = 1.0 / (1.0 + e)
    return t >= 0, d, e * d


def _expit_and_weight(t):
    """Return (expit(t), expit(t) * expit(-t)) from one exp, without overflow.

    The weight p(1 - p) = d * ed needs no 1 - p, so it does not round to 0
    when p rounds to 1 (|t| > ~37).
    """
    pos, d, ed = _expit_parts(t)
    return np.where(pos, d, ed), ed * d


def _expit(t):
    """Logistic function 1 / (1 + exp(-t)), without overflow.

    For |t| > ~745, exp(-|t|) underflows quietly to 0 and the result is
    exactly 0 or 1.
    """
    t = np.asarray(t, dtype=float)
    e = np.exp(-np.abs(t))
    d = 1.0 / (1.0 + e)
    return np.where(t >= 0, d, e * d)


def separated_by_treatment(X, Y):
    """True if either arm has a constant outcome (or is empty).

    Then the log-likelihood keeps increasing as beta1 -> +/-inf with
    beta2 -> -/+inf (control arm constant) or as beta2 -> +/-inf (treated arm
    constant), so the MLE does not exist. For 2-D X and Y of shape (S, n),
    returns a boolean array with one entry per row.
    """
    X, Y = np.asarray(X), np.asarray(Y)
    n = X.shape[-1]
    n_T = X.sum(axis=-1)
    y_T = (Y * X).sum(axis=-1)
    y_C = Y.sum(axis=-1) - y_T
    sep = (y_T == 0) | (y_T == n_T) | (y_C == 0) | (y_C == n - n_T)
    return sep if X.ndim > 1 else bool(sep)


def logit_mle(X, Y, Z, tol=1e-10, max_iter=100, full_output=False):
    """MLE (beta1, beta2, beta3) of the logit model P(Y=1) = expit(b1 + b2*X + b3*Z).

    The log-likelihood (12) is strictly concave, so when the MLE exists
    Newton-Raphson converges to the unique maximum. This is the single-fit
    interface to ``logit_mle_batch``.

    Returns beta, or (beta, info) if ``full_output`` is True, where info is a
    dict with keys ``status`` (one of the STATUS codes), ``n_iter`` and
    ``max_step`` (size of the last Newton step). If the fit fails, beta is all
    NaN; without ``full_output`` a ConvergenceWarning is also issued.
    """
    B, binfo = logit_mle_batch(np.asarray(X)[None], np.asarray(Y)[None], Z,
                               tol=tol, max_iter=max_iter)
    beta = B[0]
    info = dict(status=int(binfo["status"][0]), n_iter=int(binfo["n_iter"][0]),
                max_step=float(binfo["max_step"][0]))
    if info["status"] != CONVERGED and not full_output:
        warnings.warn(f"logit_mle failed: {STATUS[info['status']]} "
                      f"after {info['n_iter']} iterations",
                      ConvergenceWarning, stacklevel=2)
    return (beta, info) if full_output else beta


def logit_mle_batch(X, Y, Z, tol=1e-10, max_iter=100, chunk_elements=CHUNK_ELEMENTS):
    """Fit S logit models at once, one per row of X and Y, sharing covariate Z.

    X and Y have shape (S, n) and Z has shape (n,). Returns (beta, info):
    beta has shape (S, 3), and info holds arrays ``status``, ``n_iter`` and
    ``max_step`` of length S with the same meaning as in ``logit_mle``. Rows
    that fail have NaN beta; no warning is issued, so check ``info["status"]``.

    Rows are processed in chunks of about ``chunk_elements`` / n rows to bound
    the size of the (rows, n) temporaries.
    """
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    Z = np.asarray(Z, dtype=float)
    S, n = X.shape
    beta = np.full((S, 3), np.nan)
    status = np.full(S, MAX_ITER)
    n_iter = np.zeros(S, dtype=int)
    max_step = np.full(S, np.nan)
    rows = max(1, chunk_elements // max(n, 1))
    for lo in range(0, S, rows):
        sl = slice(lo, lo + rows)
        beta[sl], status[sl], n_iter[sl], max_step[sl] = _newton_batch(
            X[sl], Y[sl], Z, tol, max_iter)
    beta[status != CONVERGED] = np.nan
    return beta, dict(status=status, n_iter=n_iter, max_step=max_step)


def _newton_batch(X, Y, Z, tol, max_iter):
    """Newton-Raphson for every row of X, Y; finished rows drop out.

    With design rows (1, X_ij, Z_j), the gradient and Hessian are built from
    row sums and products with the shared Z, so no (S, n, 3) array is formed.
    """
    S = X.shape[0]
    beta = np.zeros((S, 3))
    status = np.full(S, MAX_ITER)
    n_iter = np.zeros(S, dtype=int)
    max_step = np.full(S, np.nan)

    sep = separated_by_treatment(X, Y)
    status[sep] = SEPARATED
    idx = np.flatnonzero(~sep)
    Xa = X[idx].astype(float)
    Ya = Y[idx].astype(float)
    b = beta[idx]
    Z2 = Z * Z

    for k in range(1, max_iter + 1):
        if idx.size == 0:
            break
        m = idx.size
        p, w = _expit_and_weight(_eta(b, Xa, Z))
        r = Ya - p
        wX = w * Xa
        g = np.column_stack([r.sum(axis=1), (r * Xa).sum(axis=1), r @ Z])
        h_cx, h_cz, h_xz = wX.sum(axis=1), w @ Z, wX @ Z
        H = np.empty((m, 3, 3))
        H[:, 0, 0] = w.sum(axis=1)
        H[:, 1, 1] = (wX * Xa).sum(axis=1)
        H[:, 2, 2] = w @ Z2
        H[:, 0, 1] = H[:, 1, 0] = h_cx
        H[:, 0, 2] = H[:, 2, 0] = h_cz
        H[:, 1, 2] = H[:, 2, 1] = h_xz
        step, ok = _solve_batch(H, g)

        n_iter[idx] = k
        b = np.where(ok[:, None], b + step, b)
        ms = np.abs(step).max(axis=1)
        max_step[idx[ok]] = ms[ok]

        singular = ~ok
        conv = ok & (ms < tol)
        nonfinite = ok & ~(ms < np.inf)
        status[idx[singular]] = SINGULAR_HESSIAN
        status[idx[nonfinite]] = NONFINITE
        if conv.any():
            boundary = np.abs(_eta(b[conv], Xa[conv], Z)).max(axis=1) > BOUNDARY_ETA
            status[idx[conv]] = np.where(boundary, FITTED_BOUNDARY, CONVERGED)

        done = singular | conv | nonfinite
        if done.any():
            beta[idx[done]] = b[done]
            keep = ~done
            idx, Xa, Ya, b = idx[keep], Xa[keep], Ya[keep], b[keep]
    beta[idx] = b
    return beta, status, n_iter, max_step


def _eta(b, X, Z):
    """Linear predictor b1 + b2*X + b3*Z for each row; b has shape (m, 3)."""
    return b[:, 0:1] + b[:, 1:2] * X + b[:, 2:3] * Z


def _solve_batch(H, g):
    """Solve H[i] step[i] = g[i] for every i; also return which were solvable.

    np.linalg.solve raises if any matrix in the stack is singular, so in that
    (rare) case the rows are solved one at a time to find which ones failed.
    """
    ok = np.ones(len(g), dtype=bool)
    try:
        return np.linalg.solve(H, g[..., None])[..., 0], ok
    except np.linalg.LinAlgError:
        step = np.zeros_like(g)
        for i in range(len(g)):
            try:
                step[i] = np.linalg.solve(H[i], g[i])
            except np.linalg.LinAlgError:
                ok[i] = False
        return step, ok


def plug_in(beta, Z):
    """Plug-in estimator of the differential log odds, eq. (10).

    Averages the predicted probabilities p(beta, 1, Z_i) and p(beta, 0, Z_i)
    over all subjects, then takes the difference of the log odds.
    Returns (delta_tilde, alpha_T_tilde, alpha_C_tilde); all NaN if beta is
    NaN (a failed fit). beta may also have shape (S, 3), one fit per row; then
    each returned value is an array of length S.
    """
    beta = np.asarray(beta, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if beta.ndim == 1:
        return _plug_in_rows(beta, Z)
    rows = max(1, CHUNK_ELEMENTS // max(Z.size, 1))
    parts = [_plug_in_rows(beta[lo:lo + rows], Z) for lo in range(0, len(beta), rows)]
    return tuple(np.concatenate(x) for x in zip(*parts))


def _plug_in_rows(beta, Z):
    b1, b2, b3 = beta[..., 0, None], beta[..., 1, None], beta[..., 2, None]
    eta_T = b1 + b2 + b3 * Z
    eta_C = b1 + b3 * Z
    aT, bT = _mean_expit_both(eta_T)
    aC, bC = _mean_expit_both(eta_C)
    delta = (np.log(aT) - np.log(bT)) - (np.log(aC) - np.log(bC))
    return delta, aT, aC


def _mean_expit_both(eta):
    """(mean(expit(eta)), mean(expit(-eta))) along the last axis; the second
    is 1 - the first, computed directly rather than by subtraction."""
    pos, d, ed = _expit_parts(eta)
    return np.where(pos, d, ed).mean(axis=-1), np.where(pos, ed, d).mean(axis=-1)
