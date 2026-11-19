"""Estimators from Freedman (2008): logit MLE and the plug-in estimator.

Numerical safeguards (see docs/BASELINE.md, Section 4):

* ``_expit`` evaluates exp only at -|t| <= 0, so it cannot overflow.
* The Newton weights p(1 - p) are computed from the same exp(-|eta|) without
  forming 1 - p by subtraction, so they stay positive until |eta| ~ 745
  instead of rounding to exactly 0 once eta exceeds ~37.
* ``logit_mle`` never fails silently. It reports one of the STATUS codes:
  data where a treatment arm has a constant outcome (separation, so the MLE
  does not exist) are detected before fitting; a singular Hessian, a
  non-finite step or hitting ``max_iter`` are caught and reported; and a fit
  that "converges" with some fitted probability numerically 0 or 1
  (|eta| > BOUNDARY_ETA) is flagged, since that is separation through Z or
  through X and Z jointly, which the arm check cannot see. In every
  failure case beta is returned as NaN, so it cannot leak into summaries as an
  arbitrary finite number.
* ``plug_in`` computes 1 - alpha directly as mean(expit(-eta)) rather than by
  subtraction.
"""
import warnings

import numpy as np

# Fit status codes, stored as integers in results/raw/estimates.pk
CONVERGED, MAX_ITER, SINGULAR_HESSIAN, SEPARATED, NONFINITE, FITTED_BOUNDARY = range(6)
STATUS = {CONVERGED: "converged", MAX_ITER: "max_iter",
          SINGULAR_HESSIAN: "singular_hessian", SEPARATED: "separated",
          NONFINITE: "nonfinite", FITTED_BOUNDARY: "fitted_boundary"}

# A fitted probability within ~1e-13 of 0 or 1 (|eta| > 30) means the
# likelihood is being maximized at infinity. In the study, |eta| < 6.
BOUNDARY_ETA = 30.0


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
    constant), so the MLE does not exist.
    """
    X, Y = np.asarray(X), np.asarray(Y)
    n_T, n = int(X.sum()), X.size
    y_T = int(Y @ X)                 # successes among treated
    y_C = int(Y.sum()) - y_T         # successes among controls
    return y_T in (0, n_T) or y_C in (0, n - n_T)


def logit_mle(X, Y, Z, tol=1e-10, max_iter=100, full_output=False):
    """MLE (beta1, beta2, beta3) of the logit model P(Y=1) = expit(b1 + b2*X + b3*Z).

    The log-likelihood (12) is strictly concave, so when the MLE exists
    Newton-Raphson converges to the unique maximum.

    Returns beta, or (beta, info) if ``full_output`` is True, where info is a
    dict with keys ``status`` (one of the STATUS codes), ``n_iter`` and
    ``max_step`` (size of the last Newton step). If the fit fails, beta is all
    NaN; without ``full_output`` a ConvergenceWarning is also issued.
    """
    beta, info = _newton(X, Y, Z, tol, max_iter)
    if info["status"] != CONVERGED:
        beta = np.full(3, np.nan)
        if not full_output:
            warnings.warn(f"logit_mle failed: {STATUS[info['status']]} "
                          f"after {info['n_iter']} iterations",
                          ConvergenceWarning, stacklevel=2)
    return (beta, info) if full_output else beta


def _newton(X, Y, Z, tol, max_iter):
    info = dict(status=MAX_ITER, n_iter=0, max_step=np.nan)
    beta = np.zeros(3)
    if separated_by_treatment(X, Y):
        info["status"] = SEPARATED
        return beta, info
    M = np.column_stack([np.ones(len(Y)), X, Z])
    status, max_step = MAX_ITER, np.nan
    for k in range(1, max_iter + 1):
        p, w = _expit_and_weight(M @ beta)
        grad = M.T @ (Y - p)
        hess = (M * w[:, None]).T @ M
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            status = SINGULAR_HESSIAN
            break
        beta = beta + step
        max_step = np.abs(step).max()
        if max_step < tol:
            status = (FITTED_BOUNDARY if np.abs(M @ beta).max() > BOUNDARY_ETA
                      else CONVERGED)
            break
        if not max_step < np.inf:    # inf or NaN step
            status = NONFINITE
            break
    info.update(status=status, n_iter=k, max_step=max_step)
    return beta, info


def plug_in(beta, Z):
    """Plug-in estimator of the differential log odds, eq. (10).

    Averages the predicted probabilities p(beta, 1, Z_i) and p(beta, 0, Z_i)
    over all subjects, then takes the difference of the log odds.
    Returns (delta_tilde, alpha_T_tilde, alpha_C_tilde); all NaN if beta is
    NaN (a failed fit).
    """
    Z = np.asarray(Z, dtype=float)
    eta_T = beta[0] + beta[1] + beta[2] * Z
    eta_C = beta[0] + beta[2] * Z
    aT, bT = _mean_expit_both(eta_T)
    aC, bC = _mean_expit_both(eta_C)
    delta = (np.log(aT) - np.log(bT)) - (np.log(aC) - np.log(bC))
    return delta, aT, aC


def _mean_expit_both(eta):
    """(mean(expit(eta)), mean(expit(-eta))); the second is 1 - the first,
    computed directly rather than by subtraction."""
    pos, d, ed = _expit_parts(eta)
    return np.where(pos, d, ed).mean(), np.where(pos, ed, d).mean()
