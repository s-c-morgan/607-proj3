import numpy as np


def _log_odds_from_counts(k, n):
    """log(k / (n - k)) for k successes out of n, without dividing by zero."""
    if k == 0 or k == n:
        raise ValueError(
            f"log odds undefined: {k} of {n} potential responses equal 1; "
            "the true differential log odds is infinite for this population")
    return np.log(k) - np.log(n - k)


def true_delta(YC, YT):
    """Differential log odds computed from the potential responses.

    Uses the counts directly, so the boundary cases (all 0s or all 1s, where
    the estimand is infinite) raise a clear error instead of returning inf
    with a divide-by-zero warning.
    """
    YC, YT = np.asarray(YC), np.asarray(YT)
    return (_log_odds_from_counts(int(YT.sum()), YT.size)
            - _log_odds_from_counts(int(YC.sum()), YC.size))
