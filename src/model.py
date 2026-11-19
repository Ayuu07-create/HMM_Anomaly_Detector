import numpy as np
from hmmlearn.hmm import GaussianHMM


def fit_hmm(X, n_states, random_state=42):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        tol=1e-3,
        random_state=random_state,
        min_covar=1e-3,
        verbose=False,
    )

    model.fit(X)

    return model


def normalized_log_likelihood(model, X):
    return float(model.score(X) / max(len(X), 1))


def bic_score(model, X):
    n, d = X.shape
    k = model.n_components

    # Parameters:
    # means = k*d
    # diagonal variances = k*d
    # transition matrix = k*(k-1)
    # initial probabilities = k-1
    params = (
        k * d
        + k * d
        + k * (k - 1)
        + (k - 1)
    )

    return float(
        -2 * model.score(X)
        + params * np.log(max(n, 2))
    )


def select_hmm_model(X_train, X_val, states, random_state=42):

    rows = []
    best_model = None
    best_key = None

    for k in states:

        print(f"\nTraining HMM with {k} hidden states...")

        try:

            model = fit_hmm(
                X_train,
                k,
                random_state=random_state
            )

            val_ll = normalized_log_likelihood(
                model,
                X_val
            )

            bic = bic_score(
                model,
                X_val
            )

            rows.append({
                "states": k,
                "validation_log_likelihood": val_ll,
                "bic": bic,
            })

            print(
                f"  validation log-likelihood: {val_ll:.6f}"
            )
            print(
                f"  BIC: {bic:.2f}"
            )
            print(
                f"  converged: {model.monitor_.converged}"
            )
            print(
                f"  iterations: {model.monitor_.iter}"
            )

            key = (-val_ll, bic)

            if best_key is None or key < best_key:
                best_key = key
                best_model = model

        except Exception as exc:

            print(
                f"  FAILED for {k} states: {exc}"
            )

            rows.append({
                "states": k,
                "validation_log_likelihood": np.nan,
                "bic": np.nan,
                "error": str(exc),
            })

    if best_model is None:
        raise RuntimeError(
            "No HMM candidate could be fitted successfully."
        )

    return best_model, rows