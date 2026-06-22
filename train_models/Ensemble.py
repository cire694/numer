import numpy as np
import pandas as pd
from typing import List, Optional, Callable, Any
from joblib import Parallel, delayed
import os

class EnsembleModel:
    """A flexible ensemble of heterogeneous models.

    Each model is trained independently and predictions are combined
    via an aggregation function (default: mean). Models can be of any
    type as long as they implement .fit() and .predict().

    Supports heterogeneous ensembles (e.g. LightGBM + Ridge + NN),
    homogeneous ensembles (e.g. multiple LightGBM on different targets),
    and custom aggregation strategies (mean, median, weighted average).

    Attributes:
        models: List of (name, model) pairs.
        features: Feature column names, set during fit.
        agg_fn: Function that aggregates a (n_models, n_rows) array
            into a (n_rows,) array.
    """

    def __init__(
        self,
        agg_fn: Callable[[np.ndarray], np.ndarray] = None,
    ) -> None:
        """
        Args:
            agg_fn: Aggregation function applied to stacked predictions
                of shape (n_models, n_rows). Defaults to np.mean along axis=0.
                Examples:
                    np.median  → median across models
                    lambda x: np.average(x, axis=0, weights=[0.6, 0.4])  → weighted
        """
        self.models: List[tuple[str, Any]] = []
        self.features: Optional[List[str]] = None
        self.agg_fn = agg_fn or (lambda x: np.mean(x, axis=0))

    def add_model(self, name: str, model: Any) -> "EnsembleModel":
        """Register a model in the ensemble.

        The model must implement .predict(X). It does not need to
        implement .fit() if it is already trained before being added.

        Args:
            name: Human-readable identifier (used for logging).
            model: Any object with a .predict(X) method.

        Returns:
            self, to allow chaining: ensemble.add_model(...).add_model(...)
        """
        self.models.append((name, model))
        return self

    def _fit_single(
        name: str, 
        model: Any, 
        train: pd.DataFrame, 
        features: List[str],
        target_col: str,
    ) -> tuple[str, Any]:
        """Fit a single model - module-level so joblib can pickle it."""

        if not hasattr(model, "fit"):
            print(f"[{name}] has no .fit() - skipping")
            return name, model
        mask = train[target_col].notna()
        model.fit(train.loc[mask, features], train.loc[mask, target_col])
        return name, model

    def fit(
        self,
        train: pd.DataFrame,
        features: List[str],
        targets: Optional[List[str]] = None,
        n_jobs: int = -1
    ) -> "EnsembleModel":
        """Fit all registered models that implement .fit().

        Models that don't have a .fit() method (e.g. already-trained
        pretrained models) are silently skipped — they'll still be
        used during predict().

        Args:
            train: Training DataFrame.
            features: Feature column names.
            targets: List of target column names, one per model.
                If None, all models train on "target".
                If provided, must be the same length as self.models.
            n_jobs: number of parallel jobs. -1 uses all available cores
        Returns:
            self
        """
        self.features = features

        # default: all models train on the primary target
        if targets is None:
            targets = ["target"] * len(self.models)

        if len(targets) != len(self.models):
            raise ValueError(
                f"targets length ({len(targets)}) must match "
                f"number of models ({len(self.models)})"
            )
        # Resolve actual number of cores used
        if n_jobs == -1:
            n_cores = os.cpu_count()
        elif n_jobs < 0:
            n_cores = max(1, os.cpu_count() + n_jobs + 1)
        else:
            n_cores = n_jobs

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_single)(name, model, train, features, target_col)
            for (name, model), target_col in zip(self.models, targets)
        )
        self.models = results
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Aggregate predictions from all models.

        Args:
            X: DataFrame containing feature columns.

        Returns:
            Array of shape (n_rows,) with aggregated predictions.

        Raises:
            RuntimeError: If no models have been added.
        """
        if not self.models:
            raise RuntimeError("No models in ensemble. Call add_model() first.")
        if self.features is None:
            raise RuntimeError("Ensemble has not been fitted. Call fit() first.")

        all_preds = []
        for name, model in self.models:
            preds = model.predict(X[self.features])
            all_preds.append(preds)
            print(f"[{name}] predicted {len(preds)} rows")

        # shape: (n_models, n_rows)
        stacked = np.stack(all_preds, axis=0)
        return self.agg_fn(stacked)

    def __repr__(self) -> str:
        model_list = ", ".join(name for name, _ in self.models)
        return f"EnsembleModel([{model_list}], agg_fn={self.agg_fn.__name__})"