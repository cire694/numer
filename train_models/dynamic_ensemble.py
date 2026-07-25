import lightgbm as lgb
from data import load_dataset, load_feature_groups
from config import Config
from train_models.Ensemble import fit_single
from utils import save_model

import numpy as np
import pandas as pd
from typing import List, Optional, Callable, Any, Union
from joblib import Parallel, delayed
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

"""
A neural network that's trained to weigh the predictions of each tree's output.

The era network takes in ALL features (including features not in the union of the tree's input features)

Network architecture: 
1. Turn each feature value into a one-hot vector of dim 5
2. Categorical embedding layer take advantage of 5-bin structure. (hidden_dim = 2 or 3)
    - (num_features, 5) @ (5, hidden_dim) -> (num_features, hidden_dim)
3. Two Linear layer to learn the weights. 
    -  (num_features * hidden_dim, ) ->  (64 or 128, ) -> (num_trees,)

Then we softmax the network's output to ensure the weights sum to 1. 


This idea is based on the following hypothesis: 
- Trees are good at local geometry of feature space
- Neural networks are good at global geometry of feature space


From NumerAI documentation:
The target is a measure of stock market returns over the next 20 (business) days. Specifically, it is a measure of "stock-specific" returns that are not explained by well-known "factors" or broader trends in the market, country, or sector. For example, if Apple went up and the tech sector also went up, we only want to know if Apple went up more or less than the tech sector.
Target values are binned into 5 unequal bins: 0, 0.25, 0.5, 0.75, 1.0. Again, this heavy regularization of target values is to avoid overfitting as the underlying values are extremely noisy.

The features are quantitative attributes of each stock: fundamentals like P/E ratio, technical signals like RSI, market data like short interest, secondary data like analyst ratings, and much more.
The underlying definition of each feature is not important, just know that Numerai has included these features in the dataset because we believe they are predictive of the target either by themselves or in combination with other features.
Feature values are binned into 5 equal bins: 0, 1, 2, 3, 4. This heavy regularization of feature values is to avoid overfitting as the underlying values are extremely noisy. Unlike the target, these are integers instead of floats to reduce the storage needs of the overall dataset.
If data for a particular feature is missing for that era (more common in early eras), then all values will be set to 2.
"""

# I think I'll train the following 
#   - Ensemble LGBM on group-specific feature subsets
#   - Add global target trees for ender_20, ender_60, jasper_20, jasper_60
# Then on top we'll train the weighted NN model
# This gives us 12 feature-group trees + 4 global trees = 16 trees.




class DynamicEnsemble(nn.Module):
    """Flexible ensemble of heterogeneous base models.

    Each base model is trained independently, then a neural meta head learns
    instance-specific weights for those base model predictions.
    """

    def __init__(self, embedding_dim: int = 2) -> None:
        super().__init__()
        self.ensemble_models: List[tuple[str, Any]] = []
        self.features: Optional[List[List[str]]] = None
        self.meta_features: Optional[List[str]] = None
        self.embedding_dim = embedding_dim
        self.head: Optional[nn.Embedding] = None
        self.mlp: Optional[nn.Module] = None
        self.register_buffer("offsets", torch.empty(0, dtype=torch.long))

    def add_model(self, name: str, model: Any) -> "DynamicEnsemble":
        """Register a base model in the ensemble.

        Args:
            name: Human-readable identifier for the base model.
            model: An object implementing .predict(X) and optionally .fit(X, y).

        Returns:
            self: Allows chaining of add_model() calls.
        """
        self.ensemble_models.append((name, model))
        return self

    def _build_meta_head(self, num_meta_features: int, num_models: int, hidden_dim: int = 64) -> None:
        """Build the meta head embedding + MLP for dynamic weights.

        Args:
            num_meta_features: Number of unique meta features used by the head.
            num_models: Number of base models in the ensemble.
            hidden_dim: Hidden layer size for the meta MLP.
        """
        self.head = nn.Embedding(5 * num_meta_features, self.embedding_dim)
        self.register_buffer("offsets", torch.arange(0, num_meta_features * 5, 5, dtype=torch.long))
        self.mlp = nn.Sequential(
            nn.Linear(num_meta_features * self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_models),
        )

    def _meta_feature_indices(self, X: pd.DataFrame) -> torch.LongTensor:
        """Convert meta features into embedding table indices.

        Args:
            X: DataFrame containing the meta feature columns.

        Returns:
            LongTensor of shape (n_rows, n_meta_features) containing embedding indices.
        """
        if self.meta_features is None:
            raise RuntimeError("DynamicEnsemble meta features are not initialized.")
        x = X.loc[:, self.meta_features].fillna(2).astype(np.int64).to_numpy()
        if x.max() >= 5 or x.min() < 0:
            raise ValueError("Meta feature values must be in [0, 4].")
        indices = torch.from_numpy(x)
        return indices + self.offsets.to(indices.device)

    def _encode_meta_features(self, X: pd.DataFrame) -> torch.Tensor:
        """Encode meta features into a flattened embedding tensor.

        Args:
            X: DataFrame containing the meta feature columns.

        Returns:
            Tensor of shape (n_rows, n_meta_features * embedding_dim).
        """
        if self.head is None:
            raise RuntimeError("DynamicEnsemble embedding head is not initialized.")
        indices = self._meta_feature_indices(X)
        embeddings = self.head(indices)
        return embeddings.view(embeddings.shape[0], -1)

    def _compute_weights(self, X: pd.DataFrame) -> torch.Tensor:
        """Compute softmax weights for each base model.

        Args:
            X: DataFrame containing the meta feature columns.

        Returns:
            Tensor of shape (n_rows, n_models) containing per-model weights.
        """
        if self.mlp is None:
            raise RuntimeError("DynamicEnsemble meta MLP is not initialized.")
        embeddings = self._encode_meta_features(X)
        logits = self.mlp(embeddings)
        return torch.softmax(logits, dim=1)

    def _train_meta_head(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        val_X: Optional[pd.DataFrame] = None,
        val_y: Optional[pd.Series] = None,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 4096,
    ) -> None:
        """Train the meta head to weight base model predictions.

        Args:
            X: Training DataFrame containing meta features.
            y: Training target series.
            val_X: Optional validation DataFrame containing meta features.
            val_y: Optional validation target series.
            epochs: Number of training epochs for the meta head.
            lr: Learning rate for the meta head optimizer.
            batch_size: Batch size for meta head training.

        Returns:
            None
        """
        if self.mlp is None or self.head is None:
            raise RuntimeError("DynamicEnsemble meta head must be initialized before training.")

        X_meta = self._encode_meta_features(X)
        base_preds = []
        for (_, model), features in zip(self.ensemble_models, self.features):
            preds = model.predict(X[features])
            base_preds.append(torch.from_numpy(np.asarray(preds, dtype=np.float32)))
        base_preds = torch.stack(base_preds, dim=1)

        y_tensor = torch.from_numpy(y.astype(np.float32).to_numpy())
        dataset = torch.utils.data.TensorDataset(X_meta, base_preds, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, preds_batch, y_batch in loader:
                optimizer.zero_grad()
                weights = torch.softmax(self.mlp(X_batch), dim=1)
                ensemble_pred = (weights * preds_batch).sum(dim=1)
                loss = F.mse_loss(ensemble_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.shape[0]

            avg_loss = epoch_loss / len(dataset)
            if val_X is not None and val_y is not None:
                with torch.no_grad():
                    val_embeddings = self._encode_meta_features(val_X)
                    val_weights = torch.softmax(self.mlp(val_embeddings), dim=1)
                    val_base_preds = []
                    for (_, model), features in zip(self.ensemble_models, self.features):
                        val_base_preds.append(torch.from_numpy(np.asarray(model.predict(val_X[features]), dtype=np.float32)))
                    val_base_preds = torch.stack(val_base_preds, dim=1)
                    val_pred = (val_weights * val_base_preds).sum(dim=1)
                    val_loss = F.mse_loss(val_pred, torch.from_numpy(val_y.astype(np.float32).to_numpy()))
                print(f"[meta head] epoch={epoch+1}/{epochs} train_loss={avg_loss:.6f} val_loss={val_loss:.6f}")
            else:
                print(f"[meta head] epoch={epoch+1}/{epochs} train_loss={avg_loss:.6f}")

    def fit(
        self,
        train: pd.DataFrame,
        ensemble_features: Union[List[str], List[List[str]]],
        targets: Optional[Union[List[str], str]] = None,
        val: Optional[pd.DataFrame] = None,
        callbacks: Optional[List[Callable]] = None,
        n_jobs: int = -1,
        head_hidden_dim: int = 64,
        head_epochs: int = 20,
        head_lr: float = 1e-3,
        head_batch_size: int = 4096,
    ) -> "DynamicEnsemble":
        """Fit the base models and train the dynamic meta head.

        Args:
            train: Training DataFrame containing features, era, and target columns.
            ensemble_features: Per-model feature sets, either a flat list or list of lists.
            targets: Target column name or list of target column names.
            val: Optional validation DataFrame for base model early stopping and meta head validation.
            callbacks: Optional callbacks passed to base model fit() implementations.
            n_jobs: Number of parallel jobs for training base models.
            head_hidden_dim: Hidden dimension size for the meta head MLP.
            head_epochs: Number of epochs to train the meta head.
            head_lr: Learning rate for the meta head optimizer.
            head_batch_size: Batch size for meta head training.

        Returns:
            self: The fitted DynamicEnsemble instance.
        """
        n_models = len(self.ensemble_models)
        if n_models == 0:
            raise RuntimeError("No base models added. Call add_model() before fit().")

        if targets is None:
            targets = ["target"] * n_models
        elif isinstance(targets, str):
            targets = [targets] * n_models

        if len(targets) != n_models:
            raise ValueError(
                f"targets length ({len(targets)}) must match number of models ({n_models})"
            )

        if len(ensemble_features) == 0:
            raise ValueError("Ensemble features must be non-empty")

        if isinstance(ensemble_features[0], str):
            per_model_features = [ensemble_features] * n_models
        else:
            per_model_features = list(ensemble_features)
            if len(per_model_features) != n_models:
                raise ValueError(
                    f"features length ({len(per_model_features)}) must match "
                    f"number of models ({n_models}) when passing per-model feature lists"
                )

        self.features = per_model_features
        self.meta_features = sorted({f for subset in self.features for f in subset})
        self._build_meta_head(len(self.meta_features), n_models, hidden_dim=head_hidden_dim)

        if n_jobs == -1:
            n_cores = os.cpu_count()
        elif n_jobs < 0:
            n_cores = max(1, os.cpu_count() + n_jobs + 1)
        else:
            n_cores = n_jobs

        results = Parallel(n_jobs=n_cores, backend='loky', max_nbytes='1G')(
            delayed(fit_single)(name, model, train, features, target_col, val, callbacks)
            for (name, model), features, target_col in zip(self.ensemble_models, self.features, targets)
        )
        self.ensemble_models = results

        meta_target = targets[0]
        val_X = val[self.meta_features] if val is not None and meta_target in val.columns else None
        val_y = val[meta_target] if val is not None and meta_target in val.columns else None

        self._train_meta_head(
            train[self.meta_features],
            train[meta_target],
            val_X=val_X,
            val_y=val_y,
            epochs=head_epochs,
            lr=head_lr,
            batch_size=head_batch_size,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with the dynamic ensemble.

        Args:
            X: DataFrame containing the meta feature columns and the base model feature columns.

        Returns:
            np.ndarray: Final ensemble predictions of shape (n_rows,).
        """
        if not self.ensemble_models:
            raise RuntimeError("No models in ensemble. Call add_model() first.")
        if self.features is None or self.meta_features is None:
            raise RuntimeError("DynamicEnsemble has not been fit yet.")

        base_preds = []
        for (_, model), features in zip(self.ensemble_models, self.features):
            base_preds.append(torch.from_numpy(np.asarray(model.predict(X[features]), dtype=np.float32)))
        base_preds = torch.stack(base_preds, dim=1)
        weights = self._compute_weights(X)
        ensemble_pred = (weights * base_preds).sum(dim=1)
        return ensemble_pred.cpu().numpy()

    def __repr__(self) -> str:
        model_list = ", ".join(name for name, _ in self.ensemble_models)
        return f"DynamicEnsemble([{model_list}], meta_features={len(self.meta_features) if self.meta_features is not None else 0})"
        
    

        






if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = Config(model_name='dynamic_ensemble', feature_set="all")
    train, ensemble_features = load_dataset(config, split="train")
    print("finish loading train")
    val, _ = load_dataset(config, split="validation")
    print("finish loading val")

    params = {
        "num_threads": 4,
        "num_leaves": 64,
        "colsample_bytree": 0.1,
        "subsample": 0.7,
        "subsample_freq": 1,
        "learning_rate": 0.001,
        "n_estimators": 10_000,
        "min_child_samples": 1000,
        "verbose": -1,
        "reg_lambda": 5.0,
        "seed": seed,
        "deterministic": True,
    }

    feature_sets = load_feature_groups()

    print("finish loading feature groups")
    feature_group_names = [
        'intelligence', 'charisma', 'strength', 'dexterity', 'constitution',
        'wisdom', 'agility', 'serenity', 'sunshine', 'rain', 'midnight', 'faith'
    ]

    missing = [g for g in feature_group_names if g not in feature_sets]
    if missing:
        raise KeyError(f"These groups aren't in feature_sets for this data version: {missing}")

    loaded_features = set(ensemble_features)
    per_model_features = []
    for g in feature_group_names:
        group_feats = [f for f in feature_sets[g] if f in loaded_features]
        if not group_feats:
            raise ValueError(f"Group '{g}' has zero overlap with loaded features — check feature_size used in load_dataset")
        per_model_features.append(group_feats)

    global_targets = [
        'target_ender_20',
        'target_ender_60',
        'target_jasper_20',
        'target_jasper_60',
    ]
    missing_targets = [t for t in global_targets if t not in train.columns]
    if missing_targets:
        raise KeyError(f"These global target columns are missing from train data: {missing_targets}")

    # Add a full-feature model for each global target.
    for _ in global_targets:
        per_model_features.append(ensemble_features)

    ensemble = DynamicEnsemble(embedding_dim=2)
    for group_name in feature_group_names:
        ensemble.add_model(group_name, lgb.LGBMRegressor(**params))
    for target_name in global_targets:
        ensemble.add_model(target_name, lgb.LGBMRegressor(**params))

    print("finish smoke test")
    print("running dynamic ensemble: each tree trains on a different feature group")
    print(f"Number of base models: {len(feature_group_names) + len(global_targets)}")
    early_stop_callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=False),
    ]
    ensemble.fit(
        train,
        per_model_features,
        targets=["target"] * len(feature_group_names) + global_targets,
        val=val,
        callbacks=early_stop_callbacks,
        n_jobs=4,
        head_hidden_dim=64,
        head_epochs=20,
        head_lr=1e-3,
        head_batch_size=4096,
    )

    print("saving model")
    last_train_era = int(train["era"].unique()[-1])
    save_model(ensemble, config, last_train_era=last_train_era)
    print("finished saving successfully")


    
