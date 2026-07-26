import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Optional, Callable, Any, Union
from joblib import Parallel, delayed

from data import load_dataset, load_feature_groups
from config import Config
from train_models.Ensemble import fit_single
from utils import save_model

"""
A neural network that's trained to weigh the predictions of each tree's output.
The era network takes in ALL features (including features not in the union of the tree's input features)

Network architecture:
1. Turn each feature value into a one-hot vector of dim 5
2. Categorical embedding layer take advantage of 5-bin structure. (hidden_dim = 2 or 3)
- (num_features, 5) @ (5, hidden_dim) -> (num_features, hidden_dim)
3. Two Linear layer to learn the weights.
- (num_features * hidden_dim, ) -> (64 or 128, ) -> (num_trees,)

Then we softmax the network's output to ensure the weights sum to 1.

To prevent the network from memorizing specific training eras, we restrict the gate's inputs to a subset
of our choosing (e.g. numer's "small" feature subset)
Network architecture:
1. An ensemble of trees composed of a subset of features and eras, plus a few global trees trained on targets of 20 and 60 days
2. A MLP head which takes in the predictions of each tree, along with our chosen subset of features



Predicting:
1. Stack each tree's prediction
2. Use an embedding/lookup to find feature's numerical representation (e.g. if feature 3 has value 4, look up the value in 2*5 + 4)
3. Run it through a MLP, then softmax to get our weights
4. 

MLP:

if self.mlp is None:
            raise RuntimeError("DynamicEnsemble meta MLP is not initialized.")
        embeddings = self._encode_meta_features(X)
        logits = self.mlp(embeddings)
        return torch.softmax(logits, dim=1)

self.eval()  # Ensure dropout is disabled for inference
        with torch.no_grad():
            base_preds = []
            for (_, model), features in zip(self.ensemble_models, self.features):
                base_preds.append(torch.from_numpy(np.asarray(model.predict(X[features]), dtype=np.float32)))
            base_preds = torch.stack(base_preds, dim=1)
            weights = self._compute_weights(X)
            ensemble_pred = (weights * base_preds).sum(dim=1)
            return ensemble_pred.cpu().numpy()

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
# - Ensemble LGBM on group-specific feature subsets
# - Add global target trees for ender_20, ender_60, jasper_20, jasper_60
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
        self.embed_drop: Optional[nn.Dropout] = None
        self.register_buffer("offsets", torch.empty(0, dtype=torch.long))

    def add_model(self, name: str, model: Any) -> "DynamicEnsemble":
        """Register a base model in the ensemble."""
        self.ensemble_models.append((name, model))
        return self

    def _build_meta_head(
        self,
        num_meta_features: int,
        num_models: int,
        hidden_dim: int = 64,
        embed_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ) -> None:
        """Build embedding + MLP meta head.

        Args:
            num_meta_features: Number of distinct meta features (columns).
            num_models: Number of base models to output weights for.
            hidden_dim: Hidden dimension size for the MLP.
            embed_dropout: Dropout probability applied to embeddings.
            hidden_dropout: Dropout probability applied to MLP hidden layer.
        """
        # embedding table: 5 bins per feature, flattened into a single table
        self.head = nn.Embedding(5 * num_meta_features, self.embedding_dim)
        # offsets to convert per-column bin indices into the flattened table
        self.register_buffer("offsets", torch.arange(0, num_meta_features * 5, 5, dtype=torch.long))
        # optional dropout modules
        self.embed_drop = nn.Dropout(embed_dropout) if embed_dropout > 0.0 else None
        # MLP: embedding_dim * num_meta_features -> hidden -> num_models
        layers = [nn.Linear(num_meta_features * self.embedding_dim, hidden_dim), nn.ReLU()]
        if hidden_dropout > 0.0:
            layers.append(nn.Dropout(hidden_dropout))
        layers.append(nn.Linear(hidden_dim, num_models))
        self.mlp = nn.Sequential(*layers)


    def _meta_feature_indices(self, X: pd.DataFrame) -> torch.LongTensor:
        """Convert meta feature DataFrame to embedding table indices.

        Assumes `self.meta_features` is set and that feature values are
        integer bins in [0, 4]. Missing values are filled with 2.
        Returns a LongTensor of shape (n_rows, n_meta_features).
        """
        if self.meta_features is None:
            raise RuntimeError("DynamicEnsemble meta features are not initialized.")
        x = X.loc[:, self.meta_features].fillna(2).astype(np.int64).to_numpy()
        if x.max() >= 5 or x.min() < 0:
            raise ValueError("Meta feature values must be in [0, 4].")
        indices = torch.from_numpy(x)
        return indices + self.offsets.to(indices.device)

    def _encode_meta_features(self, X: pd.DataFrame) -> torch.Tensor:
        """Encode meta features via the embedding table and flatten.

        Returns a FloatTensor of shape (n_rows, n_meta_features * embedding_dim).
        """
        if self.head is None:
            raise RuntimeError("DynamicEnsemble embedding head is not initialized.")
        indices = self._meta_feature_indices(X)
        embeddings = self.head(indices)
        if self.embed_drop is not None:
            embeddings = self.embed_drop(embeddings)
        return embeddings.view(embeddings.shape[0], -1)

    def _compute_weights(self, X: pd.DataFrame) -> torch.Tensor:
        """Compute per-model softmax weights from meta features.

        Args:
            X: DataFrame containing the meta feature columns.

        Returns:
            Tensor of shape (n_rows, n_models) with softmax-normalized weights.
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
        entropy_coef: float = 0.02,
        weight_decay_head: float = 1e-2,
        weight_decay_mlp: float = 1e-4,
    ) -> None:
        """Train the meta head (embeddings + MLP) to weight base models.

        The head is trained to minimize MSE between the weighted ensemble
        prediction and the target, with an optional entropy regularization
        term to encourage spread across models.

        Args:
            X: Training DataFrame (meta feature columns).
            y: Training target series.
            val_X: Optional validation DataFrame for monitoring.
            val_y: Optional validation targets for monitoring.
            epochs: Number of training epochs.
            lr: Learning rate for the optimizer.
            batch_size: Batch size for training.
            entropy_coef: Coefficient for entropy regularization on weights.
        """
        if self.mlp is None or self.head is None:
            raise RuntimeError("DynamicEnsemble meta head must be initialized before training.")

        # Pre-compute base predictions
        self.eval()
        with torch.no_grad():
            X_meta = self._encode_meta_features(X)
            base_preds = []
            for (_, model), features in zip(self.ensemble_models, self.features):
                preds = model.predict(X[features])
                base_preds.append(torch.from_numpy(np.asarray(preds, dtype=np.float32)))
            base_preds = torch.stack(base_preds, dim=1)

        y_tensor = torch.from_numpy(y.astype(np.float32).to_numpy())
        dataset = torch.utils.data.TensorDataset(X_meta, base_preds, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # AdamW with grouped weight decay
        optimizer = torch.optim.AdamW([
            {"params": self.head.parameters(), "weight_decay": weight_decay_head},
            {"params": self.mlp.parameters(), "weight_decay": weight_decay_mlp},
        ], lr=lr)

        def entropy_penalty(weights: torch.Tensor) -> torch.Tensor:
            """Negative entropy. Minimizing this maximizes entropy (spreads weights)."""
            eps = 1e-8
            entropy = -(weights * (weights + eps).log()).sum(dim=1)
            return -entropy.mean()

        for epoch in range(epochs):
            self.train() # Enable Dropout and LayerNorm behavior
            epoch_loss = 0.0
            epoch_mse = 0.0
            
            for X_batch, preds_batch, y_batch in loader:
                optimizer.zero_grad()
                
                weights = torch.softmax(self.mlp(X_batch), dim=1)
                ensemble_pred = (weights * preds_batch).sum(dim=1)
                
                mse_loss = F.mse_loss(ensemble_pred, y_batch)
                reg_loss = entropy_coef * entropy_penalty(weights)
                
                loss = mse_loss + reg_loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * X_batch.shape[0]
                epoch_mse += mse_loss.item() * X_batch.shape[0]

            avg_loss = epoch_loss / len(dataset)
            avg_mse = epoch_mse / len(dataset)

            if val_X is not None and val_y is not None:
                self.eval() # Disable Dropout for validation
                with torch.no_grad():
                    val_embeddings = self._encode_meta_features(val_X)
                    val_weights = torch.softmax(self.mlp(val_embeddings), dim=1)
                    val_base_preds = []
                    for (_, model), features in zip(self.ensemble_models, self.features):
                        val_base_preds.append(torch.from_numpy(np.asarray(model.predict(val_X[features]), dtype=np.float32)))
                    val_base_preds = torch.stack(val_base_preds, dim=1)
                    val_pred = (val_weights * val_base_preds).sum(dim=1)
                    val_mse = F.mse_loss(val_pred, torch.from_numpy(val_y.astype(np.float32).to_numpy()))
                print(f"[meta head] epoch={epoch+1}/{epochs} train_mse={avg_mse:.6f} val_mse={val_mse:.6f}")
            else:
                print(f"[meta head] epoch={epoch+1}/{epochs} train_mse={avg_mse:.6f}")

    def fit(
        self,
        train: pd.DataFrame,
        ensemble_features: Union[List[str], List[List[str]]],
        meta_features: Optional[List[str]] = None,
        targets: Optional[Union[List[str], str]] = None,
        val: Optional[pd.DataFrame] = None,
        callbacks: Optional[List[Callable]] = None,
        n_jobs: int = -1,
        head_hidden_dim: int = 64,
        head_epochs: int = 20,
        head_lr: float = 1e-3,
        head_batch_size: int = 4096,
        entropy_coef: float = 0.02,
        embed_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        weight_decay_head: float = 1e-2,
        weight_decay_mlp: float = 1e-4,
    ) -> "DynamicEnsemble":
        """Fit base models and train the dynamic meta head.

        This first fits all registered base models (in parallel) and then
        fits the meta head on top of their predictions.

        Args:
            train: Training DataFrame with features and target columns.
            ensemble_features: Feature lists for each base model.
            meta_features: Optional curated list of meta features. If None,
                the union of `ensemble_features` will be used.
            targets: Target column name or list of names (one per base model).
            val: Optional validation DataFrame for early stopping and validation.
            callbacks: Optional fit callbacks passed to base models.
            n_jobs: Number of parallel jobs for base model training.
            head_hidden_dim: Hidden dimension for the meta MLP.
            head_epochs: Epochs to train the meta head.
            head_lr: Learning rate for the meta head optimizer.
            head_batch_size: Batch size for the meta head.
            entropy_coef: Entropy regularization coefficient for the head.

        Returns:
            self
        """
        n_models = len(self.ensemble_models)
        if n_models == 0:
            raise RuntimeError("No base models added. Call add_model() before fit().")

        if targets is None:
            targets = ["target"] * n_models
        elif isinstance(targets, str):
            targets = [targets] * n_models

        if isinstance(ensemble_features[0], str):
            per_model_features = [ensemble_features] * n_models
        else:
            per_model_features = list(ensemble_features)

        self.features = per_model_features
        
        # Use provided curated meta-features, or fallback to union of all tree features
        if meta_features is not None:
            self.meta_features = meta_features
        else:
            self.meta_features = sorted({f for subset in self.features for f in subset})
            
        self._build_meta_head(
            len(self.meta_features),
            n_models,
            hidden_dim=head_hidden_dim,
            embed_dropout=embed_dropout,
            hidden_dropout=hidden_dropout,
        )

        # Base Models Training 
        n_cores = os.cpu_count() if n_jobs == -1 else max(1, os.cpu_count() + n_jobs + 1) if n_jobs < 0 else n_jobs
        results = Parallel(
            n_jobs=n_cores,
            backend='threading',
            prefer='threads',
            max_nbytes='1G',
        )(
            delayed(fit_single)(name, model, train, features, target_col, val, callbacks)
            for (name, model), features, target_col in zip(self.ensemble_models, self.features, targets)
        )
        self.ensemble_models = results

        # Meta Head Training
        meta_target = targets[0]
        val_X = val[self.meta_features] if val is not None and meta_target in val.columns else None
        val_y = val[meta_target] if val is not None and meta_target in val.columns else None

        self._train_meta_head(
            train,
            train[meta_target],
            val_X,
            val_y,
            epochs=head_epochs,
            lr=head_lr,
            batch_size=head_batch_size,
            entropy_coef=entropy_coef,
            weight_decay_head=weight_decay_head,
            weight_decay_mlp=weight_decay_mlp,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Compute dynamic ensemble predictions for input DataFrame.

        Args:
            X: DataFrame with feature columns required by base models and
               meta features used by the head.

        Returns:
            NumPy array of shape (n_rows,) with ensemble predictions.
        """
        if not self.ensemble_models:
            raise RuntimeError("No models in ensemble. Call add_model() first.")

        self.eval()  # Ensure dropout is disabled for inference
        with torch.no_grad():
            base_preds = []
            for (_, model), features in zip(self.ensemble_models, self.features):
                base_preds.append(torch.from_numpy(np.asarray(model.predict(X[features]), dtype=np.float32)))
            base_preds = torch.stack(base_preds, dim=1)
            weights = self._compute_weights(X)
            ensemble_pred = (weights * base_preds).sum(dim=1)
            return ensemble_pred.cpu().numpy()



def prelim():
    """Train and validate"""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = Config(model_name='dynamic_ensemble', feature_set="all")
    train, ensemble_features = load_dataset(config, split="train")
    val, _ = load_dataset(config, split="validation")

    params = {
        "num_threads": 1,
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
    feature_group_names = [
        'intelligence', 'charisma', 'strength', 'dexterity', 'constitution',
        'wisdom', 'agility', 'serenity', 'sunshine', 'rain', 'midnight', 'faith'
    ]

    loaded_features = set(ensemble_features)
    per_model_features = []
    for g in feature_group_names:
        group_feats = [f for f in feature_sets[g] if f in loaded_features]
        per_model_features.append(group_feats)

    global_targets = ['target_ender_20', 'target_ender_60', 'target_jasper_20', 'target_jasper_60']
    for _ in global_targets:
        per_model_features.append(ensemble_features)

    # Use a curated, smaller set of features for the gating network to prevent era memorization
    # Fallback to the first 40 features
    meta_feats_curated = feature_sets.get("small", list(loaded_features)[:40])
    # Ensure they exist in the loaded data
    meta_feats_curated = [f for f in meta_feats_curated if f in loaded_features]

    ensemble = DynamicEnsemble(embedding_dim=2)
    
    for group_name in feature_group_names:
        ensemble.add_model(group_name, lgb.LGBMRegressor(**params))
    for target_name in global_targets:
        ensemble.add_model(target_name, lgb.LGBMRegressor(**params))

    early_stop_callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
    
    ensemble.fit(
        train,
        ensemble_features=per_model_features,
        meta_features=meta_feats_curated, # Explicitly pass the curated subset
        targets=["target"] * len(feature_group_names) + global_targets,
        val=val,
        callbacks=early_stop_callbacks,
        n_jobs=1,
        head_hidden_dim=64,
        head_epochs=20,
        head_lr=1e-3,
        head_batch_size=1024,
        entropy_coef=0.02
    )

    last_train_era = int(train["era"].unique()[-1])
    save_model(ensemble, config, last_train_era=last_train_era)

def final()
    """
    Train on both the train and validation splits and save the final model.

    This is the submission path: it uses all available historical data
    for model training and does not reserve a validation split for
    meta-head monitoring or early stopping.
    """
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = Config(model_name='dynamic_ensemble', feature_set="all")
    train, ensemble_features = load_dataset(config, split="train")
    val, _ = load_dataset(config, split="validation")
    train_val = pd.concat([train, val], ignore_index=True)

    params = {
        "num_threads": 1,
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
    feature_group_names = [
        'intelligence', 'charisma', 'strength', 'dexterity', 'constitution',
        'wisdom', 'agility', 'serenity', 'sunshine', 'rain', 'midnight', 'faith'
    ]

    loaded_features = set(ensemble_features)
    per_model_features = []
    for g in feature_group_names:
        group_feats = [f for f in feature_sets[g] if f in loaded_features]
        per_model_features.append(group_feats)

    global_targets = ['target_ender_20', 'target_ender_60', 'target_jasper_20', 'target_jasper_60']
    for _ in global_targets:
        per_model_features.append(ensemble_features)

    meta_feats_curated = feature_sets.get("small", list(loaded_features)[:40])
    meta_feats_curated = [f for f in meta_feats_curated if f in loaded_features]

    ensemble = DynamicEnsemble(embedding_dim=2)
    for group_name in feature_group_names:
        ensemble.add_model(group_name, lgb.LGBMRegressor(**params))
    for target_name in global_targets:
        ensemble.add_model(target_name, lgb.LGBMRegressor(**params))

    ensemble.fit(
        train_val,
        ensemble_features=per_model_features,
        meta_features=meta_feats_curated,
        targets=["target"] * len(feature_group_names) + global_targets,
        val=None,
        callbacks=None,
        n_jobs=1,
        head_hidden_dim=64,
        head_epochs=20,
        head_lr=1e-3,
        head_batch_size=1024,
        entropy_coef=0.02,
    )

    last_train_era = int(train_val["era"].unique()[-1])
    save_model(ensemble, config, last_train_era=last_train_era)


if __name__ == "__main__":
    prelim()