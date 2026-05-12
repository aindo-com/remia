import pandas as pd
import torch
import numpy as np
import tqdm
from sklearn.preprocessing import OrdinalEncoder


class DF2Tensor:
    def __init__(self, one_hot: bool = True, normalize: bool = True):
        self.encoders = {}
        self.one_hot = one_hot
        self.normalize = normalize

    def fit_transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Fit encoders and transform dataframe to tensor."""
        df_processed = df.copy()

        # Identify categorical and numerical columns
        self.categorical_cols = df_processed.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.numerical_cols = df_processed.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        tensor_cols = []

        numeric_cols = torch.tensor(
            df_processed[self.numerical_cols].values, dtype=torch.float32
        )
        if self.normalize:
            self.mean = numeric_cols.mean(dim=0)
            self.std = numeric_cols.std(dim=0)
            numeric_cols = (numeric_cols - self.mean) / self.std
        tensor_cols.append(numeric_cols)

        for col in self.categorical_cols:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
            )
            encoded = encoder.fit_transform(df_processed[[col]])
            self.encoders[col] = encoder
            # Store number of classes for one-hot encoding
            self.encoders[f"{col}_n_classes"] = len(encoder.categories_[0])
            tensor_cat_col = torch.tensor(encoded.squeeze(), dtype=torch.long)
            if self.one_hot:
                # Handle -1 (unknown) by mapping to 0
                tensor_cat_col = torch.where(tensor_cat_col == -1, 0, tensor_cat_col)
                tensor_cat_col = torch.nn.functional.one_hot(
                    tensor_cat_col, num_classes=self.encoders[f"{col}_n_classes"]
                ).float()
            tensor_cols.append(tensor_cat_col)

        output_tensor = torch.cat(tensor_cols, dim=1)

        return output_tensor

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Transform dataframe using fitted encoders."""
        df_processed = df.copy()

        tensor_cols = []

        numeric_cols = torch.tensor(
            df_processed[self.numerical_cols].values, dtype=torch.float32
        )
        if self.normalize:
            numeric_cols = (numeric_cols - self.mean) / self.std
        tensor_cols.append(numeric_cols)

        for col in self.categorical_cols:
            encoder = self.encoders[col]
            n_classes = self.encoders[f"{col}_n_classes"]
            encoded = encoder.transform(df_processed[[col]]).squeeze()
            tensor_cat_col = torch.tensor(encoded, dtype=torch.long)
            if self.one_hot:
                # Handle -1 (unknown) by mapping to 0
                tensor_cat_col = torch.where(tensor_cat_col == -1, 0, tensor_cat_col)
                tensor_cat_col = torch.nn.functional.one_hot(
                    tensor_cat_col, num_classes=n_classes
                ).float()
            tensor_cols.append(tensor_cat_col)

        output_tensor = torch.cat(tensor_cols, dim=1)
        return output_tensor


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        units,
        dropout: float = 0.0,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout)
        self.units = torch.nn.ModuleList()

        if len(units) == 0:
            module_list = [torch.nn.Linear(in_features=input_shape, out_features=1)]
        else:
            module_list = [
                torch.nn.Linear(
                    in_features=units[i - 1] if i > 0 else input_shape,
                    out_features=units[i],
                )
                for i in range(len(units))
            ] + [torch.nn.Linear(in_features=units[-1], out_features=1)]
        self.units = torch.nn.ModuleList(module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = x.flatten(start_dim=1)
        for layer in self.units[:-1]:
            x_out = torch.nn.functional.layer_norm(x_out, x_out.shape[1:])
            activated = self.activation(layer(x_out))
            activated = self.dropout(activated)
            # Only add residual connection if dimensions match
            if activated.shape[1] == x_out.shape[1]:
                x_out = activated + x_out
            else:
                x_out = activated
        return self.units[-1](x_out).squeeze(1)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class TabularMLP:
    def __init__(
        self,
        units,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        self.units = units
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.activation = activation
        self.preprocessor_x = None
        self.preprocessor_y = None

    def preprocess_df(self, df: pd.DataFrame, target_col: str, device: str):
        if self.preprocessor_x is None:
            self.preprocessor_x = DF2Tensor(one_hot=True, normalize=True)
        if self.preprocessor_y is None:
            self.preprocessor_y = DF2Tensor(one_hot=False, normalize=False)

        X = (
            self.preprocessor_x.fit_transform(df.drop(columns=[target_col]))
            .float()
            .to(device)
        )
        y = (
            self.preprocessor_y.fit_transform(df[[target_col]])
            .float()
            .squeeze()
            .to(device)
        )
        return X, y

    def train(
        self,
        *,
        df: pd.DataFrame,
        target_col: str,
        epochs: int = 1000,
        iterations: int | None = 100_000,
        batch_size: int,
        patience: int,
        device: str,
        hooks: list = [],
        hook_period: int = 100,
        lr: float = 1e-2,
    ):

        self.device = device
        df_train = df.sample(
            frac=1.0, random_state=42, ignore_index=True
        )  # Shuffle data

        X, y = self.preprocess_df(df_train, target_col, device)

        self.model = MLP(
            input_shape=X.shape[1],
            units=self.units,
            dropout=self.dropout,
            activation=self.activation,
        ).to(device)
        # self.model = torch.compile(self.model)
        optimizer = torch.optim.RAdam(
            self.model.parameters(), lr=lr, weight_decay=self.weight_decay
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        best_loss = float("inf")
        patience_counter = 0
        iterations_counter = 0
        pbar = tqdm.tqdm(range(epochs), desc="Training MLP", unit="epoch")
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            shuffled_indices = torch.randperm(X.size(0))
            X = X[shuffled_indices]
            y = y[shuffled_indices]
            for i in range(0, len(X), batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                optimizer.zero_grad()
                y_logits = self.model(X_batch)
                loss = criterion(y_logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
                iterations_counter += 1

                for hook in hooks:
                    if iterations_counter % hook_period == 0:
                        hook(iterations_counter)

                if iterations is not None and iterations_counter >= iterations:
                    break
            epoch_loss /= len(X)
            pbar.set_postfix({"loss": f"{epoch_loss:.4f}"})

            acc = (
                ((self.model.probabilities(X) > 0.5).float() == y).float().mean().item()
            )
            pbar.set_postfix({"acc": f"{acc:.4f}"})
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
            if iterations is not None and iterations_counter >= iterations:
                print("Reached maximum iterations.")
                break

    def predict(self, df: pd.DataFrame | torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if isinstance(df, pd.DataFrame):
            X = self.preprocessor_x.transform(df).float().to(self.device)
        else:
            X = df.float().to(self.device)
        with torch.no_grad():
            y_probs = self.model.probabilities(X)
        return y_probs.cpu()


if __name__ == "__main__":
    df = pd.read_csv("data/adult/data.csv")
    model = TabularMLP(units=[50, 50, 20])
    training_result = model.train(
        df=df,
        target_col="income",
        epochs=100,
        batch_size=512,
        patience=5,
        device="cuda:0",
    )
    prediction = model.predict(df.drop(columns=["income"]))
    y_true = pd.get_dummies(df["income"], drop_first=True).values.squeeze()
    accuracy = (prediction.round() == y_true).mean()
    print(f"Accuracy: {accuracy:.4f}")
