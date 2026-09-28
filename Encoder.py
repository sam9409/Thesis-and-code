import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class EncoderNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim1=128,
        hidden_dim2=64,
        embedding_dim=32,
        dropout=0.2
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim1
            ),

            nn.BatchNorm1d(
                hidden_dim1
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim1,
                hidden_dim2
            ),

            nn.BatchNorm1d(
                hidden_dim2
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim2,
                embedding_dim
            ),

            nn.ReLU()
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )


    def forward(
        self,
        x
    ):

        embedding = self.encoder(
            x
        )

        logits = self.classifier(
            embedding
        )

        return logits


    def get_embedding(
        self,
        x
    ):

        return self.encoder(
            x
        )


class Encoder:

    def __init__(
        self,
        input_dim,
        num_classes,
        model_output_dir,
        random_seed=42,
        learning_rate=0.001,
        batch_size=64,
        epochs=20
    ):

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model_output_dir = model_output_dir
        self.random_seed = random_seed

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        torch.manual_seed(
            self.random_seed
        )

        np.random.seed(
            self.random_seed
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                self.random_seed
            )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Encoder device: "
            f"{self.device}"
        )

        if self.device.type == "cuda":

            print(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

        self.model = EncoderNetwork(
            input_dim=self.input_dim,
            num_classes=self.num_classes
        ).to(
            self.device
        )

        self.criterion = (
            nn.CrossEntropyLoss()
        )

        self.optimizer = (
            torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
        )

        self.scaler = StandardScaler()

        self.feature_columns = None

        self.X_train = None
        self.y_train = None

        self.X_test = None
        self.y_test = None


    def prepare_data(
        self,
        dataset,
        train_df,
        test_df
    ):

        print(
            "\nPreparing encoder data..."
        )

        self.feature_columns = []

        for column in train_df.columns:

            if column == "label":
                continue

            if pd.api.types.is_numeric_dtype(
                train_df[column]
            ):

                self.feature_columns.append(
                    column
                )

        if not self.feature_columns:

            raise ValueError(
                "No numeric features found "
                "for Encoder."
            )

        if (
            len(self.feature_columns)
            != self.input_dim
        ):

            raise ValueError(
                f"Expected {self.input_dim} "
                f"input features but found "
                f"{len(self.feature_columns)}."
            )

        print(
            f"Using "
            f"{len(self.feature_columns)} "
            f"numeric features."
        )

        X_train = (
            train_df[
                self.feature_columns
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
            .to_numpy(
                dtype=np.float32
            )
        )

        X_test = (
            test_df[
                self.feature_columns
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
            .to_numpy(
                dtype=np.float32
            )
        )

        y_train = (
            train_df["label"]
            .to_numpy(
                dtype=np.int64
            )
        )

        y_test = (
            test_df["label"]
            .to_numpy(
                dtype=np.int64
            )
        )

        print(
            f"Training samples: "
            f"{len(X_train)}"
        )

        print(
            f"Testing samples: "
            f"{len(X_test)}"
        )

        print(
            "Scaling numeric features..."
        )

        X_train = (
            self.scaler.fit_transform(
                X_train
            )
            .astype(
                np.float32
            )
        )

        X_test = (
            self.scaler.transform(
                X_test
            )
            .astype(
                np.float32
            )
        )

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        train_data = (
            self.X_train,
            self.y_train
        )

        test_data = (
            self.X_test,
            self.y_test
        )

        return (
            train_data,
            test_data
        )


    def train(
        self
    ):

        if self.X_train is None:

            raise ValueError(
                "Training data has not been "
                "prepared. Call prepare_data() "
                "before train()."
            )

        X_tensor = torch.tensor(
            self.X_train,
            dtype=torch.float32
        )

        y_tensor = torch.tensor(
            self.y_train,
            dtype=torch.long
        )

        training_dataset = (
            TensorDataset(
                X_tensor,
                y_tensor
            )
        )

        drop_last = (
            len(training_dataset)
            % self.batch_size == 1
        )

        training_loader = (
            DataLoader(
                training_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=(
                    self.device.type
                    == "cuda"
                ),
                drop_last=drop_last
            )
        )

        print(
            "\nTRAINING ENCODER"
        )

        print(
            f"Training samples: "
            f"{len(training_dataset)}"
        )

        print(
            f"Batch size: "
            f"{self.batch_size}"
        )

        print(
            f"Epochs: "
            f"{self.epochs}"
        )

        self.model.train()

        for epoch in range(
            self.epochs
        ):

            total_loss = 0.0

            correct = 0

            total = 0

            for (
                X_batch,
                y_batch
            ) in training_loader:

                X_batch = X_batch.to(
                    self.device,
                    non_blocking=True
                )

                y_batch = y_batch.to(
                    self.device,
                    non_blocking=True
                )

                self.optimizer.zero_grad()

                outputs = self.model(
                    X_batch
                )

                loss = self.criterion(
                    outputs,
                    y_batch
                )

                loss.backward()

                self.optimizer.step()

                total_loss += (
                    loss.item()
                )

                predictions = (
                    torch.argmax(
                        outputs,
                        dim=1
                    )
                )

                correct += (
                    predictions
                    == y_batch
                ).sum().item()

                total += (
                    y_batch.size(0)
                )

            average_loss = (
                total_loss
                / len(training_loader)
            )

            accuracy = (
                correct / total
            )

            print(
                f"Epoch "
                f"[{epoch + 1}/"
                f"{self.epochs}] "
                f"Loss: "
                f"{average_loss:.4f} "
                f"Accuracy: "
                f"{accuracy:.4f}"
            )

        print(
            "\nEncoder training complete."
        )


    def predict(
        self,
        X
    ):

        self.model.eval()

        X = np.asarray(
            X,
            dtype=np.float32
        )

        if X.ndim == 1:

            X = np.expand_dims(
                X,
                axis=0
            )

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():

            outputs = self.model(
                X_tensor
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

        return [
            int(value)
            for value
            in predictions.cpu().numpy()
        ]


    def predict_one(
        self,
        X
    ):

        predictions = self.predict(
            [X]
        )

        return predictions[0]


    def get_test_samples(
        self,
        test_data
    ):

        X_test, _ = test_data

        return X_test


    def get_embeddings(
        self,
        X
    ):

        self.model.eval()

        X = np.asarray(
            X,
            dtype=np.float32
        )

        if X.ndim == 1:

            X = np.expand_dims(
                X,
                axis=0
            )

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():

            embeddings = (
                self.model.get_embedding(
                    X_tensor
                )
            )

        return (
            embeddings
            .cpu()
            .numpy()
        )


    def save_model(
        self
    ):

        os.makedirs(
            self.model_output_dir,
            exist_ok=True
        )

        model_path = os.path.join(
            self.model_output_dir,
            "encoder_model.pt"
        )

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "input_dim":
                    self.input_dim,

                "num_classes":
                    self.num_classes,

                "feature_columns":
                    self.feature_columns,

                "scaler_mean":
                    self.scaler.mean_,

                "scaler_scale":
                    self.scaler.scale_,

                "random_seed":
                    self.random_seed,

                "learning_rate":
                    self.learning_rate,

                "batch_size":
                    self.batch_size,

                "epochs":
                    self.epochs
            },
            model_path
        )

        print(
            f"\nEncoder saved to: "
            f"{model_path}"
        )