
import os
import time
import random

import numpy as np
import pandas as pd
import torch
from UNSWDataPrepSetFit import UNSWDataPrepSetFit

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from MAML import MAML


class MAMLRunner:

    def __init__(
        self,
        shot_size=5,
        query_size=5,
        test_size=100,
        random_seed=42,
        hidden_size=128,
        embedding_size=64,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        meta_batch_size=4,
        meta_epochs=100,
        first_order=False
    ):

        self.shot_size = shot_size
        self.query_size = query_size
        self.test_size = test_size
        self.random_seed = random_seed

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

        self.meta_batch_size = meta_batch_size
        self.meta_epochs = meta_epochs
        self.first_order = first_order

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        self.rng = np.random.default_rng(random_seed)


    def prepare_data(self, dataset):

        dataset.load_data()
        dataset.clean_data()
        dataset.encode_labels()

        if isinstance(dataset, UNSWDataPrepSetFit):

            train_df = dataset.get_train_data().copy()
            test_df = dataset.get_test_data().copy()

            label_column = "label"

        else:

            df = dataset.get_data().copy()

            label_column = "Label"

            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=self.random_seed,
                stratify=df[label_column]
            )

        # Remove invalid numeric values.
        train_df = train_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        test_df = test_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        train_df = train_df.dropna(
            subset=[label_column]
        )

        test_df = test_df.dropna(
            subset=[label_column]
        )

        # Exclude labels and identifiers from features.
        excluded_columns = {
            "label",
            "Label",
            "binary_label",
            "attack_cat",
            "id",
            "text"
        }

        feature_columns = [
            col
            for col in train_df.columns
            if col not in excluded_columns
            and col in test_df.columns
        ]

        X_train_df = train_df[
            feature_columns
        ].copy()

        X_test_df = test_df[
            feature_columns
        ].copy()

        # Encode categorical features using training data.
        categorical_columns = X_train_df.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        X_train_df = pd.get_dummies(
            X_train_df,
            columns=categorical_columns,
            dtype=np.float32
        )

        X_test_df = pd.get_dummies(
            X_test_df,
            columns=categorical_columns,
            dtype=np.float32
        )

        X_test_df = X_test_df.reindex(
            columns=X_train_df.columns,
            fill_value=0
        )

        # Convert all features to numeric.
        X_train_df = X_train_df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X_test_df = X_test_df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Fill missing values using training medians.
        train_medians = X_train_df.median().fillna(0)

        X_train_df = X_train_df.fillna(
            train_medians
        )

        X_test_df = X_test_df.fillna(
            train_medians
        )

        # Create numeric arrays.
        X_train = X_train_df.to_numpy(
            dtype=np.float64
        )

        X_test = X_test_df.to_numpy(
            dtype=np.float64
        )

        # Standardize using training statistics only.
        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )

        X_train = np.nan_to_num(
            X_train,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

        X_test = np.nan_to_num(
            X_test,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

        # Ensure labels are consecutive integers.
        all_labels = sorted(
            set(train_df[label_column].unique())
            | set(test_df[label_column].unique())
        )

        label_mapping = {
            label: index
            for index, label in enumerate(all_labels)
        }

        y_train = train_df[label_column].map(
            label_mapping
        ).to_numpy(dtype=np.int64)

        y_test = test_df[label_column].map(
            label_mapping
        ).to_numpy(dtype=np.int64)

        # Save metadata for model checkpoints.
        self.scaler = scaler
        self.feature_columns = X_train_df.columns.tolist()
        self.label_mapping = label_mapping

        print("\nDataset preparation complete")
        print("Training samples:", len(X_train))
        print("Testing samples:", len(X_test))
        print("Input features:", X_train.shape[1])
        print("Classes:", len(all_labels))

        print("\nTraining class counts:")
        print(pd.Series(y_train).value_counts().sort_index())

        print("\nTesting class counts:")
        print(pd.Series(y_test).value_counts().sort_index())

        return (
            (X_train, y_train),
            (X_test, y_test),
            X_train.shape[1],
            len(all_labels)
        )


    def create_episode(
        self,
        X,
        y,
        query_size=None
    ):

        if query_size is None:
            query_size = self.query_size

        support_indices = []
        query_indices = []

        for label in np.unique(y):

            class_indices = np.where(
                y == label
            )[0]

            available = len(class_indices)

            # Every class needs enough support samples
            # and at least one separate query sample.
            if available < self.shot_size + 1:
                raise ValueError(
                    f"Class {label} has {available} samples. "
                    f"At least {self.shot_size + 1} are required."
                )

            actual_query_size = min(
                query_size,
                available - self.shot_size
            )

            required = (
                self.shot_size + actual_query_size
            )

            selected = self.rng.choice(
                class_indices,
                size=required,
                replace=False
            )

            support_indices.extend(
                selected[:self.shot_size]
            )

            query_indices.extend(
                selected[self.shot_size:]
            )

        support_x = torch.tensor(
            X[support_indices],
            dtype=torch.float32,
            device=self.device
        )

        support_y = torch.tensor(
            y[support_indices],
            dtype=torch.long,
            device=self.device
        )

        query_x = torch.tensor(
            X[query_indices],
            dtype=torch.float32,
            device=self.device
        )

        query_y = torch.tensor(
            y[query_indices],
            dtype=torch.long,
            device=self.device
        )

        return (
            support_x,
            support_y,
            query_x,
            query_y
        )

    def train(
        self,
        model,
        X_train,
        y_train
    ):

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.outer_lr
        )

        for epoch in range(self.meta_epochs):

            episodes = []

            for _ in range(self.meta_batch_size):

                episode = self.create_episode(
                    X_train,
                    y_train
                )

                episodes.append(episode)

            loss = model.meta_train_step(
                episodes,
                optimizer
            )

            if (
                (epoch + 1) % 10 == 0
                or epoch == 0
            ):
                print(
                    f"Epoch {epoch + 1}/"
                    f"{self.meta_epochs} - "
                    f"Meta Loss: {loss:.4f}"
                )

    def evaluate(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        model.eval()

        support_indices = []

        for label in np.unique(y_train):

            class_indices = np.where(
                y_train == label
            )[0]

            if len(class_indices) < self.shot_size:
                raise ValueError(
                    f"Class {label} has insufficient "
                    f"training samples for adaptation."
                )

            selected = self.rng.choice(
                class_indices,
                size=self.shot_size,
                replace=False
            )

            support_indices.extend(selected)

        support_x = torch.tensor(
            X_train[support_indices],
            dtype=torch.float32,
            device=self.device
        )

        support_y = torch.tensor(
            y_train[support_indices],
            dtype=torch.long,
            device=self.device
        )

        # Adapt using training samples only.
        with torch.enable_grad():

            adapted_params = model.adapt(
                support_x,
                support_y
            )

        all_predictions = []
        latencies = []

        batch_size = 256

        for start_index in range(
            0,
            len(X_test),
            batch_size
        ):

            end_index = start_index + batch_size

            batch_x = torch.tensor(
                X_test[start_index:end_index],
                dtype=torch.float32,
                device=self.device
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.no_grad():

                predictions = model.predict(
                    batch_x,
                    adapted_params
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            latencies.append(
                elapsed * 1000 / len(batch_x)
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

        accuracy = accuracy_score(
            y_test,
            all_predictions
        )

        precision = precision_score(
            y_test,
            all_predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            all_predictions,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            all_predictions,
            average="macro",
            zero_division=0
        )

        avg_latency = float(np.mean(latencies))
        median_latency = float(np.median(latencies))
        worst_latency = float(np.max(latencies))

        throughput = (
            1000 / avg_latency
            if avg_latency > 0
            else 0
        )

        return {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "average_latency_ms": avg_latency,
            "median_latency_ms": median_latency,
            "worst_latency_ms": worst_latency,
            "throughput_samples_per_second": throughput
        }

    def save_model(
        self,
        model,
        model_output_dir
    ):

        os.makedirs(
            model_output_dir,
            exist_ok=True
        )

        path = os.path.join(
            model_output_dir,
            "maml_model.pth"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": model.input_size,
                "num_classes": model.num_classes,
                "hidden_size": model.hidden_size,
                "embedding_size": model.embedding_size,
                "inner_lr": model.inner_lr,
                "inner_steps": model.inner_steps,
                "first_order": model.first_order,
                "feature_columns": self.feature_columns,
                "label_mapping": self.label_mapping,
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_
            },
            path
        )

        print(f"Model saved to: {path}")

    def run(
        self,
        dataset,
        dataset_name,
        model_output_dir
    ):

        print(f"\nSTART MAML EXPERIMENT")
        print(f"Dataset: {dataset_name}")
        print(f"Shot Size: {self.shot_size}")
        print(f"Device: {self.device}")

        (
            train_data,
            test_data,
            input_size,
            num_classes
        ) = self.prepare_data(dataset)

        X_train, y_train = train_data
        X_test, y_test = test_data

        model = MAML(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size,
            inner_lr=self.inner_lr,
            inner_steps=self.inner_steps,
            first_order=self.first_order
        ).to(self.device)

        self.train(
            model,
            X_train,
            y_train
        )

        results = self.evaluate(
            model,
            X_train,
            y_train,
            X_test,
            y_test
        )

        print(f"\n{dataset_name} Results")

        for metric, value in results.items():
            print(f"{metric}: {value}")

        self.save_model(
            model,
            model_output_dir
        )

        return results
