import os

import numpy as np
import pandas as pd
import torch

from Proto import ProtoNet
from ProtoNetPrep import ProtoNetPrep
from Evaluator import Evaluator


class ProtoNetRunner:

    def __init__(
        self,
        shot_size=5,
        test_size=100,
        random_seed=42,
        number_of_episodes=100,
        learning_rate=0.001
    ):

        self.shot_size = shot_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.number_of_episodes = number_of_episodes
        self.learning_rate = learning_rate

    def run(
        self,
        dataset,
        dataset_name,
        model_output_dir
    ):

        print("\nPROTONET EXPERIMENT")
        print(f"Dataset: {dataset_name}")
        print(f"Shot Size: {self.shot_size}")
        print(f"Test Size: {self.test_size}")

        self.prepare_dataset(
            dataset=dataset,
            dataset_name=dataset_name
        )

        training_pool, testing_pool = (
            self.get_data_pools(
                dataset=dataset,
                dataset_name=dataset_name
            )
        )

        label_column = "label"

        feature_columns = (
            self.get_feature_columns(
                training_pool,
                label_column
            )
        )

        input_size = len(
            feature_columns
        )

        print(
            f"\nProtoNet input features: "
            f"{input_size}"
        )

        model = ProtoNet(
            input_size=input_size,
            hidden_size=128,
            embedding_size=64
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model = model.to(
            device
        )

        print(
            f"ProtoNet device: "
            f"{device}"
        )

        prep = ProtoNetPrep(
            shot_size=self.shot_size,
            query_size=self.shot_size,
            random_seed=self.random_seed
        )

        self.train(
            model=model,
            prep=prep,
            training_pool=training_pool,
            feature_columns=feature_columns,
            label_column=label_column,
            device=device
        )

        support_df = (
            self.create_support_set(
                training_pool=training_pool,
                label_column=label_column
            )
        )

        print(
            "\nSupport set class counts:"
        )

        print(
            support_df[
                label_column
            ]
            .value_counts()
            .sort_index()
        )

        (
            prototypes,
            prototype_labels
        ) = self.create_prototypes(
            model=model,
            support_df=support_df,
            feature_columns=feature_columns,
            label_column=label_column,
            device=device
        )

        test_df = (
            self.create_test_set(
                testing_pool=testing_pool,
                label_column=label_column
            )
        )

        results = self.evaluate(
            model=model,
            dataset=dataset,
            test_df=test_df,
            feature_columns=feature_columns,
            label_column=label_column,
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            device=device
        )

        self.save_model(
            model=model,
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            feature_columns=feature_columns,
            input_size=input_size,
            model_output_dir=model_output_dir
        )

        print(
            "\nEXPERIMENT FINISHED"
        )

        print(
            "Model: ProtoNet"
        )

        print(
            f"Dataset: "
            f"{dataset_name}"
        )

        return results

    def prepare_dataset(
        self,
        dataset,
        dataset_name
    ):

        print(
            f"\nPreparing dataset: "
            f"{dataset_name}"
        )

        if dataset_name == "CIC-IDS2017":

            dataset.load_csv_files()

        elif dataset_name == "UNSW-NB15":

            dataset.load_data()

        else:

            raise ValueError(
                f"Unsupported dataset: "
                f"{dataset_name}"
            )

        dataset.clean_data()

        dataset.encode_labels()

        print(
            f"{dataset_name} "
            f"preparation complete."
        )

    def train(
        self,
        model,
        prep,
        training_pool,
        feature_columns,
        label_column,
        device
    ):

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate
        )

        print(
            f"\nTraining episodes: "
            f"{self.number_of_episodes}"
        )

        model.train()

        for episode in range(
            self.number_of_episodes
        ):

            (
                support_x,
                support_y,
                query_x,
                query_y
            ) = prep.create_episode(
                data=training_pool,
                feature_columns=feature_columns,
                label_column=label_column
            )

            support_x = support_x.to(
                device
            )

            support_y = support_y.to(
                device
            )

            query_x = query_x.to(
                device
            )

            query_y = query_y.to(
                device
            )

            loss = model.episode_loss(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                query_y=query_y
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if (
                episode == 0
                or (episode + 1) % 10 == 0
                or episode + 1
                == self.number_of_episodes
            ):

                print(
                    f"Episode "
                    f"{episode + 1}/"
                    f"{self.number_of_episodes}"
                    f" | Loss: "
                    f"{loss.item():.4f}"
                )

        print(
            "\nProtoNet training complete."
        )

    def create_prototypes(
        self,
        model,
        support_df,
        feature_columns,
        label_column,
        device
    ):

        support_x = torch.tensor(
            support_df[
                feature_columns
            ].to_numpy(
                dtype=np.float32
            ),
            dtype=torch.float32,
            device=device
        )

        support_y = torch.tensor(
            support_df[
                label_column
            ].to_numpy(),
            dtype=torch.long,
            device=device
        )

        model.eval()

        with torch.no_grad():

            support_embeddings = model(
                support_x
            )

            (
                prototypes,
                prototype_labels
            ) = model.calculate_prototypes(
                support_embeddings,
                support_y
            )

        return (
            prototypes,
            prototype_labels
        )

    def evaluate(
        self,
        model,
        dataset,
        test_df,
        feature_columns,
        label_column,
        prototypes,
        prototype_labels,
        device
    ):

        print(
            "\nTesting class counts:"
        )

        print(
            test_df[
                label_column
            ]
            .value_counts()
            .sort_index()
        )

        evaluator = Evaluator(
            class_names=dataset.unique_labels
        )

        y_true = (
            test_df[
                label_column
            ]
            .to_numpy()
        )

        def predict_function(
            sample
        ):

            sample = np.asarray(
                sample,
                dtype=np.float32
            )

            if sample.ndim == 1:

                sample = np.expand_dims(
                    sample,
                    axis=0
                )

            sample_tensor = torch.tensor(
                sample,
                dtype=torch.float32,
                device=device
            )

            model.eval()

            with torch.no_grad():

                embedding = model(
                    sample_tensor
                )

                distances = (
                    model.calculate_distances(
                        embedding,
                        prototypes
                    )
                )

                nearest_prototype = (
                    torch.argmin(
                        distances,
                        dim=1
                    )
                )

                prediction = (
                    prototype_labels[
                        nearest_prototype
                    ]
                )

            return int(
                prediction[0].item()
            )

        test_samples = (
            test_df[
                feature_columns
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        (
            predictions,
            inference_results
        ) = (
            evaluator.measure_inference_performance(
                predict_function=predict_function,
                samples=test_samples
            )
        )

        results = (
            evaluator.evaluate_all(
                y_true=y_true,
                y_pred=predictions,
                inference_results=inference_results
            )
        )

        evaluator.print_classification_report(
            y_true,
            predictions
        )

        matrix = (
            evaluator.get_confusion_matrix(
                y_true,
                predictions
            )
        )

        print(
            "\nConfusion Matrix:"
        )

        print(
            matrix
        )

        evaluator.print_results(
            results
        )

        return results

    def get_data_pools(
        self,
        dataset,
        dataset_name
    ):

        if dataset_name == "UNSW-NB15":

            training_pool = (
                dataset.get_train_data()
                .copy()
            )

            testing_pool = (
                dataset.get_test_data()
                .copy()
            )

            return (
                training_pool,
                testing_pool
            )

        if dataset_name == "CIC-IDS2017":

            data = (
                dataset.get_data()
                .copy()
            )

            training_parts = []
            testing_parts = []

            required_training = (
                self.shot_size
                + self.shot_size
            )

            for label, group in (
                data.groupby(
                    "label"
                )
            ):

                shuffled = group.sample(
                    frac=1,
                    random_state=self.random_seed
                )

                training_count = min(
                    required_training,
                    len(shuffled)
                )

                training_part = (
                    shuffled.iloc[
                        :training_count
                    ]
                )

                testing_part = (
                    shuffled.iloc[
                        training_count:
                        training_count
                        + self.test_size
                    ]
                )

                if len(
                    training_part
                ) > 0:

                    training_parts.append(
                        training_part
                    )

                if len(
                    testing_part
                ) > 0:

                    testing_parts.append(
                        testing_part
                    )

            if not training_parts:

                raise ValueError(
                    "No CIC-IDS2017 training "
                    "samples were created."
                )

            if not testing_parts:

                raise ValueError(
                    "No CIC-IDS2017 testing "
                    "samples were created."
                )

            training_pool = pd.concat(
                training_parts,
                ignore_index=True
            )

            testing_pool = pd.concat(
                testing_parts,
                ignore_index=True
            )

            return (
                training_pool,
                testing_pool
            )

        raise ValueError(
            f"Unsupported dataset: "
            f"{dataset_name}"
        )

    def get_feature_columns(
        self,
        data,
        label_column
    ):

        feature_columns = []

        for column in data.columns:

            if column == label_column:
                continue

            if pd.api.types.is_numeric_dtype(
                data[column]
            ):

                feature_columns.append(
                    column
                )

        if not feature_columns:

            raise ValueError(
                "No numeric features found."
            )

        return feature_columns

    def create_support_set(
        self,
        training_pool,
        label_column
    ):

        support_parts = []

        for label, group in (
            training_pool.groupby(
                label_column
            )
        ):

            if len(group) < self.shot_size:

                print(
                    f"Skipping class {label}: "
                    f"only {len(group)} samples."
                )

                continue

            selected = group.sample(
                n=self.shot_size,
                random_state=self.random_seed
            )

            support_parts.append(
                selected
            )

        if not support_parts:

            raise ValueError(
                "No classes contain enough "
                "samples for the support set."
            )

        support_df = pd.concat(
            support_parts,
            ignore_index=True
        )

        return support_df

    def create_test_set(
        self,
        testing_pool,
        label_column
    ):

        test_parts = []

        for label, group in (
            testing_pool.groupby(
                label_column
            )
        ):

            sample_size = min(
                self.test_size,
                len(group)
            )

            if sample_size == 0:
                continue

            selected = group.sample(
                n=sample_size,
                random_state=self.random_seed
            )

            test_parts.append(
                selected
            )

        if not test_parts:

            raise ValueError(
                "No testing samples "
                "were created."
            )

        test_df = pd.concat(
            test_parts,
            ignore_index=True
        )

        return test_df

    def save_model(
        self,
        model,
        prototypes,
        prototype_labels,
        feature_columns,
        input_size,
        model_output_dir
    ):

        os.makedirs(
            model_output_dir,
            exist_ok=True
        )

        model_path = os.path.join(
            model_output_dir,
            "protonet_model.pt"
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "prototypes":
                    prototypes.detach().cpu(),

                "prototype_labels":
                    prototype_labels.detach().cpu(),

                "feature_columns":
                    feature_columns,

                "input_size":
                    input_size,

                "hidden_size":
                    model.hidden_size
                    if hasattr(
                        model,
                        "hidden_size"
                    )
                    else 128,

                "embedding_size":
                    model.embedding_size
                    if hasattr(
                        model,
                        "embedding_size"
                    )
                    else 64,

                "shot_size":
                    self.shot_size,

                "test_size":
                    self.test_size,

                "number_of_episodes":
                    self.number_of_episodes,

                "learning_rate":
                    self.learning_rate,

                "random_seed":
                    self.random_seed
            },
            model_path
        )

        print(
            f"\nProtoNet saved to: "
            f"{model_path}"
        )