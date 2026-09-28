from trainingSplit import trainingSplit
from ModelFactory import ModelFactory
from Evaluator import Evaluator
from ProtoNetPrep import ProtoNetPrep
from numbers import Integral
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import torch
import os


class ExperimentRunner:

    def __init__(
        self,
        shot_size=5,
        test_size=100,
        random_seed=42
    ):

        self.shot_size = shot_size
        self.test_size = test_size
        self.random_seed = random_seed


    def run(
        self,
        model_name,
        dataset,
        dataset_name,
        model_output_dir
    ):

        print("\nSTART EXPERIMENT")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")

        dataset.load_data()
        dataset.clean_data()
        dataset.encode_labels()

        # ProtoNet uses its own training process
        if model_name == "ProtoNet":

            return self.run_protonet(
                dataset=dataset,
                dataset_name=dataset_name,
                model_output_dir=model_output_dir
            )

        # Autoencoder uses the full training dataset
        if model_name == "Autoencoder":

            train_df, test_df = (
                self.create_traditional_encoder_split(
                    dataset,
                    dataset_name
                )
            )

        # Other models use few-shot training
        else:

            print(f"Shot Size: {self.shot_size}")
            print(f"Test Size: {self.test_size}")

            if dataset_name == "UNSW-NB15":

                train_df, test_df = (
                    self.create_unsw_split(
                        dataset
                    )
                )

            else:

                train_df, test_df = (
                    self.create_cic_split(
                        dataset
                    )
                )

        print("\nTraining class counts:")

        print(
            train_df["label"]
            .value_counts()
            .sort_index()
        )

        print("\nTesting class counts:")

        print(
            test_df["label"]
            .value_counts()
            .sort_index()
        )

        # Autoencoder needs the number
        # of numerical input features
        input_size = None

        if model_name == "Autoencoder":

            feature_columns = (
                self.get_feature_columns(
                    train_df,
                    "label"
                )
            )

            input_size = len(
                feature_columns
            )

            print(
                f"\nEncoder input features: "
                f"{input_size}"
            )

        model = ModelFactory.create_model(
            model_name=model_name,
            unique_labels=dataset.unique_labels,
            model_output_dir=model_output_dir,
            random_seed=self.random_seed,
            input_size=input_size
        )

        train_data, test_data = (
            model.prepare_data(
                dataset=dataset,
                train_df=train_df,
                test_df=test_df
            )
        )

        print("\nTraining model...")

        model.train()

        evaluator = Evaluator(
            class_names=dataset.unique_labels
        )

        y_true = (
            test_df["label"]
            .to_numpy()
        )

        def predict_function(sample):

            prediction = model.predict(
                [sample]
            )

            predicted_label = prediction[0]

            if isinstance(
                predicted_label,
                Integral
            ):

                return int(
                    predicted_label
                )

            return dataset.label_to_id[
                predicted_label
            ]

        test_samples = (
            model.get_test_samples(
                test_data
            )
        )

        predictions, inference_results = (
            evaluator.measure_inference_performance(
                predict_function=predict_function,
                samples=test_samples
            )
        )

        results = evaluator.evaluate_all(
            y_true=y_true,
            y_pred=predictions,
            inference_results=inference_results
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

        print("\nConfusion Matrix:")
        print(matrix)

        evaluator.print_results(
            results
        )

        model.save_model()

        print("\nEXPERIMENT FINISHED")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")

        return results


    def run_protonet(
        self,
        dataset,
        dataset_name,
        model_output_dir
    ):

        print("\nPROTONET EXPERIMENT")

        training_pool, testing_pool = (
            self.get_protonet_pools(
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

        encoded_labels = sorted(
            training_pool[
                label_column
            ].unique()
        )

        model = ModelFactory.create_model(
            model_name="ProtoNet",
            unique_labels=encoded_labels,
            model_output_dir=model_output_dir,
            random_seed=self.random_seed,
            input_size=input_size
        )

        prep = ProtoNetPrep(
            shot_size=self.shot_size,
            query_size=self.shot_size,
            random_seed=self.random_seed
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
            f"ProtoNet device: {device}"
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

        number_of_episodes = 100

        print(
            f"Training episodes: "
            f"{number_of_episodes}"
        )

        model.train()

        for episode in range(
            number_of_episodes
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
                == number_of_episodes
            ):

                print(
                    f"Episode "
                    f"{episode + 1}/"
                    f"{number_of_episodes}"
                    f" | Loss: "
                    f"{loss.item():.4f}"
                )

        print(
            "\nProtoNet training complete."
        )

        support_df = (
            self.create_support_set(
                training_pool,
                label_column
            )
        )

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

        test_df = (
            self.create_test_set(
                testing_pool,
                label_column
            )
        )

        print("\nTraining class counts:")

        print(
            support_df[
                label_column
            ]
            .value_counts()
            .sort_index()
        )

        print("\nTesting class counts:")

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
            ].to_numpy()
        )

        def predict_function(sample):

            sample = np.asarray(
                sample,
                dtype=np.float32
            )

            if sample.ndim == 1:

                sample = np.expand_dims(
                    sample,
                    axis=0
                )

            sample = torch.tensor(
                sample,
                dtype=torch.float32,
                device=device
            )

            model.eval()

            with torch.no_grad():

                embedding = model(
                    sample
                )

                distances = (
                    model.calculate_distances(
                        embedding,
                        prototypes
                    )
                )

                nearest = torch.argmin(
                    distances,
                    dim=1
                )

                prediction = (
                    prototype_labels[
                        nearest
                    ]
                )

            return int(
                prediction.item()
            )

        test_samples = (
            test_df[
                feature_columns
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        predictions, inference_results = (
            evaluator.measure_inference_performance(
                predict_function=predict_function,
                samples=test_samples
            )
        )

        results = evaluator.evaluate_all(
            y_true=y_true,
            y_pred=predictions,
            inference_results=inference_results
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

        print("\nConfusion Matrix:")
        print(matrix)

        evaluator.print_results(
            results
        )

        self.save_protonet(
            model=model,
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            feature_columns=feature_columns,
            input_size=input_size,
            model_output_dir=model_output_dir
        )

        print("\nEXPERIMENT FINISHED")
        print("Model: ProtoNet")
        print(f"Dataset: {dataset_name}")

        return results


    def get_protonet_pools(
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

        data = (
            dataset.get_data()
            .copy()
        )

        training_parts = []
        testing_parts = []

        required_training = (
            self.shot_size * 2
        )

        for label, group in (
            data.groupby("label")
        ):

            shuffled = group.sample(
                frac=1,
                random_state=self.random_seed
            )

            training_count = min(
                max(
                    required_training,
                    self.shot_size
                ),
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

            if len(training_part) > 0:

                training_parts.append(
                    training_part
                )

            if len(testing_part) > 0:

                testing_parts.append(
                    testing_part
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

            sample_size = min(
                self.shot_size,
                len(group)
            )

            selected = group.sample(
                n=sample_size,
                random_state=self.random_seed
            )

            support_parts.append(
                selected
            )

        return pd.concat(
            support_parts,
            ignore_index=True
        )


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

            selected = group.sample(
                n=sample_size,
                random_state=self.random_seed
            )

            test_parts.append(
                selected
            )

        return pd.concat(
            test_parts,
            ignore_index=True
        )


    def save_protonet(
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
                    prototypes.cpu(),

                "prototype_labels":
                    prototype_labels.cpu(),

                "feature_columns":
                    feature_columns,

                "input_size":
                    input_size,

                "shot_size":
                    self.shot_size,

                "random_seed":
                    self.random_seed
            },
            model_path
        )

        print(
            f"\nProtoNet saved to: "
            f"{model_path}"
        )


    def create_cic_split(
        self,
        dataset
    ):

        print(
            f"\nCreating "
            f"{self.shot_size}-shot "
            f"CIC-IDS2017 training set"
        )

        data = dataset.get_data()

        splitter = trainingSplit(
            data=data,
            shot_size=self.shot_size,
            test_size=self.test_size,
            random_seed=self.random_seed
        )

        train_df, test_df = (
            splitter.create_split()
        )

        splitter.print_split_info()
        splitter.validate_split()
        splitter.check_overlap()

        return (
            train_df,
            test_df
        )


    def create_unsw_split(
        self,
        dataset
    ):

        print(
            f"\nCreating "
            f"{self.shot_size}-shot "
            f"UNSW-NB15 training set"
        )

        training_pool = (
            dataset.get_train_data()
        )

        testing_pool = (
            dataset.get_test_data()
            .copy()
        )

        train_parts = []
        test_parts = []

        for label, group in (
            training_pool.groupby(
                "label"
            )
        ):

            if (
                len(group)
                < self.shot_size
            ):

                raise ValueError(
                    f"Class {label} only has "
                    f"{len(group)} training "
                    f"examples."
                )

            selected = group.sample(
                n=self.shot_size,
                random_state=self.random_seed
            )

            train_parts.append(
                selected
            )

        for label, group in (
            testing_pool.groupby(
                "label"
            )
        ):

            sample_size = min(
                self.test_size,
                len(group)
            )

            selected = group.sample(
                n=sample_size,
                random_state=self.random_seed
            )

            test_parts.append(
                selected
            )

        train_df = pd.concat(
            train_parts,
            ignore_index=True
        )

        test_df = pd.concat(
            test_parts,
            ignore_index=True
        )

        return (
            train_df,
            test_df
        )


    def create_traditional_encoder_split(
        self,
        dataset,
        dataset_name
    ):

        if dataset_name == "UNSW-NB15":

            train_df = (
                dataset.get_train_data()
                .copy()
            )

            test_df = (
                dataset.get_test_data()
                .copy()
            )

        elif dataset_name == "CIC-IDS2017":

            data = (
                dataset.get_data()
                .copy()
            )

            train_df, test_df = (
                train_test_split(
                    data,
                    test_size=0.2,
                    random_state=self.random_seed,
                    stratify=data["label"]
                )
            )

        else:

            raise ValueError(
                f"Unsupported dataset: "
                f"{dataset_name}"
            )

        return (
            train_df,
            test_df
        )