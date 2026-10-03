from numbers import Integral

import pandas as pd

from SetFit import SetFit
from Evaluator import Evaluator
from trainingSplit import trainingSplit


class SetFitRunner:

    def __init__(
        self,
        shot_size=5,
        test_size=100,
        random_seed=42,
        batch_size=16,
        num_epochs=1
    ):

        self.shot_size = shot_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def run(
        self,
        dataset,
        dataset_name,
        model_output_dir
    ):

        print("\nSETFIT EXPERIMENT")
        print(f"Dataset: {dataset_name}")
        print(f"Shot Size: {self.shot_size}")
        print(f"Test Size: {self.test_size}")

        dataset.load_data()
        dataset.clean_data()
        dataset.encode_labels()

        train_df, test_df = self.create_split(
            dataset=dataset,
            dataset_name=dataset_name
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

        print(
            "\nConverting network flows to text..."
        )

        train_text_df = (
            dataset.convert_to_text(
                train_df.copy()
            )
        )

        test_text_df = (
            dataset.convert_to_text(
                test_df.copy()
            )
        )

        model = SetFit(
            unique_labels=dataset.unique_labels,
            model_output_dir=model_output_dir,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            random_seed=self.random_seed
        )

        train_dataset = (
            model.create_dataset(
                train_text_df
            )
        )

        test_dataset = (
            model.create_dataset(
                test_text_df
            )
        )

        model.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset
        )

        results = self.evaluate(
            model=model,
            dataset=dataset,
            test_df=test_text_df
        )

        model.save_model()

        print("\nEXPERIMENT FINISHED")
        print("Model: SetFit")
        print(f"Dataset: {dataset_name}")

        return results

    def evaluate(
        self,
        model,
        dataset,
        test_df
    ):

        evaluator = Evaluator(
            class_names=dataset.unique_labels
        )

        y_true = (
            test_df["label"]
            .to_numpy()
        )

        test_samples = (
            test_df["text"]
            .tolist()
        )

        def predict_function(
            sample
        ):

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

            if hasattr(
                predicted_label,
                "item"
            ):

                predicted_label = (
                    predicted_label.item()
                )

                if isinstance(
                    predicted_label,
                    Integral
                ):

                    return int(
                        predicted_label
                    )

            if predicted_label in (
                dataset.label_to_id
            ):

                return dataset.label_to_id[
                    predicted_label
                ]

            return int(
                predicted_label
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

    def create_split(
        self,
        dataset,
        dataset_name
    ):

        if dataset_name == "CIC-IDS2017":

            return self.create_cic_split(
                dataset
            )

        if dataset_name == "UNSW-NB15":

            return self.create_unsw_split(
                dataset
            )

        raise ValueError(
            f"Unsupported dataset: "
            f"{dataset_name}"
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

            if len(group) < self.shot_size:

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