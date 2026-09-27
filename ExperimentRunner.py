from trainingSplit import trainingSplit
from ModelFactory import ModelFactory
from Evaluator import Evaluator
from numbers import Integral
import pandas as pd


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
        print(f"Shot Size: {self.shot_size}")
        print(f"Test Size: {self.test_size}")

        dataset.load_data()
        dataset.clean_data()
        dataset.encode_labels()

        if dataset_name == "UNSW-NB15":

            train_df, test_df = self.create_unsw_split(
                dataset
            )

        else:

            train_df, test_df = self.create_cic_split(
                dataset
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

        model = ModelFactory.create_model(
            model_name=model_name,
            unique_labels=dataset.unique_labels,
            model_output_dir=model_output_dir,
            random_seed=self.random_seed
        )

        train_data, test_data = model.prepare_data(
            dataset=dataset,
            train_df=train_df,
            test_df=test_df
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

                return int(predicted_label)

            return dataset.label_to_id[
                predicted_label
            ]

        test_samples = model.get_test_samples(
            test_data
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

        matrix = evaluator.get_confusion_matrix(
            y_true,
            predictions
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


    def create_cic_split(
        self,
        dataset
    ):

        print(
            f"\nCreating {self.shot_size}-shot "
            "CIC-IDS2017 training set"
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

        return train_df, test_df


    def create_unsw_split(
        self,
        dataset
    ):

        print(
            f"\nCreating {self.shot_size}-shot "
            "UNSW-NB15 training set"
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

        for label_id, group in (
            training_pool.groupby("label")
        ):

            if len(group) < self.shot_size:

                raise ValueError(
                    f"Class {label_id} only has "
                    f"{len(group)} training examples."
                )

            selected = group.sample(
                n=self.shot_size,
                random_state=self.random_seed
            )

            train_parts.append(
                selected
            )

        for label_id, group in (
            testing_pool.groupby("label")
        ):

            number_of_samples = min(
                self.test_size,
                len(group)
            )

            selected = group.sample(
                n=number_of_samples,
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

        return train_df, test_df