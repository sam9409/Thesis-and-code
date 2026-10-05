import pandas as pd

from sklearn.model_selection import train_test_split

from Encoder import Encoder
from Evaluator import Evaluator


class EncoderRunner:

    def __init__(
        self,
        test_size=0.2,
        random_seed=42,
        learning_rate=0.001,
        batch_size=64,
        epochs=20
    ):

        self.test_size = test_size
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def run(
        self,
        dataset,
        dataset_name,
        model_output_dir
    ):

        print("\nENCODER EXPERIMENT")
        print(f"Dataset: {dataset_name}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch Size: {self.batch_size}")

        self.prepare_dataset(
            dataset=dataset,
            dataset_name=dataset_name
        )

        train_df, test_df = (
            self.create_split(
                dataset=dataset,
                dataset_name=dataset_name
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

        feature_columns = (
            self.get_feature_columns(
                train_df
            )
        )

        input_dim = len(
            feature_columns
        )

        num_classes = len(
            train_df["label"].unique()
        )

        print(
            f"\nEncoder input features: "
            f"{input_dim}"
        )

        print(
            f"Number of classes: "
            f"{num_classes}"
        )

        model = Encoder(
            input_dim=input_dim,
            num_classes=num_classes,
            model_output_dir=model_output_dir,
            random_seed=self.random_seed,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs
        )

        train_data, test_data = (
            model.prepare_data(
                train_df=train_df,
                test_df=test_df
            )
        )

        model.train()

        results = self.evaluate(
            model=model,
            dataset=dataset,
            test_data=test_data
        )

        model.save_model()

        print("\nEXPERIMENT FINISHED")
        print("Model: Encoder")
        print(f"Dataset: {dataset_name}")

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

    def create_split(
        self,
        dataset,
        dataset_name
    ):

        if dataset_name == "CIC-IDS2017":

            data = (
                dataset.get_data()
                .copy()
            )

            train_df, test_df = (
                train_test_split(
                    data,
                    test_size=self.test_size,
                    random_state=self.random_seed,
                    stratify=data["label"]
                )
            )

            train_df = (
                train_df
                .reset_index(
                    drop=True
                )
            )

            test_df = (
                test_df
                .reset_index(
                    drop=True
                )
            )

            return (
                train_df,
                test_df
            )

        if dataset_name == "UNSW-NB15":

            train_df = (
                dataset.get_train_data()
                .copy()
            )

            test_df = (
                dataset.get_test_data()
                .copy()
            )

            return (
                train_df,
                test_df
            )

        raise ValueError(
            f"Unsupported dataset: "
            f"{dataset_name}"
        )

    def get_feature_columns(
        self,
        data
    ):

        feature_columns = []

        for column in data.columns:

            if column == "label":
                continue

            if pd.api.types.is_numeric_dtype(
                data[column]
            ):

                feature_columns.append(
                    column
                )

        if not feature_columns:

            raise ValueError(
                "No numeric features found "
                "for Encoder."
            )

        return feature_columns

    def evaluate(
        self,
        model,
        dataset,
        test_data
    ):

        evaluator = Evaluator(
            class_names=dataset.unique_labels
        )

        X_test, y_test = test_data

        def predict_function(
            sample
        ):

            prediction = (
                model.predict_one(
                    sample
                )
            )

            return int(
                prediction
            )

        (
            predictions,
            inference_results
        ) = (
            evaluator.measure_inference_performance(
                predict_function=predict_function,
                samples=X_test
            )
        )

        results = (
            evaluator.evaluate_all(
                y_true=y_test,
                y_pred=predictions,
                inference_results=inference_results
            )
        )

        evaluator.print_classification_report(
            y_test,
            predictions
        )

        matrix = (
            evaluator.get_confusion_matrix(
                y_test,
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