from trainingSplit import trainingSplit
from SetFit import SetFit
from Evaluator import Evaluator
import pandas as pd

class ExperimentRunner:
    "I will add comments to this class later, I will also add more functionality to it, like saving the results to a file and maybe plotting the results."
    def __init__(
        self,
        shot_size=5,
        test_size=100,
        random_seed=42
    ):

        self.shot_size = shot_size
        self.test_size = test_size
        self.random_seed = random_seed


    def run_setfit(
        self,
        dataset,
        dataset_name,
        model_output_dir
    ):

        print(f"DATASET: {dataset_name}")

        dataset.load_data()
        dataset.clean_data()
        dataset.encode_labels()

        if dataset_name == "UNSW-NB15":

            print(
                f"\nCreating {self.shot_size}-shot "
                "UNSW-NB15 training set"
            )

            training_pool = (dataset.get_train_data())

            test_df = (
                dataset.get_test_data()
                .copy()
            )


            train_parts = []

            for label_id, group in (training_pool.groupby("label")):

                if len(group) < self.shot_size:

                    raise ValueError(
                        f"Class {label_id} only has "
                        f"{len(group)} examples. "
                    )

                selected = group.sample(
                    n=self.shot_size,
                    random_state=self.random_seed
                )

                train_parts.append(selected)

            train_df = pd.concat(
                train_parts,
                ignore_index=True
            )


        else:
            data = dataset.get_data()

            splitter = trainingSplit(
                data=data,
                shot_size=self.shot_size,
                test_size=self.test_size,
                random_seed=self.random_seed
            )


            train_df, test_df = (splitter.create_split())

            splitter.print_split_info()
            splitter.validate_split()
            splitter.check_overlap()

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

        print("\nConverting data to text...")

        train_df = dataset.convert_to_text(train_df)

        test_df = dataset.convert_to_text(test_df)

        classifier = SetFit(
            unique_labels=dataset.unique_labels,
            model_name=("sentence-transformers/all-MiniLM-L6-v2"),
            model_output_dir=model_output_dir,
            batch_size=16,
            num_epochs=1,
            random_seed=self.random_seed
        )

        classifier.create_datasets(
            train_df=train_df,
            test_df=test_df
        )

        classifier.load_model()
        classifier.create_trainer()
        classifier.train()

        evaluator = Evaluator(class_names=dataset.unique_labels)


        def setfit_predict(text):
            prediction = (classifier.model.predict([text]))
            predicted_label = prediction[0]

            return dataset.label_to_id[predicted_label]

        test_samples = (
            test_df["text"]
            .tolist()
        )

        y_true = (
            test_df["label"].to_numpy()
        )

        predictions, inference_results = (
            evaluator.measure_inference_performance(
                predict_function=setfit_predict,
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
        #AI reccomened the matrix
        matrix = (
            evaluator.get_confusion_matrix(
                y_true,
                predictions
            )
        )

        print("\nConfusion Matrix:")
        print(matrix)

        evaluator.print_results(results)

        classifier.save_model()

        print(
            f"\nFinished SetFit on "
            f"{dataset_name}"
        )

        return results