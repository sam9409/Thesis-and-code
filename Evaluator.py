import time
import numpy as np
import os
import re

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class Evaluator:

    CIC_LABELS = {
        0: "BENIGN",
        1: "Bot",
        2: "DDoS",
        3: "DoS GoldenEye",
        4: "DoS Hulk",
        5: "FTP-Patator",
        6: "PortScan",
        7: "SSH-Patator"
    }

    UNSW_LABELS = {
        0: "Analysis",
        1: "Backdoor",
        2: "DoS",
        3: "Exploits",
        4: "Fuzzers",
        5: "Generic",
        6: "Normal",
        7: "Reconnaissance",
        8: "Shellcode",
        9: "Worms"
    }

    def __init__(self, class_names=None):

        self.class_names = class_names


    def evaluate_classification(
        self,
        y_true,
        y_pred
    ):

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        macro_precision = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        results = {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1
        }

        return results


    def get_labels(
        self,
        y_true,
        y_pred
    ):

        observed_labels = np.unique(
            np.concatenate([
                y_true,
                y_pred
            ])
        ).tolist()

        if self.class_names is not None:

            return sorted(
                set(observed_labels)
                | set(range(len(self.class_names)))
            )

        return observed_labels


    def print_classification_report(
        self,
        y_true,
        y_pred
    ):

        labels = self.get_labels(
            y_true,
            y_pred
        )

        if self.class_names is not None:

            label_mapping = dict(
                enumerate(self.class_names)
            )

            target_names = [
                label_mapping.get(label, str(label))
                for label in labels
            ]

        else:

            target_names = [
                str(label)
                for label in labels
            ]

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        )

        print("\nClassification Report:")
        print(report)

        return report


    def get_confusion_matrix(
        self,
        y_true,
        y_pred
    ):

        labels = self.get_labels(
            y_true,
            y_pred
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        )

        return matrix


    def evaluate_attack_types(
        self,
        y_true,
        y_pred
    ):

        labels = self.get_labels(
            y_true,
            y_pred
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        )

        matrix = self.get_confusion_matrix(
            y_true,
            y_pred
        )

        attack_results = {}

        for index, label in enumerate(labels):

            true_positive = matrix[index, index]

            false_positive = (
                matrix[:, index].sum()
                - true_positive
            )

            false_negative = (
                matrix[index, :].sum()
                - true_positive
            )

            true_negative = (
                matrix.sum()
                - true_positive
                - false_positive
                - false_negative
            )

            class_result = report[str(label)]

            attack_results[label] = {

                "precision":
                    class_result["precision"],

                "recall":
                    class_result["recall"],

                "f1_score":
                    class_result["f1-score"],

                "support":
                    int(class_result["support"]),

                "true_positives":
                    int(true_positive),

                "true_negatives":
                    int(true_negative),

                "false_positives":
                    int(false_positive),

                "false_negatives":
                    int(false_negative)
            }

        return attack_results


    def measure_inference_performance(
        self,
        predict_function,
        samples
    ):

        latencies = []
        predictions = []

        total_start = time.perf_counter()

        for sample in samples:

            start = time.perf_counter()

            prediction = predict_function(sample)

            end = time.perf_counter()

            latency_ms = (
                end - start
            ) * 1000

            latencies.append(latency_ms)
            predictions.append(prediction)

        total_end = time.perf_counter()

        total_time = (
            total_end - total_start
        )

        average_latency = (
            np.mean(latencies)
            if latencies else 0
        )

        median_latency = (
            np.median(latencies)
            if latencies else 0
        )

        worst_latency = (
            np.max(latencies)
            if latencies else 0
        )

        if total_time > 0:

            throughput = (
                len(predictions)
                / total_time
            )

        else:

            throughput = 0

        results = {

            "average_latency_ms":
                average_latency,

            "median_latency_ms":
                median_latency,

            "worst_latency_ms":
                worst_latency,

            "throughput_samples_per_second":
                throughput
        }

        return predictions, results


    def evaluate_all(
        self,
        y_true,
        y_pred,
        inference_results=None
    ):

        results = self.evaluate_classification(
            y_true,
            y_pred
        )

        if inference_results is not None:

            results.update(inference_results)

        results["attack_results"] = (
            self.evaluate_attack_types(
                y_true,
                y_pred
            )
        )

        results["confusion_matrix"] = (
            self.get_confusion_matrix(
                y_true,
                y_pred
            )
        )

        results["labels"] = self.get_labels(
            y_true,
            y_pred
        )

        results["class_names"] = self.class_names

        return results


    @staticmethod
    def print_results(results):

        print("\nEvaluation Results")

        excluded = {
            "attack_results",
            "confusion_matrix",
            "labels",
            "class_names"
        }

        for metric, value in results.items():

            if metric not in excluded:

                print(
                    f"{metric}: {value}"
                )


    @staticmethod
    def create_results_directory(
        model_name,
        dataset_name,
        shot_size=None
    ):

        shot_label = (
            f"{shot_size}shot"
            if shot_size is not None
            else "traditional"
        )

        project_directory = os.path.dirname(
            os.path.abspath(__file__)
        )

        results_directory = os.path.join(
            project_directory,
            "Results",
            model_name,
            dataset_name,
            shot_label
        )

        os.makedirs(
            results_directory,
            exist_ok=True
        )

        return results_directory


    @staticmethod
    def get_label_mapping(
        dataset_name,
        class_names=None
    ):

        if class_names is not None:

            return dict(
                enumerate(class_names)
            )

        if dataset_name == "CIC-IDS2017":

            return Evaluator.CIC_LABELS

        elif dataset_name == "UNSW-NB15":

            return Evaluator.UNSW_LABELS

        return {}


    @staticmethod
    def save_attack_results(
        results,
        model_name,
        dataset_name,
        model_output_dir,
        shot_size=None
    ):

        shot_label = (
            f"{shot_size}shot"
            if shot_size is not None
            else "traditional"
        )

        attack_directory = os.path.join(
            model_output_dir,
            "AttackResults"
        )

        os.makedirs(
            attack_directory,
            exist_ok=True
        )

        label_mapping = Evaluator.get_label_mapping(
            dataset_name=dataset_name,
            class_names=results.get("class_names")
        )

        attack_results = results.get(
            "attack_results",
            {}
        )

        for label, metrics in attack_results.items():

            attack_name = label_mapping.get(
                label,
                str(label)
            )

            safe_name = re.sub(
                r"[^a-zA-Z0-9_-]",
                "_",
                attack_name
            )

            file_path = os.path.join(
                attack_directory,
                f"{safe_name}.txt"
            )

            with open(
                file_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    f"Model: {model_name}\n"
                )

                file.write(
                    f"Dataset: {dataset_name}\n"
                )

                file.write(
                    f"Shot Size: {shot_label}\n"
                )

                file.write(
                    f"Attack: {attack_name}\n"
                )

                file.write(
                    f"Encoded Label: {label}\n"
                )

                file.write(
                    "\nAttack Evaluation Results\n"
                )

                for metric, value in metrics.items():

                    file.write(
                        f"{metric}: {value}\n"
                    )

        print(
            f"Individual attack results saved to: "
            f"{attack_directory}"
        )

        return attack_directory


    @staticmethod
    def save_classification_report(
        results,
        dataset_name,
        results_directory
    ):

        label_mapping = Evaluator.get_label_mapping(
            dataset_name=dataset_name,
            class_names=results.get("class_names")
        )

        attack_results = results.get(
            "attack_results",
            {}
        )

        file_path = os.path.join(
            results_directory,
            "classification_report.txt"
        )

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write("Classification Report\n\n")

            file.write(
                f"{'Attack':<25}"
                f"{'Precision':>12}"
                f"{'Recall':>12}"
                f"{'F1-score':>12}"
                f"{'Support':>12}\n"
            )

            file.write("-" * 73 + "\n")

            for label, metrics in attack_results.items():

                attack_name = label_mapping.get(
                    label,
                    str(label)
                )

                file.write(
                    f"{attack_name:<25}"
                    f"{metrics['precision']:>12.4f}"
                    f"{metrics['recall']:>12.4f}"
                    f"{metrics['f1_score']:>12.4f}"
                    f"{metrics['support']:>12}\n"
                )

        print(
            f"Classification report saved to: "
            f"{file_path}"
        )

        return file_path


    @staticmethod
    def save_confusion_matrix(
        results,
        dataset_name,
        results_directory
    ):

        label_mapping = Evaluator.get_label_mapping(
            dataset_name=dataset_name,
            class_names=results.get("class_names")
        )

        file_path = os.path.join(
            results_directory,
            "confusion_matrix.txt"
        )

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write("Confusion Matrix\n\n")

            file.write(
                "Rows: Actual classes\n"
            )

            file.write(
                "Columns: Predicted classes\n\n"
            )

            file.write("Class Order:\n")

            for label in results["labels"]:

                attack_name = label_mapping.get(
                    label,
                    str(label)
                )

                file.write(
                    f"{label}: {attack_name}\n"
                )

            file.write("\n")

            np.savetxt(
                file,
                results["confusion_matrix"],
                fmt="%d"
            )

        print(
            f"Confusion matrix saved to: "
            f"{file_path}"
        )

        return file_path


    @staticmethod
    def save_results(
        results,
        model_name,
        dataset_name,
        model_output_dir,
        shot_size=None,
        experiment_settings=None
    ):

        results_directory = (
            Evaluator.create_results_directory(
                model_name=model_name,
                dataset_name=dataset_name,
                shot_size=shot_size
            )
        )

        shot_label = (
            f"{shot_size}shot"
            if shot_size is not None
            else "traditional"
        )

        file_name = (
            f"{model_name.lower()}_"
            f"{shot_label}_results.txt"
        )

        file_path = os.path.join(
            results_directory,
            file_name
        )

        excluded = {
            "attack_results",
            "confusion_matrix",
            "labels",
            "class_names"
        }

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                f"Model: {model_name}\n"
            )

            file.write(
                f"Dataset: {dataset_name}\n"
            )

            file.write(
                f"Shot Size: {shot_label}\n"
            )

            if experiment_settings:

                file.write(
                    "\nExperiment Settings\n"
                )

                for key, value in experiment_settings.items():

                    file.write(
                        f"{key}: {value}\n"
                    )

            file.write(
                "\nEvaluation Results\n"
            )

            for metric, value in results.items():

                if metric not in excluded:

                    file.write(
                        f"{metric}: {value}\n"
                    )

        print(
            f"Results saved to: {file_path}"
        )

        if "attack_results" in results:

            Evaluator.save_attack_results(
                results=results,
                model_name=model_name,
                dataset_name=dataset_name,
                model_output_dir=results_directory,
                shot_size=shot_size
            )

            Evaluator.save_classification_report(
                results=results,
                dataset_name=dataset_name,
                results_directory=results_directory
            )

        else:

            print(
                "Warning: Individual attack results "
                "were not provided."
            )

        if "confusion_matrix" in results:

            Evaluator.save_confusion_matrix(
                results=results,
                dataset_name=dataset_name,
                results_directory=results_directory
            )

        print(
            f"\nAll evaluation results saved to: "
            f"{results_directory}"
        )

        return file_path