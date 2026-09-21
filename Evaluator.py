import time
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

class Evaluator:

    def __init__(self, class_names=None):
        self.class_names = class_names


    def evaluate_classification(
        self,
        y_true,
        y_pred
    ):
        """
        Calculate classification metrics.

        Metrics:
        - Accuracy
        - Macro Precision
        - Macro Recall
        - Macro F1
        """

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


    def print_classification_report(
        self,
        y_true,
        y_pred
    ):
  
        if self.class_names is not None:

            labels = list(
                range(len(self.class_names))
            )

            report = classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=self.class_names,
                zero_division=0
            )

        else:

            report = classification_report(
                y_true,
                y_pred,
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

        matrix = confusion_matrix(
            y_true,
            y_pred
        )
        return matrix


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
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            predictions.append(prediction)

        total_end = time.perf_counter()
        total_time = (total_end - total_start)
        average_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        worst_latency = np.max(latencies)

        if total_time > 0:

            throughput = (
                len(samples)
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

        results = (
            self.evaluate_classification(
                y_true,
                y_pred
            )
        )

        if inference_results is not None:

            results.update(inference_results)

        return results


    @staticmethod
    def print_results(results):

        print("Evaluation Results")
        for metric, value in results.items():

            print(
                f"{metric}: "
                f"{value}"
            )