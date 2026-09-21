import pandas as pd
import numpy as np


class UNSWDataPrepSetFit:

    def __init__(self, train_file_path, test_file_path):

        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.train_data = None
        self.test_data = None
        self.unique_labels = None
        self.label_to_id = None
        self.id_to_label = None

    def load_data(self):

        print("Loading UNSW-NB15 training data")
        self.train_data = pd.read_csv(self.train_file_path)

        print("Loading UNSW-NB15 testing data")
        self.test_data = pd.read_csv(
            self.test_file_path
        )

        print(
            "Training rows:",
            len(self.train_data)
        )

        print(
            "Testing rows:",
            len(self.test_data)
        )

    def clean_data(self):

        self.train_data.columns = (self.train_data.columns.str.strip())
        self.test_data.columns = (self.test_data.columns.str.strip())        
        self.train_data["attack_cat"] = (
            self.train_data["attack_cat"]
            .astype(str)
            .str.strip()
        )

        self.test_data["attack_cat"] = (
            self.test_data["attack_cat"]
            .astype(str)
            .str.strip()
        )

        self.train_data.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        self.test_data.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        print("UNSW-NB15 data cleaned.")

    def encode_labels(self):


        self.train_data.rename(
            columns={"label": "binary_label"},
            inplace=True
        )

        self.test_data.rename(
            columns={"label": "binary_label"},
            inplace=True
        )


        all_labels = pd.concat(
            [
                self.train_data["attack_cat"],
                self.test_data["attack_cat"]
            ]
        )

        self.unique_labels = sorted(
            all_labels.unique()
        )


        self.label_to_id = {
            label: label_id
            for label_id, label
            in enumerate(self.unique_labels)
        }


        self.id_to_label = {
            label_id: label
            for label, label_id
            in self.label_to_id.items()
        }

        self.train_data["label"] = (
            self.train_data["attack_cat"]
            .map(self.label_to_id)
        )

        self.test_data["label"] = (
            self.test_data["attack_cat"]
            .map(self.label_to_id)
        )


        print("\nUNSW-NB15 label mapping:")

        for label, label_id in self.label_to_id.items():
            print(
                f"{label} -> {label_id}"
            )

    def row_to_text(self, row):

        features = [
            "proto",
            "service",
            "state",
            "dur",
            "spkts",
            "dpkts",
            "sbytes",
            "dbytes",
            "rate",
            "sttl",
            "dttl",
            "sload",
            "dload",
            "sloss",
            "dloss",
            "sinpkt",
            "dinpkt",
            "sjit",
            "djit",
            "tcprtt",
            "synack",
            "ackdat",
            "smean",
            "dmean"
        ]

        text_parts = []

        for feature in features:

            if feature in row.index:

                value = row[feature]

                text_parts.append(
                    f"{feature}={value}"
                )

        return "; ".join(text_parts)


    def convert_to_text(self, data):

        data = data.copy()

        data["text"] = data.apply(
            self.row_to_text,
            axis=1
        )

        return data

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data