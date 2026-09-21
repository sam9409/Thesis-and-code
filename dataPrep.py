import pandas as pd
import numpy as np
import glob
import os


class dataPrep:

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.data = None
        self.unique_labels = None
        self.label_to_id = None
        self.id_to_label = None


    def load_csv_files(self):

        csv_files = glob.glob(
            os.path.join(
                self.folder_path,
                "*.csv"
            )
        )

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in:\n"
                f"{self.folder_path}"
            )

        dataframes = []

        print(
            f"Loading CIC-IDS2017 files from:\n"
            f"{self.folder_path}"
        )

        for csv_file in csv_files:

            print(
                f"Loading: "
                f"{os.path.basename(csv_file)}"
            )

            df = pd.read_csv(csv_file)

            df.columns = (
                df.columns
                .str.strip()
            )

            # Save source file for reference only
            df["SourceFile"] = (os.path.basename(csv_file))

            dataframes.append(df)


        self.data = pd.concat(
            dataframes,
            ignore_index=True
        )

        print(
            f"\nFiles loaded: "
            f"{len(csv_files)}"
        )

        print(
            f"Rows before cleaning: "
            f"{len(self.data)}"
        )

        return self.data

    def load_data(self):
        return self.load_csv_files()


    def clean_data(self):
        self._check_data_loaded()

        # Replace infinity with NaN
        self.data.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        # Remove rows containing missing values
        self.data.dropna(inplace=True)

        # Remove extra whitespace around labels
        self.data["Label"] = (
            self.data["Label"]
            .astype(str)
            .str.strip()
        )

        # Reset indexes after rows were removed
        self.data.reset_index(
            drop=True,
            inplace=True
        )

        print("\nClasses found:")

        print(
            self.data["Label"]
            .value_counts()
        )

        print(
            f"\nRows after cleaning: "
            f"{len(self.data)}"
        )

        return self.data

    def encode_labels(self):
        self._check_data_loaded()


        self.unique_labels = sorted(
            self.data["Label"].unique()
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

        self.data["label"] = (
            self.data["Label"]
            .map(self.label_to_id)
        )

        print("\nLabel mapping:")

        for label, label_id in self.label_to_id.items():

            print(
                f"{label_id}: {label}"
            )


        return self.data


    @staticmethod
    def row_to_text(row):
        """
        used AI for this part of code, I will change it later to a map or maybe another class 
        """

        return (
            f"Network flow to destination port "
            f"{int(row['Destination Port'])}. "

            f"Flow duration was "
            f"{row['Flow Duration']} microseconds. "

            f"The flow contained "
            f"{int(row['Total Fwd Packets'])} "
            f"forward packets and "
            f"{int(row['Total Backward Packets'])} "
            f"backward packets. "

            f"Total forward traffic was "
            f"{row['Total Length of Fwd Packets']} "
            f"bytes and total backward traffic was "
            f"{row['Total Length of Bwd Packets']} "
            f"bytes. "

            f"Maximum forward packet length was "
            f"{row['Fwd Packet Length Max']} bytes. "

            f"Minimum forward packet length was "
            f"{row['Fwd Packet Length Min']} bytes. "

            f"Mean forward packet length was "
            f"{row['Fwd Packet Length Mean']:.2f} bytes. "

            f"Maximum backward packet length was "
            f"{row['Bwd Packet Length Max']} bytes. "

            f"Minimum backward packet length was "
            f"{row['Bwd Packet Length Min']} bytes. "

            f"Mean backward packet length was "
            f"{row['Bwd Packet Length Mean']:.2f} bytes. "

            f"Flow rate was "
            f"{row['Flow Bytes/s']:.2f} bytes per second "
            f"and "
            f"{row['Flow Packets/s']:.2f} "
            f"packets per second. "

            f"Mean flow inter-arrival time was "
            f"{row['Flow IAT Mean']:.2f} microseconds. "

            f"Maximum flow inter-arrival time was "
            f"{row['Flow IAT Max']} microseconds. "

            f"Forward packet rate was "
            f"{row['Fwd Packets/s']:.2f} "
            f"packets per second and backward packet rate was "
            f"{row['Bwd Packets/s']:.2f} "
            f"packets per second. "

            f"SYN flag count was "
            f"{int(row['SYN Flag Count'])}. "

            f"ACK flag count was "
            f"{int(row['ACK Flag Count'])}. "

            f"PSH flag count was "
            f"{int(row['PSH Flag Count'])}. "

            f"RST flag count was "
            f"{int(row['RST Flag Count'])}. "

            f"Average packet size was "
            f"{row['Average Packet Size']:.2f} bytes."
        )


    def convert_to_text(self, data: pd.DataFrame):
        data = data.copy()

        data["text"] = data.apply(
            self.row_to_text,
            axis=1
        )

        return data


    def print_label_counts(self):
        self._check_data_loaded()

        print(
            self.data["Label"]
            .value_counts()
        )


    def get_data(self):
        self._check_data_loaded()

        return self.data


    def _check_data_loaded(self):

        if self.data is None:

            raise ValueError(
                "Dataset has not been loaded. "
                "Run load_csv_files() first."
            )