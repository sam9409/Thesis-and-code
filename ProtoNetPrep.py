import numpy as np
import torch


class ProtoNetPrep:

    def __init__(
        self,
        shot_size,
        query_size,
        random_seed
    ):
        self.shot_size = shot_size
        self.query_size = query_size
        self.random_seed = random_seed

    def create_episode(
        self,
        data,
        feature_columns,
        label_column
    ):

        support_x = []
        support_y = []
        query_x = []
        query_y = []

        labels = sorted(
            data[label_column].unique()
        )

        required_samples = (
            self.shot_size
            + self.query_size
        )

        for label in labels:

            class_data = data[
                data[label_column] == label
            ]

            if len(class_data) < required_samples:
                continue

            samples = class_data.sample(
                n=required_samples
            )

            support = samples.iloc[
                :self.shot_size
            ]

            query = samples.iloc[
                self.shot_size:
            ]

            support_x.append(
                support[
                    feature_columns
                ].to_numpy(
                    dtype=np.float32
                )
            )

            query_x.append(
                query[
                    feature_columns
                ].to_numpy(
                    dtype=np.float32
                )
            )

            support_y.extend(
                [label] * len(support)
            )

            query_y.extend(
                [label] * len(query)
            )

        if len(support_x) == 0:
            raise ValueError(
                "No classes contain enough samples "
                "to create an episode."
            )

        support_x = torch.from_numpy(
            np.concatenate(
                support_x,
                axis=0
            )
        )

        query_x = torch.from_numpy(
            np.concatenate(
                query_x,
                axis=0
            )
        )

        support_y = torch.as_tensor(
            support_y,
            dtype=torch.long
        )

        query_y = torch.as_tensor(
            query_y,
            dtype=torch.long
        )

        return (
            support_x,
            support_y,
            query_x,
            query_y
        )