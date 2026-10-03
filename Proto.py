import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=128,
        embedding_size=64
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_size,
                embedding_size
            )
        )

    def forward(
        self,
        x
    ):
        return self.embedding(
            x
        )

    def calculate_prototypes(
        self,
        support_embeddings,
        support_labels
    ):

        labels = torch.unique(
            support_labels
        )

        prototypes = []

        for label in labels:

            class_embeddings = (
                support_embeddings[
                    support_labels == label
                ]
            )

            prototype = (
                class_embeddings.mean(
                    dim=0
                )
            )

            prototypes.append(
                prototype
            )

        prototypes = torch.stack(
            prototypes
        )

        return (
            prototypes,
            labels
        )

    def calculate_distances(
        self,
        query_embeddings,
        prototypes
    ):

        distances = torch.cdist(
            query_embeddings,
            prototypes
        )

        return distances ** 2

    def episode_loss(
        self,
        support_x,
        support_y,
        query_x,
        query_y
    ):

        support_embeddings = self(
            support_x
        )

        query_embeddings = self(
            query_x
        )

        prototypes, labels = (
            self.calculate_prototypes(
                support_embeddings,
                support_y
            )
        )

        distances = (
            self.calculate_distances(
                query_embeddings,
                prototypes
            )
        )

        targets = self.create_targets(
            query_y,
            labels
        )

        logits = -distances

        loss = F.cross_entropy(
            logits,
            targets
        )

        return loss

    def create_targets(
        self,
        query_labels,
        prototype_labels
    ):

        targets = []

        for label in query_labels:

            target = (
                prototype_labels
                == label
            ).nonzero(
                as_tuple=True
            )[0].item()

            targets.append(
                target
            )

        return torch.tensor(
            targets,
            dtype=torch.long,
            device=query_labels.device
        )

    def predict_from_prototypes(
        self,
        samples,
        prototypes,
        prototype_labels
    ):

        embeddings = self(
            samples
        )

        distances = (
            self.calculate_distances(
                embeddings,
                prototypes
            )
        )

        nearest_prototypes = (
            torch.argmin(
                distances,
                dim=1
            )
        )

        predictions = (
            prototype_labels[
                nearest_prototypes
            ]
        )

        return predictions