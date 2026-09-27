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

        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, embedding_size)
        )


    def forward(self, x):
        return self.embedding(x)


    def calculate_prototypes(
        self,
        support_embeddings,
        support_labels
    ):

        prototypes = []
        labels = torch.unique(
            support_labels
        )

        for label in labels:

            class_embeddings = (
                support_embeddings[
                    support_labels == label
                ]
            )

            prototype = class_embeddings.mean(
                dim=0
            )

            prototypes.append(prototype)

        return torch.stack(prototypes), labels


    def calculate_distances(
        self,
        query_embeddings,
        prototypes
    ):

        return torch.cdist(
            query_embeddings,
            prototypes
        ) ** 2


    def episode_loss(
        self,
        support_x,
        support_y,
        query_x,
        query_y
    ):

        # Create embeddings
        support_embeddings = self(
            support_x
        )

        query_embeddings = self(
            query_x
        )

        # Create prototypes
        prototypes, labels = (
            self.calculate_prototypes(
                support_embeddings,
                support_y
            )
        )

        # Calculate distances
        distances = self.calculate_distances(
            query_embeddings,
            prototypes
        )

        # Convert labels to prototype positions
        targets = torch.tensor(
            [
                (labels == label)
                .nonzero(as_tuple=True)[0]
                .item()

                for label in query_y
            ],
            device=query_y.device
        )

        # Negative distance = logits
        logits = -distances

        loss = F.cross_entropy(
            logits,
            targets
        )

        return loss