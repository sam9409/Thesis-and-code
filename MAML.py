
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.func import functional_call


class MAML(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes,
        hidden_size=128,
        embedding_size=64,
        inner_lr=0.01,
        inner_steps=5,
        first_order=False
    ):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),

            nn.Linear(embedding_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)

    def adapt(
        self,
        support_x,
        support_y
    ):
        params = dict(self.named_parameters())

        for _ in range(self.inner_steps):

            logits = functional_call(
                self,
                params,
                (support_x,)
            )

            loss = F.cross_entropy(
                logits,
                support_y
            )

            grads = torch.autograd.grad(
                loss,
                tuple(params.values()),
                create_graph=not self.first_order
            )

            params = {
                name: param - self.inner_lr * grad
                for (name, param), grad
                in zip(params.items(), grads)
            }

        return params

    def query_loss(
        self,
        query_x,
        query_y,
        adapted_params
    ):
        logits = functional_call(
            self,
            adapted_params,
            (query_x,)
        )

        return F.cross_entropy(
            logits,
            query_y
        )

    def meta_train_step(
        self,
        episodes,
        optimizer
    ):
        self.train()
        optimizer.zero_grad()

        losses = []

        for support_x, support_y, query_x, query_y in episodes:

            adapted_params = self.adapt(
                support_x,
                support_y
            )

            loss = self.query_loss(
                query_x,
                query_y,
                adapted_params
            )

            losses.append(loss)

        meta_loss = torch.stack(losses).mean()

        meta_loss.backward()
        optimizer.step()

        return meta_loss.item()

    def predict(
        self,
        query_x,
        adapted_params=None
    ):
        if adapted_params is None:
            with torch.no_grad():
                logits = self.forward(query_x)
        else:
            logits = functional_call(
                self,
                adapted_params,
                (query_x,)
            )

        return torch.argmax(
            logits,
            dim=1
        )
