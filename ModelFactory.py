from SetFit import SetFit
from Proto import ProtoNet
# from MAML import MAML
from Encoder import Encoder


class ModelFactory:

    @staticmethod
    def create_model(
        model_name,
        unique_labels,
        model_output_dir,
        random_seed=42,
        input_size=None
    ):

        if model_name == "SetFit":

            return SetFit(
                unique_labels=unique_labels,
                model_name=(
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"
                ),
                model_output_dir=model_output_dir,
                batch_size=16,
                num_epochs=1,
                random_seed=random_seed
            )

        elif model_name == "ProtoNet":

            if input_size is None:
                raise ValueError(
                    "input_size is required for ProtoNet."
                )

            return ProtoNet(
                input_size=input_size,
                hidden_size=128,
                embedding_size=64
            )

        elif model_name == "MAML":

            return MAML(
                unique_labels=unique_labels,
                model_output_dir=model_output_dir,
                random_seed=random_seed
            )

        elif model_name == "Autoencoder":

            if input_size is None:
                raise ValueError(
                    "input_size is required for Autoencoder."
                )

            return Encoder(
                input_dim=input_size,
                num_classes=len(unique_labels),
                model_output_dir=model_output_dir,
                random_seed=random_seed
            )

        else:

            raise ValueError(
                f"Unknown model: {model_name}"
            )