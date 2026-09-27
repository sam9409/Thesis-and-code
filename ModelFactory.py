from SetFit import SetFit
#from ProtoNet import ProtoNet
#from MAML import MAML
#from Autoencoder import Autoencoder


class ModelFactory:

    @staticmethod
    def create_model(
        model_name,
        unique_labels,
        model_output_dir,
        random_seed=42
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

            return ProtoNet(
                unique_labels=unique_labels,
                model_output_dir=model_output_dir,
                random_seed=random_seed
            )

        elif model_name == "MAML":

            return MAML(
                unique_labels=unique_labels,
                model_output_dir=model_output_dir,
                random_seed=random_seed
            )

        elif model_name == "Autoencoder":

            return Autoencoder(
                unique_labels=unique_labels,
                model_output_dir=model_output_dir,
                random_seed=random_seed
            )

        else:

            raise ValueError(
                f"Unknown model: {model_name}"
            )