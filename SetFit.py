from datasets import Dataset
from setfit import (
    SetFitModel,
    Trainer,
    TrainingArguments
)


class SetFit:

    def __init__(
        self,
        unique_labels,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_output_dir="setfit_model",
        batch_size=16,
        num_epochs=1,
        random_seed=42
    ):

        self.unique_labels = unique_labels
        self.model_name = model_name
        self.model_output_dir = model_output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_seed = random_seed

        self.model = None
        self.trainer = None

    def load_model(
        self
    ):

        print(
            f"\nLoading SetFit model: "
            f"{self.model_name}"
        )

        self.model = (
            SetFitModel.from_pretrained(
                self.model_name,
                labels=self.unique_labels
            )
        )

        print(
            f"Model loaded on: "
            f"{self.model.device}"
        )

        return self.model

    def create_dataset(
        self,
        data
    ):

        return Dataset.from_pandas(
            data[
                [
                    "text",
                    "label"
                ]
            ],
            preserve_index=False
        )

    def create_trainer(
        self,
        train_dataset,
        test_dataset=None
    ):

        training_args = (
            TrainingArguments(
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                seed=self.random_seed
            )
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        return self.trainer

    def train(
        self,
        train_dataset,
        test_dataset=None
    ):

        if self.model is None:
            self.load_model()

        self.create_trainer(
            train_dataset=train_dataset,
            test_dataset=test_dataset
        )

        print(
            "\nStarting SetFit training"
        )

        self.trainer.train()

        print(
            "\nSetFit training complete"
        )

    def predict(
        self,
        samples
    ):

        if self.model is None:

            raise ValueError(
                "SetFit model has not been loaded."
            )

        return self.model.predict(
            samples
        )

    def save_model(
        self
    ):

        if self.model is None:

            raise ValueError(
                "SetFit model has not been loaded."
            )

        print(
            f"\nSaving model to: "
            f"{self.model_output_dir}"
        )

        self.model.save_pretrained(
            self.model_output_dir
        )

        print(
            "\nSetFit model saved"
        )