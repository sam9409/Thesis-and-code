from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments


class SetFit:

    def __init__(
        self,
        unique_labels,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_output_dir: str = "setfit_model",
        batch_size: int = 16,
        num_epochs: int = 1,
        random_seed: int = 42
    ):
        self.unique_labels = unique_labels
        self.model_name = model_name
        self.model_output_dir = model_output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_seed = random_seed
        self.model = None
        self.trainer = None
        self.train_dataset = None
        self.test_dataset = None

    def create_datasets(
        self,
        train_df,
        test_df
    ):

        self.train_dataset = Dataset.from_pandas(
            train_df[["text", "label"]],
            preserve_index=False
        )

        self.test_dataset = Dataset.from_pandas(
            test_df[["text", "label"]],
            preserve_index=False
        )

        print("\nTraining Dataset:")
        print(self.train_dataset)

        print("\nTest Dataset:")
        print(self.test_dataset)

        return self.train_dataset, self.test_dataset


    def load_model(self):
      
        print(f"\nLoading SetFit model: {self.model_name}")

        self.model = SetFitModel.from_pretrained(
            self.model_name,
            labels=self.unique_labels
        )

        print(f"Model loaded on: {self.model.device}")

        return self.model

    def create_trainer(self):
        training_args = TrainingArguments(
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            seed=self.random_seed
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset
        )

        print("\nTrainer created")

        return self.trainer

    def train(self):

        print("\nStarting SetFit training")
        self.trainer.train()
        print("\nTraining complete")


    def evaluate(self):
  
        print("\nEvaluating model")
        metrics = self.trainer.evaluate()
        print("\nEvaluation Results:")

        for name, value in metrics.items():
            print(f"{name}: {value}")

        return metrics



    def predict(self, text):
        prediction = self.model.predict([text])

        return prediction


    def save_model(self):

        print(
            f"\nSaving model to:\n"
            f"{self.model_output_dir}"
        )

        self.model.save_pretrained(
            self.model_output_dir
        )

        print("\nModel saved")


    def run_training(
        self,
        train_df,
        test_df
    ):

        self.create_datasets(
            train_df,
            test_df
        )

        self.load_model()
        self.create_trainer()
        self.train()
        metrics = self.evaluate()
        self.save_model()

        return metrics