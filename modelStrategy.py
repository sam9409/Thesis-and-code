from abc import ABC, abstractmethod


class ModelStrategy(ABC):

    @abstractmethod
    def prepare_data(
        self,
        dataset,
        train_df,
        test_df
    ):
        pass


    @abstractmethod
    def train(self):
        pass


    @abstractmethod
    def predict(
        self,
        samples
    ):
        pass


    @abstractmethod
    def get_test_samples(
        self,
        test_data
    ):
        pass


    @abstractmethod
    def save_model(self):
        pass