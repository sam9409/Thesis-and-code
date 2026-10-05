from dataPrep import dataPrep
from UNSWDataPrepSetFit import UNSWDataPrepSetFit

from SetFitRunner import SetFitRunner
from ProtoRunner import ProtoNetRunner
from EncoderRunner import EncoderRunner 


def main():

    print("\nSelect Model")
    print("1 SetFit")
    print("2 ProtoNet")
    print("3 MAML")
    print("4 Autoencoder")

    model_input = input(
        "\nEnter model choice: "
    )

    model_options = {
        "1": "SetFit",
        "2": "ProtoNet",
        "3": "MAML",
        "4": "Autoencoder"
    }

    if model_input not in model_options:
        raise ValueError(
            "Invalid model choice."
        )

    model_choice = model_options[
        model_input
    ]

    print("\nSelect Dataset")
    print("1 CIC-IDS2017")
    print("2 UNSW-NB15")
    print("3 Both")

    dataset_input = input(
        "\nEnter dataset choice: "
    )

    dataset_options = {
        "1": "CIC-IDS2017",
        "2": "UNSW-NB15",
        "3": "Both"
    }

    if dataset_input not in dataset_options:
        raise ValueError(
            "Invalid dataset choice."
        )

    dataset_choice = dataset_options[
        dataset_input
    ]

    print("\nSelect Shot Size")
    print("1 5-shot")
    print("2 10-shot")
    print("3 20-shot")

    shot_input = input(
        "\nEnter shot size choice: "
    )

    shot_options = {
        "1": 5,
        "2": 10,
        "3": 20
    }

    if shot_input not in shot_options:
        raise ValueError(
            "Invalid shot size choice."
        )

    shot_size = shot_options[
        shot_input
    ]

    test_size = 100
    random_seed = 42

    if model_choice == "SetFit":

        runner = SetFitRunner(
            shot_size=shot_size,
            test_size=test_size,
            random_seed=random_seed,
            batch_size=16,
            num_epochs=1
        )

    elif model_choice == "ProtoNet":

        runner = ProtoNetRunner(
            shot_size=shot_size,
            test_size=test_size,
            random_seed=random_seed,
            number_of_episodes=100,
            learning_rate=0.001
        )

    elif model_choice == "MAML":

        raise NotImplementedError(
            "MAMLRunner has not been implemented yet."
        )

    elif model_choice == "Autoencoder":

        runner = EncoderRunner(
            test_size=test_size,
            random_seed=random_seed,
            learning_rate=0.001,
            batch_size=64,
            epochs=20
        )

    cic_folder = (
        r"C:\Users\snmoh\Desktop\Thesis and code"
        r"\MachineLearningCSV\MachineLearningCVE"
    )

    unsw_train_file = (
        r"C:\Users\snmoh\Desktop\Thesis and code"
        r"\UNSWDATA\CSV Files"
        r"\Training and Testing Sets"
        r"\UNSW_NB15_training-set.csv"
    )

    unsw_test_file = (
        r"C:\Users\snmoh\Desktop\Thesis and code"
        r"\UNSWDATA\CSV Files"
        r"\Training and Testing Sets"
        r"\UNSW_NB15_testing-set.csv"
    )

    if dataset_choice in (
        "CIC-IDS2017",
        "Both"
    ):

        cic_dataset = dataPrep(
            folder_path=cic_folder
        )

        cic_output_dir = (
            rf"{model_choice.lower()}"
            rf"-cicids2017-model"
        )

        runner.run(
            dataset=cic_dataset,
            dataset_name="CIC-IDS2017",
            model_output_dir=cic_output_dir
        )

    if dataset_choice in (
        "UNSW-NB15",
        "Both"
    ):

        unsw_dataset = UNSWDataPrepSetFit(
            train_file_path=unsw_train_file,
            test_file_path=unsw_test_file
        )

        unsw_output_dir = (
            rf"{model_choice.lower()}"
            rf"-unsw-nb15-model"
        )

        runner.run(
            dataset=unsw_dataset,
            dataset_name="UNSW-NB15",
            model_output_dir=unsw_output_dir
        )


if __name__ == "__main__":
    main()