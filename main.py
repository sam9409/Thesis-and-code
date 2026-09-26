from dataPrep import dataPrep
from UNSWDataPrepSetFit import UNSWDataPrepSetFit
from ExperimentRunner import ExperimentRunner
from Interface import interface


def main():

    random_seed = 42
    test_size = 100

    # Get user selections
    user_interface = interface()

    model_choice = user_interface.select_model()
    dataset_choice = user_interface.select_dataset()
    shot_size = user_interface.select_shot_size()

    print("\nEXPERIMENT SETTINGS")
    print(f"Model: {model_choice}")
    print(f"Dataset: {dataset_choice}")
    print(f"Shot Size: {shot_size}")
    print(f"Test Size: {test_size}")

    # Create experiment runner
    runner = ExperimentRunner(
        shot_size=shot_size,
        test_size=test_size,
        random_seed=random_seed
    )

    # CIC-IDS2017
    if dataset_choice == "CIC-IDS2017":

        cic_folder = (
            r"C:\Users\snmoh\Desktop\Thesis and code"
            r"\MachineLearningCSV\MachineLearningCVE"
        )

        dataset = dataPrep(
            folder_path=cic_folder
        )

        model_output_dir = (
            r"C:\Users\snmoh\Desktop\Thesis and code"
            r"\setfit-cicids2017-model"
        )

    # UNSW-NB15
    elif dataset_choice == "UNSW-NB15":

        unsw_train_file = (
            r"C:\Users\snmoh\Desktop\Thesis and code"
            r"\UNSWDATA\CSV Files\Training and Testing Sets"
            r"\UNSW_NB15_training-set.csv"
        )

        unsw_test_file = (
            r"C:\Users\snmoh\Desktop\Thesis and code"
            r"\UNSWDATA\CSV Files\Training and Testing Sets"
            r"\UNSW_NB15_testing-set.csv"
        )

        dataset = UNSWDataPrepSetFit(
            train_file_path=unsw_train_file,
            test_file_path=unsw_test_file
        )

        model_output_dir = (
            r"C:\Users\snmoh\Desktop\Thesis and code"
            r"\setfit-unsw-nb15-model"
        )

    # Run selected model
    if model_choice == "SetFit":

        results = runner.run_setfit(
            dataset=dataset,
            dataset_name=dataset_choice,
            model_output_dir=model_output_dir
        )

        print("\nFINAL RESULTS")

        for metric, value in results.items():
            print(f"{metric}: {value}")

    elif model_choice == "ProtoNet":
        print("ProtoNet has not been implemented yet.")

    elif model_choice == "MAML":
        print("MAML has not been implemented yet.")

    elif model_choice == "Autoencoder":
        print("Autoencoder has not been implemented yet.")


if __name__ == "__main__":
    main()