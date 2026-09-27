from dataPrep import dataPrep
from UNSWDataPrepSetFit import UNSWDataPrepSetFit
from ExperimentRunner import ExperimentRunner
from Interface import interface


def main():
    random_seed = 42
    test_size = 100

    user_interface = interface()

    model_choice = user_interface.select_model()
    dataset_choice = user_interface.select_dataset()
    shot_size = user_interface.select_shot_size()

    print("\nEXPERIMENT SETTINGS")
    print(f"Model: {model_choice}")
    print(f"Dataset: {dataset_choice}")
    print(f"Shot Size: {shot_size}")
    print(f"Test Size: {test_size}")

    runner = ExperimentRunner(
        shot_size=shot_size,
        test_size=test_size,
        random_seed=random_seed
    )

    cic_folder = (
        r"C:\Users\snmoh\Desktop\Thesis and code"
        r"\MachineLearningCSV\MachineLearningCVE"
    )

    cic_model_output = (
        r"C:\Users\snmoh\Desktop\Thesis and code"
        r"\cicids2017-model"
    )

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

    unsw_model_output = (
        r"C:\Users\snmoh\Desktop\Thesis and code"
        r"\unsw-nb15-model"
    )

    if dataset_choice == "CIC-IDS2017":

        dataset = dataPrep(
            folder_path=cic_folder
        )

        results = runner.run(
            model_name=model_choice,
            dataset=dataset,
            dataset_name="CIC-IDS2017",
            model_output_dir=cic_model_output
        )

        print("\nCIC-IDS2017 RESULTS")

        for metric, value in results.items():
            print(f"{metric}: {value}")


    elif dataset_choice == "UNSW-NB15":

        dataset = UNSWDataPrepSetFit(
            train_file_path=unsw_train_file,
            test_file_path=unsw_test_file
        )

        results = runner.run(
            model_name=model_choice,
            dataset=dataset,
            dataset_name="UNSW-NB15",
            model_output_dir=unsw_model_output
        )

        print("\nUNSW-NB15 RESULTS")

        for metric, value in results.items():
            print(f"{metric}: {value}")


    elif dataset_choice == "Both":

        print("\nSTARTING CIC-IDS2017")

        cic_dataset = dataPrep(
            folder_path=cic_folder
        )

        cic_results = runner.run(
            model_name=model_choice,
            dataset=cic_dataset,
            dataset_name="CIC-IDS2017",
            model_output_dir=cic_model_output
        )

        print("\nFINISHED CIC-IDS2017")

        print("\nSTARTING UNSW-NB15")

        unsw_dataset = UNSWDataPrepSetFit(
            train_file_path=unsw_train_file,
            test_file_path=unsw_test_file
        )

        unsw_results = runner.run(
            model_name=model_choice,
            dataset=unsw_dataset,
            dataset_name="UNSW-NB15",
            model_output_dir=unsw_model_output
        )

        print("\nFINISHED UNSW-NB15")

        print("\nFINAL RESULTS")

        print("\nCIC-IDS2017 Results")

        for metric, value in cic_results.items():
            print(f"{metric}: {value}")

        print("\nUNSW-NB15 Results")

        for metric, value in unsw_results.items():
            print(f"{metric}: {value}")


    print("\nFINISHED")


if __name__ == "__main__":
    main()