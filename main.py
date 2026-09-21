from dataPrep import dataPrep
from UNSWDataPrepSetFit import UNSWDataPrepSetFit
from ExperimentRunner import ExperimentRunner


def main():

    shot_size = 5
    test_size = 100
    random_seed = 42

    runner = ExperimentRunner(
        shot_size=shot_size,
        test_size=test_size,
        random_seed=random_seed
    )

    print("\n")
    print("Start CIC-IDS2017")


    cic_folder = (
        r"C:\Users\snmoh\Desktop\Thesis and code"
        r"\MachineLearningCSV\MachineLearningCVE"
    )

    cic_model_output = (
        r"C:\Users\snmoh\Desktop\Thesis and code"
        r"\setfit-cicids2017-model"
    )

    cic_dataset = dataPrep(
        folder_path=cic_folder
    )

    cic_results = runner.run_setfit(
        dataset=cic_dataset,
        dataset_name="CIC-IDS2017",
        model_output_dir=cic_model_output
    )

    print("\nFinished CIC-IDS2017")
    print("\nSTARTING UNSW-NB15 EXPERIMENT")
 
    unsw_train_file = (
        r"C:\Users\snmoh\Desktop\Thesis and code\UNSWDATA\CSV Files\Training and Testing Sets"
        r"\UNSW_NB15_training-set.csv"
    )

    unsw_test_file = (
        r"C:\Users\snmoh\Desktop\Thesis and code\UNSWDATA\CSV Files\Training and Testing Sets"
        r"\UNSW_NB15_testing-set.csv"
    )

    unsw_model_output = (
        r"C:\Users\snmoh\Desktop\Thesis and code"
        r"\setfit-unsw-nb15-model"
    )


    unsw_dataset = UNSWDataPrepSetFit(
        train_file_path=unsw_train_file,
        test_file_path=unsw_test_file
    )


    unsw_results = runner.run_setfit(
        dataset=unsw_dataset,
        dataset_name="UNSW-NB15",
        model_output_dir=unsw_model_output
    )


    print("\nFinished UNSW-NB15 Experiment\n")
    print("SerFit Results")
    print("\nCIC-IDS2017 Results")

    for metric, value in cic_results.items():
        print(f"{metric}: {value}")

    print("\nUNSW-NB15 Results")

    for metric, value in unsw_results.items():
        print(f"{metric}: {value}")

    print("\nfinished")

if __name__ == "__main__":
    main()