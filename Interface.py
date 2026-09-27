class interface:

    def select_model(self):

        print("\nSELECT MODEL")
        print("1. SetFit")
        print("2. Prototypical Network")
        print("3. MAML")
        print("4. Autoencoder")

        while True:

            choice = input("\nEnter model number (1-4): ")

            if choice == "1":
                return "SetFit"

            elif choice == "2":
                return "ProtoNet"

            elif choice == "3":
                return "MAML"

            elif choice == "4":
                return "Autoencoder"

            else:
                print("Invalid choice. Please enter 1-4.")


    def select_dataset(self):

        print("\nSELECT DATASET")
        print("1. CIC-IDS2017")
        print("2. UNSW-NB15")
        print("3. Both")

        while True:

            choice = input("\nEnter dataset number (1-3): ")

            if choice == "1":
                return "CIC-IDS2017"

            elif choice == "2":
                return "UNSW-NB15"

            elif choice == "3":
                return "Both"

            else:
                print("Invalid choice. Please enter 1-3.")

    def select_shot_size(self):

        print("\nSELECT TRAINING SAMPLE SIZE")
        print("1. 5-shot")
        print("2. 10-shot")
        print("3. 20-shot")

        while True:

            choice = input("\nEnter sample size option (1-3): ")

            if choice == "1":
                return 5

            elif choice == "2":
                return 10

            elif choice == "3":
                return 20

            else:
                print("Invalid choice. Please enter 1-3.")


    def select_test_size(self):

        print("\nSELECT TEST SAMPLE SIZE")

        while True:

            try:

                test_size = int(
                    input(
                        "\nEnter number of test samples per class: "
                    )
                )

                if test_size > 0:
                    return test_size

                print("Test size must be greater than 0.")

            except ValueError:
                print("Please enter a whole number.")