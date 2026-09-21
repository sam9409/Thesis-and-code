import pandas as pd


class trainingSplit:

    def __init__(
        self,
        data: pd.DataFrame,
        shot_size: int = 5,
        test_size: int = 100,
        random_seed: int = 42
    ):
        self.data = data.copy()
        self.shot_size = shot_size
        self.test_size = test_size
        self.random_seed = random_seed

        self.train_df = None
        self.test_df = None

        # Keep original row identity so overlap can be checked
        if "original_index" not in self.data.columns:
            self.data["original_index"] = self.data.index


    def create_split(self):
        """
        Create a few-shot training set and evaluation set
        for each encoded class.
        """

        if "label" not in self.data.columns:
            raise ValueError(
                "The DataFrame must contain an encoded 'label' column."
            )

        train_slices = []
        test_slices = []

        for label_id, group in self.data.groupby("label"):

            # Shuffle examples within each class
            group = group.sample(
                frac=1,
                random_state=self.random_seed
            )

            available_samples = len(group)

            # Need enough examples for training
            if available_samples <= self.shot_size:

                print(
                    f"Warning: class {label_id} only has "
                    f"{available_samples} examples."
                )

                continue

            # Exactly N training examples
            train_g = group.iloc[
                :self.shot_size
            ].copy()

            # Up to test_size evaluation examples
            available_test_samples = (
                available_samples - self.shot_size
            )

            current_test_size = min(
                self.test_size,
                available_test_samples
            )

            test_g = group.iloc[
                self.shot_size:
                self.shot_size + current_test_size
            ].copy()

            train_slices.append(train_g)
            test_slices.append(test_g)


        if not train_slices:
            raise ValueError(
                "No classes had enough samples "
                "to create a training set."
            )

        if not test_slices:
            raise ValueError(
                "No test samples were created."
            )


        # Combine all classes
        self.train_df = pd.concat(
            train_slices,
            ignore_index=True
        )

        self.test_df = pd.concat(
            test_slices,
            ignore_index=True
        )


        # Shuffle completed training dataset
        self.train_df = self.train_df.sample(
            frac=1,
            random_state=self.random_seed
        ).reset_index(drop=True)


        # Shuffle completed test dataset
        self.test_df = self.test_df.sample(
            frac=1,
            random_state=self.random_seed
        ).reset_index(drop=True)


        return self.train_df, self.test_df


    def print_split_info(self):
        """
        Print information about the generated split.
        """

        self._check_split_created()

        print(
            f"\n{self.shot_size}-shot training set: "
            f"{len(self.train_df)} rows"
        )

        print(
            f"Evaluation set: "
            f"{len(self.test_df)} rows"
        )


        print("\nTraining examples per class:")

        if "Label" in self.train_df.columns:
            print(
                self.train_df["Label"].value_counts()
            )
        else:
            print(
                self.train_df["label"].value_counts()
            )


        print("\nTest examples per class:")

        if "Label" in self.test_df.columns:
            print(
                self.test_df["Label"].value_counts()
            )
        else:
            print(
                self.test_df["label"].value_counts()
            )


    def validate_split(self):
        """
        Check that every training class contains
        exactly shot_size examples.
        """

        self._check_split_created()

        counts = (
            self.train_df["label"]
            .value_counts()
            .sort_index()
        )

        valid = True

        print("\nValidating training split:")

        for label_id, count in counts.items():

            if count != self.shot_size:

                print(
                    f"Warning: class {label_id} contains "
                    f"{count} examples instead of "
                    f"{self.shot_size}."
                )

                valid = False

            else:

                print(
                    f"Class {label_id}: "
                    f"{count} examples - OK"
                )


        if valid:
            print(
                "\nAll training classes have the "
                "correct number of examples."
            )

        return valid


    def check_overlap(self):
        """
        Check whether any original rows occur in both
        the training and test datasets.
        """

        self._check_split_created()

        train_indices = set(
            self.train_df["original_index"]
        )

        test_indices = set(
            self.test_df["original_index"]
        )

        overlap = (
            train_indices
            .intersection(test_indices)
        )


        if overlap:

            print(
                f"Warning: {len(overlap)} rows appear "
                "in both training and testing."
            )

        else:

            print(
                "\nNo overlap between "
                "training and test sets."
            )


        return overlap


    def get_train_data(self):
        """
        Return the training DataFrame.
        """

        self._check_split_created()

        return self.train_df


    def get_test_data(self):
        """
        Return the test DataFrame.
        """

        self._check_split_created()

        return self.test_df


    def _check_split_created(self):
        """
        Internal helper method that checks whether
        create_split() has already been called.
        """

        if self.train_df is None or self.test_df is None:

            raise ValueError(
                "Training and test data have not "
                "been created. Run create_split() first."
            )