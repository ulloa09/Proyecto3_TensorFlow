import pandas as pd

# --- Purpose: split a DataFrame into Train/Test/Validation while maintaining temporal order ---

def split_dfs(data: pd.DataFrame, train:int, test:int, validation:int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a time-ordered DataFrame into three sets:
    Training, Test, and Validation.

    Args:
        data (pd.DataFrame): The complete, time-sorted DataFrame.
        train (int): Percentage for the training set (e.g., 60).
        test (int): Percentage for the test set (e.g., 20).
        validation (int): Percentage for the validation set (e.g., 20).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - train_df
            - test_df
            - validation_df
    """

    # --- Validate proportions ---
    # Ensure the three percentages cover 100% of the dataset.
    assert train + test + validation == 100, "The sum of train, test, and validation must be exactly 100."
    
    # --- Calculate cutoff indices ---
    # Define the boundaries of each block based on the percentages.
    n = len(data)
    train_cutoff = int(n * train / 100)
    test_cutoff = train_cutoff + int(n * test / 100)

    # --- Create subsets ---
    # Extract the partitions in order: Train (start -> train_cutoff),
    # Test (train_cutoff -> test_cutoff), and Validation (test_cutoff -> end).
    train_df = data.iloc[:train_cutoff]
    test_df = data.iloc[train_cutoff:test_cutoff]
    validation_df = data.iloc[test_cutoff:]

    # --- Return ---
    # Return the three partitions
    print(f"Data split successfully:")
    print(f"  Train:      {len(train_df)} rows ({train}%)")
    print(f"  Test:       {len(test_df)} rows ({test}%)")
    print(f"  Validation: {len(validation_df)} rows ({validation}%)")
    
    return train_df, test_df, validation_df