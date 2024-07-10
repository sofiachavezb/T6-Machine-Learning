from typing import List, Tuple
import pandas as pd
from .config import SEED, TRAIN_PERCENTAGE, VAL_PERCENTAGE, TEST_PERCENTAGE, DATASET_PATH

LABEL_COLUMN = 'diagnosis'

def load_data(  train_percentage:float = TRAIN_PERCENTAGE, 
                val_percentage:float = VAL_PERCENTAGE, 
                test_percentage:float = TEST_PERCENTAGE,
                no_validation:bool = False,
                verbose:bool = False,
                selected_columns:List[str] = None,
              )-> Tuple[  Tuple[pd.DataFrame, pd.DataFrame], 
                          Tuple[pd.DataFrame, pd.DataFrame],
                          Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load the data from the dataset and split it into training, validation, and test sets.

    Args:
    train_percentage: The percentage of the data to use for training.
    val_percentage: The percentage of the data to use for validation.
    test_percentage: The percentage of the data to use for testing.
    no_validation: If True, the validation set will not be used.
    verbose: If True, print the size of each set.
    selected_columns: The columns to use from the dataset. If None, all columns will be used.

    Returns:
    A tuple containing the training, validation, and test sets. 
    Each set is a tuple containing the features and the target variable.
    """
    assert train_percentage > 0, "The percentage of data to use for training must be greater than 0."
    assert test_percentage > 0, "The percentage of data to use for testing must be greater than 0."
    
    if no_validation:
      val_percentage = 0
      test_percentage = 1 - train_percentage
    else:
      assert val_percentage > 0, "The percentage of data to use for validation must be greater than 0."
    
    assert abs (train_percentage + val_percentage + test_percentage-1)<0.01, "The sum of the percentages must be equal to 1." 
    
    data = pd.read_csv(DATASET_PATH)
    data = data.drop(columns=['id'])
    if selected_columns:
      data = data[selected_columns+[LABEL_COLUMN]]

    if verbose:  
      print("Data size:           ", data.shape[0])
    data = data.sample(frac=1, random_state=SEED)
    train = data.sample(frac=train_percentage, random_state=SEED)
    train_x = train.drop(columns=[LABEL_COLUMN])
    train_y = train[LABEL_COLUMN]
    if verbose:  
      print("   Train size:       ", train.shape[0])
    test_val = data.drop(train.index)
    test = test_val.sample(frac=test_percentage/(test_percentage+val_percentage), random_state=SEED)
    test_x = test.drop(columns=[LABEL_COLUMN])
    test_y = test[LABEL_COLUMN]
    if verbose:  
      print("   Test size:        ", test.shape[0])
    val = test_val.drop(test.index)
    val_x = val.drop(columns=[LABEL_COLUMN])
    val_y = val[LABEL_COLUMN]
    if verbose:  
      print("   Validation size:  ", val.shape[0])
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def load_data_for_unsupervised(  selected_columns:List[str] = None,
                                verbose:bool = False,
                                )-> pd.DataFrame:
    """
    Load the data from the dataset and return it as a DataFrame.
    Useful for unsupervised learning.

    Args:
    selected_columns: The columns to use from the dataset. If None, all columns will be used.
    verbose: If True, print the size of the dataset.

    Returns:
    The dataset as a DataFrame.
    """
    data = pd.read_csv(DATASET_PATH)
    data = data.drop(columns=['id', LABEL_COLUMN])


    if selected_columns:
      data = data[selected_columns]
    
    if verbose:
      print(f"Data size: {data.shape[0]}")

    return data


