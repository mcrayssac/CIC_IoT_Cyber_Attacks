# Import libraries
import os
import pandas as pd
from tqdm import tqdm

# Define variables
DATASET_DIRECTORY = ".\Files\\"

def x_columns(df, y_column='label'):
    """
    Returns a list of column names for the x variables.
    """
    return [col for col in df.columns if col != y_column]

def path_to_datasets():
    """
    Returns the path to the datasets folder.
    """
    return DATASET_DIRECTORY

def get_all_datasets_and_sort(path_to_datasets=path_to_datasets()):
    """
    Returns a list of all datasets in the datasets folder.
    """
    return [k for k in os.listdir(path_to_datasets) if k.endswith('.csv')]

def get_train_and_test_files(train_size=0.8, path_to_datasets=path_to_datasets()):
    """
    Returns a list of the train and test datasets.
    """
    df_sets = get_all_datasets_and_sort(path_to_datasets)
    return df_sets[:int(len(df_sets)*train_size)], df_sets[int(len(df_sets)*train_size):]

def read_csv_file(file_name, path_to_datasets=path_to_datasets()):
    """
    Returns a dataframe from a csv file.
    """
    return pd.read_csv(path_to_datasets + file_name)

def refactor_dataframe(sets, new_dictionary, scaler, new_file_path, X_columns, y_column='label', drop_other=True):
    """
    Returns .
    """
    i = 0
    res = pd.read_csv(DATASET_DIRECTORY + sets[0])

    res[X_columns] = scaler.transform(res[X_columns])
    new_y = [new_dictionary[k] for k in res[y_column]]
    res[y_column] = new_y

    if drop_other:
        res = res[res[y_column] != "Other"]
    
    for set in tqdm(sets[1:]):
        d = pd.read_csv(DATASET_DIRECTORY + set)

        d[X_columns] = scaler.transform(d[X_columns])
        new_y = [new_dictionary[k] for k in d[y_column]]
        d[y_column] = new_y

        if drop_other:
            res = res[res[y_column] != "Other"]

        if res.shape[0] > 300000:
            res.to_csv(new_file_path + "dataset" + str(i) + ".csv", index=False)
            i += 1
            res = d
        else:
            res = pd.concat([res, d], ignore_index=True)
        
        del d

    if drop_other:
            res = res[res[y_column] != "Other"]

    res.to_csv(new_file_path + "dataset.csv", index=False)
    return