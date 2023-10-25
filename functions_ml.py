# Import libraries
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import joblib
import seaborn as sns
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

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
    Refactor dataset with new dictionnary.
    """
    i = 0
    res = pd.read_csv(DATASET_DIRECTORY + sets[0])

    res[X_columns] = scaler.transform(res[X_columns])
    new_y = [new_dictionary[k] for k in res[y_column]]
    res['Binary'] = new_y

    if drop_other:
        res = res[res['Binary'] != "Other"]
    
    for set in tqdm(sets[1:]):
        d = pd.read_csv(DATASET_DIRECTORY + set)

        d[X_columns] = scaler.transform(d[X_columns])
        new_y = [new_dictionary[k] for k in d[y_column]]
        d['Binary'] = new_y

        if drop_other:
            res = res[res['Binary'] != "Other"]

        if res.shape[0] > 300000:
            res.to_csv(new_file_path + "dataset" + str(i) + ".csv", index=False)
            i += 1
            res = d
        else:
            res = pd.concat([res, d], ignore_index=True)
        
        del d

    if drop_other:
            res = res[res['Binary'] != "Other"]

    res.to_csv(new_file_path + "dataset.csv", index=False)

    del res
    return

def count_label(datasets, file_path=path_to_datasets()):
    """
    Returns all datasets counts group by classes.
    """
    results = []

    for dataset in datasets:
        df = pd.read_csv(file_path + dataset)
        class_counts = df['Binary'].value_counts()
        results.append(class_counts)

    class_counts_combined = pd.concat(results)

    return class_counts_combined.groupby(class_counts_combined.index).sum()

def plot_bar_chart(dataframe, title, xLabel, yLabel, figX, figY, log_scale=False):
    """
    Plot bar chart of a dataframe.
    """
    plt.figure(figsize=(figX, figY))

    plt.bar(dataframe.index, dataframe.values)
    if log_scale:
        plt.yscale('log')  # Utilisation une échelle logarithmique sur l'axe des y
    plt.xlabel(xLabel)
    if log_scale:
        plt.ylabel(f'{yLabel} (échelle logarithmique)')
    else:
        plt.ylabel(f'{yLabel} (échelle non logarithmique)')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def calculate_false_upper_and_false_lower(y_true, y_pred, confusionMatrix=False):
    """
    Returns the number of false upper and false lower.
    """
    # Normalize in one dimension
    if y_pred.ndim != 1:
        tableau_1d = y_pred.flatten()
        y_pred = np.ravel(tableau_1d)

    # Mapping labels to index
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_index = {label: index for index, label in enumerate(labels)}
    
    fu_dict = {}
    fl_dict = {}
    i = 1

    for label in tqdm(labels):
        # Mapping labels to index
        label_index = label_to_index[label]

        # Create confusion matrix for the current label
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Extract the false upper and false lower
        fl = sum(cm[i:, label_index])
        temp = cm[label_index, :]
        fu = sum(temp[i:])
        
        fu_dict[label] = fu
        fl_dict[label] = fl

        i += 1

    if confusionMatrix:
        # Plot the confusion matrix as a heatmap
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        # Normalize the confusion matrix
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        plt.figure(figsize=(30, 22))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', annot_kws={"size": 16})
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    # Create false upper and false lower dataframes
    data_fu = {
        'Category': list(fu_dict.keys()),
        'Count': list(fu_dict.values())
    }
    df_fu = pd.DataFrame(data_fu)
    df_fu.sort_values(by=['Count'], inplace=True, ascending=False)
    fu = df_fu['Count'].sum()

    data_fl = {
        'Category': list(fl_dict.keys()),
        'Count': list(fl_dict.values())
    }
    df_fl = pd.DataFrame(data_fl)
    df_fl.sort_values(by=['Count'], inplace=True, ascending=False)
    fl = df_fl['Count'].sum()

    return fu, fl

def get_test_performance(model, X_test, y_test, confusionMatrix=False):
    """
    Returns the performance of a model.
    """
    y_pred = model.predict(X_test)

    fu, fl = calculate_false_upper_and_false_lower(y_test, y_pred, confusionMatrix=confusionMatrix)

    return accuracy_score(y_test, y_pred), recall_score(y_test, y_pred, average='macro'), precision_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro'), fu, fl

def build_model(model, model_name, train_sets, test_sets, path_to_datasets, performance_df, path_to_model, X_columns, y_column='label', z_column='Binary', encoder=LabelEncoder(), scaler=StandardScaler(), confusionMatrix=False):
    """
    Build a model.
    """
    # Define variables for performance
    res_y_train = []
    res_y_pred_train = []

    res_X_test = []
    res_y_test = []

    for train_set in tqdm(train_sets):
        # Load data
        df = read_csv_file(train_set, path_to_datasets=path_to_datasets)
        X_train = df[X_columns]
        y_train = df[z_column]
        # print(y_train[:5])
        # print(X_train[:5])

        # Scale and encode the train set
        X_train = scaler.fit_transform(X_train)
        y_train_encoded = encoder.fit_transform(y_train)

        # Fit the model
        model.fit(X_train, y_train_encoded)
        y_pred_train = model.predict(X_train)

        # Add y to lists
        res_y_train += list(y_train_encoded)
        res_y_pred_train += list(y_pred_train)

        # Del variables
        del df, X_train, y_train, y_train_encoded, y_pred_train

    for test_set in tqdm(test_sets):
        # Load data
        df = read_csv_file(test_set, path_to_datasets=path_to_datasets)
        X_test = df[X_columns]
        y_test = df[z_column]

        # Scale and encode the test set
        X_test = scaler.transform(X_test)
        y_test_encoded = encoder.transform(y_test)

        # Add y to lists
        res_X_test += list(X_test)
        res_y_test += list(y_test_encoded)

        # Del variables
        del df, X_test, y_test, y_test_encoded

    # Get the performance and save it
    accuracy_train, recall_train, precision_train, f1_train = accuracy_score(res_y_train, res_y_pred_train), recall_score(res_y_train, res_y_pred_train, average='macro'), precision_score(res_y_train, res_y_pred_train, average='macro'), f1_score(res_y_train, res_y_pred_train, average='macro')
    accuracy_testing, recall_testing, precision_testing, f1_testing, fu, fl = get_test_performance(model, res_X_test, res_y_test, confusionMatrix=confusionMatrix)
    performance_df.loc[model_name] = [model_name, accuracy_train, recall_train, precision_train, f1_train, accuracy_testing, recall_testing, precision_testing, f1_testing, fu/len(res_y_test), fl/len(res_y_test), fu, fl, len(res_y_test)]

    # Save model
    joblib.dump(model, f"{path_to_model}model_{model_name}.joblib")

    return performance_df, scaler, encoder

def get_all_sets_3_sets(datasets, X_columns, y_column='label', z_column='Binary', path_to_datasets=path_to_datasets()):
    """
    Returns concatenated dataframe of all datasets.
    """
    i=1
    for dataset in datasets:
        if i==1:
            df = read_csv_file(dataset, path_to_datasets=path_to_datasets)
            X = df[X_columns]
            y = df[y_column]
            z = df[z_column]
            i+=1
        else:
            df = read_csv_file(dataset, path_to_datasets=path_to_datasets)
            X = pd.concat([X, df[X_columns]])
            y = pd.concat([y, df[y_column]])
            z = pd.concat([z, df[z_column]])

    return X, y, z

def optimize_hyperparameters(model, modelName, path_to_datasets, path_to_model, param_space, train_sets, X_columns, y_column='label', encoder=LabelEncoder(), scaler=StandardScaler(), n_splits=5, n_iter=10):
    """
    Optimize hyperparameters of a model.
    """

    # Optimiser les hyperparamètres du modèle
    # Utilisation de la validation croisée stratifiée
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialisation de la recherche Bayesienne
    bayes_search = BayesSearchCV(
        model,
        param_space,
        n_iter=n_iter,  # Nombre d'itérations de la recherche Bayesienne
        cv=cv,
        n_jobs=-1,  # Utilisation de tous les cœurs disponibles
        verbose=0,  # Affichage des détails de la recherche
        random_state=42
    )

    for train_set in tqdm(train_sets):
        # Load data
        df = read_csv_file(train_set, path_to_datasets=path_to_datasets)
        X_train = df[X_columns]
        y_train = df[y_column]

        # Scale and encode the train set
        X_train = scaler.fit_transform(X_train)
        y_train_encoded = encoder.fit_transform(y_train)

        # Lancer la recherche Bayesienne
        bayes_search.fit(X_train, y_train_encoded)

        # Del variables
        del df, X_train, y_train, y_train_encoded

    # Retourner le modèle avec les meilleurs hyperparamètres
    best_rf_model = bayes_search.best_estimator_

    # Save model
    joblib.dump(model, f"{path_to_model}tuning_model_{modelName}.joblib")

    return best_rf_model

def test_model(model, model_name, test_sets, path_to_datasets, performance_df, accuracy_train, recall_train, precision_train, f1_train, X_columns, y_column='label', encoder=LabelEncoder(), scaler=StandardScaler(), confusionMatrix=False):
    """
    Test a model.
    """
    res_X_test = []
    res_y_test = []

    for test_set in tqdm(test_sets):
        # Load data
        df = read_csv_file(test_set, path_to_datasets=path_to_datasets)
        X_test = df[X_columns]
        y_test = df[y_column]

        # Scale and encode the test set
        X_test = scaler.transform(X_test)
        y_test_encoded = encoder.transform(y_test)

        # Add y to lists
        res_X_test += list(X_test)
        res_y_test += list(y_test_encoded)

        # Del variables
        del df, X_test, y_test, y_test_encoded

    # Get the performance and save it
    accuracy_testing, recall_testing, precision_testing, f1_testing, fu, fl = get_test_performance(model, res_X_test, res_y_test, confusionMatrix=confusionMatrix)
    performance_df.loc[model_name] = [model_name, accuracy_train, recall_train, precision_train, f1_train, accuracy_testing, recall_testing, precision_testing, f1_testing, fu/len(res_y_test), fl/len(res_y_test), fu, fl, len(res_y_test)]

    return performance_df, scaler, encoder

def get_prediction_by_model(model, test_sets, path_to_datasets, X_columns, y_column='label', z_column='Binary', scaler=StandardScaler(), encoder=LabelEncoder()):
    """
    Get the prediction of a model.
    """
    res_X_test = []
    res_y_test_encoded = []
    res_z_test = []

    for test_set in tqdm(test_sets):
        # Load data
        df = read_csv_file(test_set, path_to_datasets=path_to_datasets)
        X_test = df[X_columns]
        y_test = df[y_column]
        z_test = df[z_column]

        # Scale and encode the test set
        X_test = scaler.fit_transform(X_test)
        y_test_encoded = encoder.fit_transform(y_test)

        # Add y to lists
        res_X_test += list(X_test)
        res_y_test_encoded += list(y_test_encoded)
        res_z_test += list(z_test)

        # Del variables
        del df, X_test, y_test, y_test_encoded, z_test
    
    res_y_pred = model.predict(res_X_test)

    #TODO: Build Final Dataframe

    return scaler, encoder, res_X_test, res_y_test_encoded, res_y_pred, res_z_test


