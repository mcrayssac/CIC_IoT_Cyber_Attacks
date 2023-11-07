# Import libraries
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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

def get_or_define_and_save_scaler(path_to_scaler, train_sets, X_columns):
    """
    Returns importing or defining scaler
    """
    try:
        scaler = joblib.load(path_to_scaler + 'scaler.joblib')
    except:
        scaler = MinMaxScaler()

        for train_set in tqdm(train_sets):
            scaler.fit(read_csv_file(train_set)[X_columns])

        joblib.dump(scaler, path_to_scaler + 'scaler.joblib')
    return scaler

def get_or_define_encoder(path_to_encoder):
    """
    Returns importing or defining encoder
    """
    try:
        encoder = joblib.load(path_to_encoder+'encoder.joblib')
    except:
        encoder = LabelEncoder()
    return encoder

def get_encoder(path_to_encoder, exception):
    """
    Returns importing encoder
    """
    try:
        encoder = joblib.load(path_to_encoder+'encoder.joblib')
    except:
        raise Exception(exception)
    return encoder

def get_col_in_csv(csv_name, model_repo, col, filter_col, filter_name, filter=True, verbose=True):
    """
    Returns specific col in a csv file
    """
    df = read_csv_file(csv_name, model_repo)
    if verbose:
        print(f"Dataframe length: {len(df)}.")

    if filter:
        df = df[df[filter_col] == filter_name]
        if verbose:
            print(f"After reduction dataframe length: {len(df)}.")

    df_col = df[col]
    del df
    return df_col

def get_or_define_performance_df(model_path, performance_path):
    """
    Returns importing or defining performance dataframe
    """
    try:
        # Load performance dataframe
        performance = read_csv_file(performance_path, model_path)
    except:
        # Define performance dataframe
        performance = pd.DataFrame(columns=['Model', 'Accuracy Training', 'Recall Training', 'Precision Training', 'F1 Training', 'Accuracy Testing', 'Recall Testing', 'Precision Testing', 'F1 Testing', 'FU_rate', 'FL_rate', 'FU', 'FL', 'Total rows'])
    return performance

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

def refactor_dataframe(sets, new_dictionary, new_file_path, y_column='label', drop_other=True):
    """
    Refactor dataset with new dictionnary.
    """
    i = 0
    res = pd.read_csv(DATASET_DIRECTORY + sets[0])

    new_y = [new_dictionary[k] for k in res[y_column]]
    res['Binary'] = new_y

    if drop_other:
        res = res[res['Binary'] != "Other"]
    
    for set in tqdm(sets[1:]):
        d = pd.read_csv(DATASET_DIRECTORY + set)

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

def build_model(model, model_name, train_sets, test_sets, path_to_datasets, performance_df, path_to_model, X_columns, y_column='label', z_column='Binary', filter_bool=False, filter_name='DoS', encoder=LabelEncoder(), scaler=MinMaxScaler(), confusionMatrix=False):
    """
    Build a model.
    """
    # Define variables for performance
    res_y_train = []
    res_y_pred_train = []

    for train_set in tqdm(train_sets):
        # Load data
        df = read_csv_file(train_set, path_to_datasets=path_to_datasets)

        if filter_bool:
            df = df[df[z_column] == filter_name]

        X_train = df[X_columns]
        y_train = df[y_column]
        # print(y_train[:5])
        # print(X_train[:5])

        # Scale and encode the train set
        X_train = scaler.transform(X_train)
        y_train_encoded = encoder.fit_transform(y_train)

        # Fit the model
        model.fit(X_train, y_train_encoded)
        y_pred_train = model.predict(X_train)

        # Add y to lists
        res_y_train += list(y_train_encoded)
        res_y_pred_train += list(y_pred_train)

        # Del variables
        del df, X_train, y_train, y_train_encoded, y_pred_train

    # Get the performance and save it
    accuracy_train, recall_train, precision_train, f1_train = accuracy_score(res_y_train, res_y_pred_train), recall_score(res_y_train, res_y_pred_train, average='macro'), precision_score(res_y_train, res_y_pred_train, average='macro'), f1_score(res_y_train, res_y_pred_train, average='macro')
    
    # Get the performance of the test set
    performance_df, encoder = test_model(model, model_name, test_sets, path_to_datasets, performance_df, accuracy_train, recall_train, precision_train, f1_train, X_columns, y_column=y_column, z_column=z_column, filter_bool=filter_bool, filter_name=filter_name, encoder=encoder, scaler=scaler, confusionMatrix=confusionMatrix)

    # Save model
    joblib.dump(model, f"{path_to_model}model_{model_name}.joblib")

    return performance_df, encoder

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

def optimize_hyperparameters(model, modelName, path_to_datasets, path_to_model, param_space, train_sets, X_columns, y_column='label', encoder=LabelEncoder(), scaler=MinMaxScaler(), n_splits=5, n_iter=10):
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
        X_train = scaler.transform(X_train)
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

def test_model(model, model_name, test_sets, path_to_datasets, performance_df, accuracy_train, recall_train, precision_train, f1_train, X_columns, y_column='label', z_column='Binary', filter_bool=False, filter_name='DoS', encoder=LabelEncoder(), scaler=MinMaxScaler(), confusionMatrix=False):
    """
    Test a model.
    """
    res_X_test = []
    res_y_test = []

    for test_set in tqdm(test_sets):
        # Load data
        df = read_csv_file(test_set, path_to_datasets=path_to_datasets)

        if filter_bool:
            df = df[df[z_column] == filter_name]

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

    return performance_df, encoder

def get_prediction_by_model(model, test_sets, path_to_datasets, X_columns, y_column='label', z_column='Binary', filter_bool=False, filter_name='DoS', scale=True, encode=True, scaler=MinMaxScaler(), encoder=LabelEncoder()):
    """
    Get the prediction of a model.
    """
    res_X_test = []
    res_y_test = []
    res_z_test = []

    for test_set in tqdm(test_sets):
        # Load data
        df = read_csv_file(test_set, path_to_datasets=path_to_datasets)
        # print(df.shape)

        if filter_bool:
            df = df[df[z_column] == filter_name]
        # print(df.shape)

        X_test = df[X_columns]
        # print(X_test[:5])
        # print(X_test.shape)
        # print('------------------------------------')
        y_test = df[y_column]
        z_test = df[z_column]

        # Scale and encode the test set
        if scale:
            X_test = scaler.transform(X_test)
        if encode:
            y_test_encoded = encoder.transform(y_test)
        else:
            y_test_encoded = y_test
        # print(X_test[:5])
        # print(y_test[:5])
        # print(z_test[:5])

        # Add y to lists
        res_X_test += list(X_test)
        res_y_test += list(y_test_encoded)
        res_z_test += list(z_test)

        # Del variables
        del df, X_test, y_test, z_test
    
    res_y_pred = model.predict(res_X_test)

    # Unscale
    if scale:
        res_X_test = scaler.inverse_transform(res_X_test)

    #TODO: Build Final Dataframe

    return res_X_test, res_y_test, res_y_pred, res_z_test

"""

Test prediction simplified

"""

def get_prediction_by_model_s(model, test_sets, path_to_datasets, X_columns, y_column='label', z_column='Binary', filter_bool=False, filter_name='DoS', scale=True, encode=True, scaler=MinMaxScaler(), encoder=LabelEncoder()):
    """
    Get the prediction of a model.
    """
    res_X_test = []
    res_y_test = []
    res_z_test = []

    for test_set in tqdm(test_sets):
        # Load data
        df = read_csv_file(test_set, path_to_datasets=path_to_datasets)
        # print(df.shape)

        if filter_bool:
            df = df[df[z_column] == filter_name]
        # print(df.shape)

        X_test = df[X_columns]
        # print(X_test[:5])
        # print(X_test.shape)
        # print('------------------------------------')
        y_test = df[y_column]
        z_test = df[z_column]

        # Scale and encode the test set
        if scale:
            X_test = scaler.transform(X_test)
        if encode:
            y_test_encoded = encoder.transform(y_test)
        else:
            y_test_encoded = y_test
        # print(X_test[:5])
        # print(y_test[:5])
        # print(z_test[:5])

        # Add y to lists
        res_X_test += list(X_test)
        res_y_test += list(y_test_encoded)
        res_z_test += list(z_test)

        # Del variables
        del df, X_test, y_test, z_test
    
    res_y_pred = model.predict(res_X_test)

    # Unscale
    if scale:
        res_X_test = scaler.inverse_transform(res_X_test)
        
    final_df = pd.DataFrame(res_X_test, columns=X_columns)

    del res_X_test

    return final_df, res_y_test, res_y_pred, res_z_test

"""

Test add cols

"""

def add_column_by_another_to_datasets(path_to_datasets, datasets, column_name, based_column_name, dictionnary):
    """
    Add column based on another to all datasets
    """
    for set in tqdm(datasets):
        # Read df
        d = pd.read_csv(path_to_datasets + set)

        # Inject the new col
        new_y = [dictionnary[k] for k in d[based_column_name]]
        d[column_name] = new_y

        # Save and Delete
        d.to_csv(path_to_datasets + set, index=False)        
        del d



"""

Test multi-filtered building model

"""

def multi_filter_df(df, filter_cols, filter_name):
    """
    Multi-filter dataframe
    """
    for i in range(0, len(filter_cols)):
        if filter_name[i]["type"] == "=":
            # print(f'Equal : {filter_cols[i]} & {filter_name[i]["name"]}')
            df = df[df[filter_cols[i]] == filter_name[i]["name"]]
        elif filter_name[i]["type"] == "!":
            # print(f'Not equal : {filter_cols[i]} & {filter_name[i]["name"]}')
            df = df[df[filter_cols[i]] != filter_name[i]["name"]]
        else:
            raise ("Dictionnary not built well ({'name': <elt>, 'type': '=' or '!'})")
    return df

def test_model_multifiltered(model, model_name, test_sets, path_to_datasets, performance_df, accuracy_train, recall_train, precision_train, f1_train, X_columns, y_column='label', filter_cols=[], filter_bool=False, filter_name=[], encoder=LabelEncoder(), scaler=MinMaxScaler(), confusionMatrix=False):
    """
    Test a model.
    """
    res_X_test = []
    res_y_test = []

    for test_set in tqdm(test_sets):
        # Load data
        df = read_csv_file(test_set, path_to_datasets=path_to_datasets)

        if filter_bool:
            df = multi_filter_df(df, filter_cols, filter_name)

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

    return performance_df, encoder

def build_model_multifiltered(model, model_name, train_sets, test_sets, path_to_datasets, performance_df, path_to_model, X_columns, y_column='label', filter_cols=[], filter_bool=False, filter_name=[], encoder=LabelEncoder(), scaler=MinMaxScaler(), confusionMatrix=False):
    """
    Build a model.
    """
    # Define variables for performance
    res_y_train = []
    res_y_pred_train = []

    for train_set in tqdm(train_sets):
        # Load data
        df = read_csv_file(train_set, path_to_datasets=path_to_datasets)
        # print(df.head(20))

        if filter_bool:
            df = multi_filter_df(df, filter_cols, filter_name)
        # print(df.head(20))

        X_train = df[X_columns]
        y_train = df[y_column]
        # print(y_train[:5])
        # print(X_train[:5])

        # Scale and encode the train set
        X_train = scaler.transform(X_train)
        y_train_encoded = encoder.fit_transform(y_train)

        # Fit the model
        model.fit(X_train, y_train_encoded)
        y_pred_train = model.predict(X_train)

        # Add y to lists
        res_y_train += list(y_train_encoded)
        res_y_pred_train += list(y_pred_train)

        # Del variables
        del df, X_train, y_train, y_train_encoded, y_pred_train

    # Get the performance and save it
    accuracy_train, recall_train, precision_train, f1_train = accuracy_score(res_y_train, res_y_pred_train), recall_score(res_y_train, res_y_pred_train, average='macro'), precision_score(res_y_train, res_y_pred_train, average='macro'), f1_score(res_y_train, res_y_pred_train, average='macro')
    
    # Get the performance of the test set
    performance_df, encoder = test_model_multifiltered(model, model_name, test_sets, path_to_datasets, performance_df, accuracy_train, recall_train, precision_train, f1_train, X_columns, y_column=y_column, filter_cols=filter_cols, filter_bool=filter_bool, filter_name=filter_name, encoder=encoder, scaler=scaler, confusionMatrix=confusionMatrix)

    # Save model
    joblib.dump(model, f"{path_to_model}model_{model_name}.joblib")

    return performance_df, encoder

def get_prediction_by_model_multifiltered(model, test_sets, path_to_datasets, X_columns, y_column='label', filter_cols=[], filter_bool=False, filter_name=[], scale=True, encode=True, scaler=MinMaxScaler(), encoder=LabelEncoder()):
    """
    Get the prediction of a model.
    """
    res_X_test = []
    res_y_test = []

    for test_set in tqdm(test_sets):
        # Load data
        df = read_csv_file(test_set, path_to_datasets=path_to_datasets)
        # print(df.shape)

        if filter_bool:
            df = multi_filter_df(df, filter_cols, filter_name)
        # print(df.shape)

        X_test = df[X_columns]
        # print(X_test[:5])
        y_test = df[y_column]

        # Scale and encode the test set
        if scale:
            X_test = scaler.transform(X_test)
        if encode:
            y_test_encoded = encoder.transform(y_test)
        else:
            y_test_encoded = y_test
        # print(X_test[:5])
        # print(y_test[:5])

        # Add y to lists
        res_X_test += list(X_test)
        res_y_test += list(y_test_encoded)

        # Del variables
        del df, X_test, y_test
    
    res_y_pred = model.predict(res_X_test)

    # Unscale
    if scale:
        res_X_test = scaler.inverse_transform(res_X_test)

    return res_X_test, res_y_test, res_y_pred

def calculate_and_plot_feature_importance(models, train_sets, X_columns, y_column, feature_names, path_to_datasets, filter_cols=[], filter_bool=False, filter_name=[], fitted_models=False, all_features=False, scaler=MinMaxScaler(), encoder=LabelEncoder()):
    """
    Calculate and plot average feature importance from multiple tree-based models.

    Parameters:
    - X: Features as a DataFrame or 2D array.
    - y: Labels as a 1D array or Series.
    - feature_names: List of feature names (optional).
    - model_names: List of model names (optional).

    Returns:
    - A DataFrame with feature names and average importance.
    - A bar plot showing the average feature importance.
    """

    # Dictionary to store feature importance for each model
    feature_importance = {model_name["Name"]: None for model_name in models}

    # Fit models and store feature importance
    for model in tqdm(models):
        if not fitted_models:
            for train_set in tqdm(train_sets):
                # Load data
                df = read_csv_file(train_set, path_to_datasets=path_to_datasets)

                if filter_bool:
                    df = multi_filter_df(df, filter_cols, filter_name)

                X_train = df[X_columns]
                y_train = df[y_column]

                # Scale and encode the train set
                X_train = scaler.transform(X_train)
                y_train_encoded = encoder.fit_transform(y_train)

                # Fit the model
                model['Model'].fit(X_train, y_train_encoded)

                # Del variables
                del df, X_train, y_train, y_train_encoded

        feature_importance[model['Name']] = model['Model'].feature_importances_/np.sum(model['Model'].feature_importances_)

    # Calculate average feature importance
    average_importance = np.mean(list(feature_importance.values()), axis=0)

    # Create a DataFrame with feature names and average importance
    # print(len(feature_names), len(average_importance))
    average_importance_df = pd.DataFrame({'Feature': feature_names, 'Average Importance': average_importance})
    if not all_features:
        average_importance_df = average_importance_df[average_importance_df['Average Importance'] > 0.01]
    average_importance_df = average_importance_df.sort_values(by=['Average Importance'], ascending=False)

    # Plot feature importance
    plt.figure(figsize=(20, 5))
    plt.bar(average_importance_df['Feature'], average_importance_df['Average Importance'])
    plt.ylabel('Average Importance')
    plt.xlabel('Feature Name')
    plt.xticks(rotation=90)
    plt.title('Average Feature Importance from Models')
    plt.show()

    return average_importance_df

def plot_feature_importance(df):
    plt.figure(figsize=(20, 5))
    plt.bar(df['Feature'], df['Average Importance'])
    plt.ylabel('Average Importance')
    plt.xlabel('Feature Name')
    plt.xticks(rotation=90)
    plt.title('Average Feature Importance from Models')
    plt.show()

def model_dict_refactor_with_load_model(simpleModelsDef, model_path):
    os_dir = [k for k in os.listdir(model_path) if k.endswith('.joblib') and k.startswith('model_')]

    for i in range(0, len(simpleModelsDef)):
        if not simpleModelsDef[i]['Name'] == 'MLP':
            temp_path = [s for s in os_dir if simpleModelsDef[i]['Name'] in s]
            if len(temp_path) == 1:
                print(model_path + temp_path[0])
                simpleModelsDef[i]['Model'] = joblib.load(model_path + temp_path[0])
            else:
                print(simpleModelsDef[i]['Name'], temp_path)
                print('More than one file in folder founded')
                simpleModelsDef[i]['Model'] = joblib.load(model_path + temp_path[0])
        else :
            simpleModelsDef.pop(i)
    
    return simpleModelsDef