# Import libraries
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
import joblib
import seaborn as sns
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

FONT_SIZE = 16
plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=FONT_SIZE) 

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

def get_or_define_and_save_scaler(path_to_scaler, train_sets, X_columns, file_path=path_to_datasets()):
    """
    Returns importing or defining scaler
    """
    try:
        scaler = joblib.load(path_to_scaler + 'scaler.joblib')
    except:
        scaler = MinMaxScaler()

        for train_set in tqdm(train_sets):
            scaler.fit(read_csv_file(train_set, file_path)[X_columns])

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

def plot_bar_chart(dataframe, title, xLabel, yLabel, figX, figY, save_directory, log_scale=False):
    """
    Plot bar chart of a dataframe.
    """
    plt.figure(figsize=(figX, figY))

    plt.bar(dataframe.index, dataframe.values)
    if log_scale:
        plt.yscale('log') 
    plt.xlabel(xLabel)
    if log_scale:
        plt.ylabel(f'{yLabel} (logarithmic scale)')
    else:
        plt.ylabel(f'{yLabel} (non-logarithmic scale)')

    # Add value to each bar of the bar chart
    for index, value in enumerate(dataframe.values):
        plt.text(index, value, value, ha='center', va='bottom')

    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(save_directory + title + '.png')
    plt.show()

def plot_pie_chart(dataframe, title, figX, figY, save_directory):
    """
    Plot pie chart of a dataframe.
    """
    plt.figure(figsize=(figX, figY))
    plt.title(title)
    plt.pie(dataframe.values, labels=dataframe.index, autopct='%1.1f%%', startangle=90, explode=[0.1]*len(dataframe))
    plt.tight_layout()
    plt.grid()
    plt.savefig(save_directory + title + '.png')
    plt.show()

# Select features with cumulative importance > 0.95 and correlation < 0.80
def select_features_by_importance(Labels, X, save_repo, threshold_percentage=0.95):
    """
    Select features by importance.
    """

    # Trie les indices des fonctionnalités par ordre décroissant d'importance
    sorted_feature_indices = np.argsort(X)[::-1] # argsort renvoie les indices qui trieraient le tableau et [::-1] inverse l'ordre

    # Calcule l'importance cumulée
    cumulative_importance = np.cumsum(X[sorted_feature_indices]) # cumsum calcule la somme cumulée et [sorted_feature_indices] réordonne les valeurs dans le même ordre que les indices
    print(cumulative_importance)

    # Sélectionne les indices des fonctionnalités à conserver
    selected_feature_indices = sorted_feature_indices[cumulative_importance <= threshold_percentage]

    # Sélectionne les colonnes correspondantes dans X
    X_selected = [Labels[i] for i in selected_feature_indices.tolist()]

    # Plot l'importance cumulée
    plt.figure(figsize=(10, 6))
    #plt.plot(cumulative_importance)
    plt.plot(Labels[sorted_feature_indices], cumulative_importance)
    plt.title('Cumulative feature importance')
    plt.xlabel('Name of features')
    plt.ylabel('Cumulative importance')
    plt.axvline(x=len(X_selected) - 0.5, color='r', linestyle='--')
    plt.grid()
    plt.xticks(rotation=90)
    plt.axhline(y=threshold_percentage, color='r', linestyle='-')
    plt.text(0, threshold_percentage + 0.01, f'Threshold of {threshold_percentage * 100}%', color='red')
    plt.tight_layout()
    plt.savefig(save_repo+'Cumulative feature importance.png')
    plt.show()

    return X_selected

def plot_correlation_matrix(df, path_to_save, title, figsize=(20, 10), lower=False, labels=True):
    """
    Plot correlation matrix of a dataframe.
    """
    corr = df.corr()
    if lower:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr = corr.mask(mask)
    plt.figure(figsize=figsize)
    plt.title(title)
    if labels:
        sns.heatmap(corr, annot=True, fmt='.2f', linewidths=1, cmap='Blues')
    else:
        sns.heatmap(corr, annot=False, fmt='.2f', linewidths=1, cmap='Blues')
    plt.tight_layout()
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.show()

def plot_pairplot(df, path_to_save, hue, title, figsize=(20, 10)):
    """
    Plot pairplot of a dataframe.
    """
    plt.figure(figsize=figsize)
    sns.pairplot(df, hue=hue)
    plt.title(title)
    plt.grid()
    plt.savefig(path_to_save)
    plt.show()

def plot_boxplot(df, path_to_save, title, figsize=(20, 10), log_scale=True):
    """
    Plot boxplot of a dataframe.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=df)
    plt.title(title)
    if log_scale:
        plt.yscale('log')
    plt.xticks(rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.show()

def plot_kde_plot(df, X_columns, path_to_save, hue='label', figsize=(25, 10), yscaleLog=False):
    """
    Plot kde plot of a dataframe.
    """
    # Define figure and axes according to the number of columns
    rows = int(np.ceil(len(X_columns)/3))
    fig, ax = plt.subplots(rows, 3, figsize=figsize)

    for i in range(0, len(X_columns)):
        plt.subplot(rows, 3, i+1)
        sns.kdeplot(x = df[X_columns[i]], color = "lightblue",hue=hue,data= df)
        plt.grid()
        if yscaleLog:
            plt.yscale('log')
        # ax[i//3, i%3].grid('on')
    if yscaleLog:
        plt.suptitle('Lineplot of the features (logarithmic scale)')
    else:
        plt.suptitle('Lineplot of the features (non-logarithmic scale)')
    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()

def plot_lineplot(df, X_columns, path_to_save, hue='label', figsize=(20, 10), yscaleLog=False):
    """
    Plot lineplot of a dataframe.
    """
    # Define figure and axes according to the number of columns
    rows = int(np.ceil(len(X_columns)/3))
    fig, ax = plt.subplots(rows, 3, figsize=figsize)

    for i in range(0, len(X_columns)):
        plt.subplot(rows, 3, i+1)
        sns.lineplot(data=df,x=df.index, y=df[X_columns[i]],hue=hue)
        # plt.legend(bbox_to_anchor = (1, 1), loc = "best");
        plt.xlabel('Index')
        plt.ylabel(X_columns[i])
        plt.grid()
        if yscaleLog:
            plt.yscale('log')
    if yscaleLog:
        plt.suptitle('Lineplot of the features (logarithmic scale)')
    else:
        plt.suptitle('Lineplot of the features (non-logarithmic scale)')
    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()

def calculate_false_upper_and_false_lower(y_true, y_pred, confusionMatrix=False, saving=False, pathToSave=".\\", figsize=(30, 20), labeled=False, encoder=LabelEncoder()):
    """
    Returns the number of false upper and false lower.
    """
    # Normalize in one dimension
    if y_pred.ndim != 1:
        tableau_1d = y_pred.flatten()
        y_pred = np.ravel(tableau_1d)

    # Mapping labels to index
    if labeled:
        y_labeled_true = encoder.inverse_transform(y_true)
        y_labeled_pred = encoder.inverse_transform(y_pred)
        labels = np.unique(np.concatenate([y_labeled_true, y_labeled_pred]))
    else:
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
        # Decode labels
        labels = encoder.inverse_transform(labels)
        # Put true labels decoded on all axis
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        # Plot the heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', annot_kws={"size": 16})
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        if saving:
            plt.tight_layout()
            plt.savefig(pathToSave+'Confusion Matrix.png', bbox_inches='tight')
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

def get_test_performance(model, X_test, y_test, confusionMatrix=False, saving=False, pathToSave=".\\", figsize=(30, 20), encoder=LabelEncoder(), labeled=False):
    """
    Returns the performance of a model.
    """
    y_pred = model.predict(X_test)

    fu, fl = calculate_false_upper_and_false_lower(y_test, y_pred, confusionMatrix=confusionMatrix, saving=saving, pathToSave=pathToSave, figsize=figsize, labeled=labeled, encoder=encoder)

    return accuracy_score(y_test, y_pred), recall_score(y_test, y_pred, average='macro'), precision_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro'), fu, fl

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

def test_model_multifiltered(model, model_name, test_sets, path_to_datasets, performance_df, accuracy_train, recall_train, precision_train, f1_train, X_columns, y_column='label', filter_cols=[], filter_bool=False, filter_name=[], encoder=LabelEncoder(), scaler=MinMaxScaler(), confusionMatrix=False, saving=False, pathToSave=".\\", figsize=(30, 20), labeled=False):
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
    accuracy_testing, recall_testing, precision_testing, f1_testing, fu, fl = get_test_performance(model, res_X_test, res_y_test, confusionMatrix=confusionMatrix, saving=saving, pathToSave=pathToSave, figsize=figsize, labeled=labeled, encoder=encoder)
    performance_df.loc[model_name] = [model_name, accuracy_train, recall_train, precision_train, f1_train, accuracy_testing, recall_testing, precision_testing, f1_testing, fu/len(res_y_test), fl/len(res_y_test), fu, fl, len(res_y_test)]

    return performance_df, encoder

def build_model_multifiltered(model, model_name, train_sets, test_sets, path_to_datasets, performance_df, path_to_model, X_columns, y_column='label', filter_cols=[], filter_bool=False, filter_name=[], encoder=LabelEncoder(), scaler=MinMaxScaler(), confusionMatrix=False, saving=False, pathToSave=".\\", figsize=(30, 20), labeled=False):
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
    performance_df, encoder = test_model_multifiltered(model, model_name, test_sets, path_to_datasets, performance_df, accuracy_train, recall_train, precision_train, f1_train, X_columns, y_column=y_column, filter_cols=filter_cols, filter_bool=filter_bool, filter_name=filter_name, encoder=encoder, scaler=scaler, confusionMatrix=confusionMatrix, saving=saving, pathToSave=pathToSave, figsize=figsize, labeled=labeled)

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

def calculate_and_plot_feature_importance(models, feature_names, save_repo, all_features=False, figsize=(20, 5)):
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
        try:
            feature_importance[model['Name']] = model['Model'].feature_importances_/np.sum(model['Model'].feature_importances_)
        except:
            raise Exception(f"Model {model['Name']} is not fitted.")

    # Calculate average feature importance
    average_importance = np.mean(list(feature_importance.values()), axis=0)

    # Create a DataFrame with feature names and average importance
    # print(len(feature_names), len(average_importance))
    average_importance_df = pd.DataFrame({'Feature': feature_names, 'Average Importance': average_importance})
    if not all_features:
        average_importance_df = average_importance_df[average_importance_df['Average Importance'] > 0.01]
    average_importance_df = average_importance_df.sort_values(by=['Average Importance'], ascending=False)

    # Plot feature importance
    plt.figure(figsize=figsize)
    bars = plt.bar(average_importance_df['Feature'][0:15], average_importance_df['Average Importance'][0:15])
    plt.ylabel('Average Importance')
    plt.xlabel('Feature Name')
    plt.xticks(rotation=90)
    plt.grid()
    plt.title('Average Feature Importance from Models')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), 
                 verticalalignment='bottom',  # position the text to start at the bar top
                 ha='center')  # align the text horizontally centered on the bar
    plt.tight_layout()
    plt.savefig(save_repo + 'Average Feature Importance from Models.png', bbox_inches='tight')
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

def remove_features_with_correlation_and_feature_importance(df, df_average_importance, save_repo, figsize=(20, 5), threshold=0.80):
    """
    Remove features with correlation > 0.80 and feature importance < 0.01
    """
    # print(df_average_importance.head(10))

    # Get correlation matrix
    corr_matrix = df.corr().abs()

    # # Plot correlation matrix
    # plt.figure(figsize=(20, 20))
    # sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=1, cmap='Blues')
    # plt.show()

    new_X = []
    # For each column in df_average_importance, add if not in new_X there is a column with it correlation > 0.80
    for index, row in df_average_importance.iterrows():
        if len(new_X) == 0:
            new_X.append(row['Feature'])
        elif df[row['Feature']].nunique() == 1:
            continue
        else:
            e = True
            for feature in new_X:
                # if row['Feature'] == 'AVG':
                #     print(feature, row['Feature'], corr_matrix[feature][row['Feature']])
                if corr_matrix[feature][row['Feature']] >= threshold:
                    e = False
                    break
            if e:
                new_X.append(row['Feature'])
    
    # Create new feature importance dataframe with features with correlation < 0.80
    new_df = pd.DataFrame()
    for index, row in df_average_importance.iterrows():
        if row['Feature'] in new_X:
            new_df = new_df.append(row)

    # Re-calculating average importance to get sum of all features = 1
    new_df['Average Importance'] = new_df['Average Importance'] / new_df['Average Importance'].sum()
    new_df = new_df.reset_index(drop=True)

    # Plot feature importance
    plt.figure(figsize=figsize)
    bars = plt.bar(new_df['Feature'][0:15], new_df['Average Importance'][0:15])
    plt.ylabel('Average Importance')
    plt.xlabel('Feature Name')
    plt.xticks(rotation=90)
    plt.grid()
    plt.title('Average Feature Importance from Models')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), 
                 verticalalignment='bottom',  # position the text to start at the bar top
                 ha='center')  # align the text horizontally centered on the bar
    plt.tight_layout()
    plt.savefig(save_repo + 'Average Feature Importance from Models.png', bbox_inches='tight')
    plt.show()

    return new_df

# Plot performance table
def plot_performance_table(performance_table, path_to_save, title, figsize=(25, 10)):
    """
    Plot performance table.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=performance_table.values, colLabels=performance_table.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.show()

# Search best hyperparameters of a model with bayesian optimization
def bayesian_optimization(model_type, model, param_space, path_to_datasets, datasets, X_columns, y_column, scaler, encoder, n_iter=10, cv=5, random_state=42):
    """
    Search best hyperparameters of a model with bayesian optimization.
    """
    # Initialisation de la recherche Bayesienne
    bayes_search = BayesSearchCV(
        model,
        param_space,
        n_iter=n_iter,  # Nombre d'itérations de la recherche Bayesienne
        cv=cv,
        n_jobs=-1,  # Utilisation de tous les cœurs disponibles
        verbose=0,  # Affichage des détails de la recherche
        random_state=random_state
    )

    # Lancer la recherche Bayesienne
    for dataset in tqdm(datasets):
        # Load data
        df = read_csv_file(dataset, path_to_datasets=path_to_datasets)
        X_train = df[X_columns]
        y_train = df[y_column]

        # Scale and encode the train set
        X_train = scaler.transform(X_train)
        y_train_encoded = encoder.fit_transform(y_train)

        # Lancer la recherche Bayesienne
        bayes_search.fit(X_train, y_train_encoded)

        # Del variables
        del df, X_train, y_train, y_train_encoded

    if model_type == 'DT':
        return DecisionTreeClassifier(**bayes_search.best_params_)
    else:
        raise Exception('Model type not defined.')