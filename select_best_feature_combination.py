import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from itertools import combinations  # Required for generating feature combinations
from sklearn.metrics import make_scorer

def plotAccuracyF1score(resAccuracy, resF1score,average_importance_df):
  born_min=1
  born_max=len(resAccuracy)
  print(born_max)
  plt.figure(figsize=(12,8))
  miny=np.min([np.min(resAccuracy),np.min(resF1score)])
  maxy=np.max([np.max(resAccuracy),np.max(resF1score)])
  plt.plot(range(0,born_max), resAccuracy, label='Accuracy',marker='s')
  plt.plot(range(0,born_max), resF1score, label='F1 Score',marker='+')
  plt.legend(fontsize=14)
  # plt.xticks(range(born_max),fontsize=16)
  plt.xticks(ticks=range(0,born_max), labels=list(average_importance_df.Feature[:born_max].values), rotation=90, ha='right',fontsize=16)
  plt.yticks(fontsize=16)
  plt.grid('on')
  plt.xlabel('Number of Features',fontsize=16)
  plt.ylabel('Cumulated Importance Score',fontsize=16)
  plt.ylim([miny-0.01*miny,maxy+0.01*maxy ])

def plot_grouped_bar(df, title="Metrics by Model"):
    # Number of models and metrics
    num_models = len(df)
    num_metrics = len(df.columns) - 1  # Exclude the 'Model' column

    # Create positions for bars
    bar_width = 0.15
    index = np.arange(num_models)

    # Define colors for each metric
    colors = ['b', 'g', 'r', 'c']

    # Set the figure size
    fig=plt.figure(figsize=(10, 6))

    # Create a bar for each metric
    for i in range(num_metrics):
        plt.bar(index + i * bar_width, df.iloc[:, i + 1], bar_width, label=df.columns[i + 1], color=colors[i])
    lt=list(df.columns)
    lt.remove('Model')
    print(lt)
    # Customize the plot
    plt.xlabel('Models')
    plt.ylabel('Metric Values')
    plt.title(title)
    plt.xticks(index + bar_width * (num_metrics / 2), df['Model'],rotation=90,fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title="Metric")
    minval=df[lt].min().min()-0.05
    maxval=df[lt].max().max()+0.01
    # print([, df[lt].max().max()+0.05])
    # plt.ylim([df[lt].min().min(), df[lt].max().max()+0.05])
    plt.ylim([minval,maxval])

    # Show the plot
    plt.show()
    return plt.gcf


def plot_combined_metrics(results):
    # Create a grouped bar chart for classification
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")
    metric_order = ['Model',"Accuracy", "Precision", "Recall", "F1 Score"]
    title = "Classification Metrics by Model"
            # Create a grouped bar chart
    plot_grouped_bar(results[metric_order], "Metrics by Model")
    plt.show()

def perform_randomized_search2(X_train, X_test, y_train, y_test,labels,features,led=1):
    results = []
    best_params = []
    models = {
        # "MLPClassifier": MLPClassifier(max_iter=1000, random_state=42),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "ExtraTreesClassifier": ExtraTreesClassifier(random_state=42),
        "XGBoostClassifier": XGBClassifier(random_state=42),
        "LightGBMClassifier":LGBMClassifier(random_state=42,verbose=-100),
        "CatBoostClassifier": CatBoostClassifier(random_state=42,verbose=0),
    }

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    for model_name, model in models.items():
            print(f"Training {model_name}...")
            cv_Model = cross_validate(model, X_train, y_train, cv=5, scoring='accuracy',return_estimator=True)
            accuracy_cv = cv_Model["test_score"].mean()
            print(f"Accuracy with CV: {accuracy_cv}")
            y_pred = cv_Model["estimator"][0].predict(X_test)
            if led==1:
              if model_name=="DecisionTreeClassifier":
                plt.figure(figsize=(32,32))
                tree.plot_tree(cv_Model["estimator"][0],
                                  feature_names=features,
                                  class_names=labels,
                                  filled=True, fontsize=9)
                plt.show()

              # viz = dtreeviz(cv_Model["estimator"][0], X_train, y_train,
              #                 # target_name="target",
              #                 feature_names=features,
              #                 class_names=labels,)

              # viz
              # text_representation = tree.export_text(cv_Model["estimator"][0])
              # print(text_representation)
            conf_matrix = confusion_matrix(y_test, y_pred,normalize='true')
            conf_matrix2 = confusion_matrix(y_test, y_pred)
# m = confusion_matrix(decoded_actual, decoded_predicted, labels=labels,normalize='true')
            cm_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
            if led==1:
              if (len(np.unique(y_train))==2):
                plt.figure(figsize=(6, 6))
                sns.heatmap(cm_df, annot=True, cmap="Blues", fmt=".6f",xticklabels=labels, yticklabels=labels)
              else:
                plt.figure(figsize=(10, 10))
              # plt.figure(figsize=(16, 16))
                sns.heatmap(cm_df, annot=True, cmap="Blues", fmt=".3f",xticklabels=labels, yticklabels=labels)
              plt.title("Confusion Matrix for: " + model_name,fontsize=16)
              plt.xlabel("Predicted",fontsize=16)
              plt.ylabel("Actual",fontsize=16)
              plt.xticks(fontsize=16)
              plt.yticks(fontsize=16)
              plt.show()

            accuracy=accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            result = {
            "Model": model_name,
            # "Best Parameters": best_model.get_params(),
        }
            result.update({
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Confusion Matrix": conf_matrix2,
                "Confusion Matrix Normalize": conf_matrix,
                # "Classification Report": class_report,
            })


            results.append(result)

    results_df = pd.DataFrame(results)
    # print(results_df.head(10))
    # print(list(results_df.columns.values))
    # results_df.set_index(['Model'], inplace=True)

    sorted_results_df = results_df.sort_values(by=[ "Accuracy","F1 Score"], ascending=[False, False])
    # print(sorted_results_df.columns )
    if led==1:
      plot_combined_metrics(sorted_results_df)
    return sorted_results_df


def plotResultsvsNumberofFeatures(average_importance_df,labels,X_train, X_test, y_train, y_test, born_min,born_max,ledto):
  # born_min=1
  # born_max=2
  # nFeatures
  # average_importance_df=average_importance_df[average_importance_df['Feature']!='A_k12']
  resAccuracy=np.zeros(born_max-born_min)
  resF1score=np.zeros(born_max-born_min)
  # labels=['Class 0','Class 1']
  for itr in range(born_min,born_max):
    print('itr:', itr)

    features=list(average_importance_df.Feature[:itr].values)
    # if 'A_k12' in features:
    #   features.remove('A_k12')
    #   continue
    print(features)
    Xtemp_train=X_train[features]
    Xtemp_test=X_test[features]
  # X_trainBin, X_testBin, y_trainBin, y_testBin
  # Xtemp_trainBin,Xtemp_testBin, y_trainBin, y_testBin
    # Xtemp.shape
    results_df=perform_randomized_search2(Xtemp_train,Xtemp_test, y_train, y_test,labels,features,led=ledto)
    results_df.set_index(['Model'], inplace=True)
    print(results_df.head(10))
    resAccuracy[itr-born_min]=results_df['Accuracy'][0]
    resF1score[itr-born_min]=results_df['F1 Score'][0]
  return resAccuracy, resF1score


def select_best_feature_combination(model, X_train, X_test, y_train, y_test, feature_list, cv=5, scoring='accuracy',top_n=10,startk=1):
    """
    Selects the best combination of features by evaluating all possible combinations with a model.

    Parameters:
        - model: The model (classifier or regressor) to evaluate the feature combinations.
        - X_train: The feature matrix of the training set.
        - X_test: The feature matrix of the test set.
        - y_train: The target vector of the training set.
        - y_test: The target vector of the test set.
        - feature_list: List of feature indices or feature names.
        - cv: Number of folds for cross-validation (default: 5).
        - scoring: The scoring metric to use for evaluation (default: 'accuracy').

    Returns:
        - best_features: The indices of the selected features.
    """

    best_score = -np.inf
    # best_scoreF = -np.inf
    best_features = None
    top_combinations = []
    scorer = make_scorer(scoring)
    scorer = make_scorer(accuracy_score)  # Default to accuracy if scoring is invalid
    if scoring in ['accuracy', 'precision', 'recall', 'f1']:
        scorer = scoring
    # Generate all possible combinations of features
    for r in range(startk, len(feature_list) + 1):
        print('r:',r)
        for feature_combination in combinations(feature_list, r):
            # Select features
            feature_combination=list(feature_combination)
            X_train_subset = X_train[:, list(feature_combination)] if isinstance(X_train, np.ndarray) else X_train[feature_combination]
            X_test_subset = X_test[:, feature_combination] if isinstance(X_test, np.ndarray) else X_test[feature_combination]
            cv_Model = cross_validate(model,  X_train_subset, y_train, cv=5, scoring=scorer,return_estimator=True)
            avg_score = cv_Model["test_score"].mean()
            # Evaluate model with cross-validation
            # scores = cross_val_score(model, X_train_subset, y_train, cv=cv, scoring=scorer)
            # avg_score = np.mean(scores)


            y_pred = np.array(cv_Model["estimator"][0].predict(X_test_subset)).flatten()
            # conf_matrix = confusion_matrix(y_test, y_pred)
            # conf_matrix2 = confusion_matrix(y_test, y_pred, normalize='true')
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            # Add combination and score to top combinations
            top_combinations.append((feature_combination, avg_score,accuracy,precision,recall,f1))
            # Update best score and features if current combination is better
            if (accuracy > best_score):
                print(feature_combination)
                best_score = accuracy
                print( best_score )
                best_features = feature_combination




    # Sort top combinations by score and select top n
    top_combinations.sort(key=lambda x: x[2], reverse=True)
    top_combinations = top_combinations[:top_n]

    # Create DataFrame from top combinations
    top_combinations_df = pd.DataFrame(top_combinations, columns=['Feature Combination', 'Training Accuracy','Accuracy','Precision','Recall','F1 Score'])
    top_combinations_df = top_combinations_df.sort_values(by=["Accuracy", "F1 Score"], ascending=[False, False])
    # # Test selected features on test set and compute score
    # top_combinations_df['Test Score'] = top_combinations_df['Feature Combination'].apply(
    #     lambda combination: model.fit(X_train[:, combination], y_train).score(X_test[:, combination], y_test))


    # Test selected features on test set and compute score
    X_train_best = X_train[:, best_features] if isinstance(X_train, np.ndarray) else X_train[best_features]
    X_test_best = X_test[:, best_features] if isinstance(X_test, np.ndarray) else X_test[best_features]
    test_score = model.fit(X_train_best, y_train).score(X_test_best, y_test)
    print(f"Test score with selected features: {test_score}")

    y_pred = np.array(model.predict(X_test_best)).flatten()
    conf_matrix = confusion_matrix(y_test, y_pred)
            # conf_matrix2 = confusion_matrix(y_test, y_pred, normalize='true')
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print("Accuracy", accuracy)
    print("Precision", precision)
    print("Recall", recall)
    print("F1 Score", f1)
    print("Confusion Matrix", conf_matrix)
    return best_features,top_combinations_df
