import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

class PostProcessor:

    @staticmethod
    def custom_scorer(scoring_metric: str) -> Any:
        scoring_dict = { 'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score, 'f1_score': f1_score}
        scoring_func = scoring_dict.get(scoring_metric)
        scorer = make_scorer(scoring_func)
        return scorer

    # defining a function to compute different metrics to check performance of a classification model built using sklearn
    @staticmethod
    def classification_metrics(model, predictors, target, prefix='val') ->dict:
        """
        Function to compute different metrics to check classification model performance

        model: classifier
        predictors: independent variables
        target: dependent variable
        """
        # predicting using the independent variables
        pred = model.predict(predictors)

        acc = accuracy_score(target, pred)  # to compute Accuracy
        recall = recall_score(target, pred)  # to compute Recall
        precision = precision_score(target, pred)  # to compute Precision
        f1 = f1_score(target, pred)  # to compute F1-score
        
        f1_score_minority = f1_score(target, pred, average=None)[1]
        recall_minority = recall_score(target, pred, average=None)[1]

        # creating a dataframe of metrics
        metrics = { f"{prefix}_Accuracy": acc, 
                    f"{prefix}_Recall": recall, 
                    f"{prefix}_Precision": precision, 
                    f"{prefix}_F1_score": f1,
                    f"{prefix}_recall_minority": recall_minority,
                    f"{prefix}_f1_score_minority": f1_score_minority
                }
        return metrics
       
    @staticmethod
    def plot_confusion_matrix(model, predictors, target):
        """
        To plot the confusion_matrix with percentages

        model: classifier
        predictors: independent variables
        target: dependent variable
        """
        # Predict the target values using the provided model and predictors
        y_pred = model.predict(predictors)

        # Compute the confusion matrix comparing the true target values with the predicted values
        cm = confusion_matrix(target, y_pred)

        # Create labels for each cell in the confusion matrix with both count and percentage
        labels = np.asarray(
            [
                ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
                for item in cm.flatten()
            ]
        ).reshape(2, 2)    # reshaping to a matrix

        # Set the figure size for the plot
        plt.figure(figsize=(6, 4))

        # Plot the confusion matrix as a heatmap with the labels
        sns.heatmap(cm, annot=labels, fmt="")

        # Add a label to the y-axis
        plt.ylabel("True label")

        # Add a label to the x-axis
        plt.xlabel("Predicted label")

    
    @staticmethod
    def plot_feature_importances(model, feature_names):
        # importance of features in the tree building
        #Model should be fit before we call this function for feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(8, 8))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.show()

    @staticmethod
    def visualize_tree(model, feature_names):
        #Function to visualize the tree
        # set the figure size for the plot
        plt.figure(figsize=(25, 25))

        # plotting the decision tree
        out = tree.plot_tree(
            model,                          # decision tree classifier model
            feature_names=feature_names,    # list of feature names in the dataset
            filled=True,                    # fill the nodes with colors based on class
            fontsize=9,                     # font size for the node text
            node_ids=False,                 # do not show the ID of each node
            class_names=['NoLoan','Loan'],               # whether or not to display class names
        )

        # add arrows to the decision tree splits if they are missing
        for o in out:
            arrow = o.arrow_patch
            if arrow is not None:
                arrow.set_edgecolor("black")    # set arrow color to black
                arrow.set_linewidth(1)          # set arrow linewidth to 1

        # displaying the plot
        plt.show()
    
    @staticmethod
    def show_tree_as_text(model, feature_names):
        # Text report showing the rules of a decision tree -
        print(tree.export_text(model, feature_names=feature_names, show_weights=True))