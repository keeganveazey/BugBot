# FILE DESCRIPTION: -------------------------------------------------------

# This file includes common functions used by multiple files in our project directory
# including data loading, visualization, and evaluation of model training + results

# --------------------------------------------------------------------------



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score



# ----------- CONSTANTS ----------------

# define directory structure
TRAIN_DIR = "PROCESSED_DATA/TRAINING_DATA/TRAINING_AUGMENTED_DATA"
VALID_DIR = "PROCESSED_DATA/VALIDATION_DATA/"
TEST_DIR = "PROCESSED_DATA/TEST_DATA/"

# image params
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NORMALIZE_FLAG = True
NO_FRILLS_DATAGEN = ImageDataGenerator()
NORM_DATAGEN = ImageDataGenerator(rescale=1. / 255)


def load_data(directory, shuffle_flag=True):
    """
    Makes data generators for efficient data loading throughout the project

    Parameters:
        directory (str) - raw data directory
        shuffle_flag (boolean) - introduces controlled stochasticity in keras directory flow

    Returns:
        (function) keras data generator
    """

    # option to apply normalization to dataset if desired
    if NORMALIZE_FLAG == True:
        generator = NORM_DATAGEN.flow_from_directory(
            directory,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical', shuffle=shuffle_flag)
        return generator

    else:
        generator = NO_FRILLS_DATAGEN.flow_from_directory(
            directory,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical', shuffle=shuffle_flag)
        return generator


def create_classification_report(y_true, y_pred, class_indices):
    """
    Generates a classification report including precision, recall, F1-score, and
    accuracy for each class outputs the report as a DataFrame for further analysis

    Parameters:
        y_true (list) - true class labels
        y_pred (list) - predicted class labels
        class_indices (dict) - mapping of class labels to class names

    Returns:
        (Dataframe) classification report
    """

    report = classification_report(y_true, y_pred, target_names=list(class_indices.keys()), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:")
    display(report_df)

    return report_df


def plot_confusion_matrix(y_true, y_pred, class_indices):
    """
    Plots a confusion matrix

    Parameters:
        y_true (list) - true class labels
        y_pred (list) - predicted class labels
        class_indices (dict) - mapping of class labels to class names
    """

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_indices.keys()))
    disp.plot(cmap=plt.cm.Blues, colorbar=True)
    plt.xticks(rotation=90)
    plt.title('Confusion Matrix')
    plt.show(block=False)
    plt.close()


def plot_loss_curves(training_history):
    """
    Plots training and validation accuracy and loss curves to evaluate model performance over epochs

    Parameters:
        training_history (keras model fit object) - object from model.fit() training history containing metrics accuracy and loss

    """
    accuracy = training_history.history.get('accuracy', [])
    val_accuracy = training_history.history.get('val_accuracy', [])
    loss = training_history.history.get('loss', [])
    val_loss = training_history.history.get('val_loss', [])
    epochs =  range(len(training_history.history['accuracy']))



    # Plot training validation accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracy, color='orange', linestyle='-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, color='blue', linestyle='-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    y_min = 0
    y_max = 1
    step_size = 0.1

    plt.yticks(np.arange(y_min, y_max + step_size, step_size))

    plt.show(block=False)

    plt.close()

    # Plot training validation loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss,color='orange', linestyle='-', label='Training Loss')
    plt.plot(epochs, val_loss, color='blue', linestyle='-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    y_min = 0
    y_max = max(max(loss, default=0), max(val_loss, default=0))  # Get max loss dynamically

    step_size = 0.1

    plt.yticks(np.arange(y_min, y_max + step_size, step_size))  # Set custom tick

    plt.show(block=False)

    # plt.xticks(np.arange(len(accuracy)), np.arange(1, len(accuracy) + 1))

    plt.close()


def plot_roc_curve(y_true, y_pred_probs, class_indices):
    """
    Plots the receiver operating characteristic (ROC) curve for each class and
    calculates the macro-averaged one vs rest (OvR) ROC AUC score

    Parameters:
        y_true (list) - true class labels
        y_pred_probs (list) - predicted class labels
        class_indices (dict) - mapping of class labels to class names

    Returns:
        (float) macro averaged one vs rest ROC AUC score
    """

    # ROC AUC reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # ROC curve and AUC for multi-class classification
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_indices))))
    n_classes = y_true_bin.shape[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = cycle(
        ["aqua", "darkorange", "cornflowerblue", "red", "purple", "green", "gold", "deeppink", "brown", "gray", "navy"])

    for i, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_true_bin[:, i],
            y_pred_probs[:, i],
            name=f"Class {i}",
            color=color,
            ax=ax
        )

    # Macro average reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # Macro average ROC AUC score using OvR strategy
    macro_roc_auc_ovr = roc_auc_score(y_true,
                                      y_pred_probs,
                                      multi_class="ovr",
                                      average="macro")

    # ROC AUC plot reference: https://scikit-learn.org/1.1/auto_examples/model_selection/plot_roc.html
    # Plot ROC AUC curve
    ax.plot([0, 1], [0, 1], "k--", label="Chance Level (0.5)")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )
    ax.legend(loc="lower right")
    plt.show(block=False)
    plt.close()

    print(f"Macro-averaged One-vs-Rest ROC AUC score: {macro_roc_auc_ovr:.2f}")
    return macro_roc_auc_ovr


def evaluation_metrics(model, generator, training_history):
    """
    Combines evaluation metrics (classification report, confusion matrix, training curves, and ROC curve)
    and outputs key metrics (accuracy, precision, recall, and F1-score)

    Parameters:
        model (keras model) - trained model
        generator (function) - data generator for the evaluation set
        training_history (keras model fit object) - object from model.fit() training history containing metrics accuracy and loss

    Returns:
        (dict) with following keys:
            accuracy: model accuracy on the evaluation data
            precision: macro averaged precision score
            recall: macro averaged recall score
            f1_score: macro averaged F1 score
            classification_report_df: classification report as a dataframe
        (png) confusion matrix plot
        (png) loss plots
        (png) macro average ROC curve plot
        (png) macro averaged one vs rest ROC AUC score
    """

    # Get true labels
    y_true = generator.classes

    # Generate predictions
    y_pred_probs = model.predict(generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    class_indices = generator.class_indices

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Classification report
    report_df = create_classification_report(y_true, y_pred, class_indices)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_indices)

    # Loss curves
    plot_loss_curves(training_history)

    # ROC AUC OvR score
    macro_roc_auc_ovr = plot_roc_curve(y_true, y_pred_probs, class_indices)

    # Get metrics from the classification report
    precision = round(report_df.loc["macro avg", "precision"], 3)
    recall = round(report_df.loc["macro avg", "recall"], 3)
    f1_score = round(report_df.loc["macro avg", "f1-score"], 3)

    # Print key metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

    return {
        "accuracy": round(accuracy, 3),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "classification_report_df": report_df
    }

