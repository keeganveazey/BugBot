# Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, Xception, DenseNet201
from tensorflow.keras.models import Model
import os
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

import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperParameters
from tensorflow.keras import backend as K

# ----------- CONSTANTS ----------------

# define directory structure
TRAIN_DIR = "PROCESSED_DATA/TRAINING_DATA/TRAINING_AUGMENTED_DATA"
VALID_DIR = "PROCESSED_DATA/VALIDATION_DATA/"
TEST_DIR = "PROCESSED_DATA/TEST_DATA/"

# Image params
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NORMALIZE_FLAG = True
NO_FRILLS_DATAGEN = ImageDataGenerator()
NORM_DATAGEN = ImageDataGenerator(rescale=1. / 255)


def load_data(directory, shuffle_flag=True):
    '''
    Param:
        - directory - str,
        - shuffle_flag - boolean, introduces constrolled stochasticity
    '''
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


TRAIN_GENERATOR = load_data(TRAIN_DIR)
VAL_GENERATOR = load_data(VALID_DIR)
TEST_GENERATOR = load_data(TEST_DIR, shuffle_flag=False)
EVAL_VAL_GENERATOR = load_data(VALID_DIR, shuffle_flag=False)

hp = HyperParameters()


# HP MODEL GENERATOR FUNCTIONS TO BE USED IN BEST MODEL WITH TUNED PARAMS

def build_transfer_learning_MobileNetV2(hp):
    '''hp tuning function specifically for MobileNetV2'''

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # freeze the base model layers
    base_model.trainable = False

    dropout_rate = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    x = Dropout(dropout_rate)(x)

    output_layer = Dense(TRAIN_GENERATOR.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)

    # tune
    learning_rate = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
    batch_size = hp.Choice('batch_size', values=[16, 32, 64])
    epochs = hp.Choice('epochs', values=[5, 10, 15, 25, 50, 100])
    optimizer = Adam(learning_rate=learning_rate)

    # tune learning rate, batch size
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_transfer_learning_DenseNet201(hp):
    '''hp tuning function specifically for DenseNet201'''

    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # freeze the base model layers
    base_model.trainable = False

    dropout_rate = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout_rate)(x)

    output_layer = Dense(TRAIN_GENERATOR.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)

    # tune
    learning_rate = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
    batch_size = hp.Choice('batch_size', values=[16, 32, 64])
    epochs = hp.Choice('epochs', values=[5, 10, 15, 25, 50, 100])
    optimizer = Adam(learning_rate=learning_rate)

    # tune learning rate, batch size
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_transfer_learning_Xception(hp):
    '''hp tuning function specifically for Xception'''

    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # freeze the base model layers
    base_model.trainable = False

    dropout_rate = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    x = Dropout(dropout_rate)(x)

    output_layer = Dense(TRAIN_GENERATOR.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)

    # tune
    learning_rate = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
    batch_size = hp.Choice('batch_size', values=[16, 32, 64])
    epochs = hp.Choice('epochs', values=[5, 10, 15, 25, 50, 100])
    optimizer = Adam(learning_rate=learning_rate)

    # tune learning rate, batch size
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# BUILD BEST MODEL WITH TUNED PARAMS
def build_best_model_transfer_learning(algorithm, model_name, hp_function):
    '''
    Param: algorithm - string in ['bayes', 'random_search'],
            model_name - string version of model
    Use: Build the best model using the desired algorithm to get the best hyperparameter
    based on validation accuracy
    '''

    if algorithm == 'bayes':
        # Define the Bayesian tuner
        tuner = kt.BayesianOptimization(
            hp_function,
            objective='val_accuracy',  # tune by improving validation accuracy
            max_trials=20,  # num different hp combos to try
            executions_per_trial=1,  # run each model once
            directory='bayesian_tuning',
            project_name=f'bayes_hp_tuning_{model_name}'
        )
    elif algorithm == 'random_search':
        tuner = kt.RandomSearch(
            hp_function,  # Your model-building function
            objective='val_accuracy',  # Tune for validation accuracy
            max_trials=20,  # Number of different hyperparameter combinations to try
            executions_per_trial=1,  # Number of times to run each model
            directory='random_search_tuning',  # Directory to store tuning results
            project_name=f'random_search_hp_tuning_{model_name}'
        )
    else:
        raise ValueError("algorithm choice must be entered as string bayes or random_search ")

    # search hp combos
    tuner.search(TRAIN_GENERATOR, validation_data=VAL_GENERATOR, epochs=10)

    # get best hps
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # save them
    best_hps_dict = {'best_lr': best_hps.get('lr'),
                     'best_dropout': best_hps.get('dropout'),
                     'best_batch_size': best_hps.get('batch_size'),
                     'best_epochs': best_hps.get('epochs')}

    # make final model with the best drop out, learning rate and batch size
    best_model = tuner.hypermodel.build(best_hps)
    best_model_training_history = best_model.fit(TRAIN_GENERATOR, validation_data=VAL_GENERATOR,
                                                 epochs=best_hps.get('epochs'), batch_size=best_hps.get('batch_size'))

    return best_hps_dict, best_model, best_model_training_history


def evaluate_model_and_save(model, filename):
    '''
    Param: model - trained keras model object,
            filename - name of fiel to save (extension must b .h5)
    Use: Saves model to h5 file, returns TEST accuracy loss and test accuracy
    '''

    # evaluate on test data
    test_loss, test_acc = model.evaluate(TEST_GENERATOR)
    print(f"Test Accuracy: {test_acc:.4f}")

    # save to file
    model.save(filename)

    return test_loss, test_acc


def get_model_and_algorithm_combos_dict():
    # dict of top 3 models to tune from eda without tuning and associated hp functions
    models_to_tune = {'MobileNetV2': build_transfer_learning_MobileNetV2,
                      'Xception': build_transfer_learning_Xception,
                      'DenseNet201': build_transfer_learning_DenseNet201}

    hp_algorithms = ['bayes', 'random_search']

    # get all possible combinations to run
    combinations_dict = {}

    # iterate through 3 models
    for model_name, model_fn in models_to_tune.items():

        # go through hp algos
        for algo in hp_algorithms:
            combinations_dict[(model_name, algo)] = model_fn

    return combinations_dict


# MODEL EVALUATION Functions ------

def create_classification_report(y_true, y_pred, class_indices):
    '''
    Params:
        y_true: true class labels
        y_pred: predicted class labels
        class_indices: mapping of class labels to class names.

    Function:
        generates a classification report including precision, recall, F1-score, and accuracy for each class
        outputs the report as a DataFrame for further analysis

    Returns:
        classification report as a dataframe
    '''
    report = classification_report(y_true, y_pred, target_names=list(class_indices.keys()), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:")
    display(report_df)
    return report_df


def plot_confusion_matrix(y_true, y_pred, class_indices):
    '''
    Params:
        y_true: true class labels
        y_pred: predicted class labels
        class_indices: Mapping of class labels to class names

    Function:
        Plots a confusion matrix

    '''
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_indices.keys()))
    disp.plot(cmap=plt.cm.Blues, colorbar=True)
    plt.xticks(rotation=90)
    plt.title('Confusion Matrix')
    plt.show()


def plot_loss_curves(training_history):
    '''
    Params:
        training_history: object from model.fit() training history containing metrics accuracy and loss

    Function:
        plots training and validation accuracy and loss curves to evaluate model performance over epochs

    '''
    accuracy = training_history.history.get('accuracy', [])
    val_accuracy = training_history.history.get('val_accuracy', [])
    loss = training_history.history.get('loss', [])
    val_loss = training_history.history.get('val_loss', [])
    epochs = range(len(accuracy))

    # Plot training validation accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot training validation loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_roc_curve(y_true, y_pred_probs, class_indices):
    '''
    Params:
        y_true: true class labels
        y_pred_probs: predicted probabilities for each class
        class_indices: mapping of class labels to class names

    Function:
        plots the receiver operating characteristic (ROC) curve for each class and calculates the macro-averaged
        one vs rest (OvR) ROC AUC score

    Returns:
        macro averaged one vs rest ROC AUC score
    '''

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
    macro_roc_auc_ovo = roc_auc_score(y_true,
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
    plt.show()
    print(f"Macro-averaged One-vs-One ROC AUC score: {macro_roc_auc_ovo:.2f}")
    return macro_roc_auc_ovo


def evaluation_metrics(model, generator, training_history):
    '''
    Params:
        model: trained model
        generator: data generator for the evaluation set
        training_history: object from model.fit() training history containing metrics accuracy and loss

    Function:
        combines evaluation metrics (classification report, confusion matrix, training curves, and ROC curve)
        outputs key metrics: accuracy, precision, recall, and F1-score

    Returns:
        dictionary containing:
            accuracy: model accuracy on the evaluation data
            precision: macro averaged precision score
            recall: macro averaged recall score
            f1_score: macro averaged F1 score
            classification_report_df: classification report as a dataframe

    Outputs:
        confusion matrix plot
        loss plots
        macro average ROC curve plot
        macro averaged one vs rest ROC AUC score
    '''

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
    macro_roc_auc_ovo = plot_roc_curve(y_true, y_pred_probs, class_indices)

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


def main():
    print('--- MAKING MODEL x HP ALGORITHM COMBINATIONS ---')
    combos_to_run = get_model_and_algorithm_combos_dict()
    print('model-alg combinations:', combos_to_run)

    # there should be 6 model combo results
    results_dict = {}

    count = 0
    for key, funct in combos_to_run.items():

        start = time.time()

        print(f' --- STARTING MODEL COMBO {count + 1}/6 --- ')
        count = count + 1

        this_algorithm = key[1]  # which hp algo to use
        this_model_name = key[0]  # which model to use
        this_hp_function = funct  # which hp tuning func to use

        # getting results for this model and optimization alg
        best_hps_dict, best_model, best_model_training_history = build_best_model_transfer_learning(
            algorithm=this_algorithm,
            model_name=this_model_name,
            hp_function=this_hp_function)

        # get final model metrics on test set and save trained model to unique file

        save_model_filename = f"{this_model_name}_{this_algorithm}.h5"
        this_test_loss, this_test_acc = evaluate_model_and_save(best_model, filename=save_model_filename)
        print(f' ---- Completed saving: {this_model_name}_{this_algorithm}.h5 ---- ')

        # Print results
        print(
            f'Model: {this_model_name}, HP Algorithm: {this_algorithm}, test_loss: {this_test_loss}, test_acc: {this_test_acc}')

        # save validation results in dictionary for comparison yeet
        performance_model_metrics_dict_validation = evaluation_metrics(best_model, EVAL_VAL_GENERATOR, best_model_training_history)

        best_validation_accuracy = performance_model_metrics_dict_validation["accuracy"]
        best_validation_precision = performance_model_metrics_dict_validation["precision"]
        best_validation_recall = performance_model_metrics_dict_validation["recall"]
        best_validation_f1_score = performance_model_metrics_dict_validation["f1_score"]

        # save test results in dictionary for comparison
        performance_model_metrics_dict_test = evaluation_metrics(best_model, TEST_GENERATOR,
                                                                 best_model_training_history)

        test_precision = performance_model_metrics_dict_test["precision"]
        test_recall = performance_model_metrics_dict_test["recall"]
        test_f1_score = performance_model_metrics_dict_test["f1_score"]

        best_lr = best_hps_dict["best_lr"]
        best_dropout = best_hps_dict["best_dropout"]
        best_batch_size = best_hps_dict["best_batch_size"]
        best_epochs = best_hps_dict["best_epochs"]

        results_dict[f'{this_model_name}_{this_algorithm}'] = {
            "test_loss": this_test_loss,
            "test_accuracy": this_test_acc,
            "best_lr":best_lr,
            "best_dropout":best_dropout,
            "best_batch_size": best_batch_size,
            "best_epochs":best_epochs,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_precision": best_validation_precision,
            "best_validation_recall": best_validation_recall,
            "best_validation_f1_score": best_validation_f1_score,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1_score
        }

        print(f"Completed this model combo: {this_model_name}_{this_algorithm}\n\n")

        # clear keras session to free memory
        K.clear_session()
        end = time.time()
        print('Time to tune and evaluate this model:', round(end-start), ' seconds')

    # save final csv with all info!
    df_results = pd.DataFrame(results_dict)
    df_results.to_csv('final_tuned_models_results.csv')


if __name__ == "__main__":
    main()
