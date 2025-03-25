# FILE DESCRIPTION: -------------------------------------------------------

# This file runs our customized densenet transfer learning model
# with hyperparameter optimization tuning. It saves the best hyperparameter
# combinations results to a dictionary, and provides functionality to
# build the best densenet model with the best hyperparameteres

# --------------------------------------------------------------------------

# IMPORTS

# make parameters easily changeable from command line for file run
import argparse



# libraries for building convolutional neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model

# libraries for tuning models
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperParameters
from tensorflow.keras import backend as K
from keras.callbacks import History, EarlyStopping

# utility libraries
import os
import sys
import time
import itertools

# our commonly used functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from global_bug_bot_functions import  *

# data set constants
TRAIN_GENERATOR = load_data(TRAIN_DIR)
VAL_GENERATOR = load_data(VALID_DIR)
TEST_GENERATOR = load_data(TEST_DIR, shuffle_flag=False)
EVAL_VAL_GENERATOR = load_data(VALID_DIR, shuffle_flag=False)

# default argument values
INPUT_MAXIMUM_EPOCHS = 5
INPUT_PATIENCE = 5
INPUT_MIN_DELTA = 0.1
INPUT_EXECUTIONS_PER_TRIAL = 1
INPUT_MAX_TRIALS = 5
PREFIX_FOR_OUTPUT_FILE = 'prefix'


def parse_arguments():
    """
    Uses argparse to get command line arguments passed in from the user for: epochs, patience, min_delta, executions_per_trial
    and max_trials. Descriptions for each argument are also provided by the help function of argparse.

    Returns: argument parser with the above-mentioned specified fields
    """

    parser = argparse.ArgumentParser(description="Run model tuners with specified hyperparameters.")

    parser.add_argument("--epochs", type=int, required=True, help="Max number of epochs to train the model.")
    parser.add_argument("--patience", type=int, required=True, help="Early stopping patience.")
    parser.add_argument("--min_delta", type=float, required=True, help="Minimum change to qualify as an improvement.")
    parser.add_argument("--executions_per_trial", type=int, required=True, help="Number of executions per trial.")
    parser.add_argument("--max_trials", type=int, required=False, help="Maximum number of trials.")

    return parser.parse_args()


def init_user_arguments():
    """
    Initializes tuner script arguments
    """

    global INPUT_MAXIMUM_EPOCHS
    global INPUT_PATIENCE
    global INPUT_MIN_DELTA
    global INPUT_EXECUTIONS_PER_TRIAL
    global INPUT_MAX_TRIALS
    global PREFIX_FOR_OUTPUT_FILE

    args = parse_arguments()

    print(f"Running model tuner with the following settings:")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print(f"Min Delta: {args.min_delta}")
    print(f"Executions per Trial: {args.executions_per_trial}")
    print(f"Max Trials: {args.max_trials}")

    INPUT_MAXIMUM_EPOCHS = args.epochs
    INPUT_PATIENCE = args.patience
    INPUT_MIN_DELTA = args.min_delta
    INPUT_EXECUTIONS_PER_TRIAL = args.executions_per_trial
    INPUT_MAX_TRIALS = args.max_trials

    PREFIX_FOR_OUTPUT_FILE = f'ep{INPUT_MAXIMUM_EPOCHS}_pat{INPUT_PATIENCE}_del{INPUT_MIN_DELTA}_ex{INPUT_EXECUTIONS_PER_TRIAL}_maxtr{INPUT_MAX_TRIALS}'

    return



class TunableDenseNet201(kt.HyperModel):

    global INPUT_MAXIMUM_EPOCHS
    global INPUT_PATIENCE
    global INPUT_MIN_DELTA
    global INPUT_EXECUTIONS_PER_TRIAL
    global INPUT_MAX_TRIALS
    global PREFIX_FOR_OUTPUT_FILE

    def build(self, hp):
        """
        Builds DenseNet201 customized transfer learning CNN model for hp tuning

        Parameters:
             hp (keras_tuner.HyperParameters) - Keras Tuner's HyperParameters class instance used to
             tune hyperparameters dropout, batch size, and learning rate

        Returns:
            (keras model) - compiled keras model
        """

        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # freeze the base model layers
        base_model.trainable = False

        # tune dropout rate
        dropout_rate = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)

        # tune batch_size
        batch_size = hp.Choice("batch_size", [16, 32, 64])  # Now explicitly included


        # global pooling and dropout layers included similar to jute pest architecture
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = Dropout(dropout_rate)(x)

        # final softmax probabilistic classification layer
        output_layer = Dense(TRAIN_GENERATOR.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output_layer)

        # tune learning rate
        learning_rate = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
        optimizer = Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.batch_size = batch_size

        return model

    def fit(self, hp, model, *args, **kwargs):
        """
        Fits the customized DenseNet201 model with a tunable batch size and parameters
        defined in the model build process.

        Parameters:
            hp (keras_tuner.HyperParameters) - instance of Keras Tuner's HyperParameters class
            model (tensorflow.keras.Model) -  keras model built using the build method with hp tuning
            *args - positional arguments to be passed to model.fit(), including training data.
            **kwargs - additional keyword arguments for model.fit() including epochs, validation data,
            and callbacks

        Returns:
        (tensorflow.keras.callbacks.History) - history object from model.fit()
    """
        return model.fit(*args, batch_size=model.batch_size,**kwargs)


# BUILD BEST MODEL WITH TUNED PARAMS
def build_best_model_transfer_learning(algorithm, model_name, hp_class = TunableDenseNet201()):
    """
    Builds the best model using the desired algorithm to get the best hyperparameter
    based on validation accuracy

    Parameters:
        algorithm (str) - optimization algorithm denoted by 'bayes' or 'random_search'
        model_name (str) - string version of model
        hp_class (function) - tunable model function, default is TunableDenseNet201() since this is the densenet script

    Returns:
        best_hps_dict (dict) -
        best_model (keras compiled model) - compiled model with best hps
        best_model_training_history (keras model history) - model history with best hps
    """

    #note: number of hp combinations is high which is why these two algorithms were utilized in addition to
    # literature review:
    # 3 options for batch size, 4 options for learning rate, 4 options for dropout level = 48 combinations
    # nCr = 11 C 3 = 165 possible unique hp combinations

    if algorithm == 'bayes':
        # Define the Bayesian tuner
        tuner = kt.BayesianOptimization(
            hp_class,
            objective='val_accuracy',  # tune by improving validation accuracy
            max_trials=INPUT_MAX_TRIALS,  # num different hp combos to try
            executions_per_trial=INPUT_EXECUTIONS_PER_TRIAL,  # run each model once
            directory='bayesian_tuning',
            project_name=f'bayes_hp_tuning_{model_name}'
        )
    elif algorithm == 'random_search':
        tuner = kt.RandomSearch(
            hp_class,  # Your model-building function
            objective='val_accuracy',  # Tune for validation accuracy
            max_trials=INPUT_MAX_TRIALS,  # Number of different hyperparameter combinations to try
            executions_per_trial=INPUT_EXECUTIONS_PER_TRIAL,  # Number of times to run each model hp combo for robustness -- try 3?
            directory='random_search_tuning',  # Directory to store tuning results
            project_name=f'random_search_hp_tuning_{model_name}'
        )
    else:
        raise ValueError("algorithm choice must be entered as string bayes or random_search ")

    # implement regularization via early stopping instead of tuning epochs iteratively
    stop_early = EarlyStopping(monitor='val_loss',min_delta=INPUT_MIN_DELTA, patience = INPUT_PATIENCE)

    # search hp combo, go for maximum of max epochs
    tuner.search(TRAIN_GENERATOR, validation_data=VAL_GENERATOR, epochs = INPUT_MAXIMUM_EPOCHS, callbacks=[stop_early])

    # Get the best trial
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

    # Extract validation accuracy history and find the epoch with the best validation accuracy
    best_epochs = INPUT_MAXIMUM_EPOCHS  # Default value in case of an issue
    if best_trial and 'val_accuracy' in best_trial.metrics.metrics:
        val_acc_history = best_trial.metrics.get_history('val_accuracy')  # Get list of val_acc per epoch
        if val_acc_history:
            best_epochs = val_acc_history.index(max(val_acc_history)) + 1  # Convert to 1-based index

    # get best hps
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # save them
    best_hps_dict = {'best_lr': best_hps.get('lr'),
                     'best_dropout': best_hps.get('dropout'),
                     'best_batch_size': best_hps.get('batch_size')}

    # make final model with the best drop out, learning rate, batch size and best epochs
    best_model = tuner.hypermodel.build(best_hps)

    best_hps_dict["best_epochs"] = best_epochs
    best_model_training_history = best_model.fit(TRAIN_GENERATOR, validation_data=VAL_GENERATOR,
                                                 epochs=best_epochs, batch_size=best_hps.get('batch_size'))


    print('Completed build_best_model_transfer_learning function call.', 'Best HPs: \n', best_hps_dict)
    return best_hps_dict, best_model, best_model_training_history


def evaluate_model_and_save(model, filename):
    """
    Saves model to .keras file

    Parameters:
        model (keras model) - trained keras model object
        filename - name of file without extension

    Returns:
        (float) test loss
        (float) test accuracy

    """

    # evaluate on test data
    test_loss, test_acc = model.evaluate(TEST_GENERATOR)
    print(f"Test Accuracy: {test_acc:.4f}")

    # save to file
    # model.save(filename)
    model.save(f'{filename}.keras')

    return test_loss, test_acc


def get_model_and_algorithm_combos_dict():
    """
    Create dictionary to save densenet model results for each optimization techniques

    Returns:
        (dict) - dictionary with model and optimization combinations to try
    """

    # model to tune and associated hp class
    models_to_tune ={'DenseNet201': TunableDenseNet201()}

    hp_algorithms = ['bayes', 'random_search']

    # get all possible combinations to run
    combinations_dict = {}

    # iterate through models
    for model_name, model_class in models_to_tune.items():

        # go through hp algos
        for algo in hp_algorithms:
            combinations_dict[(model_name, algo)] = model_class

    return combinations_dict


def main():

    # 1) initialize the epochs, patience, min_delta, executions_per_trial and max_trials from user
    init_user_arguments()

    print('--- MAKING MODEL x HP ALGORITHM COMBINATIONS ---')

    combos_to_run = get_model_and_algorithm_combos_dict()
    print('model-alg combinations:', combos_to_run)

    # there will be results for each of the optimization methods, saved in this initialized dictionary
    results_dict = {}

    count = 0 # used to keep track of file run progress

    # for each optimization method for this transfer model,
    # tune and get the best hps
    for key, funct in combos_to_run.items():

        start = time.time()

        print(f' --- STARTING MODEL COMBO {count + 1}/2 --- ')

        count = count + 1

        this_algorithm = key[1]  # which hp algo to use
        this_model_name = key[0]  # which model to use
        this_hp_class = funct  # which hp tuning func to use

        # getting results for this model and optimization algo
        best_hps_dict, best_model, best_model_training_history = build_best_model_transfer_learning(
            algorithm=this_algorithm,
            model_name=this_model_name,
            hp_class=this_hp_class)

        # get final model metrics on test set and save trained model to unique file
        save_model_filename = f"{PREFIX_FOR_OUTPUT_FILE}_{this_model_name}_{this_algorithm}"
        this_test_loss, this_test_acc = evaluate_model_and_save(best_model, filename=save_model_filename)
        print(f' ---- Completed saving: {this_model_name}_{this_algorithm}.keras ---- ')

        # Display results
        print(
            f'Model: {this_model_name}, HP Algorithm: {this_algorithm}, test_loss: {this_test_loss}, test_acc: {this_test_acc}')

        # save validation results in dictionary for comparison
        performance_model_metrics_dict_validation = evaluation_metrics(best_model, EVAL_VAL_GENERATOR, best_model_training_history)

        print('Completed call to evaluation_metrics for validation data.')

        # getting the result metrics
        best_validation_accuracy = performance_model_metrics_dict_validation["accuracy"]
        best_validation_precision = performance_model_metrics_dict_validation["precision"]
        best_validation_recall = performance_model_metrics_dict_validation["recall"]
        best_validation_f1_score = performance_model_metrics_dict_validation["f1_score"]

        # save test results in dictionary for comparison
        performance_model_metrics_dict_test = evaluation_metrics(best_model, TEST_GENERATOR,
                                                                 best_model_training_history)

        print('Completed call to evaluation_metrics for test data.')

        test_precision = performance_model_metrics_dict_test["precision"]
        test_recall = performance_model_metrics_dict_test["recall"]
        test_f1_score = performance_model_metrics_dict_test["f1_score"]

        best_lr = best_hps_dict["best_lr"]
        best_dropout = best_hps_dict["best_dropout"]
        best_batch_size = best_hps_dict["best_batch_size"]
        best_epochs = best_hps_dict["best_epochs"]

        # save the metric results and model info to a dictionary
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
        end = time.time()
        print('Time to tune and evaluate this model:', round(end-start), ' seconds')

    # save final csv with all info!
    df_results = pd.DataFrame(results_dict)
    df_results.to_csv(f'{PREFIX_FOR_OUTPUT_FILE}_from_tuning_script_final_tuned_models_results_DenseNet201.csv')

    print('Completed tuning and saving final model.')


if __name__ == "__main__":

    # Directions: Open a terminal and navigate to the bugbot directory

    # Run the script 4 times with the following (you can copy and paste) :

    # run 1)  python "Model Tuning Scripts/DenseNet201_run_tuner_script.py" --epochs 30 --patience 5 --min_delta 0.001 --executions_per_trial 1 --max_trials 20
    # run 2) python "Model Tuning Scripts/DenseNet201_run_tuner_script.py" --epochs 30 --patience 5 --min_delta 0.001 --executions_per_trial 3 --max_trials 60
    # run 3) python "Model Tuning Scripts/DenseNet201_run_tuner_script.py" --epochs 64 --patience 10 --min_delta 0.0001 --executions_per_trial 1 --max_trials 20
    # run 4) python "Model Tuning Scripts/DenseNet201_run_tuner_script.py" --epochs 20 --patience 3 --min_delta 0.001 --executions_per_trial 1 --max_trials 20

    main()

    # tested on: python DenseNet201_run_tuner_script.py --epochs 5 --patience 3 --min_delta 0.01 --executions_per_trial 1 --max_trials 5