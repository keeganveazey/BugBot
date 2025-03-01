
# our commonly used functions
from global_bug_bot_functions import  *

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
import time
import itertools

# data set constants
TRAIN_GENERATOR = load_data(TRAIN_DIR)
VAL_GENERATOR = load_data(VALID_DIR)
TEST_GENERATOR = load_data(TEST_DIR, shuffle_flag=False)
EVAL_VAL_GENERATOR = load_data(VALID_DIR, shuffle_flag=False)

maximum_epochs = 64

class TunableDenseNet201(kt.HyperModel):

    def build(self, hp):
        '''builds DenseNet201 customized transfer learning CNN model for hp tuning '''

        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # freeze the base model layers
        base_model.trainable = False

        # tune dropout rate
        dropout_rate = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)

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
        return model

    def fit(self, hp, model, *args, **kwargs):
        '''fits customized MobileNetV2 model with tunable batch size'''
        return model.fit(*args, batch_size=hp.Choice("batch_size", [16, 32, 64]),**kwargs)


# BUILD BEST MODEL WITH TUNED PARAMS
def build_best_model_transfer_learning(algorithm, model_name, hp_class = TunableDenseNet201()):
    '''
    Param: algorithm - string in ['bayes', 'random_search'],
            model_name - string version of model
    Use: Build the best model using the desired algorithm to get the best hyperparameter
    based on validation accuracy
    '''

    #note: number of hp combinations is high which is why these two algorithms were utilized in addition to
    # literature review:
    # 3 options for batch size, 4 options for learning rate, 4 options for dropout level = 48 combinations
    # nCr = 11 C 3 = 165 possible unique hp combinations

    if algorithm == 'bayes':
        # Define the Bayesian tuner
        tuner = kt.BayesianOptimization(
            hp_class,
            objective='val_accuracy',  # tune by improving validation accuracy
            max_trials=20,  # num different hp combos to try
            executions_per_trial=1,  # run each model once
            directory='bayesian_tuning',
            project_name=f'bayes_hp_tuning_{model_name}'
        )
    elif algorithm == 'random_search':
        tuner = kt.RandomSearch(
            hp_class,  # Your model-building function
            objective='val_accuracy',  # Tune for validation accuracy
            max_trials=20,  # Number of different hyperparameter combinations to try
            executions_per_trial=1,  # Number of times to run each model hp combo for robustness -- try 3?
            directory='random_search_tuning',  # Directory to store tuning results
            project_name=f'random_search_hp_tuning_{model_name}'
        )
    else:
        raise ValueError("algorithm choice must be entered as string bayes or random_search ")

    # implement regularization via early stopping instead of tuning epochs iteratively
    # set changes smaller than 0.0001 to be understood as same as a change of 0
    # halt training when 5 epochs have passed without improvement, so make sure to save the weights from the best epoch
    patience = 10
    stop_early = EarlyStopping(monitor='val_loss',min_delta=0.0001, patience = patience)


    # search hp combo, go for maximum of max epochs
    tuner.search(TRAIN_GENERATOR, validation_data=VAL_GENERATOR, epochs = maximum_epochs, callbacks=[stop_early])

    best_epochs = stop_early.stopped_epoch-patience+1 if stop_early.stopped_epoch > 0 else maximum_epochs
    print('here: epoch number is ', best_epochs)

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
    '''
    Param: model - trained keras model object,
            filename - name of file without extension
    Use: Saves model to h5 file, returns TEST accuracy loss and test accuracy
    '''

    # evaluate on test data
    test_loss, test_acc = model.evaluate(TEST_GENERATOR)
    print(f"Test Accuracy: {test_acc:.4f}")

    # save to file
    # model.save(filename)
    model.save(f'{filename}.keras')

    return test_loss, test_acc


def get_model_and_algorithm_combos_dict():

    # model to tune and associated hp class
    models_to_tune ={'DenseNet201': TunableDenseNet201()}

    hp_algorithms = ['bayes', 'random_search']

    # get all possible combinations to run
    combinations_dict = {}

    # iterate through 3 models
    for model_name, model_class in models_to_tune.items():

        # go through hp algos
        for algo in hp_algorithms:
            combinations_dict[(model_name, algo)] = model_class

    return combinations_dict



def main():

    print('--- MAKING MODEL x HP ALGORITHM COMBINATIONS ---')
    combos_to_run = get_model_and_algorithm_combos_dict()
    print('model-alg combinations:', combos_to_run)

    # there should be 6 model combo results
    results_dict = {}

    count = 0
    for key, funct in combos_to_run.items():

        start = time.time()

        print(f' --- STARTING MODEL COMBO {count + 1}/2 --- ')
        count = count + 1

        this_algorithm = key[1]  # which hp algo to use
        this_model_name = key[0]  # which model to use
        this_hp_class = funct  # which hp tuning func to use

        # getting results for this model and optimization alg
        best_hps_dict, best_model, best_model_training_history = build_best_model_transfer_learning(
            algorithm=this_algorithm,
            model_name=this_model_name,
            hp_class=this_hp_class)

        # get final model metrics on test set and save trained model to unique file

        save_model_filename = f"{this_model_name}_{this_algorithm}"
        this_test_loss, this_test_acc = evaluate_model_and_save(best_model, filename=save_model_filename)
        print(f' ---- Completed saving: {this_model_name}_{this_algorithm}.keras ---- ')

        # Print results
        print(
            f'Model: {this_model_name}, HP Algorithm: {this_algorithm}, test_loss: {this_test_loss}, test_acc: {this_test_acc}')

        # save validation results in dictionary for comparison yeet
        performance_model_metrics_dict_validation = evaluation_metrics(best_model, EVAL_VAL_GENERATOR, best_model_training_history)

        print('Completed call to evaluation_metrics for validation data.')

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
    df_results.to_csv('final_tuned_models_results_densenet.csv')


if __name__ == "__main__":
    main()
