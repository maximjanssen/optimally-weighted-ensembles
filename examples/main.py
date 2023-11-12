'''
Program to test the performance of the convex combinations.
Uses a list of models and a data set (wine quality) to call the different solvers in solvers.py
'''

import solvers
from read_datasets import *
import numpy as np

from sklearn.utils import all_estimators
import sklearn.ensemble as all_ensembles
import sklearn.multioutput as all_multioutput

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning

# For experiments:
import pandas as pd
import os
import multiprocessing as mp
import warnings

# Override line to prevent warnings from showing (increase overview when running experiments)
# remove line to display warnings
# DO NOT USE during debugging/programming, no warnings will be displayed by the program AT ALL
warnings.filterwarnings("ignore")

datasets_dict = {'winequality_white': winequality_white, 'winequality_red': winequality_red,
                 'abalone': abalone, 'obesity': obesity, 'forestfires': forestfires}

# process ConvergenceWarning as error for try-except
# To exclude regressors that were not able to converge during fit
# remove line to included regressors that failed to converge in the ensemble (not recommended)
warnings.filterwarnings('error', category=ConvergenceWarning)

# List of names of regressors to exclude from the ensemble
# Currently only leaves out QuantileRegressor as this regressor takes too long to fit for experiments
# TODO hoe zit het met de andere?
leave_out = ['QuantileRegressor', 'RadiusNeighborsRegressor', 'SGDRegressor']

# Ignore all multioutput regressors from sklearn
# (target for all models is single output)
leave_out += all_multioutput.__all__

# Ignore all ensemble regressors from sklearn
# (remove line if you want to include them)
leave_out += all_ensembles.__all__

# definition of which datasets to use from read_datasets.py
# class includes definition of which data to load and which getXy function to use
# change this to use a different dataset
# dataset = winequality_red


def getModels() -> list:
    '''
    Returns a list of all regressor methods to be used
    models can be only sklearn models at the moment
    input variables:
        exclude: list of names of models to exclude
    returns a list of models with (name, sklearn_model) combinations
    New models to be added can be added here, including multiple models with different parameters
    '''
    estimators = all_estimators(type_filter='regressor')

    models = list()
    for name, RegressorClass in estimators:
        if name.startswith('MultiTask'):  # ignore multitask regressors
            continue
        if name in leave_out:  # ignore all models requested to leave out
            continue
        try:
            regressor = RegressorClass()
            models += [(name, regressor)]
        except Exception:
            # ignore any model that throws an error
            continue

    return models


def fitModels(models: list, X: list, y: list) -> None:
    '''
    fits a list of models using a training set
    input variables:
        models  - list of (sklearn) models to be fitted
        X       - list of features to be used for fitting
        y       - list of output values to be predicted using X
    result:
        all models in list 'models' are fitted
    '''
    to_remove = list()
    for name, model in models:
        try:
            model.fit(X, y)
        except ConvergenceWarning:
            # Remove any model that fails to converge from models list
            to_remove += [(name, model)]
        except Exception:
            # Remove any model that throws an error from models list (should not happen)
            to_remove += [(name, model)]

    for remove_pair in to_remove:  # only remove models after for loop to prevent skipping models
        models.remove(remove_pair)


def getPrediction(model, X: np.ndarray, required_shape: tuple) -> np.ndarray:
    prediction = model.predict(X)
    if prediction.shape != required_shape:
        prediction = np.reshape(prediction, required_shape)
    return prediction


def getPredictions(models: list, X_test: list, required_shape, include_average: bool = False) -> dict[str, list]:
    predictions = list()
    predictions_dict = dict()
    for name, model in models:
        prediction = getPrediction(model, X_test, required_shape)
        predictions += [prediction]
        predictions_dict[name] = prediction

    if include_average:
        predictions_dict['Average'] = np.average(predictions, axis=0)

    return predictions_dict


def getWeightedPrediction(predictions: dict[str, list], weights: dict[str, float], prediction_size: int) -> list[float]:
    '''
    returns a weighted prediction
    input variables:
        predictions - dict of predictions made by different models with model names as keys
        weights     - dict of weights to be assigned with model names as keys
    output:
        a list of weighted predictions
    '''
    weighted_prediction = [0] * prediction_size
    for model_name in weights.keys():

        weighted_prediction += predictions[model_name] * \
            weights[model_name]

    return weighted_prediction


def compare_ensemble(dataset_name: str, lambda_reg: float) -> None:
    '''
    compare the performance of the ensembles to individual model performace of a given dataset
    input variables:
        dataset_name    - name of the dataset tests are to be run on
        lambda_reg      - regularization paramter to trade off ensemble performance an individual model performance
    output:
        results for both the test set and the training set, including the errors and the names of the models
    '''
    dataset = datasets_dict[dataset_name]

    # read dataset and split into train and test sets:
    data = dataset.data
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
    X_train, y_train = dataset.getXy(df_train)
    X_test, y_test = dataset.getXy(df_test)

    df_val, df_train_S = train_test_split(
        df_train, test_size=0.2, random_state=42)
    X_train_S, y_train_S = dataset.getXy(df_train_S)
    X_val, y_val = dataset.getXy(df_val)

    models = getModels()
    fitModels(models, X_train, y_train)

    train_predictions = getPredictions(
        models, X_train, y_train.shape, include_average=True)

    train_performance = {'real': y_val}

    model_predictions = getPredictions(
        models, X_test, y_test.shape, include_average=True)

    test_performance = {'real': y_test}
    errors_dict = {'model_name': list(), 'train error': list(),
                   'test error': list()}
    for model_name in model_predictions.keys():
        test_prediction = model_predictions[model_name]
        train_prediction = train_predictions[model_name]
        test_error = mean_squared_error(y_test, test_prediction, squared=False)
        train_error = mean_squared_error(
            y_train, train_prediction, squared=False)
        errors_dict['model_name'] += [model_name]
        errors_dict['test error'] += [test_error]
        errors_dict['train error'] += [train_error]

        test_performance[model_name] = test_prediction

    models = getModels()
    fitModels(models, X_train_S, y_train_S)

    train_predictions = getPredictions(
        models, X_val, y_val.shape, include_average=False)

    weights_dict = solvers.getWeights_solveqp(
        models, X_val, y_val, lambda_reg=lambda_reg)

    weighted_prediction = getWeightedPrediction(
        model_predictions, weights_dict, y_test.shape[0])

    w_train_prediction = getWeightedPrediction(
        train_predictions, weights_dict, y_val.shape[0])

    test_error = mean_squared_error(
        y_test, weighted_prediction, squared=False)
    train_error = mean_squared_error(
        y_val, w_train_prediction, squared=False)
    errors_dict['model_name'] += [f'WA-{lambda_reg}']
    errors_dict['test error'] += [test_error]
    errors_dict['train error'] += [train_error]

    test_performance[f'WA-{lambda_reg}'] = weighted_prediction
    train_performance[f'WA-{lambda_reg}'] = w_train_prediction

    return test_performance, train_performance
