'''
Functions to create and evaluate convex combinations of weights and models
Requires a predetermined list of fitted models and predetermined list of weights
Can be used to calculate optimal combination of weights in a weighted average
'''

import numpy as np

# For lambda regularization:
from sklearn.metrics import mean_squared_error


def calculateConvexCombination(models, weights, input):
    '''
    calculates a weighted predicution using multiple models
    input variables:
        models  - a list of (name, model) combinations
        weights - a list of weights to be assigned to the the predictions of the models
        input   - a list of input variables to be used for each model to make a prediction
            TODO: making predictions in advance & using these as input variables can decrease repetition & speed up computation time
    output:
        a prediction made using a weighted assemble of models and input

    corresponds to co(F) in page 4 of paper
    '''

    if not len(models) == len(weights):
        raise ValueError(
            'lengths of predictions and weights lists are not equal')

    # creates a list of 0's of length of list models
    weighted_prediction = [0] * len(models)
    for (name, model), weight in zip(models, weights):
        prediction = model.predict(input)
        weighted_prediction += weight * prediction
    return weighted_prediction


def lossFunction_ols(models, weights, X_validate, y_validate, lambda_reg):
    '''
    calculates the empirical loss of OLS regression for a training sample (X_train, y_train)
    input variables:
        models      - a list of models fitted to a training set
        weights     - a list of weights to be assigned to the predictions of models to calculate a convex combination
        X_validate  - a list of features used to predict y_validate
        y_validate  - a list of output values to be predicted using X_validate
        lambda_reg  - regularization parameter (lambda),  trades off the importance given to the loss of the ensemble regressor and to the selective sparsity of the base regressors used
    output:
        the total loss of the convex combination of models using weights

    corresponds to (1) and (3) on page 6 of paper
    '''
    total_loss = 0
    for X, y in zip(X_validate, y_validate):
        weighted_prediction = calculateConvexCombination(models, weights, [X])
        total_loss += np.power((y - weighted_prediction), 2)

    if lambda_reg == 0:
        return total_loss
        # prevent unnecessary computation time, we're not using the predictions anyways

    weighted_errors = 0
    for (name, model), weight in zip(models, weights):
        prediction = model.predict(X_validate)
        weighted_errors += weight * mean_squared_error(y_validate, prediction)

    total_loss += lambda_reg * weighted_errors

    return total_loss
